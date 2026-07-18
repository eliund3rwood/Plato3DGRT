"""
src/eval.py — Evaluation on the scene-disjoint test split.

Runs multi-step UniPC sampling for each (I_A, D_B) pair in the manifest,
compares against authentic I_B, reports PSNR / SSIM / LPIPS per scene and
aggregate, and saves qualitative (I_A, D_B, Î_B, I_B) panels.

Usage:
    python -m src.eval \\
        --config      configs/train.yaml \\
        --checkpoint  checkpoints/run_001/step_0100000.pt \\
        --manifest    data/manifests/test.jsonl \\
        --output-dir  eval_results/run_001 \\
        --num-steps   50 \\
        --guidance-scale 2.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.build import build_all, load_checkpoint
from src.data.dataset import NVSDataset
from src.data.loader import build_loader
from src.infer import sample_single_step
from src.distributed import resolve_amp_dtype, setup_distributed, cleanup_distributed


_IMG_SIZE   = 512
_LATENT_HW  = 64
_VAE_SCALE  = 0.18215


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR between two [-1,1] tensors (B×3×H×W)."""
    mse = F.mse_loss(pred.float(), gt.float()).item()
    return float("inf") if mse < 1e-10 else -10.0 * np.log10(mse + 1e-10)


def ssim_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean SSIM over a batch (B×3×H×W) in [-1,1]."""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(pred.device)
        return metric(pred, gt).item()
    except ImportError:
        # Fallback: rough estimate
        sigma_pred = pred.std().item()
        sigma_gt   = gt.std().item()
        mu_pred    = pred.mean().item()
        mu_gt      = gt.mean().item()
        C1, C2 = 0.01**2, 0.03**2
        num = (2*mu_pred*mu_gt + C1) * (2*sigma_pred*sigma_gt + C2)
        den = (mu_pred**2 + mu_gt**2 + C1) * (sigma_pred**2 + sigma_gt**2 + C2)
        return float(num / (den + 1e-8))


def lpips_batch(pred: torch.Tensor, gt: torch.Tensor, lpips_fn) -> float:
    return lpips_fn(pred.float(), gt.float()).mean().item()


# ── Main eval ─────────────────────────────────────────────────────────────────

def evaluate(
    cfg,
    checkpoint_path: str,
    manifest_path: str,
    output_dir: Path,
    num_steps: int = 50,
    guidance_scale: float = 2.0,
    max_samples: int | None = None,
) -> dict:
    is_dist    = dist.is_available() and dist.is_initialized()
    rank       = dist.get_rank()       if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1
    device     = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype  = resolve_amp_dtype(cfg.model.amp_dtype)

    if rank == 0:
        print(f"[eval] world_size={world_size}  device={device}")

    # Each rank loads its own model copy
    components = build_all(cfg, device)
    load_checkpoint(
        checkpoint_path,
        components["unet"], components["controlnet"],
        components.get("image_proj"), None, None, None, device
    )
    components["unet"].eval()
    components["controlnet"].eval()

    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    ds = NVSDataset(manifest_path, use_cached_latents=False)
    loader, _ = build_loader(
        ds, micro_batch=1, num_workers=2,
        distributed=is_dist, rank=rank, world_size=world_size, shuffle=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(exist_ok=True)

    all_psnr, all_ssim, all_lpips = [], [], []
    per_scene: dict[str, list] = {}

    n_done = 0
    for batch in tqdm(loader, desc=f"Eval [rank {rank}]", disable=(rank != 0)):
        if max_samples and n_done >= max_samples:
            break

        I_A  = batch["I_A"].to(device)
        D_B  = batch["D_B"].to(device)
        I_B  = batch["I_B"].to(device)
        meta = batch["meta"]

        I_hat = sample_single_step(
            I_A, D_B, components, cfg,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            amp_dtype=amp_dtype,
        )

        p = psnr(I_hat, I_B)
        s = ssim_batch(I_hat, I_B)
        l = lpips_batch(I_hat, I_B, lpips_fn)
        all_psnr.append(p);  all_ssim.append(s);  all_lpips.append(l)

        sid = meta["scene_id"][0] if isinstance(meta["scene_id"], list) else meta["scene_id"]
        per_scene.setdefault(sid, []).append({"psnr": p, "ssim": s, "lpips": l})

        # Panel filename: r{rank}_{local_idx} keeps all ranks collision-free
        _save_panel(I_A, D_B, I_hat, I_B, panels_dir / f"r{rank:02d}_{n_done:06d}.png")
        n_done += 1

    # ── Aggregate metrics across all ranks ───────────────────────────────────
    n_local = len(all_psnr)
    if is_dist and n_local > 0:
        def _reduce(vals):
            t = torch.tensor(sum(vals), dtype=torch.float64, device=device)
            dist.all_reduce(t)
            return t.item()
        n_t = torch.tensor(n_local, dtype=torch.float64, device=device)
        dist.all_reduce(n_t)
        total_n    = int(n_t.item())
        g_psnr     = _reduce(all_psnr)  / total_n
        g_ssim     = _reduce(all_ssim)  / total_n
        g_lpips    = _reduce(all_lpips) / total_n
        g_psnr_std = float(np.std(all_psnr))  # local std; global std needs gather
    else:
        total_n    = n_local
        g_psnr     = float(np.mean(all_psnr))  if all_psnr  else 0.0
        g_ssim     = float(np.mean(all_ssim))  if all_ssim  else 0.0
        g_lpips    = float(np.mean(all_lpips)) if all_lpips else 0.0
        g_psnr_std = float(np.std(all_psnr))   if all_psnr  else 0.0

    agg = {
        "psnr_mean":  g_psnr,
        "psnr_std":   g_psnr_std,
        "ssim_mean":  g_ssim,
        "lpips_mean": g_lpips,
        "n_samples":  total_n,
    }

    if rank == 0:
        print(f"\n[eval] Results ({total_n} samples, {world_size} GPUs):")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        per_scene_agg = {
            sid: {
                "psnr":  float(np.mean([m["psnr"]  for m in ms])),
                "ssim":  float(np.mean([m["ssim"]  for m in ms])),
                "lpips": float(np.mean([m["lpips"] for m in ms])),
                "n":     len(ms),
            }
            for sid, ms in per_scene.items()
        }
        results = {"aggregate": agg, "per_scene": per_scene_agg}
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] Saved metrics → {output_dir / 'metrics.json'}")

    if is_dist:
        dist.barrier()

    return agg


def _save_panel(I_A, D_B, I_hat, I_B, out_path: Path) -> None:
    import torchvision.utils as vutils
    def to01(x): return (x.clamp(-1, 1) + 1) / 2
    grid = vutils.make_grid(
        torch.cat([to01(I_A), D_B, to01(I_hat), to01(I_B)], dim=0),
        nrow=4, padding=4
    )
    vutils.save_image(grid, str(out_path))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate PlatoControlNet")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--manifest",    default="data/manifests/test.jsonl")
    parser.add_argument("--output-dir",  default="eval_results/default")
    parser.add_argument("--num-steps",   type=int,   default=50)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int,   default=None)
    parser.add_argument("overrides",     nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/model.yaml"),
        OmegaConf.load("configs/data.yaml"),
        OmegaConf.load(args.config),
    )
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    setup_distributed()
    evaluate(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=Path(args.output_dir),
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        max_samples=args.max_samples,
    )
    cleanup_distributed()


if __name__ == "__main__":
    main()

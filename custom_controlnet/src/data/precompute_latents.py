"""
src/data/precompute_latents.py — Pre-compute and cache z_A, z_B from frozen VAE.

Eliminates per-step VAE encode during training (frozen VAE = no gradient needed).
Writes float16 .npy per sample under latent_cache_dir/.
Rewrites manifest with z_A_path / z_B_path fields.

Usage:
    python -m src.data.precompute_latents \\
        --manifest-dir   data/manifests \\
        --output-dir     data/latent_cache \\
        --model-id       runwayml/stable-diffusion-v1-5 \\
        --batch-size     16 \\
        --num-workers    4
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


_LATENT_SCALE = 0.18215
_IMG_SIZE = 512
_LATENT_HW = 64


class _RGBOnlyDataset(Dataset):
    """Minimal dataset that returns (I_A, I_B, scene_id, frame_id_A, frame_id_B, row_idx)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import cv2
        row = self.rows[idx]

        def load_rgb(p):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (_IMG_SIZE, _IMG_SIZE):
                img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            return t * 2.0 - 1.0  # [-1, 1]

        return {
            "I_A": load_rgb(row["I_A_path"]),
            "I_B": load_rgb(row["I_B_path"]),
            "idx": idx,
        }


def _frame_cache_path(cache_dir: Path, img_path: str, suffix: str) -> Path:
    """Stable cache path based on image path hash (reuse across manifest splits)."""
    h = hashlib.sha1(img_path.encode()).hexdigest()[:12]
    stem = Path(img_path).stem
    return cache_dir / f"{stem}_{h}_{suffix}.npy"


def precompute_latents(
    manifest_dir: Path,
    output_dir: Path,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    batch_size: int = 16,
    num_workers: int = 4,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VAE (frozen)
    print(f"[precompute_latents] Loading VAE from {model_id} …")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.eval().requires_grad_(False).to(device)

    # Use AMP for speed but store as float16
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    for split in splits:
        manifest_path = manifest_dir / f"{split}.jsonl"
        if not manifest_path.exists():
            print(f"  [skip] {manifest_path} not found")
            continue

        rows = []
        with open(manifest_path) as f:
            for line in f:
                r = json.loads(line.strip())
                if r:
                    rows.append(r)

        print(f"[precompute_latents] {split}: {len(rows)} triples")

        ds = _RGBOnlyDataset(rows)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, shuffle=False, drop_last=False)

        updated_rows = list(rows)  # will patch z_A_path / z_B_path

        for batch in tqdm(loader, desc=f"VAE encode {split}"):
            I_A = batch["I_A"].to(device)
            I_B = batch["I_B"].to(device)
            indices = batch["idx"].tolist()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_A = vae.encode(I_A).latent_dist.sample() * _LATENT_SCALE
                z_B = vae.encode(I_B).latent_dist.sample() * _LATENT_SCALE

            # Validate shapes
            assert z_A.shape[-2:] == (_LATENT_HW, _LATENT_HW), \
                f"z_A latent shape {z_A.shape} != (...,{_LATENT_HW},{_LATENT_HW})"
            assert z_A.shape[1] == 4, f"z_A channels {z_A.shape[1]} != 4"

            z_A_np = z_A.float().cpu().numpy().astype(np.float16)
            z_B_np = z_B.float().cpu().numpy().astype(np.float16)

            for bi, idx in enumerate(indices):
                row = updated_rows[idx]

                cache_A = _frame_cache_path(output_dir, row["I_A_path"], "zA")
                cache_B = _frame_cache_path(output_dir, row["I_B_path"], "zB")

                if not cache_A.exists():
                    np.save(str(cache_A), z_A_np[bi])
                if not cache_B.exists():
                    np.save(str(cache_B), z_B_np[bi])

                updated_rows[idx]["z_A_path"] = str(cache_A)
                updated_rows[idx]["z_B_path"] = str(cache_B)

        # Rewrite manifest with cache paths
        out_manifest = manifest_dir / f"{split}.jsonl"
        with open(out_manifest, "w") as f:
            for r in updated_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  Manifest updated: {out_manifest}")

    print(f"[precompute_latents] Done. Latents cached under {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--output-dir",   default="data/latent_cache")
    parser.add_argument("--model-id",     default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--splits",       nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    precompute_latents(
        manifest_dir=Path(args.manifest_dir),
        output_dir=Path(args.output_dir),
        model_id=args.model_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=tuple(args.splits),
    )


if __name__ == "__main__":
    main()

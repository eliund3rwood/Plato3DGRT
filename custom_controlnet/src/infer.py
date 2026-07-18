"""
src/infer.py — Multi-step inference (UniPC/DDIM) for novel view synthesis.

Supports:
  - Batch or single-image inference
  - Optional classifier-free guidance on the appearance/text stream
  - Configurable number of denoising steps

The inference loop mirrors §1.6:
  1. Start from Gaussian noise z_T
  2. Each step: z_A concatenated, D_B drives ControlNet
  3. UniPC multi-step sampling (or DDIM)
  4. Final decode through frozen VAE → 512×512 Î_B

Usage (CLI):
    python -m src.infer \\
        --checkpoint  checkpoints/run_001/final.pt \\
        --source      path/to/I_A.png \\
        --depth       path/to/D_B.npy \\
        --output      output/I_B_hat.png \\
        --num-steps   50 \\
        --guidance-scale 2.0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.distributed import resolve_amp_dtype
from src.models.build import (
    build_all, load_checkpoint,
    set_reference_capture, set_reference_inject, clear_reference_bank,
)
from src.models.conditioning import ReferenceAttnProcessor
from src.train import get_text_embeds


_IMG_SIZE  = 512
_LATENT_HW = 64
_VAE_SCALE = 0.18215


def _clip_preprocess(I_A: torch.Tensor) -> torch.Tensor:
    """Resize and normalise I_A ([-1,1]) to CLIP ViT-L/14 input format."""
    x = F.interpolate(I_A.float(), size=(224, 224), mode="bicubic", align_corners=False)
    x = (x.clamp(-1, 1) + 1) / 2
    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=I_A.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=I_A.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ── Core sampling function ────────────────────────────────────────────────────

@torch.no_grad()
def sample_single_step(
    I_A: torch.Tensor,          # B×3×512×512 in [-1,1], on device
    D_B: torch.Tensor,          # B×3×512×512 in [0,1],  on device
    components: dict,
    cfg,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.0,
    amp_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Run multi-step denoising to produce Î_B (V3 architecture).
    Returns B×3×512×512 float32 tensor in [-1,1] on the same device as I_A.

    Three conditioning paths:
      1. ControlNet  — D_B (3ch depth) → spatial residuals
      2. IP-Adapter+ — CLIP patch tokens → Resampler → 16 tokens → decoder cross-attn
      3. Ref-attn    — VAE(I_A) at t=0 → K/V bank → decoder self-attn injection
    """
    device = I_A.device
    B      = I_A.shape[0]

    vae             = components["vae"]
    unet            = components["unet"]
    controlnet      = components["controlnet"]
    infer_sched     = components["infer_scheduler"]
    tokenizer       = components["tokenizer"]
    text_encoder    = components["text_encoder"]
    image_proj      = components.get("image_proj")
    clip_image_enc  = components.get("clip_image_encoder")
    use_ip          = cfg.model.get("use_ip_adapter", False) and image_proj is not None

    assert I_A.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)
    assert D_B.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)

    _unet = unet.module if hasattr(unet, "module") else unet
    use_cfg = guidance_scale > 1.0

    infer_sched.set_timesteps(num_inference_steps, device=device)

    # ── Text embeddings (empty prompt) ───────────────────────────────────────
    text_embeds = get_text_embeds(tokenizer, text_encoder, [""] * B, device)

    # ── IP-Adapter+: CLIP patch tokens → Resampler → 16 appearance tokens ───
    ip_tokens = None
    if use_ip and clip_image_enc is not None:
        clip_patch_tokens = clip_image_enc(_clip_preprocess(I_A)).last_hidden_state  # B×257×1024
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            ip_tokens = image_proj(clip_patch_tokens.to(amp_dtype))  # B×16×768

    # ── Reference attention: capture K/V from clean I_A at t=0 ──────────────
    if use_ip:
        z_A    = vae.encode(I_A.float()).latent_dist.sample() * _VAE_SCALE
        t_zero = torch.zeros(B, dtype=torch.long, device=device)
        set_reference_capture(_unet, True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _unet(z_A, t_zero, encoder_hidden_states=text_embeds)
        set_reference_capture(_unet, False)

        # CFG doubles the batch — expand K/V bank from B → 2B so both
        # uncond and cond halves receive the reference features.
        if use_cfg:
            for proc in _unet.attn_processors.values():
                if isinstance(proc, ReferenceAttnProcessor) and proc._bank_k is not None:
                    proc._bank_k = proc._bank_k.repeat(2, 1, 1)
                    proc._bank_v = proc._bank_v.repeat(2, 1, 1)

        set_reference_inject(_unet, True)

    # ── CFG setup ─────────────────────────────────────────────────────────────
    if use_cfg:
        uncond_text     = get_text_embeds(tokenizer, text_encoder, [""] * B, device)
        text_embeds_in  = torch.cat([uncond_text, text_embeds], dim=0)
        D_B_in          = D_B.repeat(2, 1, 1, 1)
        ip_tokens_in    = torch.cat([torch.zeros_like(ip_tokens), ip_tokens], dim=0) \
                          if ip_tokens is not None else None
    else:
        text_embeds_in = text_embeds
        D_B_in         = D_B
        ip_tokens_in   = ip_tokens

    # ── Denoising loop ────────────────────────────────────────────────────────
    z = torch.randn(B, 4, _LATENT_HW, _LATENT_HW, device=device, dtype=amp_dtype)

    for t in infer_sched.timesteps:
        z_in = z.repeat(2, 1, 1, 1) if use_cfg else z
        t_in = t.unsqueeze(0).expand(z_in.shape[0])

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            down_res, mid_res = controlnet(
                z_in, t_in,
                encoder_hidden_states=text_embeds_in,
                controlnet_cond=D_B_in,
                return_dict=False,
            )

            ip_kw = {"ip_hidden_states": ip_tokens_in} if ip_tokens_in is not None else {}

            noise_pred = unet(
                z_in, t_in,
                encoder_hidden_states=text_embeds_in,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
                cross_attention_kwargs=ip_kw or None,
            ).sample

        if use_cfg:
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        z = infer_sched.step(noise_pred, t, z, return_dict=False)[0]

    # ── Cleanup reference bank ────────────────────────────────────────────────
    if use_ip:
        set_reference_inject(_unet, False)
        clear_reference_bank(_unet)

    # ── Decode ────────────────────────────────────────────────────────────────
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        I_hat = vae.decode(z / _VAE_SCALE).sample

    assert I_hat.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)
    return I_hat.clamp(-1, 1).float()


# ── Single-image CLI ──────────────────────────────────────────────────────────

def _load_rgb_tensor(path: str, device: torch.device) -> torch.Tensor:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (_IMG_SIZE, _IMG_SIZE):
        img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def _load_depth_tensor(path: str, near: float, far: float, device: torch.device) -> torch.Tensor:
    depth = np.load(str(path)).astype(np.float32)
    if depth.shape != (_IMG_SIZE, _IMG_SIZE):
        import cv2
        depth = cv2.resize(depth, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    depth = np.clip((depth - near) / (far - near + 1e-6), 0.0, 1.0)
    t = torch.from_numpy(depth).unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
    return t.to(device)


def _save_rgb_tensor(t: torch.Tensor, path: str) -> None:
    import cv2
    img = ((t.squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)
    print(f"[infer] Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="PlatoControlNet inference")
    parser.add_argument("--checkpoint",      required=True)
    parser.add_argument("--source",          required=True, help="Path to I_A (RGB .png)")
    parser.add_argument("--depth",           required=True, help="Path to D_B (.npy float32)")
    parser.add_argument("--output",          required=True, help="Output path (.png)")
    parser.add_argument("--config",          default="configs/train.yaml")
    parser.add_argument("--num-steps",       type=int,   default=50)
    parser.add_argument("--guidance-scale",  type=float, default=2.0)
    parser.add_argument("--depth-near",      type=float, default=0.1)
    parser.add_argument("--depth-far",       type=float, default=10.0)
    parser.add_argument("overrides",         nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/model.yaml"),
        OmegaConf.load("configs/data.yaml"),
        OmegaConf.load(args.config),
    )
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = resolve_amp_dtype(cfg.model.amp_dtype)

    print(f"[infer] Building models …")
    components = build_all(cfg, device)
    load_checkpoint(
        args.checkpoint,
        components["unet"], components["controlnet"],
        components.get("image_proj"), None, None, None, device
    )

    for m in [components["unet"], components["controlnet"]]:
        m.eval()

    I_A = _load_rgb_tensor(args.source, device)
    D_B = _load_depth_tensor(args.depth, args.depth_near, args.depth_far, device)

    print(f"[infer] Sampling {args.num_steps} steps, guidance={args.guidance_scale} …")
    I_hat = sample_single_step(
        I_A, D_B, components, cfg,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        amp_dtype=amp_dtype,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    _save_rgb_tensor(I_hat, args.output)


if __name__ == "__main__":
    main()

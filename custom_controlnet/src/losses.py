"""
src/losses.py — Training objectives for PlatoControlNet.

Primary objective (mandatory):
    L_diff = MSE(ε_pred, ε)           latent denoising loss (§1.5)

Auxiliary objectives (computed from decoded ẑ_0, scheduled, configurable):
    L_pixel = MSE(Î_B, I_B)           over the full 512×512 Pose-B grid
    L_lpips = LPIPS(Î_B, I_B)         perceptual loss

Total:
    L_total = L_diff + λ_pix·L_pixel + λ_lpips·L_lpips

Pixel-level terms are expensive (VAE decode per step); compute them
on a configurable schedule (aux_loss_every_n_steps) to control VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMG_SIZE = 512
_LATENT_HW = 64


# ── ẑ_0 reconstruction from ε-prediction ─────────────────────────────────────

def predict_z0(
    noise_pred: torch.Tensor,   # B×4×64×64
    z_t: torch.Tensor,          # B×4×64×64
    t: torch.Tensor,            # B  (integer timesteps)
    alphas_cumprod: torch.Tensor,  # (T,) on same device
) -> torch.Tensor:
    """
    Derive ẑ_0 from ε-prediction:
        ẑ_0 = (z_t − √(1−ᾱ_t) · ε_pred) / √ᾱ_t
    """
    assert noise_pred.shape[-2:] == (_LATENT_HW, _LATENT_HW), \
        f"noise_pred shape {noise_pred.shape} — expected latent 64×64"
    assert z_t.shape[-2:] == (_LATENT_HW, _LATENT_HW), \
        f"z_t shape {z_t.shape} — expected latent 64×64"

    alpha_t = alphas_cumprod[t].to(z_t.device)          # B
    alpha_t = alpha_t[:, None, None, None]               # B×1×1×1
    beta_t  = 1.0 - alpha_t

    z0 = (z_t - beta_t.sqrt() * noise_pred) / (alpha_t.sqrt() + 1e-8)
    return z0


# ── Loss module ───────────────────────────────────────────────────────────────

class NVSLoss(nn.Module):
    """
    Combined NVS training loss.

    Args:
        lpips_net: backbone for LPIPS ("alex" | "vgg"). "alex" is faster.
    """

    def __init__(self, lpips_net: str = "alex"):
        super().__init__()
        try:
            import lpips as lpips_lib
            self.lpips_fn = lpips_lib.LPIPS(net=lpips_net)
        except ImportError:
            raise ImportError("lpips not installed: pip install lpips")

        # Freeze LPIPS network
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)

    # ── Sub-losses ────────────────────────────────────────────────────────────

    @staticmethod
    def diffusion_loss(noise_pred: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Primary: MSE in latent space. Both tensors B×4×64×64."""
        assert noise_pred.shape[-2:] == (_LATENT_HW, _LATENT_HW)
        assert noise.shape[-2:] == (_LATENT_HW, _LATENT_HW)
        return F.mse_loss(noise_pred.float(), noise.float())

    @staticmethod
    def pixel_loss(
        pred_rgb: torch.Tensor,    # B×3×512×512 in [-1,1]
        gt_rgb: torch.Tensor,      # B×3×512×512 in [-1,1]
        alpha_mask: torch.Tensor | None = None,  # B×1×512×512 in [0,1]
    ) -> torch.Tensor:
        """
        MSE over the full 512×512 Pose-B grid (§1.5, §4.3).
        Optional alpha mask down-weights empty/holey regions.
        """
        assert pred_rgb.shape[-2:] == (_IMG_SIZE, _IMG_SIZE), \
            f"pred_rgb {pred_rgb.shape} — expected 512×512"
        assert gt_rgb.shape[-2:] == (_IMG_SIZE, _IMG_SIZE), \
            f"gt_rgb {gt_rgb.shape} — expected 512×512"

        loss = F.mse_loss(pred_rgb.float(), gt_rgb.float(), reduction="none")  # B×3×H×W
        if alpha_mask is not None:
            loss = loss * alpha_mask
        return loss.mean()

    def lpips_loss(
        self,
        pred_rgb: torch.Tensor,   # B×3×512×512 in [-1,1]
        gt_rgb: torch.Tensor,     # B×3×512×512 in [-1,1]
    ) -> torch.Tensor:
        assert pred_rgb.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)
        assert gt_rgb.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)
        # LPIPS expects [-1,1] input
        self.lpips_fn = self.lpips_fn.to(pred_rgb.device)
        return self.lpips_fn(pred_rgb.float(), gt_rgb.float()).mean()

    # ── Combined forward ──────────────────────────────────────────────────────

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        pred_rgb: torch.Tensor | None = None,
        gt_rgb: torch.Tensor | None = None,
        lambda_pix: float = 0.0,
        lambda_lpips: float = 0.0,
        alpha_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns (total_loss, {L_diff, L_pixel, L_lpips}).
        Auxiliary terms are only computed when the corresponding λ > 0
        and pred_rgb is provided.
        """
        L_diff = self.diffusion_loss(noise_pred, noise)

        L_pixel = torch.zeros(1, device=noise.device, dtype=torch.float32)
        L_lpips = torch.zeros(1, device=noise.device, dtype=torch.float32)

        if pred_rgb is not None and gt_rgb is not None:
            if lambda_pix > 0.0:
                L_pixel = self.pixel_loss(pred_rgb, gt_rgb, alpha_mask)
            if lambda_lpips > 0.0:
                L_lpips = self.lpips_loss(pred_rgb, gt_rgb)

        total = L_diff + lambda_pix * L_pixel + lambda_lpips * L_lpips

        metrics = {
            "L_diff":  L_diff.item(),
            "L_pixel": L_pixel.item(),
            "L_lpips": L_lpips.item(),
            "L_total": total.item(),
        }
        return total, metrics


# ── λ schedule ────────────────────────────────────────────────────────────────

def get_lambda(step: int, warmup_steps: int, ramp_steps: int, target: float) -> float:
    """
    Linear ramp schedule:
      0 … warmup_steps                    → 0.0
      warmup_steps … warmup+ramp_steps    → linear 0→target
      warmup+ramp_steps …                 → target
    """
    if step < warmup_steps:
        return 0.0
    if step < warmup_steps + ramp_steps:
        frac = (step - warmup_steps) / max(ramp_steps, 1)
        return frac * target
    return target

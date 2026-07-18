"""
src/vsd_prior.py — Variational Score Distillation (VSD) using the
custom_controlnet novel-view diffusion prior, for texture (and optionally
geometry) refinement of a trained Plato3DGRT / 3DGRT Gaussian scene.

The prior model (vendored at <repo_root>/custom_controlnet/) predicts a
novel-view RGB image given (I_A, D_B): a fixed reference RGB photo I_A and a
target-pose depth map D_B. It never needs pose A explicitly — I_A only
supplies appearance via CLIP+IP-Adapter tokens and a VAE-latent reference
K/V bank injected into the UNet's decoder self-attention (see
custom_controlnet/src/models/conditioning.py). D_B drives a frozen
ControlNet branch. This module treats that whole pipeline as a frozen score
function and produces a differentiable surrogate loss whose gradient w.r.t.
a *rendered* (I_B, D_B) pair equals the score-distillation gradient — so
backpropagating it through 3DGRT's renderer updates the Gaussians.

Two variants:
  "vsd" (default) — true Variational Score Distillation (ProlificDreamer):
      a small LoRA adapter on the frozen UNet is trained online (its own
      tiny Adam optimizer) to approximate the noise-prediction distribution
      of the *current* renders, conditioned the same way as the pretrained
      model (same D_B/I_A). The Gaussian gradient uses the difference
      between the pretrained and LoRA noise predictions. This avoids the
      mode-seeking blur that plain SDS produces when many (t, noise) samples
      all regress toward a single point estimate — this is the mechanism
      that gives VSD sharper texture than SDS.
  "sds" — classic DreamFusion-style Score Distillation Sampling. No LoRA, no
      extra dependency (skips `peft`), useful as a cheap first sanity check
      that the wiring (rendering, depth normalisation, conditioning) is
      correct before paying for the LoRA optimizer.

Usage:
    prior = DiffusionPrior(checkpoint_path=".../final.pt", device=device)
    prior.set_reference_image(load_reference_image("chair_smooth_walls.png", device))
    ...
    loss_vsd, metrics = prior.step(rgb_512, depth_512_3ch)
    (loss_geom_shadow + vsd_weight * loss_vsd).backward()
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


_DEFAULT_CONTROLNET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "custom_controlnet")
)

_IMG_SIZE = 512
_VAE_SCALE = 0.18215

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _ensure_on_path(path: str) -> None:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)


# ── Depth conditioning (per-image percentile normalisation, matching how the
#    ControlNet's training data was built — see
#    custom_controlnet/src/data/render_depth.py: DEPTH_NEAR_PCT/FAR_PCT) ──────

def normalize_depth_percentile(
    depth: torch.Tensor,          # [H, W] metric depth
    alpha: torch.Tensor,          # [H, W] accumulated opacity, in [0, 1]
    near_pct: float = 2.0,
    far_pct: float = 98.0,
    empty_value: float = 1.0,     # background/empty pixels -> far
    min_range: float = 1e-3,
) -> torch.Tensor:
    """
    Returns depth_norm [H, W] float32 in [0, 1].

    Differentiable w.r.t. `depth` — the near/far window is computed from
    `depth.detach()` (treated as fixed per-step statistics, not something we
    want gradient through), but the affine map applied to `depth` keeps its
    graph. When the caller has frozen geometry (see --vsd_freeze_geometry in
    run_platonerf_3dgrt_vsd.py), `depth` has no grad anyway and this is moot.
    """
    valid = (depth > 0) & (alpha > 0.1)
    depth_const = depth.detach()
    if valid.any():
        near = torch.quantile(depth_const[valid], near_pct / 100.0)
        far = torch.quantile(depth_const[valid], far_pct / 100.0)
    else:
        near = depth_const.new_tensor(0.0)
        far = depth_const.new_tensor(1.0)
    rng = torch.clamp(far - near, min=min_range)

    d_norm = torch.clamp((depth - near) / rng, 0.0, 1.0)
    d_norm = torch.where(alpha > 0.1, d_norm, torch.full_like(d_norm, empty_value))
    return d_norm


def make_depth_cond(depth_hw: torch.Tensor, alpha_hw: torch.Tensor, **kwargs) -> torch.Tensor:
    """[H,W] depth/alpha -> 1x3xHxW depth-conditioning tensor for ControlNet."""
    d = normalize_depth_percentile(depth_hw, alpha_hw, **kwargs)
    return d.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).contiguous()


def _get_text_embeds(tokenizer, text_encoder, prompts: list[str], device: torch.device) -> torch.Tensor:
    """Encode a list of prompts (always [""] here — text conditioning is
    unused by this model). Inlined from custom_controlnet/src/train.py so
    this module doesn't need to import the whole training entrypoint (with
    its NVSDataset/build_loader/wandb dependencies) just for this helper."""
    tokens = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(tokens.input_ids.to(device))[0]
    return embeds


def load_reference_image(path: str, device: torch.device) -> torch.Tensor:
    """Load I_A -> 1x3x512x512 float tensor in [-1, 1] (mirrors infer.py)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read reference RGB: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (_IMG_SIZE, _IMG_SIZE):
        img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


class DiffusionPrior:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        controlnet_repo_root: str = _DEFAULT_CONTROLNET_ROOT,
        variant: str = "vsd",              # "vsd" or "sds"
        guidance_scale: float = 3.0,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_lr: float = 1e-4,
        t_min_frac: float = 0.02,
        t_max_frac: float = 0.75,
        grad_weight: str = "snr",          # "snr" -> (1-alpha_cumprod[t]); "uniform" -> 1
    ):
        assert variant in ("vsd", "sds"), f"unknown variant: {variant}"
        _ensure_on_path(controlnet_repo_root)

        from src.models.build import build_all, load_checkpoint, set_reference_capture
        from src.distributed import resolve_amp_dtype, unwrap_module
        from src.models.conditioning import ReferenceAttnProcessor

        self._unwrap = unwrap_module
        self._set_reference_capture = set_reference_capture
        self._ReferenceAttnProcessor = ReferenceAttnProcessor

        self.device = device
        self.variant = variant
        self.guidance_scale = guidance_scale
        self.t_min_frac = t_min_frac
        self.t_max_frac = t_max_frac
        self.grad_weight = grad_weight

        cfg = OmegaConf.load(os.path.join(controlnet_repo_root, "configs", "model.yaml"))
        self.amp_dtype = resolve_amp_dtype(cfg.model.amp_dtype)

        print(f"[vsd_prior] Building diffusion prior (variant={variant}, "
              f"amp_dtype={self.amp_dtype}) from {controlnet_repo_root} ...")
        self.components = build_all(cfg, device)
        load_checkpoint(
            checkpoint_path,
            self.components["unet"], self.components["controlnet"],
            self.components.get("image_proj"), None, None, None, device,
        )

        for key in ("vae", "unet", "controlnet", "text_encoder"):
            self.components[key].eval().requires_grad_(False)
        for key in ("clip_image_encoder", "image_proj"):
            if self.components.get(key) is not None:
                self.components[key].eval().requires_grad_(False)

        self.unet = self.components["unet"]
        self.lora_optimizer = None
        if self.variant == "vsd":
            self._add_lora(rank=lora_rank, alpha=lora_alpha, lr=lora_lr)

        self._ref_cache = None
        print(f"[vsd_prior] Ready. Call set_reference_image() before step().")

    # ------------------------------------------------------------------
    def _add_lora(self, rank: int, alpha: int, lr: float) -> None:
        try:
            from peft import LoraConfig
        except ImportError as e:
            raise ImportError(
                "variant='vsd' requires the `peft` package for the LoRA "
                "particle network (pip install peft). Use variant='sds' to "
                "run without it."
            ) from e

        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
        )
        try:
            self.unet.add_adapter(lora_config, adapter_name="vsd_particle")
        except AttributeError as e:
            raise RuntimeError(
                "unet.add_adapter() not found — this diffusers version is "
                "too old for LoRA/PEFT adapters. Upgrade with "
                "`pip install -U diffusers` or use variant='sds'."
            ) from e

        lora_params = [p for p in self.unet.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in lora_params)
        print(f"[vsd_prior] LoRA particle adapter: rank={rank}, {n_params:,} trainable params")
        self.lora_optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_reference_image(self, I_A: torch.Tensor) -> None:
        """
        I_A: 1x3x512x512 float tensor in [-1, 1], on `device`.

        Precomputes and caches everything that depends only on I_A (fixed
        for the whole run): CLIP->Resampler IP-Adapter tokens, the VAE
        latent + decoder self-attention K/V reference bank, and the
        (always empty-string) text embedding — so step() never has to
        recompute them.
        """
        vae = self.components["vae"]
        unet = self._unwrap(self.unet)
        tokenizer, text_encoder = self.components["tokenizer"], self.components["text_encoder"]
        clip_enc = self.components.get("clip_image_encoder")
        image_proj = self.components.get("image_proj")

        assert I_A.shape[-2:] == (_IMG_SIZE, _IMG_SIZE), f"I_A shape {I_A.shape}"
        B = I_A.shape[0]

        text_embeds = _get_text_embeds(tokenizer, text_encoder, [""] * B, self.device)
        text_embeds = text_embeds.to(self.amp_dtype)

        ip_tokens = None
        if clip_enc is not None and image_proj is not None:
            clip_in = F.interpolate(I_A.float(), size=(224, 224), mode="bicubic", align_corners=False)
            clip_in = (clip_in.clamp(-1, 1) + 1) / 2
            mean = torch.tensor(_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(_CLIP_STD, device=self.device).view(1, 3, 1, 1)
            clip_in = (clip_in - mean) / std
            clip_tokens = clip_enc(clip_in).last_hidden_state
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                ip_tokens = self._unwrap(image_proj)(clip_tokens.to(self.amp_dtype))

        z_A = vae.encode(I_A.float()).latent_dist.sample() * _VAE_SCALE
        z_A = z_A.to(self.amp_dtype)
        t_zero = torch.zeros(B, dtype=torch.long, device=self.device)

        # Capture with the LoRA adapter disabled — the reference bank should
        # reflect pretrained decoder features, exactly like at inference time.
        if self.variant == "vsd":
            unet.disable_adapters()
        self._set_reference_capture(unet, True)
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            unet(z_A, t_zero, encoder_hidden_states=text_embeds)
        self._set_reference_capture(unet, False)

        bank = {}
        for name, proc in unet.attn_processors.items():
            if isinstance(proc, self._ReferenceAttnProcessor):
                bank[name] = (proc._bank_k.clone(), proc._bank_v.clone())
                proc._bank_k, proc._bank_v = None, None

        self._ref_cache = {"text_embeds": text_embeds, "ip_tokens": ip_tokens, "ref_bank": bank}
        print(f"[vsd_prior] Reference image cached ({len(bank)} reference-attn layers banked).")

    def _inject_reference(self, batch_size: int) -> None:
        unet = self._unwrap(self.unet)
        for name, proc in unet.attn_processors.items():
            if isinstance(proc, self._ReferenceAttnProcessor):
                k, v = self._ref_cache["ref_bank"][name]
                proc._bank_k = k.repeat(batch_size, 1, 1)
                proc._bank_v = v.repeat(batch_size, 1, 1)
                proc.do_inject = True

    def _clear_reference(self) -> None:
        unet = self._unwrap(self.unet)
        for proc in unet.attn_processors.values():
            if isinstance(proc, self._ReferenceAttnProcessor):
                proc.do_inject = False
                proc._bank_k, proc._bank_v = None, None

    # ------------------------------------------------------------------
    def step(self, rgb_512: torch.Tensor, depth_512_3ch: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        rgb_512:        1x3x512x512 float32 in [-1, 1]. Differentiable w.r.t.
                         the 3DGRT render.
        depth_512_3ch:  1x3x512x512 float32 in [0, 1] (3ch-replicated,
                         per-image-normalised depth; see make_depth_cond).

        Returns (loss, metrics). `loss` is a surrogate: loss.backward() (or
        `(other_loss + w*loss).backward()`) deposits exactly the (V)SD
        gradient into rgb_512/depth_512_3ch, which then flows on into
        whatever produced them (3DGRT Gaussian parameters).
        """
        assert self._ref_cache is not None, "call set_reference_image() first"
        assert rgb_512.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)
        assert depth_512_3ch.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)

        vae = self.components["vae"]
        controlnet = self.components["controlnet"]
        unet = self._unwrap(self.unet)
        train_sched = self.components["train_scheduler"]

        B = rgb_512.shape[0]
        text_embeds = self._ref_cache["text_embeds"]
        ip_tokens = self._ref_cache["ip_tokens"]

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            z0 = vae.encode(rgb_512.to(self.amp_dtype)).latent_dist.sample() * _VAE_SCALE
        z0 = z0.float()

        num_train_t = train_sched.config.num_train_timesteps
        t_lo = int(self.t_min_frac * num_train_t)
        t_hi = max(t_lo + 1, int(self.t_max_frac * num_train_t))
        t = torch.randint(t_lo, t_hi, (B,), device=self.device)
        noise = torch.randn_like(z0)
        z_t = train_sched.add_noise(z0, noise, t)
        z_t_in = z_t.detach().to(self.amp_dtype)

        alphas_cumprod = train_sched.alphas_cumprod.to(self.device)
        if self.grad_weight == "uniform":
            w = torch.ones_like(t, dtype=torch.float32)
        else:
            w = (1.0 - alphas_cumprod[t]).float()
        w = w.view(-1, 1, 1, 1)

        depth_in = depth_512_3ch.to(self.amp_dtype)
        metrics = {"t_mean": float(t.float().mean().item())}

        # ---- Pretrained score, with CFG over IP-adapter appearance conditioning ----
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            if self.variant == "vsd":
                unet.disable_adapters()

            text_in = text_embeds.repeat(2, 1, 1)
            depth_in2 = depth_in.repeat(2, 1, 1, 1)
            z_t_in2 = z_t_in.repeat(2, 1, 1, 1)
            t_in2 = t.repeat(2)
            ip_in = None
            if ip_tokens is not None:
                ip_in = torch.cat([torch.zeros_like(ip_tokens), ip_tokens], dim=0)

            self._inject_reference(batch_size=2)
            down_res, mid_res = controlnet(
                z_t_in2, t_in2, encoder_hidden_states=text_in,
                controlnet_cond=depth_in2, return_dict=False,
            )
            cross_kw = {"ip_hidden_states": ip_in} if ip_in is not None else None
            eps_2 = unet(
                z_t_in2, t_in2, encoder_hidden_states=text_in,
                cross_attention_kwargs=cross_kw,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample
            self._clear_reference()

            eps_uncond, eps_cond = eps_2.chunk(2)
            eps_pretrained = (eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)).float()

        if self.variant == "sds":
            grad = w * (eps_pretrained - noise)
            target = (z0 - grad).detach()
            loss = 0.5 * F.mse_loss(z0, target, reduction="sum") / B
            metrics["loss_lora"] = 0.0
            metrics["loss_vsd"] = float(loss.item())
            return loss, metrics

        # ---- VSD: LoRA "particle" score, no CFG, same D_B/I_A conditioning ----
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            unet.enable_adapters()
            self._inject_reference(batch_size=1)
            down_res_1, mid_res_1 = controlnet(
                z_t_in, t, encoder_hidden_states=text_embeds,
                controlnet_cond=depth_in, return_dict=False,
            )
            cross_kw_1 = {"ip_hidden_states": ip_tokens} if ip_tokens is not None else None
            eps_lora = unet(
                z_t_in, t, encoder_hidden_states=text_embeds,
                cross_attention_kwargs=cross_kw_1,
                down_block_additional_residuals=down_res_1,
                mid_block_additional_residual=mid_res_1,
            ).sample
            self._clear_reference()

        # Train the LoRA particle network on this same sample (standard
        # denoising loss). Detached from the render — this step only updates
        # LoRA weights, never the Gaussians.
        loss_lora = F.mse_loss(eps_lora.float(), noise.float())
        self.lora_optimizer.zero_grad(set_to_none=True)
        loss_lora.backward()
        self.lora_optimizer.step()
        unet.disable_adapters()  # leave UNet pretrained-only by default

        eps_lora_value = eps_lora.detach().float()
        grad = w * (eps_pretrained - eps_lora_value)
        target = (z0 - grad).detach()
        loss = 0.5 * F.mse_loss(z0, target, reduction="sum") / B

        metrics["loss_lora"] = float(loss_lora.item())
        metrics["loss_vsd"] = float(loss.item())
        return loss, metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def preview(self, depth_512_3ch: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
        """
        Full multi-step denoising -> predicted Î_B, for qualitative
        monitoring only (not used for gradients). Reuses the cached
        reference encoding set by set_reference_image().
        """
        assert self._ref_cache is not None, "call set_reference_image() first"
        vae = self.components["vae"]
        controlnet = self.components["controlnet"]
        unet = self._unwrap(self.unet)
        infer_sched = self.components["infer_scheduler"]

        text_embeds = self._ref_cache["text_embeds"]
        ip_tokens = self._ref_cache["ip_tokens"]
        B = depth_512_3ch.shape[0]

        if self.variant == "vsd":
            unet.disable_adapters()

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            infer_sched.set_timesteps(num_steps, device=self.device)
            z = torch.randn(B, 4, 64, 64, device=self.device, dtype=self.amp_dtype)

            depth_in = depth_512_3ch.to(self.amp_dtype)
            use_cfg = self.guidance_scale > 1.0 and ip_tokens is not None
            if use_cfg:
                text_in = text_embeds.repeat(2, 1, 1)
                depth_in_batch = depth_in.repeat(2, 1, 1, 1)
                ip_in = torch.cat([torch.zeros_like(ip_tokens), ip_tokens], dim=0)
                self._inject_reference(batch_size=2)
            else:
                text_in, depth_in_batch, ip_in = text_embeds, depth_in, ip_tokens
                self._inject_reference(batch_size=1)

            for t in infer_sched.timesteps:
                z_in = z.repeat(2, 1, 1, 1) if use_cfg else z
                t_in = t.unsqueeze(0).expand(z_in.shape[0])
                down_res, mid_res = controlnet(
                    z_in, t_in, encoder_hidden_states=text_in,
                    controlnet_cond=depth_in_batch, return_dict=False,
                )
                cross_kw = {"ip_hidden_states": ip_in} if ip_in is not None else None
                noise_pred = unet(
                    z_in, t_in, encoder_hidden_states=text_in,
                    cross_attention_kwargs=cross_kw,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample
                if use_cfg:
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
                z = infer_sched.step(noise_pred, t, z, return_dict=False)[0]

            self._clear_reference()
            I_hat = vae.decode(z.float() / _VAE_SCALE).sample.clamp(-1, 1)

        return I_hat

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """LoRA weights + its optimizer state, for checkpoint/resume."""
        if self.variant != "vsd":
            return {}
        unet = self._unwrap(self.unet)
        lora_sd = {k: v for k, v in unet.state_dict().items() if "lora" in k}
        return {"lora_state_dict": lora_sd, "lora_optimizer": self.lora_optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if self.variant != "vsd" or not state:
            return
        unet = self._unwrap(self.unet)
        unet.load_state_dict(state["lora_state_dict"], strict=False)
        self.lora_optimizer.load_state_dict(state["lora_optimizer"])

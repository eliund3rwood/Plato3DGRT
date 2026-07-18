"""
src/train.py — Full training loop for PlatoControlNet.

Handles:
  - Hardware-aware AMP (bf16 on Ampere+, fp16+GradScaler on Turing)
  - Gradient accumulation
  - Distributed / single-GPU via config flag only
  - Rank-0 discipline: all logging, image dumps, checkpoints on rank 0 only
  - Auxiliary loss schedule with configurable frequency
  - VRAM + throughput profiling mode (M5 gate)
  - Single-triple overfit mode (M4 smoke test)

Launch:
    Single GPU:  python -m src.train --config configs/train.yaml
    Multi-GPU:   torchrun --nproc_per_node=N -m src.train --config configs/train.yaml
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from src.distributed import (
    all_reduce_mean,
    barrier,
    get_local_rank,
    get_rank,
    get_world_size,
    is_main_process,
    maybe_wrap_ddp,
    seed_everything,
    setup_distributed,
    cleanup_distributed,
    resolve_amp_dtype,
)
from src.models.build import (
    build_all,
    get_trainable_params,
    save_checkpoint,
    load_checkpoint,
    set_reference_capture,
    set_reference_inject,
    clear_reference_bank,
)
from src.losses import NVSLoss, get_lambda, predict_z0
from src.data.dataset import NVSDataset
from src.data.loader import build_loader


_VAE_SCALE = 0.18215
_LATENT_HW = 64
_IMG_SIZE   = 512


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_text_embeds(tokenizer, text_encoder, prompts: list[str], device: torch.device):
    """Encode a list of prompts. Empty prompts for unconditional conditioning."""
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(tokens.input_ids.to(device))[0]
    return embeds



def make_optimizer(params, cfg):
    lr = cfg.train.lr
    wd = cfg.train.weight_decay
    betas = tuple(cfg.train.betas)
    eps = cfg.train.eps

    if cfg.model.get("use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        except ImportError:
            print("[train] bitsandbytes not installed; falling back to AdamW")

    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)


def make_lr_scheduler(optimizer, cfg, total_steps: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

    warmup = cfg.train.warmup_steps

    def warmup_fn(step):
        if step < warmup:
            return float(step) / max(warmup, 1)
        return 1.0

    sched_name = cfg.train.lr_scheduler
    if sched_name == "constant_with_warmup":
        return LambdaLR(optimizer, lr_lambda=warmup_fn)

    if sched_name == "cosine_with_restarts":
        def cosine_fn(step):
            warmup_lr = warmup_fn(step)
            if step < warmup:
                return warmup_lr
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda=cosine_fn)

    raise ValueError(f"Unknown lr_scheduler: {sched_name}")


# ── VRAM / throughput profiling ───────────────────────────────────────────────

def run_vram_profile(cfg, device, components) -> None:
    """Run a short forward+backward pass and report peak VRAM and steps/sec."""
    print("\n[profile] VRAM + throughput gate (M5)")
    vae          = components["vae"]
    unet         = components["unet"]
    controlnet   = components["controlnet"]
    train_sched  = components["train_scheduler"]
    tokenizer    = components["tokenizer"]
    text_encoder = components["text_encoder"]

    amp_dtype    = resolve_amp_dtype(cfg.model.amp_dtype)
    use_scaler   = amp_dtype == torch.float16
    scaler       = torch.cuda.amp.GradScaler(enabled=use_scaler)

    params = get_trainable_params(unet, controlnet)
    optimizer = make_optimizer(params, cfg)

    B = cfg.train.micro_batch
    dummy_rgb   = torch.zeros(B, 3, _IMG_SIZE, _IMG_SIZE, device=device)
    dummy_depth = torch.zeros(B, 1, _IMG_SIZE, _IMG_SIZE, device=device)

    for trial_batch in [B]:
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        N_STEPS = 10

        for step in range(N_STEPS):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                with torch.no_grad():
                    z_B = vae.encode(dummy_rgb).latent_dist.sample() * _VAE_SCALE

                t_idx = torch.randint(0, train_sched.config.num_train_timesteps,
                                      (trial_batch,), device=device)
                noise = torch.randn_like(z_B)
                z_t = train_sched.add_noise(z_B, noise, t_idx)
                cond_6ch = torch.cat([dummy_depth, dummy_rgb], dim=1)  # 3ch depth + 3ch RGB

                text_embeds = get_text_embeds(tokenizer, text_encoder,
                                              [""] * trial_batch, device)

                down_res, mid_res = controlnet(
                    z_t, t_idx,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=cond_6ch,
                    return_dict=False,
                )

                noise_pred = unet(
                    z_t, t_idx,
                    encoder_hidden_states=text_embeds,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample

                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        elapsed = time.perf_counter() - t0
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
        steps_sec = N_STEPS / elapsed

        print(f"  micro_batch={trial_batch}  peak_VRAM={peak_vram_gb:.2f} GB  "
              f"steps/sec={steps_sec:.2f}")
        print(f"\n  *** Paste these numbers into DECISIONS.md §D-006 ***\n")


# ── Single-step forward (shared by train + overfit) ───────────────────────────

def forward_step(
    batch: dict,
    components: dict,
    cfg,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_fn: NVSLoss,
    step: int,
    compute_aux: bool,
) -> tuple[torch.Tensor, dict]:
    vae                = components["vae"]
    unet               = components["unet"]
    controlnet         = components["controlnet"]
    train_sched        = components["train_scheduler"]
    tokenizer          = components["tokenizer"]
    text_encoder       = components["text_encoder"]
    clip_image_encoder = components.get("clip_image_encoder")
    image_proj         = components.get("image_proj")

    I_A = batch["I_A"].to(device, non_blocking=True)
    D_B = batch["D_B"].to(device, non_blocking=True)
    I_B = batch["I_B"].to(device, non_blocking=True)
    alpha_B = batch.get("alpha_B")
    if alpha_B is not None:
        alpha_B = alpha_B.to(device, non_blocking=True)

    assert I_A.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)
    assert D_B.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)
    assert I_B.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)

    B = I_A.shape[0]

    # ── IP-Adapter+: CLIP encode I_A (fp32, outside autocast) ───────────────
    clip_patch_tokens = None
    if clip_image_encoder is not None and image_proj is not None:
        clip_in = F.interpolate(I_A.float(), size=(224, 224), mode="bicubic", align_corners=False)
        clip_in = (clip_in.clamp(-1, 1) + 1) / 2
        _mean = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=device).view(1, 3, 1, 1)
        _std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        clip_in = (clip_in - _mean) / _std
        with torch.no_grad():
            # last_hidden_state: B × 257 × 1024 (all patch tokens, not pooled)
            clip_patch_tokens = clip_image_encoder(clip_in).last_hidden_state

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        if clip_patch_tokens is not None:
            ip_tokens = image_proj(clip_patch_tokens.to(amp_dtype))  # B × 16 × 768
        else:
            ip_tokens = None

        # ── VAE encode target (or use cached latent) ─────────────────────────
        if "z_B" in batch:
            z_B = batch["z_B"].to(device, non_blocking=True).to(amp_dtype)
        else:
            with torch.no_grad():
                z_B = vae.encode(I_B).latent_dist.sample() * _VAE_SCALE

        assert z_B.shape == (B, 4, _LATENT_HW, _LATENT_HW)

        # ── Diffusion forward ────────────────────────────────────────────────
        t = torch.randint(
            0, train_sched.config.num_train_timesteps, (B,), device=device
        )
        noise = torch.randn_like(z_B)
        z_t = train_sched.add_noise(z_B, noise, t)

        # ── Text conditioning (empty prompt) ─────────────────────────────────
        text_embeds = get_text_embeds(
            tokenizer, text_encoder, [""] * B, device
        ).to(amp_dtype)

        # ── Reference attention: encode z_A and collect decoder K/V ──────────
        # Run I_A through VAE → z_A, then through frozen UNet at t=0 to bank
        # the decoder self-attention K/V for injection during the actual forward.
        _unet = unet.module if hasattr(unet, "module") else unet
        with torch.no_grad():
            z_A = vae.encode(I_A.float()).latent_dist.sample() * _VAE_SCALE
            z_A = z_A.to(amp_dtype)
            t_zeros = torch.zeros(B, dtype=torch.long, device=device)
            set_reference_capture(_unet, True)
            _unet(z_A, t_zeros, encoder_hidden_states=text_embeds)
            set_reference_capture(_unet, False)
        set_reference_inject(_unet, True)

        # ── ControlNet conditioning: D_B depth (3ch, native pretrained format) ─
        cond_3ch = D_B  # already [0,1] 3ch replicated from dataset

        # ── ControlNet ────────────────────────────────────────────────────────
        down_res, mid_res = controlnet(
            z_t, t,
            encoder_hidden_states=text_embeds,
            controlnet_cond=cond_3ch.to(amp_dtype),
            return_dict=False,
        )

        # ── UNet: IP-Adapter tokens in cross-attn + reference K/V in self-attn ─
        cross_attn_kwargs = {"ip_hidden_states": ip_tokens} if ip_tokens is not None else None
        noise_pred = unet(
            z_t, t,
            encoder_hidden_states=text_embeds,
            cross_attention_kwargs=cross_attn_kwargs,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        ).sample

        set_reference_inject(_unet, False)
        clear_reference_bank(_unet)

        assert noise_pred.shape[-2:] == (_LATENT_HW, _LATENT_HW)

        # ── Auxiliary loss (VAE decode) ───────────────────────────────────────
        pred_rgb = None
        if compute_aux:
            z0_pred = predict_z0(noise_pred, z_t, t,
                                 train_sched.alphas_cumprod.to(device))
            with torch.no_grad():
                pred_rgb = vae.decode(z0_pred / _VAE_SCALE).sample.clamp(-1, 1)
            assert pred_rgb.shape[-2:] == (_IMG_SIZE, _IMG_SIZE)

        # ── Compute λ values ──────────────────────────────────────────────────
        lam_pix  = get_lambda(step, cfg.train.aux_warmup_steps,
                              cfg.train.aux_ramp_steps, cfg.train.lambda_pix_target)
        lam_lpips= get_lambda(step, cfg.train.aux_warmup_steps,
                              cfg.train.aux_ramp_steps, cfg.train.lambda_lpips_target)

        total_loss, metrics = loss_fn(
            noise_pred, noise,
            pred_rgb=pred_rgb,
            gt_rgb=I_B.to(amp_dtype) if pred_rgb is not None else None,
            lambda_pix=lam_pix,
            lambda_lpips=lam_lpips,
            alpha_mask=alpha_B,
        )

    return total_loss, metrics


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg) -> None:
    # ── Distributed init ─────────────────────────────────────────────────────
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    seed_everything(cfg.train.seed, rank)

    amp_dtype  = resolve_amp_dtype(cfg.model.amp_dtype)
    use_scaler = amp_dtype == torch.float16
    if is_main_process():
        print(f"[train] rank={rank}/{world_size}  device={device}  "
              f"amp_dtype={amp_dtype}  GradScaler={use_scaler}")

    # ── Build models ─────────────────────────────────────────────────────────
    components = build_all(cfg, device)
    vae         = components["vae"]
    unet        = components["unet"]
    controlnet  = components["controlnet"]
    image_proj  = components.get("image_proj")

    # UNet is wrapped too: IP-Adapter processors inside it have trainable params.
    # DDP only syncs grads for requires_grad=True params, so frozen UNet weights
    # add no sync overhead.
    unet = maybe_wrap_ddp(unet, local_rank)
    controlnet = maybe_wrap_ddp(controlnet, local_rank)
    if image_proj is not None:
        image_proj = maybe_wrap_ddp(image_proj, local_rank)
    components["unet"] = unet
    components["controlnet"] = controlnet
    components["image_proj"] = image_proj

    # ── Optimizer & GradScaler ────────────────────────────────────────────────
    params    = get_trainable_params(unet, controlnet, image_proj)
    optimizer = make_optimizer(params, cfg)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ── Dataset & loaders ─────────────────────────────────────────────────────
    distributed = world_size > 1
    manifest_dir = Path(cfg.data.manifest_dir)
    use_cached   = cfg.data.use_cached_latents

    if cfg.train.get("overfit_single", False):
        # Single-triple overfit mode for M4 smoke test
        manifest = manifest_dir / "train.jsonl"
        train_ds = NVSDataset(manifest, use_cached_latents=use_cached)
        # Subset to 1 sample
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, [0])
        if is_main_process():
            print("[train] *** OVERFIT MODE: single triple ***")
    else:
        train_ds = NVSDataset(
            manifest_dir / "train.jsonl", use_cached_latents=use_cached
        )
    val_ds = NVSDataset(
        manifest_dir / "val.jsonl", use_cached_latents=use_cached
    ) if (manifest_dir / "val.jsonl").exists() else None

    train_loader, train_sampler = build_loader(
        train_ds, cfg.train.micro_batch, cfg.data.num_workers,
        distributed=distributed, rank=rank, world_size=world_size,
    )
    val_loader = None
    if val_ds is not None:
        val_loader, _ = build_loader(
            val_ds, cfg.train.micro_batch, cfg.data.num_workers,
            distributed=False, rank=0, world_size=1, shuffle=True,
        )

    max_steps       = cfg.train.max_steps
    lr_scheduler    = make_lr_scheduler(optimizer, cfg, max_steps)
    loss_fn         = NVSLoss(lpips_net=cfg.train.lpips_net).to(device)
    grad_accum      = cfg.train.grad_accum_steps
    aux_every       = cfg.train.aux_loss_every_n_steps
    log_every       = cfg.train.log_every_n_steps
    save_every      = cfg.train.save_every_n_steps
    eval_every      = cfg.train.eval_every_n_steps
    log_img_every   = cfg.train.log_image_every_n_steps
    ckpt_dir        = Path(cfg.train.checkpoint_dir)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    if cfg.train.resume_from:
        start_step, start_epoch = load_checkpoint(
            cfg.train.resume_from,
            unet, controlnet, image_proj, optimizer, scaler, lr_scheduler, device
        )
        if is_main_process():
            print(f"[train] Resumed from {cfg.train.resume_from} at step {start_step}")

    # ── W&B / TensorBoard (rank 0 only) ──────────────────────────────────────
    writer = None
    if is_main_process():
        if cfg.train.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.train.wandb_project,
                    entity=cfg.train.get("wandb_entity"),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    resume="allow",
                )
            except Exception as e:
                print(f"[train] W&B init failed: {e}")
        if cfg.train.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=cfg.train.log_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── VRAM profile mode (M5) ─────────────────────────────────────────────────
    if cfg.train.get("profile_vram", False):
        if is_main_process():
            run_vram_profile(cfg, device, components)
        cleanup_distributed()
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    step      = start_step
    epoch     = start_epoch
    optimizer.zero_grad(set_to_none=True)
    running_metrics: dict[str, float] = {}
    keep_last = cfg.train.get("keep_last_n_checkpoints", 5)
    saved_ckpts: list[Path] = []

    unet.train()
    controlnet.train()
    if image_proj is not None:
        image_proj.train()

    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            compute_aux = (step % aux_every == 0)

            total_loss, metrics = forward_step(
                batch, components, cfg, device, amp_dtype,
                loss_fn, step, compute_aux
            )
            total_loss = total_loss / grad_accum

            scaler.scale(total_loss).backward()

            # Accumulation step
            if (step + 1) % grad_accum == 0 or (step + 1) == max_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            # Accumulate metrics
            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + v

            step += 1

            # ── Logging (rank 0) ──────────────────────────────────────────────
            if is_main_process() and step % log_every == 0:
                avg = {k: v / log_every for k, v in running_metrics.items()}
                running_metrics.clear()
                lr = optimizer.param_groups[0]["lr"]
                print(f"step {step:6d}  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in avg.items()) +
                      f"  lr={lr:.2e}")
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"train/" + k: v for k, v in avg.items()} |
                                  {"train/lr": lr}, step=step)
                except Exception:
                    pass
                if writer is not None:
                    for k, v in avg.items():
                        writer.add_scalar(f"train/{k}", v, step)

            # ── Qualitative image grid (rank 0) ───────────────────────────────
            if is_main_process() and step % log_img_every == 0:
                _log_image_grid(batch, components, cfg, device, amp_dtype, step)

            # ── Checkpoint (rank 0) ───────────────────────────────────────────
            if is_main_process() and step % save_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                save_checkpoint(
                    str(ckpt_path), unet, controlnet, image_proj,
                    optimizer, scaler if use_scaler else None,
                    lr_scheduler, step, epoch
                )
                saved_ckpts.append(ckpt_path)
                if len(saved_ckpts) > keep_last:
                    old = saved_ckpts.pop(0)
                    if old.exists():
                        old.unlink()
                print(f"[train] Saved checkpoint → {ckpt_path}")
            barrier()

            # ── Val eval (rank 0) ─────────────────────────────────────────────
            if is_main_process() and step % eval_every == 0 and val_loader is not None:
                val_metrics = run_eval(
                    val_loader, components, cfg, device, amp_dtype,
                    n_samples=cfg.train.eval_num_samples,
                    step=step,
                )
                print(f"[eval] step {step}  " +
                      "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

            if step >= max_steps:
                break

        epoch += 1

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if is_main_process():
        final_path = ckpt_dir / "final.pt"
        save_checkpoint(
            str(final_path), unet, controlnet, image_proj,
            optimizer, scaler if use_scaler else None,
            lr_scheduler, step, epoch
        )
        print(f"[train] Final checkpoint → {final_path}")
        if writer is not None:
            writer.close()
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    cleanup_distributed()


# ── Eval pass ─────────────────────────────────────────────────────────────────

def run_eval(
    val_loader,
    components: dict,
    cfg,
    device: torch.device,
    amp_dtype: torch.dtype,
    n_samples: int = 64,
    step: int = 0,
) -> dict:
    """Quick val pass: MSE/PSNR in latent space (proxy for quality) + image grid."""
    unet        = components["unet"]
    controlnet  = components["controlnet"]
    train_sched = components["train_scheduler"]

    unet.eval()
    controlnet.eval()

    total_mse = 0.0
    total_n   = 0
    first_batch = None

    with torch.no_grad():
        for batch in val_loader:
            if total_n >= n_samples:
                break
            if first_batch is None:
                first_batch = batch
            loss, metrics = forward_step(
                batch, components, cfg, device, amp_dtype,
                NVSLoss().to(device), step=0, compute_aux=False
            )
            total_mse += metrics["L_diff"] * batch["I_A"].shape[0]
            total_n   += batch["I_A"].shape[0]

    psnr = -10.0 * torch.log10(torch.tensor(total_mse / max(total_n, 1) + 1e-8)).item()

    # Save val image grid from first batch
    if first_batch is not None:
        _log_image_grid(first_batch, components, cfg, device, amp_dtype, step, prefix="val")

    unet.train()
    controlnet.train()

    return {"val_L_diff": total_mse / max(total_n, 1), "val_PSNR_proxy": psnr}


# ── Image grid logging ────────────────────────────────────────────────────────

@torch.no_grad()
def _run_inference(I_A, D_B, components, cfg, device, amp_dtype):
    """
    Full denoising inference: pure Gaussian noise → N DDIM steps → VAE decode.
    ControlNet cond: D_B depth (3ch, native pretrained format).
    IP-Adapter: CLIP(I_A) → appearance tokens → UNet cross-attention.
    Returns I_hat in [-1, 1]. Models must already be in eval mode.
    """
    from src.distributed import unwrap_module
    vae                = components["vae"]
    unet               = components["unet"]
    controlnet         = components["controlnet"]
    infer_sched        = components["infer_scheduler"]
    tokenizer          = components["tokenizer"]
    text_encoder       = components["text_encoder"]
    clip_image_encoder = components.get("clip_image_encoder")
    image_proj         = components.get("image_proj")
    n_steps            = cfg.train.num_inference_steps

    B = I_A.shape[0]
    cond_3ch = D_B  # already 3ch [0,1]

    # ── IP-Adapter+: CLIP patch tokens (fp32) ───────────────────────────────
    ip_tokens = None
    if clip_image_encoder is not None and image_proj is not None:
        clip_in = F.interpolate(I_A.float(), size=(224, 224), mode="bicubic", align_corners=False)
        clip_in = (clip_in.clamp(-1, 1) + 1) / 2
        _mean = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=device).view(1, 3, 1, 1)
        _std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        clip_in = (clip_in - _mean) / _std
        clip_patch_tokens = clip_image_encoder(clip_in).last_hidden_state  # B × 257 × 1024

    _unet = unwrap_module(unet)

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        if clip_image_encoder is not None and image_proj is not None:
            ip_tokens = unwrap_module(image_proj)(clip_patch_tokens.to(amp_dtype))  # B × 16 × 768

        text_embeds = get_text_embeds(
            tokenizer, text_encoder, [""] * B, device
        ).to(amp_dtype)

        # ── Reference attention: collect z_A decoder features once ───────────
        z_A = vae.encode(I_A.float()).latent_dist.sample() * _VAE_SCALE
        z_A = z_A.to(amp_dtype)
        t_zeros = torch.zeros(B, dtype=torch.long, device=device)
        set_reference_capture(_unet, True)
        _unet(z_A, t_zeros, encoder_hidden_states=text_embeds)
        set_reference_capture(_unet, False)
        set_reference_inject(_unet, True)

        infer_sched.set_timesteps(n_steps, device=device)
        z_t = torch.randn(B, 4, _LATENT_HW, _LATENT_HW, device=device, dtype=amp_dtype)

        cross_attn_kwargs = {"ip_hidden_states": ip_tokens} if ip_tokens is not None else None

        for t in infer_sched.timesteps:
            t_batch = t.unsqueeze(0).expand(B)

            down_res, mid_res = unwrap_module(controlnet)(
                z_t, t_batch,
                encoder_hidden_states=text_embeds,
                controlnet_cond=cond_3ch.to(amp_dtype),
                return_dict=False,
            )
            noise_pred = _unet(
                z_t, t_batch,
                encoder_hidden_states=text_embeds,
                cross_attention_kwargs=cross_attn_kwargs,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample

            z_t = infer_sched.step(noise_pred, t, z_t).prev_sample

        set_reference_inject(_unet, False)
        clear_reference_bank(_unet)

        I_hat = vae.decode(z_t / _VAE_SCALE).sample.clamp(-1, 1)

    return I_hat


def _log_image_grid(batch, components, cfg, device, amp_dtype, step, prefix="train"):
    """Save a qualitative (I_A, D_B, Î_B, I_B) grid using full denoising inference."""
    try:
        import torchvision.utils as vutils
        from src.distributed import unwrap_module

        n = min(4, batch["I_A"].shape[0])
        I_A = batch["I_A"][:n].to(device)
        D_B = batch["D_B"][:n].to(device)
        I_B = batch["I_B"][:n].to(device)

        # Switch to eval for inference
        unwrap_module(components["unet"]).eval()
        unwrap_module(components["controlnet"]).eval()

        I_hat = _run_inference(I_A, D_B, components, cfg, device, amp_dtype)

        unwrap_module(components["unet"]).train()
        unwrap_module(components["controlnet"]).train()

        def to01(x): return (x.clamp(-1, 1) + 1) / 2

        grid = vutils.make_grid(
            torch.cat([to01(I_A), D_B.expand(-1, 3, -1, -1), to01(I_hat), to01(I_B)], dim=0),
            nrow=n, padding=2
        )
        img_dir = Path(cfg.train.checkpoint_dir) / "images"
        img_dir.mkdir(exist_ok=True)
        vutils.save_image(grid, img_dir / f"{prefix}_step_{step:07d}.png")
        print(f"[train] Saved {prefix} image grid → {prefix}_step_{step:07d}.png")

    except Exception as e:
        print(f"[train] Image grid failed ({prefix}): {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PlatoControlNet")
    parser.add_argument("--config", required=True,
                        help="Base config file (e.g. configs/train.yaml)")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra-style overrides: key=value …")
    args = parser.parse_args()

    # Load config hierarchy: model + data + train merged
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/model.yaml"),
        OmegaConf.load("configs/data.yaml"),
        OmegaConf.load(args.config),
    )
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    if is_main_process():
        print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()

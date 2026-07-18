"""
src/models/build.py — Assemble the full model graph.

Components loaded:
  - AutoencoderKL            (frozen)
  - UNet2DConditionModel     (frozen except IP-Adapter processors)
  - ControlNetModel          (trainable; depth-conditioned, native 3ch pretrained input)
  - CLIPVisionModelWithProjection (frozen; encodes source view I_A for IP-Adapter)
  - ImageProjection          (trainable; maps CLIP pooled features → 4 appearance tokens)
  - IPAttnProcessor2_0       (trainable; replaces UNet cross-attention processors)
  - CLIPTextModel / CLIPTokenizer (frozen; empty prompt always)

Conditioning format:
  ControlNet cond = D_B depth 1ch expanded to 3ch  (native pretrained weights, no expansion)
  IP-Adapter      = CLIP(I_A) → ImageProjection → decoupled cross-attn in every UNet decoder layer
  UNet input      = z_t (noisy target latent, 4 channels, standard SD1.5)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.distributed import resolve_amp_dtype
from src.models.conditioning import Resampler, IPAttnProcessor2_0, ReferenceAttnProcessor


_EXPECTED_BLOCK_CHANNELS = [320, 640, 1280, 1280]
_VAE_SCALE = 0.18215


# ── Block-channel assertion ───────────────────────────────────────────────────

def assert_unet_block_channels(unet) -> None:
    block_channels = list(unet.config.block_out_channels)
    assert block_channels == _EXPECTED_BLOCK_CHANNELS, (
        f"UNet block_out_channels {block_channels} != {_EXPECTED_BLOCK_CHANNELS}. "
        "Use runwayml/stable-diffusion-v1-5 (SD1.5)."
    )


# ── VAE ───────────────────────────────────────────────────────────────────────

def build_vae(model_id: str, device: torch.device) -> nn.Module:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.eval().requires_grad_(False).to(device)
    assert abs(vae.config.scaling_factor - _VAE_SCALE) < 1e-5
    return vae


# ── UNet ──────────────────────────────────────────────────────────────────────

def build_unet(model_id: str, device: torch.device, unfreeze_unet: bool = False) -> nn.Module:
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    assert_unet_block_channels(unet)
    unet.requires_grad_(False)
    if unfreeze_unet:
        unet.requires_grad_(True)
    unet.to(device)
    return unet


# ── ControlNet ────────────────────────────────────────────────────────────────

def build_controlnet(ckpt_id: str, device: torch.device) -> nn.Module:
    from diffusers import ControlNetModel
    controlnet = ControlNetModel.from_pretrained(ckpt_id)
    # Pretrained depth ControlNet expects 3ch input — no weight expansion needed.
    controlnet.requires_grad_(True).to(device)
    return controlnet


# ── CLIP vision encoder ───────────────────────────────────────────────────────

def build_clip_image_encoder(model_id: str, device: torch.device) -> nn.Module:
    from transformers import CLIPVisionModelWithProjection
    encoder = CLIPVisionModelWithProjection.from_pretrained(model_id)
    encoder.eval().requires_grad_(False).to(device)
    return encoder


# ── IP-Adapter ────────────────────────────────────────────────────────────────

def set_ip_adapter_processors(
    unet: nn.Module,
    cross_attention_dim: int = 768,
    ip_scale: float = 1.0,
) -> None:
    """
    Replace UNet cross-attention (attn2) processors with IPAttnProcessor2_0.
    Self-attention (attn1) processors are left unchanged.
    IP processor params are explicitly unfrozen after replacement.
    """
    new_procs = {}
    for name in unet.attn_processors.keys():
        if "attn1" in name:
            new_procs[name] = unet.attn_processors[name]
        else:
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = cross_attention_dim
            new_procs[name] = IPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                ip_scale=ip_scale,
            )
    unet.set_attn_processor(new_procs)

    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor2_0):
            for p in proc.parameters():
                p.requires_grad_(True)


def build_resampler(device: torch.device) -> nn.Module:
    """PerceiverResampler: 257 CLIP patch tokens → 16 appearance tokens (IP-Adapter+)."""
    return Resampler(
        clip_dim=1024, depth=4, heads=16, head_dim=64,
        num_queries=16, output_dim=768,
    ).to(device)


def set_reference_attn_processors(unet: nn.Module) -> None:
    """Replace mid_block + up_blocks self-attention (attn1) with ReferenceAttnProcessor."""
    new_procs = {}
    for name, proc in unet.attn_processors.items():
        if ("mid_block" in name or "up_blocks" in name) and "attn1" in name:
            new_procs[name] = ReferenceAttnProcessor()
        else:
            new_procs[name] = proc
    unet.set_attn_processor(new_procs)


def set_reference_capture(unet: nn.Module, flag: bool) -> None:
    from src.distributed import unwrap_module
    for proc in unwrap_module(unet).attn_processors.values():
        if isinstance(proc, ReferenceAttnProcessor):
            proc.do_capture = flag
            if flag:
                proc._bank_k = None
                proc._bank_v = None


def set_reference_inject(unet: nn.Module, flag: bool) -> None:
    from src.distributed import unwrap_module
    for proc in unwrap_module(unet).attn_processors.values():
        if isinstance(proc, ReferenceAttnProcessor):
            proc.do_inject = flag


def clear_reference_bank(unet: nn.Module) -> None:
    from src.distributed import unwrap_module
    for proc in unwrap_module(unet).attn_processors.values():
        if isinstance(proc, ReferenceAttnProcessor):
            proc._bank_k = None
            proc._bank_v = None


# ── CLIP text encoder ─────────────────────────────────────────────────────────

def build_text_encoder(model_id: str, device: torch.device):
    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder.eval().requires_grad_(False).to(device)
    return tokenizer, text_encoder


# ── Noise schedulers ──────────────────────────────────────────────────────────

def build_schedulers(model_id: str):
    from diffusers import DDPMScheduler, UniPCMultistepScheduler
    train_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    infer_scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    return train_scheduler, infer_scheduler


# ── Gradient checkpointing ────────────────────────────────────────────────────

def enable_gradient_checkpointing(unet: nn.Module, controlnet: nn.Module) -> None:
    # UNet grad checkpointing is incompatible with ReferenceAttnProcessor:
    # the K/V bank is cleared before backward(), so the checkpointing recompute
    # sees an empty bank and produces different-shaped tensors → CheckpointError.
    # The UNet body is frozen anyway, so there are no weight gradients to save.
    controlnet.enable_gradient_checkpointing()


# ── Memory-efficient attention ────────────────────────────────────────────────

def enable_efficient_attention(unet: nn.Module, controlnet: nn.Module, backend: str) -> None:
    if backend == "xformers":
        try:
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
            print("[build] xformers memory-efficient attention enabled")
            return
        except Exception as e:
            print(f"[build] xformers unavailable ({e}), falling back to SDPA")


# ── Trainable parameter groups ────────────────────────────────────────────────

def get_trainable_params(
    unet: nn.Module,
    controlnet: nn.Module,
    image_proj=None,
) -> list[nn.Parameter]:
    from src.distributed import unwrap_module
    params = list(unwrap_module(controlnet).parameters())
    if image_proj is not None:
        params += list(unwrap_module(image_proj).parameters())
    for proc in unwrap_module(unet).attn_processors.values():
        if isinstance(proc, IPAttnProcessor2_0):
            params += list(proc.parameters())
    return params


# ── Full build ────────────────────────────────────────────────────────────────

def build_all(cfg: DictConfig, device: torch.device) -> dict:
    """
    Build all model components. Returns a dict with keys:
      vae, unet, controlnet, train_scheduler, infer_scheduler,
      tokenizer, text_encoder,
      clip_image_encoder (if use_ip_adapter), image_proj (if use_ip_adapter)
    """
    model_id       = cfg.model.base_model_id
    controlnet_id  = cfg.model.controlnet_ckpt_id
    unfreeze_unet  = cfg.model.get("unfreeze_unet", False)
    attention_bk   = cfg.model.attention_backend
    grad_ckpt      = cfg.model.use_gradient_checkpointing
    use_ip_adapter = cfg.model.get("use_ip_adapter", True)

    print("[build] Loading VAE …")
    vae = build_vae(model_id, device)

    print("[build] Loading UNet …")
    unet = build_unet(model_id, device, unfreeze_unet=unfreeze_unet)

    print(f"[build] Loading ControlNet from {controlnet_id} (native 3ch depth input) …")
    controlnet = build_controlnet(controlnet_id, device)

    print("[build] Loading CLIP text encoder …")
    tokenizer, text_encoder = build_text_encoder(model_id, device)

    print("[build] Loading noise schedulers …")
    train_scheduler, infer_scheduler = build_schedulers(model_id)

    result = {
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "train_scheduler": train_scheduler,
        "infer_scheduler": infer_scheduler,
    }

    if use_ip_adapter:
        clip_id  = cfg.model.get("clip_image_model_id", "openai/clip-vit-large-patch14")
        ip_scale = cfg.model.get("ip_scale", 1.0)

        print(f"[build] Loading CLIP vision encoder from {clip_id} …")
        clip_image_encoder = build_clip_image_encoder(clip_id, device)

        print("[build] Patching UNet cross-attention with IP-Adapter processors …")
        set_ip_adapter_processors(unet, cross_attention_dim=768, ip_scale=ip_scale)

        print("[build] Patching UNet decoder self-attention with reference processors …")
        set_reference_attn_processors(unet)

        # Processors were registered after unet.to(device) — move them now.
        unet.to(device)

        print("[build] Building Resampler (IP-Adapter+, 16 queries) …")
        image_proj = build_resampler(device)

        n_ip  = sum(1 for p in unet.attn_processors.values() if isinstance(p, IPAttnProcessor2_0))
        n_ref = sum(1 for p in unet.attn_processors.values() if isinstance(p, ReferenceAttnProcessor))
        print(f"[build] IP-Adapter: {n_ip} cross-attn layers  |  Reference: {n_ref} self-attn layers")

        result["clip_image_encoder"] = clip_image_encoder
        result["image_proj"] = image_proj

    if grad_ckpt:
        print("[build] Enabling gradient checkpointing …")
        enable_gradient_checkpointing(unet, controlnet)

    if attention_bk != "none":
        enable_efficient_attention(unet, controlnet, attention_bk)

    amp_dtype = resolve_amp_dtype(cfg.model.amp_dtype)
    print(f"[build] AMP dtype = {amp_dtype}")

    return result


# ── Checkpoint save / load ────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    unet: nn.Module,
    controlnet: nn.Module,
    image_proj,
    optimizer,
    scaler,
    lr_scheduler,
    step: int,
    epoch: int,
) -> None:
    from src.distributed import unwrap_module
    ckpt = {
        "step": step,
        "epoch": epoch,
        "controlnet": unwrap_module(controlnet).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    if image_proj is not None:
        ckpt["image_proj"] = unwrap_module(image_proj).state_dict()
    ip_proc_state = {
        name: proc.state_dict()
        for name, proc in unwrap_module(unet).attn_processors.items()
        if isinstance(proc, IPAttnProcessor2_0)
    }
    if ip_proc_state:
        ckpt["ip_processors"] = ip_proc_state
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    unet: nn.Module,
    controlnet: nn.Module,
    image_proj,
    optimizer,
    scaler,
    lr_scheduler,
    device: torch.device,
) -> tuple[int, int]:
    from src.distributed import unwrap_module
    ckpt = torch.load(path, map_location=device)
    unwrap_module(controlnet).load_state_dict(ckpt["controlnet"])
    if image_proj is not None and "image_proj" in ckpt:
        unwrap_module(image_proj).load_state_dict(ckpt["image_proj"])
    if "ip_processors" in ckpt:
        procs = unwrap_module(unet).attn_processors
        for name, state in ckpt["ip_processors"].items():
            if name in procs and isinstance(procs[name], IPAttnProcessor2_0):
                procs[name].load_state_dict(state)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0), ckpt.get("epoch", 0)

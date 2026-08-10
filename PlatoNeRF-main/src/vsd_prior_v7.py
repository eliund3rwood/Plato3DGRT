"""
src/vsd_prior_v7.py — VSD/SDS diffusion prior backed by a PlatoControlNet **V7
stage-1** checkpoint.

Kept SEPARATE from vsd_prior.py rather than switched inside it: that module
drives the V3/V4-era `final.pt` and is the only VSD path with real training
history behind it (see VSD_HANDOFF.md run1-run5). V7's architecture differs in
four ways that each reach into `step()`, so a flag would have meant four
branches through the hot path of the working baseline. Nothing here changes
vsd_prior.py.

## What V7 needs that V3/V4 did not

1. **8-channel `conv_in`.** V7 concatenates a spatially-aligned reference-latent
   "splat" onto the 4 noise channels (`latent_splat.render_latent_map` ->
   `train._unet_input`). Passing a bare 4-channel latent is a shape error.
2. **Per-view GEOMETRY, at every call.** The splat is computed by unprojecting
   the reference's latents with its depth+pose and reprojecting into each
   target's camera. So `step()` needs each novel view's RAW METRIC depth, w2c
   and K — not just the [0,1] ControlNet depth image V3/V4 took. Same for the
   reference, but that is fixed for a run, so it goes in
   `set_reference_geometry()`.
3. **Reference K/V, not the old reference-attention.** V7's path
   (`models/reference_kv.py`) captures the bank by running the UNet once on the
   clean reference at t=0 with ZEROED hint channels, and is driven by
   `set_reference_mode(capture/inject/n_views)` rather than by poking
   `proc._bank_k` directly. It also broadcasts one scene's bank across N views
   internally, which is exactly VSD's shape (1 scene, K novel poses).
4. **The triplane bottleneck**, which pools features across the N jointly
   processed views into one 3D volume. It needs geometry installed immediately
   before each forward and cleared immediately after.

## The trap that made this more than a port

**CFG must run SEQUENTIALLY, never as a doubled batch.** vsd_prior.py does
`z_t.repeat(2)` and splits the result — standard, and fine for V3/V4. Under V7
the triplane would receive a 2*K batch and pool the unconditional and
conditional halves into ONE 3D volume, silently mixing them; V7's own
`generate_ring` documents this and takes the sequential path for the same
reason (plus D-045's OOM). So this module runs the two branches as two forwards.
That is the single most important difference from the file it is modelled on.

## Mapping VSD onto V7's batch convention

V7 trains as B scenes x N views, jointly denoised at one shared timestep. VSD
renders K novel poses of ONE scene per step. So B=1, N=K — a natural fit, and
the reason `--vsd_batch_size` doubles as V7's `n_views` here. V7's reference-K/V
path asserts B==1, which VSD always satisfies.

One real mismatch: V7 samples ONE timestep per scene (all N views share it),
while VSD's `step()` samples one per view. Sharing is kept here, since the
triplane and reference-K/V paths were both trained under it.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_IMG_SIZE = 512
_VAE_SCALE = 0.18215
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_DEFAULT_CONTROLNET_ROOT = "/home/tzofi/orcd/scratch/eli/platocontrolnet"


def _ensure_on_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


class V7DiffusionPrior:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        controlnet_repo_root: str = _DEFAULT_CONTROLNET_ROOT,
        variant: str = "vsd",
        guidance_scale: float = 3.0,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_lr: float = 1e-4,
        t_min_frac: float = 0.02,
        t_max_frac: float = 0.75,
        grad_weight: str = "snr",
        model_overrides: list[str] | None = None,
    ):
        assert variant in ("vsd", "sds"), f"unknown variant: {variant}"
        _ensure_on_path(controlnet_repo_root)

        from src.models.build import build_all, load_checkpoint
        from src.distributed import resolve_amp_dtype, unwrap_module
        from src.models.triplane import set_triplane_geometry, clear_triplane_geometry
        from src.models.reference_kv import set_reference_mode
        from src.models.latent_splat import render_latent_map
        from src.train import _unet_input

        self._unwrap = unwrap_module
        self._set_triplane_geometry = set_triplane_geometry
        self._clear_triplane_geometry = clear_triplane_geometry
        self._set_reference_mode = set_reference_mode
        self._render_latent_map = render_latent_map
        self._unet_input = _unet_input

        self.device = device
        self.variant = variant
        self.guidance_scale = guidance_scale
        self.t_min_frac = t_min_frac
        self.t_max_frac = t_max_frac
        self.grad_weight = grad_weight

        cfg = OmegaConf.load(os.path.join(controlnet_repo_root, "configs", "model.yaml"))
        train_cfg_path = os.path.join(controlnet_repo_root, "configs", "train.yaml")
        if os.path.exists(train_cfg_path):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(train_cfg_path))
        # The checkpoint's architecture is NOT recorded in the checkpoint, so it
        # has to be declared. archive/k2/refkv_step_0071000.pt is
        # use_reference_kv=true + use_triplane=true; a mismatch here surfaces as
        # a confusing load_state_dict error at best and silently-missing
        # mechanisms at worst.
        if model_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(model_overrides))
        self.cfg = cfg
        self.amp_dtype = resolve_amp_dtype(cfg.model.amp_dtype)
        self.use_ref_kv = bool(cfg.model.get("use_reference_kv", False))

        print(f"[vsd_prior_v7] Building V7 prior (variant={variant}, "
              f"amp={self.amp_dtype}, use_reference_kv={self.use_ref_kv}, "
              f"use_triplane={cfg.model.get('use_triplane', False)}, "
              f"unet_in_channels={cfg.model.get('unet_in_channels', 8)}) "
              f"from {controlnet_repo_root} ...")
        self.components = build_all(cfg, device)
        load_checkpoint(
            checkpoint_path,
            self.components["unet"], self.components["controlnet"],
            self.components.get("image_proj"), None, None, None, device,
            ref_slot_embed=self.components.get("ref_slot_embed"),
        )

        for key in ("vae", "unet", "controlnet", "text_encoder"):
            self.components[key].eval().requires_grad_(False)
        for key in ("clip_image_encoder", "image_proj", "ref_slot_embed"):
            if self.components.get(key) is not None:
                self.components[key].eval().requires_grad_(False)

        # V7's build_all turns gradient checkpointing ON (model.yaml
        # use_gradient_checkpointing) because TRAINING backprops into conv_in,
        # the triplane and the IP projections, at N=16 views. VSD does neither:
        # the only trainable tensors here are the LoRA particle weights, and
        # every input to the UNet is detached (z_t_in, text, depth). Reentrant
        # checkpointing given no grad-requiring INPUT returns activations with
        # no grad_fn regardless of whether parameters inside require grad, so
        # loss_lora.backward() dies with "element 0 of tensors does not require
        # grad". vsd_prior.py never hit this because the V3 build left
        # checkpointing off.
        # Called UNCONDITIONALLY, not gated on a flag read off the top-level
        # module: diffusers stores `gradient_checkpointing` on the individual
        # blocks, so `getattr(unet, "gradient_checkpointing", False)` is always
        # False and a guarded version silently skips the disable. It did.
        for _m in ("unet", "controlnet"):
            _mod = self._unwrap(self.components[_m])
            if hasattr(_mod, "disable_gradient_checkpointing"):
                _mod.disable_gradient_checkpointing()
            _n_ckpt = sum(1 for _sm in _mod.modules()
                          if getattr(_sm, "gradient_checkpointing", False))
            print(f"[vsd_prior_v7] {_m}: gradient checkpointing disabled "
                  f"({_n_ckpt} submodules still flagged; expected 0)")
            assert _n_ckpt == 0, (
                f"{_m} still has {_n_ckpt} submodules with gradient "
                "checkpointing on after disable_gradient_checkpointing(); the "
                "LoRA backward will have no graph to walk")

        self.unet = self.components["unet"]
        self.lora_optimizer = None
        if self.variant == "vsd":
            self._add_lora(rank=lora_rank, alpha=lora_alpha, lr=lora_lr)

        self._ref_cache = None
        self._ref_geom = None
        print("[vsd_prior_v7] Ready. Call set_reference_image() AND "
              "set_reference_geometry() before step().")

    # ------------------------------------------------------------------
    def set_timestep_range(self, t_min_frac: float, t_max_frac: float) -> None:
        self.t_min_frac = t_min_frac
        self.t_max_frac = t_max_frac

    # ------------------------------------------------------------------
    def _add_lora(self, rank: int, alpha: int, lr: float) -> None:
        try:
            from peft import LoraConfig
        except ImportError as e:
            raise ImportError(
                "variant='vsd' requires `peft` for the LoRA particle network "
                "(pip install peft). Use variant='sds' to run without it."
            ) from e
        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
        )
        self.unet.add_adapter(lora_config, adapter_name="vsd_particle")
        lora_params = [p for p in self.unet.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in lora_params)
        print(f"[vsd_prior_v7] LoRA particle adapter: rank={rank}, "
              f"{n_params:,} trainable params")
        self.lora_optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_reference_geometry(self, D_ref_metric, w2c_ref, K_ref) -> None:
        """
        The reference view's geometry, in PlatoControlNet's convention already
        (z-depth, OpenCV w2c) — convert with `PlatoControlNet/src/models/
        pose_convert.py` before calling, since Plato3DGRT natively produces
        Euclidean ray distance in NeRF/OpenGL poses and neither difference is
        visible in a tensor's shape.

        Each argument: (512,512) / (4,4) / (3,3), any float dtype.
        """
        def _t(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            return x
        D = _t(D_ref_metric)
        assert D.shape == (_IMG_SIZE, _IMG_SIZE), \
            f"D_ref_metric {tuple(D.shape)} != ({_IMG_SIZE},{_IMG_SIZE})"
        K = _t(K_ref)
        assert float(K[0, 2]) > 64, (
            f"K_ref principal point cx={float(K[0,2]):.1f} looks like a "
            "latent-resolution intrinsic; pass K at 512x512")
        self._ref_geom = {"D": D.unsqueeze(0), "w2c": _t(w2c_ref).unsqueeze(0),
                          "K": K.unsqueeze(0)}
        print("[vsd_prior_v7] Reference geometry set.")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_reference_image(self, I_A: torch.Tensor) -> None:
        """I_A: 1x3x512x512 in [-1,1]. Caches text embeds, IP tokens and the
        reference VAE latent. The reference-K/V BANK is not cached here — V7
        re-captures it every forward (see `_capture_ref_kv`), which is what makes
        D-048's stale-bank bug structurally unreachable."""
        from src.train import get_text_embeds

        vae = self.components["vae"]
        tokenizer, text_encoder = self.components["tokenizer"], self.components["text_encoder"]
        clip_enc = self.components.get("clip_image_encoder")
        image_proj = self.components.get("image_proj")

        assert I_A.shape[-2:] == (_IMG_SIZE, _IMG_SIZE), f"I_A shape {I_A.shape}"
        assert I_A.shape[0] == 1, (
            f"V7's reference-K/V path asserts B==1 (got {I_A.shape[0]}); VSD "
            "always has exactly one scene, so this should not fire")

        text_embeds = get_text_embeds(tokenizer, text_encoder, [""], self.device)
        text_embeds = text_embeds.to(self.amp_dtype)

        ip_tokens = None
        if clip_enc is not None and image_proj is not None:
            clip_in = F.interpolate(I_A.float(), size=(224, 224), mode="bicubic",
                                    align_corners=False)
            clip_in = (clip_in.clamp(-1, 1) + 1) / 2
            mean = torch.tensor(_CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(_CLIP_STD, device=self.device).view(1, 3, 1, 1)
            clip_in = (clip_in - mean) / std
            clip_tokens = clip_enc(clip_in).last_hidden_state
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                ip_tokens = self._unwrap(image_proj)(clip_tokens.to(self.amp_dtype))

        z_A = vae.encode(I_A.float()).latent_dist.mode() * _VAE_SCALE

        self._ref_cache = {"text_embeds": text_embeds, "ip_tokens": ip_tokens,
                           "z_ref": z_A.float()}
        print("[vsd_prior_v7] Reference image cached.")

    # ------------------------------------------------------------------
    def _capture_ref_kv(self, n_views: int, inject: bool) -> None:
        """V7's reference-K/V capture, mirroring train._capture_reference_kv:
        one UNet pass on the CLEAN reference at t=0 with zeroed hint channels,
        then switch to inject. Re-captured per forward rather than cached, so a
        bank can never go stale against a different batch shape."""
        if not self.use_ref_kv:
            return
        unet = self._unwrap(self.unet)
        z_ref = self._ref_cache["z_ref"].to(self.amp_dtype)
        text_B = self._ref_cache["text_embeds"]
        in_ch = int(self.cfg.model.get("unet_in_channels", 8))
        pad = torch.zeros(1, in_ch - z_ref.shape[1], *z_ref.shape[2:],
                          device=z_ref.device, dtype=z_ref.dtype)
        self._set_reference_mode(unet, capture=True, inject=False, n_views=n_views)
        try:
            with torch.no_grad():
                unet(torch.cat([z_ref, pad], dim=1),
                     torch.zeros(1, device=self.device, dtype=torch.long),
                     encoder_hidden_states=text_B).sample
        finally:
            self._set_reference_mode(unet, capture=False, inject=inject,
                                     n_views=n_views)

    def _clear_ref_kv(self) -> None:
        if not self.use_ref_kv:
            return
        from src.models.reference_kv import clear_reference_bank
        clear_reference_bank(self._unwrap(self.unet))

    # ------------------------------------------------------------------
    def _splat(self, D_tgt_metric, w2c_tgt, K_tgt, N):
        """Reference latents warped into each target view -> conv_in[4:8]."""
        g = self._ref_geom
        z_ref = self._ref_cache["z_ref"]

        def _per_view(x):
            return x.expand(N, *x.shape[1:])

        with torch.autocast(device_type="cuda", enabled=False):
            rendered, coverage = self._render_latent_map(
                z_refs=[_per_view(z_ref).float()],
                ref_depths_metric=[_per_view(g["D"]).float()],
                ref_w2cs=[_per_view(g["w2c"]).float()],
                ref_Ks=[_per_view(g["K"]).float()],
                tgt_depth_metric=D_tgt_metric.float(),
                tgt_w2c=w2c_tgt.float(),
                tgt_K=K_tgt.float(),
                mode=self.cfg.model.get("latent_render_mode", "reproject"),
                depth_tol=self.cfg.train.get("latent_render_tol", 0.10)
                if "train" in self.cfg else 0.10,
            )
        return rendered.to(self.amp_dtype), coverage.to(self.amp_dtype)

    def _unet_forward(self, z_t_in, t, text, ip, depth_cond, rendered, coverage,
                      D_tgt_metric, w2c_tgt, K_tgt, N):
        """One conditioned forward: ControlNet + reference-K/V + triplane + the
        8-channel conv_in input. Geometry is installed immediately before and
        cleared immediately after, per triplane.set_triplane_geometry's
        contract — a stale payload applied to the next batch is worse than none."""
        controlnet = self.components["controlnet"]
        unet = self._unwrap(self.unet)

        down_res, mid_res = controlnet(
            z_t_in, t, encoder_hidden_states=text,
            controlnet_cond=depth_cond, return_dict=False,
        )
        self._capture_ref_kv(n_views=N, inject=True)
        self._set_triplane_geometry(unet, D_tgt_metric, w2c_tgt, K_tgt, N)
        try:
            cross_kw = {"ip_hidden_states": ip} if ip is not None else None
            eps = unet(
                self._unet_input(z_t_in, rendered, coverage, self.cfg), t,
                encoder_hidden_states=text,
                cross_attention_kwargs=cross_kw,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample
        finally:
            self._clear_triplane_geometry(unet)
            self._clear_ref_kv()
        return eps

    # ------------------------------------------------------------------
    def step(self, rgb_512, depth_512_3ch, D_tgt_metric, w2c_tgt, K_tgt):
        """
        rgb_512:        Kx3x512x512 in [-1,1], differentiable w.r.t. the render.
        depth_512_3ch:  Kx3x512x512 in [0,1], ControlNet-normalised depth.
        D_tgt_metric:   Kx512x512 RAW METRIC z-depth (V7 convention).
        w2c_tgt:        Kx4x4 OpenCV world-to-camera.
        K_tgt:          Kx3x3 at 512x512.

        The three geometry arguments are what V7 needs and V3/V4 did not: they
        drive both the reference splat and the triplane. Returns
        (loss, metrics), same surrogate-loss contract as vsd_prior.DiffusionPrior.
        """
        assert self._ref_cache is not None, "call set_reference_image() first"
        assert self._ref_geom is not None, "call set_reference_geometry() first"

        vae = self.components["vae"]
        train_sched = self.components["train_scheduler"]
        unet = self._unwrap(self.unet)
        N = rgb_512.shape[0]

        text = self._ref_cache["text_embeds"].expand(N, -1, -1)
        ip_tokens = self._ref_cache["ip_tokens"]
        ip = ip_tokens.expand(N, -1, -1) if ip_tokens is not None else None

        D_tgt_metric = D_tgt_metric.to(self.device).float()
        w2c_tgt = w2c_tgt.to(self.device).float()
        K_tgt = K_tgt.to(self.device).float()

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            z0 = vae.encode(rgb_512.to(self.amp_dtype)).latent_dist.sample() * _VAE_SCALE
        z0 = z0.float()

        num_train_t = train_sched.config.num_train_timesteps
        t_lo = int(self.t_min_frac * num_train_t)
        t_hi = max(t_lo + 1, int(self.t_max_frac * num_train_t))
        # ONE timestep shared across the N views, matching how V7 was trained
        # (a scene draws one t; the triplane and reference-K/V paths have only
        # ever seen views that agree on it).
        t_scene = torch.randint(t_lo, t_hi, (1,), device=self.device)
        t = t_scene.expand(N)
        noise = torch.randn_like(z0)
        z_t = train_sched.add_noise(z0, noise, t)
        z_t_in = z_t.detach().to(self.amp_dtype)

        alphas_cumprod = train_sched.alphas_cumprod.to(self.device)
        w = (torch.ones_like(t, dtype=torch.float32) if self.grad_weight == "uniform"
             else (1.0 - alphas_cumprod[t]).float())
        w = w.view(-1, 1, 1, 1)

        depth_in = depth_512_3ch.to(self.amp_dtype)
        metrics = {"t_mean": float(t.float().mean().item())}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            rendered, coverage = self._splat(D_tgt_metric, w2c_tgt, K_tgt, N)
            metrics["splat_cov"] = float(coverage.float().mean().item())

            if self.variant == "vsd":
                unet.disable_adapters()

            # SEQUENTIAL CFG. A doubled batch would hand the triplane 2N views
            # of "one scene" and pool the uncond and cond halves into a single
            # 3D volume — V7's own generate_ring takes the sequential path for
            # exactly this reason (and D-045's OOM).
            eps_cond = self._unet_forward(
                z_t_in, t, text, ip, depth_in, rendered, coverage,
                D_tgt_metric, w2c_tgt, K_tgt, N)
            # The null branch is "depth only": zeroed IP tokens AND a zeroed
            # hint, matching the conditioning dropout V7 trained its CFG null
            # branch against (configs/train.yaml cond_dropout_prob).
            eps_uncond = self._unet_forward(
                z_t_in, t, text,
                torch.zeros_like(ip) if ip is not None else None,
                depth_in, torch.zeros_like(rendered), torch.zeros_like(coverage),
                D_tgt_metric, w2c_tgt, K_tgt, N)
            eps_pretrained = (
                eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)).float()

        if self.variant == "sds":
            grad = w * (eps_pretrained - noise)
            target = (z0 - grad).detach()
            loss = 0.5 * F.mse_loss(z0, target, reduction="sum") / N
            metrics["loss_lora"] = 0.0
            metrics["loss_vsd"] = float(loss.item())
            return loss, metrics

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            unet.enable_adapters()
            if not any(p.requires_grad for p in unet.parameters()):
                raise RuntimeError(
                    "no UNet parameter requires grad after enable_adapters() -- "
                    "the LoRA adapter is not active, so there is nothing for "
                    "the particle network to train. This is a DIFFERENT failure "
                    "from gradient checkpointing severing the graph.")
            eps_lora = self._unet_forward(
                z_t_in, t, text, ip, depth_in, rendered, coverage,
                D_tgt_metric, w2c_tgt, K_tgt, N)

        assert eps_lora.requires_grad, (
            "the LoRA forward produced a tensor with no grad_fn, so there is "
            "nothing to train. Usual cause: gradient checkpointing is on while "
            "every UNet input is detached -- reentrant checkpointing then "
            "returns activations detached from the graph even though the LoRA "
            "parameters require grad. __init__ disables it for exactly this "
            "reason; check that it actually took effect.")
        loss_lora = F.mse_loss(eps_lora.float(), noise.float())
        self.lora_optimizer.zero_grad(set_to_none=True)
        loss_lora.backward()
        self.lora_optimizer.step()
        unet.disable_adapters()

        grad = w * (eps_pretrained - eps_lora.detach().float())
        target = (z0 - grad).detach()
        loss = 0.5 * F.mse_loss(z0, target, reduction="sum") / N

        metrics["loss_lora"] = float(loss_lora.item())
        metrics["loss_vsd"] = float(loss.item())
        return loss, metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def preview(self, depth_512_3ch, D_tgt_metric, w2c_tgt, K_tgt,
                num_steps: int = 20) -> torch.Tensor:
        """Full multi-step denoise -> predicted RGB, for qualitative panels.
        Same conditioning as step(), including sequential CFG."""
        assert self._ref_cache is not None and self._ref_geom is not None
        vae = self.components["vae"]
        sched = self.components["infer_scheduler"]
        unet = self._unwrap(self.unet)
        N = depth_512_3ch.shape[0]

        D_tgt_metric = D_tgt_metric.to(self.device).float()
        w2c_tgt = w2c_tgt.to(self.device).float()
        K_tgt = K_tgt.to(self.device).float()

        text = self._ref_cache["text_embeds"].expand(N, -1, -1)
        ip_tokens = self._ref_cache["ip_tokens"]
        ip = ip_tokens.expand(N, -1, -1) if ip_tokens is not None else None
        depth_in = depth_512_3ch.to(self.amp_dtype)

        if self.variant == "vsd":
            unet.disable_adapters()

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            rendered, coverage = self._splat(D_tgt_metric, w2c_tgt, K_tgt, N)
            sched.set_timesteps(num_steps, device=self.device)
            z = torch.randn(N, 4, _IMG_SIZE // 8, _IMG_SIZE // 8,
                            device=self.device, dtype=self.amp_dtype)
            for t_i in sched.timesteps:
                t_b = t_i.repeat(N).to(self.device)
                eps_c = self._unet_forward(z, t_b, text, ip, depth_in,
                                           rendered, coverage,
                                           D_tgt_metric, w2c_tgt, K_tgt, N)
                eps_u = self._unet_forward(
                    z, t_b, text,
                    torch.zeros_like(ip) if ip is not None else None,
                    depth_in, torch.zeros_like(rendered), torch.zeros_like(coverage),
                    D_tgt_metric, w2c_tgt, K_tgt, N)
                eps = eps_u + self.guidance_scale * (eps_c - eps_u)
                z = sched.step(eps, t_i, z).prev_sample
            img = vae.decode(z / _VAE_SCALE).sample.clamp(-1, 1)
        return img

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """LoRA weights + its optimizer state, for checkpoint/resume.

        Deliberately IDENTICAL to vsd_prior.DiffusionPrior's (plain key filter,
        not peft's get_peft_model_state_dict) — the training script saves and
        reloads these under one filename regardless of which prior built them,
        so a different key layout here would produce checkpoints that load
        `strict=False` into nothing and silently resume with a fresh LoRA."""
        if self.variant != "vsd":
            return {}
        unet = self._unwrap(self.unet)
        lora_sd = {k: v for k, v in unet.state_dict().items() if "lora" in k}
        return {"lora_state_dict": lora_sd,
                "lora_optimizer": self.lora_optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if self.variant != "vsd" or not state:
            return
        unet = self._unwrap(self.unet)
        unet.load_state_dict(state["lora_state_dict"], strict=False)
        self.lora_optimizer.load_state_dict(state["lora_optimizer"])

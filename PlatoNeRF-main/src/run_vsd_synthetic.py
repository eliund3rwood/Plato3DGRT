#!/usr/bin/env python
"""
src/run_vsd_synthetic.py — run the Phase-3 VSD texture stage on a SYNTHETIC
scene's fitted, textureless 3DGS geometry. The control for "does VSD work on
clean geometry?"

WHY THIS IS A SEPARATE SCRIPT AND NOT A FLAG ON run_platonerf_3dgrt_vsd.py
-------------------------------------------------------------------------
That script's Phase 1/2 is not a stage that can be skipped — it is the shape of
the whole training loop. `load_tof_data` is unconditional, and the loop body
slices `tof`, `light_inters`, `light_dists`, `noise`, `walls_cam` and the ToF
ray array every iteration, reshuffling them at each epoch boundary. A synthetic
scene has none of those. Threading a mode flag through all of it would mean
editing the one file whose Phase 1/2 the design notes describe as
"byte-identical, always runs", for a change that cannot be GPU-tested before it
runs, on the script that owns the only validated geometry result in the
project. The cost of a mistake there is a corrupted 14-hour run; the cost of a
mistake here is this experiment only.

So the chair path is left literally untouched, and everything that CAN be
shared is imported rather than copied:

    create_3dgrt_conf      the model/optimizer/strategy config
    render_rays_3dgrt      the tracer wrapper, incl. its ray layout contract
    _save_vsd_preview      the I_A | D_B | render | prediction panel
    _save_lpips_ref_preview
    DiffusionPrior / V7DiffusionPrior, load_reference_image, make_depth_cond

What is RESTATED here is the Phase-3 loop body itself (~120 lines): LR decay,
timestep annealing, pose sampling, the batched render, the prior step, the
smoothness term, LPIPS grounding, the EMA, and checkpointing. That is a real
drift risk and it is the price of not touching the chair path. Every restated
piece keeps the same flag names and the same defaults, and the three fixes the
handoff records as hard-won are reproduced deliberately and marked:

  * base colour LR is read from the CONFIG, never from the optimizer's restored
    param_groups, so a resumed run does not compound the decay;
  * depth conditioning is rendered ONCE and frozen, never re-rendered live;
  * `--i_weights` is REQUIRED here rather than falling back to a large default
    that leaves a long run almost never checkpointing.

WHAT DIFFERS FROM THE CHAIR, AND WHY
------------------------------------
1. NO ToF LOSS. There is no ToF data for a synthetic scene, so geometry has no
   physical grounding term. Geometry is therefore frozen HARD for the entire
   run — position, density, rotation, scale — not merely gated around the VSD
   render as in the chair path. That is also what makes this a clean control:
   the geometry under test stays exactly the geometry that was measured.

2. POSES COME FROM THE SCENE. `utils/novel_views.orbit_poses` is hardcoded to
   the chair's world frame. Here the orbit is fitted from the scene's own
   cameras.json (`utils.synthetic_scene.fit_ring`, validated to regenerate the
   24 input poses to 0.017 deg) and rays are built with `rays_from_w2c`.

3. NO NeRF->OpenCV FLIP. cameras.json is already OpenCV w2c, so V7 is handed
   those matrices unchanged. `pose_convert.nerf_c2w_to_opencv_w2c` exists for
   the chair's NeRF poses and applying it here by analogy is a silent
   inside-out render (scripts/test_02 asserts it drifts 512 px).
   `ray_distance_to_z_depth` IS still applied — that one is about 3DGRT's
   ToF-native renderer, not about the scene.

4. THE REFERENCE IS ONE OF THE SCENE'S OWN RING VIEWS, at `--vsd_ref_index`,
   supplied by scripts/extract_ring_reference.py. The chair's equivalent is
   dolly pose 30 / chair_smooth_walls.png.

Usage (cluster, GPU node) — see the handoff for the sbatch form.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from threedgrut.model.model import MixtureOfGaussians          # noqa: E402
from run_platonerf_3dgrt_vsd import (                          # noqa: E402
    create_3dgrt_conf, render_rays_3dgrt,
    _save_vsd_preview, _save_lpips_ref_preview,
)
from utils.synthetic_scene import (                            # noqa: E402
    load_scene_cameras, scale_intrinsics, rays_from_w2c, fit_ring,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--expname", required=True)
    p.add_argument("--basedir", default="./logs/")
    p.add_argument("--cameras", required=True, help="the scene's cameras.json")
    p.add_argument("--ft_path", default=None,
                   help="explicit checkpoint; default is the newest in logs/<expname>/")
    p.add_argument("--N_iters", type=int, default=35000,
                   help="Phase-3 starts after this; the converted checkpoint stores "
                        "global_step=35000 so the first VSD iteration is 35001")

    # --- VSD flags: same names and defaults as run_platonerf_3dgrt_vsd.py ---
    p.add_argument("--vsd_iters", type=int, required=True,
                   help="RELATIVE to --N_iters, matching the chair script's semantics")
    p.add_argument("--vsd_checkpoint", type=str, required=True)
    p.add_argument("--vsd_controlnet_root", type=str, default=None)
    p.add_argument("--vsd_arch", type=str, default="v7", choices=["v3", "v7"])
    p.add_argument("--vsd_model_overrides", type=str, default="")
    p.add_argument("--vsd_variant", type=str, default="vsd", choices=["vsd", "sds"])
    p.add_argument("--vsd_ref_image", type=str, required=True,
                   help="one of THIS scene's ring views (extract_ring_reference.py)")
    p.add_argument("--vsd_ref_index", type=int, default=0,
                   help="which ring view --vsd_ref_image is. Attaching the reference "
                        "to the wrong camera is silent and looks fine.")
    p.add_argument("--vsd_weight", type=float, default=1.0)
    p.add_argument("--vsd_guidance_scale", type=float, default=3.0)
    p.add_argument("--vsd_lora_rank", type=int, default=8)
    p.add_argument("--vsd_lora_lr", type=float, default=1e-4)
    p.add_argument("--vsd_t_min_frac", type=float, default=0.02)
    p.add_argument("--vsd_t_max_frac_start", type=float, default=0.98)
    p.add_argument("--vsd_t_max_frac", type=float, default=0.5)
    p.add_argument("--vsd_grad_weight", type=str, default="snr", choices=["snr", "uniform"])
    p.add_argument("--vsd_render_res", type=int, default=512)
    p.add_argument("--vsd_n_orbit_poses", type=int, default=100)
    p.add_argument("--vsd_batch_size", type=int, default=8)
    p.add_argument("--vsd_every_n_steps", type=int, default=1)
    p.add_argument("--vsd_color_lr_decay", type=float, default=0.1)
    p.add_argument("--vsd_smooth_weight", type=float, default=1000.0)
    p.add_argument("--vsd_lpips_weight", type=float, default=10.0)
    p.add_argument("--vsd_lpips_every_n_steps", type=int, default=4)
    p.add_argument("--vsd_color_ema_decay", type=float, default=0.999)
    p.add_argument("--vsd_preview_steps", type=int, default=20)

    # Required, not defaulted: the handoff records a 47-minute run that
    # checkpointed zero times because this silently fell back to a large value.
    p.add_argument("--i_weights", type=int, required=True,
                   help="checkpoint/preview cadence. REQUIRED on purpose.")
    p.add_argument("--i_print", type=int, default=25)
    p.add_argument("--render_chunk", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = config_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # NOTE: unlike run_platonerf_3dgrt_vsd.py's __main__, this script never sets
    # torch.set_default_tensor_type('torch.cuda.FloatTensor'). That global only
    # existed so PlatoNeRF's ToF loader's bare torch.Tensor(...) calls landed on
    # GPU; there is no ToF loader here. Leaving the default alone sidesteps the
    # diffusers/scheduler breakage that global caused entirely, rather than
    # setting it and then working around it.

    savedir = os.path.join(args.basedir, args.expname)
    img_savedir = os.path.join(savedir, "progress")
    os.makedirs(img_savedir, exist_ok=True)

    # ---------------------------------------------------------------- scene --
    cams = load_scene_cameras(args.cameras)
    ring = fit_ring(cams)
    R = args.vsd_render_res
    K_vsd = scale_intrinsics(cams.K, cams.width, cams.height, R).astype(np.float32)
    K_vsd_t = torch.from_numpy(K_vsd).to(device)
    print(f"[VSD-syn] {len(cams)} ring views; orbit fitted: target="
          f"{np.round(ring.target, 4).tolist()} radius={ring.radius:.4f} "
          f"height={ring.height:.4f}")
    print(f"[VSD-syn] intrinsics at {R}x{R}: focal={K_vsd[0,0]:.2f}")

    # ---------------------------------------------------------------- model --
    conf = create_3dgrt_conf(args)
    ckpt_dir = savedir
    if args.ft_path:
        ckpt_path = args.ft_path
    else:
        cands = sorted(f for f in os.listdir(ckpt_dir)
                       if f.endswith(".tar") and not f.endswith("_strategy.tar"))
        if not cands:
            sys.exit(f"no .tar checkpoint in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, cands[-1])
    print(f"[VSD-syn] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    scene_extent = float(ckpt.get("scene_extent") or ring.radius * 1.1)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
    model.init_from_checkpoint(ckpt, setup_optimizer=True)
    model.build_acc(rebuild=True)
    start = int(ckpt.get("global_step", args.N_iters))
    print(f"[VSD-syn] {model.num_gaussians:,} Gaussians, resuming at global_step={start}")

    # Geometry is frozen HARD for the whole run. In the chair script the freeze
    # is gated around the VSD render only, because the ToF loss still needs
    # gradient to geometry. There is no ToF loss here, so anything that reached
    # geometry would be pure VSD drift into an unregularised fit — and run4
    # showed that degrades it (visibly streaky depth maps). Freezing outright
    # also keeps the geometry under test identical to the geometry measured by
    # scripts/test_03 and test_04.
    for p_ in (model.positions, model.density, model.rotation, model.scale):
        p_.requires_grad_(False)
    model.features_specular.requires_grad_(False)   # --vsd_freeze_specular 1 equivalent
    model.features_albedo.requires_grad_(True)
    print("[VSD-syn] frozen: positions, density, rotation, scale, features_specular; "
          "training: features_albedo")

    # ---------------------------------------------------------------- prior --
    from vsd_prior import DiffusionPrior, load_reference_image, make_depth_cond
    use_v7 = (args.vsd_arch == "v7")
    if use_v7:
        from vsd_prior_v7 import V7DiffusionPrior

    prior_kwargs = dict(
        checkpoint_path=args.vsd_checkpoint, device=device, variant=args.vsd_variant,
        guidance_scale=args.vsd_guidance_scale, lora_rank=args.vsd_lora_rank,
        lora_lr=args.vsd_lora_lr, t_min_frac=args.vsd_t_min_frac,
        t_max_frac=args.vsd_t_max_frac_start, grad_weight=args.vsd_grad_weight,
    )
    if args.vsd_controlnet_root:
        prior_kwargs["controlnet_repo_root"] = args.vsd_controlnet_root
    if use_v7:
        prior_kwargs["model_overrides"] = (
            args.vsd_model_overrides.split() if args.vsd_model_overrides else None)
        vsd_prior = V7DiffusionPrior(**prior_kwargs)
    else:
        vsd_prior = DiffusionPrior(**prior_kwargs)

    I_A = load_reference_image(args.vsd_ref_image, device)
    vsd_prior.set_reference_image(I_A)
    if not (0 <= args.vsd_ref_index < len(cams)):
        sys.exit(f"--vsd_ref_index {args.vsd_ref_index} outside [0,{len(cams)})")
    ref_w2c = cams.w2c[args.vsd_ref_index].astype(np.float64)
    print(f"[VSD-syn] reference image {os.path.basename(args.vsd_ref_image)} "
          f"attached to ring view {args.vsd_ref_index} "
          f"({cams.frame_ids[args.vsd_ref_index]})")

    # ------------------------------------------------------------ pose bank --
    orbit = ring.orbit(args.vsd_n_orbit_poses)
    vsd_rays_gpu = [torch.from_numpy(rays_from_w2c(w, K_vsd, R)).to(device) for w in orbit]
    vsd_w2c_cache = [torch.from_numpy(w.astype(np.float32)).to(device) for w in orbit]
    print(f"[VSD-syn] {len(vsd_rays_gpu)} orbit poses cached at {R}x{R} "
          f"(batch_size={args.vsd_batch_size}, every_n_steps={args.vsd_every_n_steps})")

    def render_pose_rays(rays_t, train, frame_id):
        return render_rays_3dgrt(torch.transpose(rays_t, 0, 1), model,
                                 train=train, frame_id=frame_id)

    # ------------------------------------------------- frozen depth cache ----
    # Built ONCE, here, from geometry that is already final and frozen — the
    # chair script defers this to the first VSD iteration only because a
    # from-scratch run would otherwise cache conditioning from a random-init
    # point cloud. That cannot happen here: the geometry is loaded from a
    # finished fit and never changes.
    from src.models.pose_convert import ray_distance_to_z_depth as _ray_dist_to_z
    print(f"[VSD-syn] caching depth conditioning for {len(vsd_rays_gpu)} poses ...")
    vsd_depth_cache, vsd_metric_cache = [], []
    with torch.no_grad():
        for pose_rays in vsd_rays_gpu:
            _, _, acc_c, depth_c, _, _ = render_pose_rays(pose_rays, False, 0)
            depth_hw = depth_c.reshape(R, R)
            acc_hw = acc_c.reshape(R, R)
            vsd_depth_cache.append(make_depth_cond(depth_hw, acc_hw).squeeze(0))
            if use_v7:
                vsd_metric_cache.append(torch.from_numpy(
                    _ray_dist_to_z(depth_hw.detach().cpu().numpy(), K_vsd)).to(device))
    print(f"[VSD-syn] depth cache built ({len(vsd_depth_cache)} poses).")

    ref_rays = torch.from_numpy(rays_from_w2c(ref_w2c, K_vsd, R)).to(device)
    if use_v7:
        with torch.no_grad():
            _, _, _, depth_r, _, _ = render_pose_rays(ref_rays, False, 0)
        D_ref = _ray_dist_to_z(depth_r.reshape(R, R).detach().cpu().numpy(), K_vsd)
        # w2c passed through UNCHANGED — see the module docstring, point 3.
        vsd_prior.set_reference_geometry(D_ref, ref_w2c.astype(np.float32), K_vsd)
        print(f"[VSD-syn] v7 reference geometry set from ring view "
              f"{args.vsd_ref_index} (z-depth valid frac {float((D_ref > 0).mean()):.3f})")

    # ------------------------------------------------------------- grounding --
    vsd_lpips_fn = None
    if args.vsd_lpips_weight > 0:
        import lpips as lpips_lib
        vsd_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
        for q in vsd_lpips_fn.parameters():
            q.requires_grad_(False)
        print(f"[VSD-syn] LPIPS grounding on ring view {args.vsd_ref_index}, "
              f"weight={args.vsd_lpips_weight}, every {args.vsd_lpips_every_n_steps}")

    # ------------------------------------------------------------ optimiser --
    # Base LR from the CONFIG, never from optimizer.param_groups. On resume,
    # setup_optimizer(state_dict=...) restores the already-decayed LR; treating
    # that as 1.0x decays it again every restart, silently crushing colour LR
    # toward zero over a preemptable run.
    color_groups = [pg for pg in model.optimizer.param_groups
                    if pg["name"] in ("features_albedo", "features_specular")]
    color_base_lr = {n: conf.optimizer.params[n].lr
                     for n in ("features_albedo", "features_specular")}

    albedo_init = model.features_albedo.detach().clone()
    if "ema_albedo" in ckpt:
        ema_albedo = ckpt["ema_albedo"].to(device)
        ema_specular = ckpt["ema_specular"].to(device)
        print("[VSD-syn] resumed EMA colour weights from checkpoint")
    else:
        ema_albedo = albedo_init.clone()
        ema_specular = model.features_specular.detach().clone()

    vsd_state_path = ckpt_path.replace(".tar", "_vsd.pt")
    if os.path.isfile(vsd_state_path):
        vsd_prior.load_state_dict(torch.load(vsd_state_path, map_location=device))
        print(f"[VSD-syn] resumed VSD/LoRA state from {vsd_state_path}")

    # ------------------------------------------------------------------ loop --
    end = args.N_iters + args.vsd_iters
    print(f"[VSD-syn] training {start + 1} -> {end}")
    t0 = time.time()
    for i in trange(start + 1, end + 1):
        if (i - args.N_iters) % args.vsd_every_n_steps != 0:
            continue

        progress = min(max((i - args.N_iters) / max(args.vsd_iters, 1), 0.0), 1.0)
        for pg in color_groups:
            pg["lr"] = color_base_lr[pg["name"]] * (args.vsd_color_lr_decay ** progress)
        t_max_now = (args.vsd_t_max_frac_start
                     + (args.vsd_t_max_frac - args.vsd_t_max_frac_start) * progress)
        vsd_prior.set_timestep_range(args.vsd_t_min_frac, t_max_now)

        n_views = args.vsd_batch_size
        pose_idxs = np.random.randint(0, len(vsd_rays_gpu), size=n_views)
        combined = torch.cat([vsd_rays_gpu[k] for k in pose_idxs], dim=0)
        _, _, _, _, _, extras_v = render_rays_3dgrt(
            torch.transpose(combined, 0, 1), model, train=True, frame_id=i)

        rgb_v = extras_v["rgb"].reshape(n_views, R, R, 3).permute(0, 3, 1, 2)
        rgb_v = rgb_v.clamp(0, 1) * 2.0 - 1.0
        depth_cond = torch.stack([vsd_depth_cache[k] for k in pose_idxs], dim=0)

        if use_v7:
            D_metric_v = torch.stack([vsd_metric_cache[k] for k in pose_idxs], dim=0)
            w2c_v = torch.stack([vsd_w2c_cache[k] for k in pose_idxs], dim=0)
            K_v = K_vsd_t.unsqueeze(0).expand(n_views, -1, -1)
            loss_vsd, metrics = vsd_prior.step(rgb_v, depth_cond, D_metric_v, w2c_v, K_v)
        else:
            loss_vsd, metrics = vsd_prior.step(rgb_v, depth_cond)

        loss_smooth = torch.tensor(0.0, device=device)
        if args.vsd_smooth_weight > 0:
            loss_smooth = ((rgb_v[:, :, 1:, :] - rgb_v[:, :, :-1, :]).abs().mean()
                           + (rgb_v[:, :, :, 1:] - rgb_v[:, :, :, :-1]).abs().mean())

        loss = args.vsd_weight * loss_vsd + args.vsd_smooth_weight * loss_smooth

        # LPIPS against the scene's own ring view. Real ground truth, not a
        # diffusion hallucination, so it is the one term that could safely be
        # given geometry access — but geometry is frozen here by design, so it
        # trains colour only.
        loss_lpips = torch.tensor(0.0, device=device)
        need_prev = (i % args.i_weights == 0)
        if vsd_lpips_fn is not None and (
                (i - args.N_iters) % args.vsd_lpips_every_n_steps == 0 or need_prev):
            _, _, _, _, _, extras_r = render_rays_3dgrt(
                torch.transpose(ref_rays, 0, 1), model, train=True, frame_id=i)
            rgb_r = extras_r["rgb"].reshape(1, R, R, 3).permute(0, 3, 1, 2)
            rgb_r = rgb_r.clamp(0, 1) * 2.0 - 1.0
            loss_lpips = vsd_lpips_fn(rgb_r, I_A).mean()
            loss = loss + args.vsd_lpips_weight * loss_lpips
            if need_prev:
                _save_lpips_ref_preview(rgb_r, I_A, img_savedir, i)

        # zero_grad AFTER vsd_prior.step(), matching the chair script's ordering
        # (run_platonerf_3dgrt_vsd.py:1526). step() runs loss_lora.backward()
        # internally on the LoRA particle network; zeroing here clears anything
        # that backward deposited on the model's own parameters before the real
        # backward runs. Zeroing before step() instead would let that stray
        # gradient survive into optimizer.step().
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        # No model.scheduler_step(i): the only scheduler entries are positions
        # (frozen here) and density (type "skip"), and features_albedo has no
        # entry at all — its LR is driven by the decay above.

        with torch.no_grad():
            d = args.vsd_color_ema_decay
            ema_albedo.mul_(d).add_(model.features_albedo.detach(), alpha=1 - d)
            ema_specular.mul_(d).add_(model.features_specular.detach(), alpha=1 - d)

        if i % args.i_print == 0:
            albedo_drift = (model.features_albedo.detach() - albedo_init).abs().mean().item()
            tqdm.write(
                f"[VSD-syn] {i} | loss_vsd={metrics['loss_vsd']:.4f} | "
                  f"loss_lora={metrics.get('loss_lora', 0.0):.4f} | "
                  f"loss_smooth={loss_smooth.item():.4f} | "
                  f"loss_lpips={loss_lpips.item():.4f} | "
                  f"t_mean={metrics.get('t_mean', 0.0):.1f} | t_max_frac={t_max_now:.3f} | "
                f"albedo_drift={albedo_drift:.5f} | "
                f"color_lr={color_groups[0]['lr']:.2e} | "
                f"{(time.time() - t0) / max(i - start, 1):.2f}s/it")

        if i % args.i_weights == 0:
            geom = ((D_metric_v[:1], w2c_v[:1], K_v[:1]) if use_v7 else None)
            _save_vsd_preview(vsd_prior, I_A, rgb_v[:1], depth_cond[:1], img_savedir, i,
                              v7_geom=geom, num_steps=args.vsd_preview_steps)
            save = model.get_model_parameters()
            save["global_step"] = i
            save["ema_albedo"] = ema_albedo
            save["ema_specular"] = ema_specular
            path = os.path.join(savedir, f"{i:06d}.tar")
            torch.save(save, path)
            torch.save(vsd_prior.state_dict(), path.replace(".tar", "_vsd.pt"))
            with open(os.path.join(savedir, "synthetic_run.json"), "w") as fh:
                json.dump({"cameras": os.path.abspath(args.cameras),
                           "ref_image": os.path.abspath(args.vsd_ref_image),
                           "ref_index": args.vsd_ref_index,
                           "ring_target": ring.target.tolist(),
                           "ring_radius": ring.radius,
                           "n_orbit_poses": args.vsd_n_orbit_poses,
                           "last_iter": i}, fh, indent=2)
            print(f"[VSD-syn] saved {path}")

    print(f"[VSD-syn] done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()

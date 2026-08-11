#!/usr/bin/env python
"""
src/run_photometric_synthetic.py — supervise texture with a PHOTOMETRIC loss
against FIXED diffusion outputs, instead of with score distillation.

THE QUESTION THIS ANSWERS
-------------------------
The VSD result conflates two things: the quality of what the prior can draw,
and the quality of the GRADIENT ESTIMATOR used to distil it. Score distillation
is a high-variance estimator by construction — that is why this pipeline needs
an EMA, an LR decay, a timestep anneal and multi-view batching just to be
readable.

So: generate the prior's actual images once, freeze them as pseudo ground
truth, and fit the Gaussians to them with a plain reconstruction loss. Then

  * clean, sharp result  -> the prior is fine and VSD's estimator is the cost.
  * blurry / washed out  -> the targets disagree with each other across views,
                            and the optimiser is averaging them. No distillation
                            scheme would have rescued that, and the fix is
                            multi-view consistency in the prior.
  * shimmering per view  -> the geometry cannot represent a consistent surface,
                            which is the needle/mush hypothesis rather than
                            anything about the loss.

Those three outcomes are distinguishable in a few thousand iterations, because
fitting fixed images is an ordinary well-posed regression — no annealing, no
particle network, no variance to average down.

TARGETS ARE GENERATED IN JOINT BATCHES, ON PURPOSE
--------------------------------------------------
V7's triplane pools its batch into ONE 3D volume (this is why the handoff
insists CFG must run sequentially rather than as a doubled batch — a 2K batch
silently pools the conditional and unconditional halves). Used deliberately,
that same pooling is a consistency mechanism: poses generated together share a
3D representation. So targets are produced in batches of --gen_batch rather
than one at a time, and --gen_batch is worth treating as an experimental knob,
not a memory setting.

Targets are written to disk as a contact sheet before any training happens.
LOOK AT IT FIRST. If the targets disagree with each other, the fit's job is
impossible and the run tells you nothing you could not have seen in that sheet.

Geometry is frozen exactly as in run_vsd_synthetic.py, so this differs from
that script in the loss and nothing else.
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

import imageio.v2 as imageio  # noqa: E402
from threedgrut.model.model import MixtureOfGaussians          # noqa: E402
from run_platonerf_3dgrt_vsd import create_3dgrt_conf, render_rays_3dgrt  # noqa: E402
from utils.synthetic_scene import (                            # noqa: E402
    load_scene_cameras, scale_intrinsics, rays_from_w2c, fit_ring,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--expname", required=True)
    p.add_argument("--basedir", default="./logs/")
    p.add_argument("--cameras", required=True)
    p.add_argument("--ft_path", default=None)
    p.add_argument("--N_iters", type=int, default=35000)
    p.add_argument("--iters", type=int, required=True,
                   help="photometric iterations to run, relative to the checkpoint")

    p.add_argument("--vsd_checkpoint", type=str, required=True)
    p.add_argument("--vsd_controlnet_root", type=str, default=None)
    p.add_argument("--vsd_arch", type=str, default="v7", choices=["v3", "v7"])
    p.add_argument("--vsd_model_overrides", type=str, default="")
    p.add_argument("--vsd_ref_image", type=str, required=True)
    p.add_argument("--vsd_ref_index", type=int, default=0)
    p.add_argument("--vsd_guidance_scale", type=float, default=3.0)
    p.add_argument("--vsd_render_res", type=int, default=512)

    # --- target generation ---
    p.add_argument("--n_target_poses", type=int, default=24,
                   help="poses to generate pseudo-GT for. 24 matches the scene's own "
                        "ring; more poses cover the orbit better but give the "
                        "triplane less overlap per batch to work with.")
    p.add_argument("--gen_batch", type=int, default=8,
                   help="poses generated jointly. V7's triplane pools its batch into "
                        "one 3D volume, so this directly controls how much cross-view "
                        "consistency the targets get. Experimental knob, not a memory "
                        "setting.")
    p.add_argument("--gen_steps", type=int, default=30,
                   help="denoise steps per target (more than the 20 used for training "
                        "previews: these are fitted against, not glanced at)")
    p.add_argument("--gen_seed", type=int, default=0)
    p.add_argument("--targets_npz", default=None,
                   help="reuse targets from a previous run instead of regenerating")
    p.add_argument("--gen_only", action="store_true",
                   help="generate targets and the contact sheet, then stop")

    # --- loss ---
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_lpips", type=float, default=1.0,
                   help="LPIPS on its own is known to chase high-frequency structure "
                        "and leave colour drifting, so it is paired with L1 rather "
                        "than used alone. Both terms are printed each --i_print so "
                        "their relative magnitude is visible rather than assumed — "
                        "this project has been burned by a loss term 1000x out of "
                        "scale with what it was added to.")
    p.add_argument("--batch_size", type=int, default=2,
                   help="target views per step. LPIPS at 512x512 is the memory cost.")
    p.add_argument("--lr_scale", type=float, default=1.0,
                   help="multiplies features_albedo's config LR. Fitting fixed images "
                        "is far better conditioned than distillation, so a larger LR "
                        "than VSD's is usually appropriate.")
    p.add_argument("--train_specular", type=int, default=0,
                   help="0 keeps features_specular frozen, matching the VSD runs. "
                        "Turning it on lets view-dependent SH absorb disagreement "
                        "BETWEEN targets, which would hide the very inconsistency "
                        "this experiment is trying to measure.")
    p.add_argument("--color_ema_decay", type=float, default=0.999)

    p.add_argument("--i_weights", type=int, required=True)
    p.add_argument("--i_print", type=int, default=25)
    p.add_argument("--render_chunk", type=int, default=65536)
    p.add_argument("--seed", type=int, default=0)
    return p


def to_uint8(img_m11):
    """[-1,1] CHW tensor -> HWC uint8."""
    x = img_m11.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1) * 127.5).permute(1, 2, 0).numpy()
    return x.astype(np.uint8)


def main():
    args = config_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    savedir = os.path.join(args.basedir, args.expname)
    img_savedir = os.path.join(savedir, "progress")
    os.makedirs(img_savedir, exist_ok=True)

    cams = load_scene_cameras(args.cameras)
    ring = fit_ring(cams)
    R = args.vsd_render_res
    K_vsd = scale_intrinsics(cams.K, cams.width, cams.height, R).astype(np.float32)
    K_vsd_t = torch.from_numpy(K_vsd).to(device)

    # ---------------------------------------------------------------- model --
    conf = create_3dgrt_conf(args)
    if args.ft_path:
        ckpt_path = args.ft_path
    else:
        cands = sorted(f for f in os.listdir(savedir)
                       if f.endswith(".tar") and not f.endswith("_strategy.tar"))
        if not cands:
            sys.exit(f"no .tar checkpoint in {savedir}")
        ckpt_path = os.path.join(savedir, cands[-1])
    print(f"[photo] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    scene_extent = float(ckpt.get("scene_extent") or ring.radius * 1.1)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
    model.init_from_checkpoint(ckpt, setup_optimizer=True)
    model.build_acc(rebuild=True)
    start = int(ckpt.get("global_step", args.N_iters))
    print(f"[photo] {model.num_gaussians:,} Gaussians, global_step={start}")

    for p_ in (model.positions, model.density, model.rotation, model.scale):
        p_.requires_grad_(False)
    model.features_specular.requires_grad_(bool(args.train_specular))
    model.features_albedo.requires_grad_(True)
    print(f"[photo] geometry frozen; training features_albedo"
          f"{' + features_specular' if args.train_specular else ''}")

    # ------------------------------------------------------------ pose bank --
    orbit = ring.orbit(args.n_target_poses)
    rays_gpu = [torch.from_numpy(rays_from_w2c(w, K_vsd, R)).to(device) for w in orbit]
    w2c_gpu = [torch.from_numpy(w.astype(np.float32)).to(device) for w in orbit]

    def render_rays_for(idx, train):
        return render_rays_3dgrt(torch.transpose(rays_gpu[idx], 0, 1), model,
                                 train=train, frame_id=idx)

    # ---------------------------------------------------------- the targets --
    tgt_path = args.targets_npz or os.path.join(savedir, "photo_targets.npz")
    if args.targets_npz and os.path.isfile(args.targets_npz):
        d = np.load(args.targets_npz)
        targets = torch.from_numpy(d["targets"]).to(device)   # [M,3,R,R] in [-1,1]
        print(f"[photo] reusing {targets.shape[0]} targets from {args.targets_npz}")
    else:
        from vsd_prior import (DiffusionPrior, load_reference_image,  # noqa: F401
                               make_depth_cond)
        use_v7 = (args.vsd_arch == "v7")
        if use_v7:
            from vsd_prior_v7 import V7DiffusionPrior

        prior_kwargs = dict(
            checkpoint_path=args.vsd_checkpoint, device=device, variant="sds",
            guidance_scale=args.vsd_guidance_scale, lora_rank=8, lora_lr=1e-4,
            t_min_frac=0.02, t_max_frac=0.98, grad_weight="snr",
        )
        if args.vsd_controlnet_root:
            prior_kwargs["controlnet_repo_root"] = args.vsd_controlnet_root
        if use_v7:
            prior_kwargs["model_overrides"] = (
                args.vsd_model_overrides.split() if args.vsd_model_overrides else None)
            prior = V7DiffusionPrior(**prior_kwargs)
        else:
            prior = DiffusionPrior(**prior_kwargs)
        # variant="sds" so no LoRA particle network is built: nothing here is
        # distilled, the prior is only ever asked to draw.

        I_A = load_reference_image(args.vsd_ref_image, device)
        prior.set_reference_image(I_A)

        from src.models.pose_convert import ray_distance_to_z_depth as _ray_dist_to_z
        ref_w2c = cams.w2c[args.vsd_ref_index].astype(np.float64)
        ref_rays = torch.from_numpy(rays_from_w2c(ref_w2c, K_vsd, R)).to(device)
        if use_v7:
            with torch.no_grad():
                _, _, _, depth_r, _, _ = render_rays_3dgrt(
                    torch.transpose(ref_rays, 0, 1), model, train=False, frame_id=0)
            D_ref = _ray_dist_to_z(depth_r.reshape(R, R).detach().cpu().numpy(), K_vsd)
            prior.set_reference_geometry(D_ref, ref_w2c.astype(np.float32), K_vsd)

        print(f"[photo] rendering depth conditioning for {len(rays_gpu)} poses ...")
        depth_cond, metric = [], []
        with torch.no_grad():
            for idx in range(len(rays_gpu)):
                _, _, acc_c, depth_c, _, _ = render_rays_for(idx, False)
                dh = depth_c.reshape(R, R)
                depth_cond.append(make_depth_cond(dh, acc_c.reshape(R, R)).squeeze(0))
                if use_v7:
                    metric.append(torch.from_numpy(
                        _ray_dist_to_z(dh.detach().cpu().numpy(), K_vsd)).to(device))

        print(f"[photo] generating {len(rays_gpu)} targets in batches of "
              f"{args.gen_batch} ({args.gen_steps} steps each) ...")
        torch.manual_seed(args.gen_seed)
        outs = []
        t_gen = time.time()
        for s in range(0, len(rays_gpu), args.gen_batch):
            sl = list(range(s, min(s + args.gen_batch, len(rays_gpu))))
            dc = torch.stack([depth_cond[j] for j in sl], dim=0)
            if use_v7:
                Dm = torch.stack([metric[j] for j in sl], dim=0)
                w2c = torch.stack([w2c_gpu[j] for j in sl], dim=0)
                Kb = K_vsd_t.unsqueeze(0).expand(len(sl), -1, -1)
                img = prior.preview(dc, Dm, w2c, Kb, num_steps=args.gen_steps)
            else:
                img = prior.preview(dc, num_steps=args.gen_steps)
            outs.append(img.detach().float().cpu())
            print(f"  targets {sl[0]}-{sl[-1]} done "
                  f"({time.time() - t_gen:.0f}s elapsed)")
        targets = torch.cat(outs, dim=0).to(device)
        np.savez_compressed(tgt_path, targets=targets.cpu().numpy(),
                            orbit=orbit.astype(np.float32))
        print(f"[photo] wrote {tgt_path}")
        del prior
        torch.cuda.empty_cache()

    # Contact sheet. LOOK AT THIS BEFORE READING ANY RESULT: if the targets
    # disagree with each other, the fit is being asked to satisfy contradictory
    # images and its blurriness says nothing about the optimiser.
    M = targets.shape[0]
    cols = min(8, M)
    rows = int(np.ceil(M / cols))
    sheet = np.ones((rows * R, cols * R, 3), dtype=np.uint8) * 255
    for j in range(M):
        r, c = divmod(j, cols)
        sheet[r * R:(r + 1) * R, c * R:(c + 1) * R] = to_uint8(targets[j])
    sheet_path = os.path.join(savedir, "photo_targets_sheet.png")
    imageio.imwrite(sheet_path, sheet)
    print(f"[photo] wrote {sheet_path}  ({M} targets, {rows}x{cols})")
    if args.gen_only:
        return

    # ------------------------------------------------------------------ loss --
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    for q in lpips_fn.parameters():
        q.requires_grad_(False)

    for pg in model.optimizer.param_groups:
        if pg["name"] in ("features_albedo", "features_specular"):
            pg["lr"] = conf.optimizer.params[pg["name"]].lr * args.lr_scale
    print(f"[photo] albedo LR = {conf.optimizer.params['features_albedo'].lr * args.lr_scale:.2e} "
          f"({args.lr_scale}x config)")

    ema_albedo = model.features_albedo.detach().clone()
    ema_specular = model.features_specular.detach().clone()

    end = start + args.iters
    print(f"[photo] training {start + 1} -> {end}")
    t0 = time.time()
    for i in trange(start + 1, end + 1):
        idxs = np.random.randint(0, M, size=args.batch_size)
        rgbs = []
        for j in idxs:
            _, _, _, _, _, extras = render_rays_for(int(j), True)
            rgbs.append(extras["rgb"].reshape(R, R, 3).permute(2, 0, 1))
        rgb = torch.stack(rgbs, dim=0).clamp(0, 1) * 2.0 - 1.0     # [B,3,R,R] in [-1,1]
        tgt = targets[idxs]

        loss_l1 = F.l1_loss(rgb, tgt)
        loss_lp = lpips_fn(rgb, tgt).mean()
        loss = args.w_l1 * loss_l1 + args.w_lpips * loss_lp

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        with torch.no_grad():
            d = args.color_ema_decay
            ema_albedo.mul_(d).add_(model.features_albedo.detach(), alpha=1 - d)
            ema_specular.mul_(d).add_(model.features_specular.detach(), alpha=1 - d)

        if i % args.i_print == 0:
            # Both terms printed with their WEIGHTED contributions, so a term
            # that is contributing nothing is visible immediately rather than
            # after three runs.
            tqdm.write(
                f"[photo] {i} | l1={loss_l1.item():.5f} (x{args.w_l1} = "
                f"{args.w_l1 * loss_l1.item():.5f}) | "
                f"lpips={loss_lp.item():.5f} (x{args.w_lpips} = "
                f"{args.w_lpips * loss_lp.item():.5f}) | "
                f"total={loss.item():.5f} | "
                f"{(time.time() - t0) / max(i - start, 1):.2f}s/it")

        if i % args.i_weights == 0:
            with torch.no_grad():
                j = int(idxs[0])
                pair = np.concatenate([to_uint8(rgb[0]), to_uint8(tgt[0])], axis=1)
            imageio.imwrite(os.path.join(img_savedir, f"photo_{i:06d}.png"), pair)
            save = model.get_model_parameters()
            save["global_step"] = i
            save["ema_albedo"] = ema_albedo
            save["ema_specular"] = ema_specular
            torch.save(save, os.path.join(savedir, f"{i:06d}.tar"))
            with open(os.path.join(savedir, "synthetic_run.json"), "w") as fh:
                json.dump({"cameras": os.path.abspath(args.cameras),
                           "ref_image": os.path.abspath(args.vsd_ref_image),
                           "ref_index": args.vsd_ref_index,
                           "mode": "photometric",
                           "n_target_poses": args.n_target_poses,
                           "gen_batch": args.gen_batch,
                           "ring_target": ring.target.tolist(),
                           "ring_radius": ring.radius,
                           "last_iter": i}, fh, indent=2)
            print(f"[photo] saved {i:06d}.tar  (render | target -> "
                  f"progress/photo_{i:06d}.png, pose {j})")

    print(f"[photo] done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

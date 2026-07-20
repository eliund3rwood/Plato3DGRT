#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# "Topping off" pass: direct multi-pose photometric reconstruction against a
# small set of PRE-SELECTED, hand-vetted diffusion-model output images
# (e.g. from a full pose-coverage sweep -- see diagnose_pose_coverage.py --
# where a human confirmed these specific renders look clean/consistent),
# rather than another round of live score distillation (VSD/SDS).
#
# Why this is a different, cheaper tool than run_platonerf_3dgrt_vsd.py's
# Phase 3: VSD re-queries the frozen diffusion prior (ControlNet + UNet +
# CFG, plus a LoRA "particle" network) fresh every step from randomly
# sampled poses, chasing a moving, stochastic score-distillation target.
# That's expensive and (per diagnose_pose_coverage.py's findings) the
# pretrained model gives genuinely bad, non-noise-averaging-away predictions
# at some poses -- no amount of extra VSD iterations fixes that, since it's
# a property of the frozen weights, not the optimization. Here, a human has
# already looked at a full sweep of the model's own output and hand-picked
# specific (pose, image) pairs known to be clean -- so instead of live
# distillation, this just directly minimizes L1 + LPIPS between the current
# render and those FIXED target images. No diffusion model forward pass
# happens during this training loop at all -- much cheaper per iteration,
# and immune to VSD's per-step noise since targets never change.
#
# Trade-off: each target image is an independent stochastic diffusion
# sample, not guaranteed to agree with its neighbors in regions where
# Gaussians are visible from multiple of the selected poses -- a smaller
# version of the same view-inconsistency risk VSD's batching/EMA/specular-
# freeze fixes addressed. Should be much milder here since every target was
# individually vetted rather than a raw per-step noisy sample, but worth
# inspecting the result for seams/blending artifacts between adjacent poses.
#
# Only features_albedo is trained by default (diffuse color) -- same
# reasoning as --vsd_freeze_specular in the VSD script: features_specular
# has never received real supervision, and letting per-pose-inconsistent
# targets train view-dependent SH risks re-introducing exactly the kind of
# artifact this is meant to avoid. Position/rotation/scale/density are
# always frozen -- this tool only ever touches color.
#
# Usage:
#   python src/refine_from_target_images.py --config configs/chair_vsd.txt \
#     --expname chair_vsd_run5 \
#     --target_dir /path/to/chair_v4_notex \
#     --pose_idxs "1,6,7,8,9,12,13,24,30,36,41,48,86,88,90,91,92,93,95" \
#     --n_iters 3000 --output_expname chair_vsd_run5_topoff

import os
import sys

import cv2
import lpips as lpips_lib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch

from utils.load_tof import load_tof_data
from utils.novel_views import orbit_poses, rays_at_resolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 3DGRT config -- same shape as render_test_depth_3dgrt.py / diagnose_pose_
# coverage.py's loaders. setup_optimizer=True this time since we DO train
# (features_albedo only; everything else gets requires_grad_(False) below,
# regardless of what this config's optimize_* flags say).
# ---------------------------------------------------------------------------

def create_3dgrt_conf(n_iters=200000):
    return OmegaConf.create({
        "render": {
            "method": "3dgrt", "pipeline_type": "reference",
            "backward_pipeline_type": "referenceBwd", "particle_kernel_degree": 4,
            "particle_kernel_density_clamping": True, "particle_kernel_min_response": 0.0113,
            "particle_kernel_min_alpha": float(1.0 / 255.0), "particle_kernel_max_alpha": 0.99,
            "particle_radiance_sph_degree": 3, "primitive_type": "instances",
            "min_transmittance": 0.001, "max_consecutive_bvh_update": 15,
            "enable_normals": False, "enable_hitcounts": False, "enable_kernel_timings": False,
        },
        "model": {
            "density_activation": "sigmoid", "scale_activation": "exp",
            "default_density": 0.1, "default_scale_factor": 1.0,
            "optimize_density": True, "optimize_features_albedo": True,
            "optimize_features_specular": True, "optimize_position": True,
            "optimize_rotation": True, "optimize_scale": True,
            "bvh_update_frequency": 1,
            "progressive_training": {
                "feature_type": "sh", "init_n_features": 0, "max_n_features": 3,
                "increase_frequency": 1000, "increase_step": 1,
            },
            "background": {"name": "background-color", "color": "black"},
            "print_stats": False,
        },
        "optimizer": {
            "type": "adam", "lr": 0.0, "eps": 1e-15,
            "params": {
                "positions": {"lr": 0.00016}, "density": {"lr": 0.05},
                "features_albedo": {"lr": 0.0025}, "features_specular": {"lr": 0.000125},
                "rotation": {"lr": 0.001}, "scale": {"lr": 0.002},
            },
        },
        "scheduler": {
            "positions": {"type": "exp", "lr_init": 0.00016, "lr_final": 0.0000016, "max_steps": n_iters},
            "density": {"type": "skip"},
        },
        "strategy": {
            "method": "GSStrategy", "print_stats": False,
            "densify": {"params": "positions", "frequency": 300, "start_iteration": 500,
                        "end_iteration": 0, "clone_grad_threshold": 0.0008,
                        "split_grad_threshold": 0.0008, "relative_size_threshold": 0.01,
                        "split": {"n_gaussians": 2}},
            "prune": {"frequency": 100, "start_iteration": 500, "end_iteration": 0,
                      "density_threshold": 0.05},
            "reset_density": {"frequency": 3000, "start_iteration": 0, "end_iteration": 0,
                               "new_max_density": 0.01},
            "density_decay": {"gamma": 0.99, "start_iteration": -1, "end_iteration": -1, "frequency": 50},
            "prune_weight": {"frequency": 100, "start_iteration": -1, "end_iteration": -1, "weight_threshold": 0.5},
            "prune_scale": {"frequency": 100, "start_iteration": -1, "end_iteration": -1, "threshold": 1.0},
        },
        "checkpoint": {"iterations": [n_iters]},
    })


def render_rays_3dgrt(batch_rays, model, train=True):
    ray_o, ray_d = batch_rays[0], batch_rays[1]
    rays_ori = ray_o.unsqueeze(0).unsqueeze(2)
    rays_dir = ray_d.unsqueeze(0).unsqueeze(2)
    T_to_world = torch.eye(4, device=ray_o.device, dtype=ray_o.dtype).unsqueeze(0)
    gpu_batch = Batch(rays_ori=rays_ori, rays_dir=rays_dir, T_to_world=T_to_world,
                       rays_in_world_space=True)
    out = model(gpu_batch, train=train, frame_id=0)
    rgb = out["pred_rgb"].squeeze(0).squeeze(1)  # [N, 3]
    return rgb


def load_target_panel(path: str, panel: str, res: int, device) -> torch.Tensor:
    """Load a target RGB image as a 1x3xRxR float tensor in [-1, 1].
    panel: 'left' | 'mid' | 'right' crops one third of a 3-panel
    (I_A | D_B | Î_B) composite PNG (e.g. chair_v4_notex); 'full' uses the
    whole image directly (e.g. single-panel DifFix3D-style output)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if panel == "full":
        crop = img
    else:
        h, w = img.shape[:2]
        third = w // 3
        idx = {"left": 0, "mid": 1, "right": 2}[panel]
        crop = img[:, idx * third: (idx + 1) * third if idx < 2 else w]
    if crop.shape[:2] != (res, res):
        crop = cv2.resize(crop, (res, res), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument("--expname", type=str, required=True,
                        help='experiment name to load the starting checkpoint from')
    parser.add_argument("--basedir", type=str, default='./logs/')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific starting checkpoint .tar; defaults to latest in basedir/expname')
    parser.add_argument("--dataset_type", type=str, default='dtof')
    parser.add_argument("--ignore", type=int, action='append', required=False, default=[])
    parser.add_argument("--per_image_thresh", type=float, action='append', required=False)
    parser.add_argument("--use_raw_weights", action='store_true',
                        help='start from raw (non-EMA) trained color weights instead of EMA')
    parser.add_argument("--vsd_checkpoint", type=str, default=None, help='unused, accepted for config compat')
    parser.add_argument("--vsd_n_orbit_poses", type=int, default=100, help='unused, accepted for config compat')

    parser.add_argument("--output_expname", type=str, required=True,
                        help='new experiment name to write topped-off checkpoints under')
    parser.add_argument("--target_dir", type=str, required=True,
                        help='directory of 3-panel (I_A|D_B|target) composite PNGs named '
                             'like depth_map_XXX_out.png / pose_XXX.png (must contain %%03d '
                             'somewhere matching --pose_idxs)')
    parser.add_argument("--target_glob", type=str, default="depth_map_{:03d}_out.png",
                        help='filename pattern (python .format with the pose index, after adding '
                             '--target_idx_offset) inside --target_dir')
    parser.add_argument("--target_idx_offset", type=int, default=0,
                        help='added to the orbit-pose index before formatting --target_glob -- e.g. '
                             '1 for a 1-indexed filename set (frame_0001.png == pose index 0)')
    parser.add_argument("--target_panel", type=str, default="right", choices=["left", "mid", "right", "full"],
                        help='which third of the composite is the actual RGB target')
    parser.add_argument("--pose_idxs", type=str, required=True,
                        help='comma-separated orbit-pose indices (0..vsd_n_orbit_poses-1) with '
                             'hand-vetted clean target images to fine-tune against')
    parser.add_argument("--render_res", type=int, default=512)
    parser.add_argument("--n_iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.0025,
                         help='features_albedo learning rate (matches the VSD script base LR)')
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_pix", type=float, default=1.0)
    parser.add_argument("--freeze_specular", type=int, default=1,
                        help='1 = only features_albedo trains (recommended -- see module docstring); '
                             '0 = also train features_specular')
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=500)
    return parser


def main():
    parser = config_parser()
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"[INFO] Ignoring unrecognised config keys: {_unknown}")

    pose_idxs = [int(x) for x in args.pose_idxs.split(",")]
    print(f"[topoff] {len(pose_idxs)} target poses: {pose_idxs}")

    tof, poses, light_o, light_d, hwf, walls_cam, walls_light = load_tof_data(
        args.datadir, args.ignore
    )
    H, W, focal = hwf
    H, W = int(H), int(W)

    conf = create_3dgrt_conf()
    model = MixtureOfGaussians(conf, scene_extent=6.0).to(device)

    ckpt_dir = os.path.join(args.basedir, args.expname)
    if args.ft_path is not None and args.ft_path != 'None':
        ckpt_path = args.ft_path
    else:
        ckpts = sorted([
            os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
            if f.endswith('.tar') and not f.endswith('_strategy.tar')
        ])
        ckpt_path = ckpts[-1]
    print("[topoff] Starting from checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    model.build_acc()

    if not args.use_raw_weights and 'ema_albedo' in ckpt:
        with torch.no_grad():
            model.features_albedo.copy_(ckpt['ema_albedo'].to(device))
            model.features_specular.copy_(ckpt['ema_specular'].to(device))
        print("[topoff] Started from EMA color weights (pass --use_raw_weights to start from raw instead)")

    # Only color trains; geometry and (by default) specular are frozen --
    # this tool only ever touches diffuse albedo, same reasoning as
    # --vsd_freeze_specular in the VSD script (see module docstring).
    for p in (model.positions, model.rotation, model.scale, model.density):
        p.requires_grad_(False)
    model.features_albedo.requires_grad_(True)
    model.features_specular.requires_grad_(not args.freeze_specular)

    # Use the model's own optimizer slot (not a bare torch.optim.Adam) --
    # setup_optimizer() only includes params still marked requires_grad=True
    # in its param groups, so this naturally ends up training just
    # features_albedo (+ features_specular if not frozen). Also required for
    # get_model_parameters() to save a checkpoint at all (it asserts
    # self.optimizer is not None, since it saves optimizer state too).
    conf.optimizer.params.features_albedo.lr = args.lr
    model.setup_optimizer()
    optimizer = model.optimizer

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Pre-render rays + load fixed target images for every whitelisted pose
    # ------------------------------------------------------------------
    R = args.render_res
    all_poses = orbit_poses(n=args.vsd_n_orbit_poses)
    rays_by_pose, target_by_pose = [], []
    for idx in pose_idxs:
        rays_np = rays_at_resolution(all_poses[idx], H, W, focal, R, R)
        rays_by_pose.append(torch.from_numpy(rays_np).to(device))
        target_path = os.path.join(args.target_dir, args.target_glob.format(idx + args.target_idx_offset))
        target_by_pose.append(load_target_panel(target_path, args.target_panel, R, device))
        print(f"[topoff]   pose {idx:03d} <- {target_path}")

    out_dir = os.path.join(args.basedir, args.output_expname)
    img_savedir = os.path.join(out_dir, "progress")
    os.makedirs(img_savedir, exist_ok=True)

    trained_params = "features_albedo" if args.freeze_specular else "features_albedo + features_specular"
    print(f"[topoff] Training {trained_params} for {args.n_iters} iterations "
          f"against {len(pose_idxs)} fixed targets ...")
    for i in trange(1, args.n_iters + 1):
        j = np.random.randint(0, len(pose_idxs))
        rays_t = torch.transpose(rays_by_pose[j], 0, 1)
        rgb = render_rays_3dgrt(rays_t, model, train=True)
        rgb = rgb.reshape(R, R, 3).permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0

        target = target_by_pose[j]
        loss_pix = F.l1_loss(rgb, target)
        loss_lpips = lpips_fn(rgb, target).mean()
        loss = args.lambda_pix * loss_pix + args.lambda_lpips * loss_lpips

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % args.i_print == 0:
            print(f"[topoff] iter {i} | pose {pose_idxs[j]:03d} | "
                  f"loss={loss.item():.4f} (pix={loss_pix.item():.4f}, lpips={loss_lpips.item():.4f})")

        if i % args.i_weights == 0 or i == args.n_iters:
            path = os.path.join(out_dir, f"{i:06d}.tar")
            save_dict = model.get_model_parameters()
            save_dict['global_step'] = ckpt.get('global_step', 0)
            torch.save(save_dict, path)
            print(f"[topoff] Saved checkpoint -> {path}")

            with torch.no_grad():
                panel = torch.cat([target, rgb], dim=-1)
                x = panel.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
                x = (np.clip(x, -1, 1) + 1) / 2
                x = (x * 255).astype(np.uint8)
                x_bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                prev_path = os.path.join(img_savedir, f"topoff_{i:06d}_pose{pose_idxs[j]:03d}.png")
                cv2.imwrite(prev_path, x_bgr)

    print("[topoff] Done.")


if __name__ == '__main__':
    main()

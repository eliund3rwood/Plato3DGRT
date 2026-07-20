#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Diagnostic: does the FROZEN, PRETRAINED custom_controlnet diffusion prior
# itself give meaningfully different-quality predictions depending on the
# novel-view pose (and D_B depth conditioning it implies), independent of
# anything the VSD training loop does?
#
# Motivation: run5's final RGB eval showed a consistent-across-angles
# iridescent/rainbow artifact (unlike run1-run4's per-view-inconsistent
# "shattered glass" symptom, which today's fixes appear to have resolved --
# the render now looks the SAME, if wrong, from every angle). Comparing two
# incidental preview panels from run5 (pose 76 at the smoke-test checkpoint:
# clean/photorealistic; pose 27 at the final checkpoint: strongly
# iridescent) suggested the pretrained model's own prediction quality might
# vary a lot by pose -- but that's only n=2, both incidental. This script
# settles it directly: run the frozen prior (LoRA disabled, nothing
# trainable involved -- variant="sds" skips the LoRA/peft machinery
# entirely) across many evenly-spaced orbit poses, 2 different noise seeds
# each, using the FINAL trained geometry/checkpoint for depth conditioning.
# No Gaussian training happens here at all -- this isolates "is the
# diffusion prior itself pose-dependent" from "is the VSD training loop
# doing something wrong."
#
# If a pose is bad across BOTH seeds -> pose/conditioning-driven (a
# generalization gap in the pretrained model, e.g. from uneven training-data
# pose coverage -- no VSD hyperparameter fixes this, would need restricting
# the sampling pose bank or expanding the diffusion model's own training
# data). If a pose is only sometimes bad -> more of a generic noisy-sample
# issue, closer to what batching/EMA already target.
#
# Usage (interactive GPU node, no --vsd_iters/training involved):
#   python src/diagnose_pose_coverage.py --config configs/chair_vsd.txt \
#     --expname chair_vsd_run5 \
#     --vsd_checkpoint /path/to/final.pt \
#     --n_poses 10 --n_seeds 2 --output_dir logs/chair_vsd_run5/pose_diag

import os
import sys

import cv2
import numpy as np
import torch

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
# 3DGRT config -- identical shape to render_test_depth_3dgrt.py's loader
# (setup_optimizer=False below, so optimizer/scheduler/strategy values here
# are irrelevant placeholders; only used to reconstruct the model's tensor
# shapes for init_from_checkpoint).
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
                        "end_iteration": n_iters, "clone_grad_threshold": 0.0008,
                        "split_grad_threshold": 0.0008, "relative_size_threshold": 0.01,
                        "split": {"n_gaussians": 2}},
            "prune": {"frequency": 100, "start_iteration": 500, "end_iteration": n_iters,
                      "density_threshold": 0.05},
            "reset_density": {"frequency": 3000, "start_iteration": 0, "end_iteration": n_iters,
                               "new_max_density": 0.01},
            "density_decay": {"gamma": 0.99, "start_iteration": -1, "end_iteration": -1, "frequency": 50},
            "prune_weight": {"frequency": 100, "start_iteration": -1, "end_iteration": -1, "weight_threshold": 0.5},
            "prune_scale": {"frequency": 100, "start_iteration": -1, "end_iteration": -1, "threshold": 1.0},
        },
        "checkpoint": {"iterations": [n_iters]},
    })


@torch.no_grad()
def render_rays_3dgrt(batch_rays, model):
    ray_o, ray_d = batch_rays[0], batch_rays[1]
    rays_ori = ray_o.unsqueeze(0).unsqueeze(2)
    rays_dir = ray_d.unsqueeze(0).unsqueeze(2)
    T_to_world = torch.eye(4, device=ray_o.device, dtype=ray_o.dtype).unsqueeze(0)
    gpu_batch = Batch(rays_ori=rays_ori, rays_dir=rays_dir, T_to_world=T_to_world,
                       rays_in_world_space=True)
    out = model(gpu_batch, train=False, frame_id=0)
    depth = out["pred_dist"].squeeze(0).squeeze(1).squeeze(-1)
    acc = out["pred_opacity"].squeeze(0).squeeze(1).squeeze(-1)
    return depth, acc


def to_uint8_rgb(x, signed=True):
    x = x.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    if signed:
        x = (np.clip(x, -1, 1) + 1) / 2
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument("--expname", type=str, required=True)
    parser.add_argument("--basedir", type=str, default='./logs/')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific checkpoint .tar; defaults to latest in basedir/expname')
    parser.add_argument("--dataset_type", type=str, default='dtof')
    parser.add_argument("--ignore", type=int, action='append', required=False, default=[])
    parser.add_argument("--per_image_thresh", type=float, action='append', required=False)
    parser.add_argument("--use_raw_weights", action='store_true',
                        help='use raw (non-EMA) trained color weights instead of EMA')
    parser.add_argument("--vsd_checkpoint", type=str, required=True,
                        help='path to custom_controlnet final.pt')
    parser.add_argument("--vsd_controlnet_root", type=str, default=None)
    parser.add_argument("--vsd_ref_image", type=str, default=None)
    parser.add_argument("--vsd_guidance_scale", type=float, default=3.0)
    parser.add_argument("--vsd_render_res", type=int, default=512)
    parser.add_argument("--n_poses", type=int, default=10,
                        help='number of evenly-spaced orbit poses to probe (out of the full 100-pose bank); '
                             'ignored if --pose_idxs is given')
    parser.add_argument("--pose_idxs", type=str, default=None,
                        help='comma-separated exact pose indices (0-99) to probe instead of an '
                             'evenly-spaced sweep -- for zooming in on a good/bad coverage boundary '
                             'found by a first coarse --n_poses pass, e.g. "20,23,26,29,32,35,38"')
    parser.add_argument("--n_seeds", type=int, default=2,
                        help='independent noise seeds per pose -- a pose that looks bad under '
                             'every seed indicates a pose/conditioning problem in the pretrained '
                             'model; a pose that is only sometimes bad indicates generic sample '
                             'noise instead')
    parser.add_argument("--preview_steps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def main():
    parser = config_parser()
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"[INFO] Ignoring unrecognised config keys: {_unknown}")

    os.makedirs(args.output_dir, exist_ok=True)

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
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    model.build_acc()
    model.eval()

    if not args.use_raw_weights and 'ema_albedo' in ckpt:
        with torch.no_grad():
            model.features_albedo.copy_(ckpt['ema_albedo'].to(device))
            model.features_specular.copy_(ckpt['ema_specular'].to(device))
        print("[diag] Using EMA color weights (the trained result being diagnosed)")

    # ------------------------------------------------------------------
    # Build the frozen diffusion prior. variant="sds" skips the LoRA/peft
    # particle-network machinery entirely -- preview() never uses it
    # anyway (always runs with adapters disabled), so this keeps the
    # diagnostic strictly about the pretrained UNet+ControlNet.
    # ------------------------------------------------------------------
    torch.set_default_tensor_type(torch.FloatTensor)  # see run_platonerf_3dgrt_vsd.py's note on why
    try:
        from vsd_prior import DiffusionPrior, load_reference_image, make_depth_cond
        vsd_ref_path = args.vsd_ref_image or os.path.join(args.datadir, "..", "chair_smooth_walls.png")
        prior_kwargs = dict(
            checkpoint_path=args.vsd_checkpoint, device=device, variant="sds",
            guidance_scale=args.vsd_guidance_scale,
        )
        if args.vsd_controlnet_root:
            prior_kwargs["controlnet_repo_root"] = args.vsd_controlnet_root
        prior = DiffusionPrior(**prior_kwargs)
        I_A = load_reference_image(vsd_ref_path, device)
        prior.set_reference_image(I_A)
    finally:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # ------------------------------------------------------------------
    # Evenly-spaced pose sweep across the SAME 100-pose orbit bank VSD
    # training samples from, so results map directly onto "which angular
    # region of the orbit does the pretrained model handle well."
    # ------------------------------------------------------------------
    R = args.vsd_render_res
    all_poses = orbit_poses(n=100)
    if args.pose_idxs:
        pose_idxs = np.array([int(x) for x in args.pose_idxs.split(",")], dtype=int)
    else:
        pose_idxs = np.linspace(0, 99, args.n_poses, dtype=int)

    rows = []
    print(f"[diag] Probing {len(pose_idxs)} poses x {args.n_seeds} seeds "
          f"({args.preview_steps}-step denoise each) ...")
    for pose_idx in pose_idxs:
        rays_np = rays_at_resolution(all_poses[pose_idx], H, W, focal, R, R)
        rays_t = torch.transpose(torch.from_numpy(rays_np).to(device), 0, 1)
        with torch.no_grad():
            depth_c, acc_c = render_rays_3dgrt(rays_t, model)
            depth_hw = depth_c.reshape(R, R)
            acc_hw = acc_c.reshape(R, R)
            depth_cond = make_depth_cond(depth_hw, acc_hw)  # 1x3xRxR

        panels = [to_uint8_rgb(depth_cond, signed=False)]
        for seed_i in range(args.n_seeds):
            torch.manual_seed(1234 + seed_i)
            I_hat = prior.preview(depth_cond, num_steps=args.preview_steps)
            panels.append(to_uint8_rgb(I_hat))
        row = np.concatenate(panels, axis=1)
        rows.append(row)

        row_bgr = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, f"pose_{pose_idx:03d}.png")
        cv2.imwrite(out_path, row_bgr)
        print(f"[diag] pose {pose_idx:03d} -> {out_path}")

    grid = np.concatenate(rows, axis=0)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    grid_path = os.path.join(args.output_dir, "pose_coverage_grid.png")
    cv2.imwrite(grid_path, grid_bgr)
    print(f"[diag] Full grid (D_B | seed0 Î_B | seed1 Î_B ... per pose row) -> {grid_path}")


if __name__ == '__main__':
    main()

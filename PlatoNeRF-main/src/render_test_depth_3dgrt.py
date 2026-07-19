#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# 3DGRT version of render_test_depth.py — loads a trained 3DGRT checkpoint
# and renders first-bounce depth maps for a set of camera poses, producing
# per-pose .npy / .png files and a flyaround video.

import cv2
import imageio
import numpy as np
import os
import sys
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make sure the 3DGRUT repo root is on the path so threedgrut is importable.
# Assumes this file lives at: <repo_root>/PlatoNeRF-main/src/
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch

from utils.load_tof import load_tof_data
from utils.nerf_helpers import get_rays_np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 3DGRT config (must match what was used during training)
# ---------------------------------------------------------------------------

def create_3dgrt_conf(n_iters=200000):
    densify_end = min(15000, n_iters // 2)
    return OmegaConf.create({
        "render": {
            "method": "3dgrt",
            "pipeline_type": "reference",
            "backward_pipeline_type": "referenceBwd",
            "particle_kernel_degree": 4,
            "particle_kernel_density_clamping": True,
            "particle_kernel_min_response": 0.0113,
            "particle_kernel_min_alpha": float(1.0 / 255.0),
            "particle_kernel_max_alpha": 0.99,
            "particle_radiance_sph_degree": 3,
            "primitive_type": "instances",
            "min_transmittance": 0.001,
            "max_consecutive_bvh_update": 15,
            "enable_normals": False,
            "enable_hitcounts": False,
            "enable_kernel_timings": False,
        },
        "model": {
            "density_activation": "sigmoid",
            "scale_activation": "exp",
            "default_density": 0.1,
            "default_scale_factor": 1.0,
            "optimize_density": True,
            "optimize_features_albedo": True,
            "optimize_features_specular": True,
            "optimize_position": True,
            "optimize_rotation": True,
            "optimize_scale": True,
            "bvh_update_frequency": 1,
            "progressive_training": {
                "feature_type": "sh",
                "init_n_features": 0,
                "max_n_features": 3,
                "increase_frequency": 1000,
                "increase_step": 1,
            },
            "background": {
                "name": "background-color",
                "color": "black",
            },
            "print_stats": False,
        },
        "optimizer": {
            "type": "adam",
            "lr": 0.0,
            "eps": 1e-15,
            "params": {
                "positions":         {"lr": 0.00016},
                "density":           {"lr": 0.05},
                "features_albedo":   {"lr": 0.0025},
                "features_specular": {"lr": 0.000125},
                "rotation":          {"lr": 0.001},
                "scale":             {"lr": 0.005},
            },
        },
        "scheduler": {
            "positions": {
                "type": "exp",
                "lr_init": 0.00016,
                "lr_final": 0.0000016,
                "max_steps": n_iters,
            },
            "density": {"type": "skip"},
        },
        "strategy": {
            "method": "GSStrategy",
            "print_stats": False,
            "densify": {
                "params": "positions",
                "frequency": 300,
                "start_iteration": 500,
                "end_iteration": densify_end,
                "clone_grad_threshold": 0.0002,
                "split_grad_threshold": 0.0002,
                "relative_size_threshold": 0.01,
                "split": {"n_gaussians": 2},
            },
            "prune": {
                "frequency": 100,
                "start_iteration": 500,
                "end_iteration": densify_end,
                "density_threshold": 0.005,
            },
            "reset_density": {
                "frequency": 3000,
                "start_iteration": 0,
                "end_iteration": densify_end,
                "new_max_density": 0.01,
            },
            "density_decay": {
                "gamma": 0.99,
                "start_iteration": -1,
                "end_iteration": -1,
                "frequency": 50,
            },
            "prune_weight": {
                "frequency": 100,
                "start_iteration": -1,
                "end_iteration": -1,
                "weight_threshold": 0.5,
            },
            "prune_scale": {
                "frequency": 100,
                "start_iteration": -1,
                "end_iteration": -1,
                "threshold": 1.0,
            },
        },
        "checkpoint": {"iterations": [n_iters]},
    })


# ---------------------------------------------------------------------------
# Rendering helper (mirrors render_rays_3dgrt in run_platonerf_3dgrt.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_rays_3dgrt(batch_rays, model):
    """Render a batch of rays with the 3DGRT model.

    Args:
        batch_rays: Tensor [2, N, 3] — (origins, directions), world space,
                    directions already normalised.
        model: MixtureOfGaussians (inference mode, no grad).

    Returns:
        depth: [N]    first-bounce distance
        acc:   [N]    accumulated opacity
        rgb:   [N, 3] predicted colour
    """
    ray_o = batch_rays[0]   # [N, 3]
    ray_d = batch_rays[1]   # [N, 3]

    rays_ori = ray_o.unsqueeze(0).unsqueeze(2)   # [1, N, 1, 3]
    rays_dir = ray_d.unsqueeze(0).unsqueeze(2)   # [1, N, 1, 3]
    T_to_world = torch.eye(4, device=ray_o.device, dtype=ray_o.dtype).unsqueeze(0)

    gpu_batch = Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=T_to_world,
        rays_in_world_space=True,
    )

    out = model(gpu_batch, train=False, frame_id=0)

    depth = out["pred_dist"].squeeze(0).squeeze(1).squeeze(-1)     # [N]
    acc   = out["pred_opacity"].squeeze(0).squeeze(1).squeeze(-1)  # [N]
    rgb   = out["pred_rgb"].squeeze(0).squeeze(1)                  # [N, 3]
    return depth, acc, rgb


# ---------------------------------------------------------------------------
# Camera helpers (identical to render_test_depth.py)
# ---------------------------------------------------------------------------

def look_at(vec_pos, vec_look_at):
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.array([0., 1., 0.]))
    x = x / np.linalg.norm(x)
    y = np.cross(x, z)
    view_mat = np.zeros((4, 4))
    view_mat[:3, 0] = x
    view_mat[:3, 1] = y
    view_mat[:3, 2] = -z
    view_mat[:3, 3] = vec_pos
    view_mat[3, :] = [0., 0., 0., 1.]
    return view_mat


def compute_points_around_circle(origin, radius, num_points, start_angle):
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, num_points, endpoint=False)
    x_coords = origin[0] + radius * np.cos(angles)
    y_coords = origin[1] + radius * np.sin(angles)
    return np.column_stack((x_coords, y_coords))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path (reuse the training .txt config)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='directory to write depth_predictions/ into')
    parser.add_argument("--expname", type=str, required=True,
                        help='experiment name (must match training)')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where checkpoints were saved')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific checkpoint .tar to load; defaults to latest in basedir/expname')
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='N_iters used during training (needed to reconstruct the model config)')
    parser.add_argument("--dataset_type", type=str, default='dtof')
    parser.add_argument("--ignore", type=int, action='append', required=False, default=[])
    parser.add_argument("--render_chunk", type=int, default=4096,
                        help='rays per chunk during inference (reduce if OOM)')
    # Training-config keys that may appear in .txt files — silently accepted
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--per_image_thresh", type=float, action='append', required=False)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = config_parser()
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"[INFO] Ignoring unrecognised config keys: {_unknown}")

    # ------------------------------------------------------------------
    # Load data (we only need poses and camera intrinsics)
    # ------------------------------------------------------------------
    if args.dataset_type != "dtof":
        print("Unknown dataset_type:", args.dataset_type)
        return

    tof, poses, light_o, light_d, hwf, walls_cam, walls_light = load_tof_data(
        args.datadir, args.ignore
    )
    print("Loaded ToF data:", tof.shape, hwf, args.datadir)

    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([
        [focal, 0,     0.5 * W],
        [0,     focal, 0.5 * H],
        [0,     0,     1      ],
    ])

    # ------------------------------------------------------------------
    # Load 3DGRT model from checkpoint
    # ------------------------------------------------------------------
    conf = create_3dgrt_conf(args.N_iters)
    far = 6.0
    scene_extent = float(far)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)

    ckpt_dir = os.path.join(args.basedir, args.expname)
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = sorted([
            os.path.join(ckpt_dir, f)
            for f in os.listdir(ckpt_dir)
            if f.endswith('.tar') and not f.endswith('_strategy.tar')
        ])

    valid_ckpts = []
    for p in ckpts:
        try:
            probe = torch.load(p, map_location='cpu', weights_only=False)
            if 'positions' in probe:
                valid_ckpts.append(p)
            else:
                print(f'[INFO] Skipping non-3DGRT checkpoint: {p}')
        except Exception as e:
            print(f'[WARN] Could not read {p}: {e}')

    if not valid_ckpts:
        print("No valid 3DGRT checkpoints found in", ckpt_dir)
        return

    ckpt_path = valid_ckpts[-1]
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    model.build_acc()
    model.eval()
    print(f"Model loaded: {model.num_gaussians} Gaussians")

    # ------------------------------------------------------------------
    # Build render poses (same construction as the original script)
    # ------------------------------------------------------------------
    render_poses = []

    # Camera orbiting on a circle around (0, y, -3)
    origin = (0, 3)
    radius = 0.99
    num_points = 100
    points = compute_points_around_circle(origin, radius, num_points, -np.pi / 2)
    y = -1.5
    lookat = np.array([0, y, -3])
    for point in points:
        cam_origin = np.array([-point[0], y, -point[1]])
        cam_extrin = look_at(cam_origin, lookat)
        render_poses.append(torch.Tensor(cam_extrin))

    render_poses = torch.stack(render_poses, 0)
    print(f"Rendering {len(render_poses)} poses at {H}x{W}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = os.path.join(args.output_dir, "depth_predictions")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------
    chunk = args.render_chunk
    depth_images = []
    rgb_images = []

    for pose_i, pose in enumerate(tqdm(render_poses)):
        png_path     = os.path.join(output_dir, f"depth_map_{str(pose_i).zfill(3)}.png")
        npy_path     = os.path.join(output_dir, f"depth_map_{str(pose_i).zfill(3)}.npy")
        rgb_png_path = os.path.join(output_dir, f"rgb_map_{str(pose_i).zfill(3)}.png")

        if os.path.exists(png_path) and os.path.exists(rgb_png_path):
            print(f"{pose_i} exists, loading for video")
            depth_img = np.load(npy_path) / 4.65
            depth_images.append(depth_img)
            rgb_images.append(cv2.cvtColor(cv2.imread(rgb_png_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            continue

        # Build rays for this pose: [H*W, 2, 3]
        pose_np = pose.unsqueeze(0).detach().cpu().numpy()  # [1, 4, 4]
        rays_np = np.stack([get_rays_np(H, W, K, p) for p in pose_np[:, :3, :4]], 0)  # [1, 2, H, W, 3]
        rays_np = np.transpose(rays_np, [0, 2, 3, 1, 4])   # [1, H, W, 2, 3]
        rays_np = rays_np.reshape(-1, 2, 3).astype(np.float32)  # [H*W, 2, 3]

        # Normalise directions
        norms = np.linalg.norm(rays_np[:, 1, :], axis=1, keepdims=True)
        rays_np[:, 1, :] /= norms

        rays_t = torch.from_numpy(rays_np).to(device)   # [H*W, 2, 3]

        depth_chunks, rgb_chunks = [], []
        for c in range(0, rays_t.shape[0], chunk):
            r = torch.transpose(rays_t[c:c + chunk], 0, 1)   # [2, C, 3]
            d, _, rgb_c = render_rays_3dgrt(r, model)
            depth_chunks.append(d.cpu())
            rgb_chunks.append(rgb_c.cpu())

        depth_map = torch.cat(depth_chunks).reshape(H, W).numpy()
        rgb_map   = torch.cat(rgb_chunks).reshape(H, W, 3).numpy()
        rgb_map   = np.clip(rgb_map, 0, 1)

        # Save raw depth values
        np.save(npy_path, depth_map)

        # Save normalised PNG (same normalisation as the original: divide by 4.65)
        depth_vis = depth_map / 4.65
        depth_vis = np.clip(depth_vis, 0, 1)
        cv2.imwrite(png_path, (depth_vis * 255).astype(np.uint8))

        # Save RGB
        cv2.imwrite(rgb_png_path, cv2.cvtColor((rgb_map * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        depth_images.append(depth_vis)
        rgb_images.append(rgb_map)

    # ------------------------------------------------------------------
    # Write videos
    # ------------------------------------------------------------------
    if depth_images:
        frames = [(np.clip(d, 0, 1) * 255).astype(np.uint8) for d in depth_images]
        imageio.mimwrite(os.path.join(output_dir, 'video.mp4'), frames, fps=15, quality=8)
        print("Depth video saved to", os.path.join(output_dir, 'video.mp4'))

    if rgb_images:
        rgb_frames = [(np.clip(r, 0, 1) * 255).astype(np.uint8) for r in rgb_images]
        imageio.mimwrite(os.path.join(output_dir, 'rgb_video.mp4'), rgb_frames, fps=15, quality=8)
        print("RGB video saved to", os.path.join(output_dir, 'rgb_video.mp4'))

    print("Done. Outputs in:", output_dir)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()

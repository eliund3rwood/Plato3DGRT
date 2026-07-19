#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Extends run_platonerf_3dgrt.py with a Phase 3: Variational Score
# Distillation (VSD) texture-refinement stage, using the custom_controlnet
# novel-view diffusion prior (I_A + D_B -> predicted I_B). Phase 1 (distance)
# and Phase 2 (distance + shadow) are byte-for-byte identical to
# run_platonerf_3dgrt.py — VSD is purely additive on top, active only for
# iterations beyond --N_iters, for --vsd_iters additional steps.
#
# See PlatoNeRF-main/src/vsd_prior.py for the diffusion-prior wrapper and
# mit_cluster_setup.md for how to run this on the cluster.

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as scisig
import sys
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

# ---------------------------------------------------------------------------
# Make sure the 3DGRUT repo root is on the path so threedgrut is importable.
# Assumes this file lives at: <repo_root>/PlatoNeRF-main/src/run_platonerf_3dgrt_vsd.py
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch
from threedgrut.strategy.gs import GSStrategy

from utils.load_tof import load_tof_data
from utils.nerf_helpers import get_rays_np   # still needed for camera ray generation
from utils.novel_views import novel_view_poses, rays_at_resolution, dolly_poses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

SPEED_OF_LIGHT = 2.99792458e8
TIME_RES_M = 0.0384
TIME_RES_S = 0.0384 / SPEED_OF_LIGHT
FWHM_TO_SIGMA = 2.35482004503


# ---------------------------------------------------------------------------
# 3DGRT config builder (identical to run_platonerf_3dgrt.py)
# ---------------------------------------------------------------------------

def create_3dgrt_conf(args):
    """Build an OmegaConf config for 3DGRT from PlatoNeRF's parsed args."""
    n_iters = args.N_iters
    densify_end = n_iters                    # clone/split runs to end; 200k safety valve in train() limits explosion
    prune_end   = n_iters                    # prune continues to end to clear ghost Gaussians

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
                "scale":             {"lr": 0.002},
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
                "clone_grad_threshold": 0.0008,
                "split_grad_threshold": 0.0008,
                "relative_size_threshold": 0.01,
                "split": {"n_gaussians": 2},
            },
            "prune": {
                "frequency": 100,
                "start_iteration": 500,
                "end_iteration": prune_end,
                "density_threshold": 0.05,
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
        "checkpoint": {"iterations": [7000, n_iters]},
    })


# ---------------------------------------------------------------------------
# 3DGRT rendering wrapper — returns full RGB in extras["rgb"] (needed for VSD)
# ---------------------------------------------------------------------------

def render_rays_3dgrt(batch_rays, model, train=True, frame_id=0, per_ray_far=None):
    """Render a batch of rays using 3DGRT and return outputs in PlatoNeRF format.

    Args:
        batch_rays: Tensor [2, N, 3] — first dim is (origins, directions),
                    both already in world space and directions already normalised.
        model: MixtureOfGaussians instance.
        train: bool — passed to model.forward() to enable/disable grad tracking.
        frame_id: int — frame counter for the BVH / SH progressive training.
        per_ray_far: optional Tensor [N] — maximum hit distance per ray.
                     Hits beyond this distance are treated as transparent (trans=1).

    Returns:
        (intensity, disp, acc, depth, trans, extras)
        - intensity: [N]  mean radiance (kept for API compat with Phase 1/2 code)
        - disp: None
        - acc: [N]  accumulated opacity (1 - transmittance)
        - depth: [N]  distance to first significant Gaussian hit
        - trans: [N]  transmittance along the ray  (1 - acc)
        - extras: dict with "batch" (the Batch object) and "rgb" ([N,3] colour, for VSD)
    """
    ray_o = batch_rays[0]   # [N, 3]
    ray_d = batch_rays[1]   # [N, 3]

    rays_ori = ray_o.unsqueeze(0).unsqueeze(2)   # [1, N, 1, 3]
    rays_dir = ray_d.unsqueeze(0).unsqueeze(2)   # [1, N, 1, 3]

    T_to_world = torch.eye(4, device=ray_o.device, dtype=ray_o.dtype).unsqueeze(0)  # [1, 4, 4]

    gpu_batch = Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=T_to_world,
        rays_in_world_space=True,
    )

    out = model(gpu_batch, train=train, frame_id=frame_id)

    depth     = out["pred_dist"].squeeze(0).squeeze(1).squeeze(-1)      # [N]
    acc       = out["pred_opacity"].squeeze(0).squeeze(1).squeeze(-1)   # [N]
    trans     = 1.0 - acc                                                 # [N]
    rgb       = out["pred_rgb"].squeeze(0).squeeze(1)                    # [N, 3]
    intensity = rgb.mean(-1)                                             # [N]

    if per_ray_far is not None:
        beyond = depth > per_ray_far
        trans = torch.where(beyond, torch.ones_like(trans), trans)
        acc   = torch.where(beyond, torch.zeros_like(acc),  acc)

    return intensity, None, acc, depth, trans, {"batch": gpu_batch, "rgb": rgb}


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged from original)
# ---------------------------------------------------------------------------

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return None
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def find_ray_intersections(rays, walls):
    pixels = []
    intersections = []
    distances = []
    for idx in range(rays.shape[0]):
        ray_o = rays[idx, 0]
        ray_d = rays[idx, 1]
        plane_normal = np.array(walls[idx, :3])
        plane_point = np.array(walls[idx, 3:6])
        x = walls[idx, 6]
        min_y = walls[idx, 7]
        max_y = walls[idx, 8]
        min_z = walls[idx, 9]
        max_z = walls[idx, 10]

        intersection_point = LinePlaneCollision(plane_normal, plane_point, ray_d, ray_o)
        if intersection_point is not None and \
                intersection_point[1] >= min_y and intersection_point[1] <= max_y and \
                intersection_point[2] >= min_z and intersection_point[2] <= max_z:
            pixels.append([idx])
            intersections.append(intersection_point)
            distances.append(np.linalg.norm(ray_o - intersection_point))
    return pixels, intersections, distances


def find_ray_intersections_from_tof(rays, tof):
    """Uses first bounce return to compute where the intersection of light and scene occurs."""
    pixels = []
    intersections = []
    distances = []
    for i in range(rays.shape[0]):
        rays_o, rays_d = rays[i][0], rays[i][1]
        tof_i = np.reshape(tof[i], [-1, tof[i].shape[-1]])
        intensities = np.max(tof_i, axis=1)
        binidxs = np.argmax(tof_i, axis=1)
        idx = binidxs[np.argmax(intensities)]
        distance = ((idx + 1) * TIME_RES_M) / 2.0
        intersection = rays_o + (distance * rays_d)
        pixels.append([i])
        intersections.append(intersection)
        distances.append(distance)
    return pixels, intersections, distances


EPSILON = 1e-5


def normalize_min_max(tensor, new_max=1.0, new_min=0.0):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + EPSILON) * (new_max - new_min) + new_min


# ---------------------------------------------------------------------------
# Phase 3 (VSD) helper: qualitative preview panel
# ---------------------------------------------------------------------------

def _save_vsd_preview(vsd_prior, I_A, rgb_render, depth_cond, img_savedir, i, num_steps=20):
    # Deliberately avoids torchvision.utils.make_grid/save_image — on this
    # cluster's torch build it reliably threw "can't convert cuda:0 device
    # type tensor to numpy" regardless of explicit .cpu() calls beforehand
    # (likely another torch-ecosystem-extension ABI mismatch, same flavor as
    # the torchaudio issue in mit_cluster_setup.md's gotchas). cv2.imwrite
    # already works for depth_*/acc_* preview images elsewhere in this file,
    # so build the 4-panel grid manually with numpy instead.
    try:
        I_hat = vsd_prior.preview(depth_cond, num_steps=num_steps)

        def to_uint8_rgb(x, signed=True):
            x = x.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
            if signed:
                x = (np.clip(x, -1, 1) + 1) / 2
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)

        panels = [
            to_uint8_rgb(I_A),
            to_uint8_rgb(depth_cond, signed=False),
            to_uint8_rgb(rgb_render),
            to_uint8_rgb(I_hat),
        ]
        grid_rgb = np.concatenate(panels, axis=1)
        grid_bgr = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(img_savedir, f"vsd_{i:06d}.png")
        cv2.imwrite(out_path, grid_bgr)
        tqdm.write(f"[VSD] Saved preview -> {out_path}  (I_A | D_B | render | diffusion)")
    except Exception:
        import traceback
        tqdm.write("[VSD] Preview failed (non-fatal):\n" + traceback.format_exc())


def _save_lpips_ref_preview(rgb_render, I_A, img_savedir, i):
    """Render at the LPIPS grounding pose (dolly pose 30) side-by-side with
    the reference photo — lets us visually confirm the pose actually lines
    up (rather than just watching loss_lpips, which can't distinguish
    "under-trained" from "supervising the wrong view") and watch convergence
    at this one known-correct view directly."""
    try:
        def to_uint8_rgb(x):
            x = x.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
            x = (np.clip(x, -1, 1) + 1) / 2
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)

        panel_rgb = np.concatenate([to_uint8_rgb(rgb_render), to_uint8_rgb(I_A)], axis=1)
        panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(img_savedir, f"lpips_ref_{i:06d}.png")
        cv2.imwrite(out_path, panel_bgr)
        tqdm.write(f"[VSD] Saved LPIPS-ref preview -> {out_path}  (render | I_A reference)")
    except Exception:
        import traceback
        tqdm.write("[VSD] LPIPS-ref preview failed (non-fatal):\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--use_all_rays", type=int, default=0,
                        help='whether or not to use all rays')
    parser.add_argument("--per_image_thresh", type=float, action='append', required=True)
    parser.add_argument("--debug", type=int, default=0,
                        help='whether or not to debug')
    parser.add_argument("--near", type=float, default=0.1,
                        help='near plane')
    parser.add_argument("--dist_weight", type=int, default=1000,
                        help='dist weight')
    parser.add_argument("--extract_first", type=int, default=0,
                        help='whether or not to extract 1b distance to compute projected illumination')
    parser.add_argument("--parallel", type=float, default=0.05,
                        help='parallel filter')
    parser.add_argument("--shadw", type=float, default=1.0,
                        help='shadow loss weight for shadow pixels')
    parser.add_argument("--nonshadw", type=float, default=1.0,
                        help='shadow loss weight for non shadow pixels')
    parser.add_argument("--ignore", type=int, action='append', required=False, default=[])
    parser.add_argument("--downsample", type=int, default=1,
                        help='downsample rays by factor of x')
    parser.add_argument("--downsample_temp", type=int, default=0,
                        help='downsample temporal bins by factor of x')
    parser.add_argument("--save_lights", type=int, default=0,
                        help='save light paths')
    parser.add_argument("--vis_rays", type=int, default=0,
                        help='visualize ray paths via a video; 1 true, 0 false')

    # training options
    parser.add_argument("--noise", type=float, default=0.0,
                        help="gaussian noise on time of arrival")
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_iters", type=int, default=35000,
                        help='total number of Phase 1+2 (geometry/shadow) iterations')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific checkpoint .tar file to reload')

    # 3DGRT-specific options
    parser.add_argument("--num_gaussians", type=int, default=100_000,
                        help='initial number of 3D Gaussians (default 100k; safe for 12GB VRAM)')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='dtof',
                        help='options: dtof')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout'),
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving'),
    parser.add_argument("--simple_mode", action='store_true',
                        help='whether or not to only render 1B to 2B rays'),
    parser.add_argument("--sigmoid", action='store_true',
                        help='whether or not to apply sigmoid on shadows'),
    parser.add_argument("--bce", action='store_false',
                        help='whether or not to use BCE loss'),

    # -----------------------------------------------------------------
    # Phase 3: VSD (Variational Score Distillation) texture refinement
    # -----------------------------------------------------------------
    parser.add_argument("--vsd_iters", type=int, default=0,
                        help='additional iterations after N_iters for VSD; 0 disables Phase 3 entirely')
    parser.add_argument("--vsd_checkpoint", type=str, default=None,
                        help='path to the custom_controlnet diffusion prior checkpoint (final.pt)')
    parser.add_argument("--vsd_controlnet_root", type=str, default=None,
                        help='path to the vendored custom_controlnet repo root; '
                             'defaults to <repo_root>/custom_controlnet')
    parser.add_argument("--vsd_ref_image", type=str, default=None,
                        help='fixed I_A reference RGB (512x512); defaults to '
                             '<datadir>/../chair_smooth_walls.png')
    parser.add_argument("--vsd_variant", type=str, default="vsd", choices=["vsd", "sds"],
                        help='"vsd" = true Variational Score Distillation (LoRA particle net, '
                             'sharper texture); "sds" = classic SDS (cheaper, no LoRA)')
    parser.add_argument("--vsd_every_n_steps", type=int, default=4,
                        help='run one VSD step every N Phase-3 iterations (ToF geometry+shadow '
                             'loss keeps running every iteration regardless)')
    parser.add_argument("--vsd_weight", type=float, default=1.0,
                        help='scalar weight on the VSD surrogate loss')
    parser.add_argument("--vsd_guidance_scale", type=float, default=3.0,
                        help='classifier-free guidance scale for the pretrained (appearance) branch')
    parser.add_argument("--vsd_lora_rank", type=int, default=8,
                        help='LoRA rank for the VSD particle UNet adapter')
    parser.add_argument("--vsd_lora_lr", type=float, default=1e-4,
                        help='learning rate for the LoRA particle network optimizer')
    parser.add_argument("--vsd_t_min_frac", type=float, default=0.02,
                        help='minimum diffusion timestep fraction sampled for VSD/SDS')
    parser.add_argument("--vsd_t_max_frac", type=float, default=0.75,
                        help='maximum diffusion timestep fraction sampled for VSD/SDS')
    parser.add_argument("--vsd_grad_weight", type=str, default="snr", choices=["snr", "uniform"],
                        help='timestep weighting for the (V)SD gradient')
    parser.add_argument("--vsd_render_res", type=int, default=512,
                        help='render resolution for VSD novel views (must match the diffusion '
                             'model spatial contract — 512)')
    parser.add_argument("--vsd_n_orbit_poses", type=int, default=100,
                        help='number of orbit poses in the VSD novel-view sampling bank')
    parser.add_argument("--vsd_include_dolly", type=int, default=1,
                        help='also include the 39-pose SPAD dolly-sweep in the sampling bank')
    parser.add_argument("--vsd_freeze_geometry", type=int, default=1,
                        help='1 = VSD only updates albedo/specular (texture); geometry '
                             '(positions/rotation/scale/density) stays driven solely by the ToF '
                             'distance+shadow loss. 0 = let VSD also refine geometry (experimental).')
    parser.add_argument("--vsd_preview_steps", type=int, default=20,
                        help='number of denoising steps for the periodic qualitative preview panel')
    parser.add_argument("--vsd_batch_size", type=int, default=4,
                        help='number of novel-view poses rendered and batched through the '
                             'diffusion prior per VSD step. >1 measurably reduces the '
                             'view-inconsistent texture that single-view-per-step SDS/VSD '
                             'produces (different Gaussians get pulled by whatever noise '
                             'sample happened to land on them); costs proportionally more '
                             'compute per step.')
    parser.add_argument("--vsd_color_lr_decay", type=float, default=0.1,
                        help='features_albedo/features_specular learning rate is multiplied by '
                             'an exponential decay from 1.0 down to this factor over the course '
                             'of Phase 3 (vsd_iters). Constant LR against the inherently noisy '
                             'VSD gradient is a known cause of textures never smoothing out; '
                             'set to 1.0 to disable (constant LR, prior behavior).')
    parser.add_argument("--vsd_geom_reg_weight", type=float, default=10000.0,
                        help='MSE anchor weight pulling positions/rotation/scale/density back '
                             'toward their value at the start of Phase 3 (the ToF-lidar-grounded '
                             'geometry). Only meaningful when --vsd_freeze_geometry 0 — lets VSD '
                             'nudge geometry for texture-driven corrections without letting it '
                             'wander far from the physically-grounded solution. 0 disables.')
    parser.add_argument("--vsd_smooth_weight", type=float, default=1.0,
                        help='total-variation smoothness weight on the rendered RGB image each '
                             'VSD step. Targets high-frequency per-Gaussian-color noise directly '
                             '(independent of view-to-view inconsistency, which batching already '
                             'addresses). 0 disables.')
    parser.add_argument("--vsd_lpips_weight", type=float, default=10.0,
                        help='LPIPS perceptual-loss weight between a render from the fixed '
                             'reference pose (dolly pose 30, matching chair_smooth_walls.png / '
                             'ground_truth_test/0030.png) and the reference image itself. Unlike '
                             'VSD (a noisy, hallucinated target), this is real ground truth, so it '
                             'always gets full gradient access to both color AND geometry '
                             'regardless of --vsd_freeze_geometry. 0 disables.')
    parser.add_argument("--vsd_lpips_every_n_steps", type=int, default=4,
                        help='compute the LPIPS grounding term every N Phase-3 iterations')

    return parser


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    parser = config_parser()
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"[INFO] Ignoring unrecognised config keys (NeRF-only): {_unknown}")

    if args.vsd_iters > 0 and not args.vsd_checkpoint:
        raise ValueError("--vsd_iters > 0 requires --vsd_checkpoint (path to custom_controlnet's final.pt)")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    K = None
    if args.dataset_type == "dtof":
        tof, poses, light_o, light_d, hwf, walls_cam, walls_light = load_tof_data(args.datadir, args.ignore)
        print('Loaded ToF data', tof.shape, light_o.shape, light_d.shape, hwf, args.datadir,
              walls_cam.shape, walls_light.shape)
        i_train = np.arange(tof.shape[0])
        i_val = []
        i_test = []
        print("Train idxs: {}".format(i_train))
        near = args.near
        far = 6.0
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # ------------------------------------------------------------------
    # Create log dir and copy config
    # ------------------------------------------------------------------
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # ------------------------------------------------------------------
    # Create 3DGRT model
    # ------------------------------------------------------------------
    conf = create_3dgrt_conf(args)
    scene_extent = float(far)

    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)

    scene_half = scene_extent * 1.1
    model.init_from_random_point_cloud(
        num_gaussians=args.num_gaussians,
        xyz_min=-scene_half,
        xyz_max=scene_half,
    )
    model.setup_optimizer()
    model.build_acc()

    strategy = GSStrategy(conf, model)
    strategy.init_densification_buffer()

    global_step = 0
    start = 0

    # ------------------------------------------------------------------
    # Optionally reload from checkpoint
    # ------------------------------------------------------------------
    ckpt_dir = os.path.join(basedir, expname)
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
    ckpts = valid_ckpts

    reloaded_ckpt_path = None
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        reloaded_ckpt_path = ckpt_path
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.init_from_checkpoint(ckpt, setup_optimizer=True)
        start = ckpt.get('global_step', 0)
        global_step = start

        strategy_path = ckpt_path.replace('.tar', '_strategy.tar')
        if os.path.isfile(strategy_path):
            strategy_ckpt = torch.load(strategy_path, map_location=device, weights_only=False)
            strategy.init_densification_buffer(checkpoint=strategy_ckpt)
        else:
            strategy.init_densification_buffer()

        model.build_acc(rebuild=True)

    # ------------------------------------------------------------------
    # Prepare ray batches
    # ------------------------------------------------------------------
    N_rand = args.N_rand

    print('get rays')
    poses[:, 2, :] *= -1
    poses[:, 0, :] *= -1
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]

    light_o = np.stack([light_o[i] for i in i_train], 0)
    light_d = np.stack([light_d[i] for i in i_train], 0)
    walls_cam = np.stack([walls_cam[i] for i in i_train], 0)
    walls_light = np.stack([walls_light[i] for i in i_train], 0)
    tof = np.stack([tof[i] for i in i_train], 0)

    rays = np.transpose(rays, [0, 2, 3, 1, 4])   # [N, H, W, ro+rd, 3]
    rays = np.stack([rays[i] for i in i_train], 0)
    rays = np.reshape(rays, [-1, 2, 3])            # [N*H*W, 2, 3]
    rays = rays.astype(np.float32)

    rays_d = rays[:, 1, :]
    norm = np.linalg.norm(rays[:, 1, :], axis=1)
    rays[:, 1] = rays_d / norm[:, None]

    walls_cam = np.tile(walls_cam, H * W).reshape((walls_cam.shape[0], H, W, walls_cam.shape[1]))
    walls_cam = np.reshape(walls_cam, [-1, walls_cam.shape[-1]])

    i_batch = 0

    # ------------------------------------------------------------------
    # Light intersections
    # ------------------------------------------------------------------
    light_rays = np.stack([light_o, light_d], 1)
    light_idx, light_inters, light_dists = None, None, None
    if args.extract_first == 1:
        light_idx, light_inters, light_dists = find_ray_intersections_from_tof(light_rays, tof)
    else:
        light_idx, light_inters, light_dists = find_ray_intersections(light_rays, walls_light)
    light_inters = np.array(light_inters)

    if args.save_lights == 1:
        np.save("{}/lights.npy".format(os.path.join(basedir, expname)), light_inters)
        exit()

    print("The following projected lights have been found:", light_idx)
    assert len(light_idx) == len(i_train)
    light_inters = np.tile(light_inters, H * W).reshape((light_inters.shape[0], H, W, light_inters.shape[1]))
    light_inters = np.reshape(light_inters, [-1, light_inters.shape[-1]])

    light_dists = np.array(light_dists)
    light_dists = np.tile(light_dists[:, None], H * W).reshape((light_dists.shape[0], H, W))
    light_dists = np.reshape(light_dists, [-1, 1])

    light_dirs = np.tile(light_d, H * W).reshape((light_d.shape[0], H, W, light_d.shape[1]))
    light_dirs = np.reshape(light_dirs, [-1, light_dirs.shape[-1]])

    # ------------------------------------------------------------------
    # Filter rays near illumination spot + matched filter shadow extraction
    # ------------------------------------------------------------------
    parallel = np.linalg.norm(np.cross(rays[:, 1, :], light_dirs), axis=1)
    mask = parallel < args.parallel
    mask = np.array(mask, dtype=np.uint8)

    b1 = parallel < 0.005
    b1 = np.array(b1, dtype=np.uint8)
    b1 = np.reshape(b1, [light_d.shape[0], H, W])

    tof = np.reshape(tof, [-1, tof.shape[-1]])
    tof = tof.astype(np.float32)

    noise = np.zeros([tof.shape[0], ])
    if args.noise != 0:
        noise_m = 1e-12 * args.noise * SPEED_OF_LIGHT
        noise = np.random.normal(0.0, noise_m / FWHM_TO_SIGMA, tof.shape[0])
        print("adding noise of {} m (FWHM)! Resulting min {}, max {} m.".format(
            noise_m, np.min(noise), np.max(noise)))
    noise = noise[::args.downsample]

    tof = np.reshape(tof, [light_d.shape[0], H, W, -1])
    mask = np.reshape(mask, [light_d.shape[0], H, W])
    tof_stack = []

    if args.debug == 0:
        tof_1b_norm = 0
        shadow_savedir = os.path.join(basedir, expname, "shadows")
        os.makedirs(shadow_savedir, exist_ok=True)
        for i, tofi in enumerate(tof):
            print("Preprocessing tof image {} of {}.".format(i + 1, tof.shape[0]))
            tof_1b = tofi[b1[i] == 1.0][0]
            tof_1b_norm = tof_1b / np.sum(tof_1b)

            tofi[mask[i] == 1.0] = 0.0
            corr = np.zeros([H, W])
            for pixeli in range(H):
                for pixelj in range(W):
                    pix = tofi[pixeli][pixelj]
                    if np.sum(pix) != 0:
                        pix = pix / np.sum(pix)
                    cc = scisig.correlate(pix, tof_1b_norm)
                    corr[pixeli, pixelj] = np.max(cc)

            thresh = args.per_image_thresh[i]
            corr[corr < thresh] = 0
            corr[corr >= thresh] = 1
            arrivals = np.multiply(tofi, np.expand_dims(corr, 2))
            tof_stack.append(arrivals)

            target_shadow = np.sum(arrivals, axis=2)
            target_shadow[target_shadow > 0.0] = 1.0
            target_shadow = np.stack([target_shadow, target_shadow, target_shadow], axis=2)
            target_shadow = np.float32(target_shadow)
            target_shadow = cv2.cvtColor(target_shadow, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("{}/shadow_{}.png".format(shadow_savedir, str(i).zfill(3)), target_shadow * 255)

        tof = np.stack(tof_stack, 0)

    tof = np.reshape(tof, [-1, tof.shape[-1]])
    tof = tof.astype(np.float32)
    rays = rays[::args.downsample]
    print("Rays: {}, Downsample: {}".format(rays.shape[0], args.downsample))
    light_inters = light_inters[::args.downsample]
    print("Light inters: {}, Downsample: {}".format(light_inters.shape[0], args.downsample))
    light_dists = light_dists[::args.downsample]
    print("Light dists: {}, Downsample: {}".format(light_dists.shape[0], args.downsample))
    light_dirs = light_dirs[::args.downsample]
    print("Light dirs: {}, Downsample: {}".format(light_dirs.shape[0], args.downsample))
    tof = tof[::args.downsample]
    print("ToF: {}, Downsample: {}".format(tof.shape[0], args.downsample))

    if args.downsample_temp:
        tof_down = np.zeros([tof.shape[0], int(tof.shape[1] / args.downsample_temp)])
        print("Integrating transient from shape {} to {} and temporal res {} to {}".format(
            tof.shape, tof_down.shape, TIME_RES_M, TIME_RES_M * args.downsample_temp))
        for i in range(0, tof_down.shape[1], 1):
            tof_down[:, i] = np.sum(tof[:, (i * args.downsample_temp):((i + 1) * args.downsample_temp)], axis=1)
        tof = tof_down
        globals()["TIME_RES_M"] = TIME_RES_M * args.downsample_temp
        globals()["TIME_RES_S"] = TIME_RES_S * args.downsample_temp
        print(tof.shape, TIME_RES_M, TIME_RES_S)

    # ------------------------------------------------------------------
    # Permute all training data in unison
    # ------------------------------------------------------------------
    print('shuffle rays and transients in unison')
    p = np.random.permutation(len(rays))
    tof = tof[p]
    rays = rays[p]
    light_inters = light_inters[p]
    light_dists = light_dists[p]
    light_dirs = light_dirs[p]
    print('done')

    # ------------------------------------------------------------------
    # Separate shadow / non-shadow rays
    # ------------------------------------------------------------------
    tof_sum = np.sum(tof, axis=1)
    indices = np.where(tof_sum == 0)[0]
    ftof = np.delete(tof, indices, axis=0)
    flight_inters = np.delete(light_inters, indices, axis=0)
    flight_dists = np.delete(light_dists, indices, axis=0)
    frays = np.delete(rays, indices, axis=0)
    fnoise = np.delete(noise, indices, axis=0)

    # Keep all training data on CPU — torch.Tensor() would put everything on GPU
    # because of set_default_tensor_type below. torch.from_numpy() is always CPU.
    # The .to(device) calls in the batch loop move only each small batch to GPU.
    # tof / ftof stay as numpy arrays — ~6.5 GB, never used for gradients.
    # Per-batch slices are converted to GPU tensors at iteration time.
    tof  = np.asarray(tof,  dtype=np.float32)
    ftof = np.asarray(ftof, dtype=np.float32)

    light_inters  = torch.Tensor(np.asarray(light_inters,  dtype=np.float32))
    light_dists   = torch.Tensor(np.asarray(light_dists,   dtype=np.float32))
    rays          = torch.Tensor(np.asarray(rays,          dtype=np.float32))
    noise         = torch.Tensor(np.asarray(noise,         dtype=np.float32))

    flight_inters = torch.Tensor(np.asarray(flight_inters, dtype=np.float32))
    flight_dists  = torch.Tensor(np.asarray(flight_dists,  dtype=np.float32))
    frays         = torch.Tensor(np.asarray(frays,         dtype=np.float32))
    fnoise        = torch.Tensor(np.asarray(fnoise,        dtype=np.float32))

    n_iters_pretrain = 10000
    DIST_WEIGHT = args.dist_weight

    N_iters = args.N_iters + 1
    total_iters = N_iters + max(args.vsd_iters, 0)

    # Pre-build a fixed set of rays for one full camera view (cam 0) used
    # solely for periodic depth/shadow preview images.
    _preview_rays_np = np.stack([get_rays_np(H, W, K, poses[0, :3, :4])], 0)  # [1, 2, H, W, 3]
    _preview_rays_np = np.transpose(_preview_rays_np, [0, 2, 3, 1, 4])        # [1, H, W, 2, 3]
    _preview_rays_np = _preview_rays_np.reshape(-1, 2, 3).astype(np.float32)  # [H*W, 2, 3]
    _prd = _preview_rays_np[:, 1, :]
    _prn = np.linalg.norm(_prd, axis=1, keepdims=True)
    _preview_rays_np[:, 1, :] = _prd / _prn
    _preview_rays = torch.Tensor(_preview_rays_np)  # [H*W, 2, 3]

    img_savedir = os.path.join(basedir, expname, "progress")
    os.makedirs(img_savedir, exist_ok=True)

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    print(f'3DGRT model: {model.num_gaussians} Gaussians | scene_extent={scene_extent}')

    criterion = torch.nn.MSELoss()

    # ------------------------------------------------------------------
    # Phase 3 setup: diffusion prior + novel-view pose bank (built
    # unconditionally whenever --vsd_iters > 0, regardless of whether we're
    # starting fresh or resuming mid-Phase-3, since the training loop below
    # branches on the *iteration index*, not on run history).
    # ------------------------------------------------------------------
    vsd_prior = None
    vsd_rays_gpu = None
    vsd_geom_params = None
    vsd_albedo_init = None
    vsd_specular_init = None
    vsd_positions_init = None
    vsd_rotation_init = None
    vsd_scale_init = None
    vsd_density_init = None
    vsd_color_param_groups = []
    vsd_color_base_lr = {}
    vsd_lpips_fn = None
    vsd_ref_rays_gpu = None
    I_A = None

    if args.vsd_iters > 0:
        from vsd_prior import DiffusionPrior, load_reference_image, make_depth_cond

        vsd_ref_path = args.vsd_ref_image or os.path.join(args.datadir, "..", "chair_smooth_walls.png")
        print(f"[VSD] Loading diffusion prior (variant={args.vsd_variant}) from {args.vsd_checkpoint} ...")
        prior_kwargs = dict(
            checkpoint_path=args.vsd_checkpoint,
            device=device,
            variant=args.vsd_variant,
            guidance_scale=args.vsd_guidance_scale,
            lora_rank=args.vsd_lora_rank,
            lora_lr=args.vsd_lora_lr,
            t_min_frac=args.vsd_t_min_frac,
            t_max_frac=args.vsd_t_max_frac,
            grad_weight=args.vsd_grad_weight,
        )
        if args.vsd_controlnet_root:
            prior_kwargs["controlnet_repo_root"] = args.vsd_controlnet_root

        # __main__ sets torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # globally so the PlatoNeRF ToF-loading code's bare torch.Tensor(...)
        # calls land on GPU without needing .to(device) everywhere. diffusers/
        # transformers were never written expecting that: internal buffers
        # built without an explicit device (e.g. a scheduler's own
        # self.alphas_cumprod) silently end up on CUDA too, and diffusers code
        # that assumes those are CPU-resident (e.g. UniPCMultistepScheduler.
        # set_timesteps() doing np.array(self.alphas_cumprod)) then crashes
        # with "can't convert cuda:0 device type tensor to numpy." Build the
        # whole diffusion prior under the real default (CPU) instead, then
        # restore the CUDA default for the rest of the script.
        torch.set_default_tensor_type(torch.FloatTensor)
        try:
            vsd_prior = DiffusionPrior(**prior_kwargs)
            I_A = load_reference_image(vsd_ref_path, device)
            vsd_prior.set_reference_image(I_A)
        finally:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        vsd_poses = novel_view_poses(
            n_orbit=args.vsd_n_orbit_poses, include_dolly=bool(args.vsd_include_dolly)
        )
        R = args.vsd_render_res
        vsd_rays_gpu = [
            torch.from_numpy(rays_at_resolution(p, H, W, focal, R, R)).to(device)
            for p in vsd_poses
        ]
        print(f"[VSD] {len(vsd_rays_gpu)} novel-view poses cached at {R}x{R} "
              f"(vsd_iters={args.vsd_iters}, every_n_steps={args.vsd_every_n_steps}, "
              f"batch_size={args.vsd_batch_size}, freeze_geometry={bool(args.vsd_freeze_geometry)}, "
              f"color_lr_decay={args.vsd_color_lr_decay})")

        # LPIPS grounding: dolly pose 30 is chair_smooth_walls.png's actual
        # camera pose (matches ground_truth_test/0030.png) — a real photo, not
        # a diffusion hallucination, so this term always gets full gradient
        # access to color AND geometry regardless of --vsd_freeze_geometry.
        if args.vsd_lpips_weight > 0:
            import lpips as lpips_lib
            vsd_lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
            for p in vsd_lpips_fn.parameters():
                p.requires_grad_(False)
            ref_pose = dolly_poses(39)[30]
            vsd_ref_rays_gpu = torch.from_numpy(
                rays_at_resolution(ref_pose, H, W, focal, R, R)
            ).to(device)
            print(f"[VSD] LPIPS grounding enabled: pose 30, weight={args.vsd_lpips_weight}, "
                  f"every_n_steps={args.vsd_lpips_every_n_steps}")

        vsd_geom_params = [model.positions, model.rotation, model.scale, model.density]

        # Base LR for features_albedo/features_specular's own optimizer param
        # groups, so we can exponentially decay them over Phase 3 (constant LR
        # against the inherently noisy VSD gradient is a known cause of
        # textures never smoothing out; neither has any scheduler entry in
        # create_3dgrt_conf so model.scheduler_step() never touches these,
        # safe to set param_group["lr"] directly each iteration below).
        vsd_color_param_groups = [
            pg for pg in model.optimizer.param_groups
            if pg["name"] in ("features_albedo", "features_specular")
        ]
        vsd_color_base_lr = {pg["name"]: pg["lr"] for pg in vsd_color_param_groups}

        # Snapshot albedo/specular at the start of Phase 3 — these have *no* other
        # supervision anywhere in this script (loss_2 is always 0), so VSD is the
        # only thing that will ever move them. Tracking drift from this snapshot
        # gives a direct, non-visual signal that gradient is actually reaching
        # texture, instead of only checking that the loss numbers look sane.
        vsd_albedo_init = model.features_albedo.detach().clone()
        vsd_specular_init = model.features_specular.detach().clone()

        # Same idea for geometry, used only when --vsd_freeze_geometry 0: this
        # is the ToF-lidar-grounded solution from Phase 1/2 (a real physical
        # measurement, not a guess) — vsd_geom_reg_weight anchors positions/
        # rotation/scale/density back toward it each VSD step so texture-driven
        # gradients can nudge geometry without letting it wander far from the
        # physically-grounded fit.
        vsd_positions_init = model.positions.detach().clone()
        vsd_rotation_init = model.rotation.detach().clone()
        vsd_scale_init = model.scale.detach().clone()
        vsd_density_init = model.density.detach().clone()

        # Resume LoRA/optimizer state if a sibling _vsd.pt exists next to the checkpoint we loaded.
        if reloaded_ckpt_path is not None:
            vsd_state_path = reloaded_ckpt_path.replace('.tar', '_vsd.pt')
            if os.path.isfile(vsd_state_path):
                print(f"[VSD] Resuming LoRA state from {vsd_state_path}")
                vsd_prior.load_state_dict(torch.load(vsd_state_path, map_location=device))

    start = start + 1
    for i in trange(start, total_iters):
        time0 = time.time()

        # ------------------------------------------------------------------
        # Sample a batch of rays
        # Phase 1 (i < n_iters_pretrain): only non-shadow rays, distance loss only
        # Phase 2 (n_iters_pretrain <= i <= N_iters): all rays, distance + shadow loss
        # Phase 3 (i > N_iters): same as Phase 2, plus an additive VSD term below
        # ------------------------------------------------------------------
        if i < n_iters_pretrain:
            dataset_size = flight_inters.shape[0]
            target_tof = torch.Tensor(ftof[i_batch:i_batch + N_rand]).to(device)
            batch_lights = flight_inters[i_batch:i_batch + N_rand].to(device)
            batch_light_dists = flight_dists[i_batch:i_batch + N_rand].squeeze().to(device)
            batch_rays = frays[i_batch:i_batch + N_rand, :, :]
            batch_rays = torch.transpose(batch_rays, 0, 1).to(device)
            batch_noise = fnoise[i_batch:i_batch + N_rand].to(device)
        else:
            dataset_size = light_inters.shape[0]
            target_tof = torch.Tensor(tof[i_batch:i_batch + N_rand]).to(device)
            batch_lights = light_inters[i_batch:i_batch + N_rand].to(device)
            batch_light_dists = light_dists[i_batch:i_batch + N_rand].squeeze().to(device)
            batch_rays = rays[i_batch:i_batch + N_rand, :, :]
            batch_rays = torch.transpose(batch_rays, 0, 1).to(device)
            batch_noise = noise[i_batch:i_batch + N_rand].to(device)

        target_dist_idx = torch.argmax(target_tof, dim=1)
        target_dist_idx[target_dist_idx > 0] += 1
        target_dist = ((target_dist_idx * TIME_RES_M) + batch_noise.squeeze()) / 15.0
        target_shadow = torch.sum(target_tof, dim=1)
        target_shadow[target_shadow > 0.0] = 1.0

        secondary_idxs = torch.arange(0, N_rand)
        shadow_idxs = torch.where(target_shadow == 0.0)[0]
        nonshadow_idxs = torch.where(target_shadow == 1.0)[0]

        if i < n_iters_pretrain:
            target_pred = target_dist
        else:
            target_pred = torch.stack([target_dist, target_shadow], dim=1)

        i_batch += N_rand
        if i_batch >= dataset_size:
            if i < n_iters_pretrain:
                rand_idx = torch.randperm(flight_inters.shape[0])
                ftof = ftof[rand_idx.cpu().numpy()]
                flight_inters = flight_inters[rand_idx]
                flight_dists = flight_dists[rand_idx]
                frays = frays[rand_idx]
                fnoise = fnoise[rand_idx]
            else:
                rand_idx = torch.randperm(light_inters.shape[0])
                tof = tof[rand_idx.cpu().numpy()]
                light_inters = light_inters[rand_idx]
                light_dists = light_dists[rand_idx]
                rays = rays[rand_idx]
                noise = noise[rand_idx]
            i_batch = 0

        # ------------------------------------------------------------------
        # Build a minimal Batch for the GSStrategy sensor-position update.
        # ------------------------------------------------------------------
        cam_pos = batch_rays[0].mean(0)  # [3] approximate camera / sensor origin
        sensor_T = torch.eye(4, device=device)
        sensor_T[:3, 3] = cam_pos
        sensor_batch = Batch(
            rays_ori=cam_pos.view(1, 1, 1, 3),
            rays_dir=torch.zeros(1, 1, 1, 3, device=device),
            T_to_world=sensor_T.unsqueeze(0),
            rays_in_world_space=True,
        )

        # ------------------------------------------------------------------
        # Core optimisation loop
        # ------------------------------------------------------------------

        # --- First bounce: camera → scene surface ---
        intensity, disp, acc, depth, trans, extras = render_rays_3dgrt(
            batch_rays, model, train=True, frame_id=i
        )

        batch_vray_o_wall = batch_rays[0] + torch.mul(batch_rays[1], depth[:, None])
        batch_vray_term = batch_lights
        batch_vray_d = batch_vray_term - batch_vray_o_wall
        norm = torch.norm(batch_vray_d, dim=1)
        batch_vray_d = batch_vray_d / norm[:, None]
        # Offset origin slightly along the ray direction to avoid self-intersection
        # with wall Gaussians (shadow acne). The original NeRF used near=0.1 for the same reason.
        VRAY_OFFSET = 0.1
        batch_vray_o = batch_vray_o_wall + batch_vray_d * VRAY_OFFSET
        batch_vrays = torch.stack([batch_vray_o, batch_vray_d], 0)

        total_distance_1 = (depth + norm + batch_light_dists) / 15.0

        loss_1 = torch.tensor(0.0, device=device)
        loss_2 = torch.tensor(0.0, device=device)

        if i < n_iters_pretrain:
            # Phase 1: distance loss only
            loss_1 = criterion(total_distance_1, target_pred)
        else:
            # --- Second bounce: scene surface → light (shadow signal) ---
            batch_vrays_sec = batch_vrays[:, secondary_idxs, :]
            norm_sec = (norm[secondary_idxs] - VRAY_OFFSET).clamp(min=1e-3)

            _, _, _, b_depth, b_trans, _ = render_rays_3dgrt(
                batch_vrays_sec, model, train=True, frame_id=i,
                per_ray_far=norm_sec
            )

            target_tof_sum = torch.sum(target_tof, dim=1)
            mask_dist = target_tof_sum > 0

            dist_loss = criterion(total_distance_1[mask_dist], target_pred[mask_dist, 0])

            shad_loss_shad    = criterion(target_pred[shadow_idxs, 1],    b_trans[shadow_idxs])
            shad_loss_nonshad = criterion(target_pred[nonshadow_idxs, 1], b_trans[nonshadow_idxs])

            shad_loss = (args.shadw * shad_loss_shad) + (args.nonshadw * shad_loss_nonshad)
            loss_1 = (DIST_WEIGHT * dist_loss) + shad_loss

        # ------------------------------------------------------------------
        # Phase 3: VSD step (purely additive; Phase 1/2 losses above are
        # untouched). Runs every --vsd_every_n_steps iterations once past
        # --N_iters. Samples --vsd_batch_size novel poses, renders them all
        # in one 512x512-per-image batched RGB+depth call, and asks the
        # diffusion prior (conditioned on the fixed I_A reference + these
        # depths) for a score-distillation gradient averaged over the batch
        # — a single view per step lets different Gaussians get pulled by
        # whatever noise/timestep sample happened to land on them that step,
        # which shows up as view-inconsistent texture (smooth in one
        # rendered angle, noisy/faceted in another); batching multiple views
        # per gradient step measurably reduces that.
        # ------------------------------------------------------------------
        loss_vsd = torch.tensor(0.0, device=device)
        if (vsd_prior is not None and i > args.N_iters
                and (i - args.N_iters) % args.vsd_every_n_steps == 0):

            # Exponentially decay features_albedo/features_specular's LR over
            # Phase 3 — constant LR against the inherently noisy VSD gradient
            # is a known cause of textures never smoothing out.
            progress = min(max((i - args.N_iters) / max(args.vsd_iters, 1), 0.0), 1.0)
            color_decay = args.vsd_color_lr_decay ** progress
            for pg in vsd_color_param_groups:
                pg["lr"] = vsd_color_base_lr[pg["name"]] * color_decay

            K = args.vsd_batch_size
            pose_idxs = np.random.randint(0, len(vsd_rays_gpu), size=K)
            combined_rays = torch.cat([vsd_rays_gpu[idx] for idx in pose_idxs], dim=0)  # [K*R*R, 2, 3]
            combined_rays_t = torch.transpose(combined_rays, 0, 1)                       # [2, K*R*R, 3]

            if args.vsd_freeze_geometry:
                for p in vsd_geom_params:
                    p.requires_grad_(False)

            _, _, acc_v, depth_v, _, extras_v = render_rays_3dgrt(
                combined_rays_t, model, train=True, frame_id=i
            )

            if args.vsd_freeze_geometry:
                for p in vsd_geom_params:
                    p.requires_grad_(True)

            R = args.vsd_render_res
            rgb_v = extras_v["rgb"].reshape(K, R, R, 3).permute(0, 3, 1, 2)  # Kx3xRxR
            rgb_v = rgb_v.clamp(0, 1) * 2.0 - 1.0                             # -> [-1, 1]
            depth_khw = depth_v.reshape(K, R, R)
            acc_khw = acc_v.reshape(K, R, R)
            # Detached: depth is a fixed ControlNet conditioning *input*, not a
            # gradient path. rgb/depth/acc are multiple outputs of the same
            # underlying 3DGRT tracer autograd node (one ctx for the whole
            # render call) — vsd_prior.step() calls .backward() internally on
            # its LoRA loss *before* the caller's own loss.backward() runs, so
            # leaving depth attached lets that inner call free the node's
            # saved tensors early, and the outer backward (via rgb_v) then
            # crashes with "Trying to backward through the graph a second
            # time." Geometry refinement (--vsd_freeze_geometry 0) still works
            # via the rgb_v path — depth just never needs to carry gradient.
            depth_cond = torch.stack(
                [make_depth_cond(depth_khw[k], acc_khw[k]).squeeze(0) for k in range(K)], dim=0
            ).detach()  # Kx3xRxR

            loss_vsd, vsd_metrics = vsd_prior.step(rgb_v, depth_cond)

            # Total-variation smoothness on the rendered RGB — targets the
            # high-frequency per-Gaussian-color noise directly, independent
            # of the view-to-view inconsistency batching already addresses.
            loss_smooth = torch.tensor(0.0, device=device)
            if args.vsd_smooth_weight > 0:
                loss_smooth = (
                    (rgb_v[:, :, 1:, :] - rgb_v[:, :, :-1, :]).abs().mean()
                    + (rgb_v[:, :, :, 1:] - rgb_v[:, :, :, :-1]).abs().mean()
                )

            # Anchor geometry back toward the Phase 1/2 ToF-lidar-grounded fit
            # — only meaningful when geometry isn't frozen; lets VSD nudge
            # positions/rotation/scale/density for texture-driven corrections
            # without letting them wander far from the physically-grounded
            # solution.
            loss_geom_reg = torch.tensor(0.0, device=device)
            if not args.vsd_freeze_geometry and args.vsd_geom_reg_weight > 0:
                loss_geom_reg = (
                    F.mse_loss(model.positions, vsd_positions_init)
                    + F.mse_loss(model.rotation, vsd_rotation_init)
                    + F.mse_loss(model.scale, vsd_scale_init)
                    + F.mse_loss(model.density, vsd_density_init)
                )

            loss_vsd = (loss_vsd + args.vsd_smooth_weight * loss_smooth
                        + args.vsd_geom_reg_weight * loss_geom_reg)

            if i % args.i_print == 0:
                albedo_drift = (model.features_albedo.detach() - vsd_albedo_init).abs().mean().item()
                specular_drift = (model.features_specular.detach() - vsd_specular_init).abs().mean().item()
                position_drift = (model.positions.detach() - vsd_positions_init).abs().mean().item()
                color_lr = vsd_color_param_groups[0]["lr"] if vsd_color_param_groups else 0.0
                tqdm.write(
                    f"[VSD] Iter: {i} | poses={list(pose_idxs)} | "
                    f"loss_vsd={vsd_metrics['loss_vsd']:.4f} | "
                    f"loss_lora={vsd_metrics.get('loss_lora', 0.0):.4f} | "
                    f"loss_smooth={loss_smooth.item():.4f} | loss_geom_reg={loss_geom_reg.item():.6f} | "
                    f"t_mean={vsd_metrics.get('t_mean', 0.0):.1f} | "
                    f"albedo_drift={albedo_drift:.5f} | specular_drift={specular_drift:.5f} | "
                    f"position_drift={position_drift:.5f} | color_lr={color_lr:.2e}"
                )

            if i % args.i_weights == 0:
                _save_vsd_preview(vsd_prior, I_A, rgb_v[:1], depth_cond[:1], img_savedir, i,
                                   num_steps=args.vsd_preview_steps)

        # ------------------------------------------------------------------
        # LPIPS grounding against chair_smooth_walls.png at its real camera
        # pose (dolly pose 30) — runs on its own cadence, independent of the
        # main VSD batch step above. This is a real photo, not a hallucinated
        # target, so it always gets full gradient access to color AND
        # geometry regardless of --vsd_freeze_geometry.
        # ------------------------------------------------------------------
        loss_lpips = torch.tensor(0.0, device=device)
        if vsd_lpips_fn is not None and i > args.N_iters:
            need_loss = (i - args.N_iters) % args.vsd_lpips_every_n_steps == 0
            need_preview = i % args.i_weights == 0
            if need_loss or need_preview:
                ref_rays_t = torch.transpose(vsd_ref_rays_gpu, 0, 1)   # [2, R*R, 3]
                _, _, _, _, _, extras_ref = render_rays_3dgrt(ref_rays_t, model, train=True, frame_id=i)
                R = args.vsd_render_res
                ref_pred = extras_ref["rgb"].reshape(R, R, 3).permute(2, 0, 1).unsqueeze(0)  # 1x3xRxR
                ref_pred = ref_pred.clamp(0, 1) * 2.0 - 1.0
                if need_preview:
                    _save_lpips_ref_preview(ref_pred, I_A, img_savedir, i)

            if need_loss:
                loss_lpips = vsd_lpips_fn(ref_pred, I_A).mean()

            if need_loss and i % args.i_print == 0:
                tqdm.write(f"[VSD] Iter: {i} | loss_lpips={loss_lpips.item():.4f}")

        # ------------------------------------------------------------------
        # Backward + optimiser step with GSStrategy densification hooks
        # ------------------------------------------------------------------
        model.optimizer.zero_grad()
        loss = (100 * (loss_1 + loss_2) + args.vsd_weight * loss_vsd
                + args.vsd_lpips_weight * loss_lpips)

        strategy.pre_backward(i, scene_extent, None, batch=sensor_batch)
        loss.backward()
        strategy.post_backward(i, scene_extent, None, batch=sensor_batch)

        model.optimizer.step()
        model.scheduler_step(i)

        # Cap Gaussian scale at 0.5m to prevent NaN over long runs.
        # Scale is log-space: scale_m = exp(scale_param), log(0.5) ≈ -0.693.
        with torch.no_grad():
            model.scale.clamp_(max=math.log(0.5))

        # --- CHANGE 2: THE SAFETY VALVE ---
        # If we hit 1.0M Gaussians, stop the growth to save your GPU
        if model.num_gaussians > 300000:
            strategy.conf.strategy.densify.end_iteration = i
            strategy.conf.strategy.reset_density.end_iteration = i
            print(f"!!! LIMIT REACHED: Disabling densification at {model.num_gaussians} !!!")

        scene_updated = strategy.post_optimizer_step(i, scene_extent, None, batch=sensor_batch)
        bvh_freq = conf.model.bvh_update_frequency
        if scene_updated or (bvh_freq > 0 and i % bvh_freq == 0):
            model.build_acc(rebuild=True)

        if model.progressive_training:
            if (i + 1) % model.feature_dim_increase_interval == 0:
                model.increase_num_active_features()

        dt = time.time() - time0

        # ------------------------------------------------------------------
        # Logging and checkpointing
        # ------------------------------------------------------------------
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = model.get_model_parameters()
            save_dict['global_step'] = global_step
            torch.save(save_dict, path)

            strategy_path = path.replace('.tar', '_strategy.tar')
            torch.save(strategy.get_strategy_parameters(), strategy_path)

            if vsd_prior is not None:
                vsd_path = path.replace('.tar', '_vsd.pt')
                torch.save(vsd_prior.state_dict(), vsd_path)

            print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} | Loss: {loss.item():.6f} | "
                f"N_gaussians: {model.num_gaussians} | "
                f"dt: {dt:.3f}s"
            )

        # ------------------------------------------------------------------
        # Periodic preview images
        # ------------------------------------------------------------------
        if i % args.i_weights == 0:
            with torch.no_grad():
                chunk_size = 4096
                all_depth, all_acc = [], []
                prev_rays = _preview_rays.to(device)
                for c in range(0, prev_rays.shape[0], chunk_size):
                    r_chunk = torch.transpose(prev_rays[c:c + chunk_size], 0, 1)  # [2, C, 3]
                    _, _, acc_c, depth_c, _, _ = render_rays_3dgrt(
                        r_chunk, model, train=False, frame_id=i
                    )
                    all_depth.append(depth_c.cpu())
                    all_acc.append(acc_c.cpu())

                depth_img = torch.cat(all_depth).reshape(H, W).numpy()
                acc_img   = torch.cat(all_acc).reshape(H, W).numpy()

                d_min, d_max = depth_img.min(), depth_img.max()
                if d_max > d_min:
                    depth_norm = ((depth_img - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_img, dtype=np.uint8)
                acc_norm = (np.clip(acc_img, 0, 1) * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(img_savedir, f'depth_{i:06d}.png'), depth_norm)
                cv2.imwrite(os.path.join(img_savedir, f'acc_{i:06d}.png'),   acc_norm)

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

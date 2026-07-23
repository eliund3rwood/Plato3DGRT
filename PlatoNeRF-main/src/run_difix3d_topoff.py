#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Difix3D+ "progressive 3D update", adapted for this project's single-real-
# photo setup. Ports the mechanism described in the Difix3D+ paper and
# implemented in NVIDIA's gsplat reference script
# (examples/gsplat/simple_trainer_difix3d.py): periodically render the
# CURRENT scene at novel poses, run each through Difix (single-step
# artifact-removal diffusion, see setup_difix.py) to get a cleaned image,
# and use freshly-generated targets as training supervision -- refreshed
# regularly throughout training rather than computed once upfront.
#
# This is a different, and hopefully better, mechanism than
# refine_from_target_images.py's one-shot topoff (tried twice: once against
# chair_v4_notex, once against a static DifFix3D-refined set -- neither
# fixed the iridescent artifact). The live-refresh design specifically
# addresses two suspected failure modes of the static approach: (1) targets
# going stale as the scene changes during training, since they were computed
# once from a fixed starting checkpoint and never updated; (2) asking for
# supervision at all poses simultaneously from iteration 1, rather than
# growing outward from a pose we have real grounding for.
#
# Adaptation from NVIDIA's dense-multi-view-capture setting to ours:
#   - Reference image: their scheme picks each novel pose's NEAREST REAL
#     training photo as Difix's per-pose appearance reference. We only have
#     ONE real photo (chair_smooth_walls.png, the ToF sensor's fixed
#     viewpoint) -- so every Difix call uses that same single image as
#     ref_image, unlike their nearest-neighbor selection.
#   - Curriculum: their scheme interpolates novel poses from many real
#     training poses toward held-out targets, expanding over successive
#     refreshes. We instead define "distance" as circular pose-index
#     distance around the 100-pose orbit bank from --anchor_pose_idx (the
#     orbit pose closest to the real reference photo's viewing angle,
#     visually confirmed as pose 0 by inspecting prior orbit renders -- see
#     VSD_HANDOFF.md), and widen the eligible pose window each refresh from
#     --curriculum_start_distance to --curriculum_end_distance (50 = the
#     full ring, since distance is symmetric).
#   - Mixing: each step samples the real-photo LPIPS grounding term (at
#     dolly pose 30, chair_smooth_walls.png's actual camera pose) with
#     probability --real_grounding_prob (default 0.7, matching the
#     70/30 real/pseudo split observed in NVIDIA's reference code), else a
#     pose from the CURRENT (most recent refresh's) pseudo-target batch,
#     regressed with L1 + LPIPS. Older batches are replaced, not
#     accumulated, matching the reference code's self.novelloaders[-1]-only
#     sampling (despite the paper text saying targets are "added to" the
#     training set).
#
# Only features_albedo trains by default (--freeze_specular); position/
# rotation/scale/density always frozen -- same reasoning as
# --vsd_freeze_specular / refine_from_target_images.py.
#
# Usage:
#   python src/run_difix3d_topoff.py --config configs/chair_vsd.txt \
#     --expname chair_vsd_run5_topoff \
#     --output_expname chair_vsd_run5_difix3d \
#     --n_iters 9000 --refresh_every 1000

import os
import sys

import cv2
import lpips as lpips_lib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import trange

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_DIFIX_ROOT = os.path.join(_REPO_ROOT, "custom_controlnet_difix")
if _DIFIX_ROOT not in sys.path:
    sys.path.insert(0, _DIFIX_ROOT)

from omegaconf import OmegaConf
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch

from utils.load_tof import load_tof_data
from utils.novel_views import orbit_poses, dolly_poses, rays_at_resolution
from setup_difix import get_difix_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 3DGRT config -- identical to refine_from_target_images.py's loader.
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


def tensor_to_pil(rgb_hw3_01: torch.Tensor) -> Image.Image:
    """[H,W,3] float in [0,1] -> PIL RGB image."""
    arr = (rgb_hw3_01.clamp(0, 1).detach().float().cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(img: Image.Image, res: int, device) -> torch.Tensor:
    """PIL RGB image -> 1x3xRxR float tensor in [-1, 1]."""
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def load_ref_image_pil(path: str, res: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (res, res):
        img = img.resize((res, res), Image.LANCZOS)
    return img


def circular_distance(i: int, a: int, n: int) -> int:
    d = abs(i - a) % n
    return min(d, n - d)


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
    parser.add_argument("--use_raw_weights", action='store_true')
    parser.add_argument("--vsd_checkpoint", type=str, default=None, help='unused, accepted for config compat')
    parser.add_argument("--vsd_n_orbit_poses", type=int, default=100, help='unused, accepted for config compat')
    parser.add_argument("--vsd_ref_image", type=str, default=None,
                        help='fixed I_A reference RGB; defaults to <datadir>/../chair_smooth_walls.png')

    parser.add_argument("--output_expname", type=str, required=True)
    parser.add_argument("--render_res", type=int, default=512)
    parser.add_argument("--n_iters", type=int, default=9000)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_pix", type=float, default=1.0)
    parser.add_argument("--lambda_real_lpips", type=float, default=10.0,
                        help='weight on the real-photo LPIPS grounding term (relative to the '
                             'pseudo-target L1+LPIPS terms, which are weighted 1.0)')
    parser.add_argument("--freeze_specular", type=int, default=1)

    parser.add_argument("--anchor_pose_idx", type=int, default=0,
                        help='orbit pose index closest to the real reference photo\'s viewing '
                             'angle -- the curriculum widens outward from here. Visually '
                             'confirmed as pose 0 by inspecting prior orbit renders.')
    parser.add_argument("--n_orbit_poses", type=int, default=100)
    parser.add_argument("--curriculum_start_distance", type=int, default=5)
    parser.add_argument("--curriculum_end_distance", type=int, default=50,
                        help='50 = half the 100-pose ring = full orbit coverage (distance is symmetric)')
    parser.add_argument("--refresh_every", type=int, default=1000,
                        help='iterations between Difix refresh cycles (paper default ~1500)')
    parser.add_argument("--refresh_batch_size", type=int, default=8,
                        help='novel poses rendered + Difix-fixed per refresh cycle')
    parser.add_argument("--real_grounding_prob", type=float, default=0.7,
                        help='probability each step uses the real-photo LPIPS grounding term '
                             'instead of the current pseudo-target batch (matches the 70/30 '
                             'real/pseudo split in NVIDIA\'s reference implementation)')
    parser.add_argument("--difix_num_inference_steps", type=int, default=1)
    parser.add_argument("--difix_timestep", type=int, default=199)

    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_weights", type=int, default=500)
    return parser


def main():
    parser = config_parser()
    args, _unknown = parser.parse_known_args()
    if _unknown:
        print(f"[INFO] Ignoring unrecognised config keys: {_unknown}")

    R = args.render_res
    N_ORBIT = args.n_orbit_poses

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
    print("[difix3d] Starting from checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    model.build_acc()

    if not args.use_raw_weights and 'ema_albedo' in ckpt:
        with torch.no_grad():
            model.features_albedo.copy_(ckpt['ema_albedo'].to(device))
            model.features_specular.copy_(ckpt['ema_specular'].to(device))
        print("[difix3d] Started from EMA color weights")

    for p in (model.positions, model.rotation, model.scale, model.density):
        p.requires_grad_(False)
    model.features_albedo.requires_grad_(True)
    model.features_specular.requires_grad_(not args.freeze_specular)

    conf.optimizer.params.features_albedo.lr = args.lr
    model.setup_optimizer()
    optimizer = model.optimizer

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Difix pipeline (see setup_difix.py for the compat patches this needs)
    # ------------------------------------------------------------------
    print("[difix3d] Loading Difix pipeline ...")
    difix_pipe = get_difix_pipeline(device="cuda")
    print("[difix3d] Difix pipeline ready.")

    # ------------------------------------------------------------------
    # Reference image: our ONE real photo, used for every Difix call (no
    # nearest-real-photo selection available, unlike NVIDIA's dense-capture
    # setting) AND for the real-photo LPIPS grounding term at dolly pose 30.
    # ------------------------------------------------------------------
    ref_path = args.vsd_ref_image or os.path.join(args.datadir, "..", "chair_smooth_walls.png")
    ref_pil = load_ref_image_pil(ref_path, R)
    I_A = pil_to_tensor(ref_pil, R, device)  # for LPIPS grounding, [-1,1] convention

    ref_pose_rays = torch.from_numpy(
        rays_at_resolution(dolly_poses(39)[30], H, W, focal, R, R)
    ).to(device)

    all_poses = orbit_poses(n=N_ORBIT)
    all_rays = [
        torch.from_numpy(rays_at_resolution(p, H, W, focal, R, R)).to(device)
        for p in all_poses
    ]

    out_dir = os.path.join(args.basedir, args.output_expname)
    img_savedir = os.path.join(out_dir, "progress")
    os.makedirs(img_savedir, exist_ok=True)

    n_refreshes = max(1, (args.n_iters + args.refresh_every - 1) // args.refresh_every)
    print(f"[difix3d] {n_refreshes} refresh cycles over {args.n_iters} iterations, "
          f"curriculum distance {args.curriculum_start_distance} -> {args.curriculum_end_distance}, "
          f"anchor pose {args.anchor_pose_idx}")

    trained_params = "features_albedo" if args.freeze_specular else "features_albedo + features_specular"
    print(f"[difix3d] Training {trained_params}.")

    active_pose_idxs = []
    active_targets = []  # list of 1x3xRxR tensors, [-1,1], aligned with active_pose_idxs

    def run_refresh(refresh_idx: int, iter_i: int) -> None:
        nonlocal active_pose_idxs, active_targets
        if n_refreshes > 1:
            frac = refresh_idx / (n_refreshes - 1)
        else:
            frac = 1.0
        max_dist = round(
            args.curriculum_start_distance
            + (args.curriculum_end_distance - args.curriculum_start_distance) * frac
        )
        eligible = [i for i in range(N_ORBIT)
                    if circular_distance(i, args.anchor_pose_idx, N_ORBIT) <= max_dist]
        batch_n = min(args.refresh_batch_size, len(eligible))
        chosen = list(np.random.choice(eligible, size=batch_n, replace=False))

        tqdm_desc = f"[difix3d] iter {iter_i} refresh {refresh_idx + 1}/{n_refreshes} (max_dist={max_dist}, {len(eligible)} eligible poses)"
        print(tqdm_desc + f" -> fixing poses {chosen}")

        new_targets = []
        first_render_pil, first_fixed_pil = None, None
        with torch.no_grad():
            for k, idx in enumerate(chosen):
                rays_t = torch.transpose(all_rays[idx], 0, 1)
                rgb = render_rays_3dgrt(rays_t, model, train=False)
                rgb_hw3 = rgb.reshape(R, R, 3).clamp(0, 1)
                render_pil = tensor_to_pil(rgb_hw3)
                fixed_pil = difix_pipe(
                    prompt="remove degradation", image=render_pil, ref_image=ref_pil,
                    num_inference_steps=args.difix_num_inference_steps,
                    timesteps=[args.difix_timestep], guidance_scale=0.0,
                ).images[0]
                new_targets.append(pil_to_tensor(fixed_pil, R, device))
                if k == 0:
                    first_render_pil, first_fixed_pil = render_pil, fixed_pil

        active_pose_idxs = chosen
        active_targets = new_targets

        # Save a quick before/after panel for the first pose of this refresh
        panel = np.concatenate([
            np.asarray(first_render_pil), np.asarray(first_fixed_pil.resize((R, R), Image.LANCZOS)),
        ], axis=1)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(img_savedir, f"refresh_{iter_i:06d}_pose{chosen[0]:03d}.png"), panel_bgr)

    refresh_iters = [1 + r * args.refresh_every for r in range(n_refreshes)]

    for i in trange(1, args.n_iters + 1):
        if i in refresh_iters:
            run_refresh(refresh_iters.index(i), i)

        use_real = (np.random.random() < args.real_grounding_prob)

        if use_real:
            rays_t = torch.transpose(ref_pose_rays, 0, 1)
            rgb = render_rays_3dgrt(rays_t, model, train=True)
            rgb = rgb.reshape(R, R, 3).permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0
            loss_lpips = lpips_fn(rgb, I_A).mean()
            loss = args.lambda_real_lpips * loss_lpips
            loss_pix_val, loss_lpips_val, pose_used = 0.0, loss_lpips.item(), "ref(30)"
        else:
            j = np.random.randint(0, len(active_pose_idxs))
            rays_t = torch.transpose(all_rays[active_pose_idxs[j]], 0, 1)
            rgb = render_rays_3dgrt(rays_t, model, train=True)
            rgb = rgb.reshape(R, R, 3).permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0
            target = active_targets[j]
            loss_pix = F.l1_loss(rgb, target)
            loss_lpips = lpips_fn(rgb, target).mean()
            loss = args.lambda_pix * loss_pix + args.lambda_lpips * loss_lpips
            loss_pix_val, loss_lpips_val, pose_used = loss_pix.item(), loss_lpips.item(), active_pose_idxs[j]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % args.i_print == 0:
            print(f"[difix3d] iter {i} | pose {pose_used} | real={use_real} | "
                  f"loss={loss.item():.4f} (pix={loss_pix_val:.4f}, lpips={loss_lpips_val:.4f})")

        if i % args.i_weights == 0 or i == args.n_iters:
            path = os.path.join(out_dir, f"{i:06d}.tar")
            save_dict = model.get_model_parameters()
            save_dict['global_step'] = ckpt.get('global_step', 0)
            torch.save(save_dict, path)
            print(f"[difix3d] Saved checkpoint -> {path}")

    print("[difix3d] Done.")


if __name__ == '__main__':
    main()

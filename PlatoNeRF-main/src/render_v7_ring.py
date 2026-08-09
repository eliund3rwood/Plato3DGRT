#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# render_v7_ring.py — render depth+pose for a ring of novel views from the
# PRISTINE (pre-VSD, textureless-color) 3DGRT reconstruction, in a format
# PlatoControlNet's V7 stage-1 checkpoint can consume directly.
#
# This is NOT the VSD Phase-3 training loop. It is a one-shot sanity check:
# does V7 (trained entirely on synthetic Blender chair rings) produce
# something reasonable when painting the REAL, ToF-reconstructed chair
# geometry? Depth+pose only, no diffusion here — V7 itself runs in a second
# step, in the PlatoControlNet repo, reading this script's .npz output. Kept
# as two steps rather than one cross-repo script so each side can be
# inspected/re-run independently and neither repo needs the other's Python
# dependencies installed.
#
# Reuses render_test_depth_3dgrt.py's OWN functions (create_3dgrt_conf,
# look_at, compute_points_around_circle, render_rays_3dgrt) rather than
# reimplementing them — a hand-copied second version could silently drift
# from whatever the validated render path actually does, exactly the class of
# bug scripts/test_platonerf_pose_convert.py exists to catch on the pose/depth
# conversion side.

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import render_test_depth_3dgrt as rtd  # noqa: E402
from utils.novel_views import dolly_poses  # noqa: E402

# PlatoControlNet's pose_convert module — no dependency on the rest of that
# repo (pure numpy/torch), so this cross-repo import is cheap and safe. Path
# matches STAGE2_HANDOFF.md / this project's established layout; override
# with --platocontrolnet_root if the clone lives elsewhere.
_DEFAULT_PCN_ROOT = r"C:\Eli Folder temp\PlatoControlNet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expname", default="chair_biggpu_v2",
                    help="the PRISTINE Phase-1/2 checkpoint, before any VSD "
                         "color training — geometry only, textureless")
    ap.add_argument("--basedir", default="./logs/")
    ap.add_argument("--ft_path", default="./logs/chair_biggpu_v2/035000.tar",
                    help="explicit checkpoint; the pristine one per "
                         "VSD_HANDOFF.md, not whatever is newest in expname/")
    ap.add_argument("--datadir", required=True,
                    help="ToF data dir (transforms_train.json etc.)")
    ap.add_argument("--N_iters", type=int, default=200000)
    ap.add_argument("--n-views", type=int, default=16,
                    help="target ring size; matches V7's own n_views=16 "
                         "training convention")
    ap.add_argument("--ring-radius", type=float, default=0.99)
    ap.add_argument("--near", type=float, default=0.1)
    ap.add_argument("--far", type=float, default=6.0,
                    help="must match create_3dgrt_conf's scene_extent below")
    ap.add_argument("--ref-photo",
                    default="./data/chair_smooth_walls.png",
                    help="the REAL reference photo, not a render")
    ap.add_argument("--out", default="./v7_ring_input.npz")
    ap.add_argument("--platocontrolnet_root", default=_DEFAULT_PCN_ROOT)
    ap.add_argument("--render_chunk", type=int, default=4096)
    args = ap.parse_args()

    if args.platocontrolnet_root not in sys.path:
        sys.path.insert(0, args.platocontrolnet_root)
    from src.models.pose_convert import platonerf_view_to_v7  # noqa: E402

    device = rtd.device

    # ── Load the ToF data (poses/intrinsics only — the Gaussian geometry is
    # what actually gets rendered) and the PRISTINE checkpoint ──────────────
    tof, poses, light_o, light_d, hwf, walls_cam, walls_light = rtd.load_tof_data(
        args.datadir, [])
    H, W, focal = hwf
    H, W = int(H), int(W)
    K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]],
                dtype=np.float32)

    conf = rtd.create_3dgrt_conf(args.N_iters)
    scene_extent = float(args.far)
    model = rtd.MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)

    ckpt = torch.load(args.ft_path, map_location=device, weights_only=False)
    assert "positions" in ckpt, (
        f"{args.ft_path} does not look like a 3DGRT geometry checkpoint "
        "('positions' key missing) — check --ft_path")
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    model.build_acc()
    model.eval()
    print(f"[render_v7_ring] loaded {model.num_gaussians} Gaussians from "
          f"{args.ft_path}")
    # Deliberately NOT checking for ema_albedo/ema_specular here (unlike
    # render_test_depth_3dgrt.py) — this script exists specifically to render
    # the PRISTINE, pre-VSD geometry. If this checkpoint has VSD color state,
    # that is a sign the wrong --ft_path was given.
    assert "ema_albedo" not in ckpt, (
        f"{args.ft_path} carries VSD Phase-3 EMA color state — this script "
        "wants the PRISTINE pre-VSD checkpoint (Phase 1/2 only). Point "
        "--ft_path at the seed checkpoint, e.g. logs/chair_biggpu_v2/035000.tar")

    # ── Build poses: reference = the REAL photo's actual camera; targets = a
    # fresh ring, independent of the 100-pose orbit bank VSD samples from ──
    ref_pose = dolly_poses(39)[30]   # confirmed (VSD_HANDOFF) to match chair_smooth_walls.png
    origin, y, lookat = (0, 3), -1.5, np.array([0, -1.5, -3])
    circle = rtd.compute_points_around_circle(
        origin, args.ring_radius, args.n_views, -np.pi / 2)
    tgt_poses = np.stack([
        rtd.look_at(np.array([-p[0], y, -p[1]]), lookat) for p in circle
    ], axis=0)

    def render_one(c2w_nerf: np.ndarray):
        rays_o, rays_d = rtd.get_rays_np(H, W, K, c2w_nerf[:3, :4])
        rays = np.stack([rays_o, rays_d], 0).transpose(1, 2, 0, 3).reshape(-1, 2, 3)
        norms = np.linalg.norm(rays[:, 1, :], axis=1, keepdims=True)
        rays[:, 1, :] /= norms
        rays_t = torch.from_numpy(rays.astype(np.float32)).to(device)
        depth_chunks, rgb_chunks = [], []
        for c in range(0, rays_t.shape[0], args.render_chunk):
            r = torch.transpose(rays_t[c:c + args.render_chunk], 0, 1)
            with torch.no_grad():
                d, _, rgb_c = rtd.render_rays_3dgrt(r, model)
            depth_chunks.append(d.cpu())
            rgb_chunks.append(rgb_c.cpu())
        depth_map = torch.cat(depth_chunks).reshape(H, W).numpy()
        rgb_map = torch.cat(rgb_chunks).reshape(H, W, 3).numpy()
        return depth_map, np.clip(rgb_map, 0, 1)

    print(f"[render_v7_ring] rendering 1 reference + {args.n_views} target views "
          f"at {H}x{W} ...")
    ref_dist, ref_rgb_pristine = render_one(ref_pose)
    tgt_dists, tgt_rgb_pristine = [], []
    for i, p in enumerate(tgt_poses):
        d, rgb = render_one(p)
        tgt_dists.append(d)
        tgt_rgb_pristine.append(rgb)
        print(f"  [render_v7_ring] target {i + 1}/{args.n_views} done")

    # ── Convert every view: ray-distance -> z-depth, NeRF c2w -> OpenCV w2c ─
    D_ref, w2c_ref, K_t = platonerf_view_to_v7(ref_dist, ref_pose, K)
    D_tgt, w2c_tgt = [], []
    for d, p in zip(tgt_dists, tgt_poses):
        Dm, w2c, _ = platonerf_view_to_v7(d, p, K)
        D_tgt.append(Dm)
        w2c_tgt.append(w2c)

    np.savez(
        args.out,
        D_ref_metric=D_ref.numpy(),
        w2c_ref=w2c_ref.numpy(),
        K_ref=K_t.numpy(),
        D_tgt_metric=np.stack([d.numpy() for d in D_tgt]),
        w2c_tgt=np.stack([w.numpy() for w in w2c_tgt]),
        K_tgt=np.tile(K_t.numpy()[None], (args.n_views, 1, 1)),
        near=args.near, far=args.far,
        # Pristine (untextured-VSD, whatever the Phase-1/2 color init gives)
        # renders too, purely for visual comparison against V7's prediction —
        # NOT fed into V7 itself.
        ref_rgb_pristine=ref_rgb_pristine,
        tgt_rgb_pristine=np.stack(tgt_rgb_pristine),
        ref_photo_path=os.path.abspath(args.ref_photo),
    )
    print(f"[render_v7_ring] wrote {args.out}")


if __name__ == "__main__":
    main()

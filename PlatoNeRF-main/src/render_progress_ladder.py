#!/usr/bin/env python
"""
src/render_progress_ladder.py — render the SAME fixed views from a LADDER of
checkpoints, so a VSD run can be read as a trajectory instead of an endpoint.

WHY
---
The synthetic control was being compared at ~20-25% progress against the real
chair's appearance at iteration 53000, which is 90%. If iridescence EMERGES
late, "the synthetic scenes look smooth" would just mean "they are early", and
a stage difference would be misread as a geometry difference. That is the same
mistake as reading the random-pose preview panels as a time series, and it
would invalidate the whole control.

The comparison is only fair at matched progress. Conveniently the arithmetic
lines up: chair_vsd_v7_run1 and both synthetic runs all use N_iters=35000 and
vsd_iters=20000, so progress = (i-35000)/20000 is the same function in all
three, and the SAME ABSOLUTE ITERATION means the same point on the timestep
anneal and the colour-LR decay. So the ladder is just a list of iteration
numbers, used identically for every run.

Both scene types are handled here rather than in two scripts, because the only
thing that differs is how poses and rays are built:

  --scene_mode synthetic : poses from the scene's own cameras.json ring
                           (already OpenCV w2c; NO NeRF->OpenCV flip)
  --scene_mode chair     : utils.novel_views.orbit_poses, the real chair's
                           hardcoded NeRF/OpenGL c2w bank, with
                           rays_at_resolution — the same poses the chair's own
                           eval flyaround uses

Chair mode deliberately reads camera_angle_x from transforms_train.json rather
than calling load_tof_data: that loader pulls the full (16,512,512,391) float32
ToF array (~6.5 GB, and several working copies) purely to arrive at H, W and
focal, which makes an eval-only job need a 64 GB allocation for nothing.

EMA colour weights are substituted wherever a checkpoint has them, as in
render_test_depth_3dgrt.py — the raw trajectory systematically overstates how
noisy a result is, and that bias would fall hardest on exactly the late
checkpoints this script exists to examine.

Usage:
    # the real chair, across its run
    python src/render_progress_ladder.py --scene_mode chair \
        --expname chair_vsd_v7_run1 --datadir ./data/chair \
        --iters 39000,43000,47000,51000,53000 --out_dir ladder_chair

    # the synthetic control, at the SAME iterations
    python src/render_progress_ladder.py --scene_mode synthetic \
        --expname geom_train_vsd --cameras $GT/train_chair_<hash>_cameras.json \
        --iters 39000,43000,47000,51000,55000 --out_dir ladder_train
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import imageio.v2 as imageio  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from threedgrut.model.model import MixtureOfGaussians                     # noqa: E402
from run_platonerf_3dgrt_vsd import create_3dgrt_conf, render_rays_3dgrt  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pose_rays(args):
    """Return (list of [N,2,3] ray arrays, label per view). Everything that
    differs between the two scene types is contained here."""
    R = args.res
    if args.scene_mode == "synthetic":
        from utils.synthetic_scene import (
            load_scene_cameras, scale_intrinsics, rays_from_w2c, fit_ring)
        cams = load_scene_cameras(args.cameras)
        ring = fit_ring(cams)
        K = scale_intrinsics(cams.K, cams.width, cams.height, R).astype(np.float64)
        orbit = ring.orbit(args.n_orbit)
        idxs = np.linspace(0, args.n_orbit - 1, args.n_views).astype(int)
        return ([rays_from_w2c(orbit[i].astype(np.float64), K, R) for i in idxs],
                [f"orbit {i}" for i in idxs])

    # chair: the real scene's own novel-view bank, in NeRF/OpenGL c2w
    from utils.novel_views import orbit_poses, rays_at_resolution
    meta_path = os.path.join(args.datadir, "transforms_train.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    # camera_angle_x is stored in DEGREES here (load_tof.py applies np.radians)
    cam_angle_x = np.radians(float(meta["camera_angle_x"]))
    H = W = args.chair_hw
    focal = 0.5 * W / np.tan(0.5 * cam_angle_x)
    print(f"[ladder] chair intrinsics from {os.path.basename(meta_path)}: "
          f"H=W={H}, focal={focal:.3f} (no ToF array loaded)")
    orbit = orbit_poses(n=args.n_orbit)
    idxs = np.linspace(0, args.n_orbit - 1, args.n_views).astype(int)
    return ([rays_at_resolution(orbit[i], H, W, focal, R, R) for i in idxs],
            [f"orbit {i}" for i in idxs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_mode", required=True, choices=["synthetic", "chair"])
    ap.add_argument("--expname", required=True)
    ap.add_argument("--basedir", default="./logs/")
    ap.add_argument("--cameras", default=None, help="synthetic mode only")
    ap.add_argument("--datadir", default="./data/chair", help="chair mode only")
    ap.add_argument("--iters", required=True,
                    help="comma-separated checkpoint iterations, e.g. "
                         "39000,43000,47000,51000,55000. The SAME list should be "
                         "used for every run being compared — equal iteration "
                         "means equal progress when N_iters and vsd_iters match.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--n_views", type=int, default=5, help="views per checkpoint")
    ap.add_argument("--n_orbit", type=int, default=100,
                    help="size of the orbit bank the views are sampled from; 100 "
                         "matches the chair's --vsd_n_orbit_poses default")
    ap.add_argument("--chair_hw", type=int, default=512)
    ap.add_argument("--use_raw_weights", action="store_true")
    ap.add_argument("--render_chunk", type=int, default=65536)
    ap.add_argument("--N_iters", type=int, default=35000)
    ap.add_argument("--vsd_iters", type=int, default=20000,
                    help="only used to label each row with its progress fraction")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="if >0, also save a centre crop of this fraction of the "
                         "frame — iridescent faceting is a high-frequency symptom "
                         "and can be invisible at thumbnail scale")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.scene_mode == "synthetic" and not args.cameras:
        sys.exit("--cameras is required in synthetic mode")

    iters = [int(x) for x in args.iters.split(",")]
    savedir = os.path.join(args.basedir, args.expname)
    rays_list, labels = build_pose_rays(args)
    R = args.res
    conf = create_3dgrt_conf(args)

    rows, row_labels = [], []
    for it in iters:
        path = os.path.join(savedir, f"{it:06d}.tar")
        if not os.path.isfile(path):
            print(f"[ladder] MISSING {path} — skipping")
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        scene_extent = float(ckpt.get("scene_extent") or 1.4)
        model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
        model.init_from_checkpoint(ckpt, setup_optimizer=False)
        used_ema = False
        if "ema_albedo" in ckpt and not args.use_raw_weights:
            with torch.no_grad():
                model.features_albedo.copy_(ckpt["ema_albedo"].to(device))
                model.features_specular.copy_(ckpt["ema_specular"].to(device))
            used_ema = True
        model.build_acc()
        model.eval()

        views = []
        for rays_np in rays_list:
            rays = torch.from_numpy(rays_np).to(device)
            chunks = []
            for c in range(0, rays.shape[0], args.render_chunk):
                r = torch.transpose(rays[c:c + args.render_chunk], 0, 1)
                with torch.no_grad():
                    _, _, _, _, _, extras = render_rays_3dgrt(
                        r, model, train=False, frame_id=0)
                chunks.append(extras["rgb"].float().cpu())
            views.append(np.clip(torch.cat(chunks).reshape(R, R, 3).numpy(), 0, 1))
        rows.append(views)
        prog = (it - args.N_iters) / max(args.vsd_iters, 1)
        row_labels.append(f"{it} ({prog * 100:.0f}%){'' if used_ema else ' RAW'}")
        print(f"[ladder] {it}  progress {prog * 100:5.1f}%  "
              f"{'EMA' if used_ema else 'raw'}  {model.num_gaussians:,} Gaussians")
        del model
        torch.cuda.empty_cache()

    if not rows:
        sys.exit("no checkpoints rendered — check --iters against what exists")

    def grid(all_rows, suffix, crop=0.0):
        nr, nc = len(all_rows), len(all_rows[0])
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.75 * nr), squeeze=False)
        for r in range(nr):
            for c in range(nc):
                img = all_rows[r][c]
                if crop > 0:
                    h = int(R * crop / 2)
                    img = img[R // 2 - h:R // 2 + h, R // 2 - h:R // 2 + h]
                axes[r][c].imshow(img)
                axes[r][c].axis("off")
                if r == 0:
                    axes[r][c].set_title(labels[c], fontsize=8)
            axes[r][0].text(-0.08, 0.5, row_labels[r], rotation=90,
                            va="center", ha="center", fontsize=9,
                            transform=axes[r][0].transAxes)
        fig.suptitle(f"{args.expname} — progress ladder "
                     f"({'EMA' if not args.use_raw_weights else 'raw'} weights)"
                     + (f", centre {crop:.0%} crop" if crop > 0 else ""),
                     fontsize=12)
        fig.tight_layout()
        p = os.path.join(args.out_dir, f"ladder{suffix}.png")
        fig.savefig(p, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print(f"[ladder] wrote {p}")

    grid(rows, "")
    if args.zoom > 0:
        grid(rows, "_zoom", crop=args.zoom)

    for r, it in enumerate([int(x) for x in args.iters.split(",")][:len(rows)]):
        for c, img in enumerate(rows[r]):
            imageio.imwrite(os.path.join(args.out_dir, f"{it:06d}_v{c}.png"),
                            (img * 255).astype(np.uint8))


if __name__ == "__main__":
    main()

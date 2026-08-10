#!/usr/bin/env python
"""
scripts/test_04_depth_vs_gt.py — diagnostic for test_03's cross-view depth
disagreement. Decides whether the converted geometry is WRONG or merely SOFT.

WHY THIS EXISTS
---------------
test_03 reported a median relative cross-view depth error of 4.4% (train) /
6.8% (val), with only 31% / 18% of reprojected points agreeing within 2%. That
is either a real convention bug or a benign property of what 3DGRT's
`pred_dist` actually is. Reasoning from plausible hypotheses is how several
bugs in this project got misdiagnosed, so this measures instead.

Four candidate explanations, each with a measurement that separates it:

  A. A CONVENTION BUG (rays, poses, or the ray-distance/z-depth conversion).
     -> Test 1 compares rendered depth against BLENDER GROUND TRUTH for the
        same 24 poses. This is external, single-view, and immune to occlusion
        and baseline. A convention bug cannot survive it; nothing else here
        can produce agreement with GT.

  B. SOFT DEPTH. 3DGRT returns an accumulation-weighted distance, and these
     Gaussians have MEDIAN OPACITY 0.099 — very transparent. Where many faint
     Gaussians overlap, "depth" is an expectation over a thick slab rather
     than a surface, and it slides as the camera moves.
     -> Test 3 stratifies cross-view agreement by accumulated alpha. If
        agreement climbs sharply with alpha, depth is soft.

  C. OCCLUSION AND GRAZING GEOMETRY. Adjacent ring views are 15 deg apart,
     which is a wide baseline. Points on a wall at grazing incidence move many
     pixels, and test_03 looked them up with nearest-neighbour rounding.
     -> Test 4 sweeps the baseline (0, 1, 2, 4, 8 steps). A convention bug is
        baseline-INDEPENDENT and large; occlusion/grazing grows smoothly with
        baseline and is ~0 at baseline 0.

  D. THE GT DEPTH'S OWN SEMANTICS. Blender's Z pass is z-depth in some
     configurations and ray distance in others, and PlatoControlNet's
     `unproject_grid` consumes it as z-depth.
     -> Test 1 scores BOTH interpretations and reports which fits, rather than
        assuming the one the docstrings imply.

Test 2 additionally checks that the h5's per-frame w2c matches the geom_test
cameras.json, i.e. that the ground truth is attached to the poses we rendered.

Requires a GPU node.

Usage:
    python scripts/test_04_depth_vs_gt.py \
        --ckpt logs/geom_train/035000.tar \
        --cameras $GT/train_chair_<hash>_cameras.json \
        --h5 /home/tzofi/orcd/scratch/eli/platocontrolnet/data/rings_1000/train.h5 \
        --scene 32f918efaa64a4d9c423490470c47d79 \
        --out_dir geom_check_train
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from threedgrut.model.model import MixtureOfGaussians                     # noqa: E402
from run_platonerf_3dgrt_vsd import create_3dgrt_conf, render_rays_3dgrt  # noqa: E402
from utils.synthetic_scene import load_scene_cameras, scale_intrinsics, rays_from_w2c  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ray_dist_to_z(dist_hw, K):
    H, W = dist_hw.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    j, i = np.meshgrid(np.arange(H, dtype=np.float64),
                       np.arange(W, dtype=np.float64), indexing="ij")
    mag = np.sqrt(((i - cx) / fx) ** 2 + ((j - cy) / fy) ** 2 + 1.0)
    return dist_hw / mag


def render_view(model, w2c, K, res, chunk, frame_id=0):
    rays = torch.from_numpy(rays_from_w2c(w2c, K, res)).to(device)
    ds, accs = [], []
    for c in range(0, rays.shape[0], chunk):
        r = torch.transpose(rays[c:c + chunk], 0, 1)
        with torch.no_grad():
            _, _, acc, dist, _, _ = render_rays_3dgrt(r, model, train=False, frame_id=frame_id)
        ds.append(dist.float().cpu())
        accs.append(acc.float().cpu())
    return (torch.cat(ds).reshape(res, res).numpy().astype(np.float64),
            torch.cat(accs).reshape(res, res).numpy().astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--scene", required=True, help="scene id or a unique substring")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--render_chunk", type=int, default=65536)
    ap.add_argument("--N_iters", type=int, default=35000)
    ap.add_argument("--alpha_thresh", type=float, default=0.5)
    ap.add_argument("--chair_radius", type=float, default=0.6,
                    help="world-space cylinder radius about the scene's vertical axis "
                         "used to separate CHAIR pixels from ROOM pixels")
    ap.add_argument("--chair_height", type=float, default=1.3,
                    help="world-space cylinder height for the same split")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cams = load_scene_cameras(args.cameras)
    K = scale_intrinsics(cams.K, cams.width, cams.height, args.res).astype(np.float64)

    import h5py
    with h5py.File(args.h5, "r") as f:
        matches = [k for k in f.keys() if args.scene in k]
        if len(matches) != 1:
            sys.exit(f"--scene '{args.scene}' matched {len(matches)}: {matches[:8]}")
        sid = matches[0]
        g = f[sid]
        gt_depth = np.asarray(g["depth"], dtype=np.float64)      # (n,512,512)
        gt_w2c = np.asarray(g["w2c"], dtype=np.float64)
        gt_K = np.asarray(g["K"], dtype=np.float64)
        gt_alpha = np.asarray(g["alpha"], dtype=np.float64) if "alpha" in g else None
        frame_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in g.attrs["frame_ids"]]
    n = len(gt_depth)
    print(f"[gt] scene key = {sid}, {n} views")
    print(f"[gt] depth range [{gt_depth.min():.4f}, {gt_depth.max():.4f}]")

    # ---- Test 2: is the GT attached to the poses we are rendering? ---------
    print("\n[test 2] h5 poses vs geom_test cameras.json")
    if len(gt_w2c) != len(cams):
        print(f"  [WARN] view count differs: h5 {len(gt_w2c)} vs cameras.json {len(cams)}")
    m = min(len(gt_w2c), len(cams))
    dw = np.abs(gt_w2c[:m] - cams.w2c[:m].astype(np.float64)).max()
    dk = np.abs(gt_K[0] - cams.K.astype(np.float64)).max()
    print(f"  max |w2c difference| = {dw:.3e}   max |K difference| = {dk:.3e}")
    print(f"  h5 frame_ids[:4] = {frame_ids[:4]}")
    print(f"  json frame_ids[:4] = {cams.frame_ids[:4]}")
    aligned = dw < 1e-4 and dk < 1e-3
    print(f"  -> {'ALIGNED' if aligned else 'MISALIGNED — the GT below is not for these poses'}")

    # ---- load model --------------------------------------------------------
    class _A:
        pass
    a = _A()
    a.N_iters = args.N_iters
    conf = create_3dgrt_conf(a)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    scene_extent = float(ckpt.get("scene_extent") or 1.4)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    model.build_acc()
    model.eval()
    print(f"\n[model] {model.num_gaussians:,} Gaussians")

    dists, accs = [], []
    for k in range(len(cams)):
        d, ac = render_view(model, cams.w2c[k].astype(np.float64), K,
                            args.res, args.render_chunk, frame_id=k)
        dists.append(d)
        accs.append(ac)
    print(f"[model] rendered {len(dists)} views")

    # ---- chair / room segmentation, in WORLD space -------------------------
    # The panels show the CHAIR is pin-sharp while the walls and floor are a
    # rolling cloud. That is not a conversion fault: gsplat fitted these scenes
    # against rgb_512_gsplat_tmp, in which the floor and walls are flattened to
    # a solid colour (run_gsplat_cluster.py:31-37), deliberately, so that
    # reconstructed depth carries no wall pattern for the prior to leak from.
    # A photometric loss on a uniform wall has ZERO depth gradient, so wall
    # geometry is unconstrained by construction and converged to mush.
    #
    # So every number below is reported per REGION. A single scene-wide median
    # would average clean geometry together with geometry that was never
    # constrained, and report neither.
    def chair_mask_from_gt(gt_z, w2c):
        """Pixels whose GT surface point lies inside a cylinder about the
        scene's vertical axis. Segmenting in world space rather than by depth
        threshold keeps the floor (near, but not the chair) out of the chair
        region."""
        H, W = gt_z.shape
        jj, ii = np.meshgrid(np.arange(H, dtype=np.float64),
                             np.arange(W, dtype=np.float64), indexing="ij")
        pix = np.stack([ii, jj, np.ones_like(ii)], axis=-1)
        cam = np.einsum("ij,hwj->hwi", np.linalg.inv(K), pix) * gt_z[..., None]
        c2w = np.linalg.inv(w2c)
        wp = np.einsum("ij,hwj->hwi", c2w[:3, :3], cam) + c2w[:3, 3]
        # the ring's up axis is the one the cameras do not move along
        ax = int(np.argmin(cams.centers.std(axis=0)))
        pl = [q for q in range(3) if q != ax]
        rad = np.sqrt((wp[..., pl[0]] - 0.0) ** 2 + (wp[..., pl[1]] - 0.0) ** 2)
        return (rad < args.chair_radius) & (wp[..., ax] < args.chair_height) & (gt_z > 1e-4)

    # ---- Test 1: rendered depth vs Blender GT, BOTH interpretations --------
    print("\n[test 1] rendered depth vs Blender GT depth (the decisive test)")
    print("  Interpreting the h5's depth as ...")
    res_z, res_r = [], []
    res_chair, res_room, chair_frac = [], [], []
    for k in range(min(len(dists), n)):
        z_rendered = ray_dist_to_z(dists[k], K)
        gt = gt_depth[k]
        valid = (gt > 1e-4) & (accs[k] > args.alpha_thresh)
        if valid.sum() == 0:
            continue
        # (a) h5 depth is already z-depth  -> compare directly
        res_z.append(np.median(np.abs(z_rendered[valid] - gt[valid]) / gt[valid]))
        # (b) h5 depth is euclidean ray distance -> convert it first
        gt_as_z = ray_dist_to_z(gt, K)
        res_r.append(np.median(np.abs(z_rendered[valid] - gt_as_z[valid]) / gt_as_z[valid]))

        cm = chair_mask_from_gt(gt, cams.w2c[k].astype(np.float64))
        rel = np.abs(z_rendered - gt) / np.maximum(gt, 1e-6)
        chair_frac.append(cm.mean())
        if (cm & valid).sum() > 100:
            res_chair.append(np.median(rel[cm & valid]))
        if ((~cm) & valid).sum() > 100:
            res_room.append(np.median(rel[(~cm) & valid]))
    res_z, res_r = np.array(res_z), np.array(res_r)
    print(f"    Z-DEPTH        : median relative error {np.median(res_z):.5f} "
          f"(per-view min {res_z.min():.5f}, max {res_z.max():.5f})")
    print(f"    RAY DISTANCE   : median relative error {np.median(res_r):.5f} "
          f"(per-view min {res_r.min():.5f}, max {res_r.max():.5f})")
    better = "Z-DEPTH" if np.median(res_z) < np.median(res_r) else "RAY DISTANCE"
    print(f"  -> the h5's depth behaves as {better}")
    best = min(np.median(res_z), np.median(res_r))

    # THE number this whole experiment turns on.
    res_chair, res_room = np.array(res_chair), np.array(res_room)
    print(f"\n  BY REGION (chair = world cylinder r<{args.chair_radius} h<{args.chair_height}; "
          f"{np.mean(chair_frac) * 100:.1f}% of pixels):")
    print(f"    CHAIR      : median relative error vs GT = {np.median(res_chair):.5f} "
          f"(per-view max {res_chair.max():.5f})")
    print(f"    ROOM       : median relative error vs GT = {np.median(res_room):.5f} "
          f"(per-view max {res_room.max():.5f})")
    ratio = np.median(res_room) / max(np.median(res_chair), 1e-9)
    print(f"    room/chair error ratio = {ratio:.2f}x")

    # The chair is the region that was actually constrained by the fit, so it is
    # the one that licenses a "the conversion is correct" verdict. The room's
    # error is a property of the FIT, not of this conversion.
    verdict_gt = np.median(res_chair) < 0.05
    print(f"  -> chair geometry {'AGREES' if verdict_gt else 'DISAGREES'} with Blender GT "
          f"(median {np.median(res_chair):.5f}); a convention bug could not produce this")
    if ratio > 3.0:
        print(f"  -> the room is {ratio:.1f}x worse than the chair, as expected from a fit "
              f"against flattened floor/wall colour. SCORE VSD PER REGION, not scene-wide.")

    # ---- Test 3: cross-view agreement stratified by alpha ------------------
    print("\n[test 3] cross-view agreement vs accumulated alpha "
          "(tests the soft-depth explanation)")
    K_inv = np.linalg.inv(K)

    def reproject_agreement(ka, kb, alpha_lo=0.0, alpha_hi=1.01, tol=0.02, n_samp=40000):
        za = ray_dist_to_z(dists[ka], K)
        sel_mask = (accs[ka] >= alpha_lo) & (accs[ka] < alpha_hi) & (za > 1e-4)
        jj, ii = np.nonzero(sel_mask)
        if len(ii) < 100:
            return np.nan, 0
        pick = np.random.default_rng(ka).choice(len(ii), size=min(n_samp, len(ii)), replace=False)
        ii, jj = ii[pick], jj[pick]
        pix = np.stack([ii, jj, np.ones_like(ii)], axis=1).astype(np.float64)
        cam_pts = (K_inv @ pix.T).T * za[jj, ii][:, None]
        c2w_a = np.linalg.inv(cams.w2c[ka].astype(np.float64))
        world = cam_pts @ c2w_a[:3, :3].T + c2w_a[:3, 3]
        w2c_b = cams.w2c[kb].astype(np.float64)
        Xb = world @ w2c_b[:3, :3].T + w2c_b[:3, 3]
        zb = Xb[:, 2]
        ub = K[0, 0] * Xb[:, 0] / zb + K[0, 2]
        vb = K[1, 1] * Xb[:, 1] / zb + K[1, 2]
        ok = (zb > 1e-6) & (ub >= 0) & (ub < args.res - 1) & (vb >= 0) & (vb < args.res - 1)
        if ok.sum() < 50:
            return np.nan, 0
        ui, vi = np.round(ub[ok]).astype(int), np.round(vb[ok]).astype(int)
        zb_obs = ray_dist_to_z(dists[kb], K)[vi, ui]
        good = accs[kb][vi, ui] > args.alpha_thresh
        if good.sum() < 50:
            return np.nan, 0
        rel = np.abs(zb_obs[good] - zb[ok][good]) / np.maximum(zb_obs[good], 1e-6)
        return float((rel < tol).mean()), int(good.sum())

    bands = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
    for lo, hi in bands:
        vals = [reproject_agreement(k, (k + 1) % len(cams), lo, hi)[0]
                for k in range(0, len(cams), 3)]
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals):
            print(f"  alpha in [{lo:.2f},{hi:.2f}): agreement within 2% = {vals.mean():.4f}")
        else:
            print(f"  alpha in [{lo:.2f},{hi:.2f}): too few samples")

    # ---- Test 4: baseline sweep -------------------------------------------
    print("\n[test 4] agreement vs angular baseline "
          "(a convention bug is baseline-independent; occlusion/grazing is not)")
    for step in (0, 1, 2, 4, 8):
        vals = [reproject_agreement(k, (k + step) % len(cams))[0]
                for k in range(0, len(cams), 3)]
        vals = np.array([v for v in vals if not np.isnan(v)])
        deg = step * 360.0 / len(cams)
        print(f"  baseline {step} views ({deg:5.1f} deg): agreement = "
              f"{vals.mean():.4f}" if len(vals) else f"  baseline {step}: n/a")

    # ---- panel -------------------------------------------------------------
    idxs = np.linspace(0, min(len(dists), n) - 1, 4).astype(int)
    fig, axes = plt.subplots(4, len(idxs), figsize=(3.1 * len(idxs), 12.4))
    for col, k in enumerate(idxs):
        z_rendered = ray_dist_to_z(dists[k], K)
        gt = gt_depth[k]
        valid = (gt > 1e-4) & (accs[k] > args.alpha_thresh)
        vmin = np.percentile(gt[gt > 1e-4], 2)
        vmax = np.percentile(gt[gt > 1e-4], 98)
        axes[0, col].imshow(z_rendered, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{frame_ids[k]}\nrendered z-depth", fontsize=8)
        axes[1, col].imshow(gt, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1, col].set_title("Blender GT depth", fontsize=8)
        err = np.where(valid, np.abs(z_rendered - gt) / np.maximum(gt, 1e-6), np.nan)
        im = axes[2, col].imshow(err, cmap="magma", vmin=0, vmax=0.15)
        axes[2, col].set_title(f"|rel error| (med {np.nanmedian(err):.4f})", fontsize=8)
        plt.colorbar(im, ax=axes[2, col], fraction=0.046)
        # The region split is what every conclusion below rests on, so draw it
        # rather than trusting the cylinder parameters. (Accumulated alpha used
        # to live in this row and was uninformative — it saturates at 1.0
        # everywhere inside a closed room.)
        cm = chair_mask_from_gt(gt, cams.w2c[k].astype(np.float64))
        axes[3, col].imshow(np.where(cm, 2.0, 1.0) * np.where(valid, 1.0, 0.3),
                            cmap="viridis", vmin=0, vmax=2)
        axes[3, col].set_title(f"chair mask ({cm.mean() * 100:.1f}% of pixels)", fontsize=8)
        for r in range(4):
            axes[r, col].axis("off")
    fig.suptitle(f"{os.path.basename(args.ckpt)} vs Blender GT — {sid}", fontsize=11)
    fig.tight_layout()
    p = os.path.join(args.out_dir, "depth_vs_gt_panel.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[gt] wrote {p}")

    print("\n" + "=" * 70)
    if verdict_gt and aligned:
        print("VERDICT: the CHAIR matches Blender ground truth for these poses, so the\n"
              "PLY->3DGRT conversion and the camera plumbing are correct. test_03's\n"
              "scene-wide cross-view number was averaging that clean geometry together\n"
              "with room geometry the fit never constrained (floor/wall were flattened\n"
              "to solid colour before fitting, so the photometric loss had no depth\n"
              "gradient there). Score the VSD control PER REGION: the chair region is\n"
              "the clean-geometry arm and the room is the mush arm, in one scene, under\n"
              "one prior, in one run.")
    elif not aligned:
        print("VERDICT: the h5 poses do NOT match cameras.json, so test 1 compared\n"
              "against the wrong frames. Fix the alignment before concluding anything.")
    else:
        print("VERDICT: rendered depth does NOT match Blender GT. This is a real bug,\n"
              "not softness. Look at the error map in the panel — a uniform offset means\n"
              "a depth-semantics error, a left-right or up-down gradient means a pose\n"
              "convention error.")


if __name__ == "__main__":
    main()

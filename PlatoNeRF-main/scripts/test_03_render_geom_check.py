#!/usr/bin/env python
"""
scripts/test_03_render_geom_check.py — render the CONVERTED synthetic-scene
checkpoint and prove the geometry survived the PLY -> 3DGRT conversion, BEFORE
committing a GPU-day to VSD on it.

`ply_to_3dgrt_ckpt.py --verify` already proves the checkpoint's NUMBERS match
the PLY (covariance round-trip). That says nothing about whether 3DGRT, driven
by rays this repo builds from `cameras.json`, actually draws the scene in the
right place — the parameters could be perfect and the camera plumbing still
wrong. This script closes that gap by rendering and measuring.

Requires a GPU node (3DGRT tracer).

WHAT IT MEASURES

  1. COVERAGE — fraction of pixels with accumulated opacity above a threshold,
     per view. This is the question the handoff flags as load-bearing: "the
     room's far walls may be thinly covered even in the synthetic fit, and
     walls are exactly where the chair fails". If the synthetic fit ALSO leaves
     walls thin, then a bad VSD result on walls means nothing, and the control
     cannot distinguish coverage from VSD quality. Reported per view and split
     into the chair region vs the rest of the frame.

  2. CROSS-VIEW DEPTH CONSISTENCY — the decisive end-to-end geometric test.
     Unproject view A's rendered z-depth into world space using w2c_A/K, then
     project those world points into view B and compare against view B's own
     rendered depth on co-visible pixels. This exercises the ENTIRE chain
     (ray construction -> tracer -> ray-distance-to-z-depth -> unprojection ->
     reprojection) and only agrees if every link uses the same convention. A
     flipped axis, a transposed rotation, or a ray-distance/z-depth mix-up all
     break it while leaving each individual render looking perfectly plausible.

  3. DEPTH PLAUSIBILITY — rendered depth against the ring geometry that is
     already known from the cameras (radius, look-at target, room bounding box
     from the PLY). Catches a scene rendered at the wrong scale.

  4. A PANEL TO LOOK AT — depth, alpha, and RGB for a spread of ring views.
     This project's own rule: metrics here have twice rewarded visibly worse
     output, so the panel is the deliverable, not the numbers.

Usage (cluster, GPU node):
    python scripts/test_03_render_geom_check.py \
        --ckpt logs/geom_train/035000.tar \
        --cameras <geom_test>/train_chair_<hash>_cameras.json \
        --ply <geom_test>/train_chair_<hash>.ply \
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

from threedgrut.model.model import MixtureOfGaussians          # noqa: E402
from run_platonerf_3dgrt_vsd import create_3dgrt_conf, render_rays_3dgrt  # noqa: E402
from utils.synthetic_scene import (                            # noqa: E402
    load_scene_cameras, scale_intrinsics, rays_from_w2c, fit_ring,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Gate:
    def __init__(self):
        self.failures = []

    def check(self, ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)


def ray_dist_to_z(dist_hw, K):
    """Euclidean ray distance -> z-depth. Same formula as PlatoControlNet's
    pose_convert.ray_distance_to_z_depth, restated here only so this gate does
    not require that repo to be present; test_02 checks the two agree."""
    H, W = dist_hw.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    j, i = np.meshgrid(np.arange(H, dtype=np.float64),
                       np.arange(W, dtype=np.float64), indexing="ij")
    mag = np.sqrt(((i - cx) / fx) ** 2 + ((j - cy) / fy) ** 2 + 1.0)
    return dist_hw / mag


def render_view(model, w2c, K, res, chunk, frame_id=0):
    rays = torch.from_numpy(rays_from_w2c(w2c, K, res)).to(device)
    ds, accs, rgbs = [], [], []
    for c in range(0, rays.shape[0], chunk):
        r = torch.transpose(rays[c:c + chunk], 0, 1)
        with torch.no_grad():
            _, _, acc, dist, _, extras = render_rays_3dgrt(
                r, model, train=False, frame_id=frame_id)
        ds.append(dist.float().cpu())
        accs.append(acc.float().cpu())
        rgbs.append(extras["rgb"].float().cpu())
    dist = torch.cat(ds).reshape(res, res).numpy().astype(np.float64)
    acc = torch.cat(accs).reshape(res, res).numpy().astype(np.float64)
    rgb = torch.cat(rgbs).reshape(res, res, 3).numpy().astype(np.float32)
    return dist, acc, np.clip(rgb, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--ply", default=None, help="optional; used for the room bbox")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--n_panel", type=int, default=6, help="views drawn in the panel")
    ap.add_argument("--alpha_thresh", type=float, default=0.5)
    ap.add_argument("--render_chunk", type=int, default=65536)
    ap.add_argument("--N_iters", type=int, default=35000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    g = Gate()

    cams = load_scene_cameras(args.cameras)
    ring = fit_ring(cams)
    K = scale_intrinsics(cams.K, cams.width, cams.height, args.res).astype(np.float64)
    print(f"[geom] {len(cams)} ring views, rendering at {args.res}x{args.res}")
    print(f"[geom] ring target={np.round(ring.target, 4).tolist()} radius={ring.radius:.4f}")

    class _A:
        pass
    a = _A()
    a.N_iters = args.N_iters
    conf = create_3dgrt_conf(a)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "positions" not in ckpt:
        sys.exit(f"{args.ckpt} is not a 3DGRT geometry checkpoint ('positions' missing)")
    scene_extent = float(ckpt.get("scene_extent") or ring.radius * 1.1)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)
    model.build_acc()
    model.eval()
    print(f"[geom] loaded {model.num_gaussians:,} Gaussians "
          f"(scene_extent={scene_extent:.4f}, global_step={ckpt.get('global_step')})")

    g.check(float(model.features_albedo.detach().abs().max()) == 0.0,
            "checkpoint is textureless (features_albedo all zero)")

    # ---- render every ring view -------------------------------------------
    dists, accs, rgbs = [], [], []
    for k in range(len(cams)):
        d, ac, rgb = render_view(model, cams.w2c[k].astype(np.float64), K,
                                 args.res, args.render_chunk, frame_id=k)
        dists.append(d)
        accs.append(ac)
        rgbs.append(rgb)
        print(f"  view {k + 1}/{len(cams)}  coverage={float((ac > args.alpha_thresh).mean()):.4f} "
              f"depth[p5,p50,p95]="
              f"{np.percentile(d[ac > args.alpha_thresh], [5, 50, 95]).round(3).tolist() if (ac > args.alpha_thresh).any() else 'n/a'}")

    cov = np.array([(a > args.alpha_thresh).mean() for a in accs])
    print(f"\n[geom] COVERAGE (alpha > {args.alpha_thresh}) over {len(cams)} views: "
          f"mean={cov.mean():.4f} min={cov.min():.4f} max={cov.max():.4f}")

    # The cameras sit inside a closed room looking at the chair, so a correct
    # fit should fill essentially the WHOLE frame — unlike the real ToF chair,
    # where a single sensor's cone leaves large empty regions. A low number here
    # would mean the control cannot separate coverage from VSD quality, which is
    # the confound this whole experiment exists to remove.
    g.check(cov.min() > 0.90,
            "every ring view is near-fully covered (walls included, so wall "
            "artifacts in VSD cannot be blamed on missing geometry)",
            f"worst view {cov.min():.4f}")

    # Centre crop ~ where the chair is; edges ~ walls and floor. Reported
    # separately because "the chair is covered but the walls are not" is
    # exactly the failure mode that would invalidate the control.
    q = args.res // 4
    edge_cov = []
    for a_ in accs:
        m = np.ones_like(a_, dtype=bool)
        m[q:3 * q, q:3 * q] = False
        edge_cov.append((a_[m] > args.alpha_thresh).mean())
    edge_cov = np.array(edge_cov)
    print(f"[geom] coverage in the OUTER frame (walls/floor only): "
          f"mean={edge_cov.mean():.4f} min={edge_cov.min():.4f}")
    g.check(edge_cov.min() > 0.90,
            "walls/floor region is covered too (not just the chair)",
            f"worst view {edge_cov.min():.4f}")

    # ---- depth plausibility ------------------------------------------------
    valid_d = np.concatenate([d[a_ > args.alpha_thresh] for d, a_ in zip(dists, accs)])
    print(f"[geom] rendered ray distance over all views: "
          f"p1={np.percentile(valid_d, 1):.3f} p50={np.percentile(valid_d, 50):.3f} "
          f"p99={np.percentile(valid_d, 99):.3f} max={valid_d.max():.3f}")
    g.check(np.percentile(valid_d, 50) > 0.2 * ring.radius
            and np.percentile(valid_d, 99) < 20.0 * ring.radius,
            "rendered depth is on the same scale as the camera ring",
            f"median {np.percentile(valid_d, 50):.3f} vs ring radius {ring.radius:.3f}")
    if args.ply:
        from plyfile import PlyData
        v = PlyData.read(args.ply)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        diag = np.linalg.norm(xyz.max(0) - xyz.min(0))
        print(f"[geom] PLY bbox diagonal = {diag:.3f}; max rendered distance = {valid_d.max():.3f}")
        g.check(valid_d.max() < 1.5 * diag,
                "no rendered surface lies far outside the PLY's bounding box",
                f"{valid_d.max():.3f} vs 1.5 x {diag:.3f}")

    # ---- cross-view depth consistency -------------------------------------
    print("\n[geom] cross-view depth consistency (the end-to-end convention test):")
    K_inv = np.linalg.inv(K)
    errs, fracs = [], []
    pairs = [(k, (k + 1) % len(cams)) for k in range(len(cams))]
    for ka, kb in pairs:
        za = ray_dist_to_z(dists[ka], K)
        va = accs[ka] > args.alpha_thresh
        jj, ii = np.nonzero(va)
        if len(ii) == 0:
            continue
        sel = np.random.default_rng(ka).choice(len(ii), size=min(20000, len(ii)), replace=False)
        ii, jj = ii[sel], jj[sel]
        pix = np.stack([ii, jj, np.ones_like(ii)], axis=1).astype(np.float64)
        cam_pts = (K_inv @ pix.T).T * za[jj, ii][:, None]
        c2w_a = np.linalg.inv(cams.w2c[ka].astype(np.float64))
        world = cam_pts @ c2w_a[:3, :3].T + c2w_a[:3, 3]

        w2c_b = cams.w2c[kb].astype(np.float64)
        Xb = world @ w2c_b[:3, :3].T + w2c_b[:3, 3]
        zb_pred = Xb[:, 2]
        ub = K[0, 0] * Xb[:, 0] / zb_pred + K[0, 2]
        vb = K[1, 1] * Xb[:, 1] / zb_pred + K[1, 2]
        inb = (zb_pred > 1e-6) & (ub >= 0) & (ub < args.res - 1) & (vb >= 0) & (vb < args.res - 1)
        if inb.sum() == 0:
            continue
        ui, vi = np.round(ub[inb]).astype(int), np.round(vb[inb]).astype(int)
        zb_obs = ray_dist_to_z(dists[kb], K)[vi, ui]
        vb_ok = accs[kb][vi, ui] > args.alpha_thresh
        if vb_ok.sum() == 0:
            continue
        # Co-visible = agrees to within 2% of depth. Disagreement beyond that is
        # occlusion (a real surface in between), which is expected and is why
        # this is scored as a FRACTION rather than a mean error.
        rel = np.abs(zb_obs[vb_ok] - zb_pred[inb][vb_ok]) / np.maximum(zb_obs[vb_ok], 1e-6)
        errs.append(np.median(rel))
        fracs.append((rel < 0.02).mean())
    errs, fracs = np.array(errs), np.array(fracs)
    print(f"  median relative depth error over {len(errs)} adjacent view pairs: "
          f"{np.median(errs):.5f}")
    print(f"  fraction of reprojected points agreeing within 2%: "
          f"mean={fracs.mean():.4f} min={fracs.min():.4f}")
    g.check(np.median(errs) < 0.02,
            "adjacent views agree on where surfaces are "
            "(rays, poses, and z-depth conversion are mutually consistent)",
            f"median relative error {np.median(errs):.5f}")
    g.check(fracs.mean() > 0.80,
            "most reprojected points land on the same surface in the neighbouring view",
            f"{fracs.mean():.4f} agree within 2%")

    # ---- panel -------------------------------------------------------------
    idxs = np.linspace(0, len(cams) - 1, min(args.n_panel, len(cams))).astype(int)
    fig, axes = plt.subplots(3, len(idxs), figsize=(3.0 * len(idxs), 9.2))
    if len(idxs) == 1:
        axes = axes[:, None]
    dmin = np.percentile(valid_d, 1)
    dmax = np.percentile(valid_d, 99)
    for col, k in enumerate(idxs):
        dm = np.where(accs[k] > args.alpha_thresh, dists[k], np.nan)
        axes[0, col].imshow(dm, cmap="turbo", vmin=dmin, vmax=dmax)
        axes[0, col].set_title(f"{cams.frame_ids[k]}\ndepth", fontsize=8)
        axes[1, col].imshow(accs[k], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"alpha (cov {cov[k]:.3f})", fontsize=8)
        axes[2, col].imshow(rgbs[k])
        axes[2, col].set_title("rgb (should be flat grey)", fontsize=8)
        for r in range(3):
            axes[r, col].axis("off")
    fig.suptitle(f"{os.path.basename(args.ckpt)} — converted geometry, {len(cams)} ring views",
                 fontsize=11)
    fig.tight_layout()
    panel = os.path.join(args.out_dir, "geom_check_panel.png")
    fig.savefig(panel, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[geom] wrote {panel}")
    np.savez_compressed(os.path.join(args.out_dir, "geom_check_stats.npz"),
                        coverage=cov, edge_coverage=edge_cov,
                        xview_median_err=errs, xview_agree_frac=fracs)

    print()
    if g.failures:
        print(f"GATE FAILED — {len(g.failures)} check(s):")
        for f in g.failures:
            print(f"  - {f}")
        print("LOOK AT THE PANEL before doing anything else.")
        sys.exit(1)
    print("GATE PASSED — geometry converted faithfully and renders consistently.")
    print("Still look at the panel: this project's metrics have twice rewarded "
          "visibly worse output.")


if __name__ == "__main__":
    main()

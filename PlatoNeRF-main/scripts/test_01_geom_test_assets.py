#!/usr/bin/env python
"""
scripts/test_01_geom_test_assets.py — gate the geom_test 3DGS primitives and
their cameras BEFORE anything is converted to a 3DGRT checkpoint.

Runs on plain numpy + plyfile, no GPU, no 3DGRT — so it can be run on the dev
machine as well as the cluster.

Every property this checks is one that is INVISIBLE in a tensor shape and would
produce a plausible-looking but geometrically wrong render if assumed wrong.
The authority for each is PlatoControlNet's `src/data/gsplat_fit.py`
`_save_gaussians_ply()` call site, which saves:

    scales    = torch.exp(splat_params["scales"])       -> LINEAR, not log
    opacities = torch.sigmoid(splat_params["opacities"]) -> LINEAR (0,1), not logit
    quats     = splat_params["quats"]                    -> RAW, NOT normalised,
                                                            [w,x,y,z] (identity
                                                            init is [1,0,0,0])

3DGRT stores the opposite for two of the three (configs/base_gs.yaml:
`scale_activation: exp`, `density_activation: sigmoid`, both applied to the
STORED value), so the converter must apply log() and logit(). This script
asserts the PLY side of that contract rather than trusting the paragraph.

Camera convention is not documented anywhere, so it is DECIDED HERE by
measurement: project every Gaussian centre under both candidate conventions and
report which one actually puts the scene in frame.

Usage:
    python scripts/test_01_geom_test_assets.py --root <dir with the 4 files>
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def _pct(x, ps=(0, 1, 50, 90, 99, 100)):
    return {p: float(np.percentile(x, p)) for p in ps}


def _fmt_pct(d):
    return "  ".join(f"p{p}={v:.6g}" for p, v in d.items())


class Gate:
    def __init__(self):
        self.failures = []

    def check(self, ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

def load_ply(path):
    from plyfile import PlyData
    ply = PlyData.read(path)
    v = ply["vertex"]
    names = [p.name for p in v.properties]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    quat = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    scale = np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)
    opacity = np.asarray(v["opacity"]).astype(np.float64)
    return names, xyz, quat, scale, opacity


def gate_ply(g, path):
    print(f"\n--- PLY: {os.path.basename(path)} ---")
    names, xyz, quat, scale, opacity = load_ply(path)
    N = len(xyz)
    print(f"  N = {N:,}   properties = {names}")

    # (1) Textureless: no colour of any kind, or the "zero the features" plan is
    #     silently discarding real data.
    colour_props = [n for n in names if n.startswith(("f_dc", "f_rest", "red", "green", "blue"))]
    g.check(not colour_props, "PLY is textureless (no SH / RGB properties)",
            f"found {colour_props}" if colour_props else "x,y,z + rot + scale + opacity only")

    # (2) SCALE IS LINEAR. A log-space 3DGS PLY (the usual convention) is
    #     dominated by NEGATIVE values — gsplat_fit inits log-scale at -4.0.
    #     Linear scale is strictly positive. This is the discriminator.
    n_nonpos = int((scale <= 0).sum())
    print(f"  scale     : {_fmt_pct(_pct(scale.ravel()))}")
    print(f"              nonpositive entries = {n_nonpos}   mean = {scale.mean():.6g}")
    g.check(n_nonpos == 0, "scale is LINEAR (strictly positive, so already exponentiated)",
            f"{n_nonpos} entries <= 0 would mean log-space")
    # log() of these is what goes into the 3DGRT checkpoint; make sure that is finite.
    g.check(np.isfinite(np.log(np.maximum(scale, 1e-30))).all(),
            "log(scale) is finite for every Gaussian")

    # (3) ANISOTROPY, in LINEAR space — the reference distribution for healthy
    #     geometry that the Phase-1/2 regulariser threshold is set from. The
    #     real chair's numbers must be computed as exp(scale) from its 3DGRT
    #     checkpoint to be comparable to these (3DGRT stores log scale).
    aniso = scale.max(axis=1) / np.maximum(scale.min(axis=1), 1e-30)
    print(f"  anisotropy (max axis / min axis, LINEAR): "
          f"{_fmt_pct(_pct(aniso, (50, 90, 99, 100)))}")
    print(f"              mean log10(aniso) = {np.log10(aniso).mean():.4f}")

    # (4) QUATERNIONS. gsplat_fit saves them RAW (normalisation happens only
    #     inside its rasterisation call), so norms will NOT be 1. 3DGRT
    #     normalises on use too (rotation_activation = "normalize"), so this is
    #     safe to pass through — but a zero-norm quaternion is not.
    qn = np.linalg.norm(quat, axis=1)
    print(f"  |quat|    : {_fmt_pct(_pct(qn, (0, 1, 50, 99, 100)))}")
    g.check(qn.min() > 1e-6, "no degenerate (zero-norm) quaternions",
            f"min |q| = {qn.min():.3g}")
    if abs(qn.mean() - 1.0) > 1e-3:
        print("  note: quaternions are NOT unit-norm (expected — gsplat_fit saves them raw). "
              "3DGRT's rotation_activation='normalize' handles this, but the converter "
              "normalises explicitly so the stored checkpoint is self-consistent.")
    # w-first evidence: component 0 was initialised to 1 and components 1..3 to
    # 0, so after a fit component 0 should still be much larger in magnitude.
    absmean = np.abs(quat).mean(axis=0)
    print(f"  mean|q_i| : w-slot={absmean[0]:.4f}  x={absmean[1]:.4f}  "
          f"y={absmean[2]:.4f}  z={absmean[3]:.4f}")
    g.check(absmean[0] > absmean[1:].max(),
            "quaternion is [w,x,y,z] (component 0 dominates, matching identity init)",
            f"w-slot {absmean[0]:.4f} vs max other {absmean[1:].max():.4f}")

    # (5) OPACITY is a PROBABILITY in (0,1) — sigmoid already applied. A logit
    #     -space PLY would contain negatives and values > 1.
    print(f"  opacity   : {_fmt_pct(_pct(opacity, (0, 1, 50, 99, 100)))}")
    g.check(opacity.min() > 0.0 and opacity.max() < 1.0,
            "opacity is LINEAR probability in (0,1) (sigmoid already applied)",
            f"range [{opacity.min():.6g}, {opacity.max():.6g}]")
    # logit() of these is what goes into the checkpoint as `density`. Only
    # EXACTLY 0.0 or 1.0 (in float32, which is what the PLY stores) gives
    # +-inf; anything strictly inside is fine no matter how close to the ends.
    # Report the worst |logit| rather than guessing an epsilon — a saturated
    # tail is normal for a converged 3DGS fit and is not a defect.
    o32 = opacity.astype(np.float32)
    n_exact = int(((o32 <= np.float32(0.0)) | (o32 >= np.float32(1.0))).sum())
    worst_logit = float(np.abs(np.log(opacity / (1.0 - opacity))).max()) if n_exact == 0 else float("inf")
    print(f"              {int((opacity > 0.999).sum())} Gaussians above 0.999 "
          f"(normal for a converged fit); max |logit| = {worst_logit:.3f}")
    g.check(n_exact == 0,
            "opacity never hits exactly 0.0 or 1.0 in float32 (logit() is finite)",
            f"{n_exact} saturated entries")

    print(f"  xyz bbox  : min={np.round(xyz.min(0), 4).tolist()}  "
          f"max={np.round(xyz.max(0), 4).tolist()}")
    return xyz, scale, opacity


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def _project(xyz, w2c, K, W, H, flip_yz):
    """Project world points with an OpenCV pinhole model. flip_yz=True first
    converts an OpenGL/NeRF-style camera frame (y up, -z forward) into the
    OpenCV frame (y down, +z forward)."""
    Xc = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
    if flip_yz:
        Xc = Xc * np.array([1.0, -1.0, -1.0])
    z = Xc[:, 2]
    in_front = z > 1e-6
    zs = np.where(in_front, z, 1.0)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    inside = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return in_front, inside, z


def gate_cameras(g, path, xyz):
    print(f"\n--- cameras: {os.path.basename(path)} ---")
    cams = json.load(open(path))
    g.check(isinstance(cams, list), "cameras.json is a list of per-frame dicts",
            f"type={type(cams).__name__}")
    n = len(cams)
    keys = sorted(cams[0].keys())
    print(f"  n_frames = {n}   keys = {keys}")

    g.check("K_512" in cams[0], "intrinsics K ARE present (key 'K_512')",
            "no separate intrinsics file needed" if "K_512" in cams[0] else "K is MISSING")
    g.check("w2c" in cams[0], "extrinsics present as 'w2c' (world-to-camera)")

    W = int(cams[0].get("width", 512))
    H = int(cams[0].get("height", 512))
    K = np.array(cams[0]["K_512"], dtype=np.float64)
    print(f"  image    : {W}x{H}   fx={K[0,0]:.4f} fy={K[1,1]:.4f} "
          f"cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    fovx = 2 * np.degrees(np.arctan(0.5 * W / K[0, 0]))
    print(f"  fov_x    : {fovx:.3f} deg")
    Ks = np.stack([np.array(c["K_512"], dtype=np.float64) for c in cams])
    g.check(np.allclose(Ks, Ks[0]), "K is identical across all frames",
            "one shared intrinsic matrix")

    w2cs = np.stack([np.array(c["w2c"], dtype=np.float64) for c in cams])
    g.check(w2cs.shape == (n, 4, 4), "every w2c is 4x4", f"shape {w2cs.shape}")
    Rs = w2cs[:, :3, :3]
    orth = np.abs(Rs @ np.transpose(Rs, (0, 2, 1)) - np.eye(3)).max()
    dets = np.linalg.det(Rs)
    g.check(orth < 1e-4, "w2c rotation blocks are orthonormal", f"max|RR^T - I| = {orth:.2e}")
    g.check(np.all(dets > 0), "w2c rotations are right-handed (det = +1)",
            f"det range [{dets.min():.6f}, {dets.max():.6f}]")
    g.check(np.allclose(w2cs[:, 3, :], np.array([0, 0, 0, 1])), "bottom row is [0,0,0,1]")

    # Camera centres: C = -R^T t. Convention-independent.
    C = np.einsum("nij,nj->ni", np.transpose(Rs, (0, 2, 1)), -w2cs[:, :3, 3])
    centroid = xyz.mean(axis=0)
    print(f"  scene centroid (Gaussian mean) = {np.round(centroid, 4).tolist()}")
    print(f"  camera centre bbox: min={np.round(C.min(0), 4).tolist()}  "
          f"max={np.round(C.max(0), 4).tolist()}")

    # ---- THE CONVENTION TEST -------------------------------------------------
    # Project every Gaussian centre under both candidate conventions. The right
    # one puts the scene in front of the camera and inside the image; the wrong
    # one puts it behind. This is the measurement that replaces the assumption.
    frac_cv, frac_gl = [], []
    for i in range(n):
        f_cv, in_cv, _ = _project(xyz, w2cs[i], K, W, H, flip_yz=False)
        f_gl, in_gl, _ = _project(xyz, w2cs[i], K, W, H, flip_yz=True)
        frac_cv.append(in_cv.mean())
        frac_gl.append(in_gl.mean())
    frac_cv = np.array(frac_cv)
    frac_gl = np.array(frac_gl)
    print(f"  Gaussians landing IN FRAME, mean over {n} views:")
    print(f"      as OpenCV w2c  (y down, +z forward) : {frac_cv.mean():.4f}   "
          f"(per-view min {frac_cv.min():.4f}, max {frac_cv.max():.4f})")
    print(f"      as OpenGL/NeRF (y up,   -z forward) : {frac_gl.mean():.4f}   "
          f"(per-view min {frac_gl.min():.4f}, max {frac_gl.max():.4f})")
    g.check(frac_cv.mean() > 10 * max(frac_gl.mean(), 1e-6),
            "cameras.json w2c is OPENCV convention",
            f"{frac_cv.mean():.3f} in-frame vs {frac_gl.mean():.3f} under OpenGL — "
            "so pose_convert.nerf_c2w_to_opencv_w2c must NOT be applied to these")
    # The ABSOLUTE in-frame fraction is not a quality signal: the cameras sit
    # INSIDE a room and look at the chair, so roughly half the room's Gaussians
    # are behind or beside the camera in every view by construction. What would
    # be a real defect is one view seeing far less than its neighbours (a
    # mis-signed or otherwise broken pose hiding among correct ones), so gate on
    # UNIFORMITY across the ring, plus a loose absolute floor.
    g.check(frac_cv.min() > 0.15,
            "no view is near-blind (loose absolute floor)",
            f"worst view has {frac_cv.min():.3f} of Gaussians in frame")
    g.check(frac_cv.min() > 0.75 * frac_cv.mean(),
            "in-frame fraction is uniform across the ring (no odd pose out)",
            f"min/mean = {frac_cv.min() / frac_cv.mean():.4f}")

    # ---- Ring geometry ------------------------------------------------------
    # Needed to synthesise a denser orbit for VSD pose sampling. Measure it
    # rather than reusing the chair's hardcoded orbit (different world frame).
    d = np.linalg.norm(C - centroid, axis=1)
    print(f"  camera distance to centroid: mean={d.mean():.4f} "
          f"std={d.std():.4f} min={d.min():.4f} max={d.max():.4f}")
    # Which axis is "up"? The ring plane normal is the axis the centres vary
    # least along.
    spread = C.std(axis=0)
    up_axis = int(np.argmin(spread))
    print(f"  camera-centre std per world axis = {np.round(spread, 4).tolist()} "
          f"-> ring plane normal / up axis = {'xyz'[up_axis]}")
    print(f"  camera height along {'xyz'[up_axis]}: mean={C[:, up_axis].mean():.4f} "
          f"std={C[:, up_axis].std():.4f}")
    g.check(spread[up_axis] < 0.05 * max(np.delete(spread, up_axis).max(), 1e-9),
            f"cameras lie on a ring in a plane normal to {'xyz'[up_axis]}",
            f"out-of-plane std {spread[up_axis]:.5f}")

    # Azimuth ordering/uniformity — a hole in the ring changes what a 360 orbit
    # render means.
    plane = [a for a in range(3) if a != up_axis]
    ang = np.degrees(np.arctan2(C[:, plane[1]] - centroid[plane[1]],
                                C[:, plane[0]] - centroid[plane[0]]))
    ang_sorted = np.sort(np.mod(ang, 360.0))
    gaps = np.diff(np.concatenate([ang_sorted, ang_sorted[:1] + 360.0]))
    print(f"  azimuth gaps: mean={gaps.mean():.2f} deg  min={gaps.min():.2f}  "
          f"max={gaps.max():.2f}  (uniform 360/{n} = {360.0/n:.2f})")
    g.check(gaps.max() < 3.0 * (360.0 / n),
            "ring has no large azimuthal hole", f"largest gap {gaps.max():.2f} deg")
    return C, K, w2cs, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory holding <tag>.ply and <tag>_cameras.json pairs")
    args = ap.parse_args()

    plys = sorted(glob.glob(os.path.join(args.root, "*.ply")))
    if not plys:
        sys.exit(f"no .ply files under {args.root}")

    g = Gate()
    for ply in plys:
        tag = os.path.splitext(os.path.basename(ply))[0]
        cam = os.path.join(args.root, f"{tag}_cameras.json")
        print("\n" + "=" * 78)
        print(f"SCENE: {tag}")
        print("=" * 78)
        xyz, scale, opacity = gate_ply(g, ply)
        if not os.path.exists(cam):
            g.check(False, f"cameras file exists for {tag}", cam)
            continue
        gate_cameras(g, cam, xyz)

    print("\n" + "=" * 78)
    if g.failures:
        print(f"GATE FAILED — {len(g.failures)} check(s):")
        for f in g.failures:
            print(f"  - {f}")
        sys.exit(1)
    print("GATE PASSED — all checks green.")


if __name__ == "__main__":
    main()

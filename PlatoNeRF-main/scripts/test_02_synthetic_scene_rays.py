#!/usr/bin/env python
"""
scripts/test_02_synthetic_scene_rays.py — gate `utils/synthetic_scene.py`
before it is used to drive a GPU render or a 14-hour VSD run.

Pure numpy, no GPU, no 3DGRT — runs on the dev machine.

Three properties, each of which would otherwise fail SILENTLY (a wrong render
that still looks like a chair in a room):

  1. RAY/PROJECTION CONSISTENCY. A point placed at distance t along the ray
     built for pixel (u,v) must project back to pixel (u,v) under the same w2c
     and K that V7 is handed. This is the one test that ties the 3DGRT side
     (which consumes rays) to the PlatoControlNet side (which consumes w2c/K)
     — if these two disagree, V7 unprojects the depth map onto geometry that
     sits somewhere other than where 3DGRT drew it, and every conditioning
     signal is subtly misaligned.

  2. NO NERF FLIP. Running the poses through `pose_convert.
     nerf_c2w_to_opencv_w2c` — the natural thing to do by analogy with the
     chair path — must make the reprojection FAIL. Asserting that the wrong
     path is detectably wrong is what stops the right path from being right by
     accident.

  3. RING RECONSTRUCTION. The fitted ring must regenerate the 24 input poses it
     was fitted from, to within a small tolerance, before it is trusted to
     synthesise the ~100 poses that were never in the file.

Usage:
    python scripts/test_02_synthetic_scene_rays.py --root <dir with the assets> \
        [--platocontrolnet_root <path>]
"""

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from utils.synthetic_scene import (  # noqa: E402
    load_scene_cameras, scale_intrinsics, rays_from_w2c, fit_ring,
    ring_reconstruction_error, look_at_w2c,
)

_DEFAULT_PCN_ROOT = "/home/tzofi/orcd/scratch/eli/platocontrolnet"


class Gate:
    def __init__(self):
        self.failures = []

    def check(self, ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)


def project(pts, w2c, K):
    Xc = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
    u = K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1), Xc[:, 2]


def gate_scene(g, cam_path, res, pcn_root):
    print(f"\n--- {os.path.basename(cam_path)} ---")
    cams = load_scene_cameras(cam_path)
    print(f"  {len(cams)} views at {cams.width}x{cams.height}")

    K_r = scale_intrinsics(cams.K, cams.width, cams.height, res).astype(np.float64)
    print(f"  K at {res}x{res}: fx={K_r[0,0]:.4f} cx={K_r[0,2]:.2f}")

    # ---- 1. rays round-trip through the projection model --------------------
    rng = np.random.default_rng(0)
    max_px, max_zerr = 0.0, 0.0
    for vi in rng.choice(len(cams), size=min(6, len(cams)), replace=False):
        w2c = cams.w2c[vi].astype(np.float64)
        rays = rays_from_w2c(w2c, K_r, res).astype(np.float64)
        # sample pixels across the frame, including the corners
        flat = rng.choice(res * res, size=512, replace=False)
        flat = np.concatenate([flat, [0, res - 1, res * (res - 1), res * res - 1]])
        px_want = np.stack([flat % res, flat // res], axis=1).astype(np.float64)
        t = rng.uniform(0.5, 4.0, size=len(flat))[:, None]
        pts = rays[flat, 0, :] + t * rays[flat, 1, :]
        px_got, z = project(pts, w2c, K_r)
        max_px = max(max_px, np.abs(px_got - px_want).max())
        # euclidean ray distance -> z-depth, the conversion the V7 path applies
        dir_mag = np.sqrt(((px_want[:, 0] - K_r[0, 2]) / K_r[0, 0]) ** 2
                          + ((px_want[:, 1] - K_r[1, 2]) / K_r[1, 1]) ** 2 + 1.0)
        max_zerr = max(max_zerr, np.abs(z - t.ravel() / dir_mag).max())
    # Tolerances are set by float32: rays_from_w2c returns float32 because that
    # is what the tracer consumes, so round-off lands around 1e-4 px on a 512px
    # image. Anything structurally wrong (a sign flip, a transposed rotation, a
    # corner-vs-centre pixel convention) is off by >=0.5 px, hundreds of times
    # larger, so this tolerance still separates right from wrong decisively.
    g.check(max_px < 1e-3,
            "rays_from_w2c round-trips through the OpenCV projection model",
            f"max pixel error {max_px:.2e} (float32 round-off; a convention error would be >=0.5)")
    g.check(max_zerr < 1e-5,
            "z-depth = ray_distance / |dir_cam| holds for these rays",
            f"max error {max_zerr:.2e}")

    # ---- 1b. against PlatoControlNet's REAL function, not a copy of it ------
    if pcn_root and os.path.isdir(pcn_root):
        sys.path.insert(0, pcn_root)
        try:
            from src.models.pose_convert import ray_distance_to_z_depth, nerf_c2w_to_opencv_w2c
        except ImportError as e:
            print(f"  [skip] PlatoControlNet import failed ({e})")
            ray_distance_to_z_depth = nerf_c2w_to_opencv_w2c = None
    else:
        print(f"  [skip] --platocontrolnet_root not found: {pcn_root}")
        ray_distance_to_z_depth = nerf_c2w_to_opencv_w2c = None

    if ray_distance_to_z_depth is not None:
        w2c = cams.w2c[0].astype(np.float64)
        rays = rays_from_w2c(w2c, K_r, res).astype(np.float64)
        t_map = np.linspace(1.0, 3.0, res * res).reshape(res, res)
        pts = rays[:, 0, :] + t_map.reshape(-1, 1) * rays[:, 1, :]
        _, z_true = project(pts, w2c, K_r)
        z_conv = ray_distance_to_z_depth(t_map.astype(np.float32), K_r.astype(np.float32))
        err = np.abs(z_conv.ravel() - z_true).max()
        g.check(err < 1e-4,
                "PlatoControlNet's own ray_distance_to_z_depth agrees on these rays",
                f"max error {err:.2e}")

        # ---- 2. the wrong path must be detectably wrong ---------------------
        # Treat the OpenCV w2c as if it were a NeRF c2w and run the chair's
        # conversion on it, as an analogy-driven mistake would.
        bad = nerf_c2w_to_opencv_w2c(np.linalg.inv(w2c).astype(np.float32)).astype(np.float64)
        rays_bad = rays_from_w2c(bad, K_r, res).astype(np.float64)
        flat = np.array([0, res * res // 2, res * res - 1])
        pts_bad = rays_bad[flat, 0, :] + 2.0 * rays_bad[flat, 1, :]
        px_bad, _ = project(pts_bad, w2c, K_r)
        px_want = np.stack([flat % res, flat // res], axis=1).astype(np.float64)
        drift = np.abs(px_bad - px_want).max()
        g.check(drift > 10.0,
                "applying the chair's NeRF->OpenCV flip to these poses IS detectably wrong",
                f"reprojection drifts {drift:.1f} px — so the flip is correctly omitted")

    # ---- 3. ring fit --------------------------------------------------------
    ring = fit_ring(cams)
    print(f"  ring: target={np.round(ring.target, 4).tolist()}  "
          f"radius={ring.radius:.4f}  height={ring.height:.4f}  "
          f"up={np.round(ring.up, 3).tolist()} (axis {'xyz'[ring.up_axis]})")
    rot_err, pos_err = ring_reconstruction_error(cams, ring)
    g.check(rot_err < 0.5,
            "fitted ring regenerates the input poses' ORIENTATION",
            f"max rotation error {rot_err:.4f} deg")
    g.check(pos_err < 0.01 * ring.radius,
            "fitted ring regenerates the input poses' POSITION",
            f"max centre error {pos_err:.5f} ({100 * pos_err / ring.radius:.3f}% of radius)")

    orbit = ring.orbit(100)
    g.check(orbit.shape == (100, 4, 4), "synthesised orbit has the right shape",
            str(orbit.shape))
    d = np.linalg.det(orbit[:, :3, :3])
    g.check(np.abs(d - 1).max() < 1e-5, "synthesised orbit rotations are valid (det=+1)",
            f"max |det-1| = {np.abs(d - 1).max():.2e}")
    c0 = -orbit[0, :3, :3].T @ orbit[0, :3, 3]
    c_in = cams.centers[0]
    g.check(np.linalg.norm(c0 - c_in) < 0.01 * ring.radius,
            "orbit pose 0 coincides with ring frame 0",
            f"offset {np.linalg.norm(c0 - c_in):.5f}")

    # look_at_w2c must be self-consistent: forward row points at the target
    w = look_at_w2c(np.array([1.0, 0.0, 0.5]), ring.target, ring.up)
    f_row = w[2, :3]
    to_t = ring.target - np.array([1.0, 0.0, 0.5])
    cos = f_row @ to_t / np.linalg.norm(to_t)
    g.check(abs(cos - 1) < 1e-9, "look_at_w2c's +z row points at the target",
            f"cos = {cos:.9f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--platocontrolnet_root", default=_DEFAULT_PCN_ROOT)
    args = ap.parse_args()

    cam_files = sorted(glob.glob(os.path.join(args.root, "*_cameras.json")))
    if not cam_files:
        sys.exit(f"no *_cameras.json under {args.root}")

    g = Gate()
    for c in cam_files:
        gate_scene(g, c, args.res, args.platocontrolnet_root)

    print()
    if g.failures:
        print(f"GATE FAILED — {len(g.failures)} check(s):")
        for f in g.failures:
            print(f"  - {f}")
        sys.exit(1)
    print("GATE PASSED — all checks green.")


if __name__ == "__main__":
    main()

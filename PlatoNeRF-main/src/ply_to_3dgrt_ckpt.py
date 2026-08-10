#!/usr/bin/env python
"""
src/ply_to_3dgrt_ckpt.py — convert a textureless gsplat 3DGS PLY (from
PlatoControlNet's `src/data/gsplat_fit.py`) into a 3DGRT `.tar` checkpoint that
`run_platonerf_3dgrt_vsd.py` / `render_test_depth_3dgrt.py` can load directly.

WHY THIS IS NOT A FIELD COPY
----------------------------
The two formats agree on the *meaning* of every field and disagree on the
*parameterisation* of three of them. Each disagreement is invisible in a tensor
shape and each one on its own yields a render that looks plausible and is
wrong, so each is converted explicitly and then PROVEN by round-trip below.

                     gsplat PLY (as written)        3DGRT checkpoint (as stored)
  scale       linear metres, exp() ALREADY applied   log-space; `scale_activation:
              (gsplat_fit saves torch.exp(scales))   exp` is applied on read
              -> converter applies log()

  opacity     probability in (0,1), sigmoid          logit-space; `density_
              ALREADY applied (gsplat_fit saves      activation: sigmoid` is
              torch.sigmoid(opacities))              applied on read
              -> converter applies logit()

  rotation    [w,x,y,z], NOT unit-norm (gsplat_fit   [w,x,y,z], normalised on
              normalises only inside its             read by rotation_activation
              rasterisation call)                    = "normalize"
              -> converter normalises explicitly, so the stored value is
                 self-consistent and the VSD loop's `rotation_drift` diagnostic
                 measures drift rather than the initial denormalisation

Getting `opacity` backwards is the quiet one: a probability dropped straight
into 3DGRT's `density` slot gets sigmoid()'d a SECOND time, mapping the whole
[0,1] range into [0.5, 0.73] — every Gaussian ends up near-identically
semi-transparent, which renders as a soft, plausible, completely
structure-free fog rather than as an obvious error.

Colour is zeroed on purpose: the PLY is genuinely textureless (no f_dc_/f_rest_
properties at all) and Phase 3 VSD is what paints it. `features_albedo` is the
0th-order SH coefficient, so zero corresponds to mid-grey, not black.

THE PROOF
---------
Rather than trusting the three conversions individually, `--verify` rebuilds
each Gaussian's 3x3 world covariance R·S·Sᵀ·Rᵀ from BOTH sides — from the raw
PLY under gsplat's convention, and from the written checkpoint under 3DGRT's
activations and 3DGRT's own `quaternion_to_so3` — and compares them
elementwise. Covariance is the quantity the renderer actually consumes, and it
is invariant to quaternion sign and to normalisation, so agreement there proves
the scale mapping AND the rotation mapping simultaneously, without depending on
either repo's sign conventions being what the docstrings claim.

Requires a GPU node: constructing MixtureOfGaussians builds the 3DGRT tracer.

Usage (cluster):
    python src/ply_to_3dgrt_ckpt.py \
        --ply  .../geom_test/train_chair_<hash>.ply \
        --cameras .../geom_test/train_chair_<hash>_cameras.json \
        --out  logs/geom_train/035000.tar
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

from threedgrut.model.model import MixtureOfGaussians                      # noqa: E402
from threedgrut.utils.misc import quaternion_to_so3, sh_degree_to_specular_dim  # noqa: E402
from run_platonerf_3dgrt_vsd import create_3dgrt_conf                      # noqa: E402


# ---------------------------------------------------------------------------
# PLY reading
# ---------------------------------------------------------------------------

def read_gsplat_ply(path):
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    names = {p.name for p in v.properties}
    required = {"x", "y", "z", "opacity"} | {f"rot_{i}" for i in range(4)} \
        | {f"scale_{i}" for i in range(3)}
    missing = required - names
    if missing:
        raise ValueError(f"{path} is missing required properties: {sorted(missing)}")
    colour = sorted(n for n in names if n.startswith(("f_dc", "f_rest")))
    if colour:
        raise ValueError(
            f"{path} carries colour properties {colour[:4]}... — this converter is for "
            "TEXTURELESS primitives and would silently discard them. Re-fit with "
            "gsplat_fit's geometry_only mode, or extend this script deliberately.")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    quat = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    scale = np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64)
    opacity = np.asarray(v["opacity"]).astype(np.float64)
    return xyz, quat, scale, opacity


def covariance_from_gsplat(quat_wxyz, scale_linear):
    """R·S·Sᵀ·Rᵀ built from the PLY's own numbers, under gsplat's convention
    (w-first quaternion, normalised at use; linear scale). Pure numpy, so it
    shares no code with 3DGRT and can act as an independent reference."""
    q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    S2 = scale_linear ** 2                       # diag(S·Sᵀ)
    return np.einsum("nij,nj,nkj->nik", R, S2, R)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", default=None,
                    help="the scene's cameras.json — used only to derive scene_extent "
                         "from the actual camera ring; falls back to the Gaussian bbox")
    ap.add_argument("--out", required=True, help="output .tar path")
    ap.add_argument("--scene_extent", type=float, default=None,
                    help="override; default is measured from the cameras/geometry")
    ap.add_argument("--global_step", type=int, default=35000,
                    help="stored in the checkpoint. Must be >= the VSD run's --N_iters "
                         "so training resumes directly into Phase 3 rather than "
                         "re-running Phase 1/2 (which needs ToF data that a synthetic "
                         "scene does not have).")
    ap.add_argument("--N_iters", type=int, default=35000,
                    help="only used to build the same conf create_3dgrt_conf builds")
    ap.add_argument("--no_verify", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xyz, quat_raw, scale_lin, opacity_lin = read_gsplat_ply(args.ply)
    N = len(xyz)
    print(f"[convert] {os.path.basename(args.ply)}: {N:,} Gaussians")

    # ---- guard the two parameterisations we are about to invert -------------
    if (scale_lin <= 0).any():
        raise ValueError(
            f"{(scale_lin <= 0).sum()} non-positive scale entries — this PLY looks like "
            "LOG-space scale, not the linear scale gsplat_fit writes. log() would "
            "produce NaN. Refusing to convert.")
    o32 = opacity_lin.astype(np.float32)
    if ((o32 <= np.float32(0.0)) | (o32 >= np.float32(1.0))).any():
        raise ValueError(
            "opacity contains exactly 0.0 or 1.0 — logit() would be infinite. Either "
            "this PLY stores LOGIT-space opacity (which would also show values outside "
            "[0,1]) or the fit saturated. Refusing to convert.")
    qn = np.linalg.norm(quat_raw, axis=1, keepdims=True)
    if (qn < 1e-8).any():
        raise ValueError("degenerate zero-norm quaternion(s) — cannot normalise.")

    # ---- the three conversions ---------------------------------------------
    positions = xyz.astype(np.float32)
    rotation = (quat_raw / qn).astype(np.float32)                  # [w,x,y,z], unit
    scale_log = np.log(scale_lin).astype(np.float32)               # linear -> log
    density_logit = np.log(opacity_lin / (1.0 - opacity_lin)).astype(np.float32)[:, None]

    print(f"[convert]   scale   : linear [{scale_lin.min():.6g}, {scale_lin.max():.6g}] "
          f"-> log [{scale_log.min():.4f}, {scale_log.max():.4f}]")
    print(f"[convert]   opacity : linear [{opacity_lin.min():.6g}, {opacity_lin.max():.6g}] "
          f"-> logit [{density_logit.min():.4f}, {density_logit.max():.4f}]")
    print(f"[convert]   |quat|  : [{qn.min():.4f}, {qn.max():.4f}] -> normalised to 1")

    # ---- scene extent -------------------------------------------------------
    if args.scene_extent is not None:
        scene_extent = float(args.scene_extent)
        src = "override"
    elif args.cameras and os.path.exists(args.cameras):
        cams = json.load(open(args.cameras))
        w2cs = np.stack([np.array(c["w2c"], dtype=np.float64) for c in cams])
        C = np.einsum("nij,nj->ni", np.transpose(w2cs[:, :3, :3], (0, 2, 1)),
                      -w2cs[:, :3, 3])
        centroid = xyz.mean(axis=0)
        scene_extent = float(np.linalg.norm(C - centroid, axis=1).max() * 1.1)
        src = f"1.1 x max camera-to-centroid distance over {len(cams)} views"
    else:
        scene_extent = float(np.linalg.norm(xyz - xyz.mean(0), axis=1).max())
        src = "max Gaussian distance from centroid"
    print(f"[convert]   scene_extent = {scene_extent:.4f}  ({src})")

    # ---- build a real 3DGRT model and let IT produce the checkpoint dict ----
    # Hand-assembling the dict would also have to invent `optimizer` and
    # `background` state, and would drift the moment 3DGRT's format changes.
    # Going through the real class means the file is correct by construction.
    class _A:
        pass
    a = _A()
    a.N_iters = args.N_iters
    conf = create_3dgrt_conf(a)

    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)

    def P(arr):
        return torch.nn.Parameter(torch.from_numpy(arr).to(device=device, dtype=torch.float32))

    model.positions = P(positions)
    model.rotation = P(rotation)
    model.scale = P(scale_log)
    model.density = P(density_logit)
    # Textureless. features_albedo is the 0th-order SH coefficient, so 0 is
    # mid-grey (SH2RGB adds 0.5), which is the right blank canvas for VSD.
    spec_dim = sh_degree_to_specular_dim(model.max_sh_degree)
    model.features_albedo = P(np.zeros((N, 3), dtype=np.float32))
    model.features_specular = P(np.zeros((N, spec_dim), dtype=np.float32))
    print(f"[convert]   features_albedo -> zeros({N},3), "
          f"features_specular -> zeros({N},{spec_dim})  [textureless]")

    # Full SH active. Progressive training ramps n_active_features from 0 over
    # Phase 1/2; a converted checkpoint has no Phase 1/2, so start it where a
    # finished run would be, or Phase 3 would begin with the higher SH bands
    # masked off and silently unable to learn.
    model.max_n_features = model.max_sh_degree
    model.n_active_features = model.max_sh_degree
    model.set_optimizable_parameters()
    model.setup_optimizer()
    model.validate_fields()

    ckpt = model.get_model_parameters()
    ckpt["global_step"] = args.global_step
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[convert] wrote {args.out}  "
          f"(global_step={args.global_step}, n_active_features={model.n_active_features})")

    if args.no_verify:
        return

    # ---------------------------------------------------------------------
    # Round-trip proof: reload the written file the way training will, then
    # compare the RENDERER-FACING quantities against the PLY.
    # ---------------------------------------------------------------------
    print("\n[verify] reloading the written checkpoint via init_from_checkpoint ...")
    model2 = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
    reloaded = torch.load(args.out, map_location=device, weights_only=False)
    model2.init_from_checkpoint(reloaded, setup_optimizer=True)

    fails = []

    def chk(ok, label, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            fails.append(label)

    chk(model2.num_gaussians == N, "Gaussian count preserved",
        f"{model2.num_gaussians:,} vs {N:,}")

    # scale: activation must undo the log
    s_back = model2.get_scale().detach().cpu().numpy().astype(np.float64)
    rel_s = np.abs(s_back - scale_lin) / np.maximum(scale_lin, 1e-12)
    chk(rel_s.max() < 1e-5, "scale_activation(exp) recovers the PLY's LINEAR scale",
        f"max relative error {rel_s.max():.3e}")

    # density: activation must undo the logit
    d_back = model2.get_density().detach().cpu().numpy().astype(np.float64).ravel()
    abs_d = np.abs(d_back - opacity_lin)
    chk(abs_d.max() < 1e-5, "density_activation(sigmoid) recovers the PLY's opacity",
        f"max absolute error {abs_d.max():.3e}")
    # And explicitly rule out the double-sigmoid failure, which is silent:
    double = 1.0 / (1.0 + np.exp(-opacity_lin))
    chk(np.abs(d_back - double).mean() > 1e-3,
        "opacity was NOT double-sigmoided (the silent-fog failure)",
        f"mean |recovered - double_sigmoid| = {np.abs(d_back - double).mean():.4f}")

    # rotation: unit norm after the write
    qn2 = torch.linalg.norm(model2.rotation, dim=1).detach().cpu().numpy()
    chk(np.abs(qn2 - 1).max() < 1e-5, "stored quaternions are unit-norm",
        f"max |‖q‖-1| = {np.abs(qn2 - 1).max():.2e}")

    # THE covariance proof — independent numpy reference vs 3DGRT's own math.
    idx = np.random.default_rng(0).choice(N, size=min(20000, N), replace=False)
    cov_ref = covariance_from_gsplat(quat_raw[idx], scale_lin[idx])
    R_3dgrt = quaternion_to_so3(model2.rotation[idx]).detach().cpu().numpy().astype(np.float64)
    s_3dgrt = s_back[idx]
    cov_got = np.einsum("nij,nj,nkj->nik", R_3dgrt, s_3dgrt ** 2, R_3dgrt)
    scale_of = np.abs(cov_ref).max(axis=(1, 2), keepdims=True)
    rel_cov = (np.abs(cov_got - cov_ref) / np.maximum(scale_of, 1e-30)).max()
    chk(rel_cov < 1e-4,
        "world covariance R·S·Sᵀ·Rᵀ matches the PLY elementwise "
        "(proves the scale AND rotation mappings together)",
        f"max relative error over {len(idx):,} Gaussians = {rel_cov:.3e}")

    # colour really is blank
    chk(float(model2.features_albedo.abs().max()) == 0.0
        and float(model2.features_specular.abs().max()) == 0.0,
        "features_albedo and features_specular are exactly zero (textureless)")

    chk(reloaded.get("global_step") == args.global_step,
        f"global_step = {args.global_step} (Phase 3 starts immediately)")
    chk(model2.n_active_features == model2.max_n_features,
        "all SH bands active on reload",
        f"n_active={model2.n_active_features} max={model2.max_n_features}")

    print()
    if fails:
        print(f"[verify] FAILED — {len(fails)} check(s): {fails}")
        sys.exit(1)
    print("[verify] PASSED — checkpoint is faithful to the PLY.")


if __name__ == "__main__":
    main()

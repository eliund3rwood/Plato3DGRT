#!/usr/bin/env python
"""
src/render_synthetic_orbit.py — render the 360 orbit from a synthetic-scene VSD
checkpoint, and score it PER REGION.

`render_test_depth_3dgrt.py` is the chair's eval path and is bound to the ToF
dataset and the chair's hardcoded pose bank, so this is its synthetic-scene
sibling. It reproduces the one behaviour of that script that matters for
judging a result: **EMA colour weights are substituted automatically** when the
checkpoint has them. SDS/VSD gradients are high-variance by construction, and
every published clean result is EMA-smoothed weights rather than the raw
trajectory — rendering raw systematically overstates how noisy the result is.
`--use_raw_weights` opts out, same flag name as the chair path.

THE REGION SPLIT IS THE POINT
-----------------------------
These scenes do not have uniformly clean geometry. gsplat fitted them against
images whose floor and walls were flattened to a solid colour
(run_gsplat_cluster.py:31-37), so a photometric loss had zero depth gradient
there and the room converged to mush, while the chair — which kept its texture
— is sharp. scripts/test_04 measured the gap: the room's depth error is 4.5x
(train) / 4.0x (val) the chair's against Blender ground truth.

That makes each scene its own within-scene control. The chair region is the
clean-geometry arm and the room is the mush arm, under one prior, in one run,
with nothing else varying. A scene-wide average would report neither, so this
script masks the two apart and reports them separately.

Usage:
    python src/render_synthetic_orbit.py --expname geom_train \
        --cameras $GT/train_chair_<hash>_cameras.json --out_dir orbit_train
"""

import argparse
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

from threedgrut.model.model import MixtureOfGaussians                      # noqa: E402
from run_platonerf_3dgrt_vsd import create_3dgrt_conf, render_rays_3dgrt   # noqa: E402
from utils.synthetic_scene import (                                        # noqa: E402
    load_scene_cameras, scale_intrinsics, rays_from_w2c, fit_ring,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expname", required=True)
    ap.add_argument("--basedir", default="./logs/")
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--ft_path", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_views", type=int, default=60)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--use_raw_weights", action="store_true",
                    help="render the raw VSD trajectory instead of the EMA weights "
                         "(the EMA is what a result should be judged on)")
    ap.add_argument("--chair_radius", type=float, default=0.6)
    ap.add_argument("--chair_height", type=float, default=1.3)
    ap.add_argument("--floor_clearance", type=float, default=0.02,
                    help="Gaussians within this of the lowest point are treated as "
                         "floor, not chair. Without it the cylinder swallows the "
                         "floor disc under the chair, which is mush geometry and "
                         "would contaminate the clean-geometry arm.")
    ap.add_argument("--depth_tol", type=float, default=0.02,
                    help="a chair pixel must be the FRONT surface to within this "
                         "relative depth tolerance")
    ap.add_argument("--render_chunk", type=int, default=65536)
    ap.add_argument("--N_iters", type=int, default=35000)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    savedir = os.path.join(args.basedir, args.expname)

    cams = load_scene_cameras(args.cameras)
    ring = fit_ring(cams)
    R = args.res
    K = scale_intrinsics(cams.K, cams.width, cams.height, R).astype(np.float64)

    if args.ft_path:
        ckpt_path = args.ft_path
    else:
        cands = sorted(f for f in os.listdir(savedir)
                       if f.endswith(".tar") and not f.endswith("_strategy.tar"))
        if not cands:
            sys.exit(f"no checkpoint in {savedir}")
        ckpt_path = os.path.join(savedir, cands[-1])
    print(f"[orbit] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    conf = create_3dgrt_conf(args)
    scene_extent = float(ckpt.get("scene_extent") or ring.radius * 1.1)
    model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
    model.init_from_checkpoint(ckpt, setup_optimizer=False)

    if "ema_albedo" in ckpt and not args.use_raw_weights:
        with torch.no_grad():
            model.features_albedo.copy_(ckpt["ema_albedo"].to(device))
            model.features_specular.copy_(ckpt["ema_specular"].to(device))
        print("[orbit] using EMA colour weights (pass --use_raw_weights to opt out)")
    elif "ema_albedo" not in ckpt:
        print("[orbit] checkpoint has no EMA colour state — rendering raw weights")
    else:
        print("[orbit] --use_raw_weights: rendering the raw VSD trajectory")

    model.build_acc()
    model.eval()
    print(f"[orbit] {model.num_gaussians:,} Gaussians, "
          f"global_step={ckpt.get('global_step')}")

    orbit = ring.orbit(args.n_views)
    ax = int(np.argmin(cams.centers.std(axis=0)))
    plane = [q for q in range(3) if q != ax]

    # ---- chair silhouette, from the GAUSSIANS rather than from depth --------
    # The first version of this masked pixels whose RENDERED depth unprojected
    # inside a world cylinder. That is contaminated two ways, both visible in
    # the mid-run panels: the cylinder swallows the floor disc under the chair,
    # and because room geometry is mush its surfaces land anywhere, including
    # inside the cylinder — so the "chair region" grew lobes of wall.
    #
    # Instead: build a second model holding ONLY the Gaussians whose centres are
    # in the cylinder AND above the floor, render it alone, and call a pixel
    # "chair" where that render is opaque AND its surface is at the same depth
    # as the full render's (i.e. the chair is what you actually see there, not
    # something hidden behind a wall). This is an occlusion-correct silhouette
    # and owes nothing to the appearance being judged.
    pos = model.positions.detach().cpu().numpy()
    rad_g = np.sqrt(pos[:, plane[0]] ** 2 + pos[:, plane[1]] ** 2)
    in_cyl = (rad_g < args.chair_radius) & (pos[:, ax] < args.chair_height)
    # Floor height from a LOW PERCENTILE inside the cylinder, not the global
    # minimum: these fits carry strays below the floor plane (the train scene's
    # bbox bottoms out at -0.074 where the floor is ~0), so a min-based
    # clearance would sit under the floor and fail to exclude it.
    floor = float(np.percentile(pos[in_cyl, ax], 1.0)) if in_cyl.any() else 0.0
    keep = in_cyl & (pos[:, ax] > floor + args.floor_clearance)
    print(f"[orbit] chair Gaussians: {int(keep.sum()):,} / {len(pos):,} "
          f"(cylinder r<{args.chair_radius}, {args.floor_clearance} above floor "
          f"at {floor:.3f})")
    if keep.sum() < 100:
        sys.exit("chair cylinder selected almost no Gaussians — check "
                 "--chair_radius/--chair_height against the scene's world frame")

    chair_model = MixtureOfGaussians(conf, scene_extent=scene_extent).to(device)
    idx = torch.from_numpy(np.nonzero(keep)[0]).to(device)
    with torch.no_grad():
        for name in ("positions", "rotation", "scale", "density",
                     "features_albedo", "features_specular"):
            setattr(chair_model, name,
                    torch.nn.Parameter(getattr(model, name).detach()[idx].clone(),
                                       requires_grad=False))
    chair_model.n_active_features = model.n_active_features
    chair_model.max_n_features = model.max_n_features
    chair_model.build_acc()
    chair_model.eval()

    rgbs, depths, masks = [], [], []
    for k, w2c in enumerate(orbit):
        rays = torch.from_numpy(rays_from_w2c(w2c.astype(np.float64), K, R)).to(device)
        rc, dc = [], []
        for c in range(0, rays.shape[0], args.render_chunk):
            r = torch.transpose(rays[c:c + args.render_chunk], 0, 1)
            with torch.no_grad():
                _, _, _, dist, _, extras = render_rays_3dgrt(r, model, train=False, frame_id=k)
            rc.append(extras["rgb"].float().cpu())
            dc.append(dist.float().cpu())
        rgb = np.clip(torch.cat(rc).reshape(R, R, 3).numpy(), 0, 1)
        dist = torch.cat(dc).reshape(R, R).numpy().astype(np.float64)

        # Chair silhouette: render the chair-only model down the same rays and
        # keep pixels where it is opaque AND its first hit is at (essentially)
        # the same distance as the full scene's — so a chair Gaussian sitting
        # behind a wall does not count as visible chair.
        ac_, ad_ = [], []
        for c in range(0, rays.shape[0], args.render_chunk):
            r = torch.transpose(rays[c:c + args.render_chunk], 0, 1)
            with torch.no_grad():
                _, _, acc_c, dist_c, _, _ = render_rays_3dgrt(
                    r, chair_model, train=False, frame_id=k)
            ac_.append(acc_c.float().cpu())
            ad_.append(dist_c.float().cpu())
        acc_chair = torch.cat(ac_).reshape(R, R).numpy().astype(np.float64)
        dist_chair = torch.cat(ad_).reshape(R, R).numpy().astype(np.float64)
        m = (acc_chair > 0.5) & (dist_chair <= dist * (1.0 + args.depth_tol))

        rgbs.append(rgb)
        depths.append(dist)
        masks.append(m)
        imageio.imwrite(os.path.join(args.out_dir, f"rgb_{k:03d}.png"),
                        (rgb * 255).astype(np.uint8))
        print(f"  view {k + 1}/{len(orbit)}  chair pixels {m.mean() * 100:5.1f}%")

    rgbs = np.stack(rgbs)
    masks = np.stack(masks)

    # ---- per-region view-consistency of APPEARANCE -------------------------
    # The chair failure mode this whole project is chasing is texture that
    # changes character with viewing angle. A single number for that: how much
    # does the region's colour DISTRIBUTION move between adjacent orbit views?
    # Geometry barely changes over 6 degrees, so a large shift is appearance
    # instability, which is exactly the shattering symptom.
    def region_stats(sel):
        means, stds = [], []
        for k in range(len(rgbs)):
            px = rgbs[k][sel[k]] if sel[k].any() else np.zeros((1, 3))
            means.append(px.mean(axis=0))
            stds.append(px.std(axis=0))
        means = np.stack(means)
        stds = np.stack(stds)
        jitter = np.abs(np.diff(means, axis=0)).mean()
        return means.mean(axis=0), stds.mean(), jitter

    print("\n[orbit] appearance statistics by region "
          "(mean RGB, within-view spatial std, adjacent-view mean jitter)")
    for name, sel in (("CHAIR", masks), ("ROOM ", ~masks)):
        mu, sd, jit = region_stats(sel)
        print(f"  {name}: mean RGB {np.round(mu, 4).tolist()}  "
              f"spatial std {sd:.4f}  adjacent-view jitter {jit:.5f}")
    print("  (higher spatial std = more high-frequency speckle; higher jitter = "
          "appearance changing with viewing angle, i.e. the shattering symptom)")

    # ---- panel -------------------------------------------------------------
    idxs = np.linspace(0, len(rgbs) - 1, 8).astype(int)
    fig, axes = plt.subplots(3, len(idxs), figsize=(2.7 * len(idxs), 8.4))
    for col, k in enumerate(idxs):
        axes[0, col].imshow(rgbs[k])
        axes[0, col].set_title(f"view {k}", fontsize=8)
        chair_only = rgbs[k].copy()
        chair_only[~masks[k]] = 1.0
        axes[1, col].imshow(chair_only)
        axes[1, col].set_title("chair region", fontsize=8)
        room_only = rgbs[k].copy()
        room_only[masks[k]] = 1.0
        axes[2, col].imshow(room_only)
        axes[2, col].set_title("room region", fontsize=8)
        for r in range(3):
            axes[r, col].axis("off")
    fig.suptitle(f"{args.expname} @ {ckpt.get('global_step')} — orbit "
                 f"({'EMA' if ('ema_albedo' in ckpt and not args.use_raw_weights) else 'raw'} weights)",
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(args.out_dir, "orbit_panel.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[orbit] wrote {p}")

    # NOTE: no `import imageio` here. imageio is imported at module level, and
    # re-importing it inside this function would rebind the name as a LOCAL for
    # the whole of main() — making the per-view imwrite above an
    # UnboundLocalError, since that runs before the import executes.
    try:
        v = os.path.join(args.out_dir, "orbit.mp4")
        imageio.mimwrite(v, (rgbs * 255).astype(np.uint8), fps=args.fps, quality=8)
        print(f"[orbit] wrote {v}")
    except Exception as e:
        print(f"[orbit] video skipped ({e})")


if __name__ == "__main__":
    main()

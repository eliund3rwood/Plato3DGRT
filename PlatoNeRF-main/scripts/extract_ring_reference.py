#!/usr/bin/env python
"""
scripts/extract_ring_reference.py — pull one scene's ground-truth ring RGB out
of PlatoControlNet's packed `train.h5` / `val.h5`, to serve as the VSD stage's
reference image I_A.

The geom_test assets are geometry only. VSD needs a reference PHOTO of the same
scene: `set_reference_image(I_A)` plus, for V7, `set_reference_geometry()` with
that same view's depth and pose. For the real chair that is
`chair_smooth_walls.png` at dolly pose 30; for a synthetic scene it has to be
one of its own 24 ring views, or the prior is being asked to transfer
appearance from a different room.

Also writes the reference view's INDEX into the ring, so the VSD driver knows
which pose the image was taken from. Getting that wrong is the synthetic-scene
version of the pose-30 alignment the chair path had to confirm by hand — the
image would be real, and attached to the wrong camera.

The h5 group key is the scene_id (`discover_scenes` uses the directory name),
which is not necessarily the geom_test filename stem, so `--scene` is matched
as a substring against the actual keys and the match is printed.

Usage (cluster):
    python scripts/extract_ring_reference.py \
        --h5 /home/tzofi/orcd/scratch/eli/platocontrolnet/data/rings_1000/train.h5 \
        --scene 32f918efaa64a4d9c423490470c47d79 \
        --frame 0 --out data/geom_test/train_chair_ref.png

    # or just see what is in there:
    python scripts/extract_ring_reference.py --h5 .../val.h5 --list
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--scene", default=None,
                    help="scene id, or any unique substring of it (e.g. the hash)")
    ap.add_argument("--frame", type=int, default=0,
                    help="index into the RING ORDER, not the frame_id string")
    ap.add_argument("--out", default=None, help="output .png")
    ap.add_argument("--list", action="store_true", help="list scene keys and exit")
    args = ap.parse_args()

    import h5py

    with h5py.File(args.h5, "r") as f:
        keys = list(f.keys())
        if args.list or not args.scene:
            print(f"{len(keys)} scenes in {args.h5}")
            for k in keys[:40]:
                print(f"  {k}")
            if len(keys) > 40:
                print(f"  ... and {len(keys) - 40} more")
            return

        matches = [k for k in keys if args.scene in k or k in args.scene]
        if len(matches) != 1:
            sys.exit(f"--scene '{args.scene}' matched {len(matches)} keys: {matches[:10]}\n"
                     f"Re-run with --list to see what is available.")
        sid = matches[0]
        print(f"[ref] scene key = {sid}")

        g = f[sid]
        n = int(g.attrs.get("n_views", g["rgb"].shape[0]))
        frame_ids = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in g.attrs["frame_ids"]]
        if not (0 <= args.frame < n):
            sys.exit(f"--frame {args.frame} out of range for {n} views")

        # pack_ring_h5 stores RGB as cv2.imread + BGR2RGB output, verbatim, so
        # this array is already in RGB order — do not swap channels again.
        rgb = np.asarray(g["rgb"][args.frame])           # (512,512,3) uint8, RGB
        w2c = np.asarray(g["w2c"][args.frame], dtype=np.float64)
        K = np.asarray(g["K"][args.frame], dtype=np.float64)
        depth = np.asarray(g["depth"][args.frame], dtype=np.float32)
        near = float(np.asarray(g["near"])[args.frame])
        far = float(np.asarray(g["far"])[args.frame])

    print(f"[ref] {n} views, ring order frame_ids[:4] = {frame_ids[:4]}")
    print(f"[ref] using ring index {args.frame} -> frame_id '{frame_ids[args.frame]}'")
    print(f"[ref] rgb {rgb.shape} {rgb.dtype}  depth {depth.shape} "
          f"[{depth.min():.4f}, {depth.max():.4f}]  near={near:.4f} far={far:.4f}")
    print(f"[ref] K fx={K[0,0]:.4f} cx={K[0,2]:.2f}")

    if args.out is None:
        print("[ref] no --out given; nothing written")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    from PIL import Image
    Image.fromarray(rgb).save(args.out)
    print(f"[ref] wrote {args.out}")

    # The sidecar is what makes the reference usable: it records WHICH ring view
    # this is, so the VSD driver can attach the image to the right camera rather
    # than assuming index 0. The stored w2c/K are also the cross-check that the
    # h5's cameras and the geom_test cameras.json describe the same ring.
    meta = {
        "scene_key": sid,
        "h5": os.path.abspath(args.h5),
        "ring_index": args.frame,
        "frame_id": frame_ids[args.frame],
        "n_views": n,
        "w2c": w2c.tolist(),
        "K": K.tolist(),
        "near": near,
        "far": far,
        "depth_is_z_depth": True,
        "note": ("depth here is the gsplat-composited METRIC depth pack_ring_h5 stores "
                 "raw; it is z-depth, matching unproject_grid. 3DGRT's pred_dist is "
                 "euclidean ray distance and must be converted before it is comparable."),
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[ref] wrote {meta_path}")


if __name__ == "__main__":
    main()

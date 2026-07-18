"""
src/data/replica_io.py — Replica dataset preprocessing.

Converts the Replica-NeRF / NICE-SLAM format into the canonical pipeline format
consumed by gsplat_fit.py:

  data/scenes/<scene_id>/
    rgb_512/<frame_id>.png      -- 512×512 uint8 RGB
    cameras.json               -- w2c + K_512 per frame

Expected Replica source layout (NICE-SLAM / iMAP style):
  <replica_root>/<scene_name>/
    rgb/
      frame000000.jpg  (or .png)  -- 1200×680
    depth/
      frame000000.png             -- uint16, depth scale = 6553.5  (1/6553.5 → metres)
    traj.txt                      -- one 4×4 c2w per line (16 space-separated floats)

Camera intrinsics are fixed for all Replica scenes (virtual pinhole camera):
  fx = fy = 600.0,  cx = 599.5,  cy = 339.5,  W = 1200,  H = 680

If your Replica download uses a different layout or intrinsics, override with
--fx / --fy / --cx / --cy / --width / --height flags.

Usage:
    python -m src.data.replica_io \\
        --scene-id room0 \\
        --replica-root /path/to/Replica \\
        --output-root data/scenes \\
        --frame-stride 5

    python -m src.data.replica_io \\
        --all-scenes \\
        --replica-root /path/to/Replica \\
        --output-root data/scenes
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TARGET_SIZE = 512

# Default Replica virtual camera parameters (NICE-SLAM version)
REPLICA_FX   = 600.0
REPLICA_FY   = 600.0
REPLICA_CX   = 599.5
REPLICA_CY   = 339.5
REPLICA_W    = 1200
REPLICA_H    = 680

# Known Replica scene names
REPLICA_SCENES = [
    "room0", "room1", "room2",
    "office0", "office1", "office2", "office3", "office4",
    "hotel0",
    "frl_apartment0", "frl_apartment1", "frl_apartment2",
    "frl_apartment3", "frl_apartment4", "frl_apartment5",
    "apartment0", "apartment1", "apartment2",
]


def _load_traj(traj_path: Path) -> list[np.ndarray]:
    """
    Load camera-to-world poses from traj.txt.
    Each line: 16 space-separated floats (4×4 row-major).
    Returns list of (4, 4) float64 arrays.
    """
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 16:
                mat = np.array(vals, dtype=np.float64).reshape(4, 4)
                poses.append(mat)
    return poses


def _adjust_K(K: np.ndarray, orig_h: int, orig_w: int, target: int = TARGET_SIZE) -> np.ndarray:
    s = min(orig_h, orig_w)
    x0 = (orig_w - s) // 2
    y0 = (orig_h - s) // 2
    scale = target / s
    K_new = K.copy()
    K_new[0, 2] = (K[0, 2] - x0) * scale
    K_new[1, 2] = (K[1, 2] - y0) * scale
    K_new[0, 0] = K[0, 0] * scale
    K_new[1, 1] = K[1, 1] * scale
    return K_new


def process_scene(
    scene_id: str,
    replica_root: Path,
    output_root: Path,
    fx: float = REPLICA_FX,
    fy: float = REPLICA_FY,
    cx: float = REPLICA_CX,
    cy: float = REPLICA_CY,
    img_w: int = REPLICA_W,
    img_h: int = REPLICA_H,
    frame_stride: int = 5,
    max_frames: int = 0,
) -> None:
    """
    Extract Replica scene → rgb_512/ + cameras.json.
    """
    src = replica_root / scene_id
    out = output_root  / scene_id

    traj_path = src / "traj.txt"
    rgb_dir   = src / "rgb"

    if not traj_path.exists():
        raise FileNotFoundError(f"traj.txt not found at {traj_path}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgb/ not found at {src}")

    poses = _load_traj(traj_path)
    rgb_paths = sorted(
        list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"),
    )

    # Align poses and images (should be 1:1 by frame index)
    n = min(len(poses), len(rgb_paths))
    poses     = poses[:n][::frame_stride]
    rgb_paths = rgb_paths[:n][::frame_stride]
    if max_frames > 0:
        poses     = poses[:max_frames]
        rgb_paths = rgb_paths[:max_frames]

    print(f"[replica_io] {scene_id}: {len(rgb_paths)} frames (stride={frame_stride})")

    K_orig = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    K_512 = _adjust_K(K_orig, img_h, img_w)

    rgb_out = out / "rgb_512"
    rgb_out.mkdir(parents=True, exist_ok=True)

    cameras = []
    for i, (c2w, cpath) in enumerate(tqdm(zip(poses, rgb_paths), desc=f"  {scene_id}", total=len(rgb_paths))):
        if not np.isfinite(c2w).all():
            continue

        img_bgr = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Recompute K_512 if actual image dims differ from defaults
        K_frame = _adjust_K(K_orig, orig_h, orig_w)

        s    = min(orig_h, orig_w)
        y0   = (orig_h - s) // 2
        x0   = (orig_w - s) // 2
        crop = img_rgb[y0:y0+s, x0:x0+s]
        img_512 = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        fid = f"{i * frame_stride:06d}"
        cv2.imwrite(str(rgb_out / f"{fid}.png"), cv2.cvtColor(img_512, cv2.COLOR_RGB2BGR))

        w2c = np.linalg.inv(c2w)
        cameras.append({
            "frame_id": fid,
            "w2c":      w2c.tolist(),
            "K_512":    K_frame.tolist(),
            "width":    TARGET_SIZE,
            "height":   TARGET_SIZE,
        })

    if not cameras:
        print(f"[replica_io] WARNING: no valid frames for {scene_id}")
        return

    with open(out / "cameras.json", "w") as f:
        json.dump(cameras, f, indent=2)

    print(f"[replica_io] {scene_id}: {len(cameras)} frames → {out}")
    print(f"  Next: python -m src.data.gsplat_fit --scene-id {scene_id} "
          f"--dataset canonical --scenes-root <output_root>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Replica (NICE-SLAM format) to pipeline canonical format"
    )
    parser.add_argument("--scene-id",     default=None)
    parser.add_argument("--all-scenes",   action="store_true")
    parser.add_argument("--replica-root", required=True)
    parser.add_argument("--output-root",  default="data/scenes")
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="Use every Nth frame (default 5; ~180 frames/scene)")
    parser.add_argument("--max-frames",   type=int, default=0)
    # Camera override flags (for non-standard Replica downloads)
    parser.add_argument("--fx",     type=float, default=REPLICA_FX)
    parser.add_argument("--fy",     type=float, default=REPLICA_FY)
    parser.add_argument("--cx",     type=float, default=REPLICA_CX)
    parser.add_argument("--cy",     type=float, default=REPLICA_CY)
    parser.add_argument("--width",  type=int,   default=REPLICA_W)
    parser.add_argument("--height", type=int,   default=REPLICA_H)
    args = parser.parse_args()

    replica_root = Path(args.replica_root)
    output_root  = Path(args.output_root)

    if args.all_scenes:
        scene_ids = [s for s in REPLICA_SCENES if (replica_root / s).exists()]
        if not scene_ids:
            # Fallback: discover any sub-directory
            scene_ids = sorted(p.name for p in replica_root.iterdir() if p.is_dir())
    elif args.scene_id:
        scene_ids = [args.scene_id]
    else:
        parser.error("Provide --scene-id or --all-scenes")
        return

    for sid in scene_ids:
        try:
            process_scene(
                sid, replica_root, output_root,
                fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                img_w=args.width, img_h=args.height,
                frame_stride=args.frame_stride, max_frames=args.max_frames,
            )
        except Exception as e:
            print(f"[replica_io] ERROR {sid}: {e}")


if __name__ == "__main__":
    main()

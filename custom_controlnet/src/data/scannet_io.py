"""
src/data/scannet_io.py — ScanNet preprocessing: RGB frames + camera poses.

Converts a pre-extracted ScanNet scene (color/, depth/, pose/, intrinsic/ dirs)
into the canonical pipeline format consumed by gsplat_fit.py:

  data/scenes/<scene_id>/
    rgb_512/<frame_id>.png      -- 512×512 uint8 RGB (input to 3DGS + training)
    cameras.json               -- w2c + K_512 per frame (same format as gsplat_fit output)

NOTE: depth_512/ and depth_meta.json are NOT written here. Depth maps are
rendered from the 3DGS model by render_depth.py. The 3DGS-rendered depth
(with Gaussian artifacts) is the training signal — not sensor depth.

Sensor depth IS used to build init_pointcloud.npy — a 3D point cloud that
seeds the 3DGS optimization. Without this, gsplat starts from random points
and produces blurry, low-quality reconstructions.

ScanNet camera model: pinhole (no fisheye undistortion needed).

Usage:
    python -m src.data.scannet_io \\
        --scene-id scene0000_00 \\
        --scannet-root /path/to/scannet/scans \\
        --output-root data/scenes \\
        --frame-stride 10

    python -m src.data.scannet_io \\
        --all-scenes \\
        --scannet-root /path/to/scannet/scans \\
        --output-root data/scenes
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TARGET_SIZE = 512


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_intrinsics(path: Path) -> np.ndarray:
    """Load 4×4 intrinsic matrix from ScanNet .txt file."""
    return np.loadtxt(str(path), dtype=np.float64)


def _load_pose(path: Path) -> np.ndarray | None:
    """Load 4×4 camera-to-world matrix. Returns None when invalid (inf/nan)."""
    try:
        mat = np.loadtxt(str(path), dtype=np.float64)
    except Exception:
        return None
    if not np.isfinite(mat).all():
        return None
    return mat


def _center_crop_square(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return arr[y0 : y0 + s, x0 : x0 + s]


def _adjust_K_for_crop_resize(
    K: np.ndarray, orig_h: int, orig_w: int, target: int = TARGET_SIZE
) -> np.ndarray:
    """Adjust 3×3 pinhole K for center-crop-to-square then resize to target."""
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


# ── Scene processing ──────────────────────────────────────────────────────────

def _build_init_pointcloud(
    src: Path,
    out: Path,
    max_points: int = 150_000,
    depth_stride: int = 20,
) -> None:
    """
    Unproject sensor depth frames → world-space point cloud for 3DGS init.
    Saves init_pointcloud.npy to the output scene directory.
    Skipped silently if depth/ or intrinsic_depth.txt are absent.
    """
    depth_dir = src / "depth"
    intrinsic_path = src / "intrinsic" / "intrinsic_depth.txt"
    if not depth_dir.exists() or not intrinsic_path.exists():
        return

    K4 = _load_intrinsics(intrinsic_path)
    K  = K4[:3, :3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    depth_paths = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem))
    depth_paths = depth_paths[::depth_stride]

    all_pts: list[np.ndarray] = []
    for dp in depth_paths:
        pose_path = src / "pose" / f"{dp.stem}.txt"
        c2w = _load_pose(pose_path)
        if c2w is None:
            continue
        depth_mm = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue
        depth_m = depth_mm.astype(np.float32) / 1000.0
        h, w = depth_m.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = depth_m > 0.1
        z = depth_m[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # N×4
        pts_world = (c2w @ pts_cam.T).T[:, :3]
        all_pts.append(pts_world.astype(np.float32))

    if not all_pts:
        return

    pts = np.concatenate(all_pts, axis=0)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    np.save(str(out / "init_pointcloud.npy"), pts)
    print(f"  [scannet_io] init_pointcloud.npy: {len(pts)} points")


def process_scene(
    scene_id: str,
    scannet_root: Path,
    output_root: Path,
    frame_stride: int = 10,
    max_frames: int = 0,
) -> None:
    """
    Produce rgb_512/ + cameras.json + init_pointcloud.npy from a pre-extracted ScanNet scene.
    The scene must have: color/, pose/, intrinsic/ subdirectories.
    """
    src = scannet_root / scene_id
    out = output_root  / scene_id

    for sub in ("color", "pose", "intrinsic"):
        if not (src / sub).exists():
            raise FileNotFoundError(
                f"'{sub}/' not found at {src}. "
                "Extract the .sens file first with ScanNet's SensReader."
            )

    K_color_4x4 = _load_intrinsics(src / "intrinsic" / "intrinsic_color.txt")
    K_color = K_color_4x4[:3, :3]

    # Discover frames sorted by integer frame number
    color_paths = sorted(
        list((src / "color").glob("*.jpg")) + list((src / "color").glob("*.png")),
        key=lambda p: int(p.stem),
    )
    color_paths = color_paths[::frame_stride]
    if max_frames > 0:
        color_paths = color_paths[:max_frames]

    print(f"[scannet_io] {scene_id}: {len(color_paths)} frames (stride={frame_stride})")

    rgb_out = out / "rgb_512"
    rgb_out.mkdir(parents=True, exist_ok=True)

    cameras = []

    for cpath in tqdm(color_paths, desc=f"  {scene_id}"):
        fid = cpath.stem

        # Pose
        pose_path = src / "pose" / f"{fid}.txt"
        if not pose_path.exists():
            continue
        c2w = _load_pose(pose_path)
        if c2w is None:
            continue  # invalid pose (common in ScanNet — inf entries)
        w2c = np.linalg.inv(c2w)

        # Color
        img_bgr = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        K_512 = _adjust_K_for_crop_resize(K_color, orig_h, orig_w)

        img_sq  = _center_crop_square(img_rgb)
        img_512 = cv2.resize(img_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(rgb_out / f"{fid}.png"), cv2.cvtColor(img_512, cv2.COLOR_RGB2BGR))

        cameras.append({
            "frame_id": fid,
            "w2c":      w2c.tolist(),
            "K_512":    K_512.tolist(),
            "width":    TARGET_SIZE,
            "height":   TARGET_SIZE,
        })

    if not cameras:
        print(f"[scannet_io] WARNING: no valid frames for {scene_id}")
        return

    with open(out / "cameras.json", "w") as f:
        json.dump(cameras, f, indent=2)

    _build_init_pointcloud(src, out)

    print(f"[scannet_io] {scene_id}: {len(cameras)} frames → {out}")
    print(f"  Next: python -m src.data.gsplat_fit --scene-id {scene_id} "
          f"--dataset canonical --scenes-root <output_root>")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ScanNet pre-extracted frames to pipeline canonical format"
    )
    parser.add_argument("--scene-id",     default=None)
    parser.add_argument("--all-scenes",   action="store_true")
    parser.add_argument("--scannet-root", required=True,
                        help="ScanNet scans/ directory")
    parser.add_argument("--output-root",  default="data/scenes")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames",   type=int, default=0)
    args = parser.parse_args()

    scannet_root = Path(args.scannet_root)
    output_root  = Path(args.output_root)

    if args.all_scenes:
        scene_ids = sorted(p.name for p in scannet_root.iterdir() if p.is_dir())
    elif args.scene_id:
        scene_ids = [args.scene_id]
    else:
        parser.error("Provide --scene-id or --all-scenes")
        return

    for sid in scene_ids:
        try:
            process_scene(sid, scannet_root, output_root, args.frame_stride, args.max_frames)
        except Exception as e:
            print(f"[scannet_io] ERROR {sid}: {e}")


if __name__ == "__main__":
    main()

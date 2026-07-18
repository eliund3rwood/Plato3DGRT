"""
src/data/tum_io.py — TUM RGB-D dataset preprocessing.

Converts TUM RGB-D sequences into the canonical pipeline format:

  data/scenes/<scene_id>/
    rgb_512/<frame_id>.png      -- 512x512 uint8 RGB
    cameras.json               -- w2c + K_512 per frame

TUM RGB-D source layout:
  <seq_root>/<seq_name>/
    rgb/        <timestamp>.png   (640x480 RGB)
    depth/      <timestamp>.png   (640x480 uint16, units = mm/0.2 i.e. / 5000 -> metres)
    rgb.txt                       (# timestamp filename)
    depth.txt
    groundtruth.txt               (# timestamp tx ty tz qx qy qz qw)  [camera-to-world]

Camera intrinsics differ per freiburg series:
  fr1: fx=517.3  fy=516.5  cx=318.6  cy=255.3
  fr2: fx=520.9  fy=521.0  cx=325.1  cy=249.7
  fr3: fx=535.4  fy=539.2  cx=320.1  cy=247.6

Usage:
    python -m src.data.tum_io \\
        --seq-id freiburg1_room \\
        --tum-root /path/to/tum \\
        --output-root data/scenes \\
        --frame-stride 5

    python -m src.data.tum_io --all-seqs \\
        --tum-root /path/to/tum \\
        --output-root data/scenes
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TARGET_SIZE   = 512
DEPTH_SCALE   = 5000.0   # TUM uint16 -> metres: value / 5000
MAX_DEPTH_M   = 10.0
MAX_TS_DIFF_S = 0.02     # max timestamp gap for RGB <-> pose association (20 ms)


# ── Per-camera intrinsics ─────────────────────────────────────────────────────

_INTRINSICS = {
    "fr1": dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3, w=640, h=480),
    "fr2": dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7, w=640, h=480),
    "fr3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6, w=640, h=480),
}

KNOWN_SEQS = [
    # freiburg1
    "freiburg1_desk", "freiburg1_desk2", "freiburg1_room",
    "freiburg1_plant", "freiburg1_teddy", "freiburg1_xyz",
    # freiburg2
    "freiburg2_desk", "freiburg2_dishes", "freiburg2_flowerbouquet",
    "freiburg2_large_no_loop", "freiburg2_room",
    # freiburg3
    "freiburg3_long_office_household",
    "freiburg3_nostructure_texture_near_withloop",
    "freiburg3_structure_texture_far",
    "freiburg3_structure_texture_near",
    "freiburg3_sitting_static", "freiburg3_sitting_xyz",
]


def _get_intrinsics(seq_id: str) -> dict:
    # Match "freiburg1" -> fr1, "freiburg2" -> fr2, "freiburg3" -> fr3
    for series, key in [("freiburg1", "fr1"), ("freiburg2", "fr2"), ("freiburg3", "fr3")]:
        if series in seq_id:
            return _INTRINSICS[key]
    for prefix, K in _INTRINSICS.items():
        if prefix in seq_id:
            return K
    print(f"  [tum_io] Unknown camera for '{seq_id}', defaulting to fr1 intrinsics")
    return _INTRINSICS["fr1"]


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = qx**2 + qy**2 + qz**2 + qw**2
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz),    s*(qx*qy - qw*qz),    s*(qx*qz + qw*qy)],
        [    s*(qx*qy + qw*qz),1 - s*(qx*qx + qz*qz),    s*(qy*qz - qw*qx)],
        [    s*(qx*qz - qw*qy),    s*(qy*qz + qw*qx),1 - s*(qx*qx + qy*qy)],
    ])


def _pose_to_c2w(tx, ty, tz, qx, qy, qz, qw) -> np.ndarray:
    """TUM groundtruth line -> 4x4 camera-to-world matrix."""
    R = _quat_to_rotmat(qx, qy, qz, qw)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3,  3] = [tx, ty, tz]
    return c2w


# ── Timestamp association ─────────────────────────────────────────────────────

def _read_stamp_list(path: Path) -> list[tuple[float, str]]:
    """Read a TUM timestamp-file list (rgb.txt / depth.txt). Returns [(ts, filename)]."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((float(parts[0]), parts[1]))
    return entries


def _read_groundtruth(path: Path) -> list[tuple]:
    """Read groundtruth.txt. Returns [(ts, tx, ty, tz, qx, qy, qz, qw)]."""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 8:
                poses.append(tuple(float(v) for v in parts))
    return poses


def _associate(
    primary: list[tuple[float, str]],
    secondary_stamps: np.ndarray,
    max_diff: float = MAX_TS_DIFF_S,
) -> dict[int, int]:
    """
    For each primary (ts, file) find the nearest secondary timestamp index.
    Returns {primary_idx: secondary_idx} for pairs within max_diff seconds.
    """
    matches = {}
    for i, (ts, _) in enumerate(primary):
        diffs = np.abs(secondary_stamps - ts)
        j = int(np.argmin(diffs))
        if diffs[j] <= max_diff:
            matches[i] = j
    return matches


# ── Intrinsic adjustment ──────────────────────────────────────────────────────

def _adjust_K(K_dict: dict, target: int = TARGET_SIZE) -> np.ndarray:
    fx, fy = K_dict["fx"], K_dict["fy"]
    cx, cy = K_dict["cx"], K_dict["cy"]
    w,  h  = K_dict["w"],  K_dict["h"]

    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)

    s     = min(h, w)
    x0    = (w - s) // 2
    y0    = (h - s) // 2
    scale = target / s

    K_new = K.copy()
    K_new[0, 2] = (cx - x0) * scale
    K_new[1, 2] = (cy - y0) * scale
    K_new[0, 0] = fx * scale
    K_new[1, 1] = fy * scale
    return K_new


# ── Scene processing ──────────────────────────────────────────────────────────

# ── Init point cloud from sensor depth ───────────────────────────────────────

def build_init_pointcloud(
    src: Path,
    K_dict: dict,
    gt_list: list,
    out: Path,
    depth_stride: int = 10,
    pts_per_frame: int = 5000,
    max_total_pts: int = 150000,
) -> None:
    """
    Unproject TUM sensor depth frames into world-space 3D points and save as
    init_pointcloud.ply for 3DGS initialization.

    Uses every `depth_stride`-th depth frame; subsamples `pts_per_frame` valid
    pixels per frame; caps total at `max_total_pts`.
    """
    depth_list = _read_stamp_list(src / "depth.txt")
    if not depth_list:
        print(f"  [tum_io] No depth.txt entries, skipping init point cloud")
        return

    gt_stamps = np.array([g[0] for g in gt_list])
    depth_assoc = _associate(depth_list, gt_stamps)
    depth_indices = sorted(depth_assoc.keys())[::depth_stride]

    fx, fy = K_dict["fx"], K_dict["fy"]
    cx, cy = K_dict["cx"], K_dict["cy"]

    all_pts: list[np.ndarray] = []

    for depth_idx in tqdm(depth_indices, desc="  build init PLY"):
        gt_idx = depth_assoc[depth_idx]
        _, depth_file = depth_list[depth_idx]

        depth_img = cv2.imread(str(src / depth_file), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            continue
        depth_m = depth_img.astype(np.float32) / DEPTH_SCALE

        h, w = depth_m.shape
        yy, xx = np.mgrid[0:h, 0:w]
        valid = (depth_m > 0.1) & (depth_m < MAX_DEPTH_M)
        if not valid.any():
            continue

        z = depth_m[valid]
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)  # N×3

        # Subsample per-frame
        if len(pts_cam) > pts_per_frame:
            idx = np.random.choice(len(pts_cam), pts_per_frame, replace=False)
            pts_cam = pts_cam[idx]

        # Transform to world space
        g = gt_list[gt_idx]
        _, tx, ty, tz, qx, qy, qz, qw = g
        c2w = _pose_to_c2w(tx, ty, tz, qx, qy, qz, qw)
        R, t = c2w[:3, :3], c2w[:3, 3]
        pts_world = (R @ pts_cam.T).T + t  # N×3

        all_pts.append(pts_world.astype(np.float32))

    if not all_pts:
        print(f"  [tum_io] No valid depth frames found, skipping init point cloud")
        return

    pts = np.concatenate(all_pts, axis=0)
    if len(pts) > max_total_pts:
        idx = np.random.choice(len(pts), max_total_pts, replace=False)
        pts = pts[idx]

    ply_path = out / "init_pointcloud.ply"
    try:
        from plyfile import PlyData, PlyElement
        arr = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply_path))
        print(f"  [tum_io] Saved init point cloud: {len(pts)} pts → {ply_path}")
    except ImportError:
        npy_path = out / "init_pointcloud.npy"
        np.save(str(npy_path), pts)
        print(f"  [tum_io] plyfile not installed — saved {npy_path} ({len(pts)} pts)")


def process_seq(
    seq_id: str,
    tum_root: Path,
    output_root: Path,
    frame_stride: int = 5,
    max_frames: int = 0,
    init_ply_only: bool = False,
) -> None:
    """
    Preprocess one TUM RGB-D sequence -> rgb_512/ + cameras.json + init_pointcloud.ply.
    If init_ply_only=True, skip RGB/camera processing and only (re)build the PLY.
    """
    src = tum_root / seq_id
    out = output_root / seq_id

    for f in ("rgb", "depth", "rgb.txt", "depth.txt", "groundtruth.txt"):
        if not (src / f).exists():
            raise FileNotFoundError(f"'{f}' not found at {src}")

    K_dict  = _get_intrinsics(seq_id)
    K_512   = _adjust_K(K_dict)

    rgb_list = _read_stamp_list(src / "rgb.txt")
    gt_list  = _read_groundtruth(src / "groundtruth.txt")
    gt_stamps = np.array([g[0] for g in gt_list])

    out.mkdir(parents=True, exist_ok=True)

    if init_ply_only:
        build_init_pointcloud(src, K_dict, gt_list, out)
        return

    # Associate each RGB frame to the nearest groundtruth pose
    assoc = _associate(rgb_list, gt_stamps)

    # Apply stride + cap
    rgb_indices = sorted(assoc.keys())[::frame_stride]
    if max_frames > 0:
        rgb_indices = rgb_indices[:max_frames]

    print(f"[tum_io] {seq_id}: {len(rgb_indices)} frames (stride={frame_stride})")

    rgb_out = out / "rgb_512"
    rgb_out.mkdir(parents=True, exist_ok=True)

    cameras = []
    w_orig, h_orig = K_dict["w"], K_dict["h"]
    s  = min(h_orig, w_orig)
    y0 = (h_orig - s) // 2
    x0 = (w_orig - s) // 2

    for i, rgb_idx in enumerate(tqdm(rgb_indices, desc=f"  {seq_id}")):
        ts_rgb, rgb_file = rgb_list[rgb_idx]
        gt = gt_list[assoc[rgb_idx]]
        _, tx, ty, tz, qx, qy, qz, qw = gt

        c2w = _pose_to_c2w(tx, ty, tz, qx, qy, qz, qw)
        if not np.isfinite(c2w).all():
            continue
        w2c = np.linalg.inv(c2w)

        # Load RGB
        img_bgr = cv2.imread(str(src / rgb_file), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Center crop + resize
        crop    = img_rgb[y0:y0+s, x0:x0+s]
        img_512 = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        fid = f"{i * frame_stride:06d}"
        cv2.imwrite(str(rgb_out / f"{fid}.png"), cv2.cvtColor(img_512, cv2.COLOR_RGB2BGR))

        cameras.append({
            "frame_id": fid,
            "w2c":      w2c.tolist(),
            "K_512":    K_512.tolist(),
            "width":    TARGET_SIZE,
            "height":   TARGET_SIZE,
        })

    if not cameras:
        print(f"[tum_io] WARNING: no valid frames for {seq_id}")
        return

    with open(out / "cameras.json", "w") as f:
        json.dump(cameras, f, indent=2)

    print(f"[tum_io] {seq_id}: {len(cameras)} frames -> {out}")

    # Build init point cloud from sensor depth for better 3DGS initialization
    build_init_pointcloud(src, K_dict, gt_list, out)

    print(f"  Next: python -m src.data.gsplat_fit --scene-id {seq_id} "
          f"--dataset canonical --scenes-root <output_root>")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TUM RGB-D sequences to pipeline canonical format"
    )
    parser.add_argument("--seq-id",      default=None,
                        help="e.g. freiburg1_room")
    parser.add_argument("--all-seqs",    action="store_true")
    parser.add_argument("--tum-root",    required=True,
                        help="Directory containing TUM sequence folders")
    parser.add_argument("--output-root", default="data/scenes")
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="Use every Nth frame (default 5; ~200-400 frames/seq)")
    parser.add_argument("--max-frames",  type=int, default=0)
    parser.add_argument("--init-ply-only", action="store_true",
                        help="Only (re)build init_pointcloud.ply from sensor depth; skip RGB/cameras")
    args = parser.parse_args()

    tum_root    = Path(args.tum_root)
    output_root = Path(args.output_root)

    if args.all_seqs:
        seq_ids = sorted(p.name for p in tum_root.iterdir() if p.is_dir())
    elif args.seq_id:
        seq_ids = [args.seq_id]
    else:
        parser.error("Provide --seq-id or --all-seqs")
        return

    for sid in seq_ids:
        try:
            process_seq(sid, tum_root, output_root, args.frame_stride, args.max_frames,
                        init_ply_only=args.init_ply_only)
        except Exception as e:
            print(f"[tum_io] ERROR {sid}: {e}")


if __name__ == "__main__":
    main()

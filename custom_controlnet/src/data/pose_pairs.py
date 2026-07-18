"""
src/data/pose_pairs.py — Phase 3: generate valid (A, B) pose-pair triples.

Validity filter (§2.3):
  - Baseline  (translation) : [0.10 m, 1.00 m]
  - Angle     (view dir.)   : [5°,  15°]
  - Co-vis    (IoU)         : [0.40,  0.70]

Outputs a per-scene pairs.jsonl with one row per accepted (A, B) pair
plus (baseline_m, angle_deg, covis) diagnostics.

Usage:
    python -m src.data.pose_pairs \\
        --scene-id <id> \\
        --scenes-root data/scenes \\
        [--max-pairs 1000]
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ── Geometry helpers ──────────────────────────────────────────────────────────

def camera_center(w2c: np.ndarray) -> np.ndarray:
    """Extract camera center from world-to-camera matrix (3,)."""
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return -R.T @ t


def camera_forward(w2c: np.ndarray) -> np.ndarray:
    """Forward direction of camera in world coords (z-axis of c2w)."""
    R = w2c[:3, :3]
    return R.T[:, 2]  # third column of c2w = forward


def baseline_meters(w2c_A: np.ndarray, w2c_B: np.ndarray) -> float:
    cA = camera_center(w2c_A)
    cB = camera_center(w2c_B)
    return float(np.linalg.norm(cA - cB))


def view_angle_deg(w2c_A: np.ndarray, w2c_B: np.ndarray) -> float:
    fA = camera_forward(w2c_A)
    fB = camera_forward(w2c_B)
    cos_a = np.clip(np.dot(fA, fB) / (np.linalg.norm(fA) * np.linalg.norm(fB) + 1e-8), -1, 1)
    return float(math.degrees(math.acos(cos_a)))


# ── Co-visibility computation ─────────────────────────────────────────────────

def compute_covisibility(
    pts_world: np.ndarray,   # (N, 3) — subsampled Gaussian centers
    w2c_A: np.ndarray,       # (4, 4)
    w2c_B: np.ndarray,       # (4, 4)
    K_A: np.ndarray,         # (3, 3)
    K_B: np.ndarray,         # (3, 3)
    depth_A: np.ndarray,     # (H, W) float32 — rendered depth at pose A
    depth_B: np.ndarray,     # (H, W) float32 — rendered depth at pose B
    img_size: int = 512,
    depth_tol: float = 0.10,
) -> float:
    """
    IoU of visible point sets in camera A and B.
    A point is visible in camera X if:
      (1) it projects inside the 512×512 frustum,
      (2) its projected depth <= rendered_depth[pixel] * (1 + depth_tol).
    """
    N = len(pts_world)
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.concatenate([pts_world, ones], axis=1)  # N×4

    def visible_in_cam(w2c, K, depth_map):
        pts_cam = (w2c @ pts_h.T).T[:, :3]  # N×3
        z = pts_cam[:, 2]
        in_front = z > 0.0

        px = np.clip(K[0, 0] * pts_cam[:, 0] / (z + 1e-8) + K[0, 2], -1e6, 1e6).astype(np.int32)
        py = np.clip(K[1, 1] * pts_cam[:, 1] / (z + 1e-8) + K[1, 2], -1e6, 1e6).astype(np.int32)

        in_frustum = in_front & (px >= 0) & (px < img_size) & (py >= 0) & (py < img_size)

        # Depth consistency
        depth_consistent = np.zeros(N, dtype=bool)
        if depth_map is not None and in_frustum.any():
            valid_idx = np.where(in_frustum)[0]
            px_v = np.clip(px[valid_idx], 0, img_size - 1)
            py_v = np.clip(py[valid_idx], 0, img_size - 1)
            rendered_d = depth_map[py_v, px_v]
            projected_z = z[valid_idx]
            depth_consistent[valid_idx] = projected_z <= rendered_d * (1.0 + depth_tol)
        elif in_frustum.any():
            depth_consistent = in_frustum.copy()

        return in_frustum & depth_consistent

    vis_A = visible_in_cam(w2c_A, K_A, depth_A)
    vis_B = visible_in_cam(w2c_B, K_B, depth_B)

    intersection = (vis_A & vis_B).sum()
    union = (vis_A | vis_B).sum()
    return float(intersection / (union + 1e-8))


# ── Pair sampling ─────────────────────────────────────────────────────────────

def load_scene_cameras(scene_dir: Path) -> list[dict]:
    cameras_path = scene_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.json not found at {cameras_path}")
    with open(cameras_path) as f:
        return json.load(f)


def load_depth(scene_dir: Path, frame_id: str) -> np.ndarray | None:
    depth_path = scene_dir / "depth_512" / f"{frame_id}.npy"
    if depth_path.exists():
        return np.load(str(depth_path))
    return None


def _load_gaussian_centers_from_ply(ply_path: Path, max_points: int) -> np.ndarray | None:
    """Load Gaussian means from PLY (3DGS pipeline). Returns None on failure."""
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(ply_path))
        v = ply["vertex"]
        pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float64)
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        return pts
    except Exception:
        return None


def _unproject_depth_to_points(
    cameras: list[dict],
    scene_dir: Path,
    max_points: int,
    num_ref_frames: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Unproject sensor depth maps to a 3-D point cloud for co-visibility estimation.
    Used when no Gaussian .ply is present (ScanNet sensor-depth pipeline).
    """
    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(len(cameras), min(num_ref_frames, len(cameras)), replace=False)
    pts_per_frame = max(1, max_points // num_ref_frames)

    all_pts: list[np.ndarray] = []
    for ci in ref_idx:
        cam      = cameras[ci]
        depth    = load_depth(scene_dir, cam["frame_id"])
        if depth is None:
            continue

        K   = np.array(cam["K_512"], dtype=np.float64)
        c2w = np.linalg.inv(np.array(cam["w2c"], dtype=np.float64))

        ys, xs = np.where(depth > 0)
        if len(ys) == 0:
            continue

        n = min(pts_per_frame, len(ys))
        sel = rng.choice(len(ys), n, replace=False)
        ys, xs = ys[sel], xs[sel]
        zs = depth[ys, xs].astype(np.float64)

        # Unproject to camera space
        pts_cam = np.stack([
            (xs - K[0, 2]) * zs / K[0, 0],
            (ys - K[1, 2]) * zs / K[1, 1],
            zs,
        ], axis=-1)   # N×3

        # Transform to world space
        ones    = np.ones((len(pts_cam), 1))
        pts_h   = np.concatenate([pts_cam, ones], axis=-1)
        pts_w   = (c2w @ pts_h.T).T[:, :3]
        all_pts.append(pts_w)

    if not all_pts:
        return np.random.default_rng(seed).normal(size=(min(1000, max_points), 3)) * 0.5

    pts = np.concatenate(all_pts, axis=0)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts


def get_scene_points(
    scene_dir: Path,
    cameras: list[dict],
    max_points: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """
    Return world-space 3-D points for co-visibility estimation.
    Tries Gaussian PLY first (3DGS pipeline); falls back to depth unprojection
    (ScanNet sensor-depth pipeline).
    """
    ply_path = scene_dir / "gaussians" / "point_cloud.ply"
    pts = _load_gaussian_centers_from_ply(ply_path, max_points)
    if pts is not None:
        return pts
    return _unproject_depth_to_points(cameras, scene_dir, max_points, seed=seed)


def generate_pairs(
    scene_dir: Path,
    max_pairs: int = 1000,
    baseline_min: float = 0.10,
    baseline_max: float = 1.00,
    angle_min: float = 5.0,
    angle_max: float = 15.0,
    covis_min: float = 0.40,
    covis_max: float = 0.70,
    max_pts: int = 50000,
    depth_tol: float = 0.10,
    seed: int = 42,
) -> list[dict]:
    """
    Generate all valid pose pairs for a scene and return up to max_pairs
    stratified by (baseline, angle, covis) to avoid bias toward easy pairs.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    cameras = load_scene_cameras(scene_dir)
    N = len(cameras)
    print(f"[pose_pairs] {N} cameras, enumerating pairs …")

    pts_world = get_scene_points(scene_dir, cameras, max_pts, seed=seed)

    # Pre-load depths (None if not rendered yet)
    depths = {cam["frame_id"]: load_depth(scene_dir, cam["frame_id"]) for cam in cameras}

    # Pre-compute camera arrays
    w2cs = {cam["frame_id"]: np.array(cam["w2c"], dtype=np.float64) for cam in cameras}
    Ks   = {cam["frame_id"]: np.array(cam["K_512"], dtype=np.float64) for cam in cameras}

    valid_pairs = []

    for i in tqdm(range(N), desc="Pair sampling"):
        for j in range(i + 1, N):
            cam_A = cameras[i]
            cam_B = cameras[j]
            id_A, id_B = cam_A["frame_id"], cam_B["frame_id"]

            w2c_A = w2cs[id_A]
            w2c_B = w2cs[id_B]

            # Fast geometric filters first
            bl = baseline_meters(w2c_A, w2c_B)
            if not (baseline_min <= bl <= baseline_max):
                continue

            ang = view_angle_deg(w2c_A, w2c_B)
            if not (angle_min <= ang <= angle_max):
                continue

            # Expensive covis check last
            covis = compute_covisibility(
                pts_world,
                w2c_A, w2c_B,
                Ks[id_A], Ks[id_B],
                depths[id_A], depths[id_B],
                depth_tol=depth_tol,
            )
            if not (covis_min <= covis <= covis_max):
                continue

            valid_pairs.append({
                "frame_id_A": id_A,
                "frame_id_B": id_B,
                "baseline_m": round(bl, 4),
                "angle_deg":  round(ang, 2),
                "covis":      round(covis, 4),
            })

    print(f"[pose_pairs] {len(valid_pairs)} valid pairs found")

    if len(valid_pairs) <= max_pairs:
        return valid_pairs

    # Stratified sampling across (baseline, angle, covis) ranges
    selected = _stratified_sample(valid_pairs, max_pairs, rng)
    print(f"[pose_pairs] Downsampled to {len(selected)} pairs (stratified)")
    return selected


def _stratified_sample(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """
    Bin pairs into a 3D grid of (baseline, angle, covis) cells and
    sample proportionally so no cell dominates.
    """
    # Define bin edges
    bl_bins  = [0.10, 0.25, 0.50, 0.75, 1.00]
    ang_bins = [10, 20, 35, 50, 60]
    cv_bins  = [0.30, 0.40, 0.55, 0.70]

    from collections import defaultdict
    cells: dict[tuple, list] = defaultdict(list)
    for p in pairs:
        bi = np.digitize(p["baseline_m"], bl_bins) - 1
        ai = np.digitize(p["angle_deg"],  ang_bins) - 1
        ci = np.digitize(p["covis"],      cv_bins)  - 1
        cells[(bi, ai, ci)].append(p)

    # Round-robin across non-empty cells until we have n samples
    selected = []
    cell_lists = [lst for lst in cells.values() if lst]
    for lst in cell_lists:
        rng.shuffle(lst)

    i = 0
    while len(selected) < n and any(cell_lists):
        cell = cell_lists[i % len(cell_lists)]
        if cell:
            selected.append(cell.pop())
        i += 1

    return selected


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_scene(args_tuple):
    scene_dir, kwargs = args_tuple
    force = kwargs.pop("force", False)
    out_path = scene_dir / "pairs.jsonl"
    if out_path.exists() and not force:
        return scene_dir.name, "SKIP"
    try:
        pairs = generate_pairs(scene_dir=scene_dir, **kwargs)
        with open(out_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        return scene_dir.name, len(pairs)
    except Exception as e:
        return scene_dir.name, f"ERROR: {e}"


def main():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Generate pose pairs for a scene")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--scene-id",    help="Single scene ID")
    grp.add_argument("--all-scenes",  action="store_true")
    parser.add_argument("--scenes-root", default="data/scenes")
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--baseline-min", type=float, default=0.10)
    parser.add_argument("--baseline-max", type=float, default=1.00)
    parser.add_argument("--angle-min",    type=float, default=5.0)
    parser.add_argument("--angle-max",    type=float, default=15.0)
    parser.add_argument("--covis-min",    type=float, default=0.40)
    parser.add_argument("--covis-max",    type=float, default=0.70)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing pairs.jsonl (default: skip)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--workers",      type=int,
                        default=min(8, multiprocessing.cpu_count()),
                        help="Parallel workers (default: min(8, cpu_count))")
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    kwargs = dict(
        max_pairs=args.max_pairs,
        baseline_min=args.baseline_min, baseline_max=args.baseline_max,
        angle_min=args.angle_min,       angle_max=args.angle_max,
        covis_min=args.covis_min,       covis_max=args.covis_max,
        seed=args.seed,
        force=args.force,
    )

    if args.scene_id:
        scene_dir = scenes_root / args.scene_id
        out_path = scene_dir / "pairs.jsonl"
        gen_kwargs = {k: v for k, v in kwargs.items() if k != "force"}
        pairs = generate_pairs(scene_dir=scene_dir, **gen_kwargs)
        with open(out_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[pose_pairs] Saved {len(pairs)} pairs → {out_path}")
        return

    # --all-scenes: only process scenes that have depth_512/ (render_depth done)
    scene_dirs = sorted(
        d for d in scenes_root.iterdir()
        if d.is_dir() and (d / "depth_512").exists()
    )
    print(f"[pose_pairs] {len(scene_dirs)} scenes to process with {args.workers} workers")

    tasks = [(d, kwargs) for d in scene_dirs]
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_scene, t): t[0].name for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            scene_id, result = fut.result()
            print(f"[{i}/{len(tasks)}] {scene_id}  {result}")
            if isinstance(result, str) and result.startswith("ERROR"):
                failed.append(scene_id)

    print(f"\n[pose_pairs] Done. Failed: {len(failed)}")
    if failed:
        print("  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()

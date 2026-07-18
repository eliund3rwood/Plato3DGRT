"""
src/data/make_manifest.py — Build train/val/test JSONL manifests.

Split is by scene (not by pair) to prevent scene-leakage.
Each line in a manifest is one (I_A, D_B, I_B) triple:
  {
    "scene_id":    str,
    "frame_id_A":  str,
    "frame_id_B":  str,
    "I_A_path":    str,   # 512×512 RGB at pose A
    "I_B_path":    str,   # 512×512 RGB at pose B
    "D_B_path":    str,   # 3DGS-rendered depth at pose B (.npy)
    "alpha_B_path":str,   # accumulation mask at pose B (.npy)
    "depth_near":  float, # for load-time normalization
    "depth_far":   float,
    "baseline_m":  float,
    "angle_deg":   float,
    "covis":       float,
  }

Usage:
    # Random scene split (default):
    python -m src.data.make_manifest \\
        --scenes-root data/scenes \\
        --output-dir  data/manifests \\
        --train-frac  0.8 \\
        --val-frac    0.1

    # Explicit eval holdout (recommended with TUM train/eval sets):
    python -m src.data.make_manifest \\
        --scenes-root data/scenes \\
        --output-dir  data/manifests \\
        --eval-scenes freiburg1_xyz freiburg3_sitting_static freiburg3_sitting_xyz
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def discover_scenes(scenes_root: Path) -> list[str]:
    """Return sorted list of scene_ids that have cameras.json + pairs.jsonl."""
    scene_ids = []
    for p in sorted(scenes_root.iterdir()):
        if p.is_dir() and (p / "cameras.json").exists() and (p / "pairs.jsonl").exists():
            scene_ids.append(p.name)
    return scene_ids


def build_scene_manifest(
    scene_id: str,
    scenes_root: Path,
) -> list[dict]:
    """
    Build manifest rows for a single scene.
    Returns list of dicts, one per valid (A, B) pair.
    """
    scene_dir = scenes_root / scene_id

    # Load cameras → frame_id → paths (handle both list and dict formats)
    with open(scene_dir / "cameras.json") as f:
        cameras = json.load(f)
    if isinstance(cameras, dict):
        cameras = [{"frame_id": k, **v} for k, v in cameras.items()]
    cam_map = {cam["frame_id"]: cam for cam in cameras}

    # Load depth metadata (near/far per frame)
    depth_meta_path = scene_dir / "depth_meta.json"
    if depth_meta_path.exists():
        with open(depth_meta_path) as f:
            depth_meta = json.load(f)
    else:
        depth_meta = {}

    rows = []
    pairs_path = scene_dir / "pairs.jsonl"
    with open(pairs_path) as f:
        for line in f:
            pair = json.loads(line.strip())
            id_A = pair["frame_id_A"]
            id_B = pair["frame_id_B"]

            if id_A not in cam_map or id_B not in cam_map:
                continue

            I_A_path = scene_dir / "rgb_512" / f"{id_A}.png"
            I_B_path = scene_dir / "rgb_512" / f"{id_B}.png"
            D_B_path = scene_dir / "depth_512" / f"{id_B}.npy"
            alpha_B_path = scene_dir / "alpha_512" / f"{id_B}.npy"

            if not I_A_path.exists() or not I_B_path.exists() or not D_B_path.exists():
                continue

            dm = depth_meta.get(id_B, {})
            rows.append({
                "scene_id":     scene_id,
                "frame_id_A":   id_A,
                "frame_id_B":   id_B,
                "I_A_path":     str(I_A_path),
                "I_B_path":     str(I_B_path),
                "D_B_path":     str(D_B_path),
                "alpha_B_path": str(alpha_B_path) if alpha_B_path.exists() else None,
                "depth_near":   dm.get("near", 0.01),
                "depth_far":    dm.get("far",  10.0),
                "baseline_m":   pair["baseline_m"],
                "angle_deg":    pair["angle_deg"],
                "covis":        pair["covis"],
            })

    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_manifests(
    scenes_root: Path,
    output_dir: Path,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
    eval_scenes: list[str] | None = None,
) -> dict[str, int]:
    """
    Discover all scenes, build per-scene rows, split by scene, write manifests.

    If eval_scenes is given, those scenes are pinned to 'test'; the rest are
    randomly split into train (train_frac) and val (val_frac). This is the
    recommended mode when using the TUM explicit train/eval scene lists.

    Returns dict of split_name → row count.
    """
    rng = random.Random(seed)
    scene_ids = discover_scenes(scenes_root)
    if not scene_ids:
        raise RuntimeError(f"No valid scenes found in {scenes_root}. "
                           "Run gsplat_fit, render_depth, and pose_pairs first.")

    print(f"[make_manifest] Found {len(scene_ids)} scenes")

    if eval_scenes:
        eval_set   = set(eval_scenes)
        test_ids   = [s for s in scene_ids if s in eval_set]
        train_pool = [s for s in scene_ids if s not in eval_set]
        missing    = eval_set - set(scene_ids)
        if missing:
            print(f"  [warn] eval_scenes not found in scenes_root: {missing}")
        rng.shuffle(train_pool)
        n       = len(train_pool)
        n_train = max(1, int(n * (train_frac / (train_frac + val_frac))))
        splits  = {
            "train": train_pool[:n_train],
            "val":   train_pool[n_train:],
            "test":  test_ids,
        }
        print(f"  Explicit eval split: {len(splits['train'])} train, "
              f"{len(splits['val'])} val, {len(splits['test'])} test (pinned)")
    else:
        rng.shuffle(scene_ids)
        n = len(scene_ids)
        n_train = max(1, int(n * train_frac))
        n_val   = max(1, int(n * val_frac))
        splits = {
            "train": scene_ids[:n_train],
            "val":   scene_ids[n_train:n_train + n_val],
            "test":  scene_ids[n_train + n_val:],
        }

    counts = {}
    for split_name, split_scenes in splits.items():
        all_rows = []
        for sid in tqdm(split_scenes, desc=f"Building {split_name}"):
            try:
                rows = build_scene_manifest(sid, scenes_root)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  [warn] Scene {sid} skipped: {e}")

        rng.shuffle(all_rows)
        out_path = output_dir / f"{split_name}.jsonl"
        write_manifest(all_rows, out_path)
        counts[split_name] = len(all_rows)
        print(f"  {split_name}: {len(split_scenes)} scenes, {len(all_rows)} triples → {out_path}")

    # Scene histogram
    _log_metric_histogram(
        [r for split in splits.values() for r in split],
        scenes_root,
        output_dir,
    )

    return counts


def _log_metric_histogram(scene_ids: list[str], scenes_root: Path, output_dir: Path) -> None:
    """Log (baseline, angle, covis) distribution across all pairs."""
    baselines, angles, covis_vals = [], [], []
    for sid in scene_ids:
        pairs_path = scenes_root / sid / "pairs.jsonl"
        if not pairs_path.exists():
            continue
        with open(pairs_path) as f:
            for line in f:
                p = json.loads(line)
                baselines.append(p["baseline_m"])
                angles.append(p["angle_deg"])
                covis_vals.append(p["covis"])

    import math
    def hist(vals, bins):
        counts = [0] * len(bins)
        for v in vals:
            for i, b in enumerate(bins):
                if v <= b:
                    counts[i] += 1
                    break
        return dict(zip([str(b) for b in bins], counts))

    summary = {
        "total_pairs": len(baselines),
        "baseline_hist": hist(baselines, [0.2, 0.4, 0.6, 0.8, 1.0]),
        "angle_hist":    hist(angles,    [20, 30, 45, 55, 60]),
        "covis_hist":    hist(covis_vals,[0.35, 0.45, 0.55, 0.65, 0.70]),
    }
    out = output_dir / "pair_histogram.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[make_manifest] Pair histogram → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build train/val/test manifests")
    parser.add_argument("--scenes-root",  default="data/scenes")
    parser.add_argument("--output-dir",   default="data/manifests")
    parser.add_argument("--train-frac",   type=float, default=0.80)
    parser.add_argument("--val-frac",     type=float, default=0.10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--eval-scenes",  nargs="+",  default=None,
                        help="Scene IDs pinned to test split (recommended). "
                             "Rest are randomly split into train/val.")
    args = parser.parse_args()

    counts = build_manifests(
        scenes_root=Path(args.scenes_root),
        output_dir=Path(args.output_dir),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        eval_scenes=args.eval_scenes,
    )
    print(f"[make_manifest] Done: {counts}")


if __name__ == "__main__":
    main()

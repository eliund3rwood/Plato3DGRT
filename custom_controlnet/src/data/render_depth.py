"""
src/data/render_depth.py — Phase 2: render 512×512 3DGS depth maps.

Depth comes from the 3DGS rasterizer (alpha-composited Gaussian depths),
NOT from sensor/mesh data, so 3DGS artifacts are baked in.

Writes per-frame:
  data/scenes/<scene_id>/depth_512/<frame_id>.npy   (float32, raw metric depth)
  data/scenes/<scene_id>/alpha_512/<frame_id>.npy   (float32 [0,1])
  data/scenes/<scene_id>/depth_meta.json            (near/far percentiles per frame)

Usage:
    python -m src.data.render_depth \\
        --scene-id <id> \\
        --scenes-root data/scenes \\
        --verify-alignment   # optional: save D_B / I_B overlay PNGs
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


DEPTH_NEAR_PCT = 2    # 2nd percentile → near
DEPTH_FAR_PCT  = 98   # 98th percentile → far


# ── PLY loading ───────────────────────────────────────────────────────────────

def load_gaussians_from_ply(ply_path: Path) -> dict:
    """
    Load Gaussian parameters from the PLY saved by gsplat_fit.py.
    Returns dict of torch tensors on CPU.
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("plyfile required: pip install plyfile")

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]

    means = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    opacities = v["opacity"].astype(np.float32)

    # Colors from DC SH term
    colors = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)

    return {
        "means":     torch.from_numpy(means),
        "quats":     torch.from_numpy(quats),
        "scales":    torch.from_numpy(scales),
        "opacities": torch.from_numpy(opacities),
        "colors":    torch.from_numpy(colors),
    }


# ── Depth rendering ───────────────────────────────────────────────────────────

def render_depth_gsplat(
    gaussians: dict,
    viewmat: torch.Tensor,    # 1×4×4  world-to-camera
    K: torch.Tensor,          # 1×3×3  intrinsics
    width: int,
    height: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rasterize 3DGS and return (depth_map, alpha_map) as float32 (H, W) arrays.
    Depth is in world metric units (metres). Alpha is accumulation in [0, 1].

    Uses gsplat.rasterization with render_mode that returns depth.
    """
    try:
        from gsplat import rasterization
    except ImportError:
        raise ImportError("gsplat not installed: pip install gsplat")

    means    = gaussians["means"].to(device)
    quats    = gaussians["quats"].to(device)
    scales   = gaussians["scales"].to(device)
    opacities = gaussians["opacities"].to(device)
    colors   = gaussians["colors"].to(device)  # N×3 (DC SH only)

    quats_norm = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)

    with torch.no_grad():
        # render_mode "RGB+D" returns rgba+depth in the last channel
        renders, alphas, _ = rasterization(
            means, quats_norm, scales, opacities, colors,
            viewmat, K,
            width=width, height=height,
            render_mode="RGB+D",
        )
        # renders: 1 × H × W × 4  (RGB + depth)
        # alphas:  1 × H × W × 1
        depth_map = renders[0, :, :, 3].cpu().numpy().astype(np.float32)
        alpha_map = alphas[0, :, :, 0].cpu().numpy().astype(np.float32)

    assert depth_map.shape == (height, width), \
        f"Depth shape {depth_map.shape} != ({height}, {width})"
    return depth_map, alpha_map


# ── Main render loop ──────────────────────────────────────────────────────────

def render_scene_depths(
    scene_dir: Path,
    output_dir: Path,
    verify_alignment: bool = False,
    frame_ids: list[str] | None = None,
) -> dict:
    """
    Render 512×512 depth for cameras in cameras.json.
    If frame_ids is given, only those frames are rendered (used for synthetic
    scenes where we only need the B-views from pairs.jsonl).
    Returns per-frame near/far metadata.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cameras
    cameras_path = scene_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.json not found: {cameras_path}. Run gsplat_fit first.")
    with open(cameras_path) as f:
        cameras = json.load(f)

    # Support both list-of-dicts (scannet_io) and dict-of-dicts (blender format)
    if isinstance(cameras, dict):
        cameras = [{"frame_id": k, "w2c": v["w2c"], "K_512": v["K_512"],
                    "width": 512, "height": 512} for k, v in cameras.items()]

    # Filter to requested frame IDs only
    if frame_ids is not None:
        wanted = set(frame_ids)
        cameras = [c for c in cameras if c["frame_id"] in wanted]

    # Load Gaussians
    ply_path = scene_dir / "gaussians" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"point_cloud.ply not found: {ply_path}. Run gsplat_fit first.")
    gaussians = load_gaussians_from_ply(ply_path)
    print(f"[render_depth] Loaded {len(gaussians['means'])} Gaussians from {ply_path}")

    depth_dir = output_dir / "depth_512"
    alpha_dir = output_dir / "alpha_512"
    depth_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    if verify_alignment:
        align_dir = output_dir / "alignment_check"
        align_dir.mkdir(parents=True, exist_ok=True)

    depth_meta = {}

    for cam in tqdm(cameras, desc="Render depth"):
        frame_id = cam["frame_id"]
        w2c = np.array(cam["w2c"], dtype=np.float32)
        K_512 = np.array(cam["K_512"], dtype=np.float32)
        width = int(cam["width"])
        height = int(cam["height"])

        assert width == 512 and height == 512, \
            f"Camera {frame_id} is {width}×{height}, expected 512×512"

        viewmat = torch.from_numpy(w2c).unsqueeze(0).to(device)  # 1×4×4
        K_t = torch.from_numpy(K_512).unsqueeze(0).to(device)    # 1×3×3

        depth_map, alpha_map = render_depth_gsplat(
            gaussians, viewmat, K_t, width, height, device
        )

        # Record near/far percentiles (ignore zero-depth/invalid regions)
        valid = (depth_map > 0) & (alpha_map > 0.1)
        if valid.any():
            near = float(np.percentile(depth_map[valid], DEPTH_NEAR_PCT))
            far  = float(np.percentile(depth_map[valid], DEPTH_FAR_PCT))
        else:
            near, far = 0.01, 10.0

        depth_meta[frame_id] = {"near": near, "far": far}

        # Save as float32 .npy (do NOT normalize — normalize at load time)
        np.save(str(depth_dir / f"{frame_id}.npy"), depth_map)
        np.save(str(alpha_dir / f"{frame_id}.npy"), alpha_map)

        # Optional pixel-alignment verification overlay
        if verify_alignment:
            rgb_path = scene_dir / "rgb_512" / f"{frame_id}.png"
            if rgb_path.exists():
                rgb = cv2.imread(str(rgb_path))
                _save_alignment_overlay(
                    rgb, depth_map, alpha_map,
                    near, far,
                    align_dir / f"{frame_id}_align.png"
                )

    # Save per-frame near/far metadata
    meta_path = output_dir / "depth_meta.json"
    with open(meta_path, "w") as f:
        json.dump(depth_meta, f, indent=2)

    print(f"[render_depth] Saved depth + alpha for {len(cameras)} frames → {output_dir}")
    return depth_meta


def _save_alignment_overlay(
    rgb: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray,
    near: float,
    far: float,
    out_path: Path,
) -> None:
    """Save a side-by-side RGB / depth-colourmap overlay for alignment inspection."""
    depth_norm = np.clip((depth - near) / (far - near + 1e-8), 0.0, 1.0)
    depth_vis = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    # Edge map from depth discontinuities
    edges = cv2.Canny((depth_norm * 255).astype(np.uint8), 50, 150)
    edge_overlay = rgb.copy()
    edge_overlay[edges > 0] = [0, 255, 0]  # green edges

    panel = np.concatenate([rgb, depth_vis, edge_overlay], axis=1)
    cv2.imwrite(str(out_path), panel)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Render 512×512 depth from 3DGS")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--scenes-root", default="data/scenes")
    parser.add_argument("--output-root", default=None, help="Defaults to --scenes-root/<scene-id>")
    parser.add_argument("--verify-alignment", action="store_true")
    parser.add_argument("--frame-ids", nargs="+", default=None,
                        help="Only render these frame IDs (e.g. frame_0010 frame_0016). "
                             "Defaults to all frames in cameras.json.")
    args = parser.parse_args()

    scene_dir  = Path(args.scenes_root) / args.scene_id
    output_dir = Path(args.output_root or args.scenes_root) / args.scene_id

    render_scene_depths(
        scene_dir=scene_dir,
        output_dir=output_dir,
        verify_alignment=args.verify_alignment,
        frame_ids=args.frame_ids,
    )


if __name__ == "__main__":
    main()

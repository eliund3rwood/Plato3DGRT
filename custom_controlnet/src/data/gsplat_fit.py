"""
src/data/gsplat_fit.py — Phase 1: fit a 3DGS model to an indoor scene.

Two dataset modes (--dataset flag):

  scannetpp (default)
    Reads ScanNet++ DSLR nerfstudio transforms.json; undistorts fisheye images.

  canonical
    Reads cameras.json + rgb_512/ produced by scannet_io.py or replica_io.py.
    No undistortion (already pinhole + cropped to 512×512).
    Use this for ScanNet, Replica, or any other dataset once preprocessed.

Outputs (both modes):
  data/scenes/<scene_id>/gaussians/point_cloud.ply
  data/scenes/<scene_id>/cameras.json       (overwrites scannet_io output)
  data/scenes/<scene_id>/rgb_512/           (written during loading for scannetpp)
  data/scenes/<scene_id>/gaussians/eval_metrics.json

Why 3DGS depth?  The downstream task uses depth maps rendered from noisy 3DGS
reconstructions. Training on the same kind of depth (Gaussian artifacts and all)
ensures the model learns to handle what it will see at inference.

Usage:
    # ScanNet (after scannet_io.py):
    python -m src.data.gsplat_fit \\
        --scene-id scene0000_00 \\
        --dataset canonical \\
        --scenes-root data/scenes \\
        --max-steps 30000

    # ScanNet++ (direct from raw data):
    python -m src.data.gsplat_fit \\
        --scene-id <id> \\
        --dataset scannetpp \\
        --data-root /path/to/scannetpp/data \\
        --output-root data/scenes \\
        --max-steps 30000

Cluster-only: needs CUDA + gsplat.
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ── ScanNet++ metadata loading ────────────────────────────────────────────────

def load_scannetpp_transforms(scene_dir: Path) -> dict:
    """
    Load the nerfstudio-format transforms.json from a ScanNet++ DSLR scene.
    Returns a dict with keys: camera_model, fl_x, fl_y, cx, cy,
    k1..k4 (fisheye distortion), frames (list of {file_path, transform_matrix}).
    """
    tf_path = scene_dir / "dslr" / "nerfstudio" / "transforms.json"
    if not tf_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {tf_path}")
    with open(tf_path) as f:
        return json.load(f)


def build_camera_intrinsics(meta: dict, orig_w: int, orig_h: int) -> np.ndarray:
    """Build 3×3 intrinsics matrix K from nerfstudio metadata."""
    K = np.array([
        [meta["fl_x"],          0.0, meta["cx"]],
        [          0.0, meta["fl_y"], meta["cy"]],
        [          0.0,          0.0,         1.0],
    ], dtype=np.float64)
    return K


def undistort_fisheye(img: np.ndarray, K: np.ndarray, D: np.ndarray,
                      target_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Undistort an OpenCV fisheye (KB4) image and return the undistorted image
    plus the new intrinsics K_new scaled to target_size (square crop).

    Returns:
        undist_img: (H, W, 3) uint8, pinhole-undistorted
        K_512:      3×3 intrinsics for the 512×512 crop
    """
    h, w = img.shape[:2]

    # Build the target pinhole intrinsics centered on the image
    K_new = K.copy()
    K_new[0, 2] = w / 2.0
    K_new[1, 2] = h / 2.0

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w, h), cv2.CV_32FC1
    )
    undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # Square center crop then resize to target_size×target_size
    short_side = min(h, w)
    y0 = (h - short_side) // 2
    x0 = (w - short_side) // 2
    cropped = undist[y0:y0 + short_side, x0:x0 + short_side]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Update K to reflect crop+resize
    scale = target_size / short_side
    K_512 = K_new.copy()
    K_512[0, 0] *= scale
    K_512[1, 1] *= scale
    K_512[0, 2] = (K_new[0, 2] - x0) * scale
    K_512[1, 2] = (K_new[1, 2] - y0) * scale

    return resized, K_512


# ── Canonical loader (cameras.json + rgb_512/) ───────────────────────────────

def load_canonical_scene(scene_dir: Path, target_size: int = 512):
    """
    Load scene data from the canonical pipeline format produced by
    scannet_io.py or replica_io.py.

    Returns:
        images_512  : list of (H, W, 3) uint8 arrays
        c2w_list    : list of (4, 4) float64 arrays
        K_512_list  : list of (3, 3) float64 arrays
        frame_ids   : list of str
    """
    cameras_path = scene_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(
            f"cameras.json not found at {cameras_path}. "
            "Run scannet_io.py or replica_io.py first."
        )
    with open(cameras_path) as f:
        cameras = json.load(f)

    # Support both list-of-dicts (scannet_io format) and dict-of-dicts (blender format)
    if isinstance(cameras, dict):
        cameras = [{"frame_id": k, "w2c": v["w2c"], "K_512": v["K_512"],
                    "width": 512, "height": 512} for k, v in cameras.items()]

    images_512, c2w_list, K_512_list, frame_ids = [], [], [], []

    for cam in tqdm(cameras, desc="Load frames"):
        fid = cam["frame_id"]
        img_path = scene_dir / "rgb_512" / f"{fid}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [warn] Cannot read {img_path}, skipping")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape != (target_size, target_size, 3):
            img_rgb = cv2.resize(img_rgb, (target_size, target_size))

        w2c = np.array(cam["w2c"], dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        K_512 = np.array(cam["K_512"], dtype=np.float64)

        images_512.append(img_rgb)
        c2w_list.append(c2w)
        K_512_list.append(K_512)
        frame_ids.append(fid)

    if not images_512:
        raise RuntimeError(f"No valid frames loaded from {scene_dir}")

    return images_512, c2w_list, K_512_list, frame_ids


# ── Point cloud loading (SfM / laser) ────────────────────────────────────────

def load_init_points(scene_dir: Path) -> np.ndarray:
    """
    Load initial point cloud for Gaussian init.
    Tries: (1) canonical init_pointcloud.ply (from tum_io sensor-depth unproject),
           (2) nerfstudio sparse_pc.ply, (3) COLMAP points3D.bin, (4) random fallback.
    Returns (N, 3) float32 array.
    """
    # Canonical format: sensor-depth point cloud built by tum_io / scannet_io
    canonical_ply = scene_dir / "init_pointcloud.ply"
    if canonical_ply.exists():
        pts = _load_ply_xyz(canonical_ply)
        print(f"  [gsplat_fit] Loaded {len(pts)} init points from {canonical_ply.name}")
        return pts

    canonical_npy = scene_dir / "init_pointcloud.npy"
    if canonical_npy.exists():
        pts = np.load(str(canonical_npy)).astype(np.float32)
        print(f"  [gsplat_fit] Loaded {len(pts)} init points from {canonical_npy.name}")
        return pts

    # Try nerfstudio-exported points
    pts_path = scene_dir / "dslr" / "nerfstudio" / "sparse_pc.ply"
    if pts_path.exists():
        return _load_ply_xyz(pts_path)

    # Try colmap
    colmap_pts = scene_dir / "dslr" / "colmap" / "sparse" / "0" / "points3D.bin"
    if colmap_pts.exists():
        return _load_colmap_pts3d(colmap_pts)

    # Fallback: random points in [-1, 1]^3
    print("  [gsplat_fit] WARNING: No init point cloud found — using 10k random points (poor quality)")
    return np.random.randn(10000, 3).astype(np.float32) * 0.5


def _load_ply_xyz(path: Path) -> np.ndarray:
    """Minimal PLY loader — extracts x, y, z vertex properties."""
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)


def _load_colmap_pts3d(path: Path) -> np.ndarray:
    """Read COLMAP binary points3D.bin and return (N, 3) xyz."""
    import struct
    pts = []
    with open(path, "rb") as f:
        n_pts = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_pts):
            f.read(8)  # point3D_id
            xyz = struct.unpack("<3d", f.read(24))
            pts.append(xyz)
            f.read(3 + 8 + 8)  # rgb, error, track_length
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)  # track elements
    return np.array(pts, dtype=np.float32)


# ── SH color evaluation ──────────────────────────────────────────────────────

def _eval_sh_colors(
    sh0: torch.Tensor,
    shN: torch.Tensor,
    means: torch.Tensor,
    viewmat: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate degree-3 SH at the current viewpoint.
    Falls back to DC term if gsplat.spherical_harmonics is unavailable.
    Returns (N, 3) colors in [0, 1].
    """
    try:
        from gsplat import spherical_harmonics as gsplat_sh
        cam_center = torch.linalg.inv(viewmat[0])[:3, 3]
        dirs = torch.nn.functional.normalize(means.detach() - cam_center, dim=-1)
        sh_all = torch.cat([sh0, shN], dim=1)  # (N, 16, 3)
        return (gsplat_sh(3, dirs, sh_all) + 0.5).clamp(0.0, 1.0)
    except Exception:
        return sh0[:, 0, :].clamp(0.0, 1.0)


# ── Gaussian initialization ───────────────────────────────────────────────────

def init_gaussians_from_points(pts: np.ndarray, device: torch.device):
    """
    Initialize Gaussian parameters from a point cloud.
    Returns (means, quats, scales, opacities, sh_coeffs) as leaf tensors.
    """
    N = len(pts)
    means = torch.tensor(pts, dtype=torch.float32, device=device, requires_grad=True)

    # Identity quaternion [w, x, y, z]
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0
    quats = quats.requires_grad_(True)

    # Initial scales: small uniform
    log_scales = torch.full((N, 3), -4.0, device=device, requires_grad=True)

    # Opacities: sigmoid(0.1) ≈ 0.52
    raw_opacities = torch.full((N,), 0.1, device=device, requires_grad=True)

    # SH degree 3: 16 coefficients per channel, 3 channels
    sh_coeffs = torch.zeros(N, 16, 3, device=device, requires_grad=True)
    sh_coeffs.data[:, 0, :] = 0.5  # DC term ≈ grey

    return means, quats, log_scales, raw_opacities, sh_coeffs


# ── Training loop ─────────────────────────────────────────────────────────────

def _load_scannetpp_frames(
    scene_dir: Path,
    output_dir: Path,
    target_size: int,
) -> tuple[list, list, list, list]:
    """
    Load + undistort ScanNet++ fisheye frames.
    Returns (images_512, c2w_list, K_512_list, frame_ids).
    """
    meta = load_scannetpp_transforms(scene_dir)
    camera_model = meta.get("camera_model", "OPENCV_FISHEYE")
    assert camera_model in ("OPENCV_FISHEYE", "OPENCV"), f"Unsupported camera model: {camera_model}"

    D = np.array([meta.get(k, 0.0) for k in ["k1", "k2", "k3", "k4"]], dtype=np.float64)
    frames = meta["frames"]
    img_w = meta.get("w", 1752)
    img_h = meta.get("h", 1168)
    K_orig = build_camera_intrinsics(meta, img_w, img_h)

    print(f"[gsplat_fit] Loading and undistorting {len(frames)} ScanNet++ images …")
    rgb_out_dir = output_dir / "rgb_512"
    rgb_out_dir.mkdir(parents=True, exist_ok=True)

    images_512, c2w_list, K_512_list, frame_ids = [], [], [], []
    for frame in tqdm(frames, desc="Undistort"):
        img_path = scene_dir / "dslr" / frame["file_path"]
        if not img_path.exists():
            img_path = scene_dir / frame["file_path"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [warn] Could not read {img_path}, skipping")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        undist, K_512 = undistort_fisheye(img_rgb, K_orig, D, target_size)
        assert undist.shape == (target_size, target_size, 3)

        fid = Path(frame["file_path"]).stem
        cv2.imwrite(str(rgb_out_dir / f"{fid}.png"), cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

        images_512.append(undist)
        c2w_list.append(np.array(frame["transform_matrix"], dtype=np.float64))
        K_512_list.append(K_512)
        frame_ids.append(fid)

    assert len(images_512) > 0, "No valid images loaded from ScanNet++ scene"
    return images_512, c2w_list, K_512_list, frame_ids


# ── Distributed helpers ───────────────────────────────────────────────────────

def _dist_init(rank: int, world_size: int) -> None:
    import os, torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    # Each spawned process must bind to its own physical GPU.
    # CUDA_VISIBLE_DEVICES from the parent shell can collapse all ranks onto one device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    torch.cuda.set_device(0)  # after restricting visibility, device 0 == physical rank GPU
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _dist_allreduce_grads(splat_params: dict, world_size: int) -> None:
    import torch.distributed as dist
    for p in splat_params.values():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def _dist_broadcast_params(splat_params: dict, optimizers: dict, rank: int) -> None:
    """
    Sync parameter tensors from rank 0 → all ranks.

    Broadcasts the current Gaussian count (single int) from rank 0. When the
    count has changed (densification), creates NEW nn.Parameter objects so that
    any autograd graph from the previous step still references the old objects —
    no in-place storage replacement. Then fills each parameter via NCCL broadcast
    and synchronizes the default CUDA stream before returning.
    """
    import torch.distributed as dist

    device = splat_params["means"].device

    # Step 1: broadcast new N (number of Gaussians) from rank 0 as a scalar tensor.
    # Avoids broadcast_object_list (pickle round-trip) which can mis-deliver on some NCCL builds.
    n_gs = torch.tensor([splat_params["means"].shape[0]], dtype=torch.long, device=device)
    dist.broadcast(n_gs, src=0)
    expected_n = int(n_gs.item())

    # Step 2: for each param, reallocate if N changed, then broadcast data.
    for k in list(splat_params.keys()):
        p = splat_params[k]
        if p.shape[0] != expected_n:
            # Create a NEW Parameter (old object stays alive for any active autograd graphs)
            new_shape = (expected_n,) + p.shape[1:]
            new_param = torch.nn.Parameter(p.data.new_empty(new_shape))
            splat_params[k] = new_param
            if k in optimizers:
                optimizers[k].param_groups[0]["params"][0] = new_param
                optimizers[k].state.clear()
        dist.broadcast(splat_params[k].data, src=0)

    # Step 3: flush NCCL → default CUDA stream so the forward sees complete data.
    torch.cuda.synchronize()


def _train_gsplat_rank(rank: int, world_size: int, kwargs: dict) -> None:
    """Per-rank worker launched by mp.spawn for 8-GPU 3DGS training."""
    _dist_init(rank, world_size)   # sets CUDA_VISIBLE_DEVICES=rank, binds to cuda:0
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    device = torch.device("cuda:0")  # only one GPU visible per process after _dist_init
    train_gsplat(**kwargs, device=device, rank=rank, world_size=world_size)
    import torch.distributed as dist
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────

def train_gsplat(
    scene_dir: Path,
    output_dir: Path,
    dataset: str = "scannetpp",
    max_steps: int = 30000,
    eval_interval: int = 1000,
    psnr_threshold: float = 20.0,
    target_size: int = 512,
    device: torch.device = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """
    Fit a 3DGS model to the scene using gsplat with DefaultStrategy densification.

    dataset: "scannetpp"  — reads from ScanNet++ nerfstudio format
             "canonical"  — reads from cameras.json + rgb_512/ (scannet_io / replica_io output)

    Multi-GPU: set rank/world_size when launched via mp.spawn (--world-size N CLI flag).
    Each rank renders a different random frame per step; gradients are all-reduced;
    rank 0 does the optimizer step, LR schedule, and densification, then broadcasts
    updated params to all ranks.

    Returns eval_metrics dict {step: {psnr, loss}} (rank 0 only; empty on other ranks).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from gsplat import rasterization
        from gsplat.strategy import DefaultStrategy
    except ImportError:
        raise ImportError("gsplat not installed. Run: pip install gsplat")

    # ── Load frames ───────────────────────────────────────────────────────────
    if dataset == "scannetpp":
        images_512, c2w_list, K_512_list, frame_ids = _load_scannetpp_frames(
            scene_dir, output_dir, target_size
        )
    elif dataset == "canonical":
        images_512, c2w_list, K_512_list, frame_ids = load_canonical_scene(
            output_dir, target_size
        )
        print(f"[gsplat_fit] Loaded {len(images_512)} frames from canonical format")
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Use 'scannetpp' or 'canonical'.")

    # ── GPU tensors ───────────────────────────────────────────────────────────
    w2c_list = [np.linalg.inv(c2w) for c2w in c2w_list]
    viewmats = torch.tensor(np.stack(w2c_list),    dtype=torch.float32, device=device)
    Ks       = torch.tensor(np.stack(K_512_list),   dtype=torch.float32, device=device)
    gt_imgs  = torch.tensor(
        np.stack(images_512).astype(np.float32) / 255.0,
        dtype=torch.float32, device=device,
    )  # N × 512 × 512 × 3

    # ── Initialize Gaussians from point cloud ────────────────────────────────
    pts = load_init_points(scene_dir)
    N   = len(pts)
    print(f"[gsplat_fit] Initializing from {N} points")

    # Keys must match what DefaultStrategy expects ("scales", "opacities", etc.)
    # Values are stored pre-activation (log-scales, logit-opacities) — standard in 3DGS.
    # After densification the strategy replaces tensors inside splat_params, so we always
    # read from the dict rather than local aliases.
    splat_params = {
        "means":     torch.nn.Parameter(torch.tensor(pts, dtype=torch.float32, device=device)),
        "quats":     torch.nn.Parameter(
            torch.cat([torch.ones(N, 1, device=device),
                       torch.zeros(N, 3, device=device)], dim=1)
        ),  # identity [w,x,y,z]
        "scales":    torch.nn.Parameter(torch.full((N, 3), -4.0, device=device)),  # log-space
        "opacities": torch.nn.Parameter(torch.full((N,), -2.0, device=device)),    # logit-space
        "sh0":       torch.nn.Parameter(torch.full((N, 1, 3), 0.5, device=device)),
        "shN":       torch.nn.Parameter(torch.zeros(N, 15, 3, device=device)),
    }

    # Optimizer keys must match splat_params keys exactly (DefaultStrategy uses them to
    # look up optimizers when cloning/splitting/pruning Gaussians)
    optimizers = {
        k: torch.optim.Adam(
            [splat_params[k]],
            lr={"means": 1.6e-4, "quats": 1e-3, "scales": 5e-3,
                "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}[k],
            eps=1e-15,
        )
        for k in splat_params
    }

    # gsplat DefaultStrategy: handles clone / split / prune densification
    strategy = DefaultStrategy(
        verbose=False,
        prune_opa=0.005,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        refine_start_iter=500,
        refine_stop_iter=int(max_steps * 0.7),  # extended: 70% of steps
        refine_every=100,
        absgrad=True,
    )
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    eval_metrics = {}
    N_frames = len(images_512)
    if rank == 0:
        print(f"[gsplat_fit] Training {max_steps} steps on {N_frames} frames  "
              f"(world_size={world_size}) …")

    for step in tqdm(range(max_steps), desc="3DGS fit"):
        # Broadcast rank 0's params to all ranks BEFORE each forward pass.
        # This ensures forward and backward always see consistent tensor shapes —
        # placing the broadcast after backward (previous approach) caused the
        # backward to see a different-sized tensor than the forward recorded.
        if world_size > 1:
            _dist_broadcast_params(splat_params, optimizers, rank)

        idx     = np.random.randint(0, N_frames)
        viewmat = viewmats[idx : idx + 1]
        K       = Ks[idx : idx + 1]
        gt      = gt_imgs[idx]  # 512×512×3

        # Always read from splat_params — tensors are replaced in-place by densification
        _means  = splat_params["means"]
        _quats  = splat_params["quats"]
        _scales = torch.exp(splat_params["scales"])
        _opacs  = torch.sigmoid(splat_params["opacities"])
        _colors = _eval_sh_colors(
            splat_params["sh0"], splat_params["shN"], _means, viewmat
        )

        renders, alphas, info = rasterization(
            _means,
            _quats / (_quats.norm(dim=-1, keepdim=True) + 1e-8),
            _scales, _opacs, _colors,
            viewmat, K,
            width=target_size, height=target_size,
            absgrad=True,
        )
        render = renders[0].clamp(0, 1)

        # DefaultStrategy expects radii [C, N]; gsplat squeezes C=1 → unsqueeze it back
        if info["radii"].dim() == 1:
            info["radii"] = info["radii"].unsqueeze(0)

        # step_pre_backward only needed on rank 0 — it registers gradient hooks
        # used by step_post_backward for densification decisions.
        if rank == 0:
            strategy.step_pre_backward(splat_params, optimizers, strategy_state, step, info)

        loss = torch.nn.functional.mse_loss(render, gt)
        loss.backward()

        # All-reduce gradients across GPUs before the optimizer step
        if world_size > 1:
            _dist_allreduce_grads(splat_params, world_size)

        for p in splat_params.values():
            if p.grad is not None:
                p.grad.nan_to_num_(0.0, 0.0, 0.0)

        # Rank 0 owns the optimizer state, LR schedule, and densification.
        if rank == 0:
            # gsplat sets means2d.absgrad as [N, 2]; unsqueeze to [1, N, 2]
            _m2d  = info.get("means2d")
            _absg = getattr(_m2d, "absgrad", None)
            if _absg is not None and _absg.dim() == 2:
                _m2d.absgrad = _absg.unsqueeze(0)

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            strategy.step_post_backward(splat_params, optimizers, strategy_state, step, info,
                                        packed=False)
        else:
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

        if rank == 0 and (step + 1) % eval_interval == 0:
            with torch.no_grad():
                psnr = _compute_psnr(render.detach(), gt)
            n_gs = splat_params["means"].shape[0]
            eval_metrics[step + 1] = {"psnr": psnr, "loss": loss.item(), "n_gaussians": n_gs}
            print(f"  step {step+1:5d}  loss={loss.item():.4f}  "
                  f"PSNR={psnr:.2f} dB  gaussians={n_gs:,}")

    # ── Save (rank 0 only) ────────────────────────────────────────────────────
    if rank != 0:
        return {}

    gaussians_dir = output_dir / "gaussians"
    gaussians_dir.mkdir(parents=True, exist_ok=True)

    _save_gaussians_ply(
        gaussians_dir / "point_cloud.ply",
        splat_params["means"].detach().cpu().numpy(),
        splat_params["quats"].detach().cpu().numpy(),
        torch.exp(splat_params["scales"]).detach().cpu().numpy(),
        torch.sigmoid(splat_params["opacities"]).detach().cpu().numpy(),
        torch.cat([splat_params["sh0"], splat_params["shN"]], dim=1).detach().cpu().numpy(),
    )

    cameras = []
    for i, (fid, c2w, K_512) in enumerate(zip(frame_ids, c2w_list, K_512_list)):
        cameras.append({
            "frame_id": fid,
            "w2c":      w2c_list[i].tolist(),
            "K_512":    K_512.tolist(),
            "width":    target_size,
            "height":   target_size,
        })
    with open(output_dir / "cameras.json", "w") as f:
        json.dump(cameras, f, indent=2)

    final_psnr = list(eval_metrics.values())[-1]["psnr"] if eval_metrics else 0.0
    flagged    = final_psnr < psnr_threshold
    with open(gaussians_dir / "eval_metrics.json", "w") as f:
        json.dump({
            "final_psnr":          final_psnr,
            "psnr_threshold":      psnr_threshold,
            "flagged_low_quality": flagged,
            "history":             eval_metrics,
        }, f, indent=2)

    if flagged:
        print(f"  [warn] PSNR {final_psnr:.1f} dB < {psnr_threshold} dB — flagged low quality")
    else:
        print(f"  [gsplat_fit] Final PSNR {final_psnr:.1f} dB  ✓")

    return eval_metrics


def _compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(pred, gt).item()
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def _save_gaussians_ply(path: Path, means, quats, scales, opacities, sh_coeffs):
    """Save Gaussian parameters as a PLY file."""
    try:
        from plyfile import PlyData, PlyElement
        import numpy as np

        N = len(means)
        sh_flat = sh_coeffs.reshape(N, -1)  # N x (16*3)
        sh_names = [f"f_dc_{c}" for c in range(3)] + \
                   [f"f_rest_{i}" for i in range(sh_flat.shape[1] - 3)]

        dtype_fields = (
            [("x", "f4"), ("y", "f4"), ("z", "f4")]
            + [(f"rot_{i}", "f4") for i in range(4)]
            + [(f"scale_{i}", "f4") for i in range(3)]
            + [("opacity", "f4")]
            + [(n, "f4") for n in sh_names]
        )
        arr = np.zeros(N, dtype=dtype_fields)
        arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
        for i in range(4):
            arr[f"rot_{i}"] = quats[:, i]
        for i in range(3):
            arr[f"scale_{i}"] = scales[:, i]
        arr["opacity"] = opacities
        arr["f_dc_0"] = sh_coeffs[:, 0, 0]
        arr["f_dc_1"] = sh_coeffs[:, 0, 1]
        arr["f_dc_2"] = sh_coeffs[:, 0, 2]
        for i in range(sh_flat.shape[1] - 3):
            arr[sh_names[3 + i]] = sh_flat[:, 3 + i]

        el = PlyElement.describe(arr, "vertex")
        PlyData([el]).write(str(path))
        print(f"  [gsplat_fit] Saved {N} Gaussians → {path}")
    except ImportError:
        np.save(str(path).replace(".ply", "_means.npy"), means)
        print(f"  [gsplat_fit] plyfile not installed — saved means only")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit 3DGS to an indoor scene")
    parser.add_argument("--scene-id",       required=True)
    parser.add_argument("--dataset",        default="scannetpp",
                        choices=["scannetpp", "canonical"],
                        help="'scannetpp': read from raw ScanNet++ nerfstudio format. "
                             "'canonical': read from cameras.json+rgb_512/ "
                             "(produced by scannet_io.py or replica_io.py)")
    parser.add_argument("--data-root",      default=None,
                        help="Path to ScanNet++ data/ root (scannetpp mode only)")
    parser.add_argument("--scenes-root",    default="data/scenes",
                        help="Pipeline output root (canonical mode); also output for scannetpp")
    parser.add_argument("--max-steps",      type=int,   default=30000)
    parser.add_argument("--eval-interval",  type=int,   default=1000)
    parser.add_argument("--psnr-threshold", type=float, default=20.0)
    parser.add_argument("--device-id",      type=int,   default=None,
                        help="CUDA device index for single-GPU mode")
    parser.add_argument("--world-size",     type=int,   default=1,
                        help="Number of GPUs for data-parallel fitting (uses mp.spawn)")
    args = parser.parse_args()

    if args.dataset == "scannetpp":
        if args.data_root is None:
            parser.error("--data-root is required for --dataset scannetpp")
        scene_dir  = Path(args.data_root)   / args.scene_id
        output_dir = Path(args.scenes_root) / args.scene_id
    else:
        # canonical: scene_dir == output_dir (cameras.json + rgb_512/ already there)
        scene_dir  = Path(args.scenes_root) / args.scene_id
        output_dir = scene_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        scene_dir=scene_dir,
        output_dir=output_dir,
        dataset=args.dataset,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        psnr_threshold=args.psnr_threshold,
    )

    if args.world_size > 1:
        import torch.multiprocessing as mp
        print(f"[gsplat_fit] Scene: {args.scene_id}  Dataset: {args.dataset}  "
              f"world_size={args.world_size}")
        mp.spawn(
            _train_gsplat_rank,
            args=(args.world_size, train_kwargs),
            nprocs=args.world_size,
            join=True,
        )
    else:
        if args.device_id is not None:
            device = torch.device(f"cuda:{args.device_id}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[gsplat_fit] Scene: {args.scene_id}  Dataset: {args.dataset}  Device: {device}")
        train_gsplat(**train_kwargs, device=device)


if __name__ == "__main__":
    main()

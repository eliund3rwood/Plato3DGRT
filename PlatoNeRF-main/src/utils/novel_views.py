"""
utils/novel_views.py — camera poses + ray construction for the VSD texture-
refinement stage (run_platonerf_3dgrt_vsd.py).

Poses are the same validated orbit/dolly set already used for qualitative
flyaround videos in render_test_depth_3dgrt.py and run_platonerf_3dgrt_rgb.py
(compute_test_poses), reproduced here standalone so the VSD script doesn't
need to import a sibling training script just for camera math.

All poses are camera-to-world matrices already in the 3DGRT coordinate system
(i.e. after the poses[:,0,:]*=-1; poses[:,2,:]*=-1 flip the training scripts
apply to the raw Mitsuba poses).
"""

import numpy as np

from utils.nerf_helpers import get_rays_np


_LOOKAT = np.array([0.0, -1.5, -3.0], dtype=np.float32)
_DOLLY_STEP = 0.06282151815625667


def _look_at(vec_pos: np.ndarray, vec_look_at: np.ndarray) -> np.ndarray:
    """Camera-to-world matrix (OpenGL convention: -z is forward)."""
    z = vec_look_at - vec_pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(x, z)
    m = np.zeros((4, 4), dtype=np.float32)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = -z
    m[:3, 3] = vec_pos
    m[3, :] = [0.0, 0.0, 0.0, 1.0]
    return m


def _circle_points(origin_2d, radius: float, n: int, start_angle: float) -> np.ndarray:
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, n, endpoint=False)
    xs = origin_2d[0] + radius * np.cos(angles)
    ys = origin_2d[1] + radius * np.sin(angles)
    return np.column_stack((xs, ys))


def orbit_poses(n: int = 100, radius: float = 0.99) -> np.ndarray:
    """[n, 4, 4] camera-to-world matrices orbiting (0, y, -3), same formula as
    render_test_depth_3dgrt.py's flyaround (already validated to render
    non-degenerate views of the chair+wall scene)."""
    y = -1.5
    pts = _circle_points((0, 3), radius, n, -np.pi / 2)
    poses = [_look_at(np.array([-p[0], y, -p[1]], dtype=np.float32), _LOOKAT) for p in pts]
    return np.stack(poses, axis=0)


def dolly_poses(n: int = 39) -> np.ndarray:
    """[n, 4, 4] camera-to-world matrices sweeping the SPAD camera along its
    z axis (mirrors the Nframes=40 loop used for eval video/GT test poses)."""
    poses = [
        _look_at(np.array([0.0, -1.5, -_DOLLY_STEP * i], dtype=np.float32), _LOOKAT)
        for i in range(n)
    ]
    return np.stack(poses, axis=0)


def novel_view_poses(n_orbit: int = 100, n_dolly: int = 39, include_dolly: bool = True) -> np.ndarray:
    """Combined pose bank to sample novel views B from during VSD."""
    poses = [orbit_poses(n_orbit)]
    if include_dolly:
        poses.insert(0, dolly_poses(n_dolly))
    return np.concatenate(poses, axis=0)


def rays_at_resolution(
    pose_c2w: np.ndarray,   # [3,4] or [4,4]
    H_train: int, W_train: int, focal_train: float,
    H_out: int, W_out: int,
) -> np.ndarray:
    """Build normalised [H_out*W_out, 2, 3] rays for pose_c2w, reconstructing
    the training FOV at an arbitrary output resolution (same pattern as
    run_platonerf_3dgrt_rgb.py's load_ref_rgb)."""
    camera_angle_x = 2.0 * np.arctan(0.5 * W_train / focal_train)
    focal_out = 0.5 * W_out / np.tan(0.5 * camera_angle_x)
    K_out = np.array([
        [focal_out, 0, 0.5 * W_out],
        [0, focal_out, 0.5 * H_out],
        [0, 0, 1],
    ], dtype=np.float32)

    rays_od = get_rays_np(H_out, W_out, K_out, pose_c2w[:3, :4])       # [2, H, W, 3]
    rays_flat = np.transpose(rays_od, [1, 2, 0, 3]).reshape(-1, 2, 3).astype(np.float32)
    norms = np.linalg.norm(rays_flat[:, 1, :], axis=1, keepdims=True)
    rays_flat[:, 1, :] /= norms
    return rays_flat

"""
utils/synthetic_scene.py — cameras and rays for a SYNTHETIC scene fitted by
PlatoControlNet's `gsplat_fit.py`, for the VSD control experiment.

This is the counterpart to `utils/novel_views.py`, which is hardcoded to the
real chair scene (orbit around (0,-1.5,-3), radius 0.99, NeRF/OpenGL c2w poses
in the ToF sensor's world frame). A synthetic scene has a completely different
world frame — Blender's, Z-up, chair near the origin, cameras on a 24-view ring
at radius ~1.28 — so none of those numbers transfer.

THE CONVENTION, AND THE TRAP
----------------------------
`cameras.json` stores **OpenCV w2c** (x right, y DOWN, +z FORWARD). Verified two
independent ways, not assumed:

  * by source — `gsplat_fit.py` feeds these matrices straight to
    `gsplat.rasterization(viewmats=...)`, which is OpenCV by definition; and
  * by measurement — `scripts/test_01_geom_test_assets.py` projects every
    Gaussian centre under both candidates and finds 57% of them in frame as
    OpenCV vs 2% as OpenGL.

So `pose_convert.nerf_c2w_to_opencv_w2c()` must **NOT** be applied to these
poses. That function exists for the chair, whose poses come out of
`novel_views._look_at()` in NeRF/OpenGL convention. Applying it here — the
natural thing to do by analogy with the chair path — flips Y and Z and produces
a render that is upside-down and inside-out, but still a plausible-looking
image of a chair-ish thing. The V7 prior consumes w2c/K from here unchanged.

The other half of `platonerf_view_to_v7`, `ray_distance_to_z_depth()`, IS still
required: that difference is about 3DGRT's renderer (which returns euclidean
ray distance because it is a ToF-native code path), not about the scene, and it
applies to any scene 3DGRT renders. Its docstring notes it is
convention-independent, so it is correct to use on OpenCV poses.

Pixel convention is INTEGER corner coordinates (`arange(W)`, no +0.5), matching
both `nerf_helpers.get_rays_np` on the 3DGRT side and `consistency_project.
unproject_grid` on the PlatoControlNet side, so a point unprojected by V7 lands
on the ray 3DGRT traced for it.
"""

import json

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class SceneCameras:
    """Cameras for one fitted synthetic scene, in OpenCV w2c convention."""

    def __init__(self, w2c, K, frame_ids, width, height):
        self.w2c = np.asarray(w2c, dtype=np.float32)        # [n,4,4]
        self.K = np.asarray(K, dtype=np.float32)            # [3,3] at (width,height)
        self.frame_ids = list(frame_ids)
        self.width = int(width)
        self.height = int(height)

    def __len__(self):
        return len(self.w2c)

    @property
    def centers(self):
        """[n,3] camera centres in world space: C = -Rᵀ t. Convention-independent."""
        R = self.w2c[:, :3, :3]
        t = self.w2c[:, :3, 3]
        return np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)

    @property
    def forwards(self):
        """[n,3] camera viewing directions (OpenCV +z axis, expressed in world)."""
        return self.w2c[:, 2, :3]


def load_scene_cameras(path):
    """Read a gsplat_fit `cameras.json` (list of {frame_id, w2c, K_512, width,
    height}). Also accepts the dict-of-dicts variant gsplat_fit tolerates."""
    with open(path) as f:
        cams = json.load(f)
    if isinstance(cams, dict):
        cams = [{"frame_id": k, **v} for k, v in cams.items()]

    Ks = np.stack([np.asarray(c["K_512"], dtype=np.float64) for c in cams])
    if not np.allclose(Ks, Ks[0], atol=1e-6):
        raise ValueError(
            f"{path}: intrinsics differ between frames. Everything downstream "
            "(the cached V7 K, the orbit synthesis) assumes one shared K.")
    return SceneCameras(
        w2c=np.stack([np.asarray(c["w2c"], dtype=np.float64) for c in cams]),
        K=Ks[0],
        frame_ids=[c["frame_id"] for c in cams],
        width=int(cams[0].get("width", 512)),
        height=int(cams[0].get("height", 512)),
    )


# ---------------------------------------------------------------------------
# Intrinsics / rays
# ---------------------------------------------------------------------------

def scale_intrinsics(K, width, height, out_res):
    """K for a square out_res x out_res render of the same field of view."""
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, :] *= out_res / float(width)
    K[1, :] *= out_res / float(height)
    return K.astype(np.float32)


def rays_from_w2c(w2c, K, out_res):
    """Rays for one OpenCV w2c pose, in the [N,2,3] (origin, unit direction)
    layout 3DGRT's tracer consumes — the same layout `novel_views.
    rays_at_resolution` produces for the chair, so the VSD loop needs no
    special-casing downstream.

    `K` must already be at out_res (see `scale_intrinsics`).
    """
    w2c = np.asarray(w2c, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    i, j = np.meshgrid(np.arange(out_res, dtype=np.float64),
                       np.arange(out_res, dtype=np.float64), indexing="xy")
    # OpenCV camera frame: x right, y DOWN, +z FORWARD. No sign flips — this is
    # exactly where the NeRF path differs (it uses (x, -y, -1)).
    dirs_cam = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)

    c2w = np.linalg.inv(w2c)
    rays_d = dirs_cam @ c2w[:3, :3].T
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)

    rays = np.stack([rays_o, rays_d], axis=0)                    # [2,H,W,3]
    rays = np.transpose(rays, (1, 2, 0, 3)).reshape(-1, 2, 3).astype(np.float32)
    # 3DGRT's tracer expects unit directions and returns euclidean distance
    # along them (hence ray_distance_to_z_depth downstream).
    rays[:, 1, :] /= np.linalg.norm(rays[:, 1, :], axis=1, keepdims=True)
    return rays


# ---------------------------------------------------------------------------
# Ring fitting / orbit synthesis
# ---------------------------------------------------------------------------

def _closest_point_to_rays(origins, dirs):
    """Least-squares point closest to all the given lines — the look-at target
    the ring was actually built around. Solved rather than assumed equal to the
    Gaussian centroid, because a renderer aims at the object, not at the mean of
    a point cloud that also contains floor and walls."""
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)     # projector onto the plane ⟂ d
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def look_at_w2c(center, target, up):
    """OpenCV w2c looking from `center` at `target`. Rows of R are the camera's
    (right, down, forward) axes expressed in world coordinates."""
    f = np.asarray(target, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, dtype=np.float64))
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)                     # r x d = f, i.e. right-handed
    R = np.stack([r, d, f], axis=0)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ np.asarray(center, dtype=np.float64)
    return w2c


class Ring:
    """The circle the fitted scene's cameras lie on, recovered from them."""

    def __init__(self, target, up, radius, height, plane_axes, up_axis, azimuths):
        self.target = target
        self.up = up
        self.radius = float(radius)
        self.height = float(height)
        self.plane_axes = plane_axes
        self.up_axis = int(up_axis)
        self.azimuths = azimuths           # radians, in input frame order

    def at_azimuth(self, az):
        c = np.zeros(3)
        c[self.plane_axes[0]] = self.target[self.plane_axes[0]] + self.radius * np.cos(az)
        c[self.plane_axes[1]] = self.target[self.plane_axes[1]] + self.radius * np.sin(az)
        c[self.up_axis] = self.height
        return look_at_w2c(c, self.target, self.up)

    def orbit(self, n):
        """n OpenCV w2c poses evenly spaced around the same circle, starting at
        the first input view's azimuth so pose 0 of a synthesised orbit and
        frame 0 of the ring agree."""
        az = self.azimuths[0] + np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.stack([self.at_azimuth(a) for a in az]).astype(np.float32)


def fit_ring(cams: SceneCameras) -> Ring:
    """Recover the circle the input cameras lie on, so a denser orbit can be
    synthesised on it. Every quantity is measured from the poses."""
    C = cams.centers.astype(np.float64)
    target = _closest_point_to_rays(C, cams.forwards.astype(np.float64))

    # The ring plane's normal is the world axis the centres do not vary along.
    spread = C.std(axis=0)
    up_axis = int(np.argmin(spread))
    plane_axes = tuple(a for a in range(3) if a != up_axis)

    # World up points the way the cameras' "down" axis does not: the OpenCV y
    # row is world-down, so -mean(y row) is world-up. Taking it from the poses
    # keeps the synthesised orbit's roll identical to the input ring's.
    up = -cams.w2c[:, 1, :3].astype(np.float64).mean(axis=0)
    up = up / np.linalg.norm(up)

    radial = C[:, plane_axes] - target[list(plane_axes)]
    radius = np.linalg.norm(radial, axis=1).mean()
    height = C[:, up_axis].mean()
    azimuths = np.arctan2(radial[:, 1], radial[:, 0])
    return Ring(target, up, radius, height, plane_axes, up_axis, azimuths)


def ring_reconstruction_error(cams: SceneCameras, ring: Ring):
    """Max error when the fitted ring is used to REGENERATE the input poses.

    The point of fitting a ring is to synthesise poses that were never in the
    file. If the model cannot reproduce the poses that WERE in the file, the
    synthesised ones are not on the real orbit either — so this is the gate that
    licenses using `Ring.orbit()` at all. Returns (max rotation error in
    degrees, max centre error in scene units)."""
    rot_err, pos_err = [], []
    for k, az in enumerate(ring.azimuths):
        got = ring.at_azimuth(az)
        want = cams.w2c[k].astype(np.float64)
        dR = got[:3, :3] @ want[:3, :3].T
        cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        rot_err.append(np.degrees(np.arccos(cos)))
        c_got = -got[:3, :3].T @ got[:3, 3]
        c_want = -want[:3, :3].T @ want[:3, 3]
        pos_err.append(np.linalg.norm(c_got - c_want))
    return float(np.max(rot_err)), float(np.max(pos_err))

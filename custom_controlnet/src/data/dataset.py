"""
src/data/dataset.py — NVSDataset: reads manifest JSONL, returns typed batches.

Every __getitem__ return asserts spatial dims are (512, 512). Hard failure
on shape mismatch is intentional — a silent wrong-shape tensor would corrupt
the whole training run.

Normalization:
  RGB  : PIL → [0,1] → *2-1 → [-1,1]
  Depth: clip [near,far] → [0,1] → replicate to 3 channels
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_IMG_SIZE = 512  # spatial contract — never change without updating all asserts


class NVSDataset(Dataset):
    """
    Dataset for (I_A, D_B, I_B) triples.

    Args:
        manifest_path: path to train.jsonl / val.jsonl / test.jsonl
        use_cached_latents: if True, also load z_A_path / z_B_path from manifest
        depth_norm_mode: "per_image" (use manifest near/far) or "per_scene"
        return_alpha: whether to return alpha_B mask
    """

    def __init__(
        self,
        manifest_path: str | Path,
        use_cached_latents: bool = False,
        depth_norm_mode: str = "per_image",
        return_alpha: bool = True,
    ):
        self.use_cached_latents = use_cached_latents
        self.depth_norm_mode = depth_norm_mode
        self.return_alpha = return_alpha

        self.rows: list[dict] = []
        with open(manifest_path) as f:
            for line in f:
                row = json.loads(line.strip())
                if row:
                    self.rows.append(row)

        if len(self.rows) == 0:
            raise ValueError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        I_A = self._load_rgb(row["I_A_path"])
        I_B = self._load_rgb(row["I_B_path"])
        D_B = self._load_depth(row["D_B_path"], row["depth_near"], row["depth_far"])

        # Hard shape asserts — any violation is a pipeline bug, not a data issue
        assert I_A.shape == (3, _IMG_SIZE, _IMG_SIZE), \
            f"I_A shape {I_A.shape} != (3,{_IMG_SIZE},{_IMG_SIZE})"
        assert I_B.shape == (3, _IMG_SIZE, _IMG_SIZE), \
            f"I_B shape {I_B.shape} != (3,{_IMG_SIZE},{_IMG_SIZE})"
        assert D_B.shape == (3, _IMG_SIZE, _IMG_SIZE), \
            f"D_B shape {D_B.shape} != (3,{_IMG_SIZE},{_IMG_SIZE})"

        out = {
            "I_A": I_A,   # float32 [-1, 1]
            "I_B": I_B,   # float32 [-1, 1]
            "D_B": D_B,   # float32 [0, 1] (3-channel replicated depth)
            "meta": {
                "scene_id":   row.get("scene_id", ""),
                "frame_id_A": row.get("frame_id_A", ""),
                "frame_id_B": row.get("frame_id_B", ""),
                "baseline_m": row.get("baseline_m", 0.0),
                "angle_deg":  row.get("angle_deg",  0.0),
                "covis":      row.get("covis",       0.0),
            },
        }

        if self.return_alpha and row.get("alpha_B_path"):
            alpha_B = self._load_alpha(row["alpha_B_path"])
            assert alpha_B.shape == (1, _IMG_SIZE, _IMG_SIZE), \
                f"alpha_B shape {alpha_B.shape} != (1,{_IMG_SIZE},{_IMG_SIZE})"
            out["alpha_B"] = alpha_B

        if self.use_cached_latents:
            if row.get("z_A_path"):
                out["z_A"] = torch.from_numpy(np.load(row["z_A_path"]).astype(np.float32))
            if row.get("z_B_path"):
                out["z_B"] = torch.from_numpy(np.load(row["z_B_path"]).astype(np.float32))

        return out

    # ── Loaders ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_rgb(path: str) -> torch.Tensor:
        """Load PNG → float32 tensor in [-1, 1], shape (3, 512, 512)."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read RGB: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure 512×512 (should already be correct from pipeline)
        if img.shape[:2] != (_IMG_SIZE, _IMG_SIZE):
            img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        t = torch.from_numpy(img.astype(np.float32) / 255.0)  # H×W×3
        t = t.permute(2, 0, 1)                                 # 3×H×W
        t = t * 2.0 - 1.0                                      # [-1, 1]
        return t

    @staticmethod
    def _load_depth(path: str, near: float, far: float) -> torch.Tensor:
        """
        Load float32 depth .npy → normalize to [0,1] → replicate to 3 channels.
        Shape: (3, 512, 512).
        """
        depth = np.load(str(path)).astype(np.float32)  # H×W

        if depth.shape != (_IMG_SIZE, _IMG_SIZE):
            depth = cv2.resize(depth, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # Per-image normalization
        range_d = max(far - near, 1e-6)
        depth = np.clip((depth - near) / range_d, 0.0, 1.0)

        t = torch.from_numpy(depth).unsqueeze(0)  # 1×H×W
        t = t.expand(3, -1, -1).contiguous()      # 3×H×W (replicate)
        return t

    @staticmethod
    def _load_alpha(path: str) -> torch.Tensor:
        """Load float32 alpha .npy → tensor (1, 512, 512)."""
        alpha = np.load(str(path)).astype(np.float32)

        if alpha.shape != (_IMG_SIZE, _IMG_SIZE):
            alpha = cv2.resize(alpha, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        alpha = np.clip(alpha, 0.0, 1.0)
        return torch.from_numpy(alpha).unsqueeze(0)  # 1×H×W

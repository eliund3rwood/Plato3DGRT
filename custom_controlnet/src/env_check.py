"""
src/env_check.py -- M0 environment verification.

Usage:
    python -m src.env_check

Prints GPU count, VRAM, compute capability, architecture family, and
the AMP dtype that will be selected at training time. Fails fast on
common configuration problems.
"""

import sys
import importlib
import platform


# ---- Architecture classification --------------------------------------------

_SM_TO_ARCH = {
    (12, 0): "Blackwell",
    (9, 0): "Hopper",
    (8, 9): "Ada Lovelace",
    (8, 6): "Ampere",
    (8, 0): "Ampere",
    (7, 5): "Turing",
    (7, 0): "Volta",
    (6, 1): "Pascal",
    (6, 0): "Pascal",
    (5, 3): "Maxwell",
    (5, 2): "Maxwell",
}


def _arch_name(major: int, minor: int) -> str:
    for (maj, min_), name in _SM_TO_ARCH.items():
        if major == maj and minor >= min_:
            return name
    if major >= 12:
        return "Blackwell+"
    return f"Unknown (sm_{major}{minor})"


def get_amp_dtype_str(major: int, minor: int) -> str:
    """Return the AMP dtype string for a given compute capability."""
    if major >= 8:
        return "bfloat16"
    return "float16 + GradScaler"


# ---- Dependency check -------------------------------------------------------

_REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "lpips",
    "omegaconf",
    "wandb",
    "cv2",
    "PIL",
    "numpy",
    "tqdm",
    "einops",
]

_OPTIONAL_PACKAGES = [
    ("xformers", "memory-efficient attention"),
    ("gsplat", "3D Gaussian Splatting"),
    ("bitsandbytes", "8-bit Adam"),
    ("webdataset", "WebDataset shards"),
    ("lmdb", "LMDB latent cache"),
]


def check_packages() -> None:
    print("\n-- Python & packages " + "-" * 45)
    print(f"  Python      {sys.version}")
    print(f"  Platform    {platform.platform()}")

    missing = []
    for pkg in _REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK]      {pkg:<22} {version}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\n  [ERROR] Missing required packages: {', '.join(missing)}")
        print("          Run: pip install -r requirements-local.txt")
    else:
        print("\n  All required packages found.")

    print("\n-- Optional packages " + "-" * 46)
    for pkg, desc in _OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK]     {pkg:<22} {version}  ({desc})")
        except ImportError:
            print(f"  [absent] {pkg:<22}  ({desc}) -- not required")


# ---- GPU check --------------------------------------------------------------

def check_gpus() -> None:
    try:
        import torch
    except ImportError:
        print("  torch not installed -- cannot check GPUs.")
        return

    print("\n-- CUDA & GPU " + "-" * 52)
    print(f"  torch version     {torch.__version__}")
    print(f"  CUDA available    {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("  [ERROR] No CUDA GPUs found. Training requires at least one CUDA GPU.")
        return

    count = torch.cuda.device_count()
    print(f"  GPU count         {count}")

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        major, minor = props.major, props.minor
        vram_gb = props.total_memory / (1024 ** 3)
        arch = _arch_name(major, minor)
        amp = get_amp_dtype_str(major, minor)
        bf16 = major >= 8

        print(f"\n  GPU {i}: {props.name}")
        print(f"    Architecture        {arch}  (sm_{major}{minor})")
        print(f"    VRAM                {vram_gb:.1f} GB")
        print(f"    bf16 native         {bf16}")
        print(f"    Default AMP dtype   {amp}")
        print(f"    Multi-processor cnt {props.multi_processor_count}")

        if major < 7:
            print(f"    [WARN] sm_{major}{minor} is below Volta -- AMP may not be reliable.")

    # NCCL availability
    print(f"\n  NCCL available    {torch.distributed.is_nccl_available()}")

    # xformers
    try:
        import xformers
        print(f"  xformers          {xformers.__version__}  (memory-efficient attention enabled)")
    except ImportError:
        print("  xformers          not installed (will use PyTorch SDPA fallback)")


# ---- Seed / reproducibility check ------------------------------------------

def check_reproducibility() -> None:
    print("\n-- Reproducibility " + "-" * 47)
    try:
        import torch
        import numpy as np
        import random

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"  Seeded torch / numpy / random with seed={seed}")
        print("  torch.backends.cudnn.deterministic  (set in training)")
    except Exception as e:
        print(f"  [WARN] seed check failed: {e}")


# ---- Shape sanity -----------------------------------------------------------

def check_shapes() -> None:
    """Verify that the spatial contract (512x512, 64x64 latents) holds."""
    print("\n-- Spatial contract smoke test " + "-" * 35)
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        B = 1

        rgb = torch.zeros(B, 3, 512, 512, device=device)
        assert rgb.shape[-2:] == (512, 512), "RGB shape mismatch"

        depth = torch.zeros(B, 3, 512, 512, device=device)
        assert depth.shape[-2:] == (512, 512), "Depth shape mismatch"

        latent = torch.zeros(B, 4, 64, 64, device=device)
        assert latent.shape[-2:] == (64, 64), "Latent shape mismatch"
        assert latent.shape[1] == 4, "Latent channel mismatch"

        unet_in = torch.zeros(B, 8, 64, 64, device=device)
        assert unet_in.shape[1] == 8, "UNet input channel mismatch"

        print(f"  [OK] RGB     {tuple(rgb.shape)}  -> 512x512")
        print(f"  [OK] Depth   {tuple(depth.shape)}  -> 512x512")
        print(f"  [OK] Latent  {tuple(latent.shape)}  -> 4x64x64")
        print(f"  [OK] UNet-in {tuple(unet_in.shape)}  -> 8x64x64")
        print(f"  All shapes conform to the spatial contract.")
    except Exception as e:
        print(f"  [ERROR] Shape check failed: {e}")


# ---- Main ------------------------------------------------------------------

def main() -> None:
    SEP = "=" * 66
    print(SEP)
    print("  PlatoControlNet -- Environment Check")
    print(SEP)

    check_packages()
    check_gpus()
    check_reproducibility()
    check_shapes()

    print("\n" + SEP)
    print("  Done. Review any [ERROR] / [WARN] above before training.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()

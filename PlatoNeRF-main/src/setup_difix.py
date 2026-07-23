"""
src/setup_difix.py — download nvidia/difix_ref and patch its bundled custom
model code (vae/autoencoder_kl.py, unet/unet_2d_condition.py) for compatibility
with this environment's diffusers (0.39.0), which is far newer than what
Difix3D's own pinned requirements.txt expects (diffusers==0.25.1).

We deliberately do NOT install Difix3D's pinned requirements.txt -- this
project's custom_controlnet VSD pipeline already needs diffusers>=0.27 (for
unet.add_adapter()/LoRA support, see requirements-vsd.txt), so downgrading
diffusers in this shared env would break that. Instead we keep the newer
diffusers and patch Difix's dynamically-loaded custom code to match its
current API surface. Four separate incompatibilities were found (each only
surfaces after fixing the previous one, since Python import errors are
sequential):

  1. vae/autoencoder_kl.py: `FromOriginalVAEMixin` was renamed to the more
     generic `FromOriginalModelMixin` at some point after 0.25.1.
  2. unet/unet_2d_condition.py: `diffusers.models.unet_2d_blocks` moved to
     `diffusers.models.unets.unet_2d_blocks`.
  3. unet/unet_2d_condition.py: `PositionNet` (GLIGEN grounded-generation
     conditioning) was removed from diffusers.models.embeddings entirely,
     not just moved. It's only ever instantiated when
     attention_type in ("gated", "gated-text-image") -- a mode Difix's plain
     SD-Turbo-based UNet never uses -- so a no-op stub class is safe; the
     real GLIGEN forward pass is simply never reached for our use case.
  4. vae/autoencoder_kl.py: `ModelMixin` used to bundle adapter/LoRA support
     (PeftAdapterMixin) automatically; that's been decoupled in current
     diffusers, so `self.add_adapter(...)` in AutoencoderKL.__init__ now
     needs PeftAdapterMixin mixed in explicitly.

Patches are applied to a LOCAL snapshot directory (via snapshot_download),
not to diffusers' own dynamic-module cache under
~/.cache/huggingface/modules/diffusers_modules/ -- that cache gets
regenerated/copied fresh on each from_pretrained() call and was observed to
silently discard direct edits, which cost real debugging time before
switching to this approach. Patching the snapshot dir once and always
loading from that local path (not the "nvidia/difix_ref" repo id) sidesteps
that entirely.

Usage:
    from setup_difix import get_difix_pipeline
    pipe = get_difix_pipeline(device="cuda")
"""

import os


_PATCH_MARKER = "# difix-compat-patched"


def _patch_vae(vae_path: str) -> None:
    with open(vae_path) as f:
        src = f.read()
    if _PATCH_MARKER in src:
        return

    src = src.replace(
        "from diffusers.loaders import FromOriginalVAEMixin",
        "from diffusers.loaders import FromOriginalModelMixin as FromOriginalVAEMixin\n"
        "from diffusers.loaders.peft import PeftAdapterMixin\n"
        f"{_PATCH_MARKER}",
    )
    src = src.replace(
        "class AutoencoderKL(ModelMixin, ConfigMixin, FromOriginalVAEMixin):",
        "class AutoencoderKL(ModelMixin, ConfigMixin, FromOriginalVAEMixin, PeftAdapterMixin):",
    )
    assert _PATCH_MARKER in src, f"vae patch did not apply cleanly: {vae_path}"
    with open(vae_path, "w") as f:
        f.write(src)
    print(f"[setup_difix] Patched: {vae_path}")


def _patch_unet(unet_path: str) -> None:
    with open(unet_path) as f:
        src = f.read()
    if _PATCH_MARKER in src:
        return

    src = src.replace(
        "from diffusers.models.unet_2d_blocks import",
        "from diffusers.models.unets.unet_2d_blocks import",
    )
    src = src.replace("    PositionNet,\n", "")

    anchor = "from diffusers.models.modeling_utils import ModelMixin\n"
    assert anchor in src, f"anchor not found in {unet_path}"
    stub = (
        anchor
        + f"{_PATCH_MARKER}\n\n"
        "class PositionNet(nn.Module):\n"
        "    # Stub: real GLIGEN PositionNet was removed from this diffusers\n"
        "    # version. Difix never sets attention_type to \"gated\"/\n"
        "    # \"gated-text-image\", so this is never actually instantiated with\n"
        "    # meaningful behavior needed -- only needs to exist so the\n"
        "    # reference at construction time does not error.\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        super().__init__()\n"
    )
    src = src.replace(anchor, stub, 1)
    with open(unet_path, "w") as f:
        f.write(src)
    print(f"[setup_difix] Patched: {unet_path}")


def prepare_difix_snapshot(repo_id: str = "nvidia/difix_ref") -> str:
    """Downloads (if needed) and patches the Difix model snapshot. Idempotent
    -- safe to call every run; already-patched files are left alone."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=repo_id)
    _patch_vae(os.path.join(local_dir, "vae", "autoencoder_kl.py"))
    _patch_unet(os.path.join(local_dir, "unet", "unet_2d_condition.py"))
    return local_dir


def get_difix_pipeline(device: str = "cuda", repo_id: str = "nvidia/difix_ref"):
    """Returns a ready-to-use DifixPipeline on `device`. Requires
    pipeline_difix.py (vendored from the Difix3D repo, see
    custom_controlnet_difix/) to be importable -- caller should
    sys.path.insert(0, <path to that dir>) before calling this."""
    from pipeline_difix import DifixPipeline

    local_dir = prepare_difix_snapshot(repo_id)
    pipe = DifixPipeline.from_pretrained(local_dir, trust_remote_code=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    return pipe

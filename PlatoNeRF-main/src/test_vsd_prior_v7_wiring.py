#!/usr/bin/env python
"""
src/test_vsd_prior_v7_wiring.py — gates for the V7 VSD prior, WITHOUT a GPU,
a checkpoint, or the 3DGRT tracer.

py_compile proves nothing about wiring. These check the properties that would
otherwise fail hours into a cluster run, or worse, run to completion while
silently conditioning on the wrong thing:

  1. NO CFG BATCH DOUBLING. The single most important difference from
     vsd_prior.py. Under V7 a doubled batch would hand the triplane 2N views of
     "one scene" and pool the unconditional and conditional halves into ONE 3D
     volume. It would not crash; it would just quietly mix the two branches.
     V7's own generate_ring takes the sequential path for the same reason.
  2. TRIPLANE GEOMETRY IS ALWAYS CLEARED. set_triplane_geometry's contract is
     install-immediately-before / clear-immediately-after; a stale payload gets
     applied to whatever batch runs next. Must hold even when the forward
     raises, hence a `finally`.
  3. THE NULL BRANCH IS DEPTH-ONLY. V7 trained its CFG null branch against
     conditioning dropout that zeroes the IP tokens AND the splatted hint. A
     null branch that kept either would be contrasting against something the
     model never saw as "unconditional".
  4. ONE SHARED TIMESTEP ACROSS VIEWS. V7 draws one t per scene; the triplane
     and reference-K/V paths have only ever seen views that agree on it.
  5. LORA CHECKPOINT KEYS MATCH vsd_prior.py's. The training script saves both
     priors' state under one filename, so a different layout would load
     strict=False into nothing and silently resume with a fresh LoRA.
  6. THE TRAINING SCRIPT PASSES GEOMETRY. step()/preview() take three extra
     arguments under v7; a call site left on the old signature is a TypeError
     at iteration 35001.

Source-level checks by design — importing the module needs torch+omegaconf and
the PlatoControlNet repo, neither of which is guaranteed wherever this is run.

    python src/test_vsd_prior_v7_wiring.py
"""

import ast
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PRIOR = os.path.join(_HERE, "vsd_prior_v7.py")
_TRAIN = os.path.join(_HERE, "run_platonerf_3dgrt_vsd.py")
_V3 = os.path.join(_HERE, "vsd_prior.py")


def _fail(gate, msg):
    print(f"\n  [FAIL] gate {gate}: {msg}")
    sys.exit(1)


def _src(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    prior = _src(_PRIOR)
    train = _src(_TRAIN)
    v3 = _src(_V3)

    # ── Gate 1: no CFG batch doubling ───────────────────────────────────────
    # vsd_prior.py legitimately does this; vsd_prior_v7.py must not. Scanned via
    # AST, not regex: the module docstring EXPLAINS the doubling it avoids, and
    # a text search flags that prose as a violation (it did, on the first run).
    tree = ast.parse(prior)
    doubling = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("repeat", "repeat_interleave")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 2):
            doubling.append(getattr(node, "lineno", "?"))
    if doubling:
        _fail("1", f"batch-doubling repeat(2, ...) at line(s) {doubling} in "
                   "vsd_prior_v7.py — under V7 this pools the CFG uncond and cond "
                   "halves into one triplane volume, silently")
    if prior.count("self._unet_forward(") < 2:
        _fail("1", "expected at least two separate _unet_forward calls (cond and "
                   "uncond run sequentially); found "
                   f"{prior.count('self._unet_forward(')}")
    print("[test] gate 1 PASS  CFG is sequential, no batch doubling")

    # ── Gate 2: triplane geometry always cleared ────────────────────────────
    fwd = next((n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_unet_forward"), None)
    if fwd is None:
        _fail("2", "_unet_forward not found")
    tries = [n for n in ast.walk(fwd) if isinstance(n, ast.Try)]
    if not tries:
        _fail("2", "_unet_forward has no try/finally — triplane geometry would "
                   "leak to the next batch if the forward raised")
    cleared = any("clear_triplane_geometry" in ast.dump(t.finalbody[0]) if t.finalbody else False
                  for t in tries) or any(
        "_clear_triplane_geometry" in ast.dump(ast.Module(body=t.finalbody, type_ignores=[]))
        for t in tries if t.finalbody)
    if not cleared:
        _fail("2", "clear_triplane_geometry is not in a finally block")
    print("[test] gate 2 PASS  triplane geometry cleared in finally")

    # ── Gate 3: null branch is depth-only ───────────────────────────────────
    # Both the IP tokens and the splat hint must be zeroed for the uncond pass.
    if "torch.zeros_like(rendered)" not in prior:
        _fail("3", "the CFG null branch does not zero the splatted hint — V7's "
                   "conditioning dropout zeroes it, so the model never saw an "
                   "'unconditional' that kept it")
    if "torch.zeros_like(ip)" not in prior:
        _fail("3", "the CFG null branch does not zero the IP tokens")
    print("[test] gate 3 PASS  null branch zeroes both IP tokens and the hint")

    # ── Gate 4: one shared timestep ─────────────────────────────────────────
    if not re.search(r"torch\.randint\([^)]*\(1,\)", prior):
        _fail("4", "timestep is not drawn as a single per-scene value — V7 "
                   "trains one t shared across all N views")
    if "t_scene.expand(" not in prior:
        _fail("4", "the per-scene timestep is not expanded across views")
    print("[test] gate 4 PASS  one timestep per scene, shared across views")

    # ── Gate 5: LoRA checkpoint keys match the V3 prior ─────────────────────
    for key in ("lora_state_dict", "lora_optimizer"):
        if key not in prior:
            _fail("5", f"state_dict does not use '{key}' — vsd_prior.py does, and "
                       "the training script saves both under one filename")
    # AST again, not a text search: the docstring NAMES this helper to explain
    # why it is deliberately not used, and a text search flags that as the
    # violation it is documenting (it did, on the first run).
    peft_calls = [
        getattr(n, "lineno", "?") for n in ast.walk(tree)
        if isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Name)
             and n.func.id in ("get_peft_model_state_dict", "set_peft_model_state_dict"))
            or (isinstance(n.func, ast.Attribute)
                and n.func.attr in ("get_peft_model_state_dict",
                                    "set_peft_model_state_dict")))
    ]
    if peft_calls:
        _fail("5", f"calls peft's state-dict helpers at line(s) {peft_calls} while "
                   "vsd_prior.py uses a plain key filter — the two layouts are "
                   "not interchangeable")
    print("[test] gate 5 PASS  LoRA checkpoint layout matches vsd_prior.py")

    # ── Gate 6: the training script passes geometry ─────────────────────────
    if "vsd_prior.step(\n" not in train and "rgb_v, depth_cond, D_metric_v, w2c_v, K_v" not in train:
        _fail("6", "the v7 step() call site does not pass D_metric/w2c/K")
    if "set_reference_geometry(" not in train:
        _fail("6", "set_reference_geometry() is never called — the splat would "
                   "have no reference to unproject from")
    if "_ray_dist_to_z" not in train:
        _fail("6", "the training script never converts Euclidean ray distance to "
                   "z-depth; 3DGRT's pred_dist is NOT what unproject_grid wants")
    if "_nerf_to_cv" not in train:
        _fail("6", "the training script never converts NeRF c2w to OpenCV w2c")
    # The v3 path must remain untouched.
    if "vsd_prior.step(rgb_v, depth_cond)" not in train:
        _fail("6", "the original 2-argument v3 step() call is gone — that path "
                   "is the working baseline and must keep running unchanged")
    print("[test] gate 6 PASS  v7 passes geometry; v3 call site intact")

    print("\n[test] ALL GATES PASS")


if __name__ == "__main__":
    main()

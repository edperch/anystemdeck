"""Persistent ONNX/DirectML demucs worker (AnyStemDeck addition).

Sibling to demucs_worker.py: same persistent-process design, same stdin/
stderr job protocol, same output layout on disk -- but runs inference
through `demucs-onnx` (https://github.com/StemSplit/demucs-onnx) and ONNX
Runtime's DirectML execution provider instead of PyTorch, so separation gets
hardware acceleration on AMD and Intel GPUs on Windows, which PyTorch's
CUDA/MPS-only acceleration cannot reach at all. See docs/plan.md for the
background and the decisions behind the choices below.

Run as its own process: `python -m app.pipeline.demucs_onnx_worker dml`.
`app/pipeline/separate.py` spawns this instead of demucs_worker.py whenever
the resolved device is "dml"; nothing else about job dispatch changes.

Protocol: identical to demucs_worker.py --
  - Parent writes one JSON line to stdin per job:
      {"source": "<path>", "job_dir": "<path>", "shifts": 1}
  - "NN%" progress lines (see below for how partial this is), then one of:
      "@@DONE@@"                          -- job ok, worker keeps serving
      "@@ERROR@@<json-encoded message>"   -- job failed, worker exits(1)
  - EOF on stdin ends the worker's loop cleanly.
  - STEMDECK_PARENT_PID, if set, arms the same exit-with-parent watchdog
    demucs_worker.py uses -- duplicated here rather than factored into a
    shared module, deliberately: this keeps the new DirectML path from
    touching the existing PyTorch worker at all, at the cost of ~15 lines
    of duplication. See demucs_worker.py's own docstring for why the
    watchdog exists.

Three real differences from demucs_worker.py, and why:

1. Progress is partial. `demucs_onnx.separate()` has no per-chunk progress
   callback in its public API (only an internal tqdm bar over chunk count,
   which isn't reachable from outside without depending on private
   functions -- see docs/plan.md's Phase 1 notes on why that trade was
   rejected). For the `htdemucs_ft` specialist bag we get real checkpoints
   for free by calling `separate()` once per specialist stem ourselves
   instead of once for the whole bag: 4 calls, 25% each. For single-file
   models (`htdemucs`, `htdemucs_6s` -- the default) there is only ever one
   `separate()` call, so this worker reports no percentage for that case.
   `separate.py`'s reader treats an unmatched stderr line as inert tail
   context (only surfaced if the job later fails), so in practice the UI
   just holds at "Separating stems..." until "@@DONE@@" arrives. Accepted
   v1 tradeoff.

2. Output writing is our own, not demucs-onnx's. `demucs_onnx.write_wav()`
   quantizes straight to 16-bit PCM with no headroom check, so a stem that
   peaks above full scale would hard-clip. PyTorch's `save_audio(clip=
   "rescale")` avoids that by scaling the whole file down first when
   needed. `_write_stem()` below reproduces that exact formula (`wav /
   max(1.01 * wav.abs().max(), 1)`) so the same input produces the same
   (non-clipped) result regardless of which worker rendered it. This is
   also why jobs call `separate(..., output_dir=None)` and get arrays back
   in memory rather than letting the library write files itself.

3. "Best" quality (2x shift-averaged separation) has no demucs-onnx
   equivalent. `separate.py` already caps `shifts` to 1 before ever
   dispatching to this worker, so the check below is a defensive fallback
   for anything that calls this worker directly, not the normal path.

Sample rate: `demucs_onnx.separate()` resamples its output back to the
*input file's native rate* (documented behavior), unlike the PyTorch path,
which always writes at the model's fixed 44.1 kHz regardless of the
source's rate. That means a 48 kHz source produces 48 kHz stems here but
44.1 kHz stems on CUDA/MPS/CPU for the identical file -- a real, deliberate
difference worth testing against AnyStemDeck's player/mixer/beat-grid code
(Phase 5), not silently normalized away here.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import DEMUCS_MODEL, DEMUCS_ONNX_PRECISION
from app.core.process import process_exists

# The htdemucs_ft specialist bag's stems, in the order we call separate()
# for each -- see point 1 in the module docstring. Only meaningful when
# DEMUCS_MODEL == "htdemucs_ft"; single-file models compute every stem in
# one call regardless of this list.
_BAG_STEMS = ("drums", "bass", "other", "vocals")


def _write_stem(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a (channels, samples) float32 array as 16-bit PCM WAV, rescaling
    first if it would clip. Matches demucs.audio.save_audio(clip="rescale")
    exactly -- see point 2 in the module docstring."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    scale = max(1.01 * peak, 1.0)
    if scale > 1.0:
        audio = audio / scale
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sample_rate, subtype="PCM_16")


_GRAPH_OPT_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def _patch_demucs_onnx_for_directml() -> None:
    """Work around session-creation issues in demucs-onnx found while testing
    on an RX 7800 XT (docs/plan.md) -- reproducibly hits "Not enough memory
    resources are available to complete this operation" (HRESULT 0x8007000E)
    inside a *fused* node (DmlFusedNode_0_0) on ordinary chunk sizes, on a
    16 GB card, running a ~150 MB model. Two independent things demucs-onnx's
    _make_session() doesn't do that this patches:

    1. Disables memory-pattern optimization. ONNX Runtime's own docs state
       the DirectML execution provider does not support it and it must be
       turned off. Necessary but -- confirmed on real hardware -- NOT
       sufficient on its own to fix the error above.

    2. Makes graph optimization level configurable via
       ANYSTEMDECK_ONNX_GRAPH_OPT ("disable" | "basic" | "extended" | "all",
       default "basic"). The crash names a node ORT's own graph-fusion pass
       created (DmlFusedNode_0_0), and ORT_ENABLE_ALL is what triggers the
       more aggressive fusion passes -- worth testing whether a lower level
       avoids creating whatever fused kernel is over-allocating. This is a
       live debugging knob, not a settled fix -- see docs/plan.md for
       results across levels once tested.

    This patches the session factory in place rather than forking/vendoring
    the library over these two options. Safe to remove once demucs-onnx
    ships a fix upstream (worth filing there regardless -- #1 at least
    affects every DirectML user of that library, not just us)."""
    import onnxruntime as ort
    from demucs_onnx import inference as _dox_inference

    level_name = os.environ.get("ANYSTEMDECK_ONNX_GRAPH_OPT", "basic").strip().lower()
    level_attr = _GRAPH_OPT_LEVELS.get(level_name, _GRAPH_OPT_LEVELS["basic"])
    level = getattr(ort.GraphOptimizationLevel, level_attr)
    print(f"[demucs_onnx_worker] DirectML session patch: graph_optimization_level={level_attr}, "
          f"enable_mem_pattern=False (set ANYSTEMDECK_ONNX_GRAPH_OPT=disable|basic|extended|all to change)")

    def _patched_make_session(onnx_path, providers):
        sess_opts = ort.SessionOptions()
        is_dml = "DmlExecutionProvider" in providers
        sess_opts.graph_optimization_level = level if is_dml else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if is_dml:
            sess_opts.enable_mem_pattern = False
        return ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=list(providers))

    _dox_inference._make_session = _patched_make_session


def _run_one_job(req: dict) -> None:
    from demucs_onnx import separate as onnx_separate

    source = Path(req["source"])
    job_dir = Path(req["job_dir"])
    shifts = int(req.get("shifts", 1))
    if shifts > 1:
        # Defensive fallback only -- separate.py already caps this to 1
        # before dispatch. See point 3 in the module docstring.
        sys.stderr.write(
            "note: shift-averaging ('best' quality) is not supported on dml; "
            "running standard quality instead\n"
        )

    # A lightweight metadata read (no full decode) so we know what rate
    # separate()'s returned arrays are actually at -- see the module
    # docstring's note on sample rate. demucs_onnx.separate() resamples its
    # output back to this same rate internally; this is not a guess.
    native_sr = sf.info(str(source)).samplerate

    out_dir = job_dir / DEMUCS_MODEL / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    is_bag = DEMUCS_MODEL == "htdemucs_ft"
    n_steps = len(_BAG_STEMS) if is_bag else 1

    def _report_progress(step: int) -> None:
        if not is_bag:
            return  # no natural checkpoint inside a single-file model's one call
        pct = int(round(100 * step / n_steps))
        sys.stderr.write(f"{pct}%\n")
        sys.stderr.flush()

    if is_bag:
        for i, stem in enumerate(_BAG_STEMS):
            out = onnx_separate(
                source,
                output_dir=None,
                model=f"htdemucs_ft_{stem}",
                providers="dml",
                precision=DEMUCS_ONNX_PRECISION,
                progress=False,
                verbose=False,
            )
            _write_stem(out_dir / f"{stem}.wav", out[stem], native_sr)
            _report_progress(i + 1)
    else:
        out = onnx_separate(
            source,
            output_dir=None,
            model=DEMUCS_MODEL,
            providers="dml",
            precision=DEMUCS_ONNX_PRECISION,
            progress=False,
            verbose=False,
        )
        for stem_name, audio in out.items():
            _write_stem(out_dir / f"{stem_name}.wav", audio, native_sr)
        _report_progress(1)


_PARENT_POLL_SECONDS = 1.0


def _watch_parent(parent_pid: int) -> None:
    """Exit as soon as the process that spawned us is gone. See
    demucs_worker.py's identical helper for the full rationale -- duplicated
    here rather than shared, deliberately (see module docstring)."""
    while True:
        if not process_exists(parent_pid):
            sys.stderr.write("@@ERROR@@parent process exited\n")
            sys.stderr.flush()
            os._exit(1)
        time.sleep(_PARENT_POLL_SECONDS)


def _arm_parent_watchdog() -> None:
    raw = os.environ.get("STEMDECK_PARENT_PID", "").strip()
    if not raw:
        return
    try:
        parent_pid = int(raw)
    except ValueError:
        return
    if parent_pid <= 0 or parent_pid == os.getpid():
        return
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()


def main() -> None:
    _arm_parent_watchdog()
    _patch_demucs_onnx_for_directml()

    from demucs_onnx import prewarm

    # Mirrors demucs_worker.py's up-front get_model() call: pay the model
    # download + first ONNX-session-compile cost once at startup, rather
    # than on whatever job happens to arrive first.
    try:
        prewarm([DEMUCS_MODEL], precision=DEMUCS_ONNX_PRECISION, providers="dml")
    except Exception as e:
        sys.stderr.write(f"@@ERROR@@{json.dumps(f'model prewarm failed: {e}')}\n")
        sys.stderr.flush()
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            _run_one_job(req)
        except Exception as e:
            sys.stderr.write(f"@@ERROR@@{json.dumps(str(e))}\n")
            sys.stderr.flush()
            sys.exit(1)
        sys.stderr.write("@@DONE@@\n")
        sys.stderr.flush()


if __name__ == "__main__":
    main()

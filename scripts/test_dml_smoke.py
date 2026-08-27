"""Standalone DirectML smoke test for AnyStemDeck's ONNX separation path.

Run from the repo root, inside the environment `pip install -e .` was run
into:

    python scripts/test_dml_smoke.py
    python scripts/test_dml_smoke.py --model htdemucs
    python scripts/test_dml_smoke.py --try-all
    python scripts/test_dml_smoke.py path\\to\\a\\real\\song.mp3 --model htdemucs_ft_vocals

With no file argument, generates a short synthetic test tone (a pure
plumbing check: does DirectML run end to end at all). With a real audio
file, runs the same thing on real material so you can actually listen to
the result.

--try-all runs a spread of models one after another (single-file 4-stem,
single-file 6-stem, and one htdemucs_ft specialist) and keeps going past a
failure instead of stopping at the first one, so a single run tells us
whether a problem is general to DirectML on this machine or specific to one
model's graph -- see docs/plan.md's DirectML debugging log.

--split runs the DirectML ConvTranspose *workaround* instead of the
ordinary path: exports a split-ISTFT ONNX graph (app/pipeline/onnx_export),
runs the network on DirectML, and finishes the ISTFT in numpy on CPU. Only
run this after `scripts/validate_split_export.py` has passed on CPU for
the same --model -- this script assumes the split export is numerically
correct and is only checking whether DirectML crashes on it.

    python scripts/validate_split_export.py --checkpoint htdemucs_6s
    python scripts/test_dml_smoke.py --split --model htdemucs_6s

Env var ANYSTEMDECK_ONNX_GRAPH_OPT ("disable" | "basic" | "extended" |
"all", default "basic") controls graph optimization level -- see
demucs_onnx_worker.py's _patch_demucs_onnx_for_directml docstring.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# Mirrors demucs_onnx_worker.py's _GRAPH_OPT_LEVELS -- duplicated rather
# than imported since this is a standalone diagnostic script, deliberately
# decoupled from the production worker module (see that file's own
# _patch_demucs_onnx_for_directml docstring for the rationale).
_GRAPH_OPT_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}

print("ONNX Runtime providers available:", ort.get_available_providers())
if "DmlExecutionProvider" not in ort.get_available_providers():
    print("\nDmlExecutionProvider is NOT available -- stopping here.")
    print("Did `pip install --force-reinstall --no-deps onnxruntime-directml` run in THIS environment?")
    sys.exit(1)

# Reuse the real worker's exact output-writing logic and DirectML session
# patch, not a re-implementation of either -- this is the actual code that
# will run in production.
from app.pipeline.demucs_onnx_worker import (  # noqa: E402
    _patch_demucs_onnx_for_directml,
    _write_stem,
)

_patch_demucs_onnx_for_directml()

from demucs_onnx import separate as onnx_separate  # noqa: E402

# A representative spread for --try-all: single-file 4-stem (simplest
# graph), single-file 6-stem (AnyStemDeck's default -- what actually failed
# first), and one htdemucs_ft specialist (the 4-network bag family, a
# structurally different export path). If only some of these fail, that
# narrows the bug to specific graphs rather than DirectML/the environment
# in general.
_TRY_ALL_MODELS = ["htdemucs", "htdemucs_6s", "htdemucs_ft_vocals"]


def _run_one(source: Path, model: str, native_sr: int) -> tuple[bool, str]:
    """Returns (ok, message)."""
    print(f"\n{'=' * 60}\nmodel={model!r}\n{'=' * 60}")
    t0 = time.time()
    try:
        out = onnx_separate(
            source,
            output_dir=None,
            model=model,
            providers="dml",
            precision="fp16weights",
            progress=True,
            verbose=True,
        )
    except Exception as e:
        traceback.print_exc()
        return False, f"FAILED after {time.time() - t0:.1f}s: {e.__class__.__name__}: {e}"

    elapsed = time.time() - t0
    out_dir = Path("smoke_out") / model
    for stem_name, audio in out.items():
        peak = float(np.abs(audio).max())
        print(f"  {stem_name}: shape={audio.shape} peak={peak:.3f}")
        _write_stem(out_dir / f"{stem_name}.wav", audio, native_sr)
    return True, f"OK in {elapsed:.1f}s -- stems: {list(out.keys())} -- written to {out_dir}"


def _run_one_split(source: Path, model: str, native_sr: int, *,
                    provider: str = "dml") -> tuple[bool, str]:
    """Same shape as _run_one, but exercises the DirectML ConvTranspose
    workaround instead of the ordinary demucs-onnx path -- see this file's
    module docstring and app/pipeline/onnx_export/export_split.py.

    `provider`: "dml" (default) or "cpu" -- pass "cpu" to run the *exact
    same* split-graph + chunking + ispec_numpy pipeline through plain CPU
    execution instead of DirectML. This isolates whether an anomaly (e.g.
    implausibly large peak values on real audio) comes from this project's
    own chunking code or from DirectML itself: if --provider cpu is also
    wrong, the bug is ours; if only --provider dml is wrong, it's a
    DirectML numerical-accuracy issue distinct from the ConvTranspose
    crash this whole module works around."""
    import tempfile

    from demucs_onnx._audio import load_audio, resample_to_native

    from app.pipeline.onnx_export import export_split_to_onnx, ispec_numpy, make_istft_kernels_numpy
    from app.pipeline.onnx_export.export_split import N_SAMPLES, SAMPLE_RATE

    ort_provider = "DmlExecutionProvider" if provider == "dml" else "CPUExecutionProvider"
    print(f"\n{'=' * 60}\nmodel={model!r} (split, provider={provider})\n{'=' * 60}")
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "split.onnx"
            print("  exporting split ONNX graph...")
            export_split_to_onnx(model, onnx_path, verbose=False)

            print(f"  creating {ort_provider} session for the split graph...")
            sess_opts = ort.SessionOptions()
            sess_opts.enable_mem_pattern = False  # see demucs_onnx_worker.py's patch -- necessary, if not sufficient, for the *ordinary* graph
            # Honor the same ANYSTEMDECK_ONNX_GRAPH_OPT knob the ordinary
            # (non-split) DirectML path respects (demucs_onnx_worker.py's
            # _patch_demucs_onnx_for_directml) -- this session was
            # previously always created with ORT's *default* level
            # (ORT_ENABLE_ALL, full fusion), ignoring whatever level was
            # set in the environment. That's a real suspect for the huge
            # (300-9000x) peak values seen on a real song despite the
            # random-noise CPU parity check passing: DirectML's own fused
            # kernels are exactly what the original ConvTranspose crash
            # implicated, and "silently wrong" instead of "crashes" is a
            # plausible failure mode for the same root cause. Defaulting
            # to "basic" here rather than ORT's own "all" default.
            level_name = os.environ.get("ANYSTEMDECK_ONNX_GRAPH_OPT", "basic").strip().lower()
            level_attr = _GRAPH_OPT_LEVELS.get(level_name, _GRAPH_OPT_LEVELS["basic"])
            sess_opts.graph_optimization_level = getattr(ort.GraphOptimizationLevel, level_attr)
            print(f"  graph_optimization_level={level_attr} (ANYSTEMDECK_ONNX_GRAPH_OPT={level_name!r})")
            sess = ort.InferenceSession(
                str(onnx_path), sess_options=sess_opts, providers=[ort_provider],
            )

            audio, _native_sr_check = load_audio(source, target_sr=SAMPLE_RATE)
            kernels = make_istft_kernels_numpy()

            overlap = N_SAMPLES // 4
            stride = N_SAMPLES - overlap
            total_len = audio.shape[1]
            n_chunks = max(1, (total_len + stride - 1) // stride)
            window = np.ones(N_SAMPLES, dtype=np.float32)
            transition = N_SAMPLES // 4
            fade = np.linspace(0, 1, transition, dtype=np.float32)
            window[:transition] = fade
            window[-transition:] = fade[::-1]

            out = None  # allocated once we know the source count
            weight = np.zeros(total_len, dtype=np.float32)
            for i in range(n_chunks):
                start = i * stride
                end = min(start + N_SAMPLES, total_len)
                chunk = audio[:, start:end]
                if chunk.shape[1] < N_SAMPLES:
                    chunk = np.pad(chunk, ((0, 0), (0, N_SAMPLES - chunk.shape[1])))
                x = chunk[np.newaxis, ...].astype(np.float32, copy=False)

                zspec, xt = sess.run(["zspec", "xt"], {"mix": x})
                stems = ispec_numpy(zspec, xt, length=N_SAMPLES, kernels=kernels)[0]  # (S, 2, N)
                if out is None:
                    out = np.zeros((stems.shape[0], 2, total_len), dtype=np.float32)

                chunk_len = end - start
                w = window[:chunk_len]
                out[:, :, start:end] += stems[:, :, :chunk_len] * w
                weight[start:end] += w
                print(f"    chunk {i + 1}/{n_chunks}: {time.time() - t0:.1f}s elapsed")

            weight = np.maximum(weight, 1e-8)
            out /= weight
            out = resample_to_native(out, SAMPLE_RATE, native_sr) if native_sr != SAMPLE_RATE else out
    except Exception as e:
        traceback.print_exc()
        return False, f"FAILED after {time.time() - t0:.1f}s: {e.__class__.__name__}: {e}"

    elapsed = time.time() - t0
    # Include the source filename so two runs against different files can
    # never leave stale output from a previous run sitting where the new
    # one is expected -- each input gets its own subfolder.
    out_dir = Path("smoke_out") / f"{model}_split_{provider}" / source.stem
    # Split export always produces the model's full stem set in a fixed
    # order; name them generically here since we don't have model metadata
    # handy in this smoke test -- fine for a plumbing check.
    for i in range(out.shape[0]):
        peak = float(np.abs(out[i]).max())
        print(f"  stem[{i}]: shape={out[i].shape} peak={peak:.3f}")
        _write_stem(out_dir / f"stem_{i}.wav", out[i], native_sr)
    return True, f"OK in {elapsed:.1f}s -- {out.shape[0]} stems -- written to {out_dir.resolve()}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", help="Real audio file to test with (default: synthetic tone)")
    p.add_argument("--model", default="htdemucs_6s", help="Model to run (default: htdemucs_6s)")
    p.add_argument("--try-all", action="store_true", help=f"Run all of {_TRY_ALL_MODELS} in sequence")
    p.add_argument("--split", action="store_true",
                    help="Use the DirectML ConvTranspose workaround (split-ISTFT export) instead "
                         "of the ordinary path. Run validate_split_export.py first.")
    p.add_argument("--provider", choices=["dml", "cpu"], default="dml",
                    help="--split only: which execution provider runs the split graph. "
                         "Use --provider cpu to isolate a DirectML-specific numerical issue "
                         "from a bug in this project's own chunking code.")
    args = p.parse_args()

    if args.file:
        source = Path(args.file)
        if not source.is_file():
            print(f"file not found: {source}")
            sys.exit(1)
        print(f"using real file: {source}")
    else:
        source = Path("smoke_test_tone.wav")
        sr = 44100
        dur = 8  # a real chunk is 7.8s -- long enough to exercise one full segment
        t = np.linspace(0, dur, sr * dur, endpoint=False)
        audio = np.stack(
            [np.sin(2 * np.pi * 220 * t) * 0.5, np.sin(2 * np.pi * 330 * t) * 0.5]
        ).T.astype("float32")
        sf.write(source, audio, sr)
        print(f"no file given -- generated synthetic test tone: {source}")

    native_sr = sf.info(str(source)).samplerate
    models = _TRY_ALL_MODELS if args.try_all else [args.model]
    if args.split and args.try_all:
        print("--split does not support --try-all (single model only for now); pick one with --model")
        sys.exit(1)

    results = {}
    for model in models:
        if args.split:
            results[model] = _run_one_split(source, model, native_sr, provider=args.provider)
        else:
            results[model] = _run_one(source, model, native_sr)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for model, (ok, msg) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {model:20s} {msg}")


if __name__ == "__main__":
    main()

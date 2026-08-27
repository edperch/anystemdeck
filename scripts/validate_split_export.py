"""Validate the split-ISTFT ONNX export before ever touching DirectML.

This is the de-risking step for the DirectML `ConvTranspose` workaround
(see docs/plan.md and app/pipeline/onnx_export/export_split.py's module
docstring): it exports a split ONNX graph, runs it on CPU (via plain
`onnxruntime`, no DirectML involved), reconstructs the final waveform with
`app.pipeline.onnx_export.ispec_numpy`, and compares the result against
the *original*, unmodified PyTorch model run on the exact same input.

If this fails, the bug is in this project's export/ISTFT code (most likely
`split_istft.py`'s numpy transcription), not in DirectML -- fix it here,
on CPU, where it's fast and easy to debug, before spending a GPU test run
on it. If this passes, the split graph is numerically correct and the
*only* remaining unknown is whether DirectML actually runs it without
crashing -- that's `scripts/test_dml_smoke.py --split`.

Usage (from the repo root, same environment `pip install -e .` ran into --
this needs `torch`+`demucs`, which are already core StemDeck dependencies,
plus `onnx`+`onnxruntime`, already present from the DirectML work):

    python scripts/validate_split_export.py
    python scripts/validate_split_export.py --checkpoint htdemucs
    python scripts/validate_split_export.py --checkpoint htdemucs_ft --stem drums

Runs entirely on CPU. Takes a couple of minutes (dominated by downloading
the pretrained checkpoint on first run, same as any other demucs/StemDeck
first use) -- no GPU, no DirectML, no AMD-specific anything required.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

DEFAULT_TOLERANCE = 1e-3  # matches demucs-onnx's own parity_check tolerance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="htdemucs_6s",
                    help="Pretrained checkpoint name (default: htdemucs_6s, AnyStemDeck's default model)")
    p.add_argument("--stem", default=None,
                    help="For a specialist bag (htdemucs_ft): which stem to export/check")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = p.parse_args()

    from demucs.pretrained import get_model

    from app.pipeline.onnx_export import export_split_to_onnx, ispec_numpy, make_istft_kernels_numpy
    from app.pipeline.onnx_export.export_split import N_SAMPLES

    print(f"Loading pretrained model: {args.checkpoint!r}"
          + (f" (stem={args.stem!r})" if args.stem else ""))
    obj = get_model(args.checkpoint)
    # demucs.pretrained.get_model wraps every checkpoint -- including
    # single-file models like htdemucs_6s -- in a BagOfModels. Only treat
    # it as a *specialist* bag (needs --stem) when it actually holds more
    # than one sub-model; a single-submodel bag is exported the same as
    # any other single-file model, using that one sub-model directly (the
    # wrapper itself has no _spec/encoder/etc of its own to trace).
    is_bag = hasattr(obj, "models")
    is_specialist_bag = is_bag and len(obj.models) > 1
    if is_specialist_bag:
        if args.stem is None:
            print("error: this checkpoint is a specialist bag; pass --stem drums|bass|other|vocals")
            sys.exit(1)
        from app.pipeline.onnx_export.export_split import STEM_TO_INDEX
        original = obj.models[STEM_TO_INDEX[args.stem]].eval().to("cpu")
    elif is_bag:
        original = obj.models[0].eval().to("cpu")
    else:
        original = obj.eval().to("cpu")

    torch.manual_seed(0)
    dummy = torch.randn(1, 2, N_SAMPLES, dtype=torch.float32)
    print(f"Running original PyTorch model on a {N_SAMPLES:,}-sample dummy input (ground truth)...")
    with torch.no_grad():
        out_torch = original(dummy).numpy()  # (1, S, C, N_SAMPLES)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "split.onnx"
        print("\nExporting split ONNX graph...")
        export_split_to_onnx(args.checkpoint, onnx_path, stem=args.stem, verbose=True)

        print("\nRunning split graph on CPU (onnxruntime, no DirectML)...")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        zspec, xt = sess.run(["zspec", "xt"], {"mix": dummy.numpy()})

        print("Reconstructing waveform with ispec_numpy (the CPU-side ISTFT this project wrote)...")
        kernels = make_istft_kernels_numpy()
        out_split = ispec_numpy(zspec, xt, length=N_SAMPLES, kernels=kernels)

    diff = float(np.abs(out_torch - out_split).max())
    print(f"\nmax abs diff vs original PyTorch model: {diff:.6g} (tolerance {args.tolerance:g})")
    if diff > args.tolerance:
        print("FAIL -- the split export + numpy ISTFT diverges from the original model.")
        print("Do not proceed to scripts/test_dml_smoke.py --split until this passes --")
        print("the bug is in this project's code, not DirectML.")
        sys.exit(1)

    print("PASS -- split export + numpy ISTFT reproduces the original model.")
    print("Safe to move on to: python scripts\\test_dml_smoke.py --split")


if __name__ == "__main__":
    main()

"""AnyStemDeck addition: export htdemucs to ONNX with ISTFT split out of the
graph, to route around a DirectML `ConvTranspose` bug that otherwise crashes
every model tested on real hardware. See docs/plan.md ("DirectML ConvTranspose
blocker" in Phase 1, and the "Phase 1.5" section for this workaround) for the
full story and `export_split.py`'s module docstring for the technical detail.

Public entry points:
    export_split_to_onnx(checkpoint, output, ...) -- like
        demucs_onnx.export.export_to_onnx, but the returned ONNX graph
        outputs (zspec, xt) -- the pre-ISTFT masked spectrogram and the
        time-branch waveform -- instead of final waveform stems.
    ispec_numpy(zspec, xt, length) -- finishes the job the graph didn't:
        runs the same ISTFT math as demucs's `_ispec`/`RealISTFT`, in numpy,
        on CPU, and adds it to `xt` to produce the final waveform. This is
        the *inverse* of what got cut from the graph -- keep it in lockstep
        with `demucs_onnx.export.stft.RealISTFT` if that ever changes.

This whole package requires `torch`+`demucs` (export side) or just `numpy`
(inference side, `ispec_numpy` only) -- mirrors demucs_onnx.export's own
lazy-import discipline so importing this package doesn't require torch on
the inference-only path.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["export_split_to_onnx", "ispec_numpy", "make_istft_kernels_numpy"]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "export_split_to_onnx": ("app.pipeline.onnx_export.export_split", "export_split_to_onnx"),
    "ispec_numpy": ("app.pipeline.onnx_export.split_istft", "ispec_numpy"),
    "make_istft_kernels_numpy": ("app.pipeline.onnx_export.split_istft", "make_istft_kernels_numpy"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        mod_path, attr = _LAZY_MAP[name]
        module = import_module(mod_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

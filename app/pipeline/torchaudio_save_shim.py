"""Replaces ``torchaudio.save`` with a ``soundfile``-based implementation.

Why this exists: demucs 4.0.1's own stem-writing code (``demucs/audio.py``'s
``save_audio()``) calls ``torchaudio.save(path, wav, sample_rate=..., encoding=...,
bits_per_sample=...)`` -- the classic dispatcher-backend signature. That still
works fine on torchaudio 2.6 (what this project shipped with originally), but by
torchaudio 2.9 -- the version paired with the AMD ROCm 7.2.1 PyTorch wheel this
project's WSL2/ROCm backend needs (see docs/plan.md's WSL2 GPU thread) --
``torchaudio.save`` is just an alias for ``save_with_torchcodec()``, a different
API that does not accept ``encoding``/``bits_per_sample`` at all. Calling it as
demucs does would raise a ``TypeError`` for unexpected keyword arguments, so
every separation job would fail at the very last step: writing the stems.

Rather than vendor ``torchcodec`` (a much bigger, FFmpeg-backed dependency) just
to write PCM WAV/FLAC files, or pin torch/torchaudio below 2.7 forever and give
up on newer ROCm/CUDA wheels, this patches ``torchaudio.save`` in place with a
small ``soundfile``-based replacement covering exactly the parameter space
``save_audio()`` can produce. ``soundfile`` is already a hard project dependency
for this exact kind of PCM writing (see pyproject.toml), and the demucs-onnx/
DirectML worker (``demucs_onnx_worker.py``'s ``_write_stem``) already writes
stems this same way -- this is that same approach, generalized into a drop-in
``torchaudio.save`` replacement so demucs's own code needs no changes at all.

Only ``torchaudio.save`` is patched. ``torchaudio.load`` is untouched and does
not need to be: demucs's ``load_track()`` (``demucs/separate.py``) always reads
audio via its own ffmpeg-based ``AudioFile.read()`` first, falling back to
``torchaudio.load()`` only if ffmpeg is missing or fails -- and this project
manages/verifies its own ffmpeg install (see app/core/config.py), so that
fallback is never actually reached in practice.
"""

from __future__ import annotations

import torch

# torchaudio's classic (uri, src, sample_rate, encoding=, bits_per_sample=)
# signature names a small, fixed set of encodings. This maps every
# (encoding, bits_per_sample) pair torchaudio itself documented for that old
# backend to the matching soundfile ``subtype`` string -- not narrowed to just
# this project's one current call site (demucs's save_audio() with
# encoding="PCM_S", bits_per_sample=16), so a future change to how demucs (or
# anything else) calls torchaudio.save keeps working without touching this map.
_SUBTYPES: dict[tuple[str, int | None], str] = {
    ("PCM_S", 8): "PCM_S8",
    ("PCM_S", 16): "PCM_16",
    ("PCM_S", 24): "PCM_24",
    ("PCM_S", 32): "PCM_32",
    ("PCM_U", 8): "PCM_U8",
    ("PCM_F", 32): "FLOAT",
    ("PCM_F", 64): "DOUBLE",
    ("ULAW", None): "ULAW",
    ("ALAW", None): "ALAW",
}


def _resolve_subtype(encoding: str | None, bits_per_sample: int | None) -> str | None:
    """Mirrors torchaudio's own encoding/bits_per_sample -> subtype resolution
    closely enough for every combination demucs actually produces. Returns
    None (soundfile's own per-format default subtype) when neither is given."""
    if encoding in ("ULAW", "ALAW"):
        return _SUBTYPES[(encoding, None)]
    if encoding is not None and bits_per_sample is not None:
        try:
            return _SUBTYPES[(encoding, bits_per_sample)]
        except KeyError:
            raise ValueError(
                "torchaudio_save_shim: unsupported encoding/bits_per_sample "
                f"combination {encoding!r}/{bits_per_sample!r} -- add it to "
                "_SUBTYPES if this is a real, intentional combination."
            ) from None
    if bits_per_sample is not None:
        # No explicit encoding: torchaudio's own default is signed PCM.
        return _resolve_subtype("PCM_S", bits_per_sample)
    return None


def _save_via_soundfile(
    uri,
    src: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    format: str | None = None,
    encoding: str | None = None,
    bits_per_sample: int | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
    compression=None,
) -> None:
    """Drop-in replacement for ``torchaudio.save`` (its classic signature),
    writing through ``soundfile`` instead of torchaudio's own encoder. Ignores
    ``buffer_size``/``backend``/``compression`` -- none of torchaudio's old
    backends gave ``save_audio()`` (or anything else in this project) a reason
    to pass anything but their defaults, so there is nothing to translate."""
    import soundfile as sf

    array = src.detach().to("cpu").contiguous().numpy()
    if array.ndim == 2 and channels_first:
        # torchaudio: (channel, time) -> soundfile: (time, channel).
        array = array.T
    sf.write(
        str(uri),
        array,
        sample_rate,
        subtype=_resolve_subtype(encoding, bits_per_sample),
        format=format,
    )


def install() -> None:
    """Monkeypatch ``torchaudio.save`` in place. Idempotent, and safe to call
    at any point -- demucs's own ``import torchaudio as ta`` binds the module
    object, not the function, so its later ``ta.save(...)`` calls resolve
    against whatever ``torchaudio.save`` is *at call time*, regardless of
    whether this patch ran before or after demucs itself was imported."""
    import torchaudio

    torchaudio.save = _save_via_soundfile

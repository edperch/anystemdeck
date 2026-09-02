"""Pure-numpy ISTFT: the CPU-side half of the DirectML ConvTranspose
workaround (AnyStemDeck addition).

This is a numpy transcription of two pieces of `demucs-onnx`'s export code
(https://github.com/StemSplit/demucs-onnx, MIT), reproduced here to run
*after* ONNX Runtime rather than *inside* the graph:

  - `demucs_onnx.export.stft.RealISTFT.forward` -- the actual inverse-STFT
    math (DFT synthesis + overlap-add), originally implemented as
    `F.conv_transpose1d` with a `(n_fft=4096)`-wide kernel. That specific op
    is what crashes DirectML (see docs/plan.md) -- everything in this file
    computes the identical result with plain numpy matrix multiply +
    overlap-add instead of an actual (transposed-)convolution, so there is
    no ConvTranspose node anywhere for DirectML to choke on.
  - `demucs_onnx.export.patch.patch_htdemucs_for_onnx`'s `_ispec_real`
    closure -- the padding/cropping wrapper around RealISTFT that matches
    HTDemucs's own `_ispec` API (restoring the dropped Nyquist bin, the
    symmetric time-axis pad, the pre/post crop).

Keep this in lockstep with `demucs_onnx.export.stft` if that module ever
changes upstream -- these two are meant to compute bit-identical results,
just on different hardware. `scripts/validate_split_export.py` is the
regression check: it compares this path's output against the ordinary
(un-split) demucs-onnx ONNX graph running the ISTFT itself on CPU, and
fails loudly on any drift.
"""

from __future__ import annotations

import math

import numpy as np

N_FFT = 4096
HOP_LENGTH = N_FFT // 4  # 1024, matches export_split.py / demucs-onnx's fixed config


def _periodic_hann(n_fft: int) -> np.ndarray:
    """torch.hann_window(n_fft, periodic=True) in float64, reproduced
    exactly: w[n] = 0.5 - 0.5*cos(2*pi*n/n_fft) for n in [0, n_fft)."""
    n = np.arange(n_fft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2 * math.pi * n / n_fft)


def make_istft_kernels_numpy(n_fft: int = N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """Build the `(inv_cos, inv_sin)` synthesis kernels, shape `(n_bins, n_fft)`
    each, float32. Bit-for-bit the same construction as
    `RealISTFT.__init__` in `demucs_onnx/export/stft.py` -- see that
    module's docstring for the reconstruction formula this implements.
    """
    window = _periodic_hann(n_fft)
    norm = 1.0 / math.sqrt(n_fft)
    n_bins = n_fft // 2 + 1
    n = np.arange(n_fft, dtype=np.float64)
    k = np.arange(n_bins, dtype=np.float64)[:, None]
    angles = 2 * math.pi * k * n[None, :] / n_fft

    inv_cos = 2.0 * (window * np.cos(angles)) * norm
    inv_sin = -2.0 * (window * np.sin(angles)) * norm
    inv_cos[0] *= 0.5  # DC: appears once, not twice (see RealISTFT docstring)
    inv_cos[-1] *= 0.5  # Nyquist: same
    inv_sin[0] *= 0.0  # imag part of DC/Nyquist is zero by construction
    inv_sin[-1] *= 0.0
    return inv_cos.astype(np.float32), inv_sin.astype(np.float32)


# Envelope only depends on (n_frames, length, n_fft, hop_length) -- cache
# across calls the same way RealISTFT._envelope_cache does (chunked
# inference reuses the same shape for every chunk but the last).
_ENVELOPE_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}


def _envelope(n_frames: int, length: int, n_fft: int, hop_length: int) -> np.ndarray:
    key = (n_frames, length, n_fft, hop_length)
    cached = _ENVELOPE_CACHE.get(key)
    if cached is not None:
        return cached
    win_sq = _periodic_hann(n_fft) ** 2
    env = np.zeros(length + n_fft, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        env[start : start + n_fft] += win_sq
    pad = n_fft // 2
    env = env[pad : pad + length]
    env = np.clip(env, 1e-11, None)
    env = env.astype(np.float32)
    _ENVELOPE_CACHE[key] = env
    return env


def _real_istft_numpy(
    z: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    inv_cos: np.ndarray,
    inv_sin: np.ndarray,
    length: int,
) -> np.ndarray:
    """numpy transcription of `RealISTFT.forward`. `z`: `(..., 2, F, T)`
    float32. Returns `(..., length)` float32.

    The original does this via `F.conv_transpose1d(real, inv_cos,
    stride=hop_length)` (and the same for imag/inv_sin), which for a
    `(F, 1, n_fft)`-shaped kernel and `out_channels=1` is exactly: for each
    frame `t`, take `real[:, :, t] @ inv_cos` (a length-`n_fft` vector) and
    add it into the output at `t * hop_length`. That's the definition of
    overlap-add DFT synthesis -- computed here with an einsum (batched
    matmul) plus an explicit OLA loop instead of an actual (transposed)
    convolution op.
    """
    *other, two, freqs, frames = z.shape
    if two != 2:
        raise ValueError(f"expected (...,2,F,T), got {z.shape}")
    z = z.reshape(-1, 2, freqs, frames).astype(np.float32, copy=False)
    real = z[:, 0]  # (BN, F, T)
    imag = z[:, 1]

    # (BN, T, n_fft): per-frame synthesized windows, cos + sin contributions.
    frame_cos = np.einsum("bft,fk->btk", real, inv_cos, optimize=True)
    frame_sin = np.einsum("bft,fk->btk", imag, inv_sin, optimize=True)
    frame_sig = (frame_cos + frame_sin).astype(np.float32)

    bn = frame_sig.shape[0]
    padded_len = (frames - 1) * hop_length + n_fft
    out = np.zeros((bn, padded_len), dtype=np.float32)
    # OLA: python loop over frames, vectorized over batch/n_fft. `frames`
    # is small (~336 for a 7.8s segment at hop=1024), so this is not the
    # bottleneck -- the einsums above dominate and are already vectorized.
    for t in range(frames):
        start = t * hop_length
        out[:, start : start + n_fft] += frame_sig[:, t, :]

    pad = n_fft // 2
    end = pad + length
    x = out[:, pad:end]
    env = _envelope(frames, x.shape[-1], n_fft, hop_length)
    x = x / env
    return x.reshape(*other, x.shape[-1])


def ispec_numpy(
    zspec: np.ndarray,
    xt: np.ndarray,
    length: int,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    kernels: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Finish what the split ONNX graph didn't: run ISTFT on `zspec` (the
    masked spectrogram output, shape `(B, S, C, 2, F, T)`) and add the
    time-branch waveform `xt` (shape `(B, S, C, length)`) to get the final
    separated waveform -- the same combine `HTDemucs.forward` does as its
    very last step, just moved out here.

    `length` must be the fixed export segment length in samples (`N_SAMPLES`
    = 343980 for the standard 7.8s @ 44.1kHz config) -- see
    `export_split.py`'s docstring for why the split forward doesn't need to
    handle any other value.

    `kernels`: pass a cached `(inv_cos, inv_sin)` pair (from
    `make_istft_kernels_numpy`) to avoid recomputing it on every call --
    the split worker should build it once at startup.
    """
    if kernels is None:
        kernels = make_istft_kernels_numpy(n_fft)
    inv_cos, inv_sin = kernels

    # Mirrors patch.py's `_ispec_real` exactly (scale is always 0 here --
    # HTDemucs only calls _ispec with scale!=0 from a code path this
    # project's fixed-segment export never exercises):
    hl = hop_length
    # Restore the Nyquist bin _spec_real dropped: pad the F axis (-2) by
    # (0, 1). z shape (..., 2, F, T).
    pad_width = [(0, 0)] * (zspec.ndim - 2) + [(0, 1), (0, 0)]
    z = np.pad(zspec, pad_width)
    # Symmetric pad on the T axis (-1) by (2, 2).
    pad_width = [(0, 0)] * (z.ndim - 1) + [(2, 2)]
    z = np.pad(z, pad_width)

    outer_pad = hl // 2 * 3
    le = hl * math.ceil(length / hl) + 2 * outer_pad
    x = _real_istft_numpy(
        z, n_fft=n_fft, hop_length=hl, inv_cos=inv_cos, inv_sin=inv_sin, length=le
    )
    x = x[..., outer_pad : outer_pad + length]  # (B, S, C, length)

    return (xt + x).astype(np.float32)

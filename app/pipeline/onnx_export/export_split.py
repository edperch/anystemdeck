"""Export htdemucs to ONNX with ISTFT split out of the graph (AnyStemDeck
addition -- workaround for a DirectML `ConvTranspose` bug).

## The problem this works around

On real hardware (RX 7800 XT), every model exported by the normal
`demucs_onnx.export.export_to_onnx` crashes on DirectML at the same node,
`/real_istft/ConvTranspose_1` -- HRESULT 0x8007000E, "not enough memory
resources," reproducible even with GPU usage nowhere near the card's
16GB, and even with graph fusion fully disabled. An unrelated ONNX model
(Kokoro TTS) hits the identical failure. This reads as a genuine,
currently-unfixed DirectML platform bug with `ConvTranspose` for
large-kernel transposed convolutions -- the exact pattern
`demucs_onnx.export.stft.RealISTFT` uses to implement inverse-STFT
(kernel size = n_fft = 4096, far larger than the small kernels the
network's own U-Net-style decoder uses elsewhere in the graph, which do
NOT crash). See docs/plan.md for the full debugging log.

## The workaround

`RealISTFT` is only ever invoked once per forward pass, right at the end
(`HTDemucs.forward`'s `x = self._ispec(zout, length)`, immediately before
combining with the time-branch waveform and returning). Everything before
that point -- the entire transformer/convolutional network, which is the
overwhelming majority of the actual compute -- has no ConvTranspose
problem. So instead of patching around the crash, this module changes
*what gets exported*: `_split_forward` below is a copy of
`demucs.htdemucs.HTDemucs.forward` (demucs==4.0.1) truncated right before
the `_ispec` call, returning the masked spectrogram (`zspec`) and the
time-branch waveform (`xt`) as two graph outputs instead of the finished
waveform. `split_istft.ispec_numpy` (a numpy, not ONNX, reimplementation
of the exact same ISTFT math) then runs on CPU afterward to finish the
job -- so DirectML never sees a ConvTranspose node at all, and the GPU
still does essentially all the real work.

## A fixed-length simplification this relies on

The original `forward()` branches on `use_train_segment`/`self.training`
and a `length_pre_pad` case (when the input is shorter than the model's
training segment) to decide which length value to pass to `_ispec` and
`xt.view(...)`. Every demucs-onnx export -- ours included -- always
traces with a dummy input whose length is *exactly* the training segment
length (`N_SAMPLES` = 343980, i.e. 7.8s @ 44.1kHz): `demucs-onnx`'s own
chunked inference (`_chunked_separate_single` et al) zero-pads every
chunk, including a short final one, up to exactly that length before
calling the model, and crops the result back down afterward. Given that
invariant, `mix.shape[-1] < training_length` is always False at
export/trace time, so `length_pre_pad` is always `None` and both branches
of the `use_train_segment` conditional resolve to the same value
(`training_length`). `_split_forward` below hardcodes that instead of
reproducing the branching -- it is not a simplification of the *math*,
only of dead branches that a fixed-length export can never take. If a
future version ever exports a variable-length graph, this assumption
needs revisiting.

## Usage

    from app.pipeline.onnx_export import export_split_to_onnx
    export_split_to_onnx("htdemucs_6s", "out/htdemucs_6s_split.onnx")
    export_split_to_onnx("htdemucs_ft", "out/", stem="drums")

Requires `torch` + `demucs` (the export/authoring side only -- the
inference side, `split_istft.ispec_numpy`, needs only numpy). Run
`scripts/validate_split_export.py` after exporting, before ever touching
DirectML with the result -- it catches transcription bugs in this file or
in `split_istft.py` by comparing against the ordinary (un-split) model on
CPU, where both should agree to numerical noise.
"""
from __future__ import annotations

import copy
import types
from pathlib import Path
from typing import Any

import torch
from einops import rearrange

# Reuse demucs-onnx's own, already-verified patches for the three blockers
# unrelated to ISTFT (Fraction segment, random pos-embedding shift, fused
# MHA) -- only the STFT/ISTFT patch and the tail of forward() differ here.
from demucs_onnx.export import coerce_segment_to_float, disable_random_pos_shift, onnx_friendly_mha_forward
from demucs_onnx.export.stft import RealSTFT

STEM_TO_INDEX = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}
SAMPLE_RATE = 44100
SEGMENT_S = 7.8
N_SAMPLES = int(SEGMENT_S * SAMPLE_RATE)  # 343,980 -- must match split_istft's callers


def _split_forward(self: Any, mix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy of `demucs.htdemucs.HTDemucs.forward` (demucs==4.0.1), truncated
    immediately before the `_ispec` (ISTFT) call. Returns `(zspec, xt)`
    instead of the finished waveform -- see this module's docstring for
    why, and `split_istft.ispec_numpy` for the code that finishes the job.
    Every line up to the truncation point is unchanged from upstream
    (including the "okay, this is a giant mess I know..." comment, kept
    as a marker for diffing against a future demucs upgrade)."""
    length = mix.shape[-1]
    training_length = int(self.segment * self.samplerate)
    # See module docstring: length_pre_pad is always None for a fixed
    # N_SAMPLES-length export/trace, so both branches upstream would take
    # here collapse to `training_length` -- assert that invariant instead
    # of silently mis-exporting if it's ever violated.
    if length != training_length:
        raise ValueError(
            f"_split_forward requires a fixed-length input equal to the "
            f"training segment ({training_length} samples); got {length}. "
            "This is an export/trace-time invariant, not a runtime one -- "
            "see this module's docstring.",
        )

    z = self._spec(mix)
    mag = self._magnitude(z).to(mix.device)
    x = mag

    B, C, Fq, T = x.shape

    # unlike previous Demucs, we always normalize because it is easier.
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)
    # x will be the freq. branch input.

    # Prepare the time branch input.
    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    xt = (xt - meant) / (1e-5 + stdt)

    # okay, this is a giant mess I know...
    saved = []  # skip connections, freq.
    saved_t = []  # skip connections, time.
    lengths = []  # saved lengths to properly remove padding, freq branch.
    lengths_t = []  # saved lengths for time branch.
    for idx, encode in enumerate(self.encoder):
        lengths.append(x.shape[-1])
        inject = None
        if idx < len(self.tencoder):
            # we have not yet merged branches.
            lengths_t.append(xt.shape[-1])
            tenc = self.tencoder[idx]
            xt = tenc(xt)
            if not tenc.empty:
                # save for skip connection
                saved_t.append(xt)
            else:
                # tenc contains just the first conv., so that now time and freq.
                # branches have the same shape and can be merged.
                inject = xt
        x = encode(x, inject)
        if idx == 0 and self.freq_emb is not None:
            # add frequency embedding to allow for non equivariant convolutions
            # over the frequency axis.
            frs = torch.arange(x.shape[-2], device=x.device)
            emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
            x = x + self.freq_emb_scale * emb

        saved.append(x)
    if self.crosstransformer:
        if self.bottom_channels:
            b, c, f, t = x.shape
            x = rearrange(x, "b c f t-> b c (f t)")
            x = self.channel_upsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = self.channel_upsampler_t(xt)

        x, xt = self.crosstransformer(x, xt)

        if self.bottom_channels:
            x = rearrange(x, "b c f t-> b c (f t)")
            x = self.channel_downsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = self.channel_downsampler_t(xt)

    for idx, decode in enumerate(self.decoder):
        skip = saved.pop(-1)
        x, pre = decode(x, skip, lengths.pop(-1))
        # `pre` contains the output just before final transposed convolution,
        # which is used when the freq. and time branch separate.

        offset = self.depth - len(self.tdecoder)
        if idx >= offset:
            tdec = self.tdecoder[idx - offset]
            length_t = lengths_t.pop(-1)
            if tdec.empty:
                assert pre.shape[2] == 1, pre.shape
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t)
            else:
                skip = saved_t.pop(-1)
                xt, _ = tdec(xt, skip, length_t)

    # Let's make sure we used all stored skip connections.
    assert len(saved) == 0
    assert len(lengths_t) == 0
    assert len(saved_t) == 0

    S = len(self.sources)
    x = x.view(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]

    # (dropped: the mps/xpu cpu-roundtrip branch upstream has here -- this
    # export always runs on cpu, so it was always a no-op for us anyway.)

    zspec = self._mask(z, x)

    # --- upstream forward() continues with: x = self._ispec(zspec, length)
    # --- that's the ConvTranspose call we're routing around. Everything
    # --- below instead reproduces just the *xt* half of the final combine
    # --- (`xt = xt.view(...) * stdt + meant`), matching the `training_length`
    # --- branch upstream takes here (see the invariant check above).
    xt = xt.view(B, S, -1, training_length)
    xt = xt * stdt[:, None] + meant[:, None]

    return zspec, xt


def _patch_htdemucs_for_split_onnx(model: torch.nn.Module) -> torch.nn.Module:
    """Like `demucs_onnx.export.patch.patch_htdemucs_for_onnx`, but installs
    `_split_forward` as the model's `forward` instead of leaving the
    original `forward` in place with `_ispec` patched to `RealISTFT`.
    Reuses three of that function's four sub-patches unchanged (segment,
    pos-embedding, MHA); the STFT patch here installs only `RealSTFT`
    (forward transform -- plain Conv1d, not implicated in the DirectML
    bug) and skips `RealISTFT` entirely, since the graph never reaches it."""
    import torch.nn as nn
    import torch.nn.functional as F

    coerce_segment_to_float(model)
    disable_random_pos_shift(model)

    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            m.forward = types.MethodType(onnx_friendly_mha_forward, m)

    n_fft = 4096
    hop_length = n_fft // 4
    real_stft = RealSTFT(n_fft, hop_length)
    model.real_stft = real_stft

    def _spec_real(self_: Any, x: torch.Tensor) -> torch.Tensor:
        import math
        hl = self_.hop_length
        nfft = self_.nfft
        if hl != nfft // 4:
            raise AssertionError(f"unexpected hop {hl} for nfft {nfft}")
        le = math.ceil(x.shape[-1] / hl)
        pad = hl // 2 * 3
        x = F.pad(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        z = self_.real_stft(x)[..., :-1, :]  # drop the Nyquist bin
        if z.shape[-1] != le + 4:
            raise AssertionError((z.shape, x.shape, le))
        return z[..., 2: 2 + le]

    def _magnitude_real(self_: Any, z: torch.Tensor) -> torch.Tensor:
        B, C, two, Fr, T = z.shape
        if two != 2:
            raise AssertionError(f"expected 2 real channels, got {two}")
        return z.reshape(B, C * two, Fr, T)

    def _mask_real(self_: Any, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, S, C, Fr, T = m.shape
        return m.view(B, S, C // 2, 2, Fr, T)

    model._spec = types.MethodType(_spec_real, model)
    model._magnitude = types.MethodType(_magnitude_real, model)
    model._mask = types.MethodType(_mask_real, model)
    model.forward = types.MethodType(_split_forward, model)

    model.eval()
    model.to("cpu")
    return model


def export_split_to_onnx(checkpoint: str | Path, output: str | Path, *,
                          stem: str | None = None,
                          stems: list[str] | None = None,
                          opset: int = 17,
                          verbose: bool = True) -> dict[str, Path]:
    """Export a demucs/htdemucs checkpoint to a *split* ONNX graph (outputs
    `zspec`, `xt` instead of `stems`). Signature mirrors
    `demucs_onnx.export.export_to_onnx` for the common case, minus the
    `parity_check`/`sample_rate`/`segment_seconds` knobs that split export
    doesn't support (fixed-length only -- see module docstring) and minus
    ONNX-side parity checking (the two-output shape means the usual
    single-tensor `_verify_onnx_parity` doesn't apply as-is; use
    `scripts/validate_split_export.py` instead, which knows how to combine
    `zspec`+`xt` via `ispec_numpy` before comparing).

    Returns `{stem_name: output_path}`, same convention as upstream.
    """
    from demucs.pretrained import get_model

    if isinstance(checkpoint, Path) or (isinstance(checkpoint, str) and "/" in checkpoint):
        raise NotImplementedError(
            "export_split_to_onnx only supports named pretrained checkpoints "
            "for now (e.g. 'htdemucs_6s'); local .th paths aren't wired up "
            "-- add if/when needed.",
        )

    if verbose:
        print(f"Loading pretrained model: {checkpoint}")
    obj = get_model(str(checkpoint))
    is_bag = hasattr(obj, "models")
    sub_models = list(obj.models) if is_bag else None
    is_specialist_bag = is_bag and len(sub_models) > 1

    if not is_bag or not is_specialist_bag:
        if stem is not None or stems is not None:
            raise ValueError(
                f"checkpoint {checkpoint!r} is a single model; cannot pass "
                "`stem`/`stems`. Export the whole model and pick the row "
                "at inference time.",
            )
        targets: list[tuple[str, int]] = [(str(checkpoint), 0)]
    else:
        if stem is not None and stems is not None:
            raise ValueError("pass either `stem` OR `stems`, not both.")
        wanted = [stem] if stem is not None else (list(stems) if stems is not None else list(STEM_TO_INDEX))
        for s in wanted:
            if s not in STEM_TO_INDEX:
                raise ValueError(f"unknown stem {s!r}; expected one of {list(STEM_TO_INDEX)}")
        targets = [(s, STEM_TO_INDEX[s]) for s in wanted]

    out_root = Path(output)
    if is_specialist_bag and len(targets) > 1:
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root.parent.mkdir(parents=True, exist_ok=True)

    out_paths: dict[str, Path] = {}
    for stem_name, idx in targets:
        if is_specialist_bag and len(targets) > 1:
            file_path = out_root / f"htdemucs_ft_{stem_name}_split.onnx"
        elif is_specialist_bag:
            file_path = out_root if out_root.suffix.lower() == ".onnx" else out_root / f"htdemucs_ft_{stem_name}_split.onnx"
        else:
            file_path = out_root if out_root.suffix.lower() == ".onnx" else out_root / f"{stem_name}_split.onnx"

        if verbose:
            print(f"\n=== Exporting {stem_name} (split, index {idx}) -> {file_path} ===")

        # demucs.pretrained.get_model wraps even a single-file model
        # (htdemucs, htdemucs_6s) in a BagOfModels with one sub-model --
        # `obj` itself is that wrapper (its own nn.Module, no `_spec`/
        # `encoder`/etc of its own) and must never be patched/exported
        # directly. Only a *true* non-bag object (not produced by
        # get_model in practice, but handled for robustness) uses `obj`.
        if is_specialist_bag:
            original = sub_models[idx].eval().to("cpu")
        elif is_bag:
            original = sub_models[0].eval().to("cpu")
        else:
            original = obj.eval().to("cpu")
        patched = _patch_htdemucs_for_split_onnx(copy.deepcopy(original))

        file_path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.randn(1, 2, N_SAMPLES, dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                patched, dummy, str(file_path),
                opset_version=opset,
                input_names=["mix"], output_names=["zspec", "xt"],
                do_constant_folding=True, export_params=True,
                dynamo=False,
            )
        size_mb = file_path.stat().st_size / 1e6
        if verbose:
            print(f"  exported {size_mb:.1f} MB")

        import onnx
        onnx.checker.check_model(onnx.load(str(file_path)))
        if verbose:
            print("  onnx.checker: PASS")

        out_paths[stem_name] = file_path

    return out_paths

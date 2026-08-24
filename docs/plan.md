# AnyStemDeck build plan

This is the working plan for turning StemDeck into AnyStemDeck: same app, plus a DirectML-accelerated inference path for AMD/Intel GPUs on Windows. It's ordered so that each phase leaves the app in a working, testable state — nothing here waits until the end to prove itself.

## Baseline: what StemDeck actually does today

Worth stating precisely, since a couple of things in the original ask turned out to be slightly off once I read the code. StemDeck's default separation model is **`htdemucs_6s`** (6 stems: vocals, drums, bass, guitar, piano, other), not `htdemucs_ft` — `htdemucs_ft` is Trama's model and is available in StemDeck too, but only if someone sets `STEMDECK_DEMUCS_MODEL=htdemucs_ft`. Good news: `demucs-onnx` ships an ONNX export of all three flavors StemDeck can use — `htdemucs_ft` (best SDR, 4 stems, ships as 4 separate specialist models), `htdemucs` (single-file 4-stem, ~30% faster), and `htdemucs_6s` (single-file 6-stem, the only ONNX export of the 6-stem model anywhere) — so whichever model StemDeck is configured for, there's a matching ONNX model to swap in.

Separation itself runs in `app/pipeline/demucs_worker.py`: a persistent subprocess, spawned once per compute device and kept warm across jobs, that loads a PyTorch Demucs model and calls `demucs.apply.apply_model()` per job, streaming `NN%` progress lines over stderr back to the parent (`app/pipeline/separate.py`), which parses them into job progress. Device choice is `auto | cuda | mps | cpu`, resolved in `app/core/config.py` (`available_torch_devices()` probes `torch.cuda.is_available()` / `torch.backends.mps.is_available()`) and persisted per-user in `app/core/settings.py`. The Settings UI (`static/js/catalog.js`, with matching strings duplicated across seven locale blocks in `static/js/i18n.js`) shows a device dropdown populated from that probe. None of this currently has any AMD-aware path — a 7800 XT resolves straight to `cpu`.

## Phase 0 — Fork setup

Pull in `stemdeckapp/stemdeck` as the actual starting point (added as a git remote so upstream fixes can be merged in later, rather than a one-time copy — see open question below) and confirm the Apache-2.0 `LICENSE` + `NOTICE` are in place, which they already are in this repo. No functional changes yet. This phase is really just "make sure the codebase we're building on is the codebase we think we're building on," since the write-up above came from reading the real source, not from the StemDeck marketing page.

## Phase 1 — ONNX inference worker

The core of the fork. Add `app/pipeline/demucs_onnx_worker.py`, a sibling to the existing `demucs_worker.py` that speaks the exact same stdin/stderr job protocol (`{"source", "job_dir", "shifts"}` in, `NN%` progress and `@@DONE@@`/`@@ERROR@@` out) but calls `demucs_onnx.separate(source, out_dir, model=DEMUCS_MODEL, providers=..., stems=...)` instead of PyTorch. `demucs-onnx`'s own `SessionPool` gives the same "load once, reuse across jobs" behavior the persistent worker already exists to provide, so the two designs line up naturally.

One real gap to solve here: `demucs_onnx.separate()` reports progress as a `tqdm` bar over chunk count, not the `NN%` stderr lines `separate.py` currently parses. Two ways to close it — call `demucs-onnx`'s lower-level (currently private) chunking functions directly from the worker and compute percentage ourselves, or send a small patch upstream to `demucs-onnx` adding an optional progress callback to `separate()`. The second is cleaner and worth doing regardless, but it means a dependency on an upstream PR landing; decide which to take before writing this file (see open questions).

## Phase 2 — device detection & routing

Extend `app/core/config.py` with an `available_onnx_providers()` probe (checks `onnxruntime.get_available_providers()` for `DmlExecutionProvider`, mirroring how `available_torch_devices()` checks CUDA/MPS today), and extend `app/core/settings.py`'s `_DEVICE_CHOICES` tuple from `("auto", "cuda", "mps", "cpu")` to include `"dml"`. `get_demucs_device()`'s auto-resolution order becomes cuda/mps first (if present — a NVIDIA or Apple user shouldn't get downgraded to DirectML), then dml, then cpu. `app/pipeline/separate.py` picks the ONNX worker instead of the PyTorch one whenever the resolved device is `dml`.

## Phase 3 — Settings UI

Add a `DirectML (AMD/Intel/NVIDIA)` option to the device `<select>` in `static/js/catalog.js` (currently hardcodes `cuda`/`mps`/`cpu` options), guarded the same way the existing options are — greyed out when `available_onnx_providers()` didn't find it. Add the matching `settings.device.dml` translation key to each of the locale blocks in `static/js/i18n.js` (mechanical, one line × 7 locales).

## Phase 4 — packaging & distribution

`pyproject.toml` gets `demucs-onnx` plus a platform-conditional `onnxruntime-directml` (Windows) — following the exact pattern already used for the CUDA/CPU torch split. Model weights: match StemDeck's existing behavior of downloading checkpoints on first use rather than bundling them in the installer, pointed at the `StemSplitio` Hugging Face org; default to the `fp16weights` variant (same speed and RAM as fp32, ~1.9x smaller download, ~6×10⁻⁵ extra error — StemSplit's own numbers). `docs/models.md` gets a new entry recording this license basis, matching the format StemDeck already uses for its other bundled models.

## Phase 5 — testing & validation

Two things need checking that the existing StemDeck test suite won't catch on its own: that separation output from the DirectML path is audibly and numerically consistent with the CPU/CUDA path on the same input (StemSplit's own parity numbers — max abs diff ~1.7×10⁻⁴ against PyTorch — are a good starting bound, but worth reproducing locally on a real track), and that it's actually faster than CPU on the 7800 XT specifically, since "DirectML works" and "DirectML is fast on this particular card" aren't the same claim. Everything else — existing CUDA/MPS/CPU paths, the job queue, the player — should be exercised by StemDeck's existing pytest suite unmodified, which is worth running early and often rather than only at the end.

## Phase 6 — docs & release

Update the README's install instructions once there's something installable, note clearly that DirectML acceleration is a Windows-only benefit (Linux/macOS AMD users still need ROCm or CPU — this fork doesn't change that), and tag a first release. If Phase 1's progress-callback patch went upstream to `demucs-onnx` in the meantime, that's also the point to open the PR back to `stemdeckapp/stemdeck` proposing the DirectML backend generally, per the README's stated intent.

---

## Decisions

Settled before work started, so later contributors don't have to guess why things are shaped this way:

1. **Model support.** Both `htdemucs_6s` and `htdemucs_ft` are shipped as selectable options (the underlying `demucs_onnx.separate()` call takes either as a plain parameter, so this isn't extra implementation work — just extra testing surface). Default stays `htdemucs_6s`, matching StemDeck's own out-of-the-box behavior: smallest download, fastest per song, and the only one with guitar/piano stems. `htdemucs_ft` is there for anyone who wants Trama-level vocals/drums/bass/other quality and doesn't need guitar/piano.
2. **Platform scope: superset.** PyTorch/CUDA/MPS/CPU stay exactly as they are today for NVIDIA, Apple, and CPU-only users — nothing about their path changes. DirectML is added as a new option alongside, not a replacement. Smallest possible change, ships fastest, zero regression risk to what StemDeck already does well.
3. **Upstream tracking: yes.** `stemdeckapp/stemdeck` is added as a git remote (`upstream`) rather than a one-time copy, so future StemDeck fixes and features can be periodically merged in rather than manually re-ported.
4. **Progress reporting: real percentage, no upstream dependency.** Rather than accept a coarser progress bar or wait on an external PR to `demucs-onnx`, the DirectML worker calls `demucs-onnx`'s internal per-chunk functions directly (currently private, but stable enough to depend on within our own fork) and computes the same smooth percentage StemDeck already shows elsewhere. No blocking on someone else's release schedule.

## Next action

Phase 0 needs one thing only I can't do from here: `git fetch` requires network access, which this session's connection to your machine doesn't have (it can read/write files, but not reach GitHub on your behalf). Run this once, from the `AnyStemDeck` folder, to pull StemDeck in as the upstream base:

```
git remote add upstream https://github.com/stemdeckapp/stemdeck.git
git fetch upstream
git merge --allow-unrelated-histories upstream/main
```

That last step will likely flag a conflict on `README.md` and `LICENSE`/`NOTICE` (both repos have their own versions) — keep this repo's copies, since they're already written for AnyStemDeck; everything else should merge clean since this repo currently has nothing else in it. Once that's done, tell me and I'll move on to Phase 1 — the actual `demucs_onnx_worker.py` and the device-routing changes — on top of the merged tree.

# AnyStemDeck build plan

This is the working plan for turning StemDeck into AnyStemDeck: same app, plus a DirectML-accelerated inference path for AMD/Intel GPUs on Windows. It's ordered so that each phase leaves the app in a working, testable state — nothing here waits until the end to prove itself.

## Baseline: what StemDeck actually does today

Worth stating precisely, since a couple of things in the original ask turned out to be slightly off once I read the code. StemDeck's default separation model is **`htdemucs_6s`** (6 stems: vocals, drums, bass, guitar, piano, other), not `htdemucs_ft` — `htdemucs_ft` is Trama's model and is available in StemDeck too, but only if someone sets `STEMDECK_DEMUCS_MODEL=htdemucs_ft`. Good news: `demucs-onnx` ships an ONNX export of all three flavors StemDeck can use — `htdemucs_ft` (best SDR, 4 stems, ships as 4 separate specialist models), `htdemucs` (single-file 4-stem, ~30% faster), and `htdemucs_6s` (single-file 6-stem, the only ONNX export of the 6-stem model anywhere) — so whichever model StemDeck is configured for, there's a matching ONNX model to swap in.

Separation itself runs in `app/pipeline/demucs_worker.py`: a persistent subprocess, spawned once per compute device and kept warm across jobs, that loads a PyTorch Demucs model and calls `demucs.apply.apply_model()` per job, streaming `NN%` progress lines over stderr back to the parent (`app/pipeline/separate.py`), which parses them into job progress. Device choice is `auto | cuda | mps | cpu`, resolved in `app/core/config.py` (`available_torch_devices()` probes `torch.cuda.is_available()` / `torch.backends.mps.is_available()`) and persisted per-user in `app/core/settings.py`. The Settings UI (`static/js/catalog.js`, with matching strings duplicated across seven locale blocks in `static/js/i18n.js`) shows a device dropdown populated from that probe. None of this currently has any AMD-aware path — a 7800 XT resolves straight to `cpu`.

## Phase 0 — Fork setup ✅ done

`stemdeckapp/stemdeck` merged in as `upstream`, Apache-2.0 `LICENSE`/`README.md`/`NOTICE` kept as AnyStemDeck's own on conflict. Confirmed on disk: `app/`, `static/`, `desktop/`, `tests/`, `pyproject.toml` all present.

## Phase 1 — ONNX inference worker ✅ done (dml provider untested on real hardware)

`app/pipeline/demucs_onnx_worker.py` is written: same persistent-process design and stdin/stderr job protocol as `demucs_worker.py`, calling `demucs_onnx.separate()` per job instead of PyTorch. Three things came up while actually writing it that changed the shape of the original plan:

- **No `stems=` filtering.** The original plan had the worker skip stems the user didn't select. Wrong — `collect.py`'s "Original" complement track needs every stem the model produces regardless of selection, same as the PyTorch worker. Fixed: the ONNX worker always computes and writes every stem.
- **Progress is real for `htdemucs_ft`, indeterminate for `htdemucs_6s`.** Rather than vendor `demucs-onnx`'s private chunking functions (rejected — see Decisions), the worker calls `separate()` once per specialist for the 4-stem bag (25% steps, public API only) and once for single-file models, which has no natural checkpoint inside it. Since `htdemucs_6s` is the default, most jobs will show an indeterminate "Separating stems..." rather than a percentage in v1 — accepted tradeoff, decided with Ed.
- **Output writing bypasses `demucs_onnx.write_wav()`.** It quantizes straight to 16-bit PCM with no headroom check, so a stem peaking above full scale would hard-clip — audibly different output from the PyTorch path's `save_audio(clip="rescale")` for the identical input. Fixed with `_write_stem()`, which reproduces PyTorch's exact rescale formula (`wav / max(1.01 * wav.abs().max(), 1)`). Verified in isolation (no network in this sandbox to pull real model weights): a signal peaking at 1.3 writes back at ≤1.0 with no clipping, a quiet signal is left untouched, and the target sample rate is honored — see `/tmp/test_write_stem.py` in the build session if you want to rerun it.
- **Sample rate differs from the PyTorch path, on purpose for now.** `demucs_onnx.separate()` resamples output back to the *source file's native rate*; the PyTorch worker always writes at the model's fixed 44.1 kHz regardless of input rate. So a 48 kHz source gets 48 kHz stems via DirectML but 44.1 kHz stems via CUDA/CPU for the identical file. Not fixed — needs a decision (below) plus Phase 5 testing against the player/mixer/beat-grid code, which may or may not already handle mixed stem sample rates gracefully.

**Now tested on real hardware (RX 7800 XT) — found a blocking issue, not fixed.** `pip install -e .` and `onnxruntime-directml` both installed cleanly, `DmlExecutionProvider` is correctly detected, model downloads and session creation succeed. But every real inference call fails identically:

```
Non-zero status code returned while running ConvTranspose node. Name:'/real_istft/ConvTranspose_1'
... 8007000E Not enough memory resources are available to complete this operation.
```

GPU memory usage stayed under 2 GB out of 16 GB when this happened — not a real memory shortage. Debugging steps taken, in order:

1. **`demucs-onnx`'s session creation doesn't disable ORT's memory-pattern optimization**, which ONNX Runtime's own docs say DirectML requires disabled. Real bug, genuinely worth fixing regardless — patched around it in `_patch_demucs_onnx_for_directml()` in `demucs_onnx_worker.py`. Did not fix the crash on its own.
2. Made graph optimization level configurable (`ANYSTEMDECK_ONNX_GRAPH_OPT`) and tried `disable` — same error, exact same node, even with ORT's fusion pass fully disabled. Ruled out graph fusion as the cause.
3. Tried three structurally different model graphs (`htdemucs`, `htdemucs_6s`, `htdemucs_ft_vocals`) — **all three fail at the identical node**, `/real_istft/ConvTranspose_1`. Every htdemucs variant reconstructs its waveform through the same inverse-STFT step, which `demucs-onnx` implements as a `ConvTranspose` (a standard way to make ISTFT ONNX-exportable). This ruled out anything model-specific.
4. Web research found an independent, unrelated project (Kokoro TTS, an ONNX text-to-speech model — nothing to do with audio separation) hitting the **exact same symptom**: a `ConvTranspose` node crashing under DirectML with a "not enough memory"/"parameter is incorrect" error, confirmed to work fine on CPU, reported in [two places](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/discussions/5) with [no fix found](https://github.com/hexgrad/kokoro/issues/79) in either.

**Conclusion**: this is a real, currently-unresolved limitation in DirectML's own `ConvTranspose` implementation for large-kernel transposed convolutions — the pattern audio-synthesis models use for ISTFT — not a bug in our code, not a bug in `demucs-onnx`, and not something fixable through session options. It affects at least two unrelated ONNX models on DirectML. Microsoft's own DirectML docs describe the execution provider as being "in sustained engineering" and point toward a newer API (WinML) for new work, which is consistent with this not being a fast-moving target. See "Next action" below for the real choices this leaves.

## Phase 1.5 — DirectML ConvTranspose workaround: split ISTFT out of the graph — no more crash, but a second, distinct DirectML correctness bug found

Decided with Ed: rather than wait on Microsoft or fall back to CPU, attempt to route around the specific broken op. `RealISTFT` (the module `demucs-onnx`'s export code uses to implement inverse-STFT) is only ever called once per forward pass, right at the very end of `HTDemucs.forward`, immediately before the result is combined with the time-branch waveform and returned. Everything before that point — the entire transformer/convolutional network, the overwhelming majority of the actual compute — has no `ConvTranspose` problem at all; only that one large-kernel (`n_fft=4096`) op does. So instead of patching around the crash, this exports a *different* ONNX graph that stops right before that op.

New code, all in `app/pipeline/onnx_export/` (AnyStemDeck's own, not vendored from anywhere):

- **`export_split.py`** — `export_split_to_onnx()`, a from-scratch export function built the same way `demucs_onnx.export.export_to_onnx()` is (loads a pretrained checkpoint, applies ONNX-export patches, traces with `torch.onnx.export`), but reuses only 3 of `demucs-onnx`'s 4 patches (segment-as-float, no-random pos-embedding shift, primitive-only MHA) and replaces the 4th (STFT/ISTFT) with a custom one: `RealSTFT` (forward transform, plain `Conv1d`, no issue) stays, but `RealISTFT` is never installed at all. Instead, `_split_forward` — a line-for-line copy of `demucs.htdemucs.HTDemucs.forward` (demucs==4.0.1) truncated immediately before the `_ispec` call — becomes the model's `forward`, and the graph's two outputs are `zspec` (the masked spectrogram, pre-ISTFT) and `xt` (the time-branch waveform) instead of the finished `stems` tensor. Relies on one invariant that's always true for a fixed-length export/trace (documented in the module docstring): the input length always exactly equals the model's training segment length, so the `length_pre_pad`/`use_train_segment` branching upstream's `forward()` has can be safely hardcoded away rather than reproduced.
- **`split_istft.py`** — `ispec_numpy()`, a pure-numpy transcription of `RealISTFT.forward` plus the padding/cropping wrapper (`_ispec_real`) that upstream's export patch normally wraps it in. Same DFT-synthesis-plus-overlap-add math, computed with `einsum` + an explicit OLA loop instead of an actual (transposed) convolution — so there's no `ConvTranspose` node anywhere for DirectML to choke on. Runs on CPU, after ONNX Runtime returns `zspec`/`xt`, and adds them together to reproduce the exact final combine `forward()` does internally.

**Validation is two separate steps, deliberately, so a numerics bug and a DirectML bug can never be mistaken for each other**:

1. `python scripts/validate_split_export.py [--checkpoint htdemucs_6s]` — exports the split graph, runs it on plain CPU `onnxruntime` (no DirectML involved at all), reconstructs the waveform with `ispec_numpy`, and compares against the *original*, unmodified PyTorch model on the same input. This is the check that our export/ISTFT code is correct — if it fails, the bug is ours, full stop, and there's no reason to touch the GPU yet.
2. Only once that passes: `python scripts/test_dml_smoke.py --split --model htdemucs_6s` — runs the same split graph through `DmlExecutionProvider` on the real GPU. This is the check that DirectML actually accepts the split graph without crashing — the thing we don't yet know.

**Both validation steps ran on Ed's machine (RX 7800 XT), and the crash is gone — but real-song testing found a second problem.** Step 1 (`validate_split_export.py`, CPU-only): max abs diff 0.000245 vs the original PyTorch model, well inside tolerance — the export + numpy ISTFT math is correct. Step 2 (`test_dml_smoke.py --split`, real DirectML, synthetic 8s tone): no crash, 6 stems in 9.2s.

Then real-song testing (`test_dml_smoke.py --split` on a full track) surfaced something new: output peaks in the thousands (should be ~0.1-1.0 for real separated audio). Lowering `ANYSTEMDECK_ONNX_GRAPH_OPT` to disable fusion brought that down to the tens — better, but still wrong. The decisive test: running the *exact same* split-graph + chunking code through plain CPU execution instead of DirectML (`test_dml_smoke.py --split --provider cpu`, new flag added for this) produced correct peaks (0.12-0.71) on the same file. Same code, same graph, same input — CPU correct, DirectML wrong. That rules out a bug in this project's export/chunking code (already double-checked by the CPU validation, now confirmed a second way) and points to a **second, distinct DirectML correctness bug** — not a crash this time, a silently wrong numerical result somewhere in how DirectML executes this graph, separate from (if possibly related to) the `ConvTranspose` crash bug Phase 1 already found.

**`scripts/diff_dml_nodes.py`** (new) — a differential diagnostic that finds exactly which op is responsible instead of guessing. v1 (add every intermediate tensor as a graph output in one shot) choked DirectML's own graph partitioning on a few node types and silently fell back to CPU-only, invalidating the comparison. v2 binary-searches the graph's node list, adding one extra output per probe (~12 probes for a ~2200-node graph) — much more robust, and ran cleanly on Ed's machine.

**Result: `InstanceNormalization`.** The probe sequence shows near-zero error (~1e-5 to 1.6e-4, ordinary floating-point noise) through an ordinary `Conv` node, a small but real jump at an `InstanceNormalization` node (0.0136), then rapid amplification two ops later through the `Mul`/`Div` that immediately follow it (0.07 → 18+, and everything downstream of that block stays in the same 2-20 range). That's the signature of a normalization op computing a slightly-wrong variance/statistic that then gets amplified multiplicatively by the gating math right after it — consistent with a real DirectML kernel bug in `InstanceNormalization` specifically, not a generic "everything is a bit off" precision issue.

**This is meaningfully worse news than the `ConvTranspose` bug.** A full op-type count of the graph (`Counter(n.op_type for n in m.graph.node)`) shows **74 `InstanceNormalization` nodes** — this op is part of the `dconv` residual block used repeatedly throughout the encoder and decoder, not a single call at one clean boundary the way `RealISTFT` was. It can't be sliced out of the graph the same way (there's no single "before/after" cut point — it's interleaved with convolutions throughout the whole network). A fix would mean either decomposing `InstanceNormalization` into primitive ops (`ReduceMean`/`Sub`/`Pow`/`Sqrt`/`Div`, the same op types that already tested clean) inside the export patch — plausible, since it's one `nn.Module` class patched once, not 74 separate interventions, similar in spirit to how `onnx_friendly_mha_forward` already replaces `nn.MultiheadAttention` — or forcing those 74 nodes onto CPU execution individually, which would mean 74 GPU↔CPU round trips per chunk and could erase most of the DirectML speed advantage. Neither is verified yet, and there's no guarantee `InstanceNormalization` is the *last* DirectML bug in this graph (`LayerNormalization`, 26 occurrences in the transformer, hasn't been individually tested — the search stopped once things were already corrupted downstream of the `InstanceNormalization` finding).

This is now the second independent DirectML correctness bug found in about two hours of real-hardware testing (crash + silent-wrong-answer), and the second one is structurally harder to route around than the first. Decision point for Ed — see "Next action."

Not wired into the real `demucs_onnx_worker.py` production path — correctly so, given this is still an open correctness bug, not a solved one.

Known limitations of what's built so far, worth being upfront about: `export_split_to_onnx()` only supports named pretrained checkpoints, not arbitrary local `.th` files (not needed yet); the chunked inference loop in `test_dml_smoke.py --split` is a straight port of `demucs_onnx.inference`'s overlap-add scheme with a Python `for` loop over ~336 STFT frames per chunk in `ispec_numpy` — fine for a smoke test, but if this proves out on real hardware, worth profiling before it becomes the production worker path (the per-frame OLA loop is the one part of `ispec_numpy` that isn't already vectorized).

## Phase 1.6 — pivot: DirectML parked, investigating WSL2 + ROCm instead

Presented with the `InstanceNormalization` finding, Ed asked whether the whole DirectML/ONNX approach was worth continuing given two independent platform bugs in one afternoon, and whether there was a non-DirectML way to use the 7800 XT at all. Researched the live options (all fresh web research, not assumed from training knowledge — this space moves fast):

- **ROCm on native Windows: still not viable.** Confirmed via AMD's own ROCm 7.2 Windows PyTorch release notes — the supported-card list is RX 9070/9070XT/9060XT, RX 7900 XTX, PRO W7900, and specific Ryzen AI chips. **RX 7800 XT is not on it.** Matches the original Phase 0 research; hasn't changed.
- **ZLUDA (CUDA-on-AMD translation layer): ruled out.** Its own maintainer describes it as back to hobby-project status after losing funding, explicitly not recommended for production, with rough edges even in its stated use cases.
- **AMD MIGraphX via the new Windows ML framework: real, but unproven and adds a runtime dependency.** Microsoft's 2026 Windows ML layer can download AMD's own MIGraphX execution provider — genuinely separate from DirectML, not sharing its bugs. Docs say Python is a supported language and the ONNX Runtime API is unchanged (same `InferenceSession`), which would mean reusing our existing ONNX models with just a provider swap. But it needs "framework-dependent" deployment (the Windows App SDK runtime installed on the machine) — a real new dependency for a project that's otherwise a plain Python venv — and nothing found confirms it actually works end-to-end yet. Parked as a fallback option, not pursued first.
- **ROCm via WSL2: the strongest lead, now confirmed for this card.** AMD's own docs were frustratingly vague ("select 7000 Series" with no itemized list), but the actual technical compatibility source they point to — the `librocdxg` WSL GPU-passthrough compatibility repo — explicitly lists **RX 7800 XT as supported**. An independent hands-on writeup corroborates it works in practice, calling the 7800 XT "the best balance of cost and capability" for this setup, while flagging two honest caveats: WSL2's paravirtualized GPU interface adds real overhead versus native Linux, and there are scattered reports of driver timeouts on some workloads. Ed's driver (26.10.35.01) is well past the 26.2.2 baseline AMD's docs mention needing.

**Why this is a materially better path than the DirectML/ONNX route, if it works**: it needs zero changes to StemDeck's existing PyTorch worker (`demucs_worker.py`) or model handling — ROCm presents itself to PyTorch as `torch.cuda`, the exact same API StemDeck already uses for NVIDIA. No ONNX export, no `demucs-onnx`, none of Phase 1/1.5's work is needed. The cost is architectural rather than numerical: the Python backend needs to actually run inside the WSL2 Linux environment to reach the GPU that way, while the Tauri desktop UI stays on Windows — a real integration question (how the two talk to each other) but a well-understood one, unlike chasing unpredictable EP correctness bugs.

Ubuntu 24.04 is now installed in WSL2 on Ed's machine (`wsl --install -d Ubuntu-24.04` completed, confirmed via `wsl --list --verbose`). Driver confirmed: `26.10.35.01-260716a-202643C-AMD-Software-Adrenalin-Edition`, comfortably past the 26.2.2 baseline.

Two source documents obtained (as PDF exports of the live AMD docs pages, per the established pattern of not trusting `WebFetch` on these particular pages — see Phase 1.6 above and Errors/fixes): the `librocdxg` GitHub README (Quickstart + WSL Compatibility Matrix, confirming RX 7800 XT support) and AMD's "Install PyTorch for ROCm" guide (PIP install method, Ubuntu 24.04 tab). Together these fully specify Stages 2-4 of the install (librocdxg, verification, PyTorch) below. Stage 1 (base ROCm package) is still an open question — see "Next action."

**✅ Confirmed working on Ed's machine, end to end.** All 4 install stages ran successfully (one hiccup along the way — see Errors/fixes: the librocdxg `.deb` needed to actually be downloaded, and pinned to the version matching ROCm 7.2.1 — `1.2.0`, not the newer `1.2.2` Ed initially found on the releases page). The actual go/no-go check:
```
python3 -c "import torch; print(torch.cuda.is_available())"       -> True
python3 -c "import torch; print(torch.cuda.get_device_name(0))"   -> AMD Radeon RX 7800 XT
```
Both pass. A `UserWarning: Can't initialize amdsmi - Error code: 34` prints alongside — expected and non-blocking, not a sign of anything wrong: the librocdxg README's own "Known Issues" section already flags AMD-SMI as only partially supported under WSL2 ("a formal release plan is under development"), and it's a monitoring/telemetry library, not the compute path PyTorch actually uses (that's `torch.cuda`/HIP, which is working correctly per the check above).

**This is the real milestone for Phase 1.6**: StemDeck's existing, unmodified PyTorch worker now has a confirmed, working path to real GPU acceleration on the RX 7800 XT, with zero ONNX/DirectML code involved. Next: the integration architecture question (see "Next action").

## Phase 2 — device detection & routing ✅ done

`app/core/config.py` gained `available_onnx_providers()` (checks `onnxruntime.get_available_providers()` for `DmlExecutionProvider`) and `detect_compute_device()` (cuda > mps > dml > cpu). `app/core/settings.py`'s `_DEVICE_CHOICES` now includes `"dml"`, `get_demucs_device()` auto-resolution routes through `detect_compute_device()`, and `set_demucs_device()` validates `"dml"` against `available_onnx_providers()` the same way it already validates cuda/mps. `app/pipeline/separate.py` picks the ONNX worker over the PyTorch one whenever the resolved device is `dml`, and defensively caps shift-averaging to 1 on that path (see Decisions — "best" quality has no DirectML equivalent yet).

## Phase 3 — Settings UI — not started

Add a `DirectML (AMD/Intel/NVIDIA)` option to the device `<select>` in `static/js/catalog.js` (currently hardcodes `cuda`/`mps`/`cpu` options), guarded the same way the existing options are — greyed out when the backend doesn't report `dml` in `demucs_devices_available`. Add the matching `settings.device.dml` translation key to each of the seven locale blocks in `static/js/i18n.js` (mechanical, one line × 7). Also grey out "Best" quality specifically when `dml` is selected, per the backlog item in Decisions.

## Phase 4 — packaging & distribution — partially scoped, not implemented

`pyproject.toml` now declares `demucs-onnx` as a normal dependency (same platform gate as `audio-separator`/`onnxruntime`), but **not** `onnxruntime-directml` directly — a real packaging conflict surfaced while writing this: `onnxruntime` and `onnxruntime-directml` are different PyPI distributions that both install into the same `onnxruntime` import path, and `demucs-onnx` already hard-depends on plain `onnxruntime>=1.17`. Declaring both in `pyproject.toml` would leave pip to silently pick whichever installs last — the exact problem StemDeck's own comment already flags for `onnxruntime-gpu`, just now with a second package hitting it. The fix has to happen at packaging time, not dependency-resolution time: `scripts/windows/make-portable.ps1` needs a forced `pip install --force-reinstall --no-deps onnxruntime-directml` as its last step, exactly mirroring how that script already handles the CUDA/CPU torch split. Not yet written. A plain `pip install -e .` dev checkout in the meantime gets CPU-only `onnxruntime`, which is correct and safe: `available_onnx_providers()` won't find `DmlExecutionProvider`, so `dml` simply won't be offered as a device — no crash, no silent wrong behavior.

Also surfaced: `demucs-onnx` requires Python ≥3.11, while `pyproject.toml`'s floor is still `>=3.10`. Worth deciding whether to raise the floor outright or gate `demucs-onnx` by Python version — noted inline in `pyproject.toml`, not resolved.

Model weights: still planned as download-on-first-use from the `StemSplitio` Hugging Face org, defaulting to `fp16weights` (now wired up via `STEMDECK_DEMUCS_ONNX_PRECISION` in `config.py`). `docs/models.md` still needs its entry for this, matching the format StemDeck uses for its other bundled models — not yet written.

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
4. **Progress reporting, revised during Phase 1.** The original plan to call `demucs-onnx`'s private chunking functions directly turned out to undersell the real cost — those functions take an already-loaded audio array and an already-open session, not a file path, so using them means reimplementing most of what `separate()` already does around them. Actual approach, decided with Ed: for the `htdemucs_ft` bag, call `separate()` once per specialist (public API, real 25%-step checkpoints); for single-file models including the default `htdemucs_6s`, accept an indeterminate progress state in v1 rather than vendor private code.
5. **"Best" quality on `dml`: disabled for v1, tracked as backlog.** No shift-averaging equivalent exists in `demucs-onnx`. `separate.py` caps `shifts` to 1 before dispatch when the device is `dml`; the Settings UI still needs to grey out the option (Phase 3).
6. **WSL2/ROCm installer setup: semi-automated.** Decided with Ed. The installer detects whether WSL2 + ROCm + PyTorch are already present and working, and *offers* to run a guided setup script if not — it never provisions any of this silently in the background. Rejected: fully-automated silent setup (too much risk of quietly breaking as AMD's driver/ROCm versions drift, per how many small version-matching issues came up even in this one guided manual install) and pure documentation-only (undersells how much friction a first-time user would hit).
7a. **WSL2 Python environment: one env, `pip install -e .` into the same environment ROCm+PyTorch already went into.** Decided with Ed. The guided-setup script (#6) installs the rest of the backend's dependencies (fastapi, uvicorn, demucs, librosa, yt-dlp, etc.) into the identical WSL2 Python that Stage 4 already verified `torch.cuda.is_available()` against, via `pip install -e .` against the repo mounted at its `/mnt/c/...` path — not a separate venv. Matches how the manual smoke test below is provisioned, and keeps `wsl2_python()`'s default of a bare `python3` on PATH correct without needing venv-activation logic in `start_backend_wsl2`.
7b. **ffmpeg inside WSL2: `apt install ffmpeg` in the guided-setup script, not a new Rust downloader.** Real gap found while reviewing: `app/core/config.py`'s `FFMPEG_BIN` resolution is platform-correct on its own (`sys.platform` inside the WSL2 process is `"linux"`, so it looks for a plain `ffmpeg`, not `.exe`) — but nothing provisions that binary. StemDeck's existing `download_linux_ffmpeg()` in `main.rs` can't be reused as-is: it's `#[cfg(all(unix, not(target_os = "macos")))]`, compiled out of the Windows build entirely. Decided with Ed: simplest fix is one more line in the guided-setup script (`sudo apt install ffmpeg` inside WSL2) rather than a new Windows-side downloader with its own failure modes.
7c. **wsl2Distro: pinned explicitly by the guided-setup script, not left on WSL's default.** `start_backend_wsl2` already reads a `wsl2Distro` store key when present (no code change needed for this); the guided-setup script should write it to the exact distro it verified (e.g. `"Ubuntu-24.04"`) rather than relying on `wsl.exe`'s ambient default distro, so a later unrelated `wsl --set-default` for some other project can't silently break StemDeck's launch.
7d. **WSL2/ROCm device selection: no new `device` value at all — corrected from the original plan.** The original wording above (auto-detect a new "rocm_wsl" device, slot it into the `cuda > mps > ? > dml > cpu` priority order) assumed the WSL2 path needed its own entry in `settings.py`/`config.py`/`separate.py`/the Settings dropdown, the same way `dml` did for the DirectML path. It doesn't. Once a backend process is actually *running inside WSL2* with ROCm+PyTorch installed, it reports `demucs_device: cuda` exactly like an NVIDIA machine would — ROCm presents to that Python process as plain `torch.cuda`, and none of `available_torch_devices()`/`detect_compute_device()`/the device dropdown needs to know or care that the CUDA it's seeing is actually ROCm-via-WSL2 underneath. The only genuinely new decision is **which backend *process* to start at all** — native Windows vs. WSL2 — and that's a Rust-level, launch-time choice (`start_backend`, gated on a persisted `wsl2BackendEnabled` setting; see Decision #8 and the implementation notes below), not a peer value in the existing device-selection logic. This meaningfully shrinks what "Phase 2 rework" means: Phase 2 (device detection & routing) stays exactly as already built for the DirectML path, with zero additional changes for WSL2/ROCm.

## Open question surfaced while building Phase 1

**Sample rate consistency across backends.** `demucs_onnx.separate()` writes stems back at the source file's native rate; the PyTorch path always writes at a fixed 44.1 kHz regardless of input rate. Right now the DirectML worker preserves `demucs-onnx`'s native-rate behavior as the more honest default (no unnecessary downsampling), but that means the *same input file* can produce differently-rated stems depending purely on which device happened to run the job — worth deciding whether that's fine (if StemDeck's player/mixer/beat-grid already read sample rate from each file rather than assuming 44.1 kHz) or whether the DirectML worker should force 44.1 kHz to match the PyTorch path exactly. Needs either a code read of `collect.py`/`analyze.py`/the player, or just testing it on a non-44.1kHz source and seeing what breaks, if anything — good candidate for Phase 5.

## Next action

**Phase 1.6's install milestone is done — WSL2 + ROCm + PyTorch confirmed working on the RX 7800 XT.** (Full history: two independent DirectML correctness bugs killed the ONNX/DirectML approach; `librocdxg` research found ROCm-via-WSL2 as a materially better path since it needs zero changes to StemDeck's existing PyTorch worker; all 4 install stages — base ROCm, librocdxg, verification, PyTorch — are now done on Ed's machine, with `torch.cuda.is_available()` returning `True` and `torch.cuda.get_device_name(0)` returning `AMD Radeon RX 7800 XT`. See Phase 1.6 above for the full trail and the one hiccup along the way.)

**Now open: the integration architecture — scoped against the real `desktop/src-tauri/src/main.rs` this time**, not designed in the abstract. Ed connected his `stemdeck` upstream clone (separate from this `AnyStemDeck` working copy) so this could be read directly instead of guessed at. Findings:

`start_backend()` today does `Command::new(python).args(["-m", "uvicorn", "app.main:app", ...]).spawn()` — a direct Windows child process, full stop. There is currently **zero WSL awareness anywhere in `main.rs`** (confirmed by search). Four mechanisms in that function and its neighbors are built on the assumption that the spawned `Child` *is* uvicorn, running in the same OS/PID namespace as Tauri:

1. **Liveness/health**: `wait_for_health()` calls `child.try_wait()` directly on the Rust `Child` handle to detect a dead backend during startup, alongside an HTTP poll of `/api/health` that checks a `STEMDECK_INSTANCE_TOKEN` echoed back (this token mechanism already exists for a different reason — telling a *foreign* process that happens to be squatting on the port apart from the one just spawned — see `new_instance_token()`'s doc comment). If launched via `wsl.exe` as an intermediary, `child` would be the `wsl.exe` bridge process, not uvicorn — `try_wait()` would no longer mean what the code assumes it means. The token-echo half of the check, being purely HTTP-based, keeps working unchanged.
2. **The parent-death watchdog breaks outright, not just needs adapting.** `start_backend` passes `STEMDECK_PARENT_PID` (Tauri's own Windows PID) to the backend; `app/main.py`'s `_desktop_parent_watchdog` polls `app/core/process.py`'s `process_exists()` — `os.kill(pid, 0)` on POSIX, `OpenProcess` on Windows — to auto-exit if Tauri dies. **PIDs do not cross the WSL2/Windows boundary** — they're different kernels. A backend running inside WSL2 checking a Windows PID via `os.kill()` is checking a number that means nothing in its own PID namespace; this watchdog would silently stop protecting against orphaned backends the moment the backend moves into WSL2. This needs a real redesign (e.g., watch for the HTTP connection/parent socket dying, or a heartbeat), not a tweak. (The *other* watchdog layer — `demucs_worker.py` watching its own parent, the backend process — is unaffected, since both ends would still be inside WSL2 together.)
3. **Port handling assumes one network stack.** `reserve_port()` claims-then-releases a port on the Windows side immediately before spawn specifically to avoid a TOCTOU race with the child binding it (documented at length in the code — a real bug, #-tagged, from `127.0.0.1` vs `0.0.0.0` binding semantics differing on Windows). None of that reservation dance means anything if the real bind happens inside WSL2's own network namespace instead. Whether `http://127.0.0.1:{port}` (hardcoded as the return URL today) actually reaches a WSL2-side listener depends on WSL2's default NAT port-forwarding behavior — this is exactly the 5-minute manual spike proposed earlier, now with a concrete reason it matters: the existing code's return value is that URL, unconditionally.
4. **Shutdown (SIGTERM then SIGKILL, sent to the tracked `Child`) has the same intermediary problem as #1.** Killing `wsl.exe` does not reliably deliver a clean shutdown to the Linux process on the other side of that bridge — a real, commonly-reported WSL rough edge, not a hypothetical. Likely needs either an explicit shutdown call over HTTP, `wsl.exe --terminate`, or restructuring the launch so the tracked child *is* uvicorn (e.g. `wsl.exe -d Ubuntu-24.04 -- bash -lc 'exec python3 -m uvicorn ...'`, using `exec` so bash doesn't stay as a wrapper layer) — though even then, the Windows-side `Child` PID is still not the same as the actual Linux-side PID, so PID-based liveness (#1) needs its own fix regardless.

Also confirmed: `python_path()`'s resolution is entirely native-path-based (`python/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere) — a WSL2 install has no natural place in that function as written; it needs its own branch entirely, not a new candidate path appended to the list.

**What this changes about scope**: this is not a small patch to `start_backend`'s command line. It's a self-contained rewrite of the backend-lifecycle subsystem specifically — spawn, health/identity verification, the Tauri→backend watchdog direction, and shutdown — bounded to `main.rs`'s backend-management functions plus `app/main.py`'s watchdog. Nothing about the job queue, the API routes, the worker protocol, or the UI needs to change; once the HTTP boundary is actually up, `app.main:app`'s routes don't care where they're physically running. That boundedness is good news — it's a real, scoped subsystem, not a rewrite of the app — but it's a more substantial piece of work than "add a WSL launch option," and the PID-watchdog issue (#2) in particular is a correctness bug waiting to happen if skipped rather than deliberately redesigned.

**The launch-mechanism design is now fully settled** (Decisions #8-#11 below), following the port-forwarding spike (passed — `python3 -m http.server 8001` inside WSL2 answered `curl 127.0.0.1:8001` from Windows with zero extra config, confirming WSL2's default NAT already forwards listening ports) and two further rounds of decisions with Ed:

8. **Launch mechanism: Tauri-initiated, not Tauri-supervised (a synthesis of the original two options, not either pole).** On startup, Tauri runs its existing HTTP health check (`/api/health` + `STEMDECK_INSTANCE_TOKEN`) against `127.0.0.1:<port>`; if nothing answers, it fires a **one-shot, detached** `wsl.exe -d Ubuntu-24.04 -- bash -lc "nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port <port> ... &"` — critically, **no `Child` handle is retained** for it, so today's `try_wait()`-based liveness check and SIGTERM/SIGKILL-based shutdown (both broken by the WSL2 boundary — see the numbered findings above) never come into play for this path. Tauri then goes back to polling the same HTTP health check until the backend answers. Rejected: a separately-managed always-on WSL2 service (Ed picked "fully stop when the app closes," which removes the main appeal of that option — see #9) and `wsl.exe`-as-tracked-parent (would still require solving the PID-across-boundary problems below, for no benefit once the detached approach avoids needing to).
9. **Backend lifetime: stops fully when the app closes**, matching today's native-Windows behavior — no idle GPU/RAM use in the background, at the cost of a cold-start (WSL2 + uvicorn + model load) on each app launch. Rejected: keep-running-in-background (instant relaunch, but holds GPU memory indefinitely) and its idle-timeout variant (moot once "fully stop on close" was chosen).
10. **Graceful shutdown: reuses an existing, proven mechanism instead of inventing one.** `app/main.py`'s current parent-death watchdog already self-terminates by calling `signal.raise_signal(signal.SIGTERM)` **inside the backend process** when it detects Tauri is gone (triggering uvicorn's normal graceful shutdown, including demucs-worker teardown) — this was built for the PID-watchdog case, but the same in-process self-signal is exactly what a WSL2 shutdown needs too. Plan: add one small loopback-only HTTP endpoint that Tauri calls on close, which internally does the same `signal.raise_signal(SIGTERM)` call. No new shutdown logic, just a new trigger for logic that already exists and is already exercised (crash-detection path).
11. **Crash-orphan detection: heartbeat with timeout, not PID-checking.** The existing `STEMDECK_PARENT_PID` / `process_exists()` watchdog (`os.kill(pid, 0)` / `OpenProcess`) cannot work for a WSL2-hosted backend — **PIDs do not cross the WSL2/Windows kernel boundary at all**, so a WSL2 process checking a Windows PID is checking a number meaningless in its own namespace; this needed replacing, not adapting. Decided with Ed: Tauri pings the backend roughly every 10s while running; a new WSL2-path-specific watchdog self-terminates (same `signal.raise_signal(SIGTERM)` reuse as #10) if no heartbeat arrives within ~30s. Rejected: connection-drop detection (reacts faster to a real crash, but more exposed to a transient blip on the WSL2 virtual NIC reading as a false crash) and skipping this for v1 (leaves a real orphaning risk on the table for no reason once the mechanism is this small).

**Net effect on implementation scope**: bounded to `main.rs`'s `start_backend`/`wait_for_health`/`stop_backend` (needs a WSL2-path branch in each, not a rewrite of the whole file) plus `app/main.py`'s watchdog (new shutdown endpoint + heartbeat-based watchdog alongside, not replacing, the existing PID-based one used by the native CUDA/MPS/CPU paths) plus Phase 2's device detection (the `/proc/version` "am I already inside WSL2" check, and where this device slots into the `cuda > mps > ? > dml > cpu` priority order — not yet decided). Nothing about the job queue, API routes, worker protocol, or UI needs to change.

Not blocking anything else — Phase 1/1.5 (DirectML/ONNX) code stays parked in the repo either way, and Phases 3-6 for the DirectML path are independent of this decision.

### Caught up with upstream (13 commits, ~70 files) ✅ done

Before implementation started, AnyStemDeck's fork was merged forward from `upstream/main` (13 commits it was behind, touching roughly 70 files once the actual diff was seen — well beyond what the one-line commit summaries suggested). Notably included `f059ad1`, "The shell recognises its own backend by token, not by PID (#460)" — the `STEMDECK_INSTANCE_TOKEN`/`HealthIdentity` mechanism that Decision #8's launch design already assumed existed, so this merge wasn't optional groundwork, it was load-bearing for the WSL2 work itself.

Mechanics, for the record: the merge was blocked at first by long-uncommitted local Phase 1/1.5/2 work (`config.py`, `settings.py`, `separate.py`, `pyproject.toml`), which had to be committed first (`ff8cdcd`) rather than stashed, to avoid tangling a stash against a 13-commit merge. That commit swept up some scratch/test-output files that don't belong in the repo (`tmp_split.onnx` ~177 MB, `smoke_out/**/*.wav`, `smoke_test_tone.wav`) — **still pending**: `git rm -r --cached smoke_out tmp_split.onnx smoke_test_tone.wav` plus `.gitignore` entries for all three, so they can't sneak back in. Two real conflicts came up in the merge itself: `README.md` (kept AnyStemDeck's own short README entirely — upstream's is a much longer document for a different-purpose project and doesn't describe this fork) and `app/core/settings.py` (both sides' docstring additions were purely additive, kept both). Both resolved and committed (`git commit --no-edit` after fixing the conflict markers). Post-merge, `main.rs` is 220,474 bytes and contains the token mechanism — confirmed identical to a fresh upstream clone at the same commit, since AnyStemDeck hadn't yet diverged from upstream in `main.rs` before this session's implementation work below.

### Decisions #8-#11 implemented ✅ done, unverified on real hardware

Both halves written directly against AnyStemDeck's actual `app/main.py` and `desktop/src-tauri/src/main.rs` (not the reference `stemdeck` clone, which was read-only throughout — used only to double-check the two files hadn't diverged before editing). Compiled clean (`cargo check` + `cargo clippy`, zero warnings) and Python-syntax-checked (`py_compile`) in the build sandbox before being sent back — but the sandbox has no WSL2, no Windows, and no Rust GUI toolchain match for the real target, so this is a "compiles and reads correctly" confirmation, not a "runs correctly" one. **Ed still needs to `cargo build`/run it for real** and report back whatever actually happens on first launch.

**Python (`app/main.py`)**: `_desktop_heartbeat_watchdog()` — the heartbeat-based counterpart to the existing PID-based `_desktop_parent_watchdog`, self-terminating via the same proven `signal.raise_signal(SIGTERM)` trick if `/api/desktop/heartbeat` goes unpinged for 30s. `lifespan()` now branches on a new `STEMDECK_DESKTOP_HEARTBEAT` env var to pick which watchdog runs. Two new endpoints, both token-gated against `STEMDECK_INSTANCE_TOKEN` (defense in depth on top of the existing network gate, since `allow_network` being on would otherwise let anything on the LAN call them): `POST /api/desktop/heartbeat` (resets the timer) and `POST /api/desktop/shutdown` (the graceful-shutdown trigger for #10, same `signal.raise_signal(SIGTERM)` call).

**Rust (`main.rs`)**: `BackendHandles.child: Child` became `BackendHandles.process: BackendProcess`, an enum of `Native(Child)` (unchanged behavior) or `Wsl2` (no handle — Decision #8's "no Child retained"). `start_backend` branches on a new `wsl2_backend_enabled()` check (reads a persisted `wsl2BackendEnabled` store key — see the open question below) and, on the WSL2 path, fires one detached `wsl.exe -- bash -lc "... nohup python3 -m uvicorn ... & disown"` (`start_backend_wsl2`), waits for health without a `Child` to watch (`wait_for_health_headless`), and starts a self-terminating heartbeat thread (`spawn_wsl2_heartbeat`, matched by instance token so it can never keep a stale backend alive after a restart). `stop_backend`/`stop_backend_and_wait` branch the same way: native keeps its existing SIGTERM/SIGKILL logic untouched, WSL2 POSTs to `/api/desktop/shutdown` (`post_desktop_shutdown`) and polls health until it stops answering, deliberately *not* escalating to a force-kill if that times out (reaching across the WSL2/Windows kernel boundary to kill-by-guess risks the wrong process; the heartbeat watchdog is the backstop instead). Every Windows path uvicorn needs (data dir, jobs dir, settings mirror, log path) is translated to its WSL2-side `/mnt/<drive>/...` form via `wslpath -a` rather than hand-rolled string surgery.

**Two things Ed should know going in, not bugs so much as consequences of the design**: first, this routes the backend's file I/O (job data, model cache, every separated stem) through the DrvFs bridge (`/mnt/c/...`), which is slower than a native WSL2 filesystem path — a known cost of keeping user data on the Windows side rather than duplicating it inside the WSL2 disk. Second, `start_backend_wsl2` calls `wslpath` once per path (4-5 separate `wsl.exe` subprocess launches, plus the final launch itself) rather than batching them into one call — correct, but adds some avoidable startup latency on top of an already cold-start-heavy design (WSL2 + uvicorn + model load per launch, since Decision #9 stops the backend fully on app close); worth revisiting once the mechanism is proven to work at all.

**Still open, deliberately not decided in code**: exactly how `wsl2BackendEnabled` (and the two other new settings, `wsl2Distro` and `wsl2Python`) get set in the long run — that's Decision #6's guided-setup flow, real, separate, unbuilt work. For the *manual* smoke test below they're set by hand instead (see Decision #8, sequencing).

8. **Sequencing: manual smoke test before the guided-setup flow.** Decided with Ed. Rather than design and build Decision #6's automated detect/install UI on top of a launch mechanism that has never run against real WSL2/Windows, Ed hand-provisions a WSL2 Python environment and hand-sets the new store keys first, so the `main.rs`/`app/main.py` mechanism itself gets debugged against reality before more gets built on top of it.

### Next action: manual smoke test

Nothing in the WSL2 launch path has run against real WSL2/Windows yet — the sandbox this was written in has neither. Runbook, combining Decisions #7a-#7c above with what Stage 2-4 already installed:

1. **✅ done — Provision the WSL2 Python environment** (7a/7b) — inside the same `Ubuntu-24.04` instance where `torch.cuda.is_available()` already returns `True`:
   ```
   sudo apt install -y ffmpeg
   cd "/mnt/d/OneDrive/Git/Stem Separation/AnyStemDeck"   # the repo lives on D:, not C:
   SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 pip install --user --break-system-packages -e .
   ```
   Two real, worth-recording snags, neither a bug in this project's code:
   - `pip install -e .` alone failed with `error: externally-managed-environment` — Ubuntu 24.04's PEP 668 protection (the same thing Stage 4's PyTorch install must have hit). Fixed with `--break-system-packages`; kept `--user` alongside it so this install lands in the same place (`~/.local/lib/python3.12/site-packages`) as the ROCm/PyTorch install, per Decision #7a.
   - Even with that fixed, the build failed a second way: `hatch-vcs` (which derives the package version from git tags) ran `git describe --dirty --tags --long ...` against the repo and it **timed out after 40 seconds**. Root cause: the repo lives on `D:`, reached from WSL2 through the DrvFs bridge (`/mnt/d/...`) — DrvFs's per-file overhead makes git's many small `.git/` reads dramatically slower than on a native filesystem, badly enough to blow past `setuptools_scm`'s internal timeout. Fixed by skipping git version-probing entirely: `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0` (a standard `setuptools_scm` escape hatch) before the `pip install` command. Harmless for a dev/editable install — it only changes what version string gets reported, not runtime behavior. **Flagging this as a general risk, not just a one-off**: any git operation against this repo from inside WSL2 will hit the same DrvFs slowness, not just this one build step — worth remembering if something else git-related feels oddly slow later. The real fix, if it ever becomes worse than an occasional annoyance, would be a native clone inside WSL2's own filesystem (e.g. `~/anystemdeck`) rather than reaching across the Windows drive — not attempted here, bigger than this smoke test needed.
2. **Set the three new store keys by hand** (7c) — close StemDeck first, then edit `user-data.json` (it's plain JSON from `tauri-plugin-store`; for a dev `cargo run` with no `portable.txt` marker and no relocated stems folder, this is `%USERPROFILE%\Documents\StemDeck\jobs\user-data.json` — Settings' stems-location shows which folder is live if unsure). **The file may not exist yet even after a fully successful wizard run** — it's only written to disk the first time something actually calls `store_set` (Rust's `documents_store_path()` creates the *jobs folder* eagerly but not the JSON file itself), and the one thing that calls `store_set` on a normal run is a one-time `localStorage` → store migration (`static/js/utils.js`) that only writes keys if there was existing `localStorage` data to migrate — nothing, on a fresh install with a fresh WebView. If it's missing, just create it yourself with exactly the two keys below as its entire contents; the app loads whatever's on disk the next time it calls `store_get`/`store_set`, so a hand-written file works the same as one the app wrote itself. Add:
   ```json
   "wsl2BackendEnabled": true,
   "wsl2Distro": "Ubuntu-24.04"
   ```
   (leave `wsl2Python` unset — it defaults to `python3`, which is what step 1 installed into.)
3. **`cargo build`/`cargo run` from `desktop/src-tauri`** (real toolchain snag hit here too: Ed's `rustc` was 1.87.0, below the project's `rust-version = "1.88"` floor — fixed with `rustup update stable`) and launch it. Watch for: does `wsl.exe` actually get invoked (Task Manager, or add a temporary `eprintln!` in `start_backend_wsl2` if it's not obvious); does `/api/health` come back with the right instance token within the 90s window; does `backend.log` (same file the native path uses) show real uvicorn/FastAPI startup output or an error; does a real separation job actually run on the GPU; does closing the app cleanly stop the WSL2-side process (check `wsl.exe -- pgrep -af uvicorn` before/after); does killing the app hard (Task Manager, not a clean close) get caught by the heartbeat watchdog within ~30-40s.

   **A real, more structural finding surfaced getting here**: the app's existing first-run onboarding wizard ("Setting up local runtime" → "Checking FFmpeg" → "Configuring compute device" → "Downloading AI models" → "Starting backend") runs **unconditionally before `start_backend` is ever called**, with zero awareness of `wsl2BackendEnabled`. On a raw git checkout it also immediately fails at the first step — `runtime-manifest.json` ships as an empty placeholder (`runtimeUrl: ""`, a stale macOS-arm64 template), so `probe_runtime()`/`python_path()` find nothing and the wizard tries to download a portable runtime that doesn't exist for a dev build. Workaround for the smoke test: a native Windows `.venv` at the repo root (`python -m venv .venv && .venv\Scripts\pip install -e .`) gives `python_path()` something to find, same idea as the WSL2-side install but on the Windows side, letting the wizard complete normally (FFmpeg and model downloads are automatic from there). This is a second, separate Python environment from WSL2's — its device detection (`ensure_torch_device`) runs against native Windows and is irrelevant noise for the WSL2 path (harmless, just wasted probe time), but the model downloads it triggers land in the same `STEMDECK_DATA_DIR` the WSL2 launch also reads (via `/mnt/...`), so that part isn't wasted. **Not fixed, just routed around** — worth Decision #6's guided-setup flow eventually addressing directly, either by adding its own WSL2-aware branch to this wizard or by giving a WSL2-only user a way to skip the now-redundant native runtime/device steps entirely.

   **A second, real bug found (and fixed) right after that**: with the `.venv` in place, the wizard still failed the exact same way. Traced it to `python_stdlib_ok()` (used only by `probe_runtime()`'s readiness check) — unlike `start_backend`'s own PYTHONHOME computation, which correctly goes through `bundled_python_home()` and sets nothing when there's no bundled stdlib to point at, `python_stdlib_ok()` had its own separate, less careful inline version that *always* set `PYTHONHOME` on Windows, falling back to the raw venv root even when nothing under it held a stdlib copy. A plain `python -m venv .venv` has no bundled stdlib at all (it resolves the system interpreter's via its own `pyvenv.cfg`) — pointing `PYTHONHOME` at it broke `import encodings`, made `python_stdlib_ok()` return `false` regardless of the venv being perfectly fine, and kept `probe_runtime()` reporting `pythonReady: false` forever, so the wizard kept retrying the (for a source checkout, nonexistent) runtime-pack download in a loop no amount of retrying could fix. **Fixed**: `python_stdlib_ok()` now calls `bundled_python_home()` directly instead of reimplementing the same logic worse, so it only sets `PYTHONHOME` when there's actually something there to set it to — same fix philosophy as Decision #7d, reusing an existing correct implementation rather than inventing a second, subtly different one. Pre-existing StemDeck code, nothing to do with WSL2 specifically — worth flagging upstream once this fork's own changes are further along, since any StemDeck contributor doing a plain source-checkout dev setup on Windows would hit the identical wall.

   **A third bug, same symptom, found right after confirming (via `cargo clean -p stemdeck && cargo run`, a genuine 21s recompile) that the `python_stdlib_ok()` fix really was compiled in and the wizard *still* failed identically**: the "empty runtimeUrl" error wasn't only reachable through `pythonReady: false` — `setup.js`'s `runSetup()` also gates on a separate `versionMismatch` check, `Boolean(expectedVersion) && installedVersion !== expectedVersion`, and that expression is unconditionally `true` for *any* fresh source checkout regardless of `pythonReady`. `expectedVersion` comes from the placeholder `runtime-manifest.json`'s `version` field (`"0.1.0-alpha.1"`, always truthy), while `installedVersion` is always `undefined` — no source checkout has ever downloaded a runtime pack, so nothing has ever recorded an installed version. So even with `python_stdlib_ok()` fixed and `pythonReady` correctly `true`, `runSetup()`'s gate (`if (!runtime.pythonReady || versionMismatch)`) still tripped on `versionMismatch` alone and routed into the same doomed `installRuntimePack()` call — explaining why the exact same error persisted through a confirmed clean rebuild. **Fixed**: added a third condition to `versionMismatch` — `Boolean(runtimeStatus.manifest?.runtimeUrl)` — so a manifest with nothing to download can't trigger a "mismatch" at all. A real packaged release always ships a populated `runtimeUrl`, so this doesn't change upgrade behavior there; it only stops a dev/source checkout's placeholder manifest from permanently forcing the update path. No JS toolchain in this sandbox to compile-check it against, so this one was verified by manual re-reading of `runSetup()`'s control flow only (`node --check` confirms it's syntactically valid) — worth confirming in practice that the wizard now gets past the runtime step before treating this as closed.

   **Confirmed on real hardware**: with both fixes applied, the wizard now completes end-to-end for the first time — the app reaches the normal library screen ("Ready to import a track"). That closes out this sub-thread of bugs. One expected, harmless thing seen along the way: a "No Nvidia GPU found, falling back to CPU" message during the wizard's compute-device step — this is the wizard's own native-Windows `ensure_torch_device` probe running against the `.venv` workaround's plain (non-ROCm) PyTorch, exactly the "irrelevant noise for the WSL2 path" called out above, not a new problem. Because this run happened with the store keys from step 2 still unset, `start_backend()` took the **native** path, not WSL2 — so step 2 (setting `wsl2BackendEnabled`) is now, for the first time, actually reachable and is the next thing to do, followed by re-running step 3 with WSL2 actually enabled to get the real signal this smoke test is after. (The runbook's step numbering reads linear but wasn't followed that way in practice — steps 2 and the WSL2-enabled part of step 3 were blocked behind getting a first successful wizard run at all, which took fixing the three bugs above first.)
4. **Report back whatever actually happens** — first-run surprises are likely (wslpath output format, whether `nohup ... & disown` really survives `wsl.exe` exiting, whether the loopback forwarding the earlier echo-server spike confirmed holds for a real FastAPI app under load) — that's the point of doing this before building more on top of it.

### A new bug found doing step 2: `user-data.json` loses the `wsl2*` keys between being written and the app reading them

Ran step 2 for real and it didn't work the way it should have. Sequence: with StemDeck fully closed, `user-data.json` (`D:\OneDrive\Documents\StemDeck\jobs\user-data.json` — the real Documents location turned out to be OneDrive-redirected, not the plain `%USERPROFILE%\Documents` the runbook originally assumed) was written with `wsl2BackendEnabled: true` and `wsl2Distro: "Ubuntu-24.04"` merged in alongside the existing `stemdeck.folders` library data, confirmed on disk via a direct read right after the write. StemDeck was then relaunched. Result: `wsl.exe -d Ubuntu-24.04 -- pgrep -af uvicorn` found nothing, `backend.log` shows an ordinary native-Windows uvicorn startup (new PID each run, no WSL2 involvement visible), and `stemdeck.log`/`setup.log` both show `device=cpu` — the actual signal that matters, since that line is written by the same Python code (`available_torch_devices()`) regardless of whether it's running natively or inside WSL2, and the WSL2/ROCm environment independently verified `torch.cuda.is_available() == True` weeks earlier. All of this says `start_backend()` took the native branch, meaning `wsl2_backend_enabled()` read `false` (or nothing) despite the keys being on disk moments before launch.

Checked `user-data.json` again after that run: **the `wsl2*` keys were gone. Only `stemdeck.folders` remained** (with a `parentId` field added that wasn't there before — clearly the app's own library-init code touching that key, so the store was written to during this run). This isn't a one-off — it happened on both attempts so far. Rust's `store_set()` (`store.set(key, value); store.save()`) should never drop an untouched key; `.save()` persists the *entire* in-memory map, not just the key that changed. Two keys vanishing while a third, unrelated key survives (and gets modified) rules out a simple "wrong file path" explanation — the app clearly is reading and writing *this* file, just apparently without the two keys ever being in its in-memory copy of it.

Not diagnosed yet, and this isn't a case worth guessing at blind — the working theory (unconfirmed) is that `app.store(path)`'s first-load-of-the-process either isn't happening before some other store write clobbers things, or something *external* to the app is involved: this file lives inside an actively-synced OneDrive folder, and a sync-conflict revert between the manual write and the app's next read is a real possibility worth ruling in or out, not just a Rust-side bug. **Added temporary debug logging** to `store_get`/`store_set` (`eprintln!` to stderr — visible directly in the `cargo run` terminal, no more file-log archaeology needed) that prints the resolved store path and value on every read, and on every write additionally prints every key already in the store *before* that write lands. Compiled clean (`cargo check`, verified in a throwaway build against the reference clone's full `desktop/` tree). This is deliberately temporary and noisy — meant to be removed once this is understood, not shipped. Next step: have Ed redo the `user-data.json` edit and relaunch once more with this build, and read the terminal output — it will show directly whether the keys were ever loaded into the store's in-memory copy at all, which settles this without further speculation.

### The `wsl2*` store keys mystery: resolved (not a real bug) — and the first genuine WSL2-path bug found

With `user-data.json` rewritten once more and a single, clean `cargo run` (no extra launches this time), the debug logging showed `store_get("wsl2BackendEnabled") -> Some(Bool(true))` and `store_get("wsl2Distro") -> Some(String("Ubuntu-24.04"))` from the very first read — the keys loaded correctly. So the earlier "keys vanish" symptom was very likely caused by the extra back-to-back `cargo run`s in the previous attempt (an old, already-running instance's own `store_set("stemdeck.folders", ...)` landing after the file was correctly written but before the debug build ever read it), not a real bug in `store_get`/`store_set` or an OneDrive sync issue. Worth remembering as a general gotcha — this app doesn't have a single-instance guard, so launching it again while a previous window/process is still around can silently stomp on store writes — but not worth chasing further as a standalone bug. The temporary `store_get`/`store_set`/`documents_store_path` debug logging has been removed now that it's served its purpose.

**With the store keys finally loading correctly, `start_backend()` took the WSL2 branch for the first time — and hit a real bug immediately**: the wizard's "Starting backend" step failed with:
```
wslpath failed for C:\Users\ed\AppData\Local\StemDeck: wslpath: C:UsersedAppDataLocalStemDeck
```
Note every backslash is simply gone from the second path, with whatever followed each one left bare (`Users`+`ed`+`AppData`+`Local`+`StemDeck` run together). This is a genuine `wsl.exe` interop quirk, not a `wslpath` bug and not a logic bug in `wsl_path()`: when `wsl.exe` forwards arguments after `--` through to the Linux side, it runs them through a layer that treats a bare backslash as a shell escape character and silently drops it — so a Windows path passed as a plain argument arrives on the Linux side already mangled, before `wslpath` (or anything else) ever sees it correctly. **Fixed** in `wsl_path()`: the Windows path is now converted from `\` to `/` separators before being handed to `wsl.exe`/`wslpath` — Windows path parsing (and `wslpath` itself) treats the two separators as equivalent, so this sidesteps `wsl.exe`'s own argument-mangling without reimplementing any of `wslpath`'s actual translation logic (drive-letter casing, UNC paths, etc. — still fully handled by the real `wslpath`, exactly as originally intended). Every other Windows path used on the WSL2 launch path (`backend_dir`, `data_dir`, `jobs_dir`, `log_path`, the settings mirror) already routes through this same `wsl_path()` function, so this one fix covers all of them. Verified via `cargo check` and `cargo clippy`, both clean, against a full reference build.

### Next: the WSL2-launched backend never comes up at all — no `backend.log`, not even empty

With the `wslpath` fix in place, `start_backend_wsl2()` ran without error (`wsl.exe` exited 0) and got past that step for the first time — but then timed out after 90s with `No backend log output was captured at ...backend.log`. Checked directly: no `backend.log` (or `.1`) exists at all, only an old rotated `.2` from an earlier native run. That's a stronger signal than it might look: the shell command is `cd {backend} && {env} nohup {python} -m uvicorn ... > {log} 2>&1 </dev/null & disown` — the `> {log}` redirect happens as soon as that line of the script executes, so if the log file was never even *created* (empty or otherwise), the redirect itself never ran, which (given `&&`) most likely means `cd {backend}` failed.

That failure wouldn't surface as an error here even if so: `cmd1 && cmd2 & disown` backgrounds the *whole* `cmd1 && cmd2` compound as one job, and the exit status Rust actually inspects is `disown`'s own (which succeeds as long as *some* job was backgrounded) — not the compound's. So a `cd` failure inside the backgrounded job is invisible to the current success/failure check, which only ever catches wsl.exe/bash itself failing to start. Given the freshly-fixed `wslpath` translation is what's producing `{backend}` (`shell_quote()`d, so should survive spaces — and this repo's path does have one, `Stem Separation`), and given the *first* wsl.exe interop bug we found was `wsl.exe` mangling characters within a forwarded argument, it's a reasonable next guess that something about the complex, multi-quoted `inner_cmd` string itself doesn't survive `wsl.exe`'s forwarding intact — but that's a guess, not confirmed, and this isn't a case worth patching blind a second time.

**Added debug logging** (temporary, same pattern as before) to `start_backend_wsl2()`: prints the exact `inner_cmd` string right before it's sent to `wsl.exe`, and unconditionally prints `wsl.exe`'s own exit status/stdout/stderr afterward (previously only captured on failure). Compiled clean (`cargo check`). Next step: one more `cargo run`, paste the `[stemdeck][debug]` lines — the printed `inner_cmd` is also worth Ed pasting directly into a real WSL2 shell by hand (bypassing `wsl.exe`-from-Windows entirely) as an even more direct isolation test, since that would settle in one step whether this is a `wsl.exe` interop issue or a problem with the command itself.

### Root cause found: a one-shot, detached `wsl.exe` invocation cannot outlive itself — Decision #8's whole design was wrong

The printed `inner_cmd` pasted directly into an already-open interactive `wsl -d Ubuntu-24.04` shell worked immediately: correct `backend.log` content, a clean `curl /api/health` response. That ruled out the command itself (quoting, env vars, the `nohup`/`disown` pattern) as the problem — it isolated the failure specifically to running that same command through a **one-shot** `wsl.exe -- bash -lc "..."` invocation from Windows.

What followed was a sequence of isolation tests, each changing exactly one variable, run directly on Ed's hardware:

- **A second, idle interactive WSL2 window left open** while retesting the one-shot launch from a third window — still failed identically. This rules out "the WSL2 VM just needs to be kept warm/alive by *some* session" as the explanation.
- **`setsid` added in front of `nohup`** inside the backgrounded command — still failed identically. Linux-side job-control isolation (`setsid` creating a new session, detached from any controlling terminal) is the textbook fix for "backgrounded process dies with its parent shell," and it didn't help either.
- **A plain synchronous (non-backgrounded) one-shot `wsl.exe -- bash -lc "echo hello > file"`** — worked fine. This isolates the problem specifically to *backgrounding* within a one-shot invocation, not to `-lc`/login-shell semantics, path translation, or DrvFs in general (also separately confirmed by a trivial direct write-to-DrvFs-path test).
- **The full foreground uvicorn command (no backgrounding, no `nohup`/`disown` at all), run synchronously one-shot** — also worked fine, server started and answered cleanly. So even a real, long-running server process is fine synchronously; only *backgrounding it and letting `wsl.exe` exit* was the failure case.

None of `nohup`, `disown`, or `setsid` — alone or combined — let a backgrounded process launched by a one-shot `wsl.exe` invocation survive that invocation's own exit. But the identical command pasted into an already-open, long-lived interactive `wsl.exe` shell always worked. The distinguishing variable across every test was never anything on the Linux side — it was whether the **Windows-side `wsl.exe` process itself** was still alive.

**Decisive confirmation**: launching `wsl.exe` via PowerShell's `Start-Process -NoNewWindow` — a genuinely detached-but-*alive* Windows background process, running uvicorn as `wsl.exe`'s own direct foreground command with **no** Linux-side backgrounding trick at all — worked. `curl http://127.0.0.1:8000/api/health` returned a full `200 OK` with valid JSON (the first time any `wsl.exe`-from-Windows path had produced a working backend). One loose end from that same test: `backend.log` briefly reported "does not exist" from the Windows side via `Get-Content` immediately afterward, even though uvicorn was demonstrably running and had the file open. Re-checked significantly later (after the process had been running a while): the file was there with fully correct content. This is a DrvFs quirk, not a second bug — a file a WSL2 process still has open for writing can be briefly invisible to Windows' own metadata cache before it catches up. It doesn't block anything real: `/api/health` polling doesn't depend on the log being visible, and `log_hint()` is only ever consulted after a failure, by which point that window has long since passed.

So Decision #8's core design assumption — "fire a one-shot detached `wsl.exe`, no Child handle retained, `nohup`/`disown` on the Linux side make the backend outlive it" — was simply wrong on real hardware, however reasonable it looked on paper (and however well the underlying `nohup`+`disown`+background pattern works on a native Linux box, verified separately in a sandbox with no WSL2 involved at all). What actually keeps a WSL2-launched process alive is keeping `wsl.exe` itself alive.

**Fixed**: `start_backend_wsl2()` now `spawn()`s `wsl.exe` (not `.output()`) with the uvicorn command as its direct, non-backgrounded child (`cd {backend} && {env} exec {python} -m uvicorn ...` — the `nohup`/`disown` wrapper is gone entirely, and `exec` replaces the shell with uvicorn once `cd` succeeds so there's one less process in the chain), and returns the `Child` handle. `BackendProcess::Wsl2` now carries that `Child` (previously a unit variant carrying nothing, per the old design) and the caller holds onto it for the backend's whole lifetime instead of letting `wsl.exe` exit. Concretely:
- `wait_for_health_headless()` now takes `&mut Child` and checks `try_wait()` on every poll iteration, so an early `wsl.exe` exit (bad distro name, `cd` failure, `python3` not found) is reported as that immediately, instead of only ever surfacing as an opaque 90-second timeout — the same improvement the Native path already had via its own `wait_for_health()`.
- `stop_backend_wsl2()` now takes `&mut Child` too, and — this is new — falls back to killing it directly if the graceful `/api/desktop/shutdown` HTTP request goes unanswered within the drain deadline, matching the Native path's SIGTERM-then-kill escalation. The old version could only ask nicely and then log a warning and give up, because there was no Child to fall back to; the risk of a name/port-based kill taking down some unrelated process doesn't apply here since `child` is exactly the `wsl.exe` process this launch spawned, running uvicorn as its own direct child.
- `stop_backend`/`stop_backend_and_wait` (the two callers) updated to match `BackendProcess::Wsl2(mut child)` and thread it through.
- The temporary `inner_cmd`/`wsl.exe`-output debug `eprintln!`s added while diagnosing the earlier "no log at all" symptom have been removed now that they've served their purpose — same as the store debug logging before them.

Verified via `cargo check` and `cargo clippy` (both clean) against a full reference build (`/tmp/verify7_root`, mirroring the established methodology).

**Confirmed on real hardware, and it works end-to-end for the first time.** `wsl.exe -d Ubuntu-24.04 -- pgrep -af uvicorn` showed uvicorn running (PID 337) while the app was open, and gone after closing it -- the first time the app's own clean-close path has actually stopped a WSL2-launched backend. `backend.log` for that run shows the complete real lifecycle: startup, the WebView loading the actual UI (`GET /`, every `.css`/`.js` asset, `200`/`304` as expected), live `/api/health` and `/api/queue` traffic, several `POST /api/desktop/heartbeat 200 OK`, then `POST /api/desktop/shutdown 200 OK` followed by a clean `Shutting down` / `Application shutdown complete` / `Finished server process [337]` -- no heartbeat-timeout kill this time, an actual graceful request-driven shutdown. `stemdeck.log` corroborates: `desktop shutdown requested` at the moment of closing, not the `no desktop heartbeat for over 30s; stopping backend` line every earlier run (native and the failed WSL2 attempts alike) had shown instead. This closes out the WSL2 backend-launch-and-lifecycle thread that's been open since Decision #8 -- start, health, live serving, and clean shutdown all now work exactly as designed.

One detail from that same run, worth recording so it isn't mistaken for a new problem: `stemdeck.log` still shows `demucs config: model=htdemucs_6s device=cpu`, even though this run was genuinely inside the ROCm-capable WSL2 environment.

**Correction to an initial (wrong) explanation**: first guess was that this just echoed the Windows-native NVIDIA-only `ensure_torch_device()` wizard step's persisted `config.json` value. Checked the actual code path the log line comes from (`app/main.py`'s startup log calls `get_demucs_device()`) and that guess doesn't hold up: `get_demucs_device()` (`app/core/settings.py`) does a **live** hardware probe every time the device choice is `"auto"` (the default) -- `detect_compute_device()` → `detect_torch_device()` → `torch.cuda.is_available()` -- run fresh inside whichever Python process is asking, WSL2-launched or not. It does not read the Rust-persisted `torchDevice`/`torchDeviceReason` at all; those are separate, wizard-UI-only bookkeeping. So `device=cpu` here means `torch.cuda.is_available()` genuinely returned `False` inside *this* WSL2-launched process -- despite Ed's own standalone check weeks earlier, in an interactive shell in the same distro/python, returning `True`.

**Working theory, not yet confirmed**: `start_backend_wsl2()` runs the uvicorn command via `bash -lc "..."`, a *non-interactive* login shell. `bash -lc` sources `/etc/profile` and `~/.profile`/`~/.bash_profile`, but critically does **not** source `~/.bashrc` unless one of those files explicitly does so -- and Ubuntu's default `~/.bashrc` has an early-return guard (`case $- in *i*) ;; *) return;; esac`) that skips the rest of the file for any non-interactive shell even if it is sourced. If whatever ROCm needs (`HSA_OVERRIDE_GFX_VERSION` -- commonly required to make an RX 7800 XT, which ROCm doesn't officially list as supported, present itself as a supported gfx target -- or ROCm's own PATH/LD_LIBRARY_PATH additions) lives in Ed's `~/.bashrc` rather than `~/.profile`, our launch simply never sees it, while his earlier interactive-shell verification did. This is a plausible, checkable gap in the new launch mechanism, not the wizard-config explanation given a moment ago -- worth confirming directly before deciding on a fix, same as every other bug this session.

Asked Ed to run one isolated check that mirrors our exact launch shell, without going through the app at all:
```
wsl.exe -d Ubuntu-24.04 -- bash -lc "python3 -c 'import torch; print(torch.cuda.is_available())'"
```
**Result: `False`.** Looked like the shell-init theory confirmed -- until the control test disproved it: forcing a genuinely interactive shell even in a one-shot invocation (`bash -ic` instead of `-lc`) **still** printed `False`, and a direct `grep` for `HSA|ROCM|HIP|GFX` across `~/.bashrc`/`~/.profile`/`~/.bash_profile` found **nothing at all** -- there was never any ROCm-specific shell setup to miss in the first place. Theory dead.

Next isolation: compare `sys.executable`/`torch.__version__`/`torch.version.hip` between the non-interactive launch shell and a genuine interactive `wsl -d Ubuntu-24.04` session, in case two different `python3`s were in play. They were **identical** in every respect -- same `/usr/bin/python3`, same `whoami` (`ed`), and, this is the actual finding: **`torch.__version__` is `2.6.0+cu124` with `torch.version.hip == None` in both.** That is a CUDA-targeted PyTorch build, not a ROCm one, on a machine with no NVIDIA GPU -- of course `torch.cuda.is_available()` returns `False` there, identically, every way it's invoked. The interactive-vs-one-shot distinction was a complete red herring; both paths were consistently exercising the same (wrong) torch build the whole time.

So the actual regression has nothing to do with anything built this session -- somewhere between Ed's original ROCm/PyTorch verification (`torch.cuda.is_available() == True`, weeks earlier) and now, the ROCm-enabled torch install was replaced by a plain CUDA build. The likely mechanism: `pyproject.toml` declares `"torch>=2.6,<2.7"` as a bare dependency with no index-url pin to ROCm wheels (`pip`'s default index only carries CUDA/CPU builds), and Decision #7a/7b's runbook step 1 (`pip install --user --break-system-packages -e .`, done earlier this same project) resolves and installs whatever satisfies that specifier from the default index -- silently overwriting a hand-installed ROCm wheel that also happened to satisfy the same version range. **Not yet confirmed as the exact mechanism** (haven't checked pip's install history/timestamps to prove *when* this happened), but it fits every fact so far and doesn't require inventing anything new: the runbook always said this `pip install -e .` step was needed, nobody flagged at the time that it could clobber a differently-sourced torch build already satisfying the same specifier.

**This is a separate, unrelated bug from everything else in this document** -- the WSL2 launch mechanism itself (start/health/serve/shutdown) is fully working, confirmed on real hardware just above; this is purely "the wrong PyTorch build is installed in that environment."

Checked the installed ROCm runtime: `7.2.1`. This immediately raised a real conflict rather than a simple reinstall: AMD's own current guidance for ROCm 7.2.1 pairs it with PyTorch 2.9.1 (wheels served from `repo.radeon.com`, not `pytorch.org`) -- but this project deliberately pins `torch>=2.6,<2.7` (see the pin's own comment: demucs 4.0.1 calls `torchaudio.save()` internally, and torchaudio 2.7+ dropped the built-in writer that call needs, requiring the separate `torchcodec` package instead, which has its own ABI issues with newer torch). Installing AMD's newest recommended build would very likely fix GPU detection and then immediately break every real separation job at the save step.

Chose the reversible option first: PyTorch's own official ROCm 6.2.4-tagged build of torch 2.6.0 (`pip install --force-reinstall --index-url https://download.pytorch.org/whl/rocm6.2.4 torch==2.6.0 torchaudio==2.6.0`), matching the project's existing pin exactly and requiring no code changes -- explicitly flagged to Ed going in that a torch wheel built against ROCm 6.2.4 running against an installed ROCm 7.2.1 userspace is a real version gap whose compatibility wasn't known in advance, just worth trying since it's cheap and reversible.

**Result: install succeeded (`torch.__version__` now correctly reports `2.6.0+rocm6.2.4`, `torch.version.hip` reports a real HIP build string `6.2.41134-65d174c3e`), but `torch.cuda.is_available()` is still `False`.** So the concern was justified -- a ROCm-6.2.4-built torch does not detect the GPU against a ROCm 7.2.1 runtime, at least not with nothing else changed. Not yet clear whether this is a hard ABI incompatibility, a missing GPU-visible-to-ROCm-at-all problem (i.e. not torch-specific), or something WSL2-GPU-passthrough-specific -- `torch.cuda.is_available()` swallows exceptions internally in some torch versions and just returns `False`, so this alone doesn't distinguish those.

**Root cause found -- this was never a torch-version problem at all.** `rocminfo` run directly (no torch involved) prints: `WSL environment detected.` / `hsa_init Failed, possibly no supported GPU devices`. So ROCm's own runtime knows it's running under WSL2 and is deliberately trying WSL-specific GPU access -- and that's failing at the HSA layer, before torch is even in the picture. Two device-node checks confirm which path is actually available: `/dev/kfd` (the native-Linux ROCm kernel-driver device) does not exist, while `/dev/dxg` (WSL2's GPU paravirtualization device, the one WSL-aware ROCm builds are supposed to talk through instead) does. `dpkg -l | grep rocm` shows the full ROCm 7.2.1 stack installed (`rocm`, `rocm-core`, `hsa-rocr`, etc., all at matching `7.2.1.70201-81~24.04` versions) -- **except** `libhsa-runtime64-1` (`5.7.1-2build1`) and `libamd-comgr2` (`6.0+git20231212...`), whose version strings don't match AMD's `X.Y.Z.70201-81~24.04` pattern at all and look like they came from Ubuntu's own repo rather than AMD's ROCm apt repo -- a plausible package-name collision shadowing the HSA runtime the 7.2.1 stack actually needs.

Researched AMD's official WSL2 install path (`rocm.docs.amd.com`, the `radeon-ryzen`/WSL install page) rather than guessing at a fix: **ROCm on WSL2 is not installed the same way as native Linux ROCm.** The documented procedure is `amdgpu-install -y --usecase=wsl,rocm --no-dkms` (note `--usecase=wsl`, not the plain `rocm` usecase, and `--no-dkms` since WSL has no kernel module of its own to build against) -- this pulls a WSL-specific HSA runtime build (historically named something like `hsa-runtime-rocr4wsl-amdgpu` in AMD's own build tooling) that knows how to talk through `/dev/dxg` instead of expecting `/dev/kfd`. It also requires a matching **WSL-tagged Windows GPU driver** ("AMD Software: Adrenalin Edition for WSL2" -- a different download from the regular Windows Radeon driver). A related upstream issue (ROCm/ROCm#4682) describes ROCm 6.4.0 installing an incorrectly-versioned `libhsa-runtime64.so` specifically on WSL2/Ubuntu 24.04, fixed in 6.4.1 -- confirming this WSL/native-Linux HSA-runtime-versioning class of bug is a real, recurring category for this project's exact platform (WSL2 + Ubuntu 24.04 + ROCm), not something specific to Ed's machine.

So the install Ed has is very likely the plain native-Linux `rocm` usecase (matching everything in `dpkg -l` except the two odd-versioned packages), not the WSL-specific one -- explaining exactly the observed symptom. **Not yet fixed.** This is now a system-level ROCm/Windows-driver setup problem, not anything in this project's code or the WSL2 launch mechanism (which remains fully working, confirmed above) -- next step is for Ed to redo the ROCm install via the WSL-specific `amdgpu-install --usecase=wsl,rocm --no-dkms` path and confirm his Windows-side AMD driver is the WSL2-tagged build, then re-run `rocminfo` to confirm `hsa_init` succeeds before touching torch again at all.

Sources consulted: [rocm.docs.amd.com WSL install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-radeon.html), [ROCm/ROCm#3402 (RX 7800 XT on WSL2)](https://github.com/ROCm/ROCm/issues/3402), [ROCm/ROCm#4682 (WSL2 libhsa-runtime64 versioning)](https://github.com/ROCm/ROCm/issues/4682).

### The WSL-side fix: confirmed working (system level) -- and a second, distinct ABI gap found underneath it

Ed asked whether to `sudo amdgpu-uninstall` and reinstall via the WSL-specific path rather than patch on top of the existing install -- correct instinct, and it matches AMD's own docs verbatim ("Radeon Software for Linux does not support in-place upgrades"). Confirmed the exact procedure via a fresh fetch of the same docs page (`amdgpu-uninstall`, then re-download/reinstall `amdgpu-install` for Ubuntu 24.04/ROCm 7.2 if needed, then `amdgpu-install -y --usecase=wsl,rocm --no-dkms` again).

Also answered a side question about the Windows driver: Ed has AMD Software: Adrenalin Edition 26.10 installed, and the WSL docs cite "26.1.1 for WSL2" -- checked AMD's own versioning (year.month.revision; 26.1.1 shipped ~Jan 21 2026, confirmed via AMD's release-notes URLs) and 26.10 is genuinely ~9 months newer, so this would have been a real downgrade. Advised against it: the docs' driver citation is almost certainly just what was current when that ROCm-7.2-pinned page was written, not a hard requirement, and there was already direct evidence the Windows side was fine -- `/dev/dxg` (the actual WSL2 GPU-passthrough device) already existed and was populated before any of this driver discussion, meaning Windows was already exposing the GPU to WSL2 correctly. Recommended fixing the Linux/ROCm side first and only revisiting the driver if that didn't resolve it. It did.

**Result: after the reinstall, `rocminfo` now succeeds completely** -- both `Agent 1` (the CPU) and `Agent 2` (`gfx1101`, "AMD Radeon RX 7800 XT") are listed correctly, no `hsa_init` failure. The WSL-specific ROCm install was exactly the fix.

**But torch still can't see it -- a second, different problem, one layer up.** `torch.cuda.is_available()` is still `False` even with `rocminfo` fully working. Rather than trust `is_available()` (which swallows its own errors), forced a real GPU op instead (`torch.zeros(1).cuda()`), which surfaced the actual exception: `RuntimeError: No HIP GPUs are available`. This is a HIP-layer error, not an HSA-layer one -- meaningfully different from the earlier `hsa_init Failed`. The likely explanation: PyTorch's ROCm wheels bundle their own copies of the HIP runtime libraries (`libamdhip64.so` and friends) inside the wheel itself rather than only relying on the system's -- so even though the system's HSA layer is now correctly WSL-aware (ROCm 7.2.1, `rocr4wsl`-flavored), the *torch wheel's own bundled HIP runtime* is still the one built for ROCm 6.2.4 (the closest official `pytorch.org` wheel to the project's `torch==2.6.0` pin, chosen earlier in this thread specifically to avoid a different, unrelated problem -- see above), and that bundled 6.2.4 HIP layer doesn't correctly enumerate a device through the newer, WSL-specific 7.2.1 HSA runtime underneath it. Not a config problem this time -- a real build-version gap between the torch wheel and the system ROCm install.

**The path forward has a real tradeoff, not yet resolved.** AMD's own recommended pairing for ROCm 7.2.1 is PyTorch 2.9.1 (via `repo.radeon.com`) -- almost certainly the one that would actually enumerate the GPU correctly, since it would ship a matching, WSL-aware HIP build. But that reopens the exact conflict flagged when this thread started: the project pins `torch>=2.6,<2.7` specifically because `demucs` 4.0.1 calls `torchaudio.save(path, wav, sample_rate=..., encoding=..., bits_per_sample=...)` (confirmed directly by downloading and reading demucs 4.0.1's actual source, `demucs/audio.py`), and torchaudio's own docs confirm why that pin exists: by torchaudio 2.9, `torchaudio.save()` is a straight alias for `save_with_torchcodec()` ("Starting with version 2.9, we have transitioned TorchAudio into a maintenance phase" -- per torchaudio's own 2.9 docs), which is TorchCodec's own encoder API, not the classic `encoding=`/`bits_per_sample=` dispatcher-backend signature demucs calls with. (For reference: torchaudio 2.7.0's docs still show the classic signature working fine with those exact kwargs -- so the break isn't at 2.7 as the pin comment's wording implies, it's specifically the `save_with_torchcodec()` aliasing that lands by 2.9. Worth a precise correction to that pin comment once this is actually resolved, so it names the real mechanism instead of just a version cutoff.) So installing torch 2.9.1 as-is would very likely make every stem-writing call in demucs raise a `TypeError` for unexpected keyword arguments, not silently work.

A real, buildable fix exists rather than just "can't do it": the project already depends on `soundfile` (`>=0.12`, already required for the *existing* torchaudio-2.6 WAV-writing path -- see the pin comment just above this one in `pyproject.toml`), so a small compatibility shim -- monkeypatching `demucs.audio.ta.save` (or `torchaudio.save` globally) to write via `soundfile` directly instead of going through torchaudio's version-drifting `save()` API at all -- would let torch/torchaudio move to whatever version ROCm 7.2.1 actually needs without depending on torchaudio's save API surface being stable across versions ever again. Raised with Ed as a real decision rather than just done, since it touches `pyproject.toml`'s platform-wide torch/torchaudio pin (today's pin is a single blanket range for all non-Intel-macOS platforms, not scoped to just the WSL2 path). **Ed chose to go ahead with the bump + shim.**

### Implemented: torch/torchaudio bumped to 2.9.x, and a soundfile-based `torchaudio.save` shim

Before writing anything, traced the actual call sites rather than assuming: downloaded demucs 4.0.1's real source (`pip download demucs==4.0.1 --no-deps`) and confirmed `demucs/audio.py`'s `save_audio()` is the *only* function that calls `torchaudio.save()`, always as `ta.save(str(path), wav, sample_rate=samplerate, encoding=encoding, bits_per_sample=bits_per_sample)` (`.wav`) or the same without `encoding` (`.flac`). Then checked which of demucs's own functions this project's own code actually reaches: `app/pipeline/demucs_worker.py`'s `_run_one_job()` is the **sole** call site project-wide for `save_audio` (confirmed via `grep` across `app/`), always with the exact same fixed arguments (`.wav`, `as_float=False` -> `encoding="PCM_S"`, `bits_per_sample=16`). Also confirmed demucs's *read* path (`load_track()` in `demucs/separate.py`) tries its own ffmpeg-based `AudioFile.read()` first and only ever falls back to `torchaudio.load()` if ffmpeg is missing or fails -- and this project manages its own ffmpeg install, so that fallback is never actually reached in practice. Net result: only `torchaudio.save()`'s API break actually matters here; `torchaudio.load()` needs no shim.

Also worth noting: this is not a new pattern for this project. `app/pipeline/demucs_onnx_worker.py` (the DirectML/ONNX path) already writes stems via `soundfile` directly (`_write_stem()`), for unrelated reasons -- this is the same approach, generalized into a drop-in `torchaudio.save` replacement.

**Added `app/pipeline/torchaudio_save_shim.py`**: a `soundfile`-based reimplementation of `torchaudio.save()`'s classic signature (`uri, src, sample_rate, channels_first=True, format=None, encoding=None, bits_per_sample=None, buffer_size=4096, backend=None, compression=None`), covering every `encoding`/`bits_per_sample` combination torchaudio itself documented for its old backend (not narrowed to just this project's one current call site, so a future change to how demucs calls it keeps working). `install()` monkeypatches `torchaudio.save` in place; safe to call at any point relative to demucs's own import, since `import torchaudio as ta` binds the module object, and `ta.save(...)` resolves against whatever `torchaudio.save` is *at call time*.

Verified the actual logic (not just syntax) despite this sandbox having no `torch` installed: extracted the pure-Python subtype-resolution and array-transpose logic (the only parts that don't need a real `torch.Tensor` -- `.detach().cpu().numpy()` on a real tensor and a plain `numpy` array are equivalent for this code's purposes) and round-tripped it through a real `soundfile` write+read, confirming (1) every documented encoding/bits_per_sample pair maps to the right `soundfile` subtype, (2) a channels-first stereo array transposes correctly to soundfile's frames-first convention and round-trips with only the small quantization error expected from 16-bit PCM, (3) a mono 1-D array is left untransposed and round-trips cleanly, (4) an unsupported combination raises rather than silently miswriting. `python3 -m py_compile` confirms both the new shim and the edited `demucs_worker.py` are syntactically valid (no local torch/cargo-equivalent toolchain in this sandbox to fully import-check against, mirroring the constraint noted for the earlier `setup.js` fix).

**Wired into `demucs_worker.py`**: `install()` is called at the top of `main()`, before `get_model()` and before the job loop -- unconditionally, on every platform/device, not just WSL2/ROCm, since it's harmless on older torchaudio too and keeps the worker simple (no version-sniffing branch).

**`pyproject.toml`**: bumped the main torch/torchaudio pin from `>=2.6,<2.7` to `>=2.9,<2.10` (the Intel-macOS-x86_64 line stays at `2.2.x`, an unrelated, already-solved constraint about wheel availability, not touched). Rewrote the pin's comment to name the real mechanism (torchaudio.save() becoming a `save_with_torchcodec()` alias by 2.9, not a version cutoff at 2.7 as the old comment implied -- see the correction earlier in this doc) and to point at the shim. Rewrote the `soundfile` dependency's comment too: it's no longer justified by torchaudio's own internal use of libsndfile, but directly by this project's own shim now depending on it explicitly. Confirmed still-valid TOML via `tomllib.load()`.

**Not yet done -- this is the actual remaining step**: `pyproject.toml`'s `>=2.9,<2.10` only stops `pip install -e .` from *fighting* whatever torch is already installed; it doesn't (can't) express "install AMD's specific ROCm 7.2.1 wheel" the way a plain dependency range never could for the earlier 6.2.4 build either. Ed still needs to manually install the actual ROCm-7.2.1-matched wheels, this time from `repo.radeon.com` (not `pytorch.org`, which doesn't publish ROCm 7.2.1 builds at all):
```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
pip install --user --break-system-packages --force-reinstall torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
```
Then the same real-op check as before (`torch.zeros(1).cuda()`, not the exception-swallowing `is_available()`) to confirm the GPU actually enumerates this time. If it does, next is an actual separation job through the app to confirm the shim writes a real, valid stem file end-to-end -- nothing above has run against the real app yet, only demucs's source and the shim's logic have been verified in isolation.

**Done -- and it worked.** One extra wheel was needed beyond torch/torchaudio: the ROCm 7.2.1 torch build declares a dependency on a specific ROCm-matched `triton` (`triton==3.5.1+rocm7.2.1.gita272dfa8`), which -- same pattern as torch/torchaudio -- only exists on `repo.radeon.com`, not PyPI, so the first install attempt failed with `No matching distribution found for triton==...`. Downloaded that wheel too (`triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl`, found via the same AMD docs page) and reinstalled all three together.

**Confirmed: `torch.zeros(1).cuda()` now returns `tensor([0., device='cuda:0')` instead of raising.** The full chain -- Windows AMD driver -> WSL2 `/dev/dxg` passthrough -> WSL-specific ROCm 7.2.1 install (`amdgpu-install --usecase=wsl,rocm`) -> matching torch/torchaudio/triton wheels from `repo.radeon.com` -- is finally working end to end, closing out the entire GPU-detection thread that started with the very first "device=cpu" observation above. Two harmless messages appeared in the same output and are worth recording so they aren't mistaken for new problems later: (1) a pip dependency-conflict warning naming the *old* `torch<2.7,>=2.6` pin -- expected, since it's pip reading the editable install's cached metadata from before today's `pyproject.toml` edit; clears up on the next `pip install -e .` and doesn't affect what's actually installed; (2) `Resource leak detected by SharedSignalPool, 2 Signals leaked` on process exit -- a known benign ROCm/HIP shutdown message.

**Next**: an actual separation job through the app itself -- confirms both that torch correctly picks `cuda` as `get_demucs_device()`'s live probe result now (not just in a standalone script) and that `torchaudio_save_shim.py` writes a real, valid stem file end to end, which nothing so far has exercised against the real app.

### Confirmed: a real separation job ran end-to-end on the GPU -- this whole thread is closed

Ed ran one. `stemdeck.log` for the job:
```
demucs config: model=htdemucs_6s device=cuda
stemdeck.pipeline [6dff07deec84] separating on device=cuda
stemdeck.pipeline [6dff07deec84] done device=cuda model=htdemucs_6s prepare=1.6s analyze=40.3s separate_startup=20.1s separate=65.3s post=17.2s beatgrid=18.8s total=163.3s
```
`device=cuda` -- `get_demucs_device()`'s live probe now genuinely resolves to the GPU inside the real app, not just a standalone script. Windows Task Manager's GPU tab corroborated it independently: `Compute 0` climbed to 86% utilization and GPU temperature rose 41°C -> 51°C during the `separate=65.3s` window, with dedicated GPU memory actually growing -- real compute, not an idle device sitting there. (The ~20s `separate_startup` is the CPU-bound worker-process/model-load cost, which is why an early glance at Task Manager mid-job can look GPU-idle before separation itself starts -- not a bug, just caught the wrong moment.)

`backend.log` confirms the write side too: all six `htdemucs_6s` stems (`vocals`/`drums`/`bass`/`guitar`/`piano`/`other`) were written, served back over HTTP with `206 Partial Content` (proof they're valid, byte-range-seekable WAV files the player could scrub), and played in the UI -- `torchaudio_save_shim.py` works correctly against a real job, not just the isolated round-trip test from earlier. The only new thing in the log worth noting is harmless: a `FutureWarning` from `rotary_embedding_torch` (a demucs dependency) about `torch.cuda.amp.autocast` being deprecated in favor of `torch.amp.autocast('cuda', ...)` -- a warning from third-party code about torch's own API evolution, not anything to fix here.

This closes the entire WSL2/ROCm GPU thread that started with "the WSL2-launched backend never comes up at all." Full chain, now confirmed working end-to-end on real hardware: Rust spawns and holds a `wsl.exe` Child for the backend's lifetime -> WSL-specific ROCm 7.2.1 install (`amdgpu-install --usecase=wsl,rocm`) -> matching torch/torchaudio/triton wheels from `repo.radeon.com` -> `torchaudio_save_shim.py` keeping demucs's stem-writing working on the newer torchaudio -> a real separation job, on the GPU, producing correct stems the app plays back. Remaining work drops back to the priority list below -- clean shutdown/heartbeat-watchdog verification on this now-GPU-working configuration specifically hasn't been re-checked since the torch bump (worth a quick pass, though nothing in this thread's changes touched that code path), then the large-file cleanup, Decision #6's guided setup flow (now with a proven, scriptable recipe to automate), and the rest of Phases 3-6.

Separately, asked Ed how to actually exercise a real separation job through the app (rather than just health-checking an idle backend) to see this resolved live in `stemdeck.log`'s `device=` line: drag an audio file onto the app window (or paste a YouTube/SoundCloud link into the search bar) to queue an import, which triggers a demucs separation automatically once the import finishes; watch progress via the import-queue rail button; and, once it's plausibly using ROCm, confirm actual GPU utilization independently of the app's own logging -- either `rocm-smi`/`radeontop` in a WSL2 shell during the job, or Windows Task Manager's Performance tab (select the GPU, watch the "Compute" engine graph, which does reflect WSL2 GPU workloads on a recent enough WSL2/driver version).

Once the mechanism itself is confirmed working end-to-end, remaining work in priority order: (1) the accidentally-committed large-file cleanup noted above — small, should happen first; (2) the guided WSL2/ROCm setup flow (Decision #6, scoped by #7a-#7c now) that automates what step 1-2 above just did by hand; (3) Phase 3 (Settings UI) — for both the DirectML `dml` option and, separately, wherever the guided-setup flow surfaces WSL2/ROCm status/controls; (4) Phase 4 (packaging, now with the smaller scope Decision #7d's correction implies — no new device enum work, but the guided-setup script itself needs to ship with the app); (5) Phase 5 (testing/validation, now also covering the WSL2 hop's latency and the DrvFs I/O cost noted in the implementation section above); (6) Phase 6 (docs & release).

The new and changed files (`app/core/config.py`, `app/core/settings.py`, `app/pipeline/separate.py`, `app/pipeline/demucs_onnx_worker.py`, `app/pipeline/onnx_export/` (new), `pyproject.toml`, `scripts/test_dml_smoke.py`, `scripts/validate_split_export.py` (new), `app/main.py`, `desktop/src-tauri/src/main.rs`, this file) are written to your working copy but not yet `git add`/`git commit`-ed — same as README/NOTICE after the Phase 0 merge, that step is yours to do, same reasoning as before: git operations need network access this session doesn't have to your machine.

---

## Preparing for a first public GitHub push

Ed asked what needed to happen before publishing this repo to GitHub as AnyStemDeck, noting the app still identified itself as "StemDeck" in most places. Two problems, investigated directly against the real repo via the device bridge (`git`, `du`, `grep` run straight against Ed's checkout, not assumed): a large-file history problem, and an incomplete rename. Ed confirmed both should be fixed now, fully: rewrite git history immediately, and rename "everything, including your data folder."

### Git history: `tmp_split.onnx` + `smoke_out/` stripped, ~690 MB removed

A single commit (`ff8cdcd`, the pre-merge commit that swept up long-uncommitted local work — see above) had accidentally committed `tmp_split.onnx` (177 MB) and twelve `smoke_out/**/*.wav` files (~43 MB each) plus `smoke_test_tone.wav`, none of which belong in the repo. `git-filter-repo` isn't installed on Ed's machine or in its WSL2 environment, so `pip install --user git-filter-repo` first. The mounted folder (`D:\OneDrive\Git\Stem Separation`) doesn't allow file deletion by default from this session's device-bridge access — `git-filter-repo` needs to unlink both working-tree files and its own lock files mid-rewrite, so the first attempt failed partway through with `Operation not permitted` on several unlinks, including `.git/HEAD.lock`. Requested and got delete permission for that folder, then re-verified state: despite the error, the object rewrite and ref update had actually completed (`git log -1` showed the new, rewritten commit hash); only the final `git reset --hard` cleanup step had failed. Cleared the stray `HEAD.lock`, ran `git reset --hard` and removed the leftover untracked files by hand, then `git reflog expire --all --expire=now --expire-unreachable=now` + `git gc --prune=now` to actually reclaim the space.

Before any of this, took a full raw copy of `.git` (`cp -r .git ../AnyStemDeck-git-backup-pre-rewrite`, 592 MB, ~85s over the mount) as a safety net — `git bundle create` was tried first but reliably timed out over the mount's I/O speed, so a plain directory copy instead. **That backup directory is still sitting on disk next to the repo and should be deleted once Ed has confirmed everything looks right after pushing.**

Result, verified: `.git` shrank from 592 MB to 14 MB. `git fsck --full` clean, no dangling objects. `git log --all --oneline -- tmp_split.onnx smoke_out smoke_test_tone.wav` returns nothing — confirmed gone from every commit, not just HEAD. The `upstream` remote survived (git-filter-repo only strips `origin` by convention, and this repo never had one). Added `.gitignore` entries for all three paths so they can't sneak back in, committed that on its own (`c723879`).

**This means the entire pre-existing commit history was rewritten — every commit hash changed.** Fine for a repo that has never been pushed anywhere (confirmed: no `origin`, nothing to force-push over), which is exactly Ed's situation. Would need `git push --force` and would break any existing clone if this repo had already been shared — not a concern here, but worth remembering if this ever needs doing again after the first public push.

### Rename: StemDeck → AnyStemDeck, scoped as "everything"

Went considerably further than the desktop app's own product name. The full audit (`git grep` for `StemDeck`/`Stemdeck`/`stemdeck` in any case, repo-wide) turned up around 100 files: not just `desktop/src-tauri/{Cargo.toml,tauri.conf.json,src/main.rs}` and `pyproject.toml`, but the entire Python backend's logger namespace (`logging.getLogger("stemdeck.*")` used in 18 files, plus the rotating log file itself, `LOGS_DIR/stemdeck.log`), the FastAPI app's self-reported `title`/`/api/health` `name` field, an `X-Stemdeck-Token` auth header (read on both the Rust and Python sides — had to change together), packaging scripts and CI release workflows for all three platforms, the Linux `.desktop` launcher, an Unraid Community Applications template, and the SVG icon/logo asset set's filenames and accessibility metadata.

**Left untouched, deliberately:**
- `STEMDECK_*` environment variables (`STEMDECK_DATA_DIR`, `STEMDECK_JOBS_DIR`, `STEMDECK_DEMUCS_DEVICE`, and ~25 others) — these are an established internal protocol between the Rust launcher and the Python backend, read and written in ~20 files including the whole test suite. The codebase already had its own convention for this: brand-new AnyStemDeck-only settings (`ANYSTEMDECK_ONNX_GRAPH_OPT`, from the DirectML work) already used the new prefix, while everything inherited from upstream kept `STEMDECK_`. Renaming ~150 occurrences across 20 files for zero user-visible benefit, with real risk of a missed occurrence silently breaking a setting, wasn't worth it — consistent with the project's own existing pattern, not a deviation from it.
- `NOTICE`, `LICENSE`, `README.md`'s "Relationship to StemDeck"/Credits sections — required Apache-2.0 upstream attribution to the real project, confirmed untouched (`git diff` empty on all three) after the rename pass.
- The actual visual artwork in `imgs/anystemdeck-svg-assets/*.svg` and `static/imgs/anystemdeck-logo-horizontal.svg` (renamed as files; `<title>`/`<desc>` accessibility metadata updated) — the wordmark/logo itself is still drawn as "StemDeck" in vector paths, not text, so it needs real redesign, not a find-replace. Flagged to Ed, not fixed here.
- Three places that need Ed's actual GitHub org/repo path once he creates it, rather than a guess: `ca_profile.xml` (Unraid CA listing, points at `stemdeckapp/stemdeck`'s real icon/repo URLs), `templates/anystemdeck.xml` (same issue — its own `<TemplateURL>` and related links were mechanically transformed to a placeholder `anystemdeckapp/anystemdeck` that doesn't exist), and `scripts/macos/make-runtime-pack.sh`'s `RELEASE_BASE_URL` default (still points at `github.com/stemdeckapp/stemdeck/releases/...`). `CONTRIBUTING.md`/`ROADMAP.md`'s issue-tracker links got the same mechanical placeholder treatment as the Unraid files — better than pointing at the real upstream tracker, but still needs Ed's real URL swapped in once the repo exists.

**One real bug caught by verification, not planning**: the mechanical `s/StemDeck/AnyStemDeck/g` pass was run blind across every file in one batch, without checking first whether any of them already contained the string "AnyStemDeck" — `docs/plan.md` (this file) did, extensively, since it's already named "AnyStemDeck" throughout its own history. Text already reading "AnyStemDeck" got the embedded "StemDeck" substring matched too, producing "AnyAnyStemDeck" in twelve files. Caught by re-grepping for the corrupted string after the batch, fixed everywhere with a second pass (`AnyAnyStemDeck` → `AnyStemDeck`). Worse, and specific to this file: this document is a historical narrative that legitimately uses "StemDeck" to mean the *original upstream app* throughout (e.g. "turning StemDeck into AnyStemDeck", "what StemDeck actually does today" describing the pre-fork baseline) — collapsing that distinction would have actively corrupted its meaning, not just doubled a word. Reverted this file's rename entirely (`git checkout -- docs/plan.md`) rather than trying to hand-fix it, and audited every other prose document (`README.md` was already excluded; `docs/models.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `SECURITY.md`, all three platforms' packaging READMEs) for the same upstream-vs-fork narrative pattern — none of the others had it; they were all already self-referential ("StemDeck is licensed under...", "StemDeck for macOS...") describing this app's own old name, so renaming those was correct.

Also caught by verification: renaming file *contents* that referenced other files by name doesn't rename the files themselves. `packaging/linux/install.sh`, `scripts/linux/make-portable.sh`, and `tests/linux/test_install_sh.sh` all got their literal string `"stemdeck.desktop.in"` mechanically rewritten to `"anystemdeck.desktop.in"` — but the actual file on disk was still `packaging/linux/stemdeck.desktop.in` until caught by grepping for filename-shaped references specifically and `git mv`-ing it (and the seven SVG asset files, and `templates/stemdeck.xml`) to match.

**Verified, not just applied**: staged `desktop/src-tauri` into a build sandbox (this container, since Ed's machine's WSL2 environment has no Rust toolchain) and ran `cargo check` (clean), `cargo clippy` (clean, no warnings), and `cargo test` (55/56 passed — the one failure, `a_read_only_app_root_is_rejected`, sets a directory to `0o555` and expects a write to fail; running as root in this sandbox, the kernel ignores that permission bit, so the test's own premise doesn't hold here — unrelated to the rename, would need re-running as a non-root user to actually verify, not expected to fail on Ed's machine). The `documents_anystemdeck_dir`/`legacy_migration_*` tests — which exercise real migration logic, not just naming — passed with the renamed destination path, confirming the rename didn't disturb that mechanism. A full repo-wide re-grep after all fixes turned up nothing left except the three URL placeholders above and the intentionally-preserved `STEMDECK_` env vars and NOTICE/LICENSE/README.

**Not yet done, needs Ed**: `pip install -e .` (or however the WSL2 environment's editable install was originally set up) should be re-run once these changes reach the machine — `app/main.py`'s version lookup calls `importlib.metadata.version("anystemdeck")`, matching `pyproject.toml`'s renamed `name`, and won't find anything until the package is reinstalled under that name (it degrades gracefully to a fallback version string in the meantime, doesn't crash). Also needs a real Windows/WSL2 build-and-run to confirm the Tauri side still launches correctly post-rename — this session's verification covers "compiles, lints, and unit-tests clean," not "launches for real," the same caveat noted for the WSL2 launch-mechanism work above.

Data-folder migration for Ed's own existing library (`%LocalAppData%\StemDeck` → `%LocalAppData%\AnyStemDeck`, `Documents\StemDeck` → `Documents\AnyStemDeck`) is a manual step, not code — covered directly with Ed rather than here.

### Follow-up: real GitHub URL wired in, wordmark redesign added to backlog

Ed's real repo is `https://github.com/edperch/anystemdeck`. Swapped the `anystemdeckapp/anystemdeck` placeholder (and, separately, the still-upstream-pointed `scripts/macos/make-runtime-pack.sh` `RELEASE_BASE_URL` default) for it everywhere it appeared: `ca_profile.xml`, `templates/anystemdeck.xml`, `.github/ISSUE_TEMPLATE/config.yml`, `.github/workflows/docker-publish.yml`'s comment, `CONTRIBUTING.md`, `ROADMAP.md`, `static/index.html`, `static/js/catalog.js`, `static/js/notifications.js`, `tests/js/report-url.test.mjs`, `tests/test_redact.py`. `NOTICE`/`README.md`'s references to `stemdeckapp/stemdeck` are correctly untouched — those name the real upstream project, not this fork.

**Backlog addition**: the SVG wordmark/logo artwork (`imgs/anystemdeck-svg-assets/anystemdeck-wordmark.svg`, `anystemdeck-logo-horizontal.svg`, `anystemdeck-logo-stacked.svg`, and `static/imgs/anystemdeck-logo-horizontal.svg`) still visually renders "StemDeck" — it's vector-drawn letterforms, not text, so the rename pass could only fix the filenames and accessibility metadata, not the artwork itself. Needs a real redesign pass before release; not blocking the initial push.

### `ROADMAP.md` was carrying StemDeck's own changelog as AnyStemDeck's — rewritten

Caught this reviewing the backlog with Ed: the earlier rename pass had cosmetically relabeled `ROADMAP.md`'s content (StemDeck → AnyStemDeck throughout) without noticing the content itself was StemDeck's real, historical "Shipped" version list — v0.1.0 "it exists" through v0.14.x, "the DAW," "the health report," all linking to `github.com/edperch/anystemdeck/issues/441` and neighbors. Those are StemDeck's real issue numbers; nothing like them exists in AnyStemDeck's own repo (which at this point has exactly one merged PR, the Dependabot bump). Reading it, there was no way to tell this was upstream's history rather than this fork's own — the same class of problem flagged and fixed for `docs/plan.md` itself during the original rename pass, just missed for this file at the time.

Rewritten to describe only what AnyStemDeck itself has actually done: the WSL2+ROCm GPU work (the fork's whole reason to exist) and the git-history/rename/CI housekeeping, both dated against real tags/commits rather than invented ones. The fabricated issue-linked "In flight"/"Next" sections are replaced with this fork's actual current backlog (guided-setup flow, Settings UI DirectML option, the packaging conflict, wordmark redesign, DirectML parity testing) with no invented issue numbers, since none exist yet. "How releases work" keeps the true mechanical description (git-tag-derived version, pre-release promotion gate, the CUDA-wheel-at-first-run vs. bundling tradeoff) since that's inherited code that will genuinely govern this fork's own releases too, but drops the specific `#318`/`#320` incident citation — that was StemDeck's own past incident, not something that has happened in this repo.

### Decision #6, made concrete: the guided WSL2/ROCm setup script

Decision #6 above settled the *shape* of this early on ("semi-automated: detect, offer, never provision silently") but not the concrete mechanism. With the manual path now fully proven (this document's own trail, and `README.md`'s Setup section written from it), there's enough real information to commit to a design rather than leave it open. Decided:

12. **A standalone PowerShell script, not a Rust-native reimplementation or a script run from inside WSL2.** `scripts/windows/setup-wsl2-rocm.ps1`, matching the existing convention of `scripts/windows/make-portable.ps1`. Rejected: reimplementing the install sequence in Rust (no real benefit — it would still just be shelling out to `wsl.exe`/`amdgpu-install`/`pip`, and this session already spent real time on `wsl.exe` interop quirks from Rust that a native Windows scripting tool doesn't have to deal with) and a script that runs from inside WSL2 (can't bootstrap WSL2's own installation, which is the first and most fragile step — `wsl --install -d Ubuntu-24.04` has to run from the Windows side).
13. **Detection reuses the two real verification commands this session actually used, not a bespoke health check.** `wsl -d <distro> -- rocminfo` (checked for a real GPU agent, not just a zero exit code — `hsa_init Failed` is a distinct, meaningful failure signal from ROCm simply not being installed) for the ROCm layer, and `wsl -d <distro> -- python3 -c "import torch; torch.zeros(1).cuda()"` for the PyTorch layer — deliberately not `torch.cuda.is_available()`, which this session learned firsthand can swallow the real error and just return `False`. If both succeed, the script's job is done before it starts: report "already set up" and let the caller move straight to flipping `wsl2BackendEnabled`.
14. **v1 automates the happy path only; it does not try to repair every failure mode it might hit.** Even this session's own hand-run of the sequence hit several genuinely distinct failure modes — the wrong ROCm install usecase, a PyTorch/ROCm version mismatch, a missing `triton` wheel, `wsl.exe` mangling backslashes in forwarded paths. (The last two are already fixed in `main.rs` itself, not something the setup script needs to work around.) Building automatic recovery for all of the install-time ones now would be guessing at failure handling nobody has hit yet through this script specifically. Instead: the script runs the known-good sequence (`wsl --install -d Ubuntu-24.04` → reboot checkpoint → `amdgpu-install -y --usecase=wsl,rocm --no-dkms` → the ROCm-matched torch/torchaudio/triton wheels from `repo.radeon.com` → `apt install ffmpeg` → `pip install --user --break-system-packages -e .` → write `wsl2BackendEnabled`/`wsl2Distro`), and on any step's failure, stops, shows that step's actual output, and points at `README.md`'s manual Setup section rather than attempting to branch or auto-repair. Matches Decision #6's "never silently provision" spirit, extended to "never silently guess at a fix" either. Revisit specific auto-repairs only once real users are actually hitting the same failure repeatedly — not before there's real signal on which ones matter.
15. **Surfaced from Settings, not folded into the existing first-run wizard.** The current onboarding wizard (`static/js/setup.js`) already works correctly for CUDA/MPS/CPU/DirectML users and assumes a bundled native-Windows runtime; routing every user through an AMD/WSL2 branch there would add a fork in a flow that doesn't need one for most people. Instead: a new "Enable AMD GPU (WSL2 + ROCm)" action in Settings, next to where the DirectML device option lands once Phase 3 is built — opt-in, for the specific hardware situation it actually applies to.
16. **Exact package versions are pinned to what `README.md`'s Setup section documents as working today, not discovered dynamically at install time.** AMD's ROCm-matched wheel filenames on `repo.radeon.com` include a git-hash suffix that isn't predictable or safely scrapable, so the script hardcodes the same versions the README does. This is a real, recurring maintenance cost — the pin will need a manual bump whenever AMD ships a new ROCm release the way this session had to work through ROCm 7.2.1 by hand — not a one-time decision. Worth a line in `CONTRIBUTING.md` once the script exists, so this doesn't get rediscovered from scratch next time it goes stale.

Not yet implemented — this is the design, not the code. Actual implementation is its own next task.

### Wordmark/logo redesign done — and a correction: it was never vector-drawn letterforms

The backlog note above ("real GitHub URL wired in, wordmark redesign added to backlog")
said the SVG wordmark/logo artwork was "vector-drawn letterforms, not text" — checked
more carefully this session, and that's wrong. All three files (`anystemdeck-wordmark.svg`,
`anystemdeck-logo-horizontal.svg`, `anystemdeck-logo-stacked.svg`) render "StemDeck" as
literal SVG `<text>`/`<tspan>` elements, `font-family="Inter, Geist, Manrope, Arial, sans-serif"`
— only `anystemdeck-icon.svg` and the tray/waveform-symbol files are actual vector shapes
(rounded-rect bars, no text at all). The earlier claim had been repeated uncorrected into
`ROADMAP.md`; fixed there alongside this entry. Worth remembering for next time: check a
file's actual contents before describing what would be needed to change it, not just its
rendered appearance.

Turned out to make the redesign simpler than planned — a text and layout change, not a
redraw. Checked `static/css/variables.css` first to see what the app's real UI font is,
since matching it (rather than picking something new) was the more defensible choice for
a wordmark representing the app: `--font-sans: 'Inter', -apple-system, system-ui, ...` —
Inter is already first in the wordmark's existing font stack (`Inter, Geist, Manrope,
Arial, sans-serif`), so it needed no revisiting.

Two decisions run past Ed before starting (he'd asked "do you have any questions before
you begin"), both picked from his answers:
- **All three brand files redone**, not just the standalone wordmark, so the wordmark,
  the icon+text horizontal lockup, and the icon+text stacked lockup stay in sync.
- **Color split: "Any" in gold (`#F2B53D`), "StemDeck" in the existing near-white
  (`#F4F6F8`)** — puts the visual emphasis on what actually distinguishes this fork (any
  GPU vendor) while keeping "StemDeck" itself as one visually unified block, closest in
  spirit to the original two-tone split.

Measured actual glyph widths against the real Inter Bold/Medium fonts (via `fontTools`,
not eyeballed) to place "Any"/"StemDeck" precisely and size each canvas to fit the three
extra letters without shrinking the type: wordmark 820→728px wide, horizontal lockup
1200→1230px, stacked lockup 720→780px (heights unchanged — vertical layout wasn't
affected by the longer word). Also dropped a stray space in the stacked lockup's original
markup (`"Stem "` + `"Deck"`, rendering as two words with a gap) that the wordmark and
horizontal files never had — "StemDeck"/"AnyStemDeck" is one word everywhere now,
consistent across all three. Rendered all three to PNG with the real Inter font installed
before committing, to check spacing and legibility rather than trusting the markup blind.
None of the three files are referenced anywhere in the app's code or `README.md` yet
(confirmed by grep) — still unwired brand assets, so this was a freer redesign than
editing a file something else depends on.

Also updated: `ROADMAP.md`'s wordmark/logo redesign backlog item, moved from
"In flight / next" to "Shipped" now that it's done.
### Settings UI: DirectML device option, and the availability gap it surfaced

Phase 3 of the DirectML work, finally wired into the UI Ed actually sees.
Added a "DirectML (AMD/Intel/NVIDIA)" `<option>` to the compute device select
in `static/js/catalog.js` -- the existing greying-out logic (keyed off a
`demucs_devices_available` list from the server) picked it up with no new
client logic needed. Also greyed out "Best" separation quality specifically
when the *resolved* device is `dml` (not the raw select value -- "auto" can
resolve to `dml` too), reusing the same "not available" suffix pattern the
device select already used. `app/pipeline/separate.py` already caps shifts
back to 1 for `dml` at job time, with a comment saying "the Settings UI is
meant to grey this combination out" -- this is that.

While wiring the device select, found and fixed a real, live gap in
`app/main.py`: `demucs_devices_available` (the field the greying-out logic
reads) was built from `available_torch_devices()` alone, so "dml" could never
appear there even on a machine where `available_onnx_providers()` genuinely
reports it -- DirectML would have been unselectable forever, on any hardware,
once the UI shipped. Fixed by including both. Added a regression test
(`tests/test_network_gate.py`) pinning it. Also noticed the existing
cuda/mps/cpu `<option>` tags had no `data-i18n` attribute, despite
`static/js/i18n.js` carrying translated strings for all three in every
locale the whole time -- they were just never applied. Wired those up too
(and added the new `settings.device.dml` key to all 9 locale blocks) while
already in that exact markup.

Verified against a clean clone with the patch applied: full suite (839/839,
including the new test), `ruff check`/`format --check` clean on every file
touched, both edited JS files pass `node --check`, and the JS i18n-detect
test still passes (42/42).

### Packaging: the onnxruntime/onnxruntime-directml conflict, and the Python floor

The other two of the three items Ed asked to tackle together. Both land in
`scripts/windows/make-portable.ps1` and `pyproject.toml`.

**The onnxruntime conflict** was already fully diagnosed in a comment above
`demucs-onnx` in `pyproject.toml` (from the original DirectML work) --
`onnxruntime` and `onnxruntime-directml` are separate PyPI distributions that
both install into the exact same `onnxruntime` import path, so declaring both
as dependencies would leave pip to silently pick whichever installed last.
That comment named the fix and where it belonged: "a forced post-install
override in scripts/windows/make-portable.ps1, not a pyproject.toml
dependency." Implemented exactly that: `pip install onnxruntime-directml==X
--force-reinstall --no-deps`, run unconditionally for every Windows package
(both `-CpuOnly` and the default NVIDIA build) since DirectML is a
torch-independent path, not part of that variant split. `--no-deps` because
onnxruntime-directml needs the same numpy/flatbuffers/etc. the plain
onnxruntime install (and uv.lock) already pinned; letting it reinstall those
too would risk a different resolution winning against those pins for
nothing.

The version pin needed its own decision: uv.lock resolves plain `onnxruntime`
to 1.29.0, but `onnxruntime-directml`'s own latest release on PyPI is 1.24.4
(checked directly, not assumed) -- the DirectML build is a slower-moving,
separate distribution and the two are not expected to track each other.
Pinned the override to onnxruntime-directml's own latest (1.24.4) rather than
trying to force version parity with plain onnxruntime, which isn't available
regardless. This is a real, recurring maintenance cost, the same shape as the
ROCm wheel pins already called out for the setup script (Decision #16) --
noted in the script's own comment so it isn't rediscovered from scratch.

**The Python floor** (`requires-python`): raised from `>=3.10` to `>=3.11`.
`demucs-onnx` already needed 3.11 as a hard, unconditional dependency (gated
by platform, not Python version), so `>=3.10` was a statement of intent that
`uv lock`/`pip install` would simply refuse to honor -- not a real, working
option today. Checked whether 3.10 support was worth preserving via the
alternative (gating `demucs-onnx` behind a Python-version marker) before
deciding: nothing else in the project exercises 3.10 either. CI only ever
runs Python 3.12 (`.github/workflows/ci.yml`'s container image),
`CONTRIBUTING.md` tells developers to use 3.12, and this same packaging
script hardcodes `py -3.12`. Gating `demucs-onnx` instead would have added a
second, permanently-untested code path (Python 3.10, `dml` never offered) to
preserve support nothing else here already provides -- raising the floor
outright was the lower-risk choice.

`uv lock` regenerated after the floor change (confirmed deterministic: ran it
independently in two different environments -- once directly on the floor
change, once again against a fresh clone in a verification sandbox -- and
got byte-identical package removals both times). It dropped three
now-unnecessary backport packages that only existed for pre-3.11 stdlib gaps:
`backports-asyncio-runner`, `exceptiongroup`, and `tomli` (`tomllib` is
stdlib from 3.11 on) -- a small, concrete confirmation the floor bump was
real, not just a version-string change.

**Found and fixed a second, unrelated bug while in this file**: the
`-CpuOnly` branch's forced torch reinstall was still pinned to `torch==2.6.0+cpu
torchaudio==2.6.0+cpu` -- stale since `pyproject.toml`'s own torch floor
moved to `>=2.9,<2.10` during the ROCm work earlier in this project's life,
and nobody had updated this script's independent pin to match. Building the
CPU-only Windows package today would have force-installed a torch version
below the project's own declared floor. Repinned to `torch==2.9.1+cpu` (uv.lock's
resolved version) and left a comment explaining *why* this has to be a
separate hand-maintained pin at all -- the CPU wheel lives on a different
index (`download.pytorch.org/whl/cpu`) than the one `pip install "$Root"`
resolves against, so it can't just inherit the project's own constraint.

Both verified against a clean clone with the diff applied (not just eyeballed
on Ed's machine): `uv lock` reproduced the same removals independently,
`uv sync --frozen --all-extras` resolved cleanly on the new 3.11 floor, the
full suite passed (838/838), `ruff check`/`format --check` stayed clean, and
the whole `make-portable.ps1` script still parses as valid PowerShell
(checked with a real `pwsh` parser, not just read by eye -- installed
PowerShell 7.4.6 fresh for this, since nothing here normally needs it). Not
verified: an actual Windows build run, since nothing in this environment can
do that -- worth a real smoke test next time a Windows package is built for
real, particularly the CPU torch wheel's exact filename on
`download.pytorch.org` (unreachable from this sandbox to check directly).

### Two real-machine bug reports from the first packaged AMD build, and a correctness bug they surfaced in the process

Ed ran the just-fixed `make-portable.ps1` end to end on real Windows hardware for
the first time and reported two things from the running app: the first-run wizard
still said "No NVIDIA GPU - stem separation will use CPU" (misleading on a fork
whose whole point is non-NVIDIA hardware), and the topbar still showed the old
two-tone "Stem"/"Deck" split instead of "AnyStemDeck".

**The topbar was a literal miss from the original rename pass.** `static/index.html`
had `<span class="fg">Stem</span><span class="accent">Deck</span>` -- the three
standalone SVG assets fixed earlier this session are confirmed unreferenced
anywhere in the app, so this markup is the actual in-app brand element and was
simply never touched. Fixed to `<span class="accent">Any</span><span
class="fg">StemDeck</span>`, reusing the existing `.fg`/`.accent` CSS classes (no
CSS changes needed) and matching the "Any" gold / "StemDeck" white split the
wordmark redesign already established -- just via the app's own `--accent`/`--fg`
variables rather than the SVGs' specific hex values, which is the more
maintainable choice for something theme-driven.

**The onboarding message led to a bigger finding.** `desktop/ui/setup.js`'s "no
GPU" branch is genuinely NVIDIA-specific -- `ensure_torch_device` (`main.rs`) only
ever probes `nvidia-smi`, non-mac -- so on this AMD-fork the message was always
going to fire for the exact hardware the fork exists for. The literal fix is a
one-line wording change (below). But before making it, checked what the message's
"will use CPU" claim actually resolves to at job time, since `detect_compute_device()`
(`auto`'s hardware probe) ranks `cuda > mps > dml > cpu` -- and `dml` (DirectML) was
just wired into both the Settings UI (Phase 3, this session) and, critically, into
*every* Windows package's Python install (the packaging fix, also this session:
`make-portable.ps1` now force-installs `onnxruntime-directml` unconditionally, not
just for one build variant). That combination means an AMD/Intel-on-Windows user
on the default "auto" setting would no longer get CPU when no NVIDIA GPU is found --
they'd get `dml`, automatically, the moment `onnxruntime-directml` is present.

That would be fine if `dml` worked. It doesn't, and this repo's own history already
proved it, at length: Phase 1/1.5 above found `demucs-onnx` under DirectML
reproducibly crashes on a `ConvTranspose` node (confirmed on a real RX 7800 XT),
and a from-scratch split-graph export written specifically to route around that
crash then hit a second, worse bug -- `InstanceNormalization` silently returning
numerically wrong output, amplified downstream through the very next ops. That's
exactly why Phase 1.6 above is titled "pivot: DirectML parked, investigating
WSL2 + ROCm instead", and why `ROADMAP.md`'s own Shipped entry for v0.1.0 already
says DirectML was "shelved after DirectML's `InstanceNormalization` operator
produced numerically wrong output on real hardware." The shipped
`app/pipeline/demucs_onnx_worker.py` calls `demucs_onnx.separate()` directly --
the original, un-split graph -- so a job actually dispatched to `dml` in
production is expected to hit the Phase 1 crash, not silently corrupt audio; still
a broken, user-visible failure, just not the more dangerous silent-corruption
outcome initially worried about while tracing this through. Either way, "auto"
routing AMD/Intel-on-Windows users into a known-broken device by default,
the moment packaging installs `onnxruntime-directml` everywhere, is a real
regression this session's own earlier two commits combined to create without
anyone connecting them at the time.

**Fix**: `detect_compute_device()` no longer falls through to `dml` -- auto
resolution is now `cuda > mps > cpu` only. `dml` remains fully selectable by
hand in Settings (`set_demucs_device()` unchanged, still validates it against
`available_onnx_providers()`) for anyone who wants to try it anyway, or once the
upstream ConvTranspose/InstanceNormalization bugs are fixed -- it's excluded from
*auto* specifically, not removed. Updated the docstrings on
`detect_compute_device()` and `get_demucs_device()` to record why, so this isn't
rediscovered from scratch later. Added
`test_detect_compute_device_never_auto_selects_dml` (`tests/test_config.py`),
which fails the old way if this regresses: it stubs a working `dml` provider
and asserts `auto` still resolves to `cpu`.

With that fixed, the onboarding message is now actually accurate as a literal
wording change: `desktop/ui/setup.js`'s no-GPU branch now reads "No compatible
GPU found - stem separation will use CPU" -- true again, since `auto` really
does mean CPU here now. (An earlier pass at this fix went further -- probing
DirectML availability in `ensure_torch_device`/`main.rs` and reporting "DirectML
acceleration available" when found -- and was reverted once the auto-routing bug
above came to light: that message would have been actively wrong, promising
working acceleration on a path known not to work.)

**Verified**: full pytest suite (840/840, including the new test) green on a
clean clone with the patch applied; `ruff check`/`format --check` clean on every
Python file touched; `node --check` clean on `setup.js`. The Rust change (the
reverted `ensure_torch_device`/`GpuSetup` DirectML-probe attempt) was also
independently compile-checked, `clippy -D warnings`-clean, and `cargo test`-clean
before being reverted, for what that's worth -- it wasn't a syntax problem, it was
a correctness-of-claim problem once the auto-routing bug was found.

**Not yet checked**: whether Ed's actual first real separation job (the one that
produced the 6-stem result confirming the packaging fix works) ran on `dml` or
`cpu` -- the packaged build's own `data/logs/stemdeck.log` from that run wasn't
found in the connected folder (the `dist/` there now looks like a fresh,
not-yet-run rebuild). Worth Ed checking that job's log line
(`stemdeck.pipeline [...] separating on device=...`) if he still has access to
it, mostly out of curiosity now that auto no longer routes through `dml` for
future jobs -- if that job did use `dml`, per the analysis above it likely means
that job actually failed/errored rather than silently corrupting output, which
doesn't match "6 stems produced and played back fine," so `cpu` is the more
consistent explanation, but it's worth confirming rather than assuming.

### Settings toggle for the WSL2/ROCm backend, replacing the hand-edit-the-store-file workaround

Ed asked to build a real "Enable AMD GPU (WSL2 + ROCm)" Settings control rather than
keep hand-editing `user-data.json` to set `wsl2BackendEnabled`/`wsl2Distro` (the
workaround this doc's own manual smoke test above used, since Decision #6's guided
setup flow -- which was meant to own this -- is still unbuilt).

**Scope, deliberately narrow.** This adds the toggle and a distro field, nothing
else: no ROCm/WSL2 installation, no validation that the distro named actually has
ROCm+PyTorch working, no first-run wizard integration. It's the same "no silent
guessing" posture Decision #6 itself already commits to for the guided-setup
script -- a raw switch to what already works at the Rust level
(`wsl2_backend_enabled()`, `start_backend`'s WSL2 branch, all shipped and
real-hardware-verified earlier in this doc), described honestly as needing WSL2
and ROCm already set up by hand (README's existing AMD GPU section) rather than
implying this button does that setup.

**New Settings section** (`static/js/catalog.js`, `openLibraryEditor`'s template,
right after the compute-device/quality sections): a toggle ("Enable AMD GPU
(WSL2 + ROCm)") and an optional "WSL distro" text field, in a
`.settings-wsl2-section` that starts `hidden` in the markup and is only revealed
by `wireWsl2Setting()` once two things are confirmed -- desktop (`window.__TAURI__`
present) and Windows (`getBuildTarget().os === "windows"`) -- so it never flashes
on a platform (macOS, Linux, server/browser mode) where WSL2 doesn't exist or the
Tauri store isn't reachable.

**Different plumbing from every other control in this modal, deliberately.** The
compute-device select and everything else here round-trips through
`/api/settings` (Python, live, applies to the next job). `wsl2BackendEnabled` is
read once by Rust's `start_backend`, before the Python backend has even started
-- there is no live equivalent to push a change into. So this goes through the
Tauri store (`storeGet`/`storeSet` from `utils.js`) instead, the same path the
language picker already uses for exactly this reason (see that function's own
comment). A change here only takes effect on the next launch; the description
text says so rather than the UI pretending otherwise.

**`wsl2Distro` semantics matched exactly to what `main.rs`'s own `wsl2_distro()`
already does with it** (`.filter(|s| !s.trim().is_empty())` -- empty/whitespace
means "let wsl.exe use its own default distro"): the JS side trims on blur and
stores `null` rather than `""` for an empty field, so the persisted value stays
meaningful (`null` = unset) instead of accumulating stray empty strings, though
either would have worked given the Rust-side filter.

**i18n**: six new keys (`settings.wsl2.subhead`/`enable.title`/`enable.desc`/
`distro.title`/`distro.desc`/`distro.placeholder`) added identically to all 9
locale blocks in `static/js/i18n.js`, left in English everywhere -- matching the
existing precedent for this exact kind of string (`settings.device.dml` is also
identical English text in all 9 blocks): "WSL2", "ROCm", and "AMD GPU" are
technical/brand terms, the same category CUDA/DirectML/MPS's option labels
already are.

**CSS**: one new rule, `.settings-wsl2-section.hidden { display: none; }` --
*not* the generic `.hidden` utility class, which is scoped to `.daw` and (per
the existing comment right above where this was added) doesn't reach the
settings overlay at all, since that's appended to `document.body`. Same
reasoning `.settings-net.hidden` already uses one row up.

**Verified**: `node --check` clean on every touched JS file, the full
`tests/js/*.test.mjs` suite still green (i18n-detect unaffected: it tests locale
*detection*, not per-key coverage, and no LANGUAGES/detection logic changed).
No Rust changed -- `wsl2_backend_enabled()`/`wsl2_distro()`/`start_backend`'s
WSL2 branch were already shipped and real-hardware-verified earlier in this doc;
this only adds a way to reach the store keys they already read, from the UI
instead of by hand.

**Not done, and worth being upfront about**: the toggle doesn't verify WSL2/ROCm
actually works before letting itself be switched on -- if the guided-setup
script (still unbuilt) later exists, wiring a real check in here first is
straightforward. Also not done: the ONBOARDING/first-run wizard still has zero
awareness of `wsl2BackendEnabled` (the "wizard runs unconditionally before
`start_backend`" finding from the manual smoke test above still applies) --
this toggle changes how a *restart* launches, not the setup screen a user
already mid-flow is looking at.

### Onboarding message: the "no GPU found" text was still wrong for WSL2/ROCm users

The Settings toggle above (previous entry) fixed *reaching* the WSL2/ROCm path
-- but `ensure_torch_device()` in `main.rs` had no idea it existed. It always
ran the native NVIDIA probe (`detect_nvidia_gpu`, `nvidia-smi`), regardless of
`wsl2BackendEnabled`, and that probe correctly finds nothing on an AMD machine
-- so an AMD user who had just turned the new toggle on and restarted would
*still* see "No compatible GPU found - stem separation will use CPU" on the
onboarding screen, exactly backwards from what was about to happen (this is
also the gap ROADMAP.md flagged, two entries up, as "onboarding wizard still
has zero awareness of wsl2BackendEnabled").

Root cause was structural, not a missed string: `ensure_torch_device` ran
*before* `start_backend`, had no `tauri::AppHandle` parameter, and so had no
way to call the already-existing `wsl2_backend_enabled()` helper at all.

**Fix**: `ensure_torch_device` now takes `app_handle: tauri::AppHandle` and
checks `wsl2_backend_enabled(&app_handle)` first, before touching the native
probe. When it's on, the native NVIDIA probe/install is skipped entirely --
running it would only ever produce a false "no GPU" (the AMD GPU only becomes
visible once the backend is actually started inside WSL2, which hasn't
happened yet at this point in setup) -- and `ensure_torch_device` returns a
new kind of `GpuSetup` result carrying `wsl2Enabled: true` and
`gpuName: "AMD GPU (WSL2 + ROCm)"` instead. `setup.js` checks `wsl2Enabled`
before either of the existing mac/native branches, so it can't be misread as
a failed CUDA verification (`gpuDetected`/`cudaVerified` describe the native
probe specifically and don't apply to this path) -- the onboarding screen now
says "AMD GPU (WSL2 + ROCm) enabled" instead of the misleading CPU-fallback
message.

**Staleness, handled the same way the existing `cpu-only-package` reason
already is (#316):** the decision is persisted (`torchDevice: "cuda"`,
`torchDeviceReason: "wsl2-rocm-enabled"`) so the fast-path setup gate can skip
re-probing on later launches, same as a verified native CUDA result. But
unlike a graphics card, `wsl2BackendEnabled` is a Settings toggle a user can
flip off again -- so `effective_device_reason()` gained a third parameter
(`wsl2_enabled`) and now drops a stale `"wsl2-rocm-enabled"` reason the same
way it already drops a stale `"cpu-only-package"` one, when the toggle no
longer matches. Without this, disabling the toggle and restarting would leave
the onboarding gate believing "cuda" was still settled and skip re-probing,
even though nothing about the actual running device is affected either way --
`app/core/config.py`'s own `detect_compute_device()` does a fully independent
live probe every job and never reads this Rust-side cache at all, so this
staleness was only ever a display/gate-freshness concern, not a functional
one.

**Verified**: `cargo check`/`cargo test`/`cargo clippy -- -D warnings`/
`cargo fmt --check` all clean against a fresh clone (two new tests --
`stale_wsl2_reason_dropped_when_toggle_is_off`,
`wsl2_reason_kept_while_toggle_is_still_on` -- plus the three existing
`effective_device_reason` tests updated for the new parameter); `node --check`
on `setup.js`; full `tests/js/*.test.mjs` suite green.

**Still open**: part two of the same question -- making GPU/acceleration
status visible somewhere in the *running* app, not just transiently during
onboarding -- is a separate, larger UI decision (placement, live source of
truth) tracked separately below / in ROADMAP.md rather than folded into this
fix.

### Persistent in-app GPU/acceleration status: a dot on Settings + a line in the notification panel

Part two of Ed's question, held back deliberately from the onboarding-message
fix above until placement was settled: onboarding's message is transient (the
window navigates away to the main app right after), and the main app itself
showed device status nowhere at all. Asked Ed where it should live, given
real trade-offs -- a topbar chip (always visible, but this app deliberately
keeps its topbar minimal today, e.g. the version number is hidden there on
purpose), the notification panel alone (matches that minimal-topbar instinct,
but needs a click), or something on the Settings rail button itself (low
visual weight, sits where the user would actually go to check or change it).
Ed picked a hybrid: a status dot on the Settings rail button, plus the fuller
device name in the notification panel.

**Implementation**: a single `refreshGpuStatus()` in the new
`static/js/ui-chrome.js` addition reads the same live source of truth the
Settings device dropdown itself uses -- `GET /api/settings`'s
`demucs_device_resolved` -- so this can never go stale relative to what the
next job will actually do, including a live device change made in Settings
with no restart. On desktop, that field alone can't distinguish a real
NVIDIA card from AMD-via-WSL2+ROCm (both report `"cuda"`, since ROCm
presents to PyTorch as plain `torch.cuda`) -- disambiguating reuses the same
`wsl2BackendEnabled` Tauri-store flag `wireWsl2Setting` (catalog.js) already
reads, gated behind `getBuildTarget().os === "windows"` the same way that
function gates its own Windows-only section.

Two surfaces, one function:
- **Settings rail button** (`#settingsBtn`, now `.rail-settings` for
  positioning): a small accent-colored dot (`#gpuStatusDot`, styled like the
  existing `.rail-badge` pattern) that only appears when the resolved device
  is not `"cpu"` -- deliberately absence-means-nothing rather than a visible
  "CPU warning" dot, so it can't read as an error state for the common case.
  The button's `title` always carries the full device text on hover either
  way, dot or no dot.
- **Notification panel** (`#notifDevice`, between the header and the
  scrollable list): a persistent, non-dismissible status strip -- kept
  deliberately outside `notifList`/`notifEmpty`'s failure+release
  bookkeeping (`notifications.js`'s `render()`) since it isn't a
  notification that gets cleared, just always-current status. Placed as a
  header-level strip rather than a list card specifically to avoid
  interacting with `render()`'s anchor-based card-insertion logic.

Refreshed on load, every time the notification panel is opened (cheap fetch,
no polling), and on a language change (three new i18n keys --
`gpu.status.title`, `gpu.status.accelerated` with a `{device}` placeholder,
`gpu.status.cpu` -- added to all 9 non-`pt-PT` locale blocks; the device
names themselves stay English, matching this project's existing precedent
for WSL2/ROCm/DirectML). No new accent color -- both surfaces reuse the
app's existing gold `--accent`, the same tint `.daw-notif-release` already
uses, rather than introducing a separate "success" green the app's palette
doesn't otherwise have.

Frontend-only change (no Rust/Python touched) -- verified with `node --check`
on the two edited JS files and the full `tests/js/*.test.mjs` suite (all
green) against a fresh clone with these files layered on top.

### README: which AMD GPUs actually work on the WSL2/ROCm path

Ed asked two things: whether a built `.exe` still needs the manual WSL2/ROCm
setup (yes -- `scripts/windows/make-portable.ps1` bundles a native-Windows
Python runtime and the Tauri build only; it never touches WSL2, ROCm, or the
separate Linux-side Python environment the README's AMD section walks
through installing by hand. The desktop app can *launch and manage* an
already-set-up WSL2 backend (the `wsl2BackendEnabled` toggle from earlier
this week), but setting that backend up in the first place -- steps 1-9 of
the README's AMD section -- is unaffected by whether you run from source or
a packaged `.exe`. A guided in-app setup flow that could someday automate
this is still open, tracked in ROADMAP.md's "In flight / next"), and to add
the actual list of AMD GPUs the WSL2 path supports to the README, since
"which cards work" wasn't documented anywhere before this -- only this
project's own confirmed RX 7800 XT was mentioned.

Researched against AMD's current compatibility matrix (ROCm 7.2.1 /
`librocdxg`, the from-scratch WSL driver stack that replaced the earlier
ROCm-on-WSL preview): RDNA3 (RX 7000 / PRO W7000 series) and RDNA4 (RX 9000 /
PRO R9000 series) are covered, plus -- new in 7.2.1 -- Strix/Strix Halo
RDNA3.5 integrated GPUs (Ryzen AI Max/HX APUs), the first WSL2 support for
an integrated part. RDNA2 (RX 6000/W6000) has never been added to AMD's WSL2
matrix at any version despite working on native Linux ROCm; RDNA1 and
Polaris/Vega were never ROCm architectures at all. Checked whether the
`HSA_OVERRIDE_GFX_VERSION` trick (commonly used on native Linux to force an
unlisted-but-similar card to identify as a supported `gfx` target) rescues
an unsupported card under WSL2 specifically -- community reports (GitHub
issues on ROCm/ROCm) describe it detecting the GPU but then hanging or
failing under WSL2's driver stack, so the README says not to count on it.

Left one item out deliberately: AMD's docs matrix listed a plain "RX 9060"
(non-XT) as WSL2-supported in one fetch but the separate native-Windows
matrix only shows "RX 9060 XT" -- an unresolved discrepancy between AMD's
own pages, not something to assert either way in the README. Only RX 9060 XT
is listed.

This list is explicitly framed in the README as a snapshot (dated to ROCm
7.2.1 / late 2026), with a link to AMD's live matrix, given how fast it's
grown already (6.1.3 to 7.2.1 added roughly 15 SKUs and a whole new
integrated-GPU family) -- matching this project's existing practice of
linking to "whatever's current" for the install-guide steps rather than
pinning exact package versions in prose.

### README: an explicit "how do I get into WSL2" step, and PowerShell/WSL2 labels on every command

Ed's follow-up on the GPU-list addition: the setup steps assumed the reader
already knew how to open a WSL2 shell (several just said "inside WSL2" with
no instruction for actually getting there), and didn't say which environment
-- Windows PowerShell or a WSL2 shell -- each command belonged in.

Inserted a new step 2, **Open a WSL2 shell** (`wsl -d Ubuntu-24.04` from
PowerShell, or launching "Ubuntu 24.04" from the Start menu; also covers the
one-time UNIX username/password prompt on first use), and labeled every
step from there on with **[PowerShell]**, **[WSL2 -- see step 2]**, or, for
the optional final step's config-file edit, **[Windows -- any text editor]**
-- that one doesn't run in either shell at all, it's just editing a JSON
file, so labeling it PowerShell or WSL2 would have been actively wrong. Old
steps 2-9 shifted to 3-10 to make room; the two "nine manual steps" /
"steps 1-8" cross-references (ROADMAP.md, the previous README entry above)
updated to match the new count (ten steps total, nine required + the
optional desktop-management one, formerly split at eight).

### `scripts/windows/setup-build-artefacts.ps1`: OneDrive sync scare, resolved with NTFS junctions

Ed's checkout lives inside OneDrive, and hit exactly the risk `.gitignore`
can't cover: OneDrive prompted to delete/upload over 7,000 files in one
sync pass, alarming on its own, and made worse by `dist/`'s bundled Python
environment surfacing `yt-dlp/extractor/youporn.py` in the file list --
`.gitignore` only stops *git* from tracking `dist/`, `.venv/`, and the
`desktop/src-tauri/target|gen`/`node_modules` build outputs; it does nothing
for a filesystem-level sync tool, which uploads whatever's physically on
disk regardless. Confirmed nothing was actually wrong: `git ls-files dist`
returns zero tracked files (the `.gitignore` entry was always doing its job
git-side), and `youporn.py` is a completely ordinary yt-dlp extractor module
-- yt-dlp ships one such module per supported site, `pyproject.toml` lists
`yt-dlp>=2026.7.4` as a real direct dependency (used for the app's
URL-import "Composer pill" feature), and an adult-site extractor is just
alphabetically one of many, not a sign of compromise.

Asked what to do about it going forward. Ed's answer: redirect these folders
out to `D:\Build Artefacts\$project-name\<same relative path>` (e.g. this
repo's `dist/` becomes a junction pointing at
`D:\Build Artefacts\AnyStemDeck\dist`), covering "the big offenders" (`dist`,
`.venv`, `desktop/src-tauri/target`, `desktop/src-tauri/gen`,
`desktop/node_modules` -- smaller tool caches like `.ruff_cache` left alone,
nowhere near the same scale), and packaged as a reusable
`scripts/windows/setup-build-artefacts.ps1` rather than a one-off manual fix.

NTFS junctions over symlinks, deliberately: junctions need no admin rights
or Developer Mode (symlinks on Windows do, for a non-elevated user), and
OneDrive is documented to skip junctions during sync entirely rather than
attempting to traverse into them -- symlinks don't get the same treatment
reliably. The script is idempotent (an already-correct junction is left
alone on re-run), moves existing real content to the artefacts location
before linking rather than discarding it, and deliberately refuses to guess
when both the source and destination already hold real, differing content
-- same "never silently provision, never silently overwrite" instinct
already established for the WSL2 setup script design (Decision #6, above).
Pre-provisions the destination and links immediately for folders that don't
exist yet, so whatever creates them later (`cargo`, `npm`, `uv`,
`make-portable.ps1`) writes straight to the artefacts location and never
touches the synced checkout even transiently.

Verified the script's logic (not just a syntax check) before touching Ed's
real repo: installed PowerShell 7.4.6 standalone in the sandbox specifically
for this (no apt/snap package available), confirmed
`[System.Management.Automation.Language.Parser]::ParseFile()` reports clean
parses, then ran the script against a synthetic mock tree
(`dist/` with a real file, `.venv/lib`, `desktop/src-tauri/`, `desktop/`)
under both `-WhatIf` and a real run, with the Windows-only guard temporarily
stripped for the test. Surfaced one real, Linux-only PowerShell Core quirk
in the process: `New-Item -ItemType Junction` silently no-ops there --
`Move-Item` correctly relocated content, but the junction creation call
neither threw (despite `$ErrorActionPreference = "Stop"`) nor created
anything, leaving the moved content with nothing pointing back to it.
Junctions aren't a native ext4 concept, so this is expected to not occur on
real Windows -- but added a defensive `Test-IsJunction` check immediately
after every `New-Item -ItemType Junction` call regardless, throwing loudly
if it ever fails silently for any reason, on any platform, rather than
leaving content moved with no link back to it.

One real constraint surfaced while finishing this: the device bridge used
to reach Ed's actual machine is a Linux VM shell (`device_bash`), not native
Windows PowerShell or cmd -- Windows-native operations like junction
creation genuinely cannot be executed through it, and `D:\Build Artefacts`
isn't among the folders connected to this session regardless. Writing the
script's own text content to the repo is a plain file write and works fine
this way; actually running it against Ed's real folders does not, and has
to happen in a native PowerShell window on his end.

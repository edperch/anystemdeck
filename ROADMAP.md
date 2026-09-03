# AnyStemDeck roadmap

Where AnyStemDeck itself has been, and what's next for this fork specifically.
AnyStemDeck is a fork of [StemDeck](https://github.com/stemdeckapp/stemdeck) —
for StemDeck's own version history (the desktop app, player, and pipeline this
fork is built on), see StemDeck's own repository directly. Everything below is
specific to what this fork itself has done; it used to duplicate StemDeck's
changelog under this project's name, which was wrong and has been corrected.

Day to day build notes, including the full trail behind everything below, live
in [`docs/plan.md`](docs/plan.md) — this file is the short version.

---

## Shipped

### v0.1.0 · 2026-09-02 · it runs on an AMD GPU

The fork's actual point, confirmed end-to-end on real hardware: an RX 7800 XT
running the full separation pipeline — vocals, drums, bass, guitar, piano,
other — through WSL2 + ROCm, GPU-accelerated, producing correct, playable
stems. Two approaches were tried before this one landed:

- **DirectML/ONNX** (parked). A `demucs-onnx` export run through ONNX
  Runtime's DirectML execution provider, which would have covered AMD, Intel,
  and NVIDIA alike via DirectX 12 instead of a vendor-specific driver stack.
  Shelved after DirectML's `InstanceNormalization` operator produced
  numerically wrong output on real hardware — a bug outside this project's
  control, not a design mistake. The code is still in the repo
  (`app/pipeline/demucs_onnx_worker.py`, the `dml` device option) in case
  that bug is ever fixed upstream.
- **WSL2 + ROCm** (shipped). Runs the existing, unmodified PyTorch backend
  inside WSL2, where ROCm presents itself to PyTorch as an ordinary
  `torch.cuda` device — no model changes, no new inference engine, same
  worker StemDeck already uses for NVIDIA GPUs. Getting there surfaced real
  bugs along the way: a WSL2-specific ROCm install usecase (the plain Linux
  one fails under WSL2's `/dev/dxg` passthrough), a `wsl.exe`
  argument-mangling quirk, a launch-mechanism redesign after learning a
  one-shot detached `wsl.exe` invocation can't outlive itself, and a
  PyTorch/ROCm version-pairing gap that needed AMD's own matched wheels
  rather than the public PyPI ones. Full trail in `docs/plan.md`.

### Housekeeping · 2026-09 · ready to be public

Git history rewritten to strip ~690 MB of accidentally-committed model and
test-output files before the first push. A full rename pass
(StemDeck → AnyStemDeck) across roughly 130 files — code, packaging scripts,
CI workflows, the desktop app's own data folder — deliberately stopping short
of the `STEMDECK_*` environment variables (an established internal protocol,
not user-facing) and the Apache-2.0 attribution in `NOTICE`/`LICENSE`/
`README.md`, which correctly still name the real upstream project.
`README.md` gained an actual step-by-step Setup section for both the AMD/WSL2
path and the standard NVIDIA/Apple/CPU path, in place of only describing the
approach conceptually. `uv.lock` regenerated after it drifted out of sync
with `pyproject.toml`'s torch version bump — it was silently making CI test
the wrong PyTorch build, and very likely breaking the browser-test job's
startup outright by triggering a multi-gigabyte re-download mid-run.

### Wordmark refresh · 2026-09 · the logos actually say "AnyStemDeck" now

The three brand SVGs (`anystemdeck-wordmark.svg`, `anystemdeck-logo-horizontal.svg`,
`anystemdeck-logo-stacked.svg`, plus the `static/imgs/` duplicate) were renamed
during the fork's file-level rename pass but still visually rendered "StemDeck" —
the rename pass fixed filenames and accessibility metadata (`<title>`/`<desc>`), not
the text itself. For the record, correcting something this file and `docs/plan.md`
both said earlier: the wordmark was never vector-drawn letterforms — it's literal SVG
`<text>`, set in Inter (the app's own UI font, via `--font-sans`), which is why this
was a text and layout change, not a redraw. "Any" reads in the same gold (`#F2B53D`)
the brand already used for emphasis; "StemDeck" stays the near-white (`#F4F6F8`) the
rest of the wordmark always was. Canvases grew slightly (728/1230/780px wide, up from
820/1200/720) to fit the three extra letters without shrinking the type.

### Settings UI · 2026-09 · DirectML is a real, selectable device now

Added a "DirectML (AMD/Intel/NVIDIA)" option to the compute device dropdown,
greyed out the same way CUDA/MPS already were when unavailable, and greyed out
"Best" separation quality specifically when DirectML is the resolved device
(no shift-averaging equivalent on that path yet — `app/pipeline/separate.py`
already enforced this server-side and said as much in a comment). Also fixed
a real gap this surfaced: the API endpoint the UI reads device availability
from only ever reported PyTorch devices, so DirectML could never have shown
up as available even on a machine where it genuinely works.

**Correction, same week**: this option combined with the packaging fix below to
silently route AMD/Intel-on-Windows users into a known-broken device by
default. Fixed — see the entry after Packaging.

### Packaging · 2026-09 · the onnxruntime/onnxruntime-directml conflict is resolved

`scripts/windows/make-portable.ps1` now force-reinstalls `onnxruntime-directml`
after the main dependency install, for every Windows package (not just one
variant) — plain `onnxruntime` and `onnxruntime-directml` are separate PyPI
distributions that install into the identical import path, so without this
override, whichever installed last would win unpredictably. Also raised
`pyproject.toml`'s Python floor from 3.10 to 3.11: `demucs-onnx` already
needed 3.11 as a hard dependency, so 3.10 was never really supported, and
nothing else in the project (CI, CONTRIBUTING.md, this same packaging script)
exercises anything below 3.12 anyway. Caught and fixed a second, unrelated
drift in the same script while in there: its CPU-only torch pin was still
2.6.0, left behind when `pyproject.toml`'s own torch floor moved to 2.9.

### Two bug reports from the first real AMD build, and the auto-device regression they surfaced

Ed's first real run of a packaged build (the wordmark, Settings UI, and packaging
work above, actually built and launched on his machine) surfaced two UI misses —
the in-app topbar still read the old two-tone "Stem"/"Deck", and first-run setup
still said "No NVIDIA GPU" — plus a real correctness bug found while fixing the
second one. The topbar was simply missed by the original rename pass; fixed to
match the wordmark's "Any" gold / "StemDeck" white split. The GPU message,
though, led somewhere more important: with DirectML now Settings-selectable
*and* bundled into every Windows package (the two changes just above, combined),
"auto" device resolution would have started silently routing AMD/Intel-on-Windows
users — exactly this fork's audience — into `dml`, a path this project's own
Phase 1/1.5 research (see `docs/plan.md`) already proved is broken on real
hardware (a DirectML `ConvTranspose` crash, and a numerically-wrong-output
`InstanceNormalization` bug found while trying to route around it). Auto
resolution no longer falls through to `dml` — it stays `cuda > mps > cpu`, with
`dml` still manually selectable in Settings for anyone who wants it. Onboarding's
"no GPU" message is now simply "No compatible GPU found — stem separation will
use CPU", which is accurate again now that CPU is genuinely where auto lands.

### Settings UI · 2026-09 · a real "Enable AMD GPU (WSL2 + ROCm)" toggle

The WSL2/ROCm backend (shipped as v0.1.0, above) previously had no UI at all —
enabling it meant hand-editing the app's config file, the workaround this
project's own manual smoke test used before the guided-setup flow existed.
Settings now has a real toggle plus an optional WSL distro field, wired to the
same store keys `start_backend` already reads at launch. Windows-desktop-only,
hidden everywhere else. Deliberately narrow: it flips the switch to what
already works, but does not install or verify WSL2/ROCm itself — that's still
the guided-setup flow below.

### Onboarding · 2026-09 · the "no GPU found" message now knows about WSL2/ROCm

The toggle above fixed *reaching* the WSL2/ROCm path, but the first-run wizard
didn't know it existed yet — turning it on and restarting still showed "No
compatible GPU found - stem separation will use CPU" on the startup screen,
exactly backwards. `ensure_torch_device()` now checks the toggle before
running its native NVIDIA-only probe, and skips straight to an honest "AMD GPU
(WSL2 + ROCm) enabled" message when it's on. See `docs/plan.md` for the fix
and its staleness handling if the toggle is later turned back off.

## In flight / next

No formal issue tracker yet for fork-specific work (unlike StemDeck's own
numbered issues, which this file used to — incorrectly — link to as if they
were AnyStemDeck's). Current list, roughly in the order it's likely to matter:

- **The WSL2/ROCm guided-setup flow.** Today's AMD setup is nine manual
  terminal steps ([README](README.md#amd-gpu-on-windows-via-wsl2--rocm)).
  Decision #6 and its follow-ups in `docs/plan.md`'s Decisions section settle
  the shape of the script that will replace them.
- **A persistent in-app GPU/acceleration indicator.** Onboarding's message is
  transient (the window navigates away to the main app right after) and the
  main app currently shows device status nowhere at all. Placement and design
  still open.
- **DirectML parity testing.** Numerical and speed parity against the
  CPU/CUDA path on the same input hasn't been verified on real hardware yet.

## How releases work

Inherited from StemDeck, unchanged so far since this fork hasn't cut a
packaged release yet: the version comes from the git tag (`pyproject.toml` is
`dynamic`, so there is no version number to hand-edit anywhere). Every release
is meant to ship as a pre-release and get promoted by hand once it's been
verified — the in-app updater only ever looks at the newest release that is
neither a draft nor a pre-release, so an unpromoted build stays invisible to
it by design, not by oversight. Desktop and Docker are expected to split CPU
and GPU differently: Docker publishes one image that runs CPU by default and
activates GPU via `--runtime=nvidia`, while the desktop build is expected to
install the CUDA torch wheel at first run rather than bundle it, to stay
under GitHub's 2 GiB release-asset cap — a real constraint StemDeck hit and
designed around, worth keeping in mind once AnyStemDeck starts cutting its
own desktop releases.

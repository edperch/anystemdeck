# AnyStemDeck

**Stem separation that runs on any GPU — not just NVIDIA's.**

AnyStemDeck is a fork of [StemDeck](https://github.com/stemdeckapp/stemdeck), the open-source stem extraction app, built to get real GPU acceleration on AMD cards that AMD's own official Windows ROCm support leaves out. An RX 7800 XT, for example, isn't on AMD's supported list for native Windows ROCm — and without it, PyTorch's CUDA-only fast path leaves the card running separation on the CPU, 5-10x slower than it needs to be.

## Why this exists

StemDeck (and the similar closed-source app Trama) both run Meta's Hybrid Transformer Demucs model through PyTorch, which only gets first-class hardware acceleration on NVIDIA (CUDA) and Apple Silicon (MPS). On an AMD card — even a current one like an RX 7800 XT — both apps silently fall back to the CPU.

## How it gets there: WSL2 + ROCm

The path that actually works, confirmed end-to-end on real RX 7800 XT hardware: ROCm's Linux support covers a much wider range of AMD cards than its native-Windows build does, and WSL2 exposes the GPU to a Linux environment through paravirtualized GPU passthrough. AnyStemDeck runs its existing, unmodified PyTorch backend *inside* WSL2, where ROCm presents itself to PyTorch as an ordinary `torch.cuda` device — the exact same code path StemDeck already uses for NVIDIA GPUs. No model changes, no new inference engine: same worker, same job queue, same desktop UI, just launched inside a Linux environment whose driver stack actually recognizes the card.

Today this setup is manual — see below. The desktop app can drive a WSL2-hosted backend itself once it's told to (Rust-side launch, health-check, and clean shutdown are all built and confirmed working), but there's no in-app guided flow yet to detect or install WSL2/ROCm for you; that's still on the roadmap. See [`docs/plan.md`](docs/plan.md) for the full build log, including the exact install stages and every version-matching issue that came up getting here (WSL2's paravirtualized GPU driver, ROCm's WSL-specific install usecase, and pairing PyTorch's ROCm build to the installed ROCm version).

## Setup

There's no packaged release yet — running AnyStemDeck today means running it from a source checkout.

### NVIDIA, Apple Silicon, or CPU-only

No AMD-specific steps needed — this works exactly like upstream StemDeck. See [`CONTRIBUTING.md`](CONTRIBUTING.md#development-setup) for the dev setup: install [`uv`](https://docs.astral.sh/uv/) and `ffmpeg`, then

```bash
uv sync --python 3.12
./run.sh start
```

and open `http://localhost:8000`.

### AMD GPU on Windows, via WSL2 + ROCm

This is the whole reason this fork exists. The recipe below is the exact one confirmed working end-to-end on an RX 7800 XT; expect some of the specific package versions to drift over time — check [AMD's ROCm-on-WSL install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html) for whatever's current, and match it to the torch version this project pins in `pyproject.toml`.

1. **Install WSL2 with Ubuntu 24.04**, from an elevated PowerShell:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

2. **Install ROCm inside WSL2 — using the WSL-specific usecase, not the plain Linux one.** This matters: a plain `--usecase=rocm` install pulls a native-Linux HSA runtime that expects `/dev/kfd` and fails with `hsa_init Failed, possibly no supported GPU devices` under WSL2, where the GPU is only reachable via `/dev/dxg`. Follow AMD's WSL install guide (link above) to download `amdgpu-install` for Ubuntu 24.04, then:
   ```bash
   sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
   ```
   A reasonably recent Windows AMD driver already exposes `/dev/dxg` to WSL2 — check `ls /dev/dxg` inside WSL2 before assuming you need AMD's separate "for WSL2" driver build.

3. **Verify ROCm itself sees the GPU**, independent of Python or PyTorch entirely:
   ```bash
   rocminfo
   ```
   You should see your card listed as a GPU agent (e.g. `gfx1101` for an RX 7800 XT) with no `hsa_init` error.

4. **Install ffmpeg** inside the same WSL2 environment:
   ```bash
   sudo apt install -y ffmpeg
   ```

5. **Install PyTorch's ROCm build, matched to your installed ROCm version** — from `repo.radeon.com`, not the default PyPI/`pytorch.org` index. `pytorch.org`'s official ROCm wheels only track a handful of ROCm releases, and a mismatched one silently fails to enumerate the GPU (a real HIP-level ABI gap between the wheel's bundled runtime and a newer ROCm userspace, not just a version-string mismatch) — so for a ROCm 7.2.1 install, use AMD's own ROCm-7.2.1-matched wheels instead, from the ROCm manylinux index on `repo.radeon.com` (check the [AMD PyTorch-for-ROCm install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installpytorch.html) for the current file names — they include a git-hash suffix that changes with each build):
   ```bash
   wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1+rocm7.2.1.<hash>-cp312-cp312-linux_x86_64.whl
   wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0+rocm7.2.1.<hash>-cp312-cp312-linux_x86_64.whl
   wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1+rocm7.2.1.<hash>-cp312-cp312-linux_x86_64.whl
   pip install --user --break-system-packages --force-reinstall \
     torch-2.9.1+rocm7.2.1.*.whl torchaudio-2.9.0+rocm7.2.1.*.whl triton-3.5.1+rocm7.2.1.*.whl
   ```
   `triton` isn't optional — the ROCm-matched torch wheel depends on a specific ROCm-matched `triton` build that also only exists on `repo.radeon.com`, not PyPI.

6. **Confirm the GPU actually enumerates**, with a real op rather than just `torch.cuda.is_available()` (which can silently swallow the real error and just return `False`):
   ```bash
   python3 -c "import torch; print(torch.zeros(1).cuda())"
   ```
   This should print a CUDA tensor, not raise `RuntimeError: No HIP GPUs are available`.

7. **Install AnyStemDeck itself into that same WSL2 Python environment**, from the repo checkout (works whether the checkout lives on a Windows drive under `/mnt/c/...`/`/mnt/d/...` or natively inside WSL2 — a native WSL2 clone avoids some DrvFs slowness on git operations, but isn't required):
   ```bash
   cd /mnt/d/path/to/AnyStemDeck   # wherever your checkout is
   pip install --user --break-system-packages -e .
   ```

8. **Run it.** The simplest route today is running the backend directly inside WSL2 and using it from a browser — this is genuinely the GPU-accelerated path, since it's the same backend code either way, just running in the environment that can see the GPU:
   ```bash
   ./run.sh start
   ```
   then open `http://localhost:8000` in a browser on Windows (WSL2 forwards `localhost` automatically). A finished separation job's log line should read `device=cuda` (ROCm presents itself to PyTorch as CUDA) — that's the confirmation it's actually using the GPU, not falling back to CPU.

9. **Optional: let the Windows desktop app manage the WSL2 backend for you**, instead of running `./run.sh start` by hand. This works (it's the fully-built, tested launch/health/shutdown mechanism the desktop app uses), but since there's no Settings UI for it yet, it means hand-editing the app's settings file once: close the app if it's running, then find (or create) `user-data.json` inside your stems/jobs folder — by default `Documents\AnyStemDeck\jobs\user-data.json` (check Settings → stems location in the app if you've relocated it, or if your Documents folder is redirected by OneDrive) — and merge in:
   ```json
   {
     "wsl2BackendEnabled": true,
     "wsl2Distro": "Ubuntu-24.04"
   }
   ```
   alongside whatever's already in that file. Relaunch the app from `desktop/src-tauri` (`cargo run`, or a built `.exe` once packaging exists) — it will spawn and manage the WSL2-hosted backend itself, including clean shutdown when you close it.

If something along the way doesn't match, [`docs/plan.md`](docs/plan.md) has the full, blow-by-blow debugging history behind this recipe — useful if you hit one of the same version-mismatch or WSL2-interop issues that came up building it.

## A parked alternative: DirectML / ONNX

An earlier approach ran a [demucs-onnx](https://github.com/StemSplit/demucs-onnx) export of HT-Demucs through ONNX Runtime's DirectML execution provider, which would have covered AMD, Intel, and NVIDIA GPUs alike through DirectX 12 rather than a vendor-specific driver stack. That code is still in the repo (`app/pipeline/demucs_onnx_worker.py`, the `dml` device option), but it's parked: DirectML's `InstanceNormalization` operator produced numerically wrong output on real hardware, a bug outside this project's control. WSL2 + ROCm sidesteps the problem entirely for AMD cards ROCm supports, so that's the path this project is actually built around now; DirectML may get revisited if that bug is ever fixed upstream, or for AMD/Intel cards ROCm doesn't cover.

## Status

Working end-to-end on real hardware: an RX 7800 XT running the full separation pipeline — vocals, drums, bass, guitar, piano, other — through WSL2 + ROCm, GPU-accelerated, producing correct, playable stems. The desktop app, player, mixer, beat grid, and job queue are otherwise identical to StemDeck's. Still ahead: a guided in-app setup flow for WSL2/ROCm (today's install is manual — see Setup above), and packaging a distributable build. See [`docs/plan.md`](docs/plan.md) for the detailed, up-to-date build log.

## Relationship to StemDeck

This is a fork under StemDeck's own [Apache-2.0 license](LICENSE), tracking [`stemdeckapp/stemdeck`](https://github.com/stemdeckapp/stemdeck) as upstream. Full credit to the StemDeck maintainers for the application this is built on; see [`NOTICE`](NOTICE) for the required attribution. The WSL2/ROCm backend needs no changes to StemDeck's own PyTorch worker at all, so it's a reasonable candidate to offer back upstream as a PR once it's had more real-world testing here.

## Credits

- [StemDeck](https://github.com/stemdeckapp/stemdeck) — the application this is forked from (Apache-2.0).
- [Demucs / HT-Demucs](https://github.com/facebookresearch/demucs) — Meta AI Research (MIT).
- [demucs-onnx](https://github.com/StemSplit/demucs-onnx) — the ONNX export and DirectML inference path this fork builds on (MIT), by [StemSplit](https://stemsplit.io).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

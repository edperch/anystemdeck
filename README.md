# AnyStemDeck

**Stem separation that runs on any GPU — not just NVIDIA's.**

AnyStemDeck is a fork of [StemDeck](https://github.com/stemdeckapp/stemdeck), the open-source stem extraction app, built to get real GPU acceleration on AMD cards that AMD's own official Windows ROCm support leaves out. An RX 7800 XT, for example, isn't on AMD's supported list for native Windows ROCm — and without it, PyTorch's CUDA-only fast path leaves the card running separation on the CPU, 5-10x slower than it needs to be.

## Why this exists

StemDeck (and the similar closed-source app Trama) both run Meta's Hybrid Transformer Demucs model through PyTorch, which only gets first-class hardware acceleration on NVIDIA (CUDA) and Apple Silicon (MPS). On an AMD card — even a current one like an RX 7800 XT — both apps silently fall back to the CPU.

## How it gets there: WSL2 + ROCm

The path that actually works, confirmed end-to-end on real RX 7800 XT hardware: ROCm's Linux support covers a much wider range of AMD cards than its native-Windows build does, and WSL2 exposes the GPU to a Linux environment through paravirtualized GPU passthrough. AnyStemDeck runs its existing, unmodified PyTorch backend *inside* WSL2, where ROCm presents itself to PyTorch as an ordinary `torch.cuda` device — the exact same code path StemDeck already uses for NVIDIA GPUs. No model changes, no new inference engine: same worker, same job queue, same desktop UI, just launched inside a Linux environment whose driver stack actually recognizes the card.

The desktop app detects whether WSL2 + ROCm + PyTorch are already set up and will offer a guided setup flow if not; it never provisions any of this silently in the background. See [`docs/plan.md`](docs/plan.md) for the full build log, including the exact install stages and the version-matching issues that came up along the way (WSL2's paravirtualized GPU driver, ROCm's WSL-specific install usecase, and pairing PyTorch's ROCm build to the installed ROCm version).

## A parked alternative: DirectML / ONNX

An earlier approach ran a [demucs-onnx](https://github.com/StemSplit/demucs-onnx) export of HT-Demucs through ONNX Runtime's DirectML execution provider, which would have covered AMD, Intel, and NVIDIA GPUs alike through DirectX 12 rather than a vendor-specific driver stack. That code is still in the repo (`app/pipeline/demucs_onnx_worker.py`, the `dml` device option), but it's parked: DirectML's `InstanceNormalization` operator produced numerically wrong output on real hardware, a bug outside this project's control. WSL2 + ROCm sidesteps the problem entirely for AMD cards ROCm supports, so that's the path this project is actually built around now; DirectML may get revisited if that bug is ever fixed upstream, or for AMD/Intel cards ROCm doesn't cover.

## Status

Working end-to-end on real hardware: an RX 7800 XT running the full separation pipeline — vocals, drums, bass, guitar, piano, other — through WSL2 + ROCm, GPU-accelerated, producing correct, playable stems. The desktop app, player, mixer, beat grid, and job queue are otherwise identical to StemDeck's. Still ahead: a guided in-app setup flow for WSL2/ROCm (today's install is manual), and packaging a distributable build. See [`docs/plan.md`](docs/plan.md) for the detailed, up-to-date build log.

## Relationship to StemDeck

This is a fork under StemDeck's own [Apache-2.0 license](LICENSE), tracking [`stemdeckapp/stemdeck`](https://github.com/stemdeckapp/stemdeck) as upstream. Full credit to the StemDeck maintainers for the application this is built on; see [`NOTICE`](NOTICE) for the required attribution. The WSL2/ROCm backend needs no changes to StemDeck's own PyTorch worker at all, so it's a reasonable candidate to offer back upstream as a PR once it's had more real-world testing here.

## Credits

- [StemDeck](https://github.com/stemdeckapp/stemdeck) — the application this is forked from (Apache-2.0).
- [Demucs / HT-Demucs](https://github.com/facebookresearch/demucs) — Meta AI Research (MIT).
- [demucs-onnx](https://github.com/StemSplit/demucs-onnx) — the ONNX export and DirectML inference path this fork builds on (MIT), by [StemSplit](https://stemsplit.io).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

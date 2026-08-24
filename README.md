# AnyStemDeck

**Stem separation that runs on any GPU — not just NVIDIA's.**

AnyStemDeck is a fork of [StemDeck](https://github.com/stemdeckapp/stemdeck), the open-source stem extraction app, with one focus: hardware-accelerated separation on AMD and Intel GPUs on Windows, where PyTorch's CUDA-only acceleration leaves everyone else running on the CPU.

> **Status: early / pre-alpha.** This repo currently holds the plan and the license. No code has been ported or written yet — see [`docs/plan.md`](docs/plan.md) for what's being built and in what order.

## Why this exists

StemDeck (and the similar closed-source app Trama) both run Meta's Hybrid Transformer Demucs model through PyTorch, which only gets hardware acceleration on NVIDIA (CUDA) and Apple Silicon (MPS). On an AMD card — even a current one like an RX 7800 XT — both apps silently fall back to the CPU, which is 5-10x slower.

The fix isn't ROCm: as of mid-2026, AMD's official Windows ROCm support only covers RDNA4 (RX 9000-series) cards, so most AMD Windows users are still locked out. Instead, AnyStemDeck uses **[demucs-onnx](https://github.com/StemSplit/demucs-onnx)** — a verified, MIT-licensed ONNX export of HT-Demucs (parity within ~1.7×10⁻⁴ of the original PyTorch weights) that runs through **ONNX Runtime's DirectML execution provider**. DirectML sits on top of DirectX 12, so it works on any modern GPU on Windows — AMD, Intel, or NVIDIA — with no vendor-specific driver stack required.

## How it works

StemDeck already isolates its inference call behind a single persistent worker process (`app/pipeline/demucs_worker.py`) that loads a PyTorch Demucs model and serves jobs over stdin/stderr. AnyStemDeck adds a sibling worker that does the same job through `demucs-onnx` and an ONNX Runtime session instead — same job protocol, same job queue, same desktop UI, different inference backend. Device selection (`auto | cuda | mps | dml | cpu`) picks which worker gets used per job, so the app degrades gracefully: DirectML where available, CUDA/MPS where NVIDIA or Apple hardware is present, CPU everywhere else.

Nothing about StemDeck's player, mixer, beat grid, or job queue changes — only the separation stage gets a second engine.

## Relationship to StemDeck

This is a fork under StemDeck's own [Apache-2.0 license](LICENSE), tracking [`stemdeckapp/stemdeck`](https://github.com/stemdeckapp/stemdeck) as upstream. Full credit to the StemDeck maintainers for the application this is built on; see [`NOTICE`](NOTICE) for the required attribution. The DirectML backend itself is intended to be useful enough to offer back upstream as a PR once it's proven out here — this fork exists to develop and validate it against real AMD hardware first.

## Credits

- [StemDeck](https://github.com/stemdeckapp/stemdeck) — the application this is forked from (Apache-2.0).
- [Demucs / HT-Demucs](https://github.com/facebookresearch/demucs) — Meta AI Research (MIT).
- [demucs-onnx](https://github.com/StemSplit/demucs-onnx) — the ONNX export and DirectML inference path this fork builds on (MIT), by [StemSplit](https://stemsplit.io).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

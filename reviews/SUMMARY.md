# StemDeck Full Codebase Review

**Branch:** main | **Version:** v0.5.0-alpha.3 | **Date:** 2026-05-21

---

## Tool Availability

| Tool | Status |
|------|--------|
| Static analysis (grep, ast, file reads) | Run across all source files |
| Bandit (Python SAST) | Not run — findings derived from code reading |
| pip-audit | CI-managed; current state documented in `.woodpecker/ci.yml` |
| cargo clippy | Not run locally — findings derived from code reading |
| node --check | Not run — findings derived from code reading |
| pytest | Not run — coverage gaps identified from source only |

---

## System Overview

**Architecture**

- FastAPI + uvicorn on 127.0.0.1:PORT (Tauri: dynamic free port; Docker: 8000)
- In-memory job registry (`_jobs: dict[str, Job]`) backed by `registry.json` for terminal jobs only
- Pipeline: asyncio task -> `asyncio.to_thread` -> blocking thread -> `subprocess.Popen` (Demucs) or `subprocess.run` (ffmpeg/ffprobe)
- SSE polling loop: 0.2 s interval, 4-hour deadline — pure poll-and-stream, no push
- Tauri shell: Rust process spawns Python backend; kills on `ExitRequested` / `CloseRequested`
- No authentication, no database — local-only, single-user by design

**Critical hot paths**

```
POST /api/jobs
  -> validate input (local)
  -> registry check: count queued jobs (lock -> iterate -> release)
  -> create_task(run_pipeline)           <- returns immediately
  -> asyncio.to_thread(_run_blocking)    <- blocks thread pool worker
      -> YoutubeDL.extract_info x2 (network, serial)
      -> subprocess.Popen (Demucs, 5-30 min)  <- stderr thread + watchdog thread
      -> subprocess.run (ffprobe, ffmpeg x2)  <- blocking, up to 300s each

GET /api/jobs/{id}/stems/{name}.mp3
  -> asyncio.create_subprocess_exec (ffmpeg pipe) <- streaming response
  -> yield 64 kB chunks until EOF / disconnect
```

---

## Security Findings

| Severity | ID | Finding | STRIDE | OWASP | Tracked |
|----------|----|---------|--------|-------|---------|
| HIGH | S1 | **Stored XSS via track title in innerHTML** — `static/js/catalog.js:851,894`: `track.title` from YouTube metadata injected directly into `innerHTML` without escaping. In Tauri desktop mode, XSS payload can invoke `window.__TAURI__` commands including `open_url`, `store_get`, and `store_set`. | T, E | A03 | — |
| HIGH | S2 | **Stored XSS via channel name in innerHTML** — `static/js/catalog.js:900`: `track.channel` (YouTube uploader field) injected directly into `innerHTML`. Same attack surface as S1. | T | A03 | — |
| HIGH | S3 | **Open URL / scheme injection via `open_url` Tauri command** — `desktop/src-tauri/src/main.rs:933`: caller-supplied URL passed directly to `cmd /c start` (Windows), `open` (macOS), or `xdg-open` (Linux) without scheme validation. No guard against `file:///`, `ms-msdt:`, or other dangerous schemes. Chained with S1/S2 XSS, gives arbitrary app/file launch. | E | A01 | — |
| MEDIUM | S4 | **SSE endpoint missing `JOB_ID_RE` validation** — `app/api/events.py:20`: the only endpoint that does not validate the job ID regex before hitting the registry. All other endpoints validate first. Future code changes inheriting this pattern could be exploitable. | T | A03 | — |
| MEDIUM | S5 | **Stored XSS via YouTube tags in innerHTML** — `static/js/catalog.js:1224`: `tag` strings from `info.get("tags")` lower-cased but not HTML-escaped before `chip.innerHTML = ...`. Lower-casing neutralizes `<SCRIPT>` but not event-handler XSS (`onerror`, `onload`). | T | A03 | — |

**Recommended fix order (security):**

1. S4 — one-line fix, closes a validation gap before any other work.
2. S3 — add `https://` / `http://` scheme whitelist in `open_url`; low effort, high impact.
3. S1 + S2 + S5 — replace `innerHTML` template literals with DOM API (`textContent`, `createElement`/`setAttribute`) for all YouTube-sourced strings.

---

## Reliability Findings

| Priority | ID | Finding | Impact | Effort | Tracked |
|----------|----|---------|--------|--------|---------|
| P1 | R1 | **Background tasks not stored** — `main.py:104,114`: `asyncio.create_task(_sweep_loop())` and `_desktop_parent_watchdog()` results discarded. Python GC can collect a task with no strong references; watchdog/sweep silently stops. | Silent task cancellation | trivial | — |
| P1 | R2 | **`free_port()` TOCTOU race in Tauri** — `main.rs:1396-1402`: binds port 0, reads port number, drops listener, then passes to uvicorn. Another process can grab the port between drop and uvicorn bind. Causes 90 s startup hang with no retry. | Backend fails to start | small | — |
| P1 | R3 | **No client-disconnect detection in SSE** — `events.py:25-48`: generator polls every 0.2 s for up to 4 hours. Zombie tabs (browser closed) keep the loop running until the next `yield` raises. Multiple zombie connections on Docker compoun. | Event loop overhead; wasted CPU | small | — |
| P2 | R4 | **Failed job directory not cleaned up** — `runner.py:152-160, 185-194`: on pipeline error, `job_dir` persists until TTL sweep (24 h default). Each failed job can leave 100-300 MB of partial output on disk. | Disk waste per failed job | small | — |
| P2 | R5 | **Capacity check race** — `jobs.py:138,154`: `pending` count read and job registration are not atomic. Two concurrent requests can both pass the `MAX_PENDING_JOBS` guard simultaneously. | Queue cap exceeded by concurrent requests | trivial | — |
| P2 | R6 | **`persist_registry` snapshot outside the lock** — `registry.py:68-80`: acquires lock to snapshot, releases, then writes. Concurrent `remove()` between snapshot and write produces inconsistent registry.json. | Stale entry survives restart | small | — |
| P2 | R7 | **ffmpeg process leak on ASGI middleware error** — `stems.py:50-83`: `finally` block kills the ffmpeg child if `returncode is None`, but if `StreamingResponse` raises before the generator starts, `aclose()` may not fire and the process leaks. | 1 leaked ffmpeg per edge-case disconnect | small | — |
| P3 | R8 | **Stall watchdog thread join races** — `separate.py:108`: `wt.join(timeout=5)` times out if watchdog is sleeping. `proc.terminate()` can fire on an already-exited process (benign on Unix). Adds up to 5 s latency on every successful Demucs job. | Minor latency | trivial | — |
| P3 | R9 | **SIGKILL without SIGTERM in Tauri stop** — `main.rs:959-967`: `child.kill()` sends SIGKILL immediately, bypassing uvicorn graceful shutdown. In-flight `_write_metadata` (non-atomic) can be truncated. | Metadata.json truncation on kill | small | — |
| P3 | R10 | **`os._exit(0)` in parent watchdog** — `main.py:98`: bypasses atexit handlers, uvicorn shutdown, and any future lifespan `finally` cleanup. | Log truncation; future cleanup silently skipped | trivial | — |
| P3 | R11 | **No SSE connection cap** — `events.py`: no limit on concurrent SSE connections per job or globally. Each holds an asyncio task. Misbehaving client could open hundreds on Docker deployment. | Event loop starvation on Docker | small | — |
| P3 | R12 | **`wait_for_health` busy-polls for up to 90 s** — `main.rs:1404-1432`: 250 ms sleep loop on Tauri blocking thread. UI invoke blocked for up to 90 s on slow startup. | UI unresponsive during startup | small | — |

---

## Code Quality Findings

| Severity | ID | Finding | Source | Tracked |
|----------|----|---------|--------|---------|
| HIGH | C1 | **`events.py` missing `JOB_ID_RE` validation** (duplicate of S4) — the only endpoint without job ID regex validation. | Security + code consistency | see S4 |
| HIGH | C2 | **`renderTrackItem` injects `track.title` unescaped into innerHTML** — `catalog.js:900` (duplicate of S1). | Security + code correctness | see S1 |
| HIGH | C3 | **`_set()` imported from `download.py` by 3 unrelated pipeline modules** — belongs in `app/core/models.py` or `pipeline/utils.py`. Import coupling between pipeline stages creates fragile dependencies. | ARCH | — |
| HIGH | C4 | **`PATCH /api/jobs/{id}/sections` has zero tests** — no happy path, no validation, no 404, no disk error coverage. | TEST | — |
| HIGH | C5 | **File upload path has no tests** — `_create_local_job` including size limits, extension validation, duration check, and orphan cleanup is entirely untested. | TEST | — |
| MEDIUM | C6 | **SSE + 1 s polling run simultaneously** — polling loop fires on every job submission even when SSE is healthy; should activate only on SSE error. | Frontend correctness | — |
| MEDIUM | C7 | **`visualAudioContext` never closed in `destroyPlayer()`** — risks hitting browser 6-context limit (`AudioContext` is capped per page in Chrome/Safari). | Resource leak | — |
| MEDIUM | C8 | **`track.thumb` URL injected via innerHTML** — `thumbHtml()` in `catalog.js:864`: `src="${track.thumb}"` — should use `setAttribute`. (Related to S1/S2 surface.) | Security + code | see S1 |
| MEDIUM | C9 | **`run_pipeline` / `run_local_pipeline` near-identical** — ~60 lines of duplicated error/cancel/sweep handling between `runner.py` functions. | Maintainability | — |
| MEDIUM | C10 | **PowerShell URL injection** — `download.py`: `download_file_with_powershell` embeds URL in a PowerShell string with single quotes; a URL containing `'` breaks the command. | Correctness on Windows | — |
| MEDIUM | C11 | **`test_stems_api.py` writes real files to `JOBS_DIR`** instead of `tmp_path` — breaks test isolation when JOBS_DIR is non-empty. | Test isolation | — |
| MEDIUM | C12 | **Module-level side effects at import time** — `ensure_runtime_dirs()` and `restore_registry()` run during import in `main.py`. Breaks test isolation for any test that imports from `app.main`. | Test isolation | — |
| MEDIUM | C13 | **Missing tests** for MP3 streaming endpoint and 503 capacity limit response. | TEST | — |
| LOW | C14 | **`Job.status` typed as `str`** instead of `Literal["queued", "processing", "done", "error", "cancelled"]` — loses exhaustiveness checking. | Type safety | — |
| LOW | C15 | **Dead code: `noneSelected` variable** in `catalog.js` — assigned but never read. | Dead code | — |
| LOW | C16 | **`structuredClone` vs JSON round-trip** — inconsistent deep-copy strategy across JS modules. | Minor inconsistency | — |
| LOW | C17 | **Untracked `ResizeObserver` leak** — observer created in player.js without a stored reference for cleanup. | Resource leak | — |
| LOW | C18 | **Deferred `numpy` import inside function body** — `analyze.py`: `import numpy as np` inside a function; unconventional, confuses static analysis. | Code style | — |
| LOW | C19 | **`loadState` error silently swallowed** — `state.js`: `try { ... } catch { }` with empty catch on localStorage load. Per project rules: catch must at minimum `console.warn(err)`. | Rule violation | — |

---

## Documentation Findings

| Severity | ID | Finding | Tracked |
|----------|----|---------|---------|
| HIGH | D1 | **`sweep_old_jobs` docstring stale** — `collect.py:183`: says "Called once per new job submission"; function is now called only from the hourly background loop. Misleads future contributors. | — |
| HIGH | D2 | **README API table missing 4 endpoints** — `README.md:247-253`: `GET /api/jobs`, `PATCH /api/jobs/{id}/sections`, `GET /api/jobs/{id}/stems/{name}.mp3`, and `/api/health` are absent. | — |
| HIGH | D3 | **README claims range requests for WAV stems** — `README.md:251`: `FileResponse` does not implement HTTP range requests. Claim is factually wrong and misleads integrators. | — |
| HIGH | D4 | **README config table omits 9 env vars** — `README.md:230-239`: `STEMDECK_DATA_DIR`, `STEMDECK_CACHE_DIR`, `STEMDECK_DOWNLOADS_DIR`, `STEMDECK_MODELS_DIR`, `STEMDECK_LOGS_DIR`, `STEMDECK_FFMPEG_DIR`, `STEMDECK_FFMPEG`, `STEMDECK_FFPROBE`, `STEMDECK_TIMEOUT_*` not documented. Critical for Docker/desktop users. | — |
| MEDIUM | D5 | **`run.sh` env vars undocumented** — `HOST`, `PORT`, `RELOAD`, `FOREGROUND` used but not listed anywhere. Particularly `RELOAD=1` is useful for dev. | — |
| MEDIUM | D6 | **All FastAPI route handlers have no docstrings** — OpenAPI `/docs` shows bare endpoint paths with no descriptions or summaries. Affects all 10 routes. | — |
| MEDIUM | D7 | **`JobRequest` / `SectionItem` Pydantic fields use inline comments, not `Field(description=...)`** — `jobs.py:105-112`: field semantics invisible in generated OpenAPI schema. | — |
| MEDIUM | D8 | **`FastAPI(title="StemDeck")` missing `description=` and `version=`** — `main.py:118`: `app_version()` defined but not passed to FastAPI constructor. | — |
| MEDIUM | D9 | **`config.py` lacks module docstring explaining two-mode design** — portable mode (`STEMDECK_DATA_DIR`) vs dev mode, and internal Tauri bridge vars, not explained anywhere. | — |
| MEDIUM | D10 | **Broken "Buy Me a Coffee" link** — `README.md:38`: URL missing `https://` scheme. | — |
| LOW | D11 | **README Python version mismatch** — `README.md:139`: says "3.10-3.13" but `run.sh setup` pins `--python 3.12`; inconsistent. | — |
| LOW | D12 | **Badge links use old `thcp/stemdeck` repo** — `README.md:9-12`: two different GitHub orgs in the same file; one is stale. | — |
| LOW | D13 | **No module-level docstrings on any API module** — `jobs.py`, `stems.py`, `events.py`: OpenAPI tag descriptions never registered. | — |
| LOW | D14 | **Dockerfile uses `pip` without explanation** — `build/Dockerfile:53`: no comment explaining why `pip` instead of `uv` for the venv install. | — |
| LOW | D15 | **Stale date in `ci.yml` comment** — `.woodpecker/ci.yml:64-65`: "no fix version available as of 2025-12-15" — date is now 6 months old. Re-check joblib PYSEC-2024-277 advisory status. | — |

---

## Recommended Fix Order

### Immediate (security)

1. **S4 / C1** — Add `JOB_ID_RE` validation to `app/api/events.py:20`. One line.
2. **S3** — Whitelist `http://` and `https://` schemes in `open_url` (`main.rs:933`).
3. **S1 + S2 + S5 / C2 + C8** — Replace all `innerHTML` template literals for YouTube-sourced strings with DOM API calls in `catalog.js`.

### High value (reliability + correctness)

4. **R1** — Store `asyncio.create_task()` results in `main.py:104,114`. Trivial.
5. **R5** — Make capacity check + registration atomic in `registry.py`. Small.
6. **R4** — Call `shutil.rmtree(job_dir)` in the error branch of `runner.py`, same as cancel branch.
7. **C4 + C5 + C13** — Add tests for sections endpoint, file upload path, MP3 streaming, and 503 response. Most impactful test gap.

### Medium (quality + docs)

8. **C3** — Move `_set()` to `app/core/models.py` or `pipeline/utils.py`.
9. **C19** — Add `console.warn(err)` to empty catch in `state.js`.
10. **C6** — Fix SSE + polling running simultaneously in the frontend.
11. **D1** — Fix stale `sweep_old_jobs` docstring.
12. **D2 + D3 + D4** — Update README: API table, range-request claim, env var table.
13. **R2** — Fix `free_port()` TOCTOU in `main.rs` (retry on port conflict, or let uvicorn pick).
14. **R6** — Hold registry lock through `persist_registry` write.

### Low (cleanup)

15. **C14** — Narrow `Job.status` to `Literal[...]`.
16. **C11 + C12** — Fix test isolation (stems test writes to real JOBS_DIR; module-level side effects).
17. **D5 + D6 + D7 + D8** — Improve OpenAPI docs quality.
18. **D10 + D12 + D15** — README link fixes and stale comment.

# StemDeck Code Review

## Tool Availability

No Go, protobuf, or static analysis tools applicable — Python/JS/Rust codebase reviewed manually.

---

## Findings

### Architecture & Design

| ID | Finding | Severity |
|----|---------|----------|
| ARCH-1 | **`_set` private function leaks across module boundaries.** `app/pipeline/download.py:138` defines `_set()` as a module-private helper but it is imported by three other modules (`runner.py:15`, `analyze.py:9`, `separate.py:25`). `_set` has nothing to do with downloading — it is a generic job-field mutator that all pipeline stages need. It should live in `app/core/models.py` (or a `app/pipeline/utils.py`) and be public. | HIGH |
| ARCH-2 | **Module-level side effects execute at import time.** `app/main.py:156-157` calls `ensure_runtime_dirs()` and `restore_registry(JOBS_DIR)` unconditionally at module level, outside the FastAPI lifespan. This means any test that imports `app.main` hits the real filesystem, breaks test isolation, and makes the module non-reusable. Move both calls inside `lifespan()`. | MEDIUM |
| ARCH-3 | **`app/pipeline/__init__.py` re-exports `STEM_NAMES` from config.** `__init__.py` exports `STEM_NAMES` but it belongs in `app/core/config`. Consumers that import from `app.pipeline` for a config constant create a misleading dependency. Remove from `__init__.py`. | LOW |
| ARCH-4 | **`analyze.py` imports `JOBS_DIR` directly for a path check.** `app/pipeline/analyze.py:7` imports `JOBS_DIR` to guard the `_load_audio_ffmpeg` path check. This couples the analysis module to the global config. The caller (`runner.py`) already has the path; the guard is appropriate, but the source should be a passed argument rather than a module-level global import. | LOW |
| ARCH-5 | **`sections` field omitted from `_write_metadata`.** `runner.py:124-142` writes `metadata.json` after pipeline completion but does not include `job.sections`. Sections are stored via `PATCH /api/jobs/{id}/sections` which writes them to `metadata.json` separately. However if a job is new, the initial metadata file has no `sections` key; recovery via `_recover_done_job` reads `meta.get("sections")` so that is fine. But a `registry.persist()` call via `to_record()` does persist sections because `_JOB_FIELDS` includes it. The inconsistency between the two metadata paths (registry.json vs metadata.json) is subtle and may cause a future regression. | LOW |

### Python Patterns

| ID | Finding | Severity |
|----|---------|----------|
| PY-1 | **`app/api/events.py` missing `JOB_ID_RE` input validation.** `events.py:20` calls `registry_get(job_id)` without first validating the `job_id` format against `JOB_ID_RE`. Every other endpoint in `jobs.py` and `stems.py` validates the regex first to block filesystem traversal attempts before touching state. Missing here means a crafted `job_id` with path separators is checked only by the dict lookup (which will miss), but the defence-in-depth pattern is broken. Add `if not JOB_ID_RE.match(job_id): raise HTTPException(404)` before the registry call. | HIGH |
| PY-2 | **Pending-job capacity check is not atomic.** `jobs.py:138-140` and `jobs.py:154-156`: reading the count of queued jobs and registering a new job are two separate operations with no lock between them. Under rapid concurrent requests two clients can both read `pending < MAX_PENDING_JOBS` and both proceed to register. The existing `threading.Lock` in `registry.py` protects individual dict mutations but not the read-then-write pattern here. Fix: move the capacity guard inside a registry function that holds `_lock` for the check + register atomically. | MEDIUM |
| PY-3 | **`_probe_duration` in `jobs.py` is synchronous and blocks the event loop.** `jobs.py:47-69` uses `subprocess.run()` (blocking) directly in an async handler. It is called at line 213 inside `await asyncio.to_thread(...)`, which is correct. But at line 69 the `float(result.stdout.strip())` call happens inside the thread too (correct). **Actually no issue — the whole function is wrapped in `to_thread`.** Annotation-only note: the docstring says nothing about threading; add a comment noting it must only be called from a thread, not the event loop. | LOW |
| PY-4 | **`_create_youtube_job` leaks internal Pydantic validation messages to clients.** `jobs.py:131`: `detail=str(e)` on a Pydantic `ValidationError` can expose schema internals. Use `"Invalid request body"` or extract only the top-level message. | LOW |
| PY-5 | **`Job.status` is typed as `str`, not a `Literal` or `Enum`.** `models.py:16` uses a bare `str` with a comment listing valid values. This means typos like `"separting"` are not caught statically. Replace with `Literal["queued", "downloading", "analyzing", "separating", "done", "error", "cancelled"]` or a `StrEnum`. | LOW |
| PY-6 | **`download.py` retry sleep blocks the event loop.** `download.py:205` calls `time.sleep(wait)` inside `download()`, which runs inside `asyncio.to_thread()`. Blocking `time.sleep` inside a thread is fine (it doesn't block the event loop), but it does hold the thread-pool thread for the backoff period, potentially delaying other `to_thread` calls during a burst of failures. Replace with an async sleep by restructuring the retry loop to be async, or at minimum document that the sleep is intentional inside a thread. | LOW |
| PY-7 | **`runner.py` duplicates error/cancel handling between `run_pipeline` and `run_local_pipeline`.** Lines 145-173 and 176-206 are near-identical. The only difference is whether `job_dir.mkdir` is called upfront. Extract a `_run_pipeline_common(job, job_dir, blocking_fn, *args)` to eliminate the duplication. | MEDIUM |
| PY-8 | **`analyze.py` has a deferred import of `numpy` inside `analyze()`.** `analyze.py:305` imports `numpy as np` inside the function body after the outer `try`. The module-level comment says librosa is pre-warmed, but numpy's deferred import inside the function means the first call pays the import cost. Move `import numpy as np` to module level alongside librosa (with the same `try/except ImportError` guard). | LOW |

### JS Patterns

| ID | Finding | Severity |
|----|---------|----------|
| JS-1 | **`renderedJobs` and `jobSources` in `job.js` grow unbounded.** `job.js:19-20`: `renderedJobs` (Set) and `jobSources` (Map) are module-level and never pruned. Every job submitted in the session accumulates. For a local app with ~hundreds of jobs over time this is negligible in practice, but the `jobSources` Map in particular keeps the source URL string for every job ever submitted in the current session. Add cleanup when a job reaches a terminal status or cap to recent N entries. | LOW |
| JS-2 | **`wireAllButton` computes `noneSelected` but never uses it.** `main.js:78`: `const noneSelected = selectedStems.size === 0;` is declared inside the click handler but never read — the branch immediately after checks `allSelected`. Dead code. | LOW |
| JS-3 | **SSE and polling run simultaneously on every job submission.** `job.js:445-446` calls `startJobPolling(jobId)` followed immediately by `connectEvents(jobId)`. During normal operation SSE is the primary mechanism, but REST polling at 1-second intervals is also active from submission until terminal status, meaning 2× the requests on happy path. The polling is an intentional fallback per the comment, but it should be disabled while SSE is connected and only activated on SSE error. | MEDIUM |
| JS-4 | **`visualAudioContext` in `player.js` is never closed.** `player.js:170`: the `AudioContext` created for visual decoding persists across track loads (by design — `??=`). However `destroyPlayer()` (line 502) does not call `visualAudioContext.close()`. The Web Audio spec allows at most 6 concurrent `AudioContext` instances in some browsers; a long session with many track loads could silently fail decoding. Close the context in `destroyPlayer()` and null the reference so a new context is created on the next load. | MEDIUM |
| JS-5 | **`drawFooterPlaceholder` creates an untracked `ResizeObserver`.** `player.js:874-876`: `new ResizeObserver(...).observe(bar)` is created without a reference. It cannot be disconnected during `destroyPlayer()`. This is a minor memory leak per track load. Assign to a module-level variable and disconnect in `destroyPlayer()`. | LOW |
| JS-6 | **`catalog.js:154` silently swallows `loadState` JSON parse errors.** The outer `catch {}` on line 154 discards all errors including bugs. Add at minimum `console.warn("[catalog] loadState error:", e)`. | LOW |
| JS-7 | **`storeSetDebounced` uses `JSON.parse(JSON.stringify(value))` for deep clone.** `utils.js:37`: this throws on values containing `undefined`, `BigInt`, or circular references. For the mixer state this is fine, but it is a fragile pattern. Use `structuredClone(value)` (available in all modern browsers and Node 17+). | LOW |
| JS-8 | **`catalog.js:864` `thumbHtml` injects `track.thumb` directly into innerHTML without sanitization.** The `src` value is a YouTube CDN URL from the server and thus trusted, but if `track.thumb` ever contained a crafted string (e.g., from a malformed server response) it would be injected raw. Use `el.setAttribute("src", track.thumb)` via DOM API instead of template-literal innerHTML. | MEDIUM |
| JS-9 | **`catalog.js:900-911` `renderTrackItem` injects `track.title` via innerHTML.** The `cat-title` div is set by `${track.title ?? "Unknown track"}` in an innerHTML template. If a YouTube video has a title containing `<script>` or `"`, it could break the layout or (in a non-CSP context) execute. Escape via `_esc()` (already defined in `sections.js`) or use `el.querySelector(".cat-title").textContent = ...` after construction. | HIGH |
| JS-10 | **`sections.js` uses `_esc()` for the label but not for the color CSS variable.** `sections.js:80`: `--sc:${section.color}` is written directly into `style.cssText`. Colors are validated server-side against `_COLOR_RE` (`#[0-9a-fA-F]{3,8}`), so this is safe. But in `_openRename`, the input style also sets `--sc: ${section.color}` (line 298) from the same validated value. No issue, but worth a comment that the color is server-validated. | LOW |

### Rust Patterns

| ID | Finding | Severity |
|----|---------|----------|
| RS-1 | **`free_port()` has a TOCTOU race.** `main.rs:1396-1401`: binds port 0, reads the allocated port, drops the listener, then later uses the port for uvicorn. Between `drop(listener)` and uvicorn binding, another process can claim the port. On a quiet local machine this is extremely unlikely, but it is a classic TOCTOU. Fix: keep the `TcpListener` open until uvicorn is spawned, then close it (or use `SO_REUSEPORT`). | LOW |
| RS-2 | **`start_backend` locks two Mutexes sequentially, risking lock ordering issues.** `main.rs:453` locks `state.url`, then `main.rs:538` locks `state.child`. If any other function ever locks them in a different order a deadlock is possible. Both lock sites should follow a single canonical order (`child` before `url`) with a comment. Currently no inversion exists, but `stop_backend` (line 959) only locks `child`. Still, document the order. | LOW |
| RS-3 | **`stop_backend` in `ExitRequested` and `CloseRequested` both call `stop_backend` but `CloseRequested` also calls `app_handle.exit(0)`.** `main.rs:162-175`: on `ExitRequested` the backend is killed but `exit(0)` is not called. On `CloseRequested` both are done. On platforms where the window close goes through `CloseRequested`, this is fine. But `ExitRequested` fires on `app_handle.exit()` calls too; if `exit(0)` isn't called there, the Tauri event loop may not shut down cleanly on all platforms. This is subtle but the existing Python watchdog (`STEMDECK_PARENT_PID`) handles the backend side, so the impact is limited. | LOW |
| RS-4 | **`download_file_with_powershell` embeds URL in a PowerShell `-Command` string.** `main.rs:1604`: the URL is interpolated directly into a PowerShell script string: `Invoke-WebRequest -Uri '{url}'`. A URL containing a single quote would break the command. Use `-Uri` with the URL as a separate argument via `-Command "... -Uri $env:TARGET_URL"` and set it as an environment variable, or use `-File` with a temp script. | MEDIUM |

### Test Coverage

| ID | Finding | Severity |
|----|---------|----------|
| TC-1 | **`PATCH /api/jobs/{id}/sections` endpoint has zero tests.** No test file covers the sections endpoint: validation (bad color, invalid time range, section ID regex), 404 on unknown job, 500 on disk write failure, or successful update. This is the only API endpoint with no test coverage. | HIGH |
| TC-2 | **File upload path (`_create_local_job`) has no tests.** The multipart upload code path in `jobs.py:152-240` including size limits, extension validation, duration check, and orphan cleanup on failure is completely untested. | HIGH |
| TC-3 | **MP3 streaming endpoint (`GET /api/jobs/{id}/stems/{name}.mp3`) has no tests.** `stems.py:37-83` is untested. | MEDIUM |
| TC-4 | **`test_registry_persistence.py` uses `setup_function`/`teardown_function` instead of `autouse` fixture.** All other test files use the `_isolate_registry` autouse fixture. `test_registry_persistence.py:14-18` uses module-level `setup_function`/`teardown_function` which are pytest xunit-style hooks that work differently from fixtures and don't compose well with other fixtures. Migrate to the autouse fixture pattern for consistency. | LOW |
| TC-5 | **No test for the capacity limit (503) when `MAX_PENDING_JOBS` is reached.** `jobs.py:139-140` has the capacity guard but there is no test that fills the queue to the limit and verifies the next request gets a 503. | MEDIUM |
| TC-6 | **`test_stems_api.py` writes real files to `JOBS_DIR`.** `test_stems_api.py:27-30` writes stem files to the real `JOBS_DIR` config path, not to `tmp_path`. This violates test isolation and can leave residue if a test crashes. Use `monkeypatch.setattr("app.api.stems.JOBS_DIR", tmp_path)` as done in `test_registry_persistence.py`. | MEDIUM |
| TC-7 | **No test for the sweep loop's interaction with `persist_registry`.** `collect.py:182-212` calls `registry_persist` when jobs are removed. The tests in `test_sweep.py` don't verify the registry file is updated after a sweep — only that the in-memory registry and directory are cleaned. | LOW |

### Dead Code / Unused Exports

| ID | Finding | Severity |
|----|---------|----------|
| DC-1 | **`app/pipeline/__init__.py` exports `STEM_NAMES` — unused by any external consumer.** Grepping shows no file imports `STEM_NAMES` from `app.pipeline`; they all import from `app.core.config`. Remove from `__init__.py`. | LOW |
| DC-2 | **`main.js:78` `noneSelected` variable is declared but never used.** Dead code inside `wireAllButton`'s click handler. | LOW |
| DC-3 | **`player.js` imports `renderMixerRow` twice.** `player.js:24` and `player.js:25` both import from `./mixer.js`. Line 24 includes `renderMixerRow` in a named import alongside others; line 25 is a standalone `import { renderMixerRow }`. The second import is redundant. | LOW |

---

## Consolidated Findings Table

| Severity | ID | Finding | File(s) |
|----------|----|---------|---------|
| HIGH | 1 | `events.py` missing `JOB_ID_RE` validation on `job_id` path param before registry lookup | `app/api/events.py:20` |
| HIGH | 2 | `renderTrackItem` injects `track.title` unescaped into innerHTML | `static/js/catalog.js:900-901` |
| HIGH | 3 | `_set()` private helper imported across 3 unrelated modules — wrong home | `app/pipeline/download.py:138`, `runner.py:15`, `analyze.py:9`, `separate.py:25` |
| HIGH | 4 | `PATCH /api/jobs/{id}/sections` endpoint has zero tests | `tests/` (missing) |
| HIGH | 5 | File upload path (`_create_local_job`) has no tests | `tests/` (missing) |
| MEDIUM | 6 | Pending-job capacity check is not atomic (read + register race) | `app/api/jobs.py:138-146`, `154-158` |
| MEDIUM | 7 | SSE + 1 s polling run concurrently on every submission — redundant load | `static/js/job.js:445-446` |
| MEDIUM | 8 | `visualAudioContext` never closed in `destroyPlayer()` — potential max-context limit | `static/js/player.js:170`, `502` |
| MEDIUM | 9 | `thumbHtml` injects `track.thumb` URL via innerHTML without DOM API | `static/js/catalog.js:864` |
| MEDIUM | 10 | `run_pipeline` / `run_local_pipeline` are near-duplicate — extract common helper | `app/pipeline/runner.py:145-206` |
| MEDIUM | 11 | `download_file_with_powershell` embeds URL in PowerShell string (quote injection) | `desktop/src-tauri/src/main.rs:1604` |
| MEDIUM | 12 | MP3 streaming endpoint has no tests | `app/api/stems.py:37-83`, `tests/` (missing) |
| MEDIUM | 13 | Capacity limit (503) has no test | `tests/` (missing) |
| MEDIUM | 14 | `test_stems_api.py` writes to real `JOBS_DIR`, not `tmp_path` | `tests/test_stems_api.py:27-30` |
| MEDIUM | 15 | Module-level side effects at import time break test isolation | `app/main.py:156-157` |
| LOW | 16 | `Job.status` should be `Literal[...]` not `str` | `app/core/models.py:16` |
| LOW | 17 | `noneSelected` variable declared but never read | `static/js/main.js:78` |
| LOW | 18 | `drawFooterPlaceholder` creates untracked `ResizeObserver` | `static/js/player.js:874-876` |
| LOW | 19 | `renderedJobs` / `jobSources` grow unbounded | `static/js/job.js:19-20` |
| LOW | 20 | `storeSetDebounced` deep-clone should use `structuredClone` | `static/js/utils.js:37` |
| LOW | 21 | `catalog.js:154` silently swallows `loadState` errors | `static/js/catalog.js:154` |
| LOW | 22 | `free_port()` TOCTOU race between drop and uvicorn bind | `desktop/src-tauri/src/main.rs:1396-1401` |
| LOW | 23 | `analyze.py` deferred `import numpy as np` inside function | `app/pipeline/analyze.py:305` |
| LOW | 24 | `setup_function`/`teardown_function` pattern inconsistent with rest of test suite | `tests/test_registry_persistence.py:14-18` |
| LOW | 25 | Duplicate `import { renderMixerRow }` in `player.js` | `static/js/player.js:24-25` |
| LOW | 26 | `STEM_NAMES` re-exported from `app/pipeline/__init__.py` — unused | `app/pipeline/__init__.py` |
| LOW | 27 | `app/pipeline/analyze.py` imports `JOBS_DIR` from config directly | `app/pipeline/analyze.py:7` |

---

## Recommended Fix Order

1. **Finding 1** — Add `JOB_ID_RE` guard to `events.py`. One-line fix, closes a defence-in-depth gap.
2. **Finding 2** — Fix `renderTrackItem` innerHTML injection. Use `textContent` for the title fields.
3. **Findings 4 + 5** — Write tests for the sections endpoint and file upload path. Highest coverage gap.
4. **Finding 6** — Make capacity check atomic in the registry to close the race.
5. **Finding 8** — Close `visualAudioContext` in `destroyPlayer()`.
6. **Finding 9** — Replace `thumbHtml` innerHTML with DOM API setAttribute for the `src`.
7. **Finding 11** — Fix PowerShell URL injection in `download_file_with_powershell`.
8. **Finding 7** — Disable polling while SSE is healthy; activate only on SSE error.
9. **Finding 3** — Move `_set` to `app/pipeline/utils.py` or `app/core/models.py`.
10. **Finding 10** — Extract `_run_pipeline_common` to eliminate duplication.
11. **Findings 12 + 13** — Add tests for MP3 streaming and capacity limit.
12. **Finding 14** — Fix `test_stems_api.py` to use `monkeypatch` for `JOBS_DIR`.
13. **Finding 15** — Move `ensure_runtime_dirs` + `restore_registry` into `lifespan()`.
14. **Finding 16** — Narrow `Job.status` to `Literal[...]`.
15. **Low findings 17–27** — Address in a single cleanup pass.

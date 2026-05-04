from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from app.core.config import DEMUCS_DEVICE, DEMUCS_MODEL
from app.core.models import Job, JobCancelled
from app.core.registry import set_proc

logger = logging.getLogger("stemdeck.pipeline")

_PCT_RE = re.compile(r"(\d{1,3})%")


def separate(job: Job, source: Path, job_dir: Path) -> Path:
    from app.pipeline.download import _set

    _set(job, status="separating", progress=0.0, stage="Separating stems...")

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        DEMUCS_MODEL,
        "-d",
        DEMUCS_DEVICE,
        "-o",
        str(job_dir),
        str(source),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,
    )
    if proc.stderr is None:
        raise RuntimeError("demucs subprocess has no stderr pipe")
    set_proc(job.id, proc)

    # tqdm uses \r to redraw -- read char-by-char and split on \r or \n.
    # Keep the last few non-progress lines so we can surface them if demucs
    # exits non-zero (otherwise the only signal would be a bare exit code).
    buf = ""
    tail: list[str] = []
    try:
        while True:
            ch = proc.stderr.read(1)
            if not ch:
                break
            if ch in ("\r", "\n"):
                line = buf.strip()
                buf = ""
                if not line:
                    continue
                m = _PCT_RE.search(line)
                if m:
                    pct = max(0, min(100, int(m.group(1))))
                    _set(job, progress=pct / 100.0, stage=f"Separating {pct}%")
                else:
                    tail.append(line)
                    if len(tail) > 40:
                        tail.pop(0)
            else:
                buf += ch

        proc.wait()
    finally:
        set_proc(job.id, None)

    # POST /cancel calls proc.terminate() directly, which causes the read loop
    # above to hit EOF and proc.wait() to return a nonzero status. Translate
    # that into JobCancelled before the generic "demucs failed" path.
    if job.cancel_requested:
        raise JobCancelled()
    if proc.returncode != 0:
        detail = "\n".join(tail[-15:]) if tail else "(no stderr captured)"
        logger.error("demucs exited %s; tail:\n%s", proc.returncode, detail)
        last = tail[-1] if tail else f"exit status {proc.returncode}"
        raise RuntimeError(f"demucs failed: {last}")

    stems_root = job_dir / DEMUCS_MODEL / source.stem
    if not stems_root.is_dir():
        raise RuntimeError(f"demucs output not found at {stems_root}")
    return stems_root

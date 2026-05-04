from __future__ import annotations

import subprocess

from app.core.models import Job

_jobs: dict[str, Job] = {}
# Active subprocesses keyed by job_id (currently only Demucs). Lets
# POST /cancel terminate the running process from the API thread instead
# of waiting for the pipeline thread to notice the cancel flag.
_procs: dict[str, subprocess.Popen] = {}


def register(job: Job) -> Job:
    _jobs[job.id] = job
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def remove(job_id: str) -> None:
    _jobs.pop(job_id, None)
    _procs.pop(job_id, None)


def all_jobs() -> dict[str, Job]:
    """Return a snapshot of the registry for sweep / cleanup."""
    return dict(_jobs)


def set_proc(job_id: str, proc: subprocess.Popen | None) -> None:
    if proc is None:
        _procs.pop(job_id, None)
    else:
        _procs[job_id] = proc


def get_proc(job_id: str) -> subprocess.Popen | None:
    return _procs.get(job_id)

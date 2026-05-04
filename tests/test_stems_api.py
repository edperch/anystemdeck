from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import JOBS_DIR
from app.core.models import Job
from app.core.registry import _jobs


@pytest.fixture(autouse=True)
def _isolate_registry():
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def _make_stem_file(job_id: str, name: str, contents: bytes = b"RIFF") -> Path:
    stems_dir = JOBS_DIR / job_id / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    path = stems_dir / f"{name}.wav"
    path.write_bytes(contents)
    return path


def test_rejects_malformed_job_id(client):
    # Anything that isn't 12 lowercase hex chars must be rejected before
    # touching the filesystem -- this is the path-traversal gate.
    for bad_id in ("../etc", "ABC", "abcdefabcdef0", "abcdefabcde", "abcd-efabcdef"):
        r = client.get(f"/api/jobs/{bad_id}/stems/vocals.wav")
        assert r.status_code == 404, f"id {bad_id!r} should 404"


def test_rejects_unknown_stem_name(client):
    job = Job(id="abcdefabcdef")
    job.status = "done"
    _jobs[job.id] = job
    r = client.get(f"/api/jobs/{job.id}/stems/banjo.wav")
    assert r.status_code == 404


def test_requires_done_status(client):
    job = Job(id="abcdefabcdef")
    job.status = "separating"
    _jobs[job.id] = job
    _make_stem_file(job.id, "vocals")
    try:
        r = client.get(f"/api/jobs/{job.id}/stems/vocals.wav")
        assert r.status_code == 404
    finally:
        (JOBS_DIR / job.id / "stems" / "vocals.wav").unlink(missing_ok=True)
        (JOBS_DIR / job.id / "stems").rmdir()
        (JOBS_DIR / job.id).rmdir()


def test_serves_done_job_stem(client):
    job = Job(id="abcdefabcdee")
    job.status = "done"
    _jobs[job.id] = job
    path = _make_stem_file(job.id, "vocals", b"RIFF1234")
    try:
        r = client.get(f"/api/jobs/{job.id}/stems/vocals.wav")
        assert r.status_code == 200
        assert r.content == b"RIFF1234"
        assert r.headers["content-type"] == "audio/wav"
    finally:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
        path.parent.parent.rmdir()

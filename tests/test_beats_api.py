from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core.models import Job
from app.core.registry import _jobs

JOB = "abcdefabcdef"


@pytest.fixture(autouse=True)
def _isolate_registry():
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.api import jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "JOBS_DIR", tmp_path)
    from app.main import app

    return TestClient(app)


@pytest.fixture
def done_job(tmp_path):
    job = Job(id=JOB)
    job.status = "done"
    _jobs[job.id] = job
    (tmp_path / JOB / "stems").mkdir(parents=True)
    return job


def _write_computed(tmp_path, **over):
    grid = {
        "version": 1,
        "source": "drums",
        "bpm": 120.0,
        "duration": 10.0,
        "confidence": 90,
        "beats": [0.0, 0.5, 1.0, 1.5, 2.0],
        "onsets": [0.0, 0.5, 1.0],
    }
    grid.update(over)
    (tmp_path / JOB / "stems" / "beats.json").write_text(json.dumps(grid), encoding="utf-8")
    return grid


# --- GET -------------------------------------------------------------------


def test_get_returns_computed_grid(client, tmp_path, done_job):
    grid = _write_computed(tmp_path)
    r = client.get(f"/api/jobs/{JOB}/beats")
    assert r.status_code == 200
    body = r.json()
    assert body["beats"] == grid["beats"]
    assert body["edited"] is False
    assert body["bars"] == []


def test_get_is_never_cached(client, tmp_path, done_job):
    """The computed artifact is immutable and cached forever; this endpoint
    changes on every edit and must not be."""
    _write_computed(tmp_path)
    r = client.get(f"/api/jobs/{JOB}/beats")
    assert r.headers.get("cache-control") == "no-store"


def test_get_prefers_user_edits(client, tmp_path, done_job):
    _write_computed(tmp_path)
    (tmp_path / JOB / "stems" / "beats.user.json").write_text(
        json.dumps({"beats": [0.1, 0.7, 1.3], "bars": [{"beat": 0, "beats_per_bar": 3}]}),
        encoding="utf-8",
    )
    body = client.get(f"/api/jobs/{JOB}/beats").json()
    assert body["beats"] == [0.1, 0.7, 1.3]
    assert body["bars"] == [{"beat": 0, "beats_per_bar": 3}]
    assert body["edited"] is True


def test_edits_keep_computed_onsets(client, tmp_path, done_job):
    """Editing beats never changes where the transients are, and the editor
    still needs them for snapping."""
    _write_computed(tmp_path)
    (tmp_path / JOB / "stems" / "beats.user.json").write_text(
        json.dumps({"beats": [0.1, 0.7]}), encoding="utf-8"
    )
    body = client.get(f"/api/jobs/{JOB}/beats").json()
    assert body["onsets"] == [0.0, 0.5, 1.0]


def test_get_ignores_corrupt_edits(client, tmp_path, done_job):
    """A truncated edit file must degrade to the detected grid, not 500."""
    _write_computed(tmp_path)
    (tmp_path / JOB / "stems" / "beats.user.json").write_text("{not json", encoding="utf-8")
    body = client.get(f"/api/jobs/{JOB}/beats").json()
    assert body["edited"] is False
    assert body["beats"] == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_get_ignores_empty_edit_list(client, tmp_path, done_job):
    _write_computed(tmp_path)
    (tmp_path / JOB / "stems" / "beats.user.json").write_text(
        json.dumps({"beats": []}), encoding="utf-8"
    )
    assert client.get(f"/api/jobs/{JOB}/beats").json()["edited"] is False


def test_get_404s_without_a_computed_grid(client, tmp_path, done_job):
    assert client.get(f"/api/jobs/{JOB}/beats").status_code == 404


def test_get_requires_done_status(client, tmp_path):
    job = Job(id=JOB)
    job.status = "separating"
    _jobs[job.id] = job
    (tmp_path / JOB / "stems").mkdir(parents=True)
    _write_computed(tmp_path)
    assert client.get(f"/api/jobs/{JOB}/beats").status_code == 404


@pytest.mark.parametrize("bad", ["../etc", "ABC", "abcdefabcdef0", "abcd-efabcdef"])
def test_get_rejects_malformed_job_id(client, bad):
    assert client.get(f"/api/jobs/{bad}/beats").status_code == 404


# --- PATCH -----------------------------------------------------------------


def test_patch_persists_edits(client, tmp_path, done_job):
    _write_computed(tmp_path)
    r = client.patch(
        f"/api/jobs/{JOB}/beats",
        json={"beats": [0.2, 0.9, 1.4], "bars": [{"beat": 0, "beats_per_bar": 7}]},
    )
    assert r.status_code == 200
    assert r.json()["beats"] == 3

    body = client.get(f"/api/jobs/{JOB}/beats").json()
    assert body["beats"] == [0.2, 0.9, 1.4]
    assert body["bars"] == [{"beat": 0, "beats_per_bar": 7}]
    assert body["edited"] is True


def test_patch_writes_a_separate_file_from_the_computed_grid(client, tmp_path, done_job):
    """Re-running analysis overwrites beats.json; edits must survive that."""
    grid = _write_computed(tmp_path)
    client.patch(f"/api/jobs/{JOB}/beats", json={"beats": [0.2, 0.9]})
    computed = json.loads((tmp_path / JOB / "stems" / "beats.json").read_text())
    assert computed["beats"] == grid["beats"], "computed grid must be untouched"
    assert (tmp_path / JOB / "stems" / "beats.user.json").is_file()


def test_patch_leaves_no_temp_file(client, tmp_path, done_job):
    _write_computed(tmp_path)
    client.patch(f"/api/jobs/{JOB}/beats", json={"beats": [0.2, 0.9]})
    leftovers = list((tmp_path / JOB / "stems").glob("*.tmp"))
    assert leftovers == []


def test_patch_rejects_unsorted_beats(client, tmp_path, done_job):
    _write_computed(tmp_path)
    r = client.patch(f"/api/jobs/{JOB}/beats", json={"beats": [1.0, 0.5, 2.0]})
    assert r.status_code == 422


def test_patch_rejects_duplicate_beats(client, tmp_path, done_job):
    _write_computed(tmp_path)
    r = client.patch(f"/api/jobs/{JOB}/beats", json={"beats": [1.0, 1.0]})
    assert r.status_code == 422


@pytest.mark.parametrize("bad", ["-1.0", "90000.0", "Infinity", "-Infinity", "NaN"])
def test_patch_rejects_out_of_range_beats(client, tmp_path, done_job, bad):
    """Infinity/NaN are not standard JSON but Python's parser accepts them, so
    a hostile client can get them as far as the validator. It must reject them
    rather than persist a grid the scheduler would binary-search into a hang."""
    _write_computed(tmp_path)
    r = client.patch(
        f"/api/jobs/{JOB}/beats",
        content=f'{{"beats": [{bad}]}}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_patch_rejects_absurd_beat_count(client, tmp_path, done_job):
    _write_computed(tmp_path)
    r = client.patch(f"/api/jobs/{JOB}/beats", json={"beats": [i * 0.001 for i in range(20001)]})
    assert r.status_code == 422


@pytest.mark.parametrize("bpb", [0, 33, -1])
def test_patch_rejects_implausible_bar_length(client, tmp_path, done_job, bpb):
    _write_computed(tmp_path)
    r = client.patch(
        f"/api/jobs/{JOB}/beats",
        json={"beats": [0.0, 0.5], "bars": [{"beat": 0, "beats_per_bar": bpb}]},
    )
    assert r.status_code == 422


def test_patch_404s_without_a_computed_grid(client, tmp_path, done_job):
    r = client.patch(f"/api/jobs/{JOB}/beats", json={"beats": [0.2, 0.9]})
    assert r.status_code == 404


@pytest.mark.parametrize("bad", ["ABC", "abcdefabcdef0", "abcd-efabcdef"])
def test_patch_rejects_malformed_job_id(client, bad):
    r = client.patch(f"/api/jobs/{bad}/beats", json={"beats": [0.2]})
    assert r.status_code == 404


# --- DELETE ----------------------------------------------------------------


def test_delete_reverts_to_detected_grid(client, tmp_path, done_job):
    grid = _write_computed(tmp_path)
    client.patch(f"/api/jobs/{JOB}/beats", json={"beats": [0.2, 0.9]})
    assert client.get(f"/api/jobs/{JOB}/beats").json()["edited"] is True

    r = client.delete(f"/api/jobs/{JOB}/beats")
    assert r.status_code == 200

    body = client.get(f"/api/jobs/{JOB}/beats").json()
    assert body["edited"] is False
    assert body["beats"] == grid["beats"]


def test_delete_is_idempotent(client, tmp_path, done_job):
    _write_computed(tmp_path)
    # The requests are made outside the assert: `python -O` strips asserts, and
    # a call hidden inside one would silently stop happening.
    first = client.delete(f"/api/jobs/{JOB}/beats")
    second = client.delete(f"/api/jobs/{JOB}/beats")
    assert first.status_code == 200
    assert second.status_code == 200


def test_delete_rejects_malformed_job_id(client):
    resp = client.delete("/api/jobs/ABC/beats")
    assert resp.status_code == 404

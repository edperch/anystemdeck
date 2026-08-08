from __future__ import annotations

import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.models import Job
from app.core.registry import _jobs
from app.pipeline.click_render import (
    ACCENT_AUTO,
    ACCENT_FREQ,
    ACCENT_OFF,
    CLICK_FREQ,
    cache_key,
    is_downbeat,
    render_click_wav,
    rescale_beats,
    source_index,
)

JOB = "abcdefabcdef"
SR = 44100


def _read(path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767.0
    return data, sr


def _click_starts(y, sr, threshold=0.2, min_gap=0.04):
    """Onset time of each click.

    A click is a 1 kHz sine under a decaying envelope, so it crosses any
    amplitude threshold about ten times per click. Grouping by a refractory
    gap longer than one click (35 ms) collapses those crossings back to one
    onset each.

    The reported time lags the true start by up to a quarter cycle plus the
    1 ms attack -- roughly 1.3 ms at 1 kHz -- which sets the measurement floor
    for the assertions below.
    """
    loud = np.flatnonzero(np.abs(y) > threshold)
    if loud.size == 0:
        return np.array([])
    keep = [loud[0]]
    for i in loud[1:]:
        if i - keep[-1] > min_gap * sr:
            keep.append(i)
    return np.array(keep) / sr


def _dominant_freq(y, sr, at, dur=0.01):
    seg = y[int(at * sr) : int((at + dur) * sr)]
    if len(seg) < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return float(np.fft.rfftfreq(len(seg), 1 / sr)[int(np.argmax(spec))])


# --- rescale / index mapping ----------------------------------------------


def test_rescale_identity():
    beats = [0.0, 0.5, 1.0]
    assert rescale_beats(beats, 1.0) == beats


def test_rescale_doubles_by_inserting_midpoints():
    assert rescale_beats([0.0, 0.5, 1.0], 2.0) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_rescale_halves_by_taking_every_other():
    assert rescale_beats([0.0, 0.5, 1.0, 1.5, 2.0], 0.5) == [0.0, 1.0, 2.0]


def test_source_index_maps_back_to_the_detected_grid():
    """Bar marks index the detected beats, so accents must be decided there.
    Without this the accent lands on the wrong beat at any rate but 1x."""
    assert [source_index(i, 2.0) for i in range(5)] == [0, None, 1, None, 2]
    assert [source_index(i, 0.5) for i in range(3)] == [0, 2, 4]
    assert [source_index(i, 1.0) for i in range(3)] == [0, 1, 2]


# --- accent decisions ------------------------------------------------------


def test_accent_off_never_accents():
    bars = [{"beat": 0, "beats_per_bar": 4}]
    assert not any(is_downbeat(i, bars, ACCENT_OFF) for i in range(8))


def test_accent_fixed_count():
    assert [is_downbeat(i, [], 3) for i in range(6)] == [True, False, False, True, False, False]


def test_accent_auto_follows_bar_marks():
    bars = [{"beat": 2, "beats_per_bar": 4}]
    assert [is_downbeat(i, bars, ACCENT_AUTO) for i in range(2, 11)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


def test_accent_auto_handles_a_meter_change():
    bars = [{"beat": 0, "beats_per_bar": 4}, {"beat": 8, "beats_per_bar": 3}]
    assert is_downbeat(8, bars, ACCENT_AUTO)
    assert is_downbeat(11, bars, ACCENT_AUTO)
    assert not is_downbeat(12, bars, ACCENT_AUTO)


def test_accent_auto_without_marks_is_silent():
    assert not is_downbeat(0, [], ACCENT_AUTO)


def test_accent_ignores_inserted_midpoints():
    """At x2 the odd entries belong to no detected beat and cannot be downbeats."""
    assert not is_downbeat(source_index(1, 2.0), [{"beat": 0, "beats_per_bar": 4}], ACCENT_AUTO)


# --- rendering -------------------------------------------------------------


def test_render_places_a_click_on_every_beat(tmp_path):
    beats = [0.5 + i * 0.5 for i in range(10)]
    out = render_click_wav(tmp_path / "c.wav", beats, [], duration=7.0)
    y, sr = _read(out)
    starts = _click_starts(y, sr)
    assert len(starts) == len(beats)
    # 3 ms covers the threshold-crossing lag described in _click_starts; a
    # misplaced click would be out by a whole beat, not by milliseconds.
    assert np.abs(starts - np.array(beats)).max() < 0.003, "clicks must land on the beats"


def test_render_spans_the_whole_track_even_when_beats_start_late(tmp_path):
    """The export's region trim puts -ss before every input, so the click has to
    cover the full duration for the offsets to line up with the stems."""
    out = render_click_wav(tmp_path / "c.wav", [5.0, 5.5, 6.0], [], duration=12.0)
    y, sr = _read(out)
    assert abs(len(y) / sr - 12.0) < 0.01


def test_render_accents_are_higher_pitched_and_louder(tmp_path):
    beats = [0.5 + i * 0.5 for i in range(8)]
    out = render_click_wav(
        tmp_path / "c.wav", beats, [{"beat": 0, "beats_per_bar": 4}], duration=6.0
    )
    y, sr = _read(out)
    assert abs(_dominant_freq(y, sr, beats[0]) - ACCENT_FREQ) < 120
    assert abs(_dominant_freq(y, sr, beats[1]) - CLICK_FREQ) < 120
    peak_at = lambda t: np.abs(y[int(t * sr) : int((t + 0.03) * sr)]).max()  # noqa: E731
    assert peak_at(beats[0]) > peak_at(beats[1])


def test_render_respects_the_rate_multiplier(tmp_path):
    beats = [0.5 + i * 0.5 for i in range(8)]
    doubled = render_click_wav(tmp_path / "d.wav", beats, [], duration=6.0, multiplier=2.0)
    halved = render_click_wav(tmp_path / "h.wav", beats, [], duration=6.0, multiplier=0.5)
    yd, sr = _read(doubled)
    yh, _ = _read(halved)
    assert len(_click_starts(yd, sr)) == 15
    assert len(_click_starts(yh, sr)) == 4


def test_render_accent_mode_off_produces_one_voice(tmp_path):
    beats = [0.5 + i * 0.5 for i in range(8)]
    out = render_click_wav(
        tmp_path / "c.wav",
        beats,
        [{"beat": 0, "beats_per_bar": 4}],
        duration=6.0,
        accent_mode=ACCENT_OFF,
    )
    y, sr = _read(out)
    freqs = [_dominant_freq(y, sr, b) for b in beats]
    assert all(abs(f - CLICK_FREQ) < 120 for f in freqs), freqs


def test_render_returns_none_without_beats(tmp_path):
    assert render_click_wav(tmp_path / "c.wav", [], [], duration=5.0) is None


def test_render_returns_none_for_zero_duration(tmp_path):
    assert render_click_wav(tmp_path / "c.wav", [0.0, 0.5], [], duration=0.0) is None


def test_render_never_clips(tmp_path):
    """Clicks overlap at very fast tempos; the sum must stay in range."""
    beats = [0.1 + i * 0.01 for i in range(200)]
    out = render_click_wav(tmp_path / "c.wav", beats, [], duration=4.0)
    y, _ = _read(out)
    assert np.abs(y).max() <= 1.0


def test_render_leaves_no_temp_file(tmp_path):
    render_click_wav(tmp_path / "c.wav", [0.5, 1.0, 1.5], [], duration=3.0)
    assert list(tmp_path.glob("*.tmp")) == []


def test_render_is_deterministic(tmp_path):
    beats = [0.5 + i * 0.5 for i in range(6)]
    a = render_click_wav(tmp_path / "a.wav", beats, [], duration=5.0)
    b = render_click_wav(tmp_path / "b.wav", beats, [], duration=5.0)
    assert a.read_bytes() == b.read_bytes()


# --- cache key -------------------------------------------------------------


def test_cache_key_separates_edited_grids():
    """An edited grid must not hit an entry rendered from the detected one."""
    a = cache_key(JOB, [0.0, 0.5], [], 5.0, SR, 1.0, ACCENT_AUTO)
    b = cache_key(JOB, [0.0, 0.6], [], 5.0, SR, 1.0, ACCENT_AUTO)
    assert a != b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"multiplier": 2.0},
        {"accent_mode": 4},
        {"sample_rate": 48000},
        {"bars": [{"beat": 0, "beats_per_bar": 3}]},
    ],
)
def test_cache_key_covers_every_render_input(kwargs):
    base = {
        "job_id": JOB,
        "beats": [0.0, 0.5],
        "bars": [],
        "duration": 5.0,
        "sample_rate": SR,
        "multiplier": 1.0,
        "accent_mode": ACCENT_AUTO,
    }
    assert cache_key(**base) != cache_key(**{**base, **kwargs})


# --- export endpoint -------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_registry():
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.api import stems as stems_mod

    monkeypatch.setattr(stems_mod, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(stems_mod, "_MIXDOWN_CACHE_DIR", tmp_path / "cache" / "mixdown")
    monkeypatch.setattr(stems_mod, "_CLICK_CACHE_DIR", tmp_path / "cache" / "click")
    from app.main import app

    return TestClient(app)


def _setup_job(tmp_path, with_grid=True):
    job = Job(id=JOB)
    job.status = "done"
    _jobs[job.id] = job
    stems = tmp_path / JOB / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    if with_grid:
        (stems / "beats.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "beats": [0.5, 1.0, 1.5, 2.0],
                    "bars": [{"beat": 0, "beats_per_bar": 4}],
                    "duration": 3.0,
                }
            ),
            encoding="utf-8",
        )
    return stems


def test_click_lane_is_off_by_default(client, tmp_path):
    from app.api import stems as stems_mod

    _setup_job(tmp_path)
    assert stems_mod._click_lane(JOB, False, 1.0, ACCENT_AUTO, 0.6) is None


def test_click_lane_renders_when_enabled(client, tmp_path):
    from app.api import stems as stems_mod

    _setup_job(tmp_path)
    lane = stems_mod._click_lane(JOB, True, 1.0, ACCENT_AUTO, 0.6)
    assert lane is not None
    path, gain = lane
    assert path.is_file()
    assert gain == pytest.approx(0.6)


def test_click_lane_is_none_without_a_beat_grid(client, tmp_path):
    from app.api import stems as stems_mod

    _setup_job(tmp_path, with_grid=False)
    assert stems_mod._click_lane(JOB, True, 1.0, ACCENT_AUTO, 0.6) is None


def test_click_lane_prefers_user_edits(client, tmp_path):
    from app.api import stems as stems_mod

    stems = _setup_job(tmp_path)
    detected = stems_mod._click_lane(JOB, True, 1.0, ACCENT_AUTO, 0.6)
    (stems / "beats.user.json").write_text(
        json.dumps({"beats": [0.25, 0.75, 1.25], "bars": []}), encoding="utf-8"
    )
    edited = stems_mod._click_lane(JOB, True, 1.0, ACCENT_AUTO, 0.6)
    assert edited is not None
    assert edited[0].name != detected[0].name, "edited grid must render its own click"


def test_click_gain_is_clamped(client, tmp_path):
    from app.api import stems as stems_mod

    _setup_job(tmp_path)
    assert stems_mod._click_lane(JOB, True, 1.0, ACCENT_AUTO, 99.0)[1] == 4.0
    assert stems_mod._click_lane(JOB, True, 1.0, ACCENT_AUTO, -5.0)[1] == 0.0


def test_mixdown_cache_key_separates_click_from_clean(client, tmp_path):
    from app.api import stems as stems_mod

    _setup_job(tmp_path)
    lane = stems_mod._click_lane(JOB, True, 1.0, ACCENT_AUTO, 0.6)
    clean = stems_mod._mixdown_cache_key(JOB, "wav", ["drums"], [1.0], None, None, None)
    clicked = stems_mod._mixdown_cache_key(JOB, "wav", ["drums"], [1.0], None, None, lane)
    assert clean != clicked, "a click export must never reuse a clean render"


@pytest.mark.parametrize("bad", ["click_accent=99", "click_accent=-2"])
def test_mixdown_rejects_out_of_range_click_params(client, tmp_path, bad):
    _setup_job(tmp_path)
    (tmp_path / JOB / "stems" / "drums.wav").write_bytes(b"RIFF")
    r = client.get(f"/api/jobs/{JOB}/mixdown.wav?stems=drums&gains=1.0&click=1&{bad}")
    assert r.status_code == 422

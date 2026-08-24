from __future__ import annotations

import numpy as np
import pytest

from app.core.config import BEATGRID_HOP, BEATGRID_SR
from app.pipeline import beat_detect as bd
from app.pipeline.beatgrid import (
    _correct_grid_phase,
    _downbeats_to_bars,
    _fill_interior_gaps,
    _interval_spread,
    _onset_support,
)


@pytest.fixture(autouse=True)
def _reset_model():
    """The detector caches its model and its failure flag process-wide."""
    bd.reset_model_cache()
    yield
    bd.reset_model_cache()


def _onset_env(hit_times, seconds=10.0, sr=BEATGRID_SR):
    """A fake fine-resolution onset envelope with a spike at each hit."""
    from app.core.config import BEATGRID_REFINE_HOP

    frames = int(seconds * sr / BEATGRID_REFINE_HOP)
    env = np.zeros(frames)
    for t in hit_times:
        f = int(round(t * sr / BEATGRID_REFINE_HOP))
        if 0 <= f < frames:
            env[f] = 1.0
    return env


# --- detector selection ----------------------------------------------------


def test_librosa_forced_skips_the_model(monkeypatch):
    """STEMDECK_BEAT_DETECTOR=librosa must not even attempt to load weights."""
    monkeypatch.setattr(bd, "BEAT_DETECTOR", "librosa")
    called = []
    monkeypatch.setattr(bd, "_get_model", lambda: called.append(1) or None)

    y = np.zeros(BEATGRID_SR * 2, dtype=np.float32)
    env = np.zeros(int(BEATGRID_SR * 2 / BEATGRID_HOP))
    _beats, downbeats, name = bd.detect(y, BEATGRID_SR, env)
    assert name == "librosa"
    assert downbeats == [], "librosa has no downbeat notion"
    assert called == [], "model must not be consulted when librosa is forced"


def test_auto_falls_back_when_the_model_is_unavailable(monkeypatch):
    """A fresh offline install has no weights; the click must still work."""
    monkeypatch.setattr(bd, "BEAT_DETECTOR", "auto")
    monkeypatch.setattr(bd, "_get_model", lambda: None)

    y = np.zeros(BEATGRID_SR * 2, dtype=np.float32)
    env = np.zeros(int(BEATGRID_SR * 2 / BEATGRID_HOP))
    _beats, _db, name = bd.detect(y, BEATGRID_SR, env)
    assert name == "librosa"


def test_model_mode_refuses_to_degrade_silently(monkeypatch):
    """Explicitly requiring the model must fail loudly, not quietly fall back --
    that mode exists to diagnose packaging problems."""
    monkeypatch.setattr(bd, "BEAT_DETECTOR", "model")
    monkeypatch.setattr(bd, "_get_model", lambda: None)

    y = np.zeros(BEATGRID_SR, dtype=np.float32)
    env = np.zeros(int(BEATGRID_SR / BEATGRID_HOP))
    with pytest.raises(RuntimeError, match="beat model required"):
        bd.detect(y, BEATGRID_SR, env)


def test_model_inference_failure_falls_back(monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("cuda exploded")

    monkeypatch.setattr(bd, "BEAT_DETECTOR", "auto")
    monkeypatch.setattr(bd, "_get_model", lambda: _boom)
    y = np.zeros(BEATGRID_SR * 2, dtype=np.float32)
    env = np.zeros(int(BEATGRID_SR * 2 / BEATGRID_HOP))
    _b, _d, name = bd.detect(y, BEATGRID_SR, env)
    assert name == "librosa"


def test_model_returning_too_few_beats_falls_back(monkeypatch):
    monkeypatch.setattr(bd, "BEAT_DETECTOR", "auto")
    monkeypatch.setattr(bd, "_get_model", lambda: lambda _y, _sr: ([1.0], []))
    y = np.zeros(BEATGRID_SR * 2, dtype=np.float32)
    env = np.zeros(int(BEATGRID_SR * 2 / BEATGRID_HOP))
    _b, _d, name = bd.detect(y, BEATGRID_SR, env)
    assert name == "librosa"


def test_model_result_is_used_when_available(monkeypatch):
    monkeypatch.setattr(bd, "BEAT_DETECTOR", "auto")
    monkeypatch.setattr(bd, "_get_model", lambda: lambda _y, _sr: ([0.5, 1.0, 1.5], [0.5, 1.5]))
    y = np.zeros(BEATGRID_SR * 2, dtype=np.float32)
    env = np.zeros(int(BEATGRID_SR * 2 / BEATGRID_HOP))
    beats, downbeats, name = bd.detect(y, BEATGRID_SR, env)
    assert name == "beat_this"
    assert beats == [0.5, 1.0, 1.5]
    assert downbeats == [0.5, 1.5]


def test_model_load_failure_is_cached(monkeypatch):
    """Retrying a failed load per job would stall every job on the same
    network timeout."""
    attempts = []

    def _fail(*_a, **_k):
        attempts.append(1)
        raise OSError("no network")

    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **k):
        if name == "beat_this.inference":
            _fail()
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert bd._get_model() is None
    assert bd._get_model() is None
    assert len(attempts) == 1, "load must only be attempted once"


# --- _interval_spread ------------------------------------------------------


def test_spread_is_zero_for_a_perfect_grid():
    assert _interval_spread([i * 0.5 for i in range(20)]) == pytest.approx(0.0, abs=1e-9)


def test_spread_ignores_a_single_large_hole():
    """The whole reason IQR is used instead of the coefficient of variation: one
    drum-free section must not disqualify a track whose pulse never wavers."""
    beats = [i * 0.5 for i in range(20)]
    beats += [beats[-1] + 5.0 + i * 0.5 for i in range(20)]  # 5 s hole
    spread = _interval_spread(beats)
    arr = np.diff(np.asarray(beats))
    cv = float(arr.std() / np.median(arr))
    assert spread < 0.05, f"IQR/median should stay small, got {spread:.3f}"
    assert cv > 0.5, "cv is the measure that would wrongly reject this grid"


def test_spread_reports_genuine_irregularity():
    rng = np.random.default_rng(0)
    beats = np.cumsum(rng.uniform(0.25, 0.75, 40)).tolist()
    assert _interval_spread(beats) > 0.2


def test_spread_handles_short_input():
    assert _interval_spread([1.0, 2.0]) == 0.0


# --- _fill_interior_gaps ---------------------------------------------------


def test_gap_fill_subdivides_a_clean_multiple():
    beats = [i * 0.5 for i in range(10)]
    beats += [beats[-1] + 5.0 + i * 0.5 for i in range(10)]  # 10-beat hole
    out, inserted = _fill_interior_gaps(beats)
    assert inserted == 9
    iv = np.diff(np.asarray(out))
    assert iv.max() == pytest.approx(0.5, abs=0.01)


def test_gap_fill_scales_its_tolerance_with_gap_size():
    """A long gap accumulates many beats' worth of period error, so the residual
    has to be judged per inserted beat. Judged on the whole gap, a 15-beat hole
    was rejected and Welcome To Paradise kept a 5 s silent stretch."""
    period = 0.34
    beats = [i * period for i in range(10)]
    beats += [beats[-1] + period * 15.5 + i * period for i in range(10)]
    _out, inserted = _fill_interior_gaps(beats)
    assert inserted > 10, f"large gap should still fill, inserted {inserted}"


def test_gap_fill_leaves_a_non_multiple_alone():
    beats = [i * 0.5 for i in range(10)]
    beats.append(beats[-1] + 0.77)  # not a multiple of the period
    beats += [beats[-1] + 0.5 * i for i in range(1, 10)]
    _out, inserted = _fill_interior_gaps(beats)
    assert inserted == 0


def test_gap_fill_no_op_on_a_regular_grid():
    beats = [i * 0.5 for i in range(30)]
    out, inserted = _fill_interior_gaps(beats)
    assert inserted == 0
    assert out == pytest.approx(beats)


# --- phase correction ------------------------------------------------------


def test_phase_shift_rescues_an_off_beat_grid():
    """Right tempo, wrong half of the beat: every click lands in a gap."""
    hits = [0.5 + i * 0.5 for i in range(20)]
    env = _onset_env(hits)
    off_beat = [t + 0.25 for t in hits]
    out, shifted = _correct_grid_phase(off_beat, env, BEATGRID_SR, duration=10.0)
    assert shifted is True
    assert _onset_support(env, BEATGRID_SR, out) > _onset_support(env, BEATGRID_SR, off_beat)


def test_phase_shift_leaves_a_correct_grid_alone():
    hits = [0.5 + i * 0.5 for i in range(20)]
    env = _onset_env(hits)
    out, shifted = _correct_grid_phase(list(hits), env, BEATGRID_SR, duration=10.0)
    assert shifted is False
    assert out == pytest.approx(hits)


def test_phase_shift_does_not_flip_on_off_beat_hats():
    """Real playing puts hi-hats on the eighths, so the wrong phase always has
    *some* support. Only a large improvement may flip the grid."""
    beats = [0.5 + i * 0.5 for i in range(20)]
    env = _onset_env(beats)
    for t in beats:  # weaker off-beat hats
        f = int(round((t + 0.25) * BEATGRID_SR / 32))
        if f < env.size:
            env[f] = 0.4
    _out, shifted = _correct_grid_phase(beats, env, BEATGRID_SR, duration=10.0)
    assert shifted is False


def test_phase_shift_needs_enough_beats():
    env = _onset_env([0.5, 1.0])
    out, shifted = _correct_grid_phase([0.5, 1.0], env, BEATGRID_SR, duration=10.0)
    assert shifted is False
    assert out == [0.5, 1.0]


# --- downbeats to bar marks ------------------------------------------------


def test_bars_collapse_steady_four_four_to_one_mark():
    beats = [i * 0.5 for i in range(41)]
    downbeats = [beats[i] for i in range(0, 41, 4)]
    bars = _downbeats_to_bars(beats, downbeats)
    assert bars == [{"beat": 0, "beats_per_bar": 4}]


def test_bars_detect_three_four():
    beats = [i * 0.5 for i in range(31)]
    downbeats = [beats[i] for i in range(0, 31, 3)]
    bars = _downbeats_to_bars(beats, downbeats)
    assert bars == [{"beat": 0, "beats_per_bar": 3}]


def test_bars_ignore_a_single_slipped_downbeat():
    """One spurious or missed downbeat must not rewrite the meter -- raw output
    produced 13 marks with lengths 1, 4, 3, 6 on a plain 4/4 punk track."""
    beats = [i * 0.5 for i in range(41)]
    idx = list(range(0, 41, 4))
    idx.remove(16)  # a missed downbeat: one 8-beat span
    bars = _downbeats_to_bars(beats, [beats[i] for i in idx])
    assert bars == [{"beat": 0, "beats_per_bar": 4}]


def test_bars_record_a_sustained_meter_change():
    beats = [i * 0.5 for i in range(61)]
    idx = list(range(0, 32, 4)) + list(range(32, 61, 7))
    bars = _downbeats_to_bars(beats, [beats[i] for i in idx])
    lengths = [b["beats_per_bar"] for b in bars]
    assert 4 in lengths and 7 in lengths, f"expected a 4 -> 7 change, got {bars}"


def test_bars_discard_implausible_spacings():
    beats = [i * 0.5 for i in range(41)]
    # Adjacent downbeats (a one-beat "bar") are always detector noise.
    bars = _downbeats_to_bars(beats, [beats[0], beats[1], beats[2], beats[3]])
    assert bars == []


def test_bars_empty_without_downbeats():
    assert _downbeats_to_bars([i * 0.5 for i in range(20)], []) == []

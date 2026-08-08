"""Full-length beat grid extraction.

`analyze.py` estimates a single global BPM from the first 180 s of the source
mix -- fine for a metadata chip, useless for a click track: a fixed period
extrapolated from a truncated, drum-plus-everything-else signal drifts audibly
long before a song ends.

This stage runs *after* separation instead, so it can track beats on the
isolated drums stem over the entire track and persist the actual beat times.
The frontend clicks at those times, so whatever tempo drift the tracker
followed is reproduced exactly rather than approximated.

Output is `stems/beats.json`; failure is non-fatal and simply leaves the click
track unavailable for that job (the endpoint 404s and the UI hides the
control).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.core.config import (
    BEATGRID_GAP_FILL_MAX_RESIDUAL,
    BEATGRID_GAP_LOCAL_WINDOW,
    BEATGRID_HOP,
    BEATGRID_MAX_IRREGULARITY,
    BEATGRID_MAX_ONSETS,
    BEATGRID_ONSET_DELTA_MULT,
    BEATGRID_ONSET_PERCENTILE,
    BEATGRID_ONSET_TOL_FRAC,
    BEATGRID_ONSET_TOL_MAX,
    BEATGRID_PHASE_FLIP_RATIO,
    BEATGRID_REFINE_HOP,
    BEATGRID_REFINE_MAX_INTERP_FRAC,
    BEATGRID_REFINE_MELS,
    BEATGRID_REFINE_MOVE_MAX,
    BEATGRID_REFINE_MOVE_MIN,
    BEATGRID_REFINE_MOVE_OF_BEAT,
    BEATGRID_REFINE_NFFT,
    BEATGRID_REFINE_PERCENTILE,
    BEATGRID_REFINE_WINDOW,
    BEATGRID_REFINE_WINDOW_MAX,
    BEATGRID_REFINE_WINDOW_OF_BEAT,
    BEATGRID_SR,
    TIMEOUT_BEATGRID,
)
from app.pipeline.analyze import _load_audio_ffmpeg
from app.pipeline.beat_detect import detect as detect_beats

logger = logging.getLogger("stemdeck.beatgrid")

# Beat tracking wants percussive transients. The isolated drums stem is by far
# the cleanest onset envelope available; the full mix is a distant fallback for
# drumless material, where the tracker latches onto whatever else is rhythmic.
_SOURCE_PREFERENCE = ("drums", "original", "mix")

# A grid with fewer than this many beats is not a grid -- it's noise. Clicking
# along to it would be worse than offering no click at all.
_MIN_BEATS = 8

# Sanity window on the derived tempo. Anything outside is a detection failure
# (usually silence or a pathological onset envelope), not an exotic song.
_MIN_BPM = 30.0
_MAX_BPM = 300.0

# How far a beat may sit from the position its neighbours predict before it is
# treated as tracker error and pulled back, as a fraction of a beat. Swept
# against synthetic grids: 0.10 fires exactly once across five tempos -- on the
# one genuinely broken beat -- and never otherwise, while 0.15 and above miss it
# entirely. Well clear of human microtiming, which lives inside ~30 ms.
_GRID_OUTLIER_FRAC = 0.10

# The tracker only places beats where there is onset evidence, so a grid
# typically starts when the drums enter and stops when they drop out -- on the
# test track, 8.9 s into a 224 s song and ending 15 s early. A metronome that
# stays silent through the intro and outro reads as broken, so a steady grid is
# extended outward at its own edge tempo to cover the whole track.
#
# Number of edge intervals averaged to get the local tempo to extend by. Enough
# to be robust to one bad interval, short enough to track a song that drifts.
_EXTRAPOLATE_EDGE_BEATS = 9

# Plausible bar lengths. 1 is always noise (two adjacent downbeat frames firing);
# past 16 it is a missed downbeat rather than an exotic meter.
_MIN_BAR = 2
_MAX_BAR = 16
# A differing bar length must repeat this many times before it is treated as a
# real meter change rather than a detector slip.
_BAR_CHANGE_PERSISTENCE = 2


def _pick_source(stems_dir: Path) -> tuple[Path, str] | None:
    """First existing stem in preference order, as (path, label)."""
    for name in _SOURCE_PREFERENCE:
        candidate = stems_dir / f"{name}.wav"
        if candidate.is_file():
            return candidate, name
    return None


def _sanitize(beat_times: object, duration: float) -> list[float]:
    """Drop non-finite, out-of-range, and non-increasing beat times.

    librosa will not normally emit these, but the click scheduler does a
    binary search over this array and assumes it is sorted and strictly
    increasing. Enforcing that here means the frontend never has to defend
    against a malformed grid.
    """
    import numpy as np

    arr = np.asarray(beat_times, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= 0.0) & (arr <= duration)]
    arr.sort()
    if arr.size == 0:
        return []
    # Strictly increasing: collapse exact/near duplicates (< 1 ms apart).
    keep = np.concatenate(([True], np.diff(arr) > 1e-3))
    return [round(float(t), 6) for t in arr[keep]]


def _fine_onset_env(y: object, sr: int) -> object:
    """High-resolution onset envelope (~1.45 ms/frame).

    Shared by beat refinement and by onset extraction for the grid editor, so
    the transient positions the editor snaps to are exactly the ones the
    refiner already agreed with. Computing it twice would be wasteful and
    could hand the editor a subtly different set of onsets.
    """
    import librosa
    import numpy as np

    return librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=BEATGRID_REFINE_HOP,
        n_fft=BEATGRID_REFINE_NFFT,
        n_mels=BEATGRID_REFINE_MELS,
        aggregate=np.median,
    )


def _fine_onsets(env: object, sr: int, limit: int = BEATGRID_MAX_ONSETS) -> list[float]:
    """Precise transient times for the editor's snap-to-onset.

    Shipped with the grid so dragging a beat snaps locally with no server
    round-trip per pointer move. Peaks are parabolically interpolated to
    sub-frame precision, matching how `_refine_beats` places beats -- a beat
    dragged onto an onset then sits exactly where the refiner would have put
    it.

    `limit` caps the payload for pathological inputs; onsets are dropped
    weakest-first so the strongest transients always survive.
    """
    import librosa
    import numpy as np

    arr = np.asarray(env, dtype=float)
    if arr.size < 3:
        return []

    frame_dur = BEATGRID_REFINE_HOP / sr
    # wait ~30 ms: two peaks closer than that are one transient, not two.
    peaks = librosa.util.peak_pick(
        arr,
        pre_max=int(round(0.015 / frame_dur)),
        post_max=int(round(0.015 / frame_dur)),
        pre_avg=int(round(0.045 / frame_dur)),
        post_avg=int(round(0.060 / frame_dur)),
        delta=float(np.percentile(arr, BEATGRID_ONSET_PERCENTILE)) * BEATGRID_ONSET_DELTA_MULT,
        wait=int(round(0.030 / frame_dur)),
    )
    if len(peaks) == 0:
        return []

    if len(peaks) > limit:
        strongest = np.argsort(arr[peaks])[-limit:]
        peaks = np.sort(peaks[strongest])

    times: list[float] = []
    for p in peaks:
        delta = 0.0
        if 0 < p < arr.size - 1:
            a, b, c = float(arr[p - 1]), float(arr[p]), float(arr[p + 1])
            denom = a - 2.0 * b + c
            if denom != 0.0:
                delta = max(-0.5, min(0.5, 0.5 * (a - c) / denom))
        times.append(round(float((p + delta) * frame_dur), 4))
    return times


def _refine_beats(
    y: object, sr: int, coarse: list[float], beat_interval: float, env: object | None = None
) -> tuple[list[float], int]:
    """Move each beat onto the true transient, to sub-frame precision.

    `beat_track` reports beats on its 512-sample hop grid, so every beat
    carries up to ~12 ms of quantisation error and, worse, that error changes
    from beat to beat -- the click jitters against a drummer who is perfectly
    steady. Two passes fix it:

    1. Snap: search a fine onset envelope within a narrow window around each
       coarse beat and take the strongest peak, refined by parabolic
       interpolation over its neighbours (sub-frame, ~1 ms).
    2. Interpolate: beats where nothing was actually played have no peak to
       snap to. Rather than leaving them quantised (which would reintroduce
       jitter between played beats), place them by linear interpolation
       across beat index between their refined neighbours -- which is exactly
       where an unplayed beat belongs.

    Measured against synthetic transients this leaves a +1.7 ms systematic bias
    (the spectral-flux peak trails the true attack) with 0.19 ms standard
    deviation. The bias is deliberately not compensated: a fixed 1.7 ms offset
    is an order of magnitude below audibility, and a constant tuned against
    synthetic clicks would not generalise to real drum attacks. The 0.19 ms
    jitter is the number that matters, since jitter is what a listener hears
    as flam.

    Returns (beats, refined_count). Falls back to the coarse grid unchanged
    if fewer than two beats could be snapped.
    """
    import numpy as np

    if not coarse:
        return coarse, 0

    env = _fine_onset_env(y, sr) if env is None else env
    if env.size < 3:
        return coarse, 0

    frame_dur = BEATGRID_REFINE_HOP / sr
    window_sec = min(
        BEATGRID_REFINE_WINDOW_MAX,
        max(BEATGRID_REFINE_WINDOW, beat_interval * BEATGRID_REFINE_WINDOW_OF_BEAT),
    )
    window_frames = max(1, int(round(window_sec / frame_dur)))
    threshold = float(np.percentile(env, BEATGRID_REFINE_PERCENTILE))
    max_move = min(
        BEATGRID_REFINE_MOVE_MAX,
        max(BEATGRID_REFINE_MOVE_MIN, beat_interval * BEATGRID_REFINE_MOVE_OF_BEAT),
    )

    refined_idx: list[int] = []
    refined_time: list[float] = []
    for i, beat in enumerate(coarse):
        center = int(round(beat / frame_dur))
        lo = max(0, center - window_frames)
        hi = min(env.size - 1, center + window_frames)
        if hi <= lo:
            continue
        local = env[lo : hi + 1]
        peak = int(lo + np.argmax(local))
        if float(env[peak]) <= threshold:
            continue  # nothing played here; leave it to interpolation
        # Parabolic interpolation through the peak and its two neighbours.
        delta = 0.0
        if 0 < peak < env.size - 1:
            a, b, c = float(env[peak - 1]), float(env[peak]), float(env[peak + 1])
            denom = a - 2.0 * b + c
            if denom != 0.0:
                delta = 0.5 * (a - c) / denom
                delta = max(-0.5, min(0.5, delta))
        snapped = (peak + delta) * frame_dur
        # Reject snaps that move further than hop quantisation can account for.
        # The search window is deliberately wider than this limit so argmax can
        # find the genuine peak among nearby artifacts; this check then throws
        # out the result when what it found was too far away to be this beat.
        if abs(snapped - beat) > max_move:
            continue
        refined_idx.append(i)
        refined_time.append(snapped)

    # One anchor cannot define a grid, and zero means the envelope disagreed
    # with the tracker everywhere -- in both cases trust the coarse result.
    if len(refined_idx) < 2:
        return coarse, len(refined_idx)

    idx_arr = np.asarray(refined_idx, dtype=float)
    time_arr = np.asarray(refined_time, dtype=float)
    all_idx = np.arange(len(coarse), dtype=float)

    # np.interp clamps outside the anchor range; extend with the local slope so
    # leading/trailing unplayed beats keep marching at the right tempo instead
    # of piling up on the first/last anchor.
    out = np.interp(all_idx, idx_arr, time_arr)
    head_slope = (time_arr[1] - time_arr[0]) / max(1e-9, idx_arr[1] - idx_arr[0])
    tail_slope = (time_arr[-1] - time_arr[-2]) / max(1e-9, idx_arr[-1] - idx_arr[-2])
    head = all_idx < idx_arr[0]
    tail = all_idx > idx_arr[-1]
    out[head] = time_arr[0] + (all_idx[head] - idx_arr[0]) * head_slope
    out[tail] = time_arr[-1] + (all_idx[tail] - idx_arr[-1]) * tail_slope

    # Interpolation across a long unrefined stretch spans whatever gap the
    # detector left, so without a bound it relocates real beats by seconds
    # instead of sharpening them. Clamp every beat to a fraction of a beat from
    # where detection put it; interior gaps are then filled deliberately by
    # _fill_interior_gaps rather than smeared away here.
    interp_limit = beat_interval * BEATGRID_REFINE_MAX_INTERP_FRAC
    original = np.asarray(coarse, dtype=float)
    out = np.clip(out, original - interp_limit, original + interp_limit)

    return [float(t) for t in out], len(refined_idx)


def _enforce_grid_consistency(beats: list[float]) -> tuple[list[float], int]:
    """Pull individual beats back onto the grid their neighbours describe.

    `beat_track(trim=False)` occasionally emits a first or last beat that does
    not sit on the tempo grid -- it is reaching into the quiet head/tail of the
    track, where there is little onset evidence to constrain it. Refinement
    then faithfully snaps that beat to whatever spectral edge artifact is
    nearby, turning a small tracker wobble into a single audibly wrong click
    (measured 46 ms out on a synthetic 140 BPM track, against ~2 ms for every
    other beat in the same grid).

    Each beat is compared against the position midway between its neighbours
    and replaced when it disagrees by more than a quarter of a beat. Real
    rubato moves the interval gradually, so a jump that large between
    *adjacent* beats is tracker error, not musical timing.

    Returns (beats, corrected_count).
    """
    import numpy as np

    if len(beats) < 3:
        return beats, 0

    arr = np.asarray(beats, dtype=float)
    median_interval = float(np.median(np.diff(arr)))
    if median_interval <= 0:
        return beats, 0
    tol = median_interval * _GRID_OUTLIER_FRAC

    out = arr.copy()
    # Interior beats: the midpoint of the neighbours is the grid prediction.
    predicted = (arr[:-2] + arr[2:]) / 2.0
    bad = np.abs(arr[1:-1] - predicted) > tol
    out[1:-1] = np.where(bad, predicted, arr[1:-1])
    corrected = int(bad.sum())

    # Endpoints have one neighbour, so extrapolate from the two beats inside
    # them (using the already-corrected interior values).
    first_pred = out[1] - (out[2] - out[1])
    if abs(out[0] - first_pred) > tol:
        out[0] = first_pred
        corrected += 1
    last_pred = out[-2] + (out[-2] - out[-3])
    if abs(out[-1] - last_pred) > tol:
        out[-1] = last_pred
        corrected += 1

    return [float(t) for t in out], corrected


def _interval_spread(beats: list[float]) -> float:
    """Interquartile range of the beat intervals over their median.

    A regularity measure that survives holes. The coefficient of variation does
    not: one drum-free section produces a single enormous interval that drags cv
    to 0.5 on a track whose pulse never wavers. IQR ignores the tails, so it
    reports what the bulk of the grid is doing.
    """
    import numpy as np

    if len(beats) < 5:
        return 0.0
    iv = np.diff(np.asarray(beats, dtype=float))
    median = float(np.median(iv))
    if median <= 0:
        return float("inf")
    return float((np.percentile(iv, 75) - np.percentile(iv, 25)) / median)


def _onset_support(env: object, sr: int, times: object) -> float:
    """Mean onset-envelope strength under a set of times.

    Sampled as a local maximum over a few frames, so a beat a millisecond off a
    transient still counts. Used to compare competing grid phases.
    """
    import numpy as np

    arr = np.asarray(env, dtype=float)
    t = np.asarray(times, dtype=float)
    if arr.size < 3 or t.size == 0:
        return 0.0
    frame_dur = BEATGRID_REFINE_HOP / sr
    centres = np.rint(t / frame_dur).astype(int)
    span = max(1, int(round(0.010 / frame_dur)))
    lo = np.clip(centres - span, 0, arr.size - 1)
    hi = np.clip(centres + span, 0, arr.size - 1)
    return float(np.mean([arr[a : b + 1].max() for a, b in zip(lo, hi, strict=False)]))


def _correct_grid_phase(
    beats: list[float], env: object, sr: int, duration: float
) -> tuple[list[float], bool]:
    """Shift the grid half a beat if that is plainly where the music is.

    Getting the tempo right and the phase wrong is a real tracker failure mode:
    every click then falls in the gap between hits. It showed up on a bare
    repeated-kick pattern where the detected grid had *zero* onset support and
    the half-beat shift had all of it.

    The guard against over-correcting is the ratio: a correct grid on real music
    still has some off-beat support (hi-hats on the eighths), so the shift has to
    be a large improvement, not a marginal one. Returns (beats, shifted).
    """
    import numpy as np

    if len(beats) < 8:
        return beats, False
    arr = np.asarray(beats, dtype=float)
    half = float(np.median(np.diff(arr))) / 2.0
    if half <= 0:
        return beats, False

    here = _onset_support(env, sr, arr)
    shifted_arr = arr + half
    shifted_arr = shifted_arr[shifted_arr <= duration]
    if shifted_arr.size < 8:
        return beats, False
    there = _onset_support(env, sr, shifted_arr)

    if there > here * BEATGRID_PHASE_FLIP_RATIO:
        logger.info("beatgrid: phase shifted by half a beat (support %.4f -> %.4f)", here, there)
        return [float(t) for t in shifted_arr], True
    return beats, False


def _fill_interior_gaps(beats: list[float]) -> tuple[list[float], int]:
    """Subdivide gaps left where the detector heard no beats.

    Detection only places beats where there is evidence, so a section with no
    drums leaves a hole in the middle of the grid -- 5.26 s on the Welcome To
    Paradise breakdown. A metronome that stops for five seconds reads as broken,
    and it is the one thing a player needs a click for.

    A gap is filled when its length divides cleanly by the local beat period.
    Both ends of a gap are real detected beats, so the inserted beats are pinned
    on both sides and cannot drift out of the section -- the worst case is
    choosing k off by one when the gap is an exact half-multiple, which the
    editor can fix with a single anchor.

    Returns (beats, inserted_count).
    """
    import numpy as np

    if len(beats) < 4:
        return beats, 0

    arr = np.asarray(beats, dtype=float)
    intervals = np.diff(arr)
    win = BEATGRID_GAP_LOCAL_WINDOW

    out: list[float] = [float(arr[0])]
    inserted = 0
    for i, d in enumerate(intervals):
        # Local period from the neighbourhood, so a track that drifts is filled
        # at the tempo in force around the gap rather than the global average.
        lo = max(0, i - win)
        hi = min(len(intervals), i + win + 1)
        neighbourhood = np.concatenate([intervals[lo:i], intervals[i + 1 : hi]])
        period = (
            float(np.median(neighbourhood)) if neighbourhood.size else float(np.median(intervals))
        )
        if period > 0:
            k = int(round(d / period))
            if k >= 2 and abs(d / k - period) < BEATGRID_GAP_FILL_MAX_RESIDUAL * period:
                for j in range(1, k):
                    out.append(float(arr[i] + d * j / k))
                    inserted += 1
        out.append(float(arr[i + 1]))

    return out, inserted


def _downbeats_to_bars(beats: list[float], downbeats: list[float]) -> list[dict]:
    """Turn detected downbeat *times* into bar marks over beat *indices*.

    Raw downbeat output is noisy: adjacent frames both firing produce a
    one-beat "bar", and a single missed downbeat produces a double-length one.
    Emitting a mark per observation gave 13 marks (lengths 1, 4, 3, 6...) on a
    plain 4/4 punk track and 256 on Dance of Eternity.

    So: implausible spacings are discarded, the modal spacing becomes the
    track's meter, and a *change* is only recorded once the new length repeats.
    A steady 4/4 song ends up with a single mark; a real, sustained meter change
    still shows up; per-bar noise does not.
    """
    import numpy as np

    if not downbeats or len(beats) < 2:
        return []

    arr = np.asarray(beats, dtype=float)
    idx = sorted({int(np.abs(arr - t).argmin()) for t in downbeats})
    if len(idx) < 3:
        return []

    spans = [(a, b - a) for a, b in zip(idx, idx[1:], strict=False)]
    # A one-beat bar is always detector noise; past 16 it is a missed downbeat.
    spans = [(a, n) for a, n in spans if _MIN_BAR <= n <= _MAX_BAR]
    if not spans:
        return []

    lengths = [n for _, n in spans]
    modal = int(np.bincount(lengths).argmax())

    bars: list[dict] = [{"beat": spans[0][0], "beats_per_bar": modal}]
    current = modal
    # A candidate change is only committed once the same differing length has
    # been seen _BAR_CHANGE_PERSISTENCE times in a row, so one slipped downbeat
    # cannot rewrite the meter.
    run_at = 0
    run_len_value = 0
    run_count = 0
    for at, n in spans:
        if n == current:
            run_count = 0
            continue
        if run_count and n == run_len_value:
            run_count += 1
        else:
            run_at, run_len_value, run_count = at, n, 1
        if run_count >= _BAR_CHANGE_PERSISTENCE:
            bars.append({"beat": run_at, "beats_per_bar": n})
            current = n
            run_count = 0
    return bars


def _extend_to_track_edges(
    beats: list[float], duration: float, irregularity: float
) -> tuple[list[float], int, int]:
    """Continue a steady grid through the intro and outro.

    Beat tracking needs onset evidence, so the grid begins where the drums
    enter and ends where they stop. The tempo does not. Extending outward at
    the local edge tempo gives the click a continuous grid over the whole
    track, which is what a metronome is expected to do.

    Returns (beats, head_added, tail_added). A grid too unsteady to trust is
    returned untouched -- extending a wandering grid invents beats with nothing
    behind them. `irregularity` is _interval_spread's IQR/median measure.
    """
    import numpy as np

    if irregularity > BEATGRID_MAX_IRREGULARITY or len(beats) < 3:
        return beats, 0, 0

    arr = np.asarray(beats, dtype=float)
    edge = min(_EXTRAPOLATE_EDGE_BEATS, len(arr))
    head_iv = float(np.median(np.diff(arr[:edge])))
    tail_iv = float(np.median(np.diff(arr[-edge:])))
    if head_iv <= 0 or tail_iv <= 0:
        return beats, 0, 0

    head: list[float] = []
    t = arr[0] - head_iv
    while t >= 0.0:
        head.append(t)
        t -= head_iv
    head.reverse()

    tail: list[float] = []
    t = arr[-1] + tail_iv
    while t <= duration:
        tail.append(t)
        t += tail_iv

    return head + [float(b) for b in arr] + tail, len(head), len(tail)


def _grid_confidence(beats: list[float], onsets: object, tol: float) -> int:
    """Percentage of beats that have a detected onset within `tol` seconds.

    This is a self-check, not a proof: a beat with no onset is normal (nothing
    has to be played on every beat), so a healthy score is well below 100. Its
    value is in catching *systematic* failure -- a half-time grid, a phase
    offset, or a tracker that locked onto nothing -- which drives the score
    toward zero and lets the UI warn instead of confidently clicking in the
    wrong place.
    """
    import numpy as np

    onset_arr = np.asarray(onsets, dtype=float).ravel()
    onset_arr = onset_arr[np.isfinite(onset_arr)]
    if onset_arr.size == 0 or not beats:
        return 0
    onset_arr.sort()
    beat_arr = np.asarray(beats, dtype=float)
    # Nearest onset for each beat via insertion point against both neighbours.
    idx = np.searchsorted(onset_arr, beat_arr)
    left = np.clip(idx - 1, 0, onset_arr.size - 1)
    right = np.clip(idx, 0, onset_arr.size - 1)
    nearest = np.minimum(
        np.abs(beat_arr - onset_arr[left]),
        np.abs(beat_arr - onset_arr[right]),
    )
    return int(round(float((nearest <= tol).mean()) * 100))


def compute_beat_grid(stems_dir: Path) -> dict | None:
    """Detect beat times across the full track and write `stems/beats.json`.

    Returns the grid dict on success, None on any failure. Never raises --
    a missing beat grid degrades the click track, it must not fail the job.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not installed -- skipping beat grid")
        return None

    try:
        picked = _pick_source(stems_dir)
        if picked is None:
            logger.warning("beatgrid: no usable source stem in %s", stems_dir)
            return None
        source, source_label = picked

        loaded = _load_audio_ffmpeg(source, sr=BEATGRID_SR, duration=None, timeout=TIMEOUT_BEATGRID)
        if loaded is None:
            return None
        y, sr = loaded
        duration = float(len(y)) / sr
        if duration <= 0:
            return None

        # Median aggregation across mel bands suppresses one-off broadband
        # noise (a cymbal wash, a vinyl click) that a mean would let dominate
        # the onset envelope.
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=BEATGRID_HOP, aggregate=np.median
        )
        if not np.any(onset_env):
            logger.warning("beatgrid: flat onset envelope for %s (silent stem?)", source.name)
            return None

        beat_times, downbeat_times, detector = detect_beats(y, sr, onset_env)
        beats = _sanitize(beat_times, duration)
        if len(beats) < _MIN_BEATS:
            logger.warning("beatgrid: only %d beats detected -- discarding", len(beats))
            return None

        # Coarse interval only sizes the refinement search window; the real
        # tempo is derived from the refined grid below.
        coarse_interval = float(np.median(np.diff(np.asarray(beats, dtype=float))))
        if coarse_interval <= 0:
            return None
        # Computed once and reused: refinement and the editor's snap targets
        # must agree, and this envelope is the expensive part of the stage.
        fine_env = _fine_onset_env(y, sr)
        # Phase before refinement: refinement snaps beats onto transients, and a
        # grid on the wrong half of the beat has none to snap to.
        beats, phase_shifted = _correct_grid_phase(beats, fine_env, sr, duration)
        beats, refined_count = _refine_beats(y, sr, beats, coarse_interval, env=fine_env)
        detected_count = len(beats)

        # Gap filling and the consistency pass both assume the pulse is locally
        # regular, so they only run when it measurably is. On material that is
        # genuinely irregular they fight the detector rather than helping it.
        irregularity = _interval_spread(beats)
        regular = irregularity <= BEATGRID_MAX_IRREGULARITY
        gap_filled = 0
        corrected_count = 0
        if regular:
            # Order matters. Both passes reason about a beat against its
            # neighbours, and a drum-free section leaves a hole several beats
            # wide. Run the consistency check first and it sees the beats
            # bounding that hole as wild outliers and drags them into the middle
            # of it. Fill first, and the grid it inspects is regular.
            beats, gap_filled = _fill_interior_gaps(beats)
            beats = _sanitize(beats, duration)
            beats, corrected_count = _enforce_grid_consistency(beats)
            beats = _sanitize(beats, duration)
        if len(beats) < _MIN_BEATS:
            logger.warning("beatgrid: post-processing left only %d beats -- discarding", len(beats))
            return None

        intervals = np.diff(np.asarray(beats, dtype=float))
        median_interval = float(np.median(intervals))
        if median_interval <= 0:
            return None
        bpm = 60.0 / median_interval
        if not (_MIN_BPM <= bpm <= _MAX_BPM):
            logger.warning("beatgrid: implausible tempo %.1f BPM -- discarding", bpm)
            return None

        # Confidence and CV are measured before the grid is extended past the
        # detected range. Extrapolated beats have no onset under them by
        # definition, so including them would drag the score down and misreport
        # a good detection as a poor one.
        tol = min(BEATGRID_ONSET_TOL_MAX, median_interval * BEATGRID_ONSET_TOL_FRAC)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=BEATGRID_HOP, units="time"
        )
        confidence = _grid_confidence(beats, onsets, tol)

        # Coefficient of variation of the beat intervals: 0 is metronomic.
        # Surfaced so the UI can distinguish "steady song" from "the tracker
        # is wandering", which look identical from BPM alone.
        # median_interval is guaranteed positive by the guard above.
        cv = float(intervals.std() / median_interval)

        beats, head_added, tail_added = _extend_to_track_edges(beats, duration, irregularity)
        beats = _sanitize(beats, duration)
        if len(beats) < _MIN_BEATS:
            return None
        # Tempo describes the grid that actually gets clicked, so recompute it
        # over the final grid. Median, not mean: refinement already removed the
        # per-beat quantisation the mean was averaging away, and the mean is
        # skewed by gap-filled and irregular sections.
        beat_arr = np.asarray(beats, dtype=float)
        final_median = float(np.median(np.diff(beat_arr)))
        bpm = 60.0 / final_median if final_median > 0 else bpm

        bars = _downbeats_to_bars(beats, downbeat_times)

        grid = {
            "version": 1,
            "source": source_label,
            # Which back-end produced this grid. Surfaced so a half-time grid on
            # a fast track can be traced to the librosa fallback rather than
            # guessed at.
            "detector": detector,
            # Bar marks from real detected downbeats, emitted only where the bar
            # length changes. Empty on the librosa fallback, which has no
            # downbeat notion.
            "bars": bars,
            "bpm": round(bpm, 3),
            "duration": round(duration, 3),
            "confidence": confidence,
            "interval_cv": round(cv, 4),
            # IQR/median: the regularity measure that gated the clean-up passes.
            "irregularity": round(irregularity, 4),
            # How many beats were snapped to a real transient vs positioned by
            # interpolation. A low ratio means the grid is mostly inferred.
            "refined": refined_count,
            # Beats the tracker placed off its own grid and that were pulled
            # back. A handful is normal (head/tail); a large number means the
            # tracker never locked on.
            "corrected": corrected_count,
            # Beats actually detected from audio, versus those continued through
            # the intro/outro at the edge tempo.
            # Transient positions the grid editor snaps dragged beats onto.
            "onsets": _fine_onsets(fine_env, sr),
            "detected": detected_count,
            "extrapolated_head": head_added,
            "extrapolated_tail": tail_added,
            # Beats inserted into drum-free sections to keep the click running.
            "gap_filled": gap_filled,
            # True when the whole grid was half a beat out and got shifted.
            "phase_shifted": phase_shifted,
            "beats": [round(float(b), 6) for b in beats],
        }

        tmp = stems_dir / "beats.json.tmp"
        tmp.write_text(json.dumps(grid), encoding="utf-8")
        tmp.replace(stems_dir / "beats.json")
        logger.info(
            "beatgrid[%s]: %d beats (%d detected, +%d gap, +%d head, +%d tail), %.2f BPM, "
            "%d bar marks, confidence %d%%, cv %.3f, irregularity %.3f%s, refined %d, "
            "corrected %d (source=%s)",
            detector,
            len(beats),
            detected_count,
            gap_filled,
            head_added,
            tail_added,
            bpm,
            len(bars),
            confidence,
            cv,
            irregularity,
            " [irregular: clean-up skipped]" if not regular else "",
            refined_count,
            corrected_count,
            source_label,
        )
        return grid
    except Exception:
        logger.exception("beatgrid failed for %s", stems_dir)
        return None

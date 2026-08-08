"""Render the click track to audio so it can be included in exports.

Playback synthesises the click in the browser with Web Audio oscillators, which
never reach the server -- the export path is ffmpeg summing stem WAVs. This
module produces an equivalent WAV so the click can be mixed in as one more
ffmpeg input.

The voice is reproduced from `static/js/metronome.js` deliberately literally,
including its exponential gain ramps, because an export that sounds different
from what the user monitored is worse than no export option at all. The two
implementations are pinned together by `tests/test_click_render.py`, which
asserts the rendered peaks land on the same beats the scheduler would use.
"""

from __future__ import annotations

import hashlib
import logging
import wave
from pathlib import Path

logger = logging.getLogger("stemdeck.clickrender")

# Mirrors the constants at the top of static/js/metronome.js. Changing either
# side without the other makes exports diverge from playback.
CLICK_FREQ = 1000.0
ACCENT_FREQ = 1500.0
CLICK_DECAY = 0.035
CLICK_ATTACK = 0.001
CLICK_PEAK = 0.7
ACCENT_PEAK = 1.0
# exponentialRampToValueAtTime cannot start from zero, so the scheduler ramps
# from this floor; matching it keeps the attack shape identical.
RAMP_FLOOR = 0.0001

# Accent modes, matching the frontend's Accent selector.
ACCENT_AUTO = -1  # follow the detected bar marks
ACCENT_OFF = 0

_VALID_MULTIPLIERS = (0.5, 1.0, 2.0)


def rescale_beats(beats: list[float], multiplier: float) -> list[float]:
    """Apply the playback rate multiplier. Mirrors `_rescale` in metronome.js:
    doubling inserts midpoints, halving takes every other beat, and both derive
    from the original list so switching never compounds."""
    if multiplier == 2.0:
        out: list[float] = []
        for i in range(len(beats) - 1):
            out.append(beats[i])
            out.append((beats[i] + beats[i + 1]) / 2.0)
        if beats:
            out.append(beats[-1])
        return out
    if multiplier == 0.5:
        return beats[::2]
    return list(beats)


def source_index(i: int, multiplier: float) -> int | None:
    """Map an index in the rescaled grid back to the original beat it came from.

    Bar marks are recorded against the *detected* beats, so accents must be
    decided in that index space. At x2 the odd entries are inserted midpoints
    that correspond to no original beat and can never be downbeats; at /2 every
    entry is an original beat two apart.
    """
    if multiplier == 2.0:
        return i // 2 if i % 2 == 0 else None
    if multiplier == 0.5:
        return i * 2
    return i


def is_downbeat(index: int | None, bars: list[dict], accent_mode: int) -> bool:
    """Whether the beat at `index` (original grid) carries an accent."""
    if index is None or accent_mode == ACCENT_OFF:
        return False
    if accent_mode > 0:
        return index % accent_mode == 0
    # Auto: follow the last bar mark at or before this beat.
    mark = None
    for b in bars:
        beat = b.get("beat")
        if isinstance(beat, int) and beat <= index:
            mark = b
        else:
            break
    if mark is None:
        return False
    per_bar = mark.get("beats_per_bar")
    if not isinstance(per_bar, int) or per_bar < 1:
        return False
    return (index - mark["beat"]) % per_bar == 0


def _voice(peak: float, freq: float, sample_rate: int):
    """One click as a float array: a sine under the scheduler's two exponential
    gain ramps (RAMP_FLOOR -> peak over the attack, then back down over the rest
    of the decay). Phase starts at zero, exactly as a fresh OscillatorNode."""
    import numpy as np

    n = int(round(CLICK_DECAY * sample_rate))
    t = np.arange(n) / sample_rate
    attack = t <= CLICK_ATTACK
    env = np.empty(n)
    env[attack] = RAMP_FLOOR * (peak / RAMP_FLOOR) ** (t[attack] / CLICK_ATTACK)
    span = CLICK_DECAY - CLICK_ATTACK
    env[~attack] = peak * (RAMP_FLOOR / peak) ** ((t[~attack] - CLICK_ATTACK) / span)
    return np.sin(2.0 * np.pi * freq * t) * env


def cache_key(
    job_id: str,
    beats: list[float],
    bars: list[dict],
    duration: float,
    sample_rate: int,
    multiplier: float,
    accent_mode: int,
) -> str:
    """Every input to the render is in the key. Beats are included by digest
    rather than by job id alone: an edited grid must not hit a cache entry
    rendered from the detected one."""
    grid = hashlib.sha1(
        ("|".join(f"{b:.6f}" for b in beats)).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    bar_sig = ",".join(f"{b.get('beat')}:{b.get('beats_per_bar')}" for b in bars)
    raw = f"{job_id}|{grid}|{bar_sig}|{duration:.3f}|{sample_rate}|{multiplier}|{accent_mode}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


def render_click_wav(
    dest: Path,
    beats: list[float],
    bars: list[dict],
    duration: float,
    sample_rate: int = 44100,
    multiplier: float = 1.0,
    accent_mode: int = ACCENT_AUTO,
) -> Path | None:
    """Write a mono WAV of the click track spanning the whole track.

    Full length regardless of where the beats start, so the export's region trim
    (`-ss` before every ffmpeg input) lines the click up with the stems without
    any special-casing. Returns the path, or None when there is nothing to
    render.
    """
    if multiplier not in _VALID_MULTIPLIERS:
        multiplier = 1.0
    grid = rescale_beats([float(b) for b in beats], multiplier)
    total = int(round(duration * sample_rate))
    if not grid or total <= 0:
        return None

    import numpy as np

    buf = np.zeros(total, dtype=np.float64)
    # Only two distinct voices, so render each once and stamp it in.
    plain = _voice(CLICK_PEAK, CLICK_FREQ, sample_rate)
    accented = _voice(ACCENT_PEAK, ACCENT_FREQ, sample_rate)

    for i, t in enumerate(grid):
        start = int(round(t * sample_rate))
        if start >= total or start < 0:
            continue
        voice = accented if is_downbeat(source_index(i, multiplier), bars, accent_mode) else plain
        n = min(len(voice), total - start)
        # Clicks can overlap at very fast tempos; summing matches the graph,
        # where every click is its own node into the same gain.
        buf[start : start + n] += voice[:n]

    np.clip(buf, -1.0, 1.0, out=buf)
    pcm = (buf * 32767.0).astype("<i2")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    tmp.replace(dest)
    logger.info("click render: %d beats, %.1f s -> %s", len(grid), duration, dest.name)
    return dest

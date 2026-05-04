import os
import re
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def _detect_device() -> str:
    """Pick best available Torch device for Demucs. Override via
    STEMDECK_DEMUCS_DEVICE env var ('cuda' | 'mps' | 'cpu'). Apple Silicon
    silently falls back to CPU otherwise -- demucs's CLI default is
    "cuda if available else cpu" and macOS has no CUDA, leaving the
    integrated GPU idle and processing 3-5x slower than necessary."""
    forced = os.environ.get("STEMDECK_DEMUCS_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = ROOT / "static"
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "other")
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Runtime knobs -- env-backed so Docker / local can tune without a code edit.
JOBS_DIR = _env_path("STEMDECK_JOBS_DIR", ROOT / "jobs")
DEMUCS_MODEL = os.environ.get("STEMDECK_DEMUCS_MODEL", "htdemucs_6s").strip() or "htdemucs_6s"
DEMUCS_DEVICE = _detect_device()
MAX_DURATION_SEC = _env_int("STEMDECK_MAX_DURATION_SEC", 1200)  # 20 min default
JOB_TTL_SECONDS = _env_int("STEMDECK_JOB_TTL_SECONDS", 24 * 3600)  # 24 h default
MAX_PENDING_JOBS = _env_int("STEMDECK_MAX_PENDING_JOBS", 3)

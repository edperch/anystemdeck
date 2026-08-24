"""Stripping locally- or personally-identifying text before it can reach a
public report.

The notification centre's "Report on GitHub"/"Report on Discord" flow and its
opt-in "include recent logs" button both end with this text pasted somewhere
public. Three things in particular have no business being there:

  - The reporter's home directory. On Windows the Python install path alone
    (`C:\\Users\\<name>\\...`) carries the OS username.
  - A track's source URL. `download.py` logs it for every job
    ("download starting: <url>"), so a raw log tail carries the YouTube/
    SoundCloud link for everything the reporter has ever imported in the
    fetched window -- not just the one that failed. The per-job failure
    report has always kept this out deliberately (see
    app/api/jobs.py's _FAILURE_PUBLIC_KEYS); a raw log has no such allowlist
    of its own, so it is enforced here instead.
  - An IP address. The mobile UI talks to this backend over the LAN when
    network access is on, so uvicorn's access log (captured into
    backend.log on desktop) can carry another device's address on the
    reporter's home network, not just their own.

Every place that can hand text to that report flow -- the per-job failure
evidence (runner.py) and the Settings -> Logs viewer (main.py) -- redacts
through this one function rather than each carrying its own copy.
"""

from __future__ import annotations

import re
from pathlib import Path

# Mirrors app/pipeline/download.py's _YOUTUBE_HOSTS | _SOUNDCLOUD_HOSTS. Kept
# as a literal copy rather than importing it: this module is used from
# main.py's request path, and pipeline.download pulls in yt_dlp, which is not
# something a log-tail redaction pass should need to import.
_SOURCE_URL_RE = re.compile(
    r"https?://(?:[\w-]+\.)*"
    r"(?:youtube\.com|youtu\.be|youtube-nocookie\.com|soundcloud\.com)\S*",
    re.IGNORECASE,
)

# IPv4 only. The app's own LAN detection (app/main.py's _is_lan_ipv4) is
# IPv4-only too, and a loose IPv6 pattern is a real hazard here: something
# like "(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}" also matches an HH:MM:SS
# timestamp (every log line has one), which would mangle every line's time
# instead of catching the rare home IPv6 address.
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact(text: str) -> str:
    """Strip the reporter's home directory, any YouTube/SoundCloud source
    URL, and any IPv4 address from `text`."""
    home = str(Path.home())
    if home and home not in (".", "/"):
        text = text.replace(home, "<home>")
    text = _SOURCE_URL_RE.sub("<source-url-redacted>", text)
    text = _IPV4_RE.sub("<ip>", text)
    return text

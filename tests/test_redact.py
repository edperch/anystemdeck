from __future__ import annotations

from pathlib import Path

from app.core.redact import redact


def test_strips_the_home_directory():
    home = str(Path.home())
    text = f"{home}\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\threads.py"
    result = redact(text)
    assert home not in result
    assert "<home>" in result
    assert "threads.py" in result


def test_strips_a_youtube_url():
    text = "download starting: https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=RDdQw4w9WgXcQ"
    result = redact(text)
    assert "youtube.com" not in result
    assert "dQw4w9WgXcQ" not in result
    assert "<source-url-redacted>" in result


def test_strips_a_youtu_be_url():
    result = redact("source: https://youtu.be/dQw4w9WgXcQ")
    assert "youtu.be" not in result
    assert "<source-url-redacted>" in result


def test_strips_a_soundcloud_url():
    result = redact("source: https://soundcloud.com/artist/track")
    assert "soundcloud.com" not in result
    assert "<source-url-redacted>" in result


def test_does_not_touch_an_unrelated_url():
    """Only the hosts this app actually pulls tracks from are source URLs --
    a link to the app's own site/repo, or anything else, is not personal
    information and must survive (it's often the useful part of a log line)."""
    text = "see https://github.com/stemdeckapp/stemdeck/issues/277 and https://stemdeck.app"
    result = redact(text)
    assert result == text


def test_strips_an_ipv4_address():
    result = redact('192.168.1.14:52341 - "GET /api/jobs HTTP/1.1" 200 OK')
    assert "192.168.1.14" not in result
    assert "<ip>" in result


def test_does_not_mistake_a_version_string_for_an_ip():
    result = redact("StemDeck v0.9.1.dev2+g682ab90d7.d20260816")
    assert "0.9.1" in result
    assert "<ip>" not in result


def test_does_not_mistake_a_timestamp_for_anything():
    """The pipeline's own log format is `YYYY-MM-DD HH:MM:SS ...` -- every
    single line has one, so any redaction pattern with false positives here
    would mangle the entire report, not just the rare real leak."""
    text = "2026-08-17 16:50:02 E stemdeck.pipeline pipeline failed for job abcdefabcdef"
    assert redact(text) == text


def test_composes_all_three_kinds_in_one_pass():
    home = str(Path.home())
    text = f"{home}\\stemdeck  192.168.1.14 requested https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = redact(text)
    assert home not in result
    assert "192.168.1.14" not in result
    assert "youtube.com" not in result
    assert "<home>" in result
    assert "<ip>" in result
    assert "<source-url-redacted>" in result

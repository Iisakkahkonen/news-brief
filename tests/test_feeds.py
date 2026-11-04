import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from news_brief import feeds
from news_brief.feeds import Outlet, fetch_all, load_outlets, parse_date, parse_feed, select, strip_html

FIXTURES = Path(__file__).parent / "fixtures"


def test_rss_items_are_parsed_with_clean_text():
    items = parse_feed((FIXTURES / "sample_rss.xml").read_bytes(), "Sample World")

    assert [i.title for i in items] == [
        "Ceasefire talks resume in Geneva",
        "Floods close roads in the north",
    ]
    first = items[0]
    assert first.summary == "Negotiators met for a second day & agreed to continue."
    assert first.link == "https://example.org/world/ceasefire-talks"
    assert first.published == datetime(2026, 11, 3, 9, 30, tzinfo=timezone.utc)


def test_rss_date_with_offset_is_converted_to_utc():
    items = parse_feed((FIXTURES / "sample_rss.xml").read_bytes(), "Sample World")
    assert items[1].published == datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)


def test_atom_items_support_alternate_links_summary_and_content():
    items = parse_feed((FIXTURES / "sample_atom.xml").read_bytes(), "Sample Atom")

    assert len(items) == 2
    assert items[0].link == "https://example.org/atom/election"
    assert items[0].summary == "Officials certified the count on Monday."
    assert items[0].published == datetime(2026, 11, 3, 8, 15, tzinfo=timezone.utc)
    assert items[1].summary == "Dockworkers rejected the latest offer."
    assert items[1].published == datetime(2026, 11, 3, 8, 0, tzinfo=timezone.utc)


def test_rdf_rss_1_0_items_are_parsed():
    items = parse_feed((FIXTURES / "sample_rdf.xml").read_bytes(), "Sample RDF")

    assert len(items) == 1
    assert items[0].title == "Leaders meet at regional summit"
    assert items[0].link == "https://example.org/rdf/summit"
    assert items[0].summary == "Talks focus on trade."
    assert items[0].published == datetime(2026, 11, 3, 7, 45, tzinfo=timezone.utc)


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self, _limit):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_429(retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("https://x.example", 429, "Too Many Requests", headers, None)


def test_rate_limited_feed_is_retried_once_after_waiting(monkeypatch):
    body = (FIXTURES / "sample_rss.xml").read_bytes()
    calls, sleeps = [], []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise _http_429("3")
        return _FakeResponse(body)

    monkeypatch.setattr(feeds.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(feeds.time, "sleep", sleeps.append)

    items = feeds.fetch_feed(Outlet("Limited", "https://x.example/rss", "en", "approved"))

    assert len(calls) == 2
    assert sleeps == [3.0]
    assert len(items) == 2


def test_retry_wait_is_capped_and_a_second_429_is_reported(monkeypatch):
    sleeps = []

    def always_limited(request, timeout):
        raise _http_429("600")

    monkeypatch.setattr(feeds.urllib.request, "urlopen", always_limited)
    monkeypatch.setattr(feeds.time, "sleep", sleeps.append)

    with pytest.raises(urllib.error.HTTPError):
        feeds.fetch_feed(Outlet("Limited", "https://x.example/rss", "en", "approved"))
    assert sleeps == [feeds.MAX_RETRY_WAIT]


def test_other_http_errors_are_not_retried(monkeypatch):
    calls = []

    def not_found(request, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("https://x.example", 404, "Not Found", {}, None)

    monkeypatch.setattr(feeds.urllib.request, "urlopen", not_found)

    with pytest.raises(urllib.error.HTTPError):
        feeds.fetch_feed(Outlet("Gone", "https://x.example/rss", "en", "approved"))
    assert len(calls) == 1


def test_item_ids_are_stable_and_unique_per_outlet():
    first = parse_feed((FIXTURES / "sample_rss.xml").read_bytes(), "Sample World")
    second = parse_feed((FIXTURES / "sample_rss.xml").read_bytes(), "Sample World")
    other_outlet = parse_feed((FIXTURES / "sample_rss.xml").read_bytes(), "Other")

    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert first[0].item_id != other_outlet[0].item_id
    assert first[0].item_id != first[1].item_id


def test_unsupported_document_is_rejected():
    with pytest.raises(ValueError, match="unsupported feed format"):
        parse_feed(b"<html><body>not a feed</body></html>", "x")


def test_rss_without_channel_is_rejected():
    with pytest.raises(ValueError, match="no <channel>"):
        parse_feed(b"<rss version='2.0'></rss>", "x")


@pytest.mark.parametrize("raw, expected", [
    ("<p>Hello <b>there</b></p>", "Hello there"),
    ("A &amp; B&nbsp;&lt;C&gt;", "A & B <C>"),
    ("  spaced\n\n out  ", "spaced out"),
    ("", ""),
    (None, ""),
])
def test_strip_html(raw, expected):
    assert strip_html(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    (None, None),
    ("", None),
    ("not a date", None),
    ("Mon, 02 Nov 2026 07:00:00 GMT", datetime(2026, 11, 2, 7, tzinfo=timezone.utc)),
    ("2026-11-02T09:00:00", datetime(2026, 11, 2, 9, tzinfo=timezone.utc)),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_load_outlets_reads_config_and_defaults_to_proposed(tmp_path):
    config = tmp_path / "outlets.toml"
    config.write_text(
        '[[outlet]]\nname = "A"\nurl = "https://a.example/rss"\n'
        '[[outlet]]\nname = "B"\nurl = "https://b.example/rss"\nstatus = "approved"\n',
        encoding="utf-8",
    )
    outlets = load_outlets(config)
    assert [o.status for o in outlets] == ["proposed", "approved"]


def test_load_outlets_rejects_unknown_status(tmp_path):
    config = tmp_path / "outlets.toml"
    config.write_text('[[outlet]]\nname = "A"\nurl = "x"\nstatus = "maybe"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="status must be one of"):
        load_outlets(config)


def test_select_uses_only_approved_outlets_by_default():
    outlets = [
        Outlet("A", "u", "en", "approved"),
        Outlet("B", "u", "en", "proposed"),
    ]
    assert [o.name for o in select(outlets)] == ["A"]
    assert [o.name for o in select(outlets, include_proposed=True)] == ["A", "B"]


def test_shipped_config_has_no_approved_outlets_yet():
    # Nothing has been approved, so the shipped config must not fetch anything.
    outlets = load_outlets(Path(__file__).parents[1] / "config" / "outlets.toml")
    assert outlets, "config should list proposed outlets"
    assert select(outlets) == []


def test_one_broken_feed_does_not_stop_the_run(monkeypatch):
    good = Outlet("Good", "https://good.example/rss", "en", "approved")
    bad = Outlet("Bad", "https://bad.example/rss", "en", "approved")

    def fake_fetch(outlet, timeout=20.0):
        if outlet is bad:
            raise OSError("connection refused")
        return parse_feed((FIXTURES / "sample_rss.xml").read_bytes(), outlet.name)

    monkeypatch.setattr(feeds, "fetch_feed", fake_fetch)
    items, errors = fetch_all([bad, good])

    assert errors == {"Bad": "OSError: connection refused"}
    assert len(items) == 2
    assert all(i.outlet == "Good" for i in items)


def test_duplicate_links_across_feeds_are_kept_once(monkeypatch):
    outlets = [Outlet("One", "u1", "en", "approved"), Outlet("Two", "u2", "en", "approved")]
    monkeypatch.setattr(
        feeds, "fetch_feed",
        lambda outlet, timeout=20.0: parse_feed((FIXTURES / "sample_rss.xml").read_bytes(), "shared"),
    )
    items, errors = fetch_all(outlets)
    assert errors == {}
    assert len(items) == 2

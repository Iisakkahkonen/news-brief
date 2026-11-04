"""Read headlines and feed summaries from RSS 2.0 and Atom feeds.

Standard library only, so the pipeline runs without installing anything.
Only the title, link, date and the summary the outlet already publishes are
kept. Full articles are never fetched or stored.
"""

from __future__ import annotations

import hashlib
import html.parser
import re
import tomllib
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

USER_AGENT = "news-brief/0.1 (+https://github.com/Iisakkahkonen/news-brief)"
MAX_FEED_BYTES = 2_000_000
ATOM = "{http://www.w3.org/2005/Atom}"
VALID_STATUSES = ("proposed", "approved")


@dataclass(frozen=True)
class Outlet:
    name: str
    url: str
    language: str
    status: str  # "proposed" until the user approves it, then "approved"


@dataclass(frozen=True)
class Item:
    item_id: str  # short stable ID; the summariser cites items by this
    outlet: str
    title: str
    link: str
    summary: str
    published: datetime | None


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(text: str) -> str:
    """Return the visible text of a snippet of HTML, with whitespace collapsed."""
    parser = _TextExtractor()
    parser.feed(text or "")
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def parse_date(value: str | None) -> datetime | None:
    """Parse an RFC 822 (RSS) or ISO 8601 (Atom) date into UTC, or None."""
    if not value or not value.strip():
        return None
    value = value.strip()
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _make_item(outlet: str, title: str, link: str, summary_html: str,
               published: str | None, guid: str) -> Item | None:
    title = " ".join(title.split())
    if not title or not link:
        return None
    item_id = hashlib.sha256(f"{outlet}|{guid or link}".encode("utf-8")).hexdigest()[:12]
    return Item(
        item_id=item_id,
        outlet=outlet,
        title=title,
        link=link.strip(),
        summary=strip_html(summary_html),
        published=parse_date(published),
    )


def _parse_rss(root: ET.Element, outlet: str) -> list[Item]:
    channel = root.find("channel")
    if channel is None:
        raise ValueError("RSS document has no <channel>")
    items = []
    for el in channel.findall("item"):
        link = el.findtext("link", "")
        item = _make_item(
            outlet,
            title=el.findtext("title", ""),
            link=link,
            summary_html="".join(el.find("description").itertext())
            if el.find("description") is not None else "",
            published=el.findtext("pubDate"),
            guid=el.findtext("guid", "") or link,
        )
        if item:
            items.append(item)
    return items


def _parse_atom(root: ET.Element, outlet: str) -> list[Item]:
    items = []
    for entry in root.findall(ATOM + "entry"):
        link = ""
        for candidate in entry.findall(ATOM + "link"):
            if candidate.get("rel", "alternate") == "alternate":
                link = candidate.get("href", "")
                break
        body = entry.find(ATOM + "summary")
        if body is None:
            body = entry.find(ATOM + "content")
        summary_html = "".join(body.itertext()) if body is not None else ""
        item = _make_item(
            outlet,
            title=entry.findtext(ATOM + "title", ""),
            link=link,
            summary_html=summary_html,
            published=entry.findtext(ATOM + "published")
            or entry.findtext(ATOM + "updated"),
            guid=entry.findtext(ATOM + "id", "") or link,
        )
        if item:
            items.append(item)
    return items


def parse_feed(xml_bytes: bytes, outlet: str) -> list[Item]:
    """Parse an RSS 2.0 or Atom document into items, newest first as published."""
    root = ET.fromstring(xml_bytes)
    if root.tag == ATOM + "feed":
        return _parse_atom(root, outlet)
    if root.tag == "rss":
        return _parse_rss(root, outlet)
    raise ValueError(f"unsupported feed format: root element <{root.tag}>")


def fetch_feed(outlet: Outlet, timeout: float = 20.0) -> list[Item]:
    request = urllib.request.Request(
        outlet.url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.5",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(MAX_FEED_BYTES + 1)
    if len(body) > MAX_FEED_BYTES:
        raise ValueError(f"feed is larger than {MAX_FEED_BYTES} bytes")
    return parse_feed(body, outlet.name)


def fetch_all(outlets: list[Outlet], max_items_per_feed: int = 50,
              ) -> tuple[list[Item], dict[str, str]]:
    """Fetch every outlet. A broken feed is reported, never allowed to stop the run.

    Returns (items, errors) where errors maps outlet name to a short reason.
    """
    items: list[Item] = []
    errors: dict[str, str] = {}
    seen_links: set[str] = set()
    for outlet in outlets:
        try:
            feed_items = fetch_feed(outlet)
        except (OSError, ValueError, SyntaxError) as exc:
            errors[outlet.name] = f"{type(exc).__name__}: {exc}"
            continue
        for item in feed_items[:max_items_per_feed]:
            if item.link not in seen_links:
                seen_links.add(item.link)
                items.append(item)
    return items, errors


def load_outlets(path: Path) -> list[Outlet]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    outlets = []
    for row in data.get("outlet", []):
        status = row.get("status", "proposed")
        if status not in VALID_STATUSES:
            raise ValueError(f"{row.get('name')}: status must be one of {VALID_STATUSES}")
        outlets.append(Outlet(
            name=row["name"],
            url=row["url"],
            language=row.get("language", "en"),
            status=status,
        ))
    return outlets


def select(outlets: list[Outlet], include_proposed: bool = False) -> list[Outlet]:
    """Only approved outlets are used unless the caller asks to trial proposed ones."""
    if include_proposed:
        return list(outlets)
    return [o for o in outlets if o.status == "approved"]

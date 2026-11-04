from datetime import datetime, timezone

import pytest

from news_brief.feeds import Item


def make_item(item_id, outlet, title, summary="", hour=8, link=None):
    return Item(
        item_id=item_id,
        outlet=outlet,
        title=title,
        link=link or f"https://{outlet.lower().replace(' ', '')}.example/{item_id}",
        summary=summary,
        published=datetime(2026, 11, 4, hour, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def day_of_news():
    """A small realistic day: two multi-outlet events and two one-off stories."""
    return [
        make_item("geneva1", "BBC", "Ceasefire talks resume in Geneva after two-week pause",
                  "Negotiators from both sides met on Tuesday and agreed to meet again.", 7),
        make_item("geneva2", "Al Jazeera", "Geneva ceasefire talks restart as mediators urge calm",
                  "Mediators said the talks would continue on Wednesday.", 8),
        make_item("geneva3", "DW", "Talks on ceasefire restart in Geneva",
                  "Delegations returned to the table in Geneva.", 9),
        make_item("flood1", "Dawn", "Floods kill 14 in northern Pakistan",
                  "Heavy rain triggered landslides and flash floods in the north.", 6),
        make_item("flood2", "The Hindu", "Pakistan floods: death toll reaches 14",
                  "Officials said flash floods hit northern districts.", 10),
        make_item("vote1", "NPR", "Voters in Brazil head to the polls for regional runoffs",
                  "Turnout was reported as steady in several states.", 11),
        make_item("tech1", "Euronews", "European regulators fine ferry operator over safety lapses",
                  "The penalty follows an inspection of three vessels.", 12),
    ]

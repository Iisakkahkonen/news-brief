import json
from pathlib import Path

from news_brief.__main__ import save_items

from conftest import make_item


def test_save_items_writes_everything_needed_to_regroup_offline(tmp_path):
    items = [make_item("a1", "BBC", "Headline one", "Summary one"),
             make_item("b2", "DW", "Headline two")]
    target = tmp_path / "items.json"

    save_items(items, target)

    rows = json.loads(target.read_text(encoding="utf-8"))
    assert [r["item_id"] for r in rows] == ["a1", "b2"]
    assert rows[0] == {
        "item_id": "a1", "outlet": "BBC", "title": "Headline one",
        "summary": "Summary one", "link": "https://bbc.example/a1",
        "published": "2026-11-04T08:00:00+00:00",
    }

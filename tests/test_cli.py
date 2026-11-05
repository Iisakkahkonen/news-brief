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


def test_run_brief_writes_the_brief_and_a_check_log(tmp_path, day_of_news):
    import json as _json
    from datetime import datetime

    from news_brief.run import run_brief
    from test_summarise import ScriptedModel

    def answer(*pairs):
        return _json.dumps({"sentences": [{"text": t, "sources": s} for t, s in pairs]})

    model = ScriptedModel(
        answer(("Ceasefire talks resumed in Geneva.", ["geneva1"]),
               ("Talks followed 40 days of fighting.", ["geneva1"])),
        answer(("Floods killed 14 people in northern Pakistan.", ["flood1"])),
    )
    result = run_brief(day_of_news, model, tmp_path / "out",
                       datetime(2026, 11, 5, 18, 46), min_outlets=2)

    brief_text = result.brief_path.read_text(encoding="utf-8")
    log_text = result.log_path.read_text(encoding="utf-8")
    assert result.brief_path.name == "brief-2026-11-05-1846.txt"
    assert result.log_path.name == "check-log-2026-11-05-1846.txt"
    assert "Ceasefire talks resumed in Geneva." in brief_text
    assert "40 days" not in brief_text
    assert "Sentences removed: 1" in log_text
    assert "SENTENCE: Talks followed 40 days of fighting." in log_text
    assert "REASON:  number not in cited items: 40" in log_text


def test_brief_without_a_key_stops_before_touching_the_network(monkeypatch, capsys):
    import sys
    from news_brief import __main__ as cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["news_brief", "--brief"])
    monkeypatch.setattr(cli, "fetch_all",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))

    assert cli.main() == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().out


def test_a_second_run_does_not_overwrite_the_first(tmp_path, day_of_news):
    from datetime import datetime

    from news_brief.run import run_brief
    from test_summarise import ScriptedModel

    first = run_brief(day_of_news, ScriptedModel(), tmp_path, datetime(2026, 11, 5, 18, 46),
                      max_events=0)
    second = run_brief(day_of_news, ScriptedModel(), tmp_path, datetime(2026, 11, 5, 19, 2),
                       max_events=0)
    assert first.brief_path != second.brief_path
    assert first.brief_path.exists() and second.brief_path.exists()


def test_run_brief_also_saves_the_fetched_items_for_later_diagnosis(tmp_path, day_of_news):
    from datetime import datetime

    from news_brief.run import run_brief
    from test_summarise import ScriptedModel

    result = run_brief(day_of_news, ScriptedModel(), tmp_path, datetime(2026, 11, 5, 18, 46),
                       max_events=0)
    rows = json.loads(result.items_path.read_text(encoding="utf-8"))

    assert result.items_path.name == "items-2026-11-05-1846.json"
    assert [r["item_id"] for r in rows] == [i.item_id for i in day_of_news]

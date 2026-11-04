import json
from datetime import date

from news_brief.model import BudgetedModel, ModelUnavailable
from news_brief.pipeline import build_brief, render_text

from test_summarise import ScriptedModel


def answer(*sentences):
    return json.dumps({"sentences": [{"text": t, "sources": list(s)} for t, s in sentences]})


GENEVA_OK = answer(
    ("Ceasefire talks resumed in Geneva after a two-week pause.", ["geneva1", "geneva3"]),
    ("Mediators said the talks would continue on Wednesday.", ["geneva2"]),
)
FLOOD_OK = answer(("Floods killed 14 people in northern Pakistan.", ["flood1", "flood2"]))


def test_good_model_output_becomes_a_brief_with_every_sentence_kept(day_of_news):
    model = ScriptedModel(GENEVA_OK, FLOOD_OK, answer(), answer())
    brief = build_brief(day_of_news, model, min_outlets=2)

    assert len(brief.sections) == 2
    assert brief.rejections == []
    assert brief.incomplete_reason is None
    assert sum(len(s.sentences) for s in brief.sections) == 3


def test_made_up_claims_are_dropped_and_logged_not_sent(day_of_news):
    lying_geneva = answer(
        ("Ceasefire talks resumed in Geneva.", ["geneva1"]),                    # fine
        ("Talks resumed after 40 days of fighting.", ["geneva1"]),              # invented number
        ("President Macron hosted the delegations.", ["geneva3"]),              # invented name
        ("Both sides signed a deal.", []),                                      # no source
    )
    lying_flood = answer(
        ("Floods killed 14 people in northern Pakistan.", ["flood1"]),          # fine
        ("Delegations returned to Geneva.", ["made-up-id"]),                    # fake source id
        ("Voters in Brazil head to the polls.", ["vote1"]),                     # other event's source
    )
    brief = build_brief(day_of_news, ScriptedModel(lying_geneva, lying_flood), min_outlets=2)

    assert [s.text for s in brief.sections[0].sentences] == ["Ceasefire talks resumed in Geneva."]
    assert [s.text for s in brief.sections[1].sentences] == [
        "Floods killed 14 people in northern Pakistan."]
    reasons = [r.rejection.reason for r in brief.rejections]
    assert len(reasons) == 5
    assert "number not in cited items: 40" in reasons
    assert any(r.startswith("name not in cited items") and "macron" in r for r in reasons)
    assert "no sources listed" in reasons
    assert any("made-up-id" in r for r in reasons)
    assert any("vote1" in r for r in reasons)

    text = render_text(brief, date(2026, 11, 5))
    for dropped in ("40 days", "Macron", "signed a deal", "Brazil", "Delegations returned"):
        assert dropped not in text


def test_event_with_no_surviving_sentence_is_left_out(day_of_news):
    all_bad = answer(("Talks collapsed after 99 hours.", ["geneva1"]))
    model = ScriptedModel(all_bad, FLOOD_OK)
    brief = build_brief(day_of_news, model, min_outlets=2)

    assert len(brief.sections) == 1
    assert brief.sections[0].lead.item_id == "flood1"
    assert brief.skipped_events == [
        ("Ceasefire talks resume in Geneva after two-week pause",
         "no sentence passed the source check"),
    ]


def test_unusable_model_output_skips_that_event_only(day_of_news):
    model = ScriptedModel("Sorry, I can't help with that.", FLOOD_OK)
    brief = build_brief(day_of_news, model, min_outlets=2)

    assert len(brief.sections) == 1
    assert len(brief.skipped_events) == 1
    assert "unusable model output" in brief.skipped_events[0][1]


def test_provider_refusal_stops_the_run_and_keeps_what_was_done(day_of_news):
    model = ScriptedModel(GENEVA_OK, ModelUnavailable("HTTP 429 quota exhausted"))
    brief = build_brief(day_of_news, model, min_outlets=2)

    assert len(brief.sections) == 1
    assert brief.incomplete_reason == "ModelUnavailable: HTTP 429 quota exhausted"
    assert len(model.prompts) == 2  # no retry loop


def test_call_budget_caps_the_number_of_model_calls(day_of_news):
    inner = ScriptedModel(GENEVA_OK, FLOOD_OK, answer(), answer())
    brief = build_brief(day_of_news, BudgetedModel(inner, max_calls=1), min_outlets=1)

    assert len(inner.prompts) == 1
    assert brief.incomplete_reason.startswith("BudgetExceeded")
    assert len(brief.sections) == 1


def test_max_events_limits_how_many_events_are_summarised(day_of_news):
    model = ScriptedModel(GENEVA_OK, FLOOD_OK)
    brief = build_brief(day_of_news, model, max_events=1, min_outlets=1)
    assert len(model.prompts) == 1
    assert len(brief.sections) == 1


def test_no_items_means_no_model_calls():
    model = ScriptedModel()
    brief = build_brief([], model)
    assert brief.sections == [] and model.prompts == []


def test_rendered_brief_credits_sources_with_outlet_and_link(day_of_news):
    brief = build_brief(day_of_news, ScriptedModel(GENEVA_OK, FLOOD_OK), min_outlets=2)
    text = render_text(brief, date(2026, 11, 5))

    assert text.startswith("WORLD BRIEF - 2026-11-05")
    assert "Ceasefire talks resume in Geneva after two-week pause (BBC)" in text
    assert "Ceasefire talks resumed in Geneva after a two-week pause. [1][2]" in text
    assert "Sources\n[1] BBC: Ceasefire talks resume in Geneva after two-week pause" in text
    assert "https://bbc.example/geneva1" in text
    assert "written by an AI model" in text


def test_rendered_brief_says_so_when_nothing_passed():
    text = render_text(build_brief([], ScriptedModel()), date(2026, 11, 5))
    assert "No events passed the source check today." in text


def test_rendered_brief_reports_removed_sentences_and_incomplete_runs(day_of_news):
    bad = answer(("Invented 77 claims.", ["geneva1"]))
    model = ScriptedModel(bad, ModelUnavailable("HTTP 429"))
    text = render_text(build_brief(day_of_news, model, min_outlets=2), date(2026, 11, 5))

    assert "Sentences removed by that check today: 1." in text
    assert "This brief is incomplete" in text

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
    brief = build_brief(day_of_news, ScriptedModel(lying_geneva, lying_flood), min_outlets=2,
                        sentences_per_event=4)

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
    assert "Ceasefire talks resumed in Geneva after a two-week pause. [1, 2]" in text
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


def test_model_sees_at_most_the_prompt_limit_and_cannot_cite_hidden_items():
    from news_brief.group import MAX_PROMPT_ITEMS
    from conftest import make_item

    items = [make_item(f"q{n:02d}", f"Outlet{n}", "Quake strikes coast", "Buildings fell.", hour=n % 20)
             for n in range(20)]
    hidden = "q19"   # latest of the identical items, so it is not among those shown
    model = ScriptedModel(answer(
        ("Quake strikes coast.", ["q00"]),
        ("Buildings fell.", [hidden]),
    ))
    brief = build_brief(items, model)

    shown = [line for line in model.prompts[0].splitlines() if line.startswith("[q")]
    assert len(shown) == MAX_PROMPT_ITEMS
    assert f"[{hidden}]" not in model.prompts[0]
    assert [s.text for s in brief.sections[0].sentences] == ["Quake strikes coast."]
    assert any(hidden in r.rejection.reason for r in brief.rejections)


# --- error policy and progress --------------------------------------------------

def test_an_overloaded_model_skips_one_event_and_carries_on(day_of_news):
    from news_brief.model import ModelError

    model = ScriptedModel(ModelError("HTTP 503 overloaded (after one retry)"), FLOOD_OK)
    brief = build_brief(day_of_news, model, min_outlets=2)

    assert brief.incomplete_reason is None
    assert len(brief.sections) == 1                      # the second event still made it
    assert brief.skipped_events[0][1].startswith("model error: HTTP 503")


def test_three_model_errors_in_a_row_end_the_run():
    from news_brief.model import ModelError
    from conftest import make_item

    items = []
    for n in range(5):                                   # five distinct two-outlet events
        items += [make_item(f"e{n}a", "A", f"Topic{n}x Topic{n}y happened", hour=1),
                  make_item(f"e{n}b", "B", f"Topic{n}x Topic{n}y occurred", hour=2)]
    model = ScriptedModel(*[ModelError("HTTP 503") for _ in range(5)])
    brief = build_brief(items, model, min_outlets=2)

    assert len(model.prompts) == 3
    assert brief.incomplete_reason.startswith("3 model errors in a row")


def test_a_success_resets_the_error_count(day_of_news):
    from news_brief.model import ModelError

    items = day_of_news
    model = ScriptedModel(ModelError("HTTP 503"), FLOOD_OK)
    brief = build_brief(items, model, min_outlets=2, max_consecutive_errors=2)
    assert brief.incomplete_reason is None


def test_progress_reports_each_event_and_its_outcome(day_of_news):
    lines = []
    build_brief(day_of_news, ScriptedModel(GENEVA_OK, FLOOD_OK), min_outlets=2,
                progress=lines.append)

    assert lines[0].startswith("7 items ->") and "up to 2 will be summarised" in lines[0]
    assert lines[1].startswith("[1/2] Ceasefire talks resume in Geneva")
    assert "    kept 2 sentence(s), removed 0" in lines
    assert any(line.startswith("[2/2]") for line in lines)


def test_rendered_brief_shows_outlets_without_their_feed_section():
    from conftest import make_item

    items = [
        make_item("g1", "The Guardian (World)", "Wong sentenced to ten years", "Prison term.", 7),
        make_item("h1", "The Hindu (International)", "Wong sentenced to ten years in prison",
                  "Prison term.", 8),
    ]
    model = ScriptedModel(answer(("Wong was sentenced to ten years.", ["g1", "h1"])))
    text = render_text(build_brief(items, model, min_outlets=2), date(2026, 11, 5))

    assert "Wong sentenced to ten years (The Guardian)" in text
    assert "[1] The Guardian: Wong sentenced to ten years" in text
    assert "[2] The Hindu: Wong sentenced to ten years in prison" in text
    assert "(World)" not in text and "(International)" not in text


# --- citations: capped inline, never dropped -------------------------------------

def test_format_marks_keeps_every_source_but_compresses_long_runs():
    from news_brief.pipeline import format_marks
    assert format_marks([1]) == "[1]"
    assert format_marks([2, 1]) == "[1, 2]"
    assert format_marks([1, 2, 3]) == "[1, 2, 3]"
    assert format_marks([1, 2, 3, 4, 5]) == "[1-5]"
    assert format_marks([1, 2, 3, 4, 7]) == "[1-4, 7]"
    assert format_marks([1, 2, 5, 6]) == "[1, 2, 5, 6]"
    assert format_marks([3, 4, 5, 9]) == "[3-5, 9]"
    assert format_marks([1, 1, 2]) == "[1, 2]"


def test_a_sentence_with_five_sources_shows_a_range_but_lists_all_five():
    from conftest import make_item

    items = [make_item(f"s{n}", f"Outlet{n}", "Wong sentenced to ten years", "Prison.", hour=n + 1)
             for n in range(5)]
    model = ScriptedModel(answer(("Wong was sentenced to ten years.", [i.item_id for i in items])))
    text = render_text(build_brief(items, model, min_outlets=2), date(2026, 11, 5))

    assert "Wong was sentenced to ten years. [1-5]" in text
    for number, outlet in enumerate(["Outlet0", "Outlet1", "Outlet2", "Outlet3", "Outlet4"], 1):
        assert f"[{number}] {outlet}: Wong sentenced to ten years" in text


# --- length: stop asking the model once the brief is long enough ---------------

def _events(n):
    from conftest import make_item
    items = []
    for k in range(n):
        items += [make_item(f"e{k}a", "A", f"Topic{k}x Topic{k}y happened today", hour=1),
                  make_item(f"e{k}b", "B", f"Topic{k}x Topic{k}y occurred today", hour=2)]
    return items


def test_the_run_stops_asking_the_model_when_the_word_budget_is_reached():
    items = _events(6)
    ten_words = "something quite notable happened here today as many outlets reported."
    model = ScriptedModel(*[answer((ten_words, [f"e{k}a"])) for k in range(6)])

    brief = build_brief(items, model, min_outlets=2, word_budget=30)

    assert len(model.prompts) < 6                      # fewer model calls were spent
    assert brief.words >= 30
    assert brief.left_for_length == 6 - len(model.prompts)
    assert "left out to keep this a short read" in render_text(brief, date(2026, 11, 5))


def test_no_budget_means_every_event_is_summarised():
    items = _events(4)
    model = ScriptedModel(*[answer(("Something happened.", [f"e{k}a"])) for k in range(4)])
    brief = build_brief(items, model, min_outlets=2)
    assert len(model.prompts) == 4 and brief.left_for_length == 0


def test_header_states_the_reading_time():
    from conftest import make_item
    items = [make_item("a", "A", "Wong sentenced", "Prison.", 1), make_item("b", "B", "Wong sentenced", "Prison.", 2)]
    text = render_text(build_brief(items, ScriptedModel(answer(("Wong was sentenced.", ["a"]))),
                                   min_outlets=2), date(2026, 11, 5))
    assert text.splitlines()[0] == "WORLD BRIEF - 2026-11-05  (about 1 min read)"


def test_a_brief_asks_for_two_short_sentences_and_cuts_a_longer_answer():
    items = _events(1)
    long_answer = answer(("something happened today.", ["e0a"]), ("more happened today.", ["e0a"]),
                         ("even more happened today.", ["e0a"]))
    model = ScriptedModel(long_answer)
    brief = build_brief(items, model, min_outlets=2)

    assert "Write 1 to 2 plain sentences" in model.prompts[0]
    assert len(brief.sections[0].sentences) == 2


# --- ranking: unranked outlets do not make an event look big -------------------------

def test_outlets_marked_unranked_do_not_count_towards_an_events_size():
    from conftest import make_item

    big = [make_item(f"big{n}", f"News{n}", "Parliament passes budget after long debate",
                     "Lawmakers voted late.", hour=n + 1) for n in range(3)]
    reports = [make_item(f"r{n}", name, "Programme strengthens outbreak ready evidence capabilities",
                         "A development report.", hour=n + 1)
               for n, name in enumerate(["ReliefWeb (Updates)", "ReliefWeb (Updates)", "Solo News"])]
    model = ScriptedModel(answer(("Parliament passed the budget.", ["big0"])))

    brief = build_brief(big + reports, model, min_outlets=2,
                        unranked_outlets={"ReliefWeb (Updates)"})

    assert len(model.prompts) == 1                    # the report group had only 1 ranked outlet
    assert brief.sections[0].lead.item_id == "big0"


def test_unranked_outlets_still_supply_items_to_an_event_that_qualifies():
    from conftest import make_item

    items = [make_item("n1", "BBC", "Cyclone hits coast", "Winds.", 1),
             make_item("n2", "Al Jazeera", "Cyclone hits the coast", "Winds.", 2),
             make_item("rw", "ReliefWeb (Updates)", "Cyclone hits coast: situation report", "Aid.", 3)]
    model = ScriptedModel(answer(("A cyclone hit the coast.", ["n1", "rw"])))
    brief = build_brief(items, model, min_outlets=2, unranked_outlets={"ReliefWeb (Updates)"})

    assert "[rw]" in model.prompts[0]
    assert [s.text for s in brief.sections[0].sentences] == ["A cyclone hit the coast."]


def test_models_that_answered_are_named_in_the_brief():
    from news_brief.model import BudgetedModel

    class Tracked(ScriptedModel):
        usage = {"gemini-a": 2, "gemini-b": 1}

    items = _events(1)
    model = BudgetedModel(Tracked(answer(("Something happened.", ["e0a"]))), max_calls=3)
    brief = build_brief(items, model, min_outlets=2)
    text = render_text(brief, date(2026, 11, 5))

    assert brief.models_used == {"gemini-a": 2, "gemini-b": 1}
    assert "Written by: gemini-a (2), gemini-b (1)." in text

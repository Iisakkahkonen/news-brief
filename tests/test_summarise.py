import json

import pytest

from news_brief.group import Cluster
from news_brief.model import BudgetExceeded, BudgetedModel, ModelUnavailable
from news_brief.summarise import (
    MAX_SENTENCES, BadModelOutput, Sentence, build_prompt, parse_response, summarise_cluster,
)

from conftest import make_item


class ScriptedModel:
    """Test double: returns canned answers in order and records the prompts."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


GOOD = json.dumps({"sentences": [
    {"text": "Floods killed 14 people.", "sources": ["flood1", "flood2"]},
    {"text": "Officials said 3,200 were displaced.", "sources": ["flood1"]},
]})


def cluster():
    return Cluster([
        make_item("flood1", "Dawn", "Floods kill 14 in northern Pakistan", "Heavy rain fell."),
        make_item("flood2", "The Hindu", "Pakistan floods: death toll reaches 14"),
    ])


def test_prompt_lists_every_item_with_id_and_outlet():
    prompt = build_prompt(cluster())
    assert "[flood1] (Dawn) Floods kill 14 in northern Pakistan - Heavy rain fell." in prompt
    assert "[flood2] (The Hindu) Pakistan floods: death toll reaches 14" in prompt


def test_prompt_states_the_grounding_rules():
    prompt = build_prompt(cluster())
    assert "ONLY the numbered items" in prompt
    assert "id of each item" in prompt
    assert "JSON only" in prompt


def test_prompt_contains_nothing_from_other_events():
    prompt = build_prompt(cluster())
    assert "Brazil" not in prompt


def test_valid_response_is_parsed_into_sentences():
    sentences = parse_response(GOOD)
    assert sentences == [
        Sentence("Floods killed 14 people.", ("flood1", "flood2")),
        Sentence("Officials said 3,200 were displaced.", ("flood1",)),
    ]


def test_response_wrapped_in_a_code_fence_is_accepted():
    assert len(parse_response(f"```json\n{GOOD}\n```")) == 2
    assert len(parse_response(f"```\n{GOOD}\n```")) == 2


def test_missing_sources_key_gives_empty_sources_for_the_verifier_to_reject():
    parsed = parse_response('{"sentences": [{"text": "No sources."}]}')
    assert parsed == [Sentence("No sources.", ())]


def test_extra_sentences_beyond_the_limit_are_ignored():
    many = json.dumps({"sentences": [{"text": f"S{n}.", "sources": ["a"]} for n in range(9)]})
    assert len(parse_response(many)) == MAX_SENTENCES


@pytest.mark.parametrize("bad", [
    "",
    "Sure! Here is your summary.",
    "{not json",
    "[]",
    '{"sentences": "text"}',
    '{"sentences": [{"sources": ["a"]}]}',
    '{"sentences": [{"text": 5, "sources": []}]}',
    '{"sentences": [{"text": "x", "sources": "a"}]}',
    '{"sentences": [{"text": "x", "sources": [1]}]}',
    '{"sentences": ["just a string"]}',
])
def test_malformed_responses_raise_bad_model_output(bad):
    with pytest.raises(BadModelOutput):
        parse_response(bad)


def test_summarise_cluster_sends_the_prompt_and_parses_the_answer():
    model = ScriptedModel(GOOD)
    sentences = summarise_cluster(cluster(), model)
    assert len(sentences) == 2
    assert len(model.prompts) == 1
    assert "[flood1]" in model.prompts[0]


def test_budgeted_model_stops_at_its_cap():
    inner = ScriptedModel("a", "b", "c")
    model = BudgetedModel(inner, max_calls=2)
    assert model.complete("1") == "a"
    assert model.complete("2") == "b"
    with pytest.raises(BudgetExceeded):
        model.complete("3")
    assert len(inner.prompts) == 2


def test_budget_must_allow_at_least_one_call():
    with pytest.raises(ValueError):
        BudgetedModel(ScriptedModel(), max_calls=0)


def test_provider_refusal_propagates_so_the_run_can_stop():
    model = ScriptedModel(ModelUnavailable("HTTP 429"))
    with pytest.raises(ModelUnavailable):
        summarise_cluster(cluster(), model)


def test_prompt_asks_for_exact_names_and_unchanged_certainty():
    prompt = build_prompt(cluster())
    assert "exactly as the items do" in prompt
    assert "do not add" in prompt and "first names" in prompt
    assert "expected, planned, alleged or claimed" in prompt
    assert "attribute a claim to an outlet" in prompt

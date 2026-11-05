"""Ask the model for a short summary of one event, as sentences tagged with sources.

The model is told to answer in JSON: a list of sentences, each with the IDs of
the feed items it is based on. Nothing the model returns is trusted here; the
parser only checks the shape, and verify.py checks the content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from news_brief.group import Cluster
from news_brief.model import ModelClient

MAX_SENTENCES = 4


class BadModelOutput(ValueError):
    """The model's answer was not the JSON shape we asked for."""


@dataclass(frozen=True)
class Sentence:
    text: str
    sources: tuple[str, ...]


RULES = """\
You write one short, neutral summary of a single news event for a daily brief.

Use ONLY the numbered items below. Do not use outside knowledge, and do not add
any name, place, number, date or detail that is not in the items.
Write 1 to {max_sentences} plain sentences in a neutral tone. Where outlets
disagree, say who said what (for example "X said ..., while Y said ...").
Report claims as claims: use "said" or "according to" for anything an outlet
attributes to a party, and never state a disputed claim as settled fact.
Refer to people, places and organisations exactly as the items do: do not add
first names, titles or descriptions that the items do not contain.
Keep the certainty and tense of the items. If an item says something is
expected, planned, alleged or claimed, do not write that it has happened.
To attribute a claim to an outlet, use the outlet name exactly as written in
the item list and cite that outlet's item.
Every sentence must list the id of each item it is based on.

Reply with JSON only, no other text, in exactly this shape:
{{"sentences": [{{"text": "...", "sources": ["id1", "id2"]}}]}}
"""


def build_prompt(cluster: Cluster) -> str:
    lines = [RULES.format(max_sentences=MAX_SENTENCES), "Items:"]
    for item in cluster.items:
        detail = f" - {item.summary}" if item.summary else ""
        lines.append(f"[{item.item_id}] ({item.outlet}) {item.title}{detail}")
    return "\n".join(lines)


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_response(text: str) -> list[Sentence]:
    cleaned = (text or "").strip()
    fenced = _FENCE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise BadModelOutput(f"not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("sentences"), list):
        raise BadModelOutput('expected an object with a "sentences" list')

    sentences = []
    for entry in data["sentences"][:MAX_SENTENCES]:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise BadModelOutput("each sentence needs a text string")
        sources = entry.get("sources", [])
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise BadModelOutput("sources must be a list of id strings")
        sentences.append(Sentence(text=entry["text"].strip(), sources=tuple(sources)))
    return sentences


def summarise_cluster(cluster: Cluster, model: ModelClient) -> list[Sentence]:
    return parse_response(model.complete(build_prompt(cluster)))

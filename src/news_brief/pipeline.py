"""Collect -> group -> summarise -> verify, and turn the result into text.

Nothing here talks to the network. Items come in, a model client is passed
in, and a Brief comes out. Every sentence in a Brief has passed verify.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from news_brief.feeds import Item
from news_brief.group import DEFAULT_THRESHOLD, group_items
from news_brief.model import ModelClient, ModelError
from news_brief.summarise import BadModelOutput, Sentence, summarise_cluster
from news_brief.verify import Rejection, verify


@dataclass
class Section:
    lead: Item  # its headline is shown verbatim, so it never comes from the model
    sentences: list[Sentence]
    items: dict[str, Item]  # every item of the event, for source links


@dataclass
class LoggedRejection:
    headline: str
    rejection: Rejection


@dataclass
class Brief:
    sections: list[Section] = field(default_factory=list)
    rejections: list[LoggedRejection] = field(default_factory=list)
    skipped_events: list[tuple[str, str]] = field(default_factory=list)  # (headline, why)
    incomplete_reason: str | None = None  # set when the model stopped the run early


def build_brief(items: list[Item], model: ModelClient, *, max_events: int = 12,
                min_outlets: int = 1, threshold: float = DEFAULT_THRESHOLD) -> Brief:
    brief = Brief()
    clusters = [c for c in group_items(items, threshold) if len(c.outlets) >= min_outlets]

    for cluster in clusters[:max_events]:
        headline = cluster.lead.title
        try:
            sentences = summarise_cluster(cluster, model)
        except BadModelOutput as exc:
            brief.skipped_events.append((headline, f"unusable model output: {exc}"))
            continue
        except ModelError as exc:
            brief.incomplete_reason = f"{type(exc).__name__}: {exc}"
            break

        accepted, rejected = verify(sentences, cluster.by_id())
        brief.rejections.extend(LoggedRejection(headline, r) for r in rejected)
        if accepted:
            brief.sections.append(Section(cluster.lead, accepted, cluster.by_id()))
        else:
            brief.skipped_events.append((headline, "no sentence passed the source check"))
    return brief


def render_text(brief: Brief, day: date) -> str:
    """Plain-text brief. Sources are numbered and credited at the bottom."""
    numbers: dict[str, int] = {}
    source_lines: list[str] = []

    def cite(item: Item) -> int:
        if item.item_id not in numbers:
            numbers[item.item_id] = len(numbers) + 1
            source_lines.append(f"[{numbers[item.item_id]}] {item.outlet}: {item.title}\n    {item.link}")
        return numbers[item.item_id]

    out = [f"WORLD BRIEF - {day.isoformat()}", ""]
    if not brief.sections:
        out += ["No events passed the source check today.", ""]
    for section in brief.sections:
        out.append(f"{section.lead.title} ({section.lead.outlet})")
        for sentence in section.sentences:
            marks = "".join(f"[{cite(section.items[s])}]" for s in sentence.sources)
            out.append(f"  {sentence.text} {marks}")
        out.append("")

    if source_lines:
        out += ["Sources", *source_lines, ""]
    out.append(
        "Summaries are written by an AI model from the headlines and summaries "
        "the outlets publish. Code removes any sentence whose sources, numbers "
        "or names cannot be found in those items."
    )
    removed = len(brief.rejections)
    if removed:
        out.append(f"Sentences removed by that check today: {removed}.")
    if brief.incomplete_reason:
        out.append(f"This brief is incomplete: the model stopped the run ({brief.incomplete_reason}).")
    return "\n".join(out) + "\n"

"""Collect -> group -> summarise -> verify, and turn the result into text.

Nothing here talks to the network. Items come in, a model client is passed
in, and a Brief comes out. Every sentence in a Brief has passed verify.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from news_brief.feeds import Item, outlet_label
from news_brief.group import DEFAULT_THRESHOLD, group_items
from news_brief.model import BudgetExceeded, ModelClient, ModelError, ModelUnavailable
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
                min_outlets: int = 1, threshold: float = DEFAULT_THRESHOLD,
                progress: Callable[[str], None] | None = None,
                max_consecutive_errors: int = 3) -> Brief:
    """Group, summarise and verify.

    Error policy: a quota, permission or budget problem ends the run at once.
    A one-off model error (for example Google being overloaded) skips that
    event and carries on, but several in a row end the run so that a
    permanent fault cannot keep hammering the service.
    """
    say = progress or (lambda message: None)
    brief = Brief()
    all_clusters = group_items(items, threshold)
    clusters = [c for c in all_clusters if len(c.outlets) >= min_outlets]
    chosen = clusters[:max_events]
    say(f"{len(items)} items -> {len(all_clusters)} groups; {len(clusters)} reported by at "
        f"least {min_outlets} outlet(s); summarising {len(chosen)}")

    errors_in_a_row = 0
    for number, cluster in enumerate(chosen, 1):
        cluster = cluster.for_prompt()
        headline = cluster.lead.title
        say(f"[{number}/{len(chosen)}] {headline[:80]} "
            f"({len(cluster.items)} items, {len(cluster.outlets)} outlets)")
        try:
            sentences = summarise_cluster(cluster, model)
        except BadModelOutput as exc:
            errors_in_a_row = 0  # the model did answer
            brief.skipped_events.append((headline, f"unusable model output: {exc}"))
            say("    unusable model answer, event skipped")
            continue
        except (ModelUnavailable, BudgetExceeded) as exc:
            brief.incomplete_reason = f"{type(exc).__name__}: {exc}"
            say(f"    stopping: {exc}")
            break
        except ModelError as exc:
            errors_in_a_row += 1
            brief.skipped_events.append((headline, f"model error: {exc}"))
            say(f"    model error, event skipped: {exc}")
            if errors_in_a_row >= max_consecutive_errors:
                brief.incomplete_reason = (
                    f"{errors_in_a_row} model errors in a row, last: {exc}")
                say("    stopping: too many model errors in a row")
                break
            continue

        errors_in_a_row = 0
        accepted, rejected = verify(sentences, cluster.by_id())
        brief.rejections.extend(LoggedRejection(headline, r) for r in rejected)
        say(f"    kept {len(accepted)} sentence(s), removed {len(rejected)}")
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
            source_lines.append(
                f"[{numbers[item.item_id]}] {outlet_label(item.outlet)}: {item.title}\n    {item.link}")
        return numbers[item.item_id]

    out = [f"WORLD BRIEF - {day.isoformat()}", ""]
    if not brief.sections:
        out += ["No events passed the source check today.", ""]
    for section in brief.sections:
        out.append(f"{section.lead.title} ({outlet_label(section.lead.outlet)})")
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

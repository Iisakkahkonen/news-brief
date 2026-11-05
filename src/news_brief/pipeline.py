"""Collect -> group -> summarise -> verify, and turn the result into text.

Nothing here talks to the network. Items come in, a model client is passed
in, and a Brief comes out. Every sentence in a Brief has passed verify.py.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import date

from news_brief.feeds import Item, outlet_label
from news_brief.group import DEFAULT_THRESHOLD, Cluster, group_items
from news_brief.model import BudgetExceeded, ModelClient, ModelError, ModelUnavailable
from news_brief.summarise import DEFAULT_SENTENCES, BadModelOutput, Sentence, summarise_cluster
from news_brief.verify import Rejection, verify

READING_WORDS_PER_MINUTE = 240
MAX_INLINE_SOURCES = 3  # more than this are shown as a range, e.g. [1-5], never dropped


@dataclass
class Section:
    lead: Item  # its headline is shown verbatim, so it never comes from the model
    sentences: list[Sentence]
    items: dict[str, Item]  # every item of the event, for source links

    @property
    def words(self) -> int:
        return len(self.lead.title.split()) + sum(len(s.text.split()) for s in self.sentences)


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
    left_for_length: int = 0  # events not summarised because the brief was long enough
    models_used: dict[str, int] = field(default_factory=dict)  # answers per model

    @property
    def words(self) -> int:
        return sum(section.words for section in self.sections)


def _ranked_outlets(cluster: Cluster, unranked: Collection[str]) -> set[str]:
    return cluster.outlets - set(unranked)


def build_brief(items: list[Item], model: ModelClient, *, max_events: int = 14,
                min_outlets: int = 1, threshold: float = DEFAULT_THRESHOLD,
                progress: Callable[[str], None] | None = None,
                max_consecutive_errors: int = 3,
                unranked_outlets: Collection[str] = (),
                sentences_per_event: int = DEFAULT_SENTENCES,
                word_budget: int | None = None) -> Brief:
    """Group, summarise and verify.

    Events are ranked by how many outlets report them, not counting
    ``unranked_outlets`` (for example a humanitarian-reports feed). Events are
    summarised in that order and the run stops once the brief holds
    ``word_budget`` words, so a short read costs few model calls.

    Error policy: a quota, permission or budget problem ends the run at once.
    A one-off model error (for example Google being overloaded) skips that
    event and carries on, but several in a row end the run so that a
    permanent fault cannot keep hammering the service.
    """
    say = progress or (lambda message: None)
    brief = Brief()
    all_clusters = group_items(items, threshold)
    ranked = [(c, _ranked_outlets(c, unranked_outlets)) for c in all_clusters]
    ranked = [(c, r) for c, r in ranked if len(r) >= min_outlets]
    ranked.sort(key=lambda pair: (-len(pair[1]), -len(pair[0].items)))
    chosen = [c for c, _ in ranked][:max_events]
    say(f"{len(items)} items -> {len(all_clusters)} groups; {len(ranked)} reported by at "
        f"least {min_outlets} outlet(s); up to {len(chosen)} will be summarised")

    errors_in_a_row = 0
    for number, cluster in enumerate(chosen, 1):
        if word_budget is not None and brief.words >= word_budget:
            brief.left_for_length = len(chosen) - (number - 1)
            say(f"The brief has reached about {brief.words} words; "
                f"{brief.left_for_length} more event(s) left out.")
            break
        cluster = cluster.for_prompt()
        headline = cluster.lead.title
        say(f"[{number}/{len(chosen)}] {headline[:80]} "
            f"({len(cluster.items)} items, {len(cluster.outlets)} outlets)")
        try:
            sentences = summarise_cluster(cluster, model, sentences_per_event)
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

    brief.models_used = dict(getattr(model, "usage", {}) or {})
    return brief


def format_marks(numbers: Collection[int]) -> str:
    """[1, 2] for a few sources; [1-5] or [1-3, 6] for many. Nothing is dropped."""
    ordered = sorted(set(numbers))
    if len(ordered) <= MAX_INLINE_SOURCES:
        return "[" + ", ".join(str(n) for n in ordered) + "]"
    parts, start, previous = [], ordered[0], ordered[0]
    for n in ordered[1:] + [None]:
        if n is not None and n == previous + 1:
            previous = n
            continue
        parts.append(str(start) if start == previous else
                     f"{start}, {previous}" if previous == start + 1 else f"{start}-{previous}")
        if n is not None:
            start = previous = n
    return "[" + ", ".join(parts) + "]"


def render_text(brief: Brief, day: date) -> str:
    """Plain-text brief. Every cited source is numbered and credited at the bottom."""
    numbers: dict[str, int] = {}
    source_lines: list[str] = []

    def cite(item: Item) -> int:
        if item.item_id not in numbers:
            numbers[item.item_id] = len(numbers) + 1
            source_lines.append(
                f"[{numbers[item.item_id]}] {outlet_label(item.outlet)}: {item.title}\n    {item.link}")
        return numbers[item.item_id]

    minutes = max(1, round(brief.words / READING_WORDS_PER_MINUTE))
    out = [f"WORLD BRIEF - {day.isoformat()}  (about {minutes} min read)", ""]
    if not brief.sections:
        out += ["No events passed the source check today.", ""]
    for section in brief.sections:
        out.append(f"{section.lead.title} ({outlet_label(section.lead.outlet)})")
        for sentence in section.sentences:
            marks = format_marks(cite(section.items[s]) for s in sentence.sources)
            out.append(f"  {sentence.text} {marks}")
        out.append("")

    if source_lines:
        out += ["Sources", *source_lines, ""]
    out.append(
        "Summaries are written by an AI model from the headlines and summaries "
        "the outlets publish. Code removes any sentence whose sources, numbers "
        "or names cannot be found in those items."
    )
    if brief.models_used:
        written = ", ".join(f"{name} ({count})" for name, count in sorted(brief.models_used.items()))
        out.append(f"Written by: {written}.")
    removed = len(brief.rejections)
    if removed:
        out.append(f"Sentences removed by that check today: {removed}.")
    if brief.left_for_length:
        out.append(f"{brief.left_for_length} more event(s) were left out to keep this a short read.")
    if brief.incomplete_reason:
        out.append(f"This brief is incomplete: the model stopped the run ({brief.incomplete_reason}).")
    return "\n".join(out) + "\n"

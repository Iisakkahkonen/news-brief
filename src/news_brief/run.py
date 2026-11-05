"""Run the whole pipeline once and write the results to files.

Two files per run: the brief itself, and a log of everything the source check
removed. The log is how the verifier is tuned and how the case study can show
what the model got wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from news_brief.feeds import Item
from news_brief.model import ModelClient
from news_brief.pipeline import Brief, build_brief, render_text


@dataclass
class RunResult:
    brief: Brief
    brief_path: Path
    log_path: Path


def render_log(brief: Brief, day: date) -> str:
    lines = [f"SOURCE CHECK LOG - {day.isoformat()}", ""]
    lines.append(f"Sentences removed: {len(brief.rejections)}")
    lines.append(f"Events left out: {len(brief.skipped_events)}")
    if brief.incomplete_reason:
        lines.append(f"Run stopped early: {brief.incomplete_reason}")
    lines.append("")

    for logged in brief.rejections:
        sentence = logged.rejection.sentence
        lines += [
            f"EVENT:   {logged.headline}",
            f"SENTENCE: {sentence.text}",
            f"SOURCES: {', '.join(sentence.sources) or '(none)'}",
            f"REASON:  {logged.rejection.reason}",
            "",
        ]
    for headline, why in brief.skipped_events:
        lines += [f"LEFT OUT: {headline}", f"REASON:   {why}", ""]
    return "\n".join(lines)


def run_brief(items: list[Item], model: ModelClient, out_dir: Path, when: datetime, *,
              max_events: int = 12, min_outlets: int = 2,
              progress: Callable[[str], None] | None = None) -> RunResult:
    """Files are stamped with the time too, so a second run never overwrites the first."""
    brief = build_brief(items, model, max_events=max_events, min_outlets=min_outlets,
                        progress=progress)
    day = when.date()
    stamp = when.strftime("%Y-%m-%d-%H%M")

    out_dir.mkdir(parents=True, exist_ok=True)
    brief_path = out_dir / f"brief-{stamp}.txt"
    log_path = out_dir / f"check-log-{stamp}.txt"
    brief_path.write_text(render_text(brief, day), encoding="utf-8")
    log_path.write_text(render_log(brief, day), encoding="utf-8")
    return RunResult(brief, brief_path, log_path)

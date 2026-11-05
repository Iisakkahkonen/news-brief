"""Command line entry point.

    python -m news_brief                  fetch the approved outlets, count items
    python -m news_brief --show-groups    ...and print how the items group into events
    python -m news_brief --save-items F   ...and save every item to a JSON file
    python -m news_brief --brief          write the AI brief (needs GEMINI_API_KEY)

Use --include-proposed to test feed URLs before approving them.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from news_brief.feeds import fetch_all, load_outlets, select
from news_brief.gemini import DEFAULT_MODELS, GeminiClient
from news_brief.group import group_items
from news_brief.model import BudgetedModel
from news_brief.pipeline import READING_WORDS_PER_MINUTE
from news_brief.run import run_brief, save_items

NO_KEY_MESSAGE = (
    "GEMINI_API_KEY is not set. Create a key (see the README), then in PowerShell:\n"
    '  $env:GEMINI_API_KEY = "your-key"\n'
    "and run this command again in the same window."
)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "outlets.toml"


def print_groups(items) -> None:
    clusters = group_items(items)
    multi = [c for c in clusters if len(c.items) > 1]
    print(f"{len(items)} items -> {len(clusters)} groups ({len(multi)} with more than one item)\n")
    for cluster in multi[:25]:
        print(f"== {len(cluster.items)} items from {len(cluster.outlets)} outlets")
        for item in cluster.items:
            print(f"   - {item.outlet}: {item.title}")
        print()


def make_brief(items, errors, args, unranked) -> int:
    api_key = os.environ["GEMINI_API_KEY"].strip()
    if errors:
        print(f"Note: {len(errors)} feed(s) failed: {', '.join(errors)}")

    def say(message: str) -> None:
        print(message, flush=True)

    client = GeminiClient(api_key, args.model, notify=say)
    model = BudgetedModel(client, max_calls=args.max_calls)
    budget = int(args.minutes * READING_WORDS_PER_MINUTE) if args.minutes > 0 else None
    say(f"Fetched {len(items)} items. Models, in order: {', '.join(client.models)}.")
    say(f"Aiming for about {args.minutes:g} minutes of reading; this takes a few minutes.")
    result = run_brief(items, model, args.out_dir, datetime.now(),
                       max_events=args.max_events, min_outlets=args.min_outlets,
                       progress=say, unranked_outlets=unranked, word_budget=budget)

    brief = result.brief
    print(f"\nModel calls used: {model.calls} of {args.max_calls} "
          f"({client.requests_made} requests sent to Google, retries included)")
    print(f"Events in brief: {len(brief.sections)}; sentences removed by the source check: "
          f"{len(brief.rejections)}; events left out: {len(brief.skipped_events)}")
    if brief.incomplete_reason:
        print(f"Run stopped early: {brief.incomplete_reason}")
    if model.usage:
        print("Answers by model: " + ", ".join(f"{k} x{v}" for k, v in sorted(model.usage.items())))
    print(f"Brief about {brief.words} words (about {brief.words / READING_WORDS_PER_MINUTE:.1f} min read)")
    print(f"Brief:     {result.brief_path}")
    print(f"Check log: {result.log_path}")
    print(f"Items:     {result.items_path}")
    return 0 if brief.sections else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--include-proposed", action="store_true",
                        help="also fetch outlets that are not approved yet")
    parser.add_argument("--show-groups", action="store_true",
                        help="fetch, then print the events the items are grouped into "
                             "(no AI model involved; use it to tune the grouping)")
    parser.add_argument("--save-items", type=Path, metavar="FILE",
                        help="write every fetched item (title, summary, link, date) to a "
                             "JSON file, so grouping can be tuned on the exact data")
    parser.add_argument("--brief", action="store_true",
                        help="write the brief with the Gemini model; reads the key from the "
                             "GEMINI_API_KEY environment variable")
    parser.add_argument("--model", default=",".join(DEFAULT_MODELS),
                        help="Gemini model(s) for --brief, comma-separated; if one is overloaded "
                             "the next is tried")
    parser.add_argument("--minutes", type=float, default=2.0,
                        help="target reading time of the brief; the run stops asking the model "
                             "for more events once it is long enough (0 = no limit)")
    parser.add_argument("--min-outlets", type=int, default=2,
                        help="only summarise events reported by at least this many outlets")
    parser.add_argument("--max-events", type=int, default=14,
                        help="most events considered (the reading-time target usually stops sooner)")
    parser.add_argument("--max-calls", type=int, default=15,
                        help="hard cap on model calls per run (keeps a run inside the free tier)")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    args = parser.parse_args()

    if args.brief and not os.environ.get("GEMINI_API_KEY", "").strip():
        print(NO_KEY_MESSAGE)
        return 2

    outlets = select(load_outlets(args.config), include_proposed=args.include_proposed)
    if not outlets:
        print("No outlets selected. Approve outlets in config/outlets.toml first.")
        return 0

    items, errors = fetch_all(outlets)
    if args.save_items:
        save_items(items, args.save_items)
        print(f"Saved {len(items)} items to {args.save_items}")
    if args.show_groups:
        print_groups(items)
        return 0
    if args.brief:
        unranked = {o.name for o in outlets if not o.ranks}
        return make_brief(items, errors, args, unranked)
    for outlet in outlets:
        count = sum(1 for item in items if item.outlet == outlet.name)
        status = "FAILED" if outlet.name in errors else f"{count} items"
        print(f"{outlet.name:<28} {status}")
    for name, reason in errors.items():
        print(f"  {name}: {reason}")
    print(f"Total: {len(items)} items from {len(outlets) - len(errors)} of {len(outlets)} feeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

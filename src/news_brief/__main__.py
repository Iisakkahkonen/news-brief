"""Command line entry point: python -m news_brief [--include-proposed]

Fetches the outlets listed in config/outlets.toml and prints a headline count
per outlet. Use --include-proposed to test feed URLs before approving them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from news_brief.feeds import fetch_all, load_outlets, select
from news_brief.group import group_items

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "outlets.toml"


def save_items(items, path: Path) -> None:
    rows = [
        {
            "item_id": i.item_id, "outlet": i.outlet, "title": i.title,
            "summary": i.summary, "link": i.link,
            "published": i.published.isoformat() if i.published else None,
        }
        for i in items
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def print_groups(items) -> None:
    clusters = group_items(items)
    multi = [c for c in clusters if len(c.items) > 1]
    print(f"{len(items)} items -> {len(clusters)} groups ({len(multi)} with more than one item)\n")
    for cluster in multi[:25]:
        print(f"== {len(cluster.items)} items from {len(cluster.outlets)} outlets")
        for item in cluster.items:
            print(f"   - {item.outlet}: {item.title}")
        print()


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
    args = parser.parse_args()

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

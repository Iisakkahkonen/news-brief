"""Command line entry point: python -m news_brief [--include-proposed]

Fetches the outlets listed in config/outlets.toml and prints a headline count
per outlet. Use --include-proposed to test feed URLs before approving them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from news_brief.feeds import fetch_all, load_outlets, select

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "outlets.toml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--include-proposed", action="store_true",
                        help="also fetch outlets that are not approved yet")
    args = parser.parse_args()

    outlets = select(load_outlets(args.config), include_proposed=args.include_proposed)
    if not outlets:
        print("No outlets selected. Approve outlets in config/outlets.toml first.")
        return 0

    items, errors = fetch_all(outlets)
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

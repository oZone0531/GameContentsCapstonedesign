"""Give existing generated files an info block and a readable name.

Usage (from the repository root):
    python migrate_artifact_names.py --dry-run          # only print the plan
    python migrate_artifact_names.py                     # do it
    python migrate_artifact_names.py --undo artifacts/migrations/migration-<time>.json

For every old module, report, replay, main-AI model, dataset and tensor cache
that has no info block yet, the script embeds one (kind, creation time, initial
conditions, summary, notes), writes a readable ``.about.md`` / ``ABOUT.md`` and
renames the file by the naming rule. Files that already have an info block are
skipped, so running it twice is safe. Nothing is deleted; the rename log is
written to artifacts/migrations/ and ``--undo`` restores the old names.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bang_rl.artifact_migration import apply_plan, plan_migration, undo_migration  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and change nothing")
    parser.add_argument("--undo", metavar="LOG", help="restore the old names recorded in a migration log")
    parser.add_argument("--root", default=None, help="project root to migrate (default: this repository)")
    args = parser.parse_args(argv)

    if args.undo:
        restored = undo_migration(args.undo)
        print(f"restored {len(restored)} names from {args.undo}")
        return 0
    plan = plan_migration(args.root)
    print(plan.render())
    if args.dry_run or not plan.to_migrate:
        return 0
    log = apply_plan(plan)
    print(f"done. Undo log: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

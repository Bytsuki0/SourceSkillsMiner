"""
Command-line entry point.

Preserves the exact CLI of the old ``ScoringSys.py`` (flags, defaults, banner,
output and the ``SSM_OUTPUT_DIR`` behaviour) so the Flask backend's subprocess
call keeps working unchanged. It simply builds a :class:`ScoringFacade` wired to
a :class:`ConsoleObserver` and runs it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from ssm.core.config import Config
from ssm.core.events import ConsoleObserver
from ssm.core.serialization import save_score_to_json
from ssm.scoring.facade import ScoringFacade

# Project root = parent of the ssm package, where json/ lives for local runs.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _reconfigure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GitHub User Scoring System")
    p.add_argument("--username", "-u", help="GitHub username (overrides config)")
    p.add_argument("--token", "-t", help="GitHub token (overrides config)")
    p.add_argument("--repo-limit", type=int, default=25,
                   help="Maximum repositories to analyze for sentiment (default: %(default)s)")
    p.add_argument("--skip-import-scan", action="store_true",
                   help="Skip import/package scan (faster, but less supplementary data)")
    p.add_argument("--import-max-repos", type=int, default=25,
                   help="Max repos for import scan (default: %(default)s)")
    p.add_argument("--import-max-files", type=int, default=30,
                   help="Max files per repo for import scan (default: %(default)s)")
    return p


def main() -> None:
    _reconfigure_stdio()

    args = _build_parser().parse_args()
    config = Config.from_file("config.ini")
    facade = ScoringFacade(config, observers=[ConsoleObserver()])

    try:
        print("=" * 70)
        print("GITHUB USER SCORING SYSTEM")
        print("=" * 70)
        print()

        summary = facade.analyze(
            username=args.username,
            token=args.token,
            repo_limit=args.repo_limit,
            include_import_scan=not args.skip_import_scan,
            import_scan_max_repos=args.import_max_repos,
            import_scan_max_files=args.import_max_files,
        )

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        saved_path = save_score_to_json(
            summary,
            username=args.username or config.username,
            base_dir=_PROJECT_ROOT,
        )
        print(f"\nSaved to: {saved_path}")

    except Exception as exc:
        print("Error computing score:", exc)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

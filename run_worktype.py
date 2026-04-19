"""
run_worktype.py

Standalone runner for GitHubLanguageCommitAnalyzer and GitHubImportScanner.

Reads credentials from config.ini in the current working directory
(RunWorkType.sh creates an isolated per-job config.ini so multiple
instances can run in parallel without colliding).

Output is written to the directory pointed to by the WTA_OUTPUT_DIR
environment variable, falling back to ./json_train/ if the variable
is not set.

Output format matches futuremint.json:
    {
        "username":       "...",
        "language_usage": { ... },   ← GitHubLanguageCommitAnalyzer
        "import_scan":    { ... }    ← GitHubImportScanner
    }

The areas / weights / final_score fields produced by ScoringSys are
intentionally absent — this file only covers the technology-stack
sections needed for training-data generation.

Usage (normally invoked by RunWorkType.sh, not directly):
    WTA_OUTPUT_DIR=/path/to/json_train python run_worktype.py
"""

import os
import sys
import json
import math
import configparser
from datetime import datetime
from typing import Dict, List, Any

# WorkTypeAnalyzer.py must be in the same directory as this script
# (or on sys.path).  RunWorkType.sh resolves the absolute path and
# sets cwd, so the import always works regardless of where the shell
# script is called from.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from WorkTypeAnalyzer import (
    GitHubLanguageCommitAnalyzer,
    GitHubImportScanner,
)


# ---------------------------------------------------------------------------
# Output directory — reads WTA_OUTPUT_DIR first, then falls back
# ---------------------------------------------------------------------------

def _resolve_output_dir() -> str:
    env_dir = os.environ.get("WTA_OUTPUT_DIR", "").strip()
    if env_dir:
        return env_dir
    return os.path.join(os.getcwd(), "json_train")


# ---------------------------------------------------------------------------
# Float sanitiser (nan / inf are invalid JSON)
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursively replace nan/inf floats with None before json.dump."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Section builders — mirror the shape used in futuremint.json exactly
# ---------------------------------------------------------------------------

def _build_language_usage(
    language_stats: Dict[str, List[int]],
) -> dict:
    """
    Convert raw GitHubLanguageCommitAnalyzer output into the language_usage
    section shape used throughout the project.

    Input:  {"Python": [lines, commits], "Go": [lines, commits], ...}
    Output: {
        "error": null,
        "languages": {...},
        "language_count": N,
        "total_commits": N,
        "total_lines": N,
        "top_5_languages": [{"language": ..., "lines": ..., "commits": ...}, ...]
    }
    """
    if not language_stats:
        return {
            "error": None,
            "languages": {},
            "language_count": 0,
            "total_commits": 0,
            "total_lines": 0,
            "top_5_languages": [],
        }

    total_commits = sum(v[1] for v in language_stats.values())
    total_lines   = sum(v[0] for v in language_stats.values())

    # Sort by commit count descending (mirrors futuremint.json ordering)
    sorted_langs = sorted(language_stats.items(), key=lambda x: -x[1][1])

    top_5 = [
        {"language": lang, "lines": stats[0], "commits": stats[1]}
        for lang, stats in sorted_langs[:5]
    ]

    return {
        "error":           None,
        "languages":       language_stats,
        "language_count":  len(language_stats),
        "total_commits":   total_commits,
        "total_lines":     total_lines,
        "top_5_languages": top_5,
    }


def _build_import_scan(
    raw_scan: dict,
    username: str,
) -> dict:
    """
    Wrap the raw GitHubImportScanner output in the import_scan envelope
    shape used in the project JSON files.
    """
    return {
        "error":                None,
        "username":             username,
        "analysis_date":        raw_scan.get("analysis_date"),
        "total_repos_analyzed": raw_scan.get("total_repos_analyzed", 0),
        "total_files_analyzed": raw_scan.get("total_files_analyzed", 0),
        "languages":            raw_scan.get("languages", {}),
        "repositories":         raw_scan.get("repositories", []),
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_profile(
    username:       str,
    language_usage: dict,
    import_scan:    dict,
    output_dir:     str,
) -> str:
    """
    Assemble and persist the profile JSON.
    Returns the absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)

    profile = _sanitize({
        "username":       username,
        "language_usage": language_usage,
        "import_scan":    import_scan,
    })

    file_path = os.path.join(output_dir, f"{username}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return file_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(
    username:          str,
    token:             str,
    output_dir:        str,
    import_max_repos:  int = 20,
    import_max_files:  int = 30,
) -> str:
    """
    Run both analyzers for one user and save the result.

    Parameters
    ----------
    username          : GitHub login.
    token             : Personal access token.
    output_dir        : Directory where the JSON file will be written.
    import_max_repos  : Maximum repositories to scan for imports.
    import_max_files  : Maximum files per repository for import scan.

    Returns
    -------
    Absolute path of the saved JSON file.
    """
    print(f"\n[run_worktype] Starting analysis for: {username}")
    print(f"[run_worktype] Output dir: {output_dir}")

    # ── Step 1: language usage ────────────────────────────────────────────
    print("\n[1/2] Analysing language usage…")
    try:
        lang_analyzer  = GitHubLanguageCommitAnalyzer(username, token)
        language_stats = lang_analyzer.analyze_language_usage()
        language_usage = _build_language_usage(language_stats)
        print(f"      {language_usage['language_count']} language(s) found, "
              f"{language_usage['total_commits']} total commit(s)")
    except Exception as exc:
        print(f"      [WARN] Language analysis failed: {exc}")
        language_usage = {
            "error":           str(exc),
            "languages":       {},
            "language_count":  0,
            "total_commits":   0,
            "total_lines":     0,
            "top_5_languages": [],
        }

    # ── Step 2: import / package scan ────────────────────────────────────
    print(f"\n[2/2] Scanning imports "
          f"(max {import_max_repos} repos, {import_max_files} files/repo)…")
    try:
        import_scanner = GitHubImportScanner(username, token)
        raw_scan       = import_scanner.analyze_imports(
            max_repos          = import_max_repos,
            max_files_per_repo = import_max_files,
        )
        import_scan = _build_import_scan(raw_scan, username)
        print(f"      {import_scan['total_repos_analyzed']} repo(s) analysed, "
              f"{import_scan['total_files_analyzed']} file(s) scanned")
    except Exception as exc:
        print(f"      [WARN] Import scan failed: {exc}")
        import_scan = {
            "error":                str(exc),
            "username":             username,
            "analysis_date":        None,
            "total_repos_analyzed": 0,
            "total_files_analyzed": 0,
            "languages":            {},
            "repositories":         [],
        }

    # ── Step 3: save ─────────────────────────────────────────────────────
    file_path = save_profile(username, language_usage, import_scan, output_dir)
    print(f"\n[run_worktype] Saved → {file_path}")
    return file_path


# ---------------------------------------------------------------------------
# Entry point — reads config.ini written by RunWorkType.sh
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")

    username = cfg.get("github", "username", fallback="").strip()
    token    = cfg.get("github", "token",    fallback="").strip()

    if not username or not token:
        print("ERROR: config.ini must contain [github] username and token", file=sys.stderr)
        sys.exit(1)

    output_dir = _resolve_output_dir()

    try:
        run(username, token, output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
ScoringSys.py

Scoring system that aggregates data from OSSanaliser, SentimentalAnaliser and StatusAnaliser
and returns scores in the range [-1, +1] for each area and a final aggregated score
(which is the weighted mean of the area scores by default). The weights are configurable
via the config.ini under section [scoring] or by passing a dict to score_user().

Also includes raw data from WorkTypeAnalyzer:
 - GitHubLanguageCommitAnalyzer  → language/commit distribution
 - GitHubImportScanner           → per-language package/import usage (no classification)

Design choices (defaults):
 - Areas: "OSS", "Status", "Sentiment", "Commitment"
 - Each area produces a score in [-1,1]. If there is no data for an area, the score is 0.0.
 - Final score = weighted mean of area scores (weights sum is normalized internally).
 - Sub-scores are computed using simple normalization heuristics (log-scale for counts,
   fraction for rates, capped streak normalization, etc.). Edit weights or normalization
   parameters to suit your project.

Usage examples:
    from ScoringSys import score_user
    summary = score_user(username, token)
    print(summary)
"""
import os
import math
import json
import statistics
import configparser as cfgparser
from typing import Dict, List, Optional

# local modules (provided by your project)
import OSSanaliser as f1
import SentimentalAnaliser as f2
import StatusAnaliser as f3
import WorkTypeAnalyzer as f4   # provides GitHubLanguageCommitAnalyzer + GitHubImportScanner

# Load defaults from config.ini if available
# Replace the top of the file with this — no module-level argparse needed

config = cfgparser.ConfigParser()
config.read('config.ini')               # reads from cwd, which RunParallel sets per-job
DEFAULT_USERNAME = config.get('github', 'username', fallback=None)
DEFAULT_TOKEN    = config.get('github', 'token',    fallback=None)

# Default weights (you can change these in config.ini under [scoring] or pass a dict)
DEFAULT_WEIGHTS = {
    'OSS':        1.0,
    'Status':     1.0,
    'Sentiment':  1.0,
    'Commitment': 1.0,
}

# ---------------------------------------------------------------------------
# Helper normalizers  →  produce values in [0,1], later converted to [-1,1]
# ---------------------------------------------------------------------------

def _safe_log_norm(x: float, scale: float = 100.0) -> float:
    """Logarithmic normalization: maps x in [0, +inf) to [0,1] using base-10 log.
    scale is an approximate value that maps to ~1.0 (for example scale=100 → log10(101)).
    If x==0 returns 0.0.
    """
    if x <= 0:
        return 0.0
    try:
        return min(1.0, math.log10(x + 1) / math.log10(scale + 1))
    except Exception:
        return 0.0


def _linear_cap(x: float, cap: float) -> float:
    """Simple linear normalization clipped to [0,1] by cap."""
    if x <= 0:
        return 0.0
    return min(1.0, x / cap)


def _to_signed01(x01: float) -> float:
    """Convert [0,1] to [-1,1]"""
    return max(-1.0, min(1.0, 2.0 * x01 - 1.0))


# ---------------------------------------------------------------------------
# Scored area computations
# ---------------------------------------------------------------------------

def compute_sentiment_score(username: str, token: str,
                            repo_full_names: Optional[List[str]] = None,
                            num_events: int = 200000) -> Dict:
    """Computes sentiment area score.

    Returns dict with keys: score (float in [-1,1]), details (dict per-repo sentiment).
    """
    try:
        f2.setup_nltk()
    except Exception:
        pass

    sentiments = {}

    for full in (repo_full_names or []):
        try:
            scores, reps = f2.get_user_activity_sentiment(full, num_events=num_events)
            if scores != 0.0:
                sentiments[full] = scores
        except Exception:
            pass

    upper = 0
    base  = 0

    for points in sentiments:
        if sentiments[points] > 0:
            base  += 1
            upper += 1
        if sentiments[points] < 0:
            base  += 1

    scr = upper / base if base > 0 else 0.0
    avg = scr

    return {'score': avg, 'details': sentiments}


def compute_oss_score(username: str, token: str, num_events: int = 1000000) -> Dict:
    """Computes OSS area score based on issues/PRs opened vs closed/merged and commit counts.
    Returns dict with keys: score, details.
    """
    try:
        opened = f1.get_user_opened_issues_and_prs(username, num_events)
    except Exception:
        opened = {'issues': [], 'prs': []}
    try:
        resolved = f1.get_user_resolved_issues_and_prs(username, num_events)
    except Exception:
        resolved = {'resolved_issues': [], 'closed_prs': [], 'merged_prs_count': 0}
    try:
        total_commits = f1.get_commit_stats_total(username)
    except Exception:
        total_commits = 0

    num_open_issues   = len(opened.get('issues', []))
    num_open_prs      = len(opened.get('prs', []))
    num_closed_issues = len(resolved.get('resolved_issues', []))
    num_closed_prs    = len(resolved.get('closed_prs', []))
    num_merged_prs    = resolved.get('merged_prs_count', 0)

    issue_total            = num_open_issues + num_closed_issues
    pr_total               = num_open_prs + num_closed_prs
    issue_resolution_rate  = (num_closed_issues / issue_total) if issue_total > 0 else 0.0
    pr_merge_rate          = (num_merged_prs / pr_total)       if pr_total    > 0 else 0.0
    commits_norm           = _safe_log_norm(total_commits, scale=200.0)

    subs = {
        'issue_resolution_rate': issue_resolution_rate,
        'pr_merge_rate':         pr_merge_rate,
        'commits_activity':      commits_norm,
    }

    if issue_total == 0 and pr_total == 0 and total_commits == 0:
        return {'score': 0.0, 'details': subs}

    avg01 = statistics.mean(list(subs.values()))
    score = _to_signed01(avg01)

    return {'score': score, 'details': subs}


def compute_commit_score(username: str, token: str) -> Dict:
    """Computes commitment area score based on advanced contributor analysis.
    Returns dict with keys: score, details.
    """
    try:
        analyzer        = f3.AdvancedContributorAnalyzer(username, token)
        commitment_json = analyzer.get_results_as_json()
        commitment      = json.loads(commitment_json)
    except Exception as e:
        return {
            'score': 0.0,
            'details': {'error': str(e), 'criteria_met': {}},
        }

    summary = commitment.get('summary')

    if summary is None:
        return {
            'score': 0.0,
            'details': {
                'error': "Missing 'summary' section in analysis result.",
                'criteria_met': {},
            },
        }

    required_keys = [
        'has_12_month_streak',
        'has_6_month_streak',
        'has_write_to_non_owned_repo',
        'has_repo_at_50th_percentile_commits',
        'at_75th_percentile_followers',
    ]

    total_points = 0
    criteria_met = {}

    for key in required_keys:
        if key not in summary or not isinstance(summary[key], bool):
            criteria_met[key] = False
            continue
        criteria_met[key]  = summary[key]
        total_points      += int(summary[key])

    normalized_score = total_points / len(required_keys)
    score            = _to_signed01(normalized_score)

    return {
        'score': normalized_score,
        'details': {
            'criteria_met': criteria_met,
            'total_points': total_points,
            'max_points':   len(required_keys),
            'percentage':   normalized_score * 100,
        },
    }


def compute_status_score(username: str, token: str) -> Dict:
    """Computes status area score using aggregate commit/lines/streak info from StatusAnaliser.
    Returns dict with keys: score, details.
    """
    try:
        analyzer = f3.GitHubStatsAnalyzerAllTime(username, token)
        agg      = analyzer.get_aggregate_stats()
    except Exception:
        agg = {
            'Linhas_trocas': 0,
            'Total_commits': 0,
            'Streak_contribuicoes_consecutivas': 0,
        }

    lines   = agg.get('Linhas_trocas',                          0)
    commits = agg.get('Total_commits',                          0)
    streak  = agg.get('Streak_contribuicoes_consecutivas',      0)

    lines_norm   = _safe_log_norm(lines,   scale=5000.0)
    commits_norm = _safe_log_norm(commits, scale=200.0)
    streak_norm  = _linear_cap(streak, cap=52)

    subs = {
        'lines_activity':   lines_norm,
        'commits_activity': commits_norm,
        'week_streak':      streak_norm,
    }

    if lines == 0 and commits == 0 and streak == 0:
        return {'score': 0.0, 'details': subs}

    avg01 = statistics.mean(list(subs.values()))
    score = _to_signed01(avg01)

    return {'score': score, 'details': subs}


# ---------------------------------------------------------------------------
# Raw data collectors (NOT scored, appended as-is to the final result)
# ---------------------------------------------------------------------------

def get_language_usage_data(username: str, token: str) -> Dict:
    """
    Collect language/commit distribution via GitHubLanguageCommitAnalyzer.

    Output shape (raw data, not scored):
        {
          "error": null | "...",
          "languages": { "Python": [lines, commits], ... },
          "language_count": N,
          "total_commits": N,
          "total_lines": N,
          "top_5_languages": [ {"language": "...", "lines": N, "commits": N}, ... ]
        }
    """
    try:
        print("Analyzing language usage...")
        analyzer       = f4.GitHubLanguageCommitAnalyzer(username, token)
        language_stats = analyzer.analyze_language_usage()

        if not language_stats:
            return {
                'error': None, 'languages': {},
                'language_count': 0, 'total_commits': 0, 'total_lines': 0,
            }

        total_commits = sum(stats[1] for stats in language_stats.values())
        total_lines   = sum(stats[0] for stats in language_stats.values())

        sorted_languages = sorted(
            language_stats.items(),
            key=lambda x: x[1][1],
            reverse=True,
        )

        return {
            'error':          None,
            'languages':      language_stats,
            'language_count': len(language_stats),
            'total_commits':  total_commits,
            'total_lines':    total_lines,
            'top_5_languages': [
                {'language': lang, 'lines': stats[0], 'commits': stats[1]}
                for lang, stats in sorted_languages[:5]
            ],
        }
    except Exception as e:
        return {
            'error': str(e), 'languages': {},
            'language_count': 0, 'total_commits': 0, 'total_lines': 0,
        }


def get_import_scan_data(username: str, token: str,
                         max_repos: int = 20,
                         max_files_per_repo: int = 30) -> Dict:
    """
    Scan repositories for import/package usage via GitHubImportScanner.

    No work-type classification is applied — results are grouped purely by
    language and list every external package found with its occurrence count.

    Output shape (raw data, not scored):
        {
          "error": null | "...",
          "username": "...",
          "analysis_date": "...",
          "total_repos_analyzed": N,
          "total_files_analyzed": N,
          "languages": {
            "Python": {
              "files_scanned": N,
              "packages": { "requests": 12, "numpy": 7, ... }
            },
            "TypeScript": { ... },
            ...
          },
          "repositories": [
            {
              "repository": "user/repo",
              "files_analyzed": N,
              "languages_found": ["Python", "TypeScript"]
            },
            ...
          ]
        }
    """
    try:
        print(
            f"Scanning imports/packages "
            f"(max {max_repos} repos, {max_files_per_repo} files/repo)..."
        )
        scanner     = f4.GitHubImportScanner(username, token)
        import_data = scanner.analyze_imports(
            max_repos=max_repos,
            max_files_per_repo=max_files_per_repo,
        )
        return {'error': None, **import_data}

    except Exception as e:
        return {
            'error':                str(e),
            'username':             username,
            'analysis_date':        None,
            'total_repos_analyzed': 0,
            'total_files_analyzed': 0,
            'languages':            {},
            'repositories':         [],
        }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_weights_from_config(cfg: cfgparser.ConfigParser = config) -> Dict[str, float]:
    """Reads weights from config.ini [scoring] section if present, otherwise returns defaults."""
    if 'scoring' not in cfg:
        return DEFAULT_WEIGHTS.copy()

    sec     = cfg['scoring']
    weights = {}
    for k in DEFAULT_WEIGHTS.keys():
        try:
            weights[k] = float(sec.get(k, DEFAULT_WEIGHTS[k]))
        except Exception:
            weights[k] = DEFAULT_WEIGHTS[k]
    return weights


def score_user(username: Optional[str] = None,
               token: Optional[str] = None,
               weights: Optional[Dict[str, float]] = None,
               repo_limit: Optional[int] = None,
               num_events_sentiment: int = 20000,
               include_import_scan: bool = True,
               import_scan_max_repos: int = 20,
               import_scan_max_files: int = 30) -> Dict:
    """Main entry point.

    Returns a summary dict:
      {
        'username': '...',
        'areas': {
            'OSS':        {'score': ..., 'details': {...}},
            'Status':     {'score': ..., 'details': {...}},
            'Sentiment':  {'score': ..., 'details': {...}},
            'Commitment': {'score': ..., 'details': {...}}
        },
        'weights':       {...},
        'final_score':   float in [-1, 1],
        'language_usage': {...},   # raw GitHubLanguageCommitAnalyzer data
        'import_scan':    {...}    # raw GitHubImportScanner data (packages by language)
      }

    If username/token are not provided the values in config.ini will be used.

    Args:
        username:             GitHub username.
        token:                GitHub personal access token.
        weights:              Custom weights for scoring areas (override config.ini).
        repo_limit:           Max repositories to analyze for sentiment.
        num_events_sentiment: Number of events to pull for sentiment analysis.
        include_import_scan:  Run GitHubImportScanner (can be slow; disable for speed).
        import_scan_max_repos:  Max repos for the import scanner.
        import_scan_max_files:  Max files per repo for the import scanner.
    """
    username = username or DEFAULT_USERNAME
    token    = token    or DEFAULT_TOKEN

    if not username:
        raise ValueError('username is required either as argument or in config.ini')

    # Build normalized weights
    cfg_weights = load_weights_from_config()
    if weights:
        cfg_weights.update(weights)
    total_w = sum(abs(v) for v in cfg_weights.values())
    if total_w == 0:
        normalized = {k: 1.0 / len(cfg_weights) for k in cfg_weights}
    else:
        normalized = {k: float(v) / total_w for k, v in cfg_weights.items()}

    # Repo list for sentiment
    try:
        _analyzer      = f3.GitHubStatsAnalyzerAllTime(username, token)
        repo_full_list = [r['full_name'] for r in _analyzer.repos]
        if repo_limit:
            repo_full_list = repo_full_list[:repo_limit]
    except Exception:
        repo_full_list = None

    # Scored areas
    print("Computing OSS score...")
    oss_res = compute_oss_score(username, token)

    print("Computing status score...")
    status_res = compute_status_score(username, token)

    print("Computing sentiment score...")
    sentiment_res = compute_sentiment_score(
        username, token,
        repo_full_names=repo_full_list,
        num_events=num_events_sentiment,
    )

    print("Computing commitment score...")
    commitment_res = compute_commit_score(username, token)

    areas = {
        'OSS':        oss_res,
        'Status':     status_res,
        'Sentiment':  sentiment_res,
        'Commitment': commitment_res,
    }

    # Weighted mean
    final = sum(areas[k]['score'] * normalized[k] for k in areas)

    # Raw supplementary data (not scored)
    print("\nCollecting supplementary data...")

    language_data = get_language_usage_data(username, token)

    if include_import_scan:
        import_scan_data = get_import_scan_data(
            username, token,
            max_repos=import_scan_max_repos,
            max_files_per_repo=import_scan_max_files,
        )
    else:
        print("Skipping import scan (pass --include-import-scan to enable).")
        import_scan_data = {
            'skipped': True,
            'reason':  'Import scan skipped by user request.',
        }

    print("\nAnalysis complete!")

    return {
        'username':       username,
        'areas':          areas,
        'weights':        normalized,
        'final_score':    float(max(-1.0, min(1.0, final))),
        'language_usage': language_data,
        'import_scan':    import_scan_data,
    }


def save_score_to_json(result: Dict, username: str) -> str:
    """
    Save the score_user result dict as a JSON file named
    '<username>.json' inside a 'json' folder located in the
    main project directory (same level as this file).

    Args:
        result (Dict): The result dict returned by score_user().
        username (str): Username used to generate the filename.

    Returns:
        str: Full path to the saved JSON file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(base_dir, 'json')
    os.makedirs(json_dir, exist_ok=True)

    filename  = f"{username}.json"
    file_path = os.path.join(json_dir, filename)

    print(f"\nSaving result for {username} → {filename}...")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return file_path
    except Exception as e:
        raise RuntimeError(f"Error saving JSON: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='GitHub User Scoring System')
    p.add_argument('--username', '-u', help='GitHub username (overrides config)')
    p.add_argument('--token',    '-t', help='GitHub token (overrides config)')
    p.add_argument('--repo-limit', type=int, default=200,
                   help='Maximum repositories to analyze for sentiment (default: %(default)s)')
    p.add_argument('--skip-import-scan', action='store_true',
                   help='Skip import/package scan (faster, but less supplementary data)')
    p.add_argument('--import-max-repos', type=int, default=20,
                   help='Max repos for import scan (default: %(default)s)')
    p.add_argument('--import-max-files', type=int, default=30,
                   help='Max files per repo for import scan (default: %(default)s)')

    args = p.parse_args()

    try:
        print('=' * 70)
        print('GITHUB USER SCORING SYSTEM')
        print('=' * 70)
        print()

        summary = score_user(
            username=args.username,
            token=args.token,
            repo_limit=args.repo_limit,
            include_import_scan=not args.skip_import_scan,
            import_scan_max_repos=args.import_max_repos,
            import_scan_max_files=args.import_max_files,
        )

        print('\n' + '=' * 70)
        print('RESULTS')
        print('=' * 70)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        saved_path = save_score_to_json(
            summary,
            username=args.username or DEFAULT_USERNAME,
        )
        print(f'\nSaved to: {saved_path}')

    except Exception as e:
        print('Error computing score:', e)
        import traceback
        traceback.print_exc()
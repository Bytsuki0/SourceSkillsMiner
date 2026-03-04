# SourceSkillsMiner

> **Automated GitHub contributor profiling and multi-dimensional scoring system.**

SourceSkillsMiner is a Python-based analytical pipeline that collects, processes, and scores GitHub user activity across four independent dimensions — Open-Source engagement, development status, communication sentiment, and long-term commitment — and enriches the result with raw language and package-usage data. A parallel execution layer (PowerShell and Bash) enables large-scale batch analysis across thousands of users.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Reference](#2-module-reference)
   - 2.1 [StatusAnaliser.py](#21-statusanaliserpy)
   - 2.2 [WorkTypeAnalyzer.py](#22-worktypeanalyzerpy)
   - 2.3 [OSSanaliser.py](#23-ossanaliserpy)
   - 2.4 [SentimentalAnaliser.py](#24-sentimentalanaliserpy)
   - 2.5 [ScoringSys.py](#25-scoringsyspy)
3. [Execution Layer](#3-execution-layer)
   - 3.1 [RunParallel.ps1 (Windows)](#31-runparallelps1-windows)
   - 3.2 [RunParallel.sh (Linux / macOS)](#32-runparallelsh-linux--macos)
4. [Data Flow](#4-data-flow)
5. [Configuration](#5-configuration)
6. [Performance Design Decisions](#6-performance-design-decisions)
7. [Scoring Model](#7-scoring-model)
8. [Known Limitations](#8-known-limitations)
9. [Dependencies](#9-dependencies)
10. [Directory Structure](#10-directory-structure)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Execution Layer                        │
│        RunParallel.ps1 / RunParallel.sh                 │
│   (creates isolated per-user config + dispatches jobs)  │
└────────────────────┬────────────────────────────────────┘
                     │  spawns N parallel processes
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   ScoringSys.py                         │
│              (main orchestration entry point)           │
└──┬──────────────┬──────────────┬────────────────┬───────┘
   │              │              │                │
   ▼              ▼              ▼                ▼
OSSanaliser  StatusAnaliser  Sentimental    WorkTypeAnalyzer
   .py            .py         Analiser.py       .py
                              
  GitHub REST API  ←──────────────────────────────────────
```

Each module communicates exclusively with the [GitHub REST API v3](https://docs.github.com/en/rest) using a personal access token. All modules are stateless between runs; results are persisted as per-user JSON files in the `json/` directory.

---

## 2. Module Reference

### 2.1 `StatusAnaliser.py`

Responsible for collecting raw commit activity metrics from GitHub's pre-aggregated contributor statistics endpoint (`/repos/{owner}/{repo}/stats/contributors`).

#### Classes

---

##### `AdvancedContributorAnalyzer`

Evaluates five binary criteria that indicate sustained, high-impact contributor behaviour.

| Method | Description |
|---|---|
| `_get_user_repositories()` | Paginates through all public repos for the user. |
| `_get_months_for_repo(repo)` | Fetches months with at least one commit for a single repo. Extracted to enable parallelism. |
| `_get_months_with_contributions()` | Dispatches `_get_months_for_repo` across all repos using a `ThreadPoolExecutor` (5 workers). Aggregates the union of all active months. |
| `_compute_monthly_streak(months)` | Computes the longest consecutive-month streak and the currently active streak from a `Set[str]` of `'YYYY-MM'` strings. |
| `check_monthly_contribution_streaks()` | Returns streak metadata including boolean flags for ≥12-month and ≥6-month streaks. |
| `check_pr_discussion_involvement(repo, threshold)` | For a given repo, checks what fraction of the user's PRs (past 12 months) contain at least one comment from the user themselves. Default threshold: 70 %. |
| `check_follower_percentile(sample_users)` | Estimates the user's follower percentile. When no sample is provided, uses a heuristic (`min(99, max(50, followers × 2))`). Pass a curated list for accurate results. |
| `check_write_rights_non_owned_repos()` | Inspects the `permissions` field of each repo in the user's repo list to detect push/admin rights on repos owned by others. |
| `_process_repo_percentile(repo)` | Fetches all contributor commit counts for a single repo and computes the user's percentile rank. Used by the thread pool in `check_commit_percentile_per_repo`. |
| `check_commit_percentile_per_repo()` | Runs `_process_repo_percentile` in parallel across all repos. Reports how many repos the user is at or above the 50th commit-percentile. |
| `analyze_all_criteria()` | Orchestrates all checks and returns a `summary` dict with five boolean keys consumed by `ScoringSys.py`. |

**Output summary keys:**

```python
{
  'has_12_month_streak':              bool,
  'has_6_month_streak':               bool,
  'has_write_to_non_owned_repo':      bool,
  'has_repo_at_50th_percentile_commits': bool,
  'at_75th_percentile_followers':     bool
}
```

---

##### `GitHubStatsAnalyzerAllTime`

Collects aggregate commit, line-change, and weekly-streak statistics across all of a user's repositories.

| Method | Description |
|---|---|
| `_estimate_lines_from_repo(repo)` | **Static.** Approximates total lines from the repo's `size` field (KB) using the formula `lines ≈ (size_kb × 1024) / 40`. Requires no API call. Accuracy note: counts entire repo size, not per-user contribution. |
| `_get_commit_stats(full_name)` | Fetches contributor stats for one repo with exponential backoff on HTTP 202. Results are stored in `self._stats_cache` so subsequent calls for the same repo are free. |
| `_process_repo(repo)` | Assembles a full stats dict for one repo (line estimate + commit stats). Designed to run inside a thread pool. |
| `analyze_all()` | Dispatches `_process_repo` for every repo in parallel (5 workers). Populates `_stats_cache` as a side effect. |
| `get_aggregate_stats()` | Calls `analyze_all()` and reduces to totals. Because `analyze_all()` uses the internal cache, calling both methods costs only one round of API calls. |

**Internal caching contract:**

```
analyze_all()         → populates _stats_cache
get_aggregate_stats() → calls analyze_all() → cache hit for every repo → 0 extra API calls
```

**`_get_commit_stats` exponential backoff:**

```
attempt 0 → sleep 1 s
attempt 1 → sleep 2 s
attempt 2 → sleep 4 s
attempt 3 → sleep 8 s
attempt 4+ → sleep 16 s (capped)
```

---

##### `GitHubLanguageCommitAnalyzer`

Distributes commit and line-change totals across programming languages by using the byte-count proportions reported by GitHub's `/languages` endpoint.

| Method | Description |
|---|---|
| `__init__(username, token, prefetched_stats)` | Accepts an optional `prefetched_stats` dict (keyed by short repo name) pre-populated from `GitHubStatsAnalyzerAllTime.analyze_all()`. Entries are stored in `_stats_cache`, eliminating redundant API calls. |
| `_get_commit_stats(full_name, repo)` | Cache-first lookup. On miss: uses `_fetch_user_stats_from_api` once to get both lines and commits, then optionally replaces the line count with the repo-size estimate if `repo` dict is provided. |
| `_fetch_user_stats_from_api(full_name)` | Single API call that extracts `(total_lines_changed, total_commits)` in **one pass** over the weeks array. Replaces the former two-function design that made two separate calls to the same endpoint. |
| `_process_repo(repo)` | Fetches languages and stats for one repo; designed for the thread pool. |
| `analyze_language_usage()` | Parallelises `_process_repo` across all repos and returns `Dict[str, [lines, commits]]`. |

**Language distribution formula:**

```
language_lines[lang]   += int(total_lines   × (lang_bytes / total_bytes))
language_commits[lang] += int(total_commits × (lang_bytes / total_bytes))
```

---

### 2.2 `WorkTypeAnalyzer.py`

Provides code-level analysis: import scanning and package-usage profiling across a user's repositories.

#### Classes

---

##### `GitHubLanguageCommitAnalyzer` *(also in WorkTypeAnalyzer)*

Identical in purpose to the one in `StatusAnaliser.py`. In `WorkTypeAnalyzer.py` this is the standalone version without the prefetch-cache integration.

---

##### `GitHubImportScanner`

Traverses repository file trees and extracts import/dependency statements from source files, grouped by language.

| Component | Description |
|---|---|
| `EXTENSION_TO_LANGUAGE` | Maps 30+ file extensions to canonical language labels. |
| `IMPORT_PATTERNS` | Per-language `re.Pattern` lists targeting the exact import syntax of each language (Python `import`/`from`, JS/TS `require`/`import`, Rust `use`/`extern crate`, etc.). |
| `GENERIC_IMPORT_KEYWORDS` | Fallback regex applied to any line that contains an import-like keyword, catching cases not covered by language-specific patterns. |
| `_normalise_package(raw, language)` | Strips raw match groups down to the top-level package name. Handles scoped npm packages (`@scope/pkg`), Python dotted paths, Java multi-segment paths, Rust `::` separators, and Dart `package:` URIs. |
| `_get_repo_tree(full_name)` | Tries `main`, `master`, then `develop` branches to retrieve the recursive file tree. |
| `_get_file_content(full_name, path)` | Fetches base64-encoded file content and decodes it. |
| `_extract_imports(content, language)` | Two-pass extraction: (1) language-specific patterns over the full file, (2) generic keyword sweep. |
| `analyze_imports(max_repos, max_files_per_repo)` | Main entry point. Returns a structured dict grouping packages by language with per-package occurrence counts. |

**Directories skipped during scan:**

`node_modules`, `vendor`, `dist`, `build`, `.git`, `__pycache__`, `.venv`, `venv`, `env`, `target`, `bin`, `obj`, `out`, `coverage`, `.nyc_output`

---

### 2.3 `OSSanaliser.py`

Collects open-source engagement signals by querying the GitHub Events and Search APIs.

| Function | Description |
|---|---|
| `get_user_opened_issues_and_prs(username, num_events)` | Returns `{'issues': [...], 'prs': [...]}` for events created by the user. |
| `get_user_resolved_issues_and_prs(username, num_events)` | Returns `{'resolved_issues': [...], 'closed_prs': [...], 'merged_prs_count': int}`. |
| `get_commit_stats_total(username)` | Returns a single integer representing total commits across all repos. |

> **Note:** These functions may return non-dict / non-int types on HTTP 403 (rate-limit or insufficient permissions). `ScoringSys.py` guards all three call sites with `isinstance` checks before use.

---

### 2.4 `SentimentalAnaliser.py`

Performs VADER-based sentiment analysis on GitHub event payloads (issue bodies, PR descriptions, comments) for a given repository.

| Function | Description |
|---|---|
| `setup_nltk()` | Downloads required NLTK corpora if not already present. |
| `get_user_activity_sentiment(repo_full_name, num_events)` | Returns `(score: float, representations: list)`. Score is positive for predominantly positive language, negative otherwise. |

---

### 2.5 `ScoringSys.py`

The top-level orchestrator. Calls all four analytical modules, normalises their outputs into `[-1, +1]` scores, computes a weighted final score, and persists the result.

#### Normalisation Helpers

| Function | Formula | Notes |
|---|---|---|
| `_safe_log_norm(x, scale)` | `log10(x+1) / log10(scale+1)`, clamped to `[0,1]` | Returns `0.0` for any non-numeric `x` (type guard against API failures) |
| `_linear_cap(x, cap)` | `min(1.0, x / cap)` | Same type guard |
| `_to_signed01(x)` | `2x − 1`, clamped to `[-1,1]` | Converts `[0,1]` to `[-1,1]` |

#### Scoring Functions

| Function | Inputs | Sub-scores |
|---|---|---|
| `compute_oss_score()` | OSSanaliser data | Issue resolution rate, PR merge rate, commit activity (log-normalised) |
| `compute_status_score()` | `GitHubStatsAnalyzerAllTime.get_aggregate_stats()` | Lines (log), commits (log), weekly streak (linear cap at 52 weeks) |
| `compute_sentiment_score()` | SentimentalAnaliser per-repo scores | Fraction of repos with positive sentiment |
| `compute_commit_score()` | `AdvancedContributorAnalyzer.analyze_all_criteria()` | Boolean criteria: 5 criteria, each worth 1 point, normalised to `[0,1]` |

#### `score_user()` — main entry point

```python
summary = score_user(
    username         = 'octocat',
    token            = 'ghp_...',
    repo_limit       = 200,          # max repos for sentiment
    include_import_scan = True,
    import_scan_max_repos  = 20,
    import_scan_max_files  = 30,
)
```

**Return shape:**

```json
{
  "username": "octocat",
  "areas": {
    "OSS":        { "score": 0.42, "details": { ... } },
    "Status":     { "score": 0.61, "details": { ... } },
    "Sentiment":  { "score": 0.75, "details": { ... } },
    "Commitment": { "score": 0.60, "details": { ... } }
  },
  "weights": { "OSS": 0.25, "Status": 0.25, "Sentiment": 0.25, "Commitment": 0.25 },
  "final_score": 0.59,
  "language_usage": { ... },
  "import_scan":    { ... }
}
```

#### `save_score_to_json(result, username)`

Writes the result to `<project_root>/json/<username>.json`. Uses `os.path.abspath(__file__)` so the output path is always relative to where `ScoringSys.py` lives, not the per-job temporary working directory created by the execution layer.

---

## 3. Execution Layer

### 3.1 `RunParallel.ps1` (Windows)

Runs up to `$MaxParallel` instances of `ScoringSys.py` simultaneously using PowerShell `Start-Job`.

**Key design choices:**

- Uses the venv's `python.exe` directly (`$VenvPython`) instead of activating the venv, because `Start-Job` spawns isolated sessions that do not inherit the parent's activated environment.
- Each job runs in its own temp directory (`%TEMP%\ssminer_{username}\`) containing a freshly written `config.ini`. This means `ScoringSys.py` finds its config through the standard `config.read('config.ini')` call with no argument changes required.
- `Wait-ForSlot` polls every 500 ms and only dispatches a new job when the running count is below `$MaxParallel`.
- Temp directories are deleted automatically after each job completes.

**`$MaxParallel` guidance:** At 5,000 API requests/hour per token, 4 parallel workers consuming ~200 requests each per user will exhaust the limit after roughly 6 users/hour. Adjust based on token count and user profile size.

---

### 3.2 `RunParallel.sh` (Linux / macOS)

Equivalent Bash implementation with an additional **token cycling** feature.

**Token cycling:**

Three tokens are read from `config_main.ini` under the keys `token`, `token_1`, and `token_2`. A `job_index` counter increments with every dispatched job; the active token is selected by `job_index % 3`:

```
job 0 → token
job 1 → token_1
job 2 → token_2
job 3 → token
job 4 → token_1
…
```

This distributes API rate-limit consumption evenly across three accounts, effectively tripling the sustained request throughput.

**Required `config_main.ini` entries:**

```ini
[github]
token   = ghp_firstToken
token_1 = ghp_secondToken
token_2 = ghp_thirdToken
```

**Other details:**

- Uses `mktemp -d` for isolated per-job directories.
- `reap_jobs` checks `kill -0 $pid` (non-destructive signal) to detect completion without blocking.
- `trap _cleanup_on_exit INT TERM` ensures child processes are killed on Ctrl-C.
- Username strings are sanitised with `sed` before use in directory names.

---

## 4. Data Flow

```
RunParallel
  │
  ├── writes  /tmp/ssminer_{user}/config.ini
  └── cd /tmp/ssminer_{user}/ && python ScoringSys.py
                │
                ├── OSSanaliser          → issues, PRs, commit count
                ├── StatusAnaliser
                │     ├── GitHubStatsAnalyzerAllTime   → lines, commits, streak
                │     └── AdvancedContributorAnalyzer  → 5 boolean criteria
                ├── SentimentalAnaliser  → per-repo VADER scores
                └── WorkTypeAnalyzer
                      ├── GitHubLanguageCommitAnalyzer → language distribution
                      └── GitHubImportScanner          → package usage
                │
                └── score_user()
                      ├── compute_oss_score()
                      ├── compute_status_score()
                      ├── compute_sentiment_score()
                      ├── compute_commit_score()
                      └── save_score_to_json()
                            └── <project_root>/json/{username}.json
```

---

## 5. Configuration

### `config.ini` (per-job, auto-generated)

```ini
[github]
username = octocat
token    = ghp_...

[scoring]          # optional — overrides default weights
OSS        = 1.0
Status     = 1.0
Sentiment  = 1.0
Commitment = 1.0
```

### `config_main.ini` (shared, read by execution layer only)

```ini
[github]
token   = ghp_primaryToken
token_1 = ghp_secondaryToken   # required by RunParallel.sh
token_2 = ghp_tertiaryToken    # required by RunParallel.sh
```

### `github_users.txt`

Plain text, one username per line. Lines starting with `#` and blank lines are ignored.

```
# Ruby core contributors
evanphx
matz
tenderlove
```

---

## 6. Performance Design Decisions

| Decision | Rationale |
|---|---|
| `ThreadPoolExecutor(max_workers=5)` inside each class | GitHub's secondary rate limit penalises bursts; 5 concurrent requests per process is a safe ceiling. With `MaxParallel=4` processes this yields up to 20 concurrent connections. |
| Exponential backoff on HTTP 202 | GitHub computes contributor stats asynchronously. A flat 2 s sleep per retry caused up to 20 s of blocking per repo; exponential backoff (1, 2, 4, 8, 16 s) reduces average wait significantly on warm caches. |
| Internal `_stats_cache` in `GitHubStatsAnalyzerAllTime` | `analyze_all()` and `get_aggregate_stats()` previously made two independent full sweeps of all repos. The cache ensures the endpoint is hit once per repo per process lifetime. |
| Repo-size line estimation | The `/stats/contributors` endpoint is the single slowest call in the pipeline (202 retries, one per repo). Replacing per-user line summation with a repo-size estimate (`size_kb × 1024 / 40`) eliminates line-counting as a reason to call this endpoint. |
| `prefetched_stats` in `GitHubLanguageCommitAnalyzer` | Allows the caller to inject stats already collected by `GitHubStatsAnalyzerAllTime`, avoiding a second pass over the same contributor-stats endpoint for the language distribution calculation. |
| `_fetch_user_stats_from_api` (single-pass) | Replaced two methods (`_fetch_commit_count_from_api`, `_fetch_commit_stats_from_api`) that called the same endpoint and iterated the same `weeks` array separately. One call, one loop, both values. |
| Isolated per-job temp directories | Prevents `config.ini` file collisions when multiple `ScoringSys.py` processes run concurrently on the same machine. Each process reads its own config from its own working directory. |
| Token cycling (Bash runner) | Distributes API usage across up to three GitHub accounts, multiplying the effective rate limit by the number of tokens. |

---

## 7. Scoring Model

### Area scores

Each area independently produces a score in `[-1, +1]`. A score of `0.0` is returned when no data is available for that area (graceful degradation).

| Area | Source | Method |
|---|---|---|
| **OSS** | OSSanaliser | Mean of: issue resolution rate, PR merge rate, log-normalised commit count. Converted to `[-1,1]`. |
| **Status** | StatusAnaliser | Mean of: log-normalised lines, log-normalised commits, linearly capped weekly streak (cap = 52 weeks). Converted to `[-1,1]`. |
| **Sentiment** | SentimentalAnaliser | Fraction of repos with positive sentiment score. Range naturally `[0,1]`; used directly as score. |
| **Commitment** | AdvancedContributorAnalyzer | Count of satisfied boolean criteria (max 5), divided by 5, converted to `[-1,1]`. |

### Final score

```
final_score = Σ (area_score[k] × normalised_weight[k])   for k in {OSS, Status, Sentiment, Commitment}
```

Weights are normalised so they always sum to 1, regardless of the raw values in `config.ini`.

### Type safety in normalisation

All normaliser functions (`_safe_log_norm`, `_linear_cap`) perform an `isinstance(x, (int, float))` check before any arithmetic. This prevents a `TypeError` crash when an upstream API call returns a list, dict, or `None` due to a 403 rate-limit or permission error — the score for that sub-dimension silently becomes `0.0` and processing continues.

---

## 8. Known Limitations

| Limitation | Impact | Notes |
|---|---|---|
| Repo-size line estimate counts all contributors | `Linhas_trocas` overestimates the target user's contribution on shared repos | Acceptable for ranking purposes; exact per-user line counts require the full contributor-stats call |
| Follower percentile heuristic | `estimated_percentile = min(99, max(50, followers × 2))` is an approximation | Provide `sample_users` to `check_follower_percentile()` for accurate results |
| Sentiment analysis scope | Only analyses repos the user owns/has access to | Contributions to external repos are not captured |
| GitHub API visibility | Only public repositories are accessible without additional OAuth scopes | Private repo contributions are invisible |
| Token cycling distributes load but not identity | All three tokens must belong to accounts with access to the target repos | Public repos: any valid token works |
| `max_retries = 100000` in original `_get_months_with_contributions` | Could block indefinitely on a permanently unavailable repo | Replaced with shared `_fetch_contributor_stats` helper capped at 10 retries with exponential backoff |

---

## 9. Dependencies

```
requests         # HTTP client for all GitHub API calls
nltk             # VADER sentiment analysis (SentimentalAnaliser)
```

Python standard library modules used: `os`, `math`, `json`, `re`, `base64`, `time`, `statistics`, `configparser`, `datetime`, `collections.defaultdict`, `concurrent.futures`, `typing`.

**Python version:** 3.8+ (required for `f-strings`, `walrus operator` support, and `as_completed` with `dict` unpacking).

**Install:**

```bash
pip install requests nltk
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 10. Directory Structure

```
SourceSkillsMiner/
│
├── ScoringSys.py              # Main orchestrator and CLI entry point
├── StatusAnaliser.py          # Commit stats, streaks, contributor analysis
├── WorkTypeAnalyzer.py        # Language distribution + import scanner
├── OSSanaliser.py             # Issue/PR/commit collection
├── SentimentalAnaliser.py     # VADER sentiment analysis
│
├── RunParallel.ps1            # Windows parallel runner
├── RunParallel.sh             # Linux/macOS parallel runner (with token cycling)
│
├── config_main.ini            # Shared config: tokens for the execution layer
├── github_users.txt           # Input: one GitHub username per line
│
├── json/                      # Output: one JSON file per processed user
│   ├── octocat.json
│   └── ...
│
└── win_venv/  (or venv/)      # Python virtual environment
    └── Scripts/python.exe
```

---

## Usage

### Single user (CLI)

```bash
python ScoringSys.py --username octocat --repo-limit 50 --skip-import-scan
```

### Batch (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File .\RunParallel.ps1
```

### Batch (Linux / macOS)

```bash
chmod +x RunParallel.sh
./RunParallel.sh
```

### As a library

```python
from ScoringSys import score_user, save_score_to_json

result = score_user(username='octocat', token='ghp_...')
print(result['final_score'])
save_score_to_json(result, 'octocat')
```

---

## Output Schema

Each `json/<username>.json` file follows this schema:

```jsonc
{
  "username": "string",
  "areas": {
    "OSS": {
      "score": "float [-1, 1]",
      "details": {
        "issue_resolution_rate": "float [0, 1]",
        "pr_merge_rate":         "float [0, 1]",
        "commits_activity":      "float [0, 1]"
      }
    },
    "Status": {
      "score": "float [-1, 1]",
      "details": {
        "lines_activity":   "float [0, 1]",
        "commits_activity": "float [0, 1]",
        "week_streak":      "float [0, 1]"
      }
    },
    "Sentiment": {
      "score": "float [0, 1]",
      "details": { "<repo_full_name>": "float" }
    },
    "Commitment": {
      "score": "float [0, 1]",
      "details": {
        "criteria_met": {
          "has_12_month_streak":              "bool",
          "has_6_month_streak":               "bool",
          "has_write_to_non_owned_repo":      "bool",
          "has_repo_at_50th_percentile_commits": "bool",
          "at_75th_percentile_followers":     "bool"
        },
        "total_points": "int [0, 5]",
        "max_points":   5,
        "percentage":   "float [0, 100]"
      }
    }
  },
  "weights": {
    "OSS": "float", "Status": "float",
    "Sentiment": "float", "Commitment": "float"
  },
  "final_score": "float [-1, 1]",
  "language_usage": {
    "error": "string | null",
    "languages": { "<Language>": ["lines_int", "commits_int"] },
    "language_count": "int",
    "total_commits":  "int",
    "total_lines":    "int",
    "top_5_languages": [
      { "language": "string", "lines": "int", "commits": "int" }
    ]
  },
  "import_scan": {
    "error": "string | null",
    "total_repos_analyzed": "int",
    "total_files_analyzed": "int",
    "languages": {
      "<Language>": {
        "files_scanned": "int",
        "packages": { "<package_name>": "int (occurrence_count)" }
      }
    },
    "repositories": [
      {
        "repository":      "string",
        "files_analyzed":  "int",
        "languages_found": ["string"]
      }
    ]
  }
}
```

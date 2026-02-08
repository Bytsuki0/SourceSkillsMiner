"""
ScoringSys.py

Scoring system that aggregates data from OSSanaliser, SentimentalAnaliser and StatusAnaliser
and returns scores in the range [-1, +1] for each area and a final aggregated score
(which is the weighted mean of the area scores by default). The weights are configurable
via the config.ini under section [scoring] or by passing a dict to score_user().

Design choices (defaults):
 - Areas: "OSS", "Status", "Sentiment"
 - Each area produces a score in [-1,1]. If there is no data for an area, the score is 0.0.
 - Final score = weighted mean of area scores (weights sum is normalized internally).
 - Sub-scores are computed using simple normalization heuristics (log-scale for counts,
   fraction for rates, capped streak normalization, etc.). Edit weights or normalization
   parameters to suit your project.

Usage examples:
    from ScoringSys import score_usersudo a
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

# Load defaults from config.ini if available
config = cfgparser.ConfigParser()
config.read('config.ini')
DEFAULT_USERNAME = config.get('github', 'username', fallback=None)
DEFAULT_TOKEN = config.get('github', 'token', fallback=None)

# Default weights (you can change these in config.ini under [scoring] or pass a dict)
DEFAULT_WEIGHTS = {
    'OSS': 1.0,
    'Status': 1.0,
    'Sentiment': 1.0
}

# Helper normalizers -> produce values in [0,1], later converted to [-1,1]

def _safe_log_norm(x: float, scale: float = 100.0) -> float:
    """Logarithmic normalization: maps x in [0, +inf) to [0,1] using base-10 log.
    scale is an approximate value that maps to ~1.0 (for example scale=100 -> log10(101)).
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



def compute_sentiment_score(username: str, token: str, repo_full_names: Optional[List[str]] = None,
                            num_events: int = 10) -> Dict:
    """Computes sentiment area score.

    Returns dict with keys: score (float in [-1,1]), details (dict per-repo sentiment)
    """
    try:
        f2.setup_nltk()
    except Exception:
        # If NLTK setup fails, continue; f2.get_user_activity_sentiment may still work
        pass

    sentiments = {}

    #try:
    #    analyzer = f3.GitHubStatsAnalyzerAllTime(username, token)
    #    repo_full_names = [f"{username}/{repo_name}" for repo_name in analyzer.repos]
    #except Exception:
    #    repo_full_names = []
    #if not repo_full_names:
    #   try:
    #        analyzer = f3.GitHubStatsAnalyzerAllTime(username, token)
    #        repo_full_names = [f"{username}/{repo_name}" for repo_name in analyzer.repos]
    #    except Exception:
    #       repo_full_names = []

    for full in (repo_full_names or []):
        try:
            scores, reps = f2.get_user_activity_sentiment(full, num_events=num_events)
            if scores != 0.0:
                score =+ scores
                sentiments[full] = scores
        except Exception:
            score =+ 0.0

    upper = 0 
    base = 0 
    
    
    for points in sentiments:
        if sentiments[points] > 0:
            base = base + 1
            upper= upper + 1
        if sentiments[points] < 0:
            base = base + 1

    scr = upper / base if base > 0 else 0.0
    
    avg = scr
    #if sentiments:
    #    avg = statistics.mean(sentiments.values())
    #else:
    #    avg = 0.0

    return {'score': avg, 'details': sentiments}


def compute_oss_score(username: str, token: str, num_events: int = 100) -> Dict:
    """Computes OSS area score based on issues/PRs opened vs closed/merged and commit counts.
    Returns dict with keys: score, details
    """
    try:
        opened = f1.get_user_opened_issues_and_prs(username, num_events)
    except Exception:
        opened = {'issues': [], 'prs': []}
    try:
        resolved = f1.get_user_resolved_issues_and_prs(username, num_events)
    except Exception:
        resolved = {'resolved_issues': [], 'closed_prs': [], 'merged_prs_count': 0}

    # Commit stats (total public commits)
    try:
        total_commits = f1.get_commit_stats_total(username)
    except Exception:
        total_commits = 0

    num_open_issues = len(opened.get('issues', []))
    num_open_prs = len(opened.get('prs', []))
    num_closed_issues = len(resolved.get('resolved_issues', []))
    num_closed_prs = len(resolved.get('closed_prs', []))
    num_merged_prs = resolved.get('merged_prs_count', 0)

    # Resolution rates
    issue_total = num_open_issues + num_closed_issues
    pr_total = num_open_prs + num_closed_prs
    issue_resolution_rate = (num_closed_issues / issue_total) if issue_total > 0 else 0.0
    pr_merge_rate = (num_merged_prs / pr_total) if pr_total > 0 else 0.0

    # Normalize commit count using log-scale (scale chosen heuristically; edit as needed)
    commits_norm = _safe_log_norm(total_commits, scale=200.0)  # 200 commits maps near 1.0

    # Combine sub-scores: resolution rates are already in [0,1], commits_norm in [0,1]
    # We'll give equal weight to (issue resolution, PR merge rate, commits activity)
    subs = {
        'issue_resolution_rate': issue_resolution_rate,
        'pr_merge_rate': pr_merge_rate,
        'commits_activity': commits_norm
    }

    # If there is no meaningful data at all, return neutral 0.0
    if issue_total == 0 and pr_total == 0 and total_commits == 0:
        return {'score': 0.0, 'details': subs}

    # average of subs
    avg01 = statistics.mean(list(subs.values()))
    score = _to_signed01(avg01)

    return {'score': score, 'details': subs}


def compute_status_score(username: str, token: str) -> Dict:
    """Computes status area score using aggregate commit/lines/streak info from StatusAnaliser.
    Returns dict with keys: score, details
    """
    try:
        analyzer = f3.GitHubStatsAnalyzerAllTime(username, token)
        agg = analyzer.get_aggregate_stats()
    except Exception:
        agg = {'Linhas_trocas': 0, 'Total_commits': 0, 'Streak_contribuicoes_consecutivas': 0}

    lines = agg.get('Linhas_trocas', 0)
    commits = agg.get('Total_commits', 0)
    streak = agg.get('Streak_contribuicoes_consecutivas', 0)

    # Normalize
    lines_norm = _safe_log_norm(lines, scale=5000.0)   # 5000 lines -> near 1.0 (heuristic)
    commits_norm = _safe_log_norm(commits, scale=200.0)
    streak_norm = _linear_cap(streak, cap=52)  # streak across weeks, cap at 52

    subs = {
        'lines_activity': lines_norm,
        'commits_activity': commits_norm,
        'week_streak': streak_norm
    }

    if lines == 0 and commits == 0 and streak == 0:
        return {'score': 0.0, 'details': subs}

    avg01 = statistics.mean(list(subs.values()))
    score = _to_signed01(avg01)

    return {'score': score, 'details': subs}


# Orchestration

def load_weights_from_config(cfg: cfgparser.ConfigParser = config) -> Dict[str, float]:
    """Reads weights from config.ini [scoring] section if present, otherwise returns defaults."""
    if 'scoring' not in cfg:
        return DEFAULT_WEIGHTS.copy()

    sec = cfg['scoring']
    weights = {}
    for k in DEFAULT_WEIGHTS.keys():
        try:
            weights[k] = float(sec.get(k, DEFAULT_WEIGHTS[k]))
        except Exception:
            weights[k] = DEFAULT_WEIGHTS[k]
    return weights


def score_user(username: Optional[str] = None, token: Optional[str] = None,
               weights: Optional[Dict[str, float]] = None, repo_limit: Optional[int] = None,
               num_events_sentiment: int = 10) -> Dict:
    """Main entry point.

    Returns a summary dict:
      {
        'areas': {
            'OSS': {'score': ..., 'details': {...}},
            'Status': {...},
            'Sentiment': {...}
         },
         'weights': {...},
         'final_score': float in [-1,1]
      }

    If username/token not provided, the values in config.ini will be used (if present).
    """
    username = username or DEFAULT_USERNAME
    token = token or DEFAULT_TOKEN

    if not username:
        raise ValueError('username is required either as argument or in config.ini')

    # Load weights
    cfg_weights = load_weights_from_config()
    if weights:
        # Merge provided weights overriding config
        cfg_weights.update(weights)
    # Normalize weights to sum 1.0 to make them intuitive
    total_w = sum(abs(v) for v in cfg_weights.values())
    if total_w == 0:
        # fallback to equal
        normalized = {k: 1.0 / len(cfg_weights) for k in cfg_weights}
    else:
        normalized = {k: float(v) / total_w for k, v in cfg_weights.items()}

    # Gather repo list for sentiment (optional limit)
    try:
        analyzer = f3.GitHubStatsAnalyzerAllTime(username, token)
        repo_full_list = [r['full_name'] for r in analyzer.repos]
        if repo_limit:
            repo_full_list = repo_full_list[:repo_limit]
    except Exception:
        repo_full_list = None

    # Compute area scores
    oss_res = compute_oss_score(username, token)
    status_res = compute_status_score(username, token)
    sentiment_res = compute_sentiment_score(username, token, repo_full_names=repo_full_list,
                                            num_events=num_events_sentiment)

    areas = {
        'OSS': oss_res,
        'Status': status_res,
        'Sentiment': sentiment_res
    }

    # Weighted mean (weights normalized earlier)
    area_scores = {k: areas[k]['score'] for k in ['OSS', 'Status', 'Sentiment']}
    final = 0.0
    for k, w in normalized.items():
        final += area_scores.get(k, 0.0) * w

    return {
        'areas': areas,
        'weights': normalized,
        'final_score': float(max(-1.0, min(1.0, final)))
    }


def save_score_to_json(result: Dict, username: str,filename: str = "score_result.json") -> str:
    """
    Save the score_user result dict as a JSON file in a 'json' folder
    located in the main project directory (same level as ScoringSys.py).
    
    Args:
        result (Dict): The result dict returned by score_user().
        filename (str): Name of the JSON file (default "score_result.json").
    
    Returns:
        str: Full path to the saved JSON file.
    """
    # Diretório principal (onde está o ScoringSys.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Criar a pasta 'json' se não existir
    json_dir = os.path.join(base_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

    # Se não passar filename, usa padrão com o username
    if not filename:
        filename = f"{username}_score.json"

    # Caminho completo do arquivo
    file_path = os.path.join(json_dir, filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return file_path
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar JSON: {e}")

# If run as script, print an example summary
if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--username', '-u', help='GitHub username (overrides config)')
    p.add_argument('--token', '-t', help='GitHub token (overrides config)')
    p.add_argument(
    '--repo-limit',
    type=int,
    default=200,
    help="Maximum number of repositories to analyze (default: %(default)s)"
)
    args = p.parse_args()

    try:
        summary = score_user(username=args.username, token=args.token, repo_limit=args.repo_limit)
        import json
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        saved_path = save_score_to_json(summary, username=args.username or DEFAULT_USERNAME)
        print(f"Resultado salvo em: {saved_path}")
    except Exception as e:
        print('Erro ao calcular pontuação:', e)



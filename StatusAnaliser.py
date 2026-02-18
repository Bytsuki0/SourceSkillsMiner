import requests
import json
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Optional, Tuple

_BYTES_PER_LINE = 40
_KB = 1024


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------

def _fetch_contributor_stats(url: str, headers: dict, max_retries: int = 10) -> Optional[list]:
    """
    Fetches /stats/contributors with exponential backoff on 202 responses.
    Returns the parsed list on success, None on failure.
    """
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers)
        if resp.status_code == 202:
            time.sleep(2 ** min(attempt, 4))   # 1, 2, 4, 8, 16 s max
            continue
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError):
            return None
        if not isinstance(data, list) or not data:
            return None
        return data
    return None


# ---------------------------------------------------------------------------
# AdvancedContributorAnalyzer
# ---------------------------------------------------------------------------

class AdvancedContributorAnalyzer:
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.repos = self._get_user_repositories()

    def _get_user_repositories(self) -> List[dict]:
        repos = []
        page = 1
        while True:
            url = f"{self.api_url}/users/{self.username}/repos?per_page=100&page={page}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            if not isinstance(data, list) or not data:
                break
            repos.extend(data)
            page += 1
        return repos

    def _get_months_for_repo(self, repo_full_name: str) -> Set[str]:
        """Fetch months with commits for a single repo."""
        url = f"{self.api_url}/repos/{repo_full_name}/stats/contributors"
        data = _fetch_contributor_stats(url, self.headers)
        if not data:
            return set()
        user_stats = next(
            (c for c in data
             if c.get('author', {}).get('login', '').lower() == self.username.lower()),
            None
        )
        if not user_stats:
            return set()
        months = set()
        for week_info in user_stats.get('weeks', []):
            if week_info.get('c', 0) > 0:
                date_obj = datetime.fromtimestamp(week_info['w'])
                months.add(date_obj.strftime('%Y-%m'))
        return months

    def _get_months_with_contributions(self) -> Set[str]:
        """Fetch months with commits across all repos in parallel."""
        months_with_commits: Set[str] = set()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._get_months_for_repo, repo['full_name']): repo
                for repo in self.repos
            }
            for future in as_completed(futures):
                try:
                    months_with_commits.update(future.result())
                except Exception:
                    pass
        return months_with_commits

    def _compute_monthly_streak(self, months: Set[str]) -> Tuple[int, int]:
        if not months:
            return 0, 0
        month_list = sorted(months)
        max_streak = current_streak = 1
        last_month = None
        for month_str in month_list:
            if last_month is None:
                current_streak = 1
            else:
                y1, m1 = map(int, last_month.split('-'))
                y2, m2 = map(int, month_str.split('-'))
                if (y2 == y1 and m2 == m1 + 1) or (y2 == y1 + 1 and m2 == 1 and m1 == 12):
                    current_streak += 1
                else:
                    current_streak = 1
            max_streak = max(max_streak, current_streak)
            last_month = month_str
        current_month = datetime.now().strftime('%Y-%m')
        includes_current = current_month in months or month_list[-1] == current_month
        current_active_streak = current_streak if includes_current else 0
        return max_streak, current_active_streak

    def check_monthly_contribution_streaks(self) -> Dict[str, any]:
        months = self._get_months_with_contributions()
        longest_streak, current_streak = self._compute_monthly_streak(months)
        return {
            'total_months_with_contributions': len(months),
            'longest_monthly_streak': longest_streak,
            'current_monthly_streak': current_streak,
            'has_12_month_streak': longest_streak >= 12,
            'has_6_month_streak': longest_streak >= 6,
            'months_list': sorted(list(months))
        }

    def _get_user_prs_for_repo(self, repo_full_name: str, since_date: datetime) -> List[dict]:
        prs = []
        page = 1
        while True:
            url = f"{self.api_url}/repos/{repo_full_name}/pulls"
            params = {'state': 'all', 'creator': self.username, 'per_page': 100, 'page': page}
            resp = requests.get(url, headers=self.headers, params=params)
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            if not isinstance(data, list) or not data:
                break
            for pr in data:
                created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                if created_at >= since_date:
                    prs.append(pr)
            page += 1
        return prs

    def _get_pr_comments(self, repo_full_name: str, pr_number: int) -> List[dict]:
        comments = []
        for endpoint in ('issues', 'pulls'):
            url = f"{self.api_url}/repos/{repo_full_name}/{endpoint}/{pr_number}/comments"
            page = 1
            while True:
                resp = requests.get(url, headers=self.headers, params={'per_page': 100, 'page': page})
                if resp.status_code != 200:
                    break
                try:
                    data = resp.json()
                except (ValueError, json.JSONDecodeError):
                    break
                if not isinstance(data, list) or not data:
                    break
                comments.extend(data)
                page += 1
        return comments

    def check_pr_discussion_involvement(self, repo_full_name: str, threshold: float = 0.7) -> Dict[str, any]:
        one_year_ago = datetime.now().replace(year=datetime.now().year - 1)
        prs = self._get_user_prs_for_repo(repo_full_name, one_year_ago)
        if not prs:
            return {
                'total_prs': 0, 'prs_with_discussion': 0,
                'discussion_rate': 0.0, 'meets_threshold': False, 'threshold': threshold
            }
        prs_with_user_comments = 0
        for pr in prs:
            comments = self._get_pr_comments(repo_full_name, pr['number'])
            if any(c.get('user', {}).get('login', '').lower() == self.username.lower() for c in comments):
                prs_with_user_comments += 1
            time.sleep(0.5)
        discussion_rate = prs_with_user_comments / len(prs)
        return {
            'repository': repo_full_name,
            'total_prs': len(prs),
            'prs_with_discussion': prs_with_user_comments,
            'discussion_rate': discussion_rate,
            'meets_threshold': discussion_rate >= threshold,
            'threshold': threshold
        }

    def get_follower_count(self) -> int:
        url = f"{self.api_url}/users/{self.username}"
        resp = requests.get(url, headers=self.headers)
        if resp.status_code != 200:
            return 0
        try:
            return resp.json().get('followers', 0)
        except (ValueError, json.JSONDecodeError):
            return 0

    def check_follower_percentile(self, sample_users: Optional[List[str]] = None) -> Dict[str, any]:
        user_followers = self.get_follower_count()
        if sample_users is None:
            estimated_percentile = min(99, max(50, user_followers * 2))
            return {
                'follower_count': user_followers,
                'estimated_percentile': estimated_percentile,
                'at_75th_percentile': estimated_percentile >= 75,
                'note': 'Estimation based on heuristic. Provide sample_users for accurate comparison.'
            }
        follower_counts = [user_followers]
        for user in sample_users:
            url = f"{self.api_url}/users/{user}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                try:
                    follower_counts.append(resp.json().get('followers', 0))
                except (ValueError, json.JSONDecodeError):
                    pass
            time.sleep(0.5)
        follower_counts.sort()
        percentile = (follower_counts.index(user_followers) / len(follower_counts)) * 100
        return {
            'follower_count': user_followers,
            'percentile': percentile,
            'at_75th_percentile': percentile >= 75,
            'sample_size': len(follower_counts)
        }

    def check_write_rights_non_owned_repos(self) -> Dict[str, any]:
        non_owned_with_write = []
        for repo in self.repos:
            owner = repo['owner']['login']
            if owner.lower() != self.username.lower():
                permissions = repo.get('permissions', {})
                if permissions.get('push', False) or permissions.get('admin', False):
                    non_owned_with_write.append({
                        'repository': repo['full_name'],
                        'owner': owner,
                        'permissions': permissions
                    })
        return {
            'has_write_to_non_owned': len(non_owned_with_write) > 0,
            'count': len(non_owned_with_write),
            'repositories': non_owned_with_write
        }

    def _get_all_contributors_for_repo(self, repo_full_name: str) -> List[Tuple[str, int]]:
        url = f"{self.api_url}/repos/{repo_full_name}/stats/contributors"
        data = _fetch_contributor_stats(url, self.headers)
        if not data:
            return []
        return [
            (c['author']['login'], sum(w.get('c', 0) for w in c.get('weeks', [])))
            for c in data if c.get('author')
        ]

    def _process_repo_percentile(self, repo: dict) -> Optional[dict]:
        """Compute commit percentile for a single repo. Returns None if user has no commits."""
        full_name = repo['full_name']
        contributors = self._get_all_contributors_for_repo(full_name)
        if not contributors:
            return None
        commit_counts = [commits for _, commits in contributors]
        user_commits = next(
            (commits for name, commits in contributors if name.lower() == self.username.lower()), 0
        )
        if user_commits == 0:
            return None
        commit_counts.sort()
        percentile = (commit_counts.index(user_commits) / len(commit_counts)) * 100
        return {
            'repository': full_name,
            'user_commits': user_commits,
            'total_contributors': len(contributors),
            'percentile': percentile,
            'at_50th_percentile': percentile >= 50
        }

    def check_commit_percentile_per_repo(self) -> Dict[str, any]:
        repo_percentiles = []
        repos_at_50th = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._process_repo_percentile, repo): repo for repo in self.repos}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        repo_percentiles.append(result)
                        if result['at_50th_percentile']:
                            repos_at_50th += 1
                except Exception:
                    pass
        return {
            'total_repos_analyzed': len(repo_percentiles),
            'repos_at_50th_percentile': repos_at_50th,
            'has_any_repo_at_50th': repos_at_50th > 0,
            'repositories': repo_percentiles
        }

    def analyze_all_criteria(self) -> Dict[str, any]:
        print("Analyzing monthly contribution streaks...")
        monthly_streaks = self.check_monthly_contribution_streaks()
        print("Checking write rights to non-owned repositories...")
        write_rights = self.check_write_rights_non_owned_repos()
        print("Analyzing commit percentiles across repositories...")
        commit_percentiles = self.check_commit_percentile_per_repo()
        print("Getting follower count...")
        follower_analysis = self.check_follower_percentile()
        return {
            'username': self.username,
            'analysis_date': datetime.now().isoformat(),
            'monthly_contribution_streaks': monthly_streaks,
            'write_rights_non_owned_repos': write_rights,
            'commit_percentile_analysis': commit_percentiles,
            'follower_analysis': follower_analysis,
            'summary': {
                'has_12_month_streak': monthly_streaks['has_12_month_streak'],
                'has_6_month_streak': monthly_streaks['has_6_month_streak'],
                'has_write_to_non_owned_repo': write_rights['has_write_to_non_owned'],
                'has_repo_at_50th_percentile_commits': commit_percentiles['has_any_repo_at_50th'],
                'at_75th_percentile_followers': follower_analysis['at_75th_percentile']
            }
        }

    def get_results_as_json(self) -> str:
        return json.dumps(self.analyze_all_criteria(), indent=2)


# ---------------------------------------------------------------------------
# GitHubStatsAnalyzerAllTime
# ---------------------------------------------------------------------------

class GitHubStatsAnalyzerAllTime:
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.repos = self._get_user_repositories()
        # Internal cache: full_name -> stats dict
        # Ensures analyze_all() and get_aggregate_stats() never re-fetch
        self._stats_cache: Dict[str, dict] = {}

    def _get_user_repositories(self) -> List[dict]:
        repos = []
        page = 1
        while True:
            url = f"{self.api_url}/users/{self.username}/repos?per_page=100&page={page}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            if not isinstance(data, list) or not data:
                break
            repos.extend(data)
            page += 1
        return repos

    @staticmethod
    def _estimate_lines_from_repo(repo: dict) -> int:
        """
        Fast approximation using the 'size' field (KB) already in repo metadata.
        No extra API call needed.  Formula: lines ≈ (size_kb * 1024) / bytes_per_line
        """
        return int(repo.get('size', 0) * _KB / _BYTES_PER_LINE)

    def _get_commit_stats(self, repo_full_name: str) -> dict:
        """
        Fetches contributor stats for one repo with exponential backoff on 202.
        Results are cached — repeated calls for the same repo are free.
        """
        if repo_full_name in self._stats_cache:
            return self._stats_cache[repo_full_name]

        empty = {'Contribuicoes_mensais': {}, 'Streak_contribuicoes_consecutivas': 0, 'Total_commits': 0}

        url = f"{self.api_url}/repos/{repo_full_name}/stats/contributors"
        data = _fetch_contributor_stats(url, self.headers)
        if not data:
            self._stats_cache[repo_full_name] = empty
            return empty

        user_stats = next(
            (c for c in data
             if c.get('author', {}).get('login', '').lower() == self.username.lower()),
            None
        )
        if not user_stats:
            self._stats_cache[repo_full_name] = empty
            return empty

        monthly_contrib: Dict[str, int] = defaultdict(int)
        weeks_with_commits: Set[str] = set()
        total_commits = 0

        for week_info in user_stats.get('weeks', []):
            commits = week_info.get('c', 0)
            if commits == 0:
                continue
            date_obj = datetime.fromtimestamp(week_info['w'])
            monthly_contrib[date_obj.strftime('%Y-%m')] += commits
            weeks_with_commits.add(date_obj.strftime('%Y-%U'))
            total_commits += commits

        result = {
            'Contribuicoes_mensais': dict(monthly_contrib),
            'Streak_contribuicoes_consecutivas': self._compute_weekly_streak(weeks_with_commits),
            'Total_commits': total_commits,
        }
        self._stats_cache[repo_full_name] = result
        return result

    def _compute_weekly_streak(self, weeks: set) -> int:
        week_list = sorted(weeks)
        max_streak = current_streak = 0
        last_week = None
        for week in week_list:
            if last_week is None:
                current_streak = 1
            else:
                y1, w1 = map(int, last_week.split('-'))
                y2, w2 = map(int, week.split('-'))
                if (y2 == y1 and w2 == w1 + 1) or (y2 == y1 + 1 and w2 == 0 and w1 == 52):
                    current_streak += 1
                else:
                    current_streak = 1
            max_streak = max(max_streak, current_streak)
            last_week = week
        return max_streak

    def _process_repo(self, repo: dict) -> Tuple[str, dict]:
        """Fetch and assemble stats for a single repo (runs inside thread pool)."""
        name = repo['name']
        full_name = repo['full_name']
        owner = repo['owner']['login']
        return name, {
            'Repositoria': name,
            'eh_dono': owner.lower() == self.username.lower(),
            'Permissao_escritura': not repo.get('fork', False),
            'Linhas_trocas': self._estimate_lines_from_repo(repo),
            'Linhas_trocas_metodo': 'estimativa_tamanho_repo',
            **self._get_commit_stats(full_name),
        }

    def analyze_all(self) -> dict:
        """
        Fetches stats for all repos in parallel (up to 5 threads).
        Results are cached internally so get_aggregate_stats() never re-fetches.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._process_repo, repo): repo for repo in self.repos}
            for future in as_completed(futures):
                try:
                    name, data = future.result()
                    results[name] = data
                except Exception as e:
                    repo = futures[future]
                    results[repo['name']] = {'error': str(e)}
        return results

    def get_results_as_json(self) -> str:
        return json.dumps(self.analyze_all(), indent=2)

    def get_aggregate_stats(self) -> dict:
        """
        Aggregates totals from analyze_all().
        Reuses internal cache — zero extra API calls.
        """
        all_stats = self.analyze_all()
        return {
            'Linhas_trocas': sum(v.get('Linhas_trocas', 0) for v in all_stats.values()),
            'Linhas_trocas_metodo': 'estimativa_tamanho_repo',
            'Total_commits': sum(v.get('Total_commits', 0) for v in all_stats.values()),
            'Streak_contribuicoes_consecutivas': max(
                (v.get('Streak_contribuicoes_consecutivas', 0) for v in all_stats.values()),
                default=0
            ),
        }


# ---------------------------------------------------------------------------
# GitHubLanguageCommitAnalyzer
# ---------------------------------------------------------------------------

class GitHubLanguageCommitAnalyzer:
    def __init__(self, username: str, token: str, prefetched_stats: Optional[Dict[str, dict]] = None):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self._stats_cache: Dict[str, Optional[tuple]] = {}
        if prefetched_stats:
            for repo_name, stats in prefetched_stats.items():
                self._stats_cache[repo_name] = (
                    stats.get('Linhas_trocas', 0),
                    stats.get('Total_commits', 0),
                )

    def _get_user_repositories(self) -> List[dict]:
        repos, page = [], 1
        while True:
            url = f"{self.api_url}/users/{self.username}/repos?per_page=100&page={page}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            if not isinstance(data, list) or not data:
                break
            repos.extend(data)
            page += 1
        return repos

    def _get_repo_languages(self, full_name: str) -> dict:
        url = f"{self.api_url}/repos/{full_name}/languages"
        resp = requests.get(url, headers=self.headers)
        return resp.json() if resp.status_code == 200 else {}

    def _get_commit_stats(self, full_name: str, repo: Optional[dict] = None) -> Optional[tuple]:
        """
        Returns (lines_changed, commits). Priority:
          1. In-memory cache (from prefetched_stats or prior calls)
          2. Single API call that extracts both lines and commits in one pass.
             If a repo dict is provided, lines are replaced with the faster
             size-based estimate — but commits always come from the API response,
             and the response is only fetched once regardless.
        """
        short_name = full_name.split('/')[-1]
        if short_name in self._stats_cache:
            return self._stats_cache[short_name]
        if full_name in self._stats_cache:
            return self._stats_cache[full_name]

        result = self._fetch_user_stats_from_api(full_name)

        if result is not None and repo is not None:
            # Swap in the faster size-based line estimate while keeping the
            # fetched commit count.  Both values came from the same API call.
            estimated_lines = GitHubStatsAnalyzerAllTime._estimate_lines_from_repo(repo)
            result = (estimated_lines, result[1])

        self._stats_cache[full_name] = result
        return result

    def _fetch_user_stats_from_api(self, full_name: str) -> Optional[tuple]:
        """
        Single API call that extracts (total_lines_changed, total_commits)
        for self.username in one pass over the weeks array.
        Replaces the former _fetch_commit_count_from_api and
        _fetch_commit_stats_from_api which hit the same endpoint separately.
        """
        url  = f"{self.api_url}/repos/{full_name}/stats/contributors"
        data = _fetch_contributor_stats(url, self.headers)
        if not data:
            return None
        for contributor in data:
            author = contributor.get('author')
            if author and author.get('login', '').lower() == self.username.lower():
                weeks        = contributor.get('weeks', [])
                total_lines  = sum(w.get('a', 0) + w.get('d', 0) for w in weeks)
                total_commits = sum(w.get('c', 0)                 for w in weeks)
                return total_lines, total_commits
        return None

    def _process_repo(self, repo: dict) -> Optional[Tuple[dict, tuple]]:
        """Fetch languages and commit stats for one repo in parallel."""
        full_name = repo['full_name']
        languages = self._get_repo_languages(full_name)
        stats = self._get_commit_stats(full_name, repo=repo)
        if not stats or not languages or sum(languages.values()) == 0:
            return None
        return languages, stats

    def analyze_language_usage(self) -> Dict[str, List[int]]:
        language_stats = defaultdict(lambda: [0, 0])
        repos = self._get_user_repositories()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._process_repo, repo): repo for repo in repos}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is None:
                        continue
                    languages, (lines_changed, commits) = result
                    total_bytes = sum(languages.values())
                    for lang, byte_count in languages.items():
                        proportion = byte_count / total_bytes
                        language_stats[lang][0] += int(lines_changed * proportion)
                        language_stats[lang][1] += int(commits * proportion)
                except Exception:
                    pass

        return dict(language_stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    username = config['github']['username']
    token = config['github']['token']

    analyzer = GitHubStatsAnalyzerAllTime(username, token)
    raw_stats = analyzer.analyze_all()
    print(json.dumps(raw_stats, indent=2))

    # Aggregate reuses the cache built by analyze_all — no extra API calls
    print(json.dumps(analyzer.get_aggregate_stats(), indent=2))

    # Language analyzer reuses the same stats — no re-fetching
    lang = GitHubLanguageCommitAnalyzer(username, token, prefetched_stats=raw_stats)
    print(lang.analyze_language_usage())
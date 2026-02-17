import requests
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

# Average bytes per line of source code — adjust if needed.
# Common estimates: Python ~40, C ~50, JS ~35. 40 is a reasonable default.
_BYTES_PER_LINE = 40
# GitHub's size field is in KB (1024 bytes).
_KB = 1024


class AdvancedContributorAnalyzer:
    """
    Analyzes advanced GitHub contributor metrics including:
    - Monthly contribution streaks (12+ months and 6+ months)
    - PR discussion involvement (70% threshold)
    - Follower percentile ranking (75th percentile)
    - Write rights to non-owned repositories
    - Commit percentile ranking (50th percentile)
    """

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
        """Fetch all repositories for the user."""
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

    def _get_months_with_contributions(self) -> Set[str]:
        """
        Get all months where the user made at least one contribution.
        Returns a set of month strings in 'YYYY-MM' format.
        """
        months_with_commits = set()

        for repo in self.repos:
            full_name = repo['full_name']
            url = f"{self.api_url}/repos/{full_name}/stats/contributors"
            max_retries = 100000

            for attempt in range(max_retries):
                resp = requests.get(url, headers=self.headers)
                if resp.status_code == 202:
                    time.sleep(2)
                    continue
                if resp.status_code != 200:
                    break
                try:
                    data = resp.json()
                except (ValueError, json.JSONDecodeError):
                    break
                if not isinstance(data, list) or not data:
                    break

                user_stats = None
                for contributor in data:
                    author = contributor.get('author')
                    if author and author.get('login', '').lower() == self.username.lower():
                        user_stats = contributor
                        break

                if not user_stats:
                    break

                for week_info in user_stats.get('weeks', []):
                    commits = week_info.get('c', 0)
                    if commits > 0:
                        week_timestamp = week_info.get('w')
                        date_obj = datetime.fromtimestamp(week_timestamp)
                        month_key = date_obj.strftime('%Y-%m')
                        months_with_commits.add(month_key)
                break

        return months_with_commits

    def _compute_monthly_streak(self, months: Set[str]) -> Tuple[int, int]:
        """
        Compute the longest streak of consecutive months with contributions.
        Returns: (longest_streak, current_streak_if_includes_current_month)
        """
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
        includes_current = current_month in months or (month_list and month_list[-1] == current_month)
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
                'total_prs': 0,
                'prs_with_discussion': 0,
                'discussion_rate': 0.0,
                'meets_threshold': False,
                'threshold': threshold
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
        for _ in range(10):
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 202:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                return []
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                return []
            if not isinstance(data, list) or not data:
                return []
            return [
                (c['author']['login'], sum(w.get('c', 0) for w in c.get('weeks', [])))
                for c in data
                if c.get('author')
            ]
        return []

    def check_commit_percentile_per_repo(self) -> Dict[str, any]:
        repo_percentiles = []
        repos_at_50th = 0
        for repo in self.repos:
            full_name = repo['full_name']
            contributors = self._get_all_contributors_for_repo(full_name)
            if not contributors:
                continue
            commit_counts = [commits for _, commits in contributors]
            user_commits = next(
                (commits for name, commits in contributors if name.lower() == self.username.lower()), 0
            )
            if user_commits == 0:
                continue
            commit_counts.sort()
            percentile = (commit_counts.index(user_commits) / len(commit_counts)) * 100
            is_at_50th = percentile >= 50
            if is_at_50th:
                repos_at_50th += 1
            repo_percentiles.append({
                'repository': full_name,
                'user_commits': user_commits,
                'total_contributors': len(contributors),
                'percentile': percentile,
                'at_50th_percentile': is_at_50th
            })
            time.sleep(1)
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

    def _get_user_repositories(self):
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
        Fast approximation of total lines in a repository using the 'size'
        field (KB) already present in the repo metadata — no extra API call.

        Formula:  lines ≈ (size_kb * 1024) / bytes_per_line
        At _BYTES_PER_LINE = 40 this gives ~25 lines per KB, which is a
        reasonable middle ground across languages.  The result is clearly an
        estimate; for exact numbers use _get_commit_stats() instead.
        """
        size_kb = repo.get('size', 0)
        return int(size_kb * _KB / _BYTES_PER_LINE)

    def _get_commit_stats(self, repo_full_name: str) -> dict:
        """
        Fetches contributor stats from GitHub and returns commit count,
        monthly breakdown, and weekly streak.

        Line counts are NO LONGER computed here — use _estimate_lines_from_repo()
        for a fast approximation, or sum additions/deletions from the weeks
        data below if exact per-user line counts are required.
        """
        url = f"{self.api_url}/repos/{repo_full_name}/stats/contributors"
        for _ in range(10):
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 202:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            if not isinstance(data, list) or not data:
                break

            user_stats = next(
                (c for c in data
                 if c.get('author', {}).get('login', '').lower() == self.username.lower()),
                None
            )
            if not user_stats:
                break

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

            return {
                'Contribuicoes_mensais': dict(monthly_contrib),
                'Streak_contribuicoes_consecutivas': self._compute_weekly_streak(weeks_with_commits),
                'Total_commits': total_commits,
            }

        return {
            'Contribuicoes_mensais': {},
            'Streak_contribuicoes_consecutivas': 0,
            'Total_commits': 0,
        }

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

    def analyze_all(self) -> dict:
        """
        Returns per-repo stats.  Linhas_trocas is now estimated from repo size
        (fast, zero extra API calls) instead of summed from weekly additions/
        deletions (slow, one contributor-stats call per repo).
        """
        results = {}
        for repo in self.repos:
            name       = repo['name']
            full_name  = repo['full_name']
            owner      = repo['owner']['login']
            eh_dono    = owner.lower() == self.username.lower()

            # Fast line estimate — no API call
            estimated_lines = self._estimate_lines_from_repo(repo)

            # Commit/streak data still needs the stats endpoint
            stats = self._get_commit_stats(full_name)

            results[name] = {
                'Repositoria': name,
                'eh_dono': eh_dono,
                'Permissao_escritura': not repo.get('fork', False),
                'Linhas_trocas': estimated_lines,          # approximation
                'Linhas_trocas_metodo': 'estimativa_tamanho_repo',
                **stats,
            }
        return results

    def get_results_as_json(self) -> str:
        return json.dumps(self.analyze_all(), indent=2)

    def get_aggregate_stats(self) -> dict:
        """
        Aggregate totals across all repos.
        Lines are estimated; commits and streak still use the stats endpoint.
        """
        aggregate_lines = sum(self._estimate_lines_from_repo(r) for r in self.repos)

        aggregate_commits = 0
        all_weeks: Set[str] = set()

        for repo in self.repos:
            stats = self._get_commit_stats(repo['full_name'])
            aggregate_commits += stats['Total_commits']
            # Rebuild the week set from monthly data isn't possible here,
            # but _get_commit_stats already builds it internally; to avoid
            # a second call we'd need caching — see StatusAnaliser_v2.py.

        return {
            'Linhas_trocas': aggregate_lines,
            'Linhas_trocas_metodo': 'estimativa_tamanho_repo',
            'Total_commits': aggregate_commits,
        }


class GitHubLanguageCommitAnalyzer:
    def __init__(self, username: str, token: str, prefetched_stats: Optional[Dict[str, dict]] = None):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        # Cache populated from GitHubStatsAnalyzerAllTime if provided
        self._stats_cache: Dict[str, Optional[tuple]] = {}
        if prefetched_stats:
            for repo_name, stats in prefetched_stats.items():
                self._stats_cache[repo_name] = (
                    stats.get('Linhas_trocas', 0),
                    stats.get('Total_commits', 0),
                )

    def _get_user_repositories(self):
        repos, page = [], 1
        while True:
            url = f"{self.api_url}/users/{self.username}/repos?per_page=100&page={page}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            repos.extend(data)
            page += 1
        return repos

    def _get_repo_languages(self, full_name):
        url = f"{self.api_url}/repos/{full_name}/languages"
        resp = requests.get(url, headers=self.headers)
        if resp.status_code != 200:
            return {}
        return resp.json()

    def _get_commit_stats(self, full_name: str, repo: Optional[dict] = None) -> Optional[tuple]:
        """
        Returns (lines_changed, commits).  Checks cache first.
        Falls back to repo-size estimate if a repo dict is provided,
        otherwise hits the GitHub API.
        """
        short_name = full_name.split('/')[-1]
        if short_name in self._stats_cache:
            return self._stats_cache[short_name]
        if full_name in self._stats_cache:
            return self._stats_cache[full_name]

        # Fast path: estimate lines from repo size if repo dict available
        if repo is not None:
            lines = GitHubStatsAnalyzerAllTime._estimate_lines_from_repo(repo)
            commits = self._fetch_commit_count_from_api(full_name)
            result = (lines, commits) if commits is not None else None
            self._stats_cache[full_name] = result
            return result

        # Slow path: full contributor stats API call
        result = self._fetch_commit_stats_from_api(full_name)
        self._stats_cache[full_name] = result
        return result

    def _fetch_commit_count_from_api(self, full_name: str) -> Optional[int]:
        """Fetches only the total commit count for the user in a repo."""
        url = f"{self.api_url}/repos/{full_name}/stats/contributors"
        for _ in range(10):
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 202:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data:
                return None
            for contributor in data:
                author = contributor.get('author')
                if author and author.get('login', '').lower() == self.username.lower():
                    return sum(w.get('c', 0) for w in contributor.get('weeks', []))
        return None

    def _fetch_commit_stats_from_api(self, full_name: str) -> Optional[tuple]:
        """Full contributor stats fetch (lines + commits). Used only on cache miss without repo dict."""
        url = f"{self.api_url}/repos/{full_name}/stats/contributors"
        for _ in range(10):
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 202:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data:
                return None
            for contributor in data:
                author = contributor.get('author')
                if author and author.get('login', '').lower() == self.username.lower():
                    total_lines = sum(w.get('a', 0) + w.get('d', 0) for w in contributor.get('weeks', []))
                    total_commits = sum(w.get('c', 0) for w in contributor.get('weeks', []))
                    return total_lines, total_commits
        return None

    def analyze_language_usage(self):
        language_stats = defaultdict(lambda: [0, 0])
        repos = self._get_user_repositories()
        for repo in repos:
            full_name = repo['full_name']
            languages = self._get_repo_languages(full_name)
            # Pass repo dict so line count can be estimated without extra API call
            commit_stats = self._get_commit_stats(full_name, repo=repo)
            if not commit_stats:
                continue
            lines_changed, commits = commit_stats
            total_bytes = sum(languages.values())
            if total_bytes == 0:
                continue
            for lang, bytes_count in languages.items():
                proportion = bytes_count / total_bytes
                language_stats[lang][0] += int(lines_changed * proportion)
                language_stats[lang][1] += int(commits * proportion)
        return dict(language_stats)


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    username = config['github']['username']
    token = config['github']['token']

    # Run all-time stats first (lines are now estimated, much faster)
    analyzer = GitHubStatsAnalyzerAllTime(username, token)
    raw_stats = analyzer.analyze_all()
    print(json.dumps(raw_stats, indent=2))

    # Pass results as cache into language analyzer — no re-fetching of lines or commits
    lang = GitHubLanguageCommitAnalyzer(username, token, prefetched_stats=raw_stats)
    print(lang.analyze_language_usage())

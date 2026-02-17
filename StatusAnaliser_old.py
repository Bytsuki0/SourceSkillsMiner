import requests
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

##d:\Code\GitBlame\StatusAnaliser.py:74: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
##  date_obj = datetime.utcfromtimestamp(week_timestamp)


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
                # Parse YYYY-MM format
                y1, m1 = map(int, last_month.split('-'))
                y2, m2 = map(int, month_str.split('-'))
                
                # Check if consecutive months
                if (y2 == y1 and m2 == m1 + 1) or (y2 == y1 + 1 and m2 == 1 and m1 == 12):
                    current_streak += 1
                else:
                    current_streak = 1
            
            max_streak = max(max_streak, current_streak)
            last_month = month_str
        
        # Check if streak includes current month
        current_month = datetime.now().strftime('%Y-%m')
        includes_current = current_month in months or (month_list and month_list[-1] == current_month)
        current_active_streak = current_streak if includes_current else 0
        
        return max_streak, current_active_streak
    
    def check_monthly_contribution_streaks(self) -> Dict[str, any]:
        """
        Check if user has:
        - At least 12 months with contributions
        - At least 6 months with contributions
        """
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
        """Get all PRs created by the user in a repository since a given date."""
        prs = []
        page = 1
        
        while True:
            url = f"{self.api_url}/repos/{repo_full_name}/pulls"
            params = {
                'state': 'all',
                'creator': self.username,
                'per_page': 100,
                'page': page
            }
            
            resp = requests.get(url, headers=self.headers, params=params)
            if resp.status_code != 200:
                break
            
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            
            if not isinstance(data, list) or not data:
                break
            
            # Filter by date
            for pr in data:
                created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                if created_at >= since_date:
                    prs.append(pr)
            
            page += 1
        
        return prs
    
    def _get_pr_comments(self, repo_full_name: str, pr_number: int) -> List[dict]:
        """Get all comments on a PR (issue comments and review comments)."""
        comments = []
        
        # Get issue comments
        url = f"{self.api_url}/repos/{repo_full_name}/issues/{pr_number}/comments"
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
        
        # Get review comments
        url = f"{self.api_url}/repos/{repo_full_name}/pulls/{pr_number}/comments"
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
        """
        Check if user is involved in discussion of their own PRs at least threshold% of the time
        in the past year for a specific repository.
        """
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
            pr_number = pr['number']
            comments = self._get_pr_comments(repo_full_name, pr_number)
            
            # Check if user commented on their own PR
            user_commented = any(
                comment.get('user', {}).get('login', '').lower() == self.username.lower()
                for comment in comments
            )
            
            if user_commented:
                prs_with_user_comments += 1
            
            # Rate limiting protection
            time.sleep(0.5)
        
        discussion_rate = prs_with_user_comments / len(prs) if prs else 0.0
        
        return {
            'repository': repo_full_name,
            'total_prs': len(prs),
            'prs_with_discussion': prs_with_user_comments,
            'discussion_rate': discussion_rate,
            'meets_threshold': discussion_rate >= threshold,
            'threshold': threshold
        }
    
    def get_follower_count(self) -> int:
        """Get the number of followers for the user."""
        url = f"{self.api_url}/users/{self.username}"
        resp = requests.get(url, headers=self.headers)
        
        if resp.status_code != 200:
            return 0
        
        try:
            data = resp.json()
            return data.get('followers', 0)
        except (ValueError, json.JSONDecodeError):
            return 0
    
    def check_follower_percentile(self, sample_users: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Check if user is at 75th percentile by follower count.
        If sample_users is not provided, uses a representative sample approach.
        
        Note: This is an approximation since we can't query all GitHub users.
        For accurate results, provide a list of relevant users to compare against.
        """
        user_followers = self.get_follower_count()
        
        if sample_users is None:
            # Use heuristic: if user has 50+ followers, they're likely in top 25%
            # This is based on GitHub statistics showing most users have <10 followers
            estimated_percentile = min(99, max(50, user_followers * 2))
            
            return {
                'follower_count': user_followers,
                'estimated_percentile': estimated_percentile,
                'at_75th_percentile': estimated_percentile >= 75,
                'note': 'Estimation based on heuristic. Provide sample_users for accurate comparison.'
            }
        
        # Compare against provided sample
        follower_counts = [user_followers]
        for user in sample_users:
            url = f"{self.api_url}/users/{user}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    follower_counts.append(data.get('followers', 0))
                except (ValueError, json.JSONDecodeError):
                    pass
            time.sleep(0.5)  # Rate limiting
        
        follower_counts.sort()
        user_rank = follower_counts.index(user_followers)
        percentile = (user_rank / len(follower_counts)) * 100
        
        return {
            'follower_count': user_followers,
            'percentile': percentile,
            'at_75th_percentile': percentile >= 75,
            'sample_size': len(follower_counts)
        }
    
    def check_write_rights_non_owned_repos(self) -> Dict[str, any]:
        """
        Check if user has write rights to repositories they don't own.
        """
        non_owned_with_write = []
        
        for repo in self.repos:
            owner = repo['owner']['login']
            is_owner = owner.lower() == self.username.lower()
            
            if not is_owner:
                permissions = repo.get('permissions', {})
                has_write = permissions.get('push', False) or permissions.get('admin', False)
                
                if has_write:
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
        """Get all contributors and their commit counts for a repository."""
        url = f"{self.api_url}/repos/{repo_full_name}/stats/contributors"
        max_retries = 10
        
        for attempt in range(max_retries):
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
            
            contributors = []
            for contributor in data:
                author = contributor.get('author')
                if author:
                    username = author.get('login', '')
                    total_commits = sum(week.get('c', 0) for week in contributor.get('weeks', []))
                    contributors.append((username, total_commits))
            
            return contributors
        
        return []
    
    def check_commit_percentile_per_repo(self) -> Dict[str, any]:
        """
        Check if user is at 50th percentile by number of commits for each repository.
        """
        repo_percentiles = []
        repos_at_50th = 0
        
        for repo in self.repos:
            full_name = repo['full_name']
            contributors = self._get_all_contributors_for_repo(full_name)
            
            if not contributors:
                continue
            
            # Find user's commits
            user_commits = 0
            commit_counts = []
            
            for username, commits in contributors:
                commit_counts.append(commits)
                if username.lower() == self.username.lower():
                    user_commits = commits
            
            if user_commits == 0:
                continue
            
            commit_counts.sort()
            user_rank = commit_counts.index(user_commits)
            percentile = (user_rank / len(commit_counts)) * 100
            
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
            
            time.sleep(1)  # Rate limiting
        
        return {
            'total_repos_analyzed': len(repo_percentiles),
            'repos_at_50th_percentile': repos_at_50th   ,
            'has_any_repo_at_50th': repos_at_50th > 0,
            'repositories': repo_percentiles
        }
    
    def analyze_all_criteria(self) -> Dict[str, any]:
        """
        Analyze all advanced contributor criteria and return comprehensive results.
        """
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
        """Return analysis results as formatted JSON string."""
        data = self.analyze_all_criteria()
        return json.dumps(data, indent=2)


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

    def _get_commit_stats(self, repo_full_name):
        url = f"{self.api_url}/repos/{repo_full_name}/stats/contributors"
        max_retries = 10
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

            total_lines = 0
            monthly_contrib = defaultdict(int)
            weeks_with_commits = set()
            total_commits = 0

            for week_info in user_stats.get('weeks', []):
                week_timestamp = week_info.get('w')
                additions = week_info.get('a', 0)
                deletions = week_info.get('d', 0)
                commits = week_info.get('c', 0)

                date_obj = datetime.utcfromtimestamp(week_timestamp)

                total_lines += additions + deletions
                month_key = date_obj.strftime('%Y-%m')
                monthly_contrib[month_key] += commits

                if commits > 0:
                    iso_week = date_obj.strftime('%Y-%U')
                    weeks_with_commits.add(iso_week)
                    total_commits += commits

            streak = self._compute_weekly_streak(weeks_with_commits)

            return {
                'Linhas_trocas': total_lines,
                'Contribuicoes_mensais': dict(monthly_contrib),
                'Streak_contribuicoes_consecutivas': streak,
                'Total_commits': total_commits
            }

        return {
            'Linhas_trocas': 0,
            'Contribuicoes_mensais': {},
            'Streak_contribuicoes_consecutivas': 0,
            'Total_commits': 0
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
        results = {}
        for repo in self.repos:
            name = repo['name']
            full_name = repo['full_name']
            owner = repo['owner']['login']
            eh_dono = owner.lower() == self.username.lower()
            stats = self._get_commit_stats(full_name)
            results[name] = {
                'Repositoria': name,
                'eh_dono': eh_dono,
                'Permissao_escritura': not repo.get('fork', False),
                **stats
            }
        return results

    def get_results_as_json(self) -> str:
        data = self.analyze_all()
        return json.dumps(data, indent=2)

    def get_aggregate_stats(self) -> dict:
        aggregate_lines = 0
        aggregate_commits = 0
        all_weeks_with_commits = set()

        for repo in self.repos:
            full_name = repo['full_name']
            url = f"{self.api_url}/repos/{full_name}/stats/contributors"
            max_retries = 10
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
                    week_timestamp = week_info.get('w')
                    additions = week_info.get('a', 0)
                    deletions = week_info.get('d', 0)
                    commits = week_info.get('c', 0)

                    date_obj = datetime.utcfromtimestamp(week_timestamp)
                    aggregate_lines += additions + deletions
                    aggregate_commits += commits
                    if commits > 0:
                        iso_week = date_obj.strftime('%Y-%U')
                        all_weeks_with_commits.add(iso_week)
                break

        aggregate_streak = self._compute_weekly_streak(all_weeks_with_commits)

        return {
            'Linhas_trocas': aggregate_lines,
            'Total_commits': aggregate_commits,
            'Streak_contribuicoes_consecutivas': aggregate_streak
        }

class GitHubLanguageCommitAnalyzer:
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def _get_user_repositories(self):
        repos = []
        page = 1
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

    def _get_commit_stats(self, full_name):
        url = f"{self.api_url}/repos/{full_name}/stats/contributors"
        max_retries = 10
        for attempt in range(max_retries):
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
                    total_lines = sum(week.get('a', 0) + week.get('d', 0) for week in contributor.get('weeks', []))
                    total_commits = sum(week.get('c', 0) for week in contributor.get('weeks', []))
                    return total_lines, total_commits
        return None

    def analyze_language_usage(self):
        language_stats = defaultdict(lambda: [0, 0])  # language: [lines_changed, commits]
        repos = self._get_user_repositories()
        for repo in repos:
            full_name = repo['full_name']
            languages = self._get_repo_languages(full_name)
            commit_stats = self._get_commit_stats(full_name)
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

    #analyzer_pro = AdvancedContributorAnalyzer(username, token)
    #print(analyzer_pro.get_results_as_json())

    lang = GitHubLanguageCommitAnalyzer(username, token)
    print(lang.analyze_language_usage())

    #analyzer = GitHubStatsAnalyzerAllTime(username, token)
    #print(analyzer.get_results_as_json())
    


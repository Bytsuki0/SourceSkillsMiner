"""
The single source of GitHub commit-contribution data.

Previously *three* classes (GitHubStatsAnalyzerAllTime, WorkTypeAnalyzer's
analyzers, AdaptabilityAnalyzer) each carried their own copy of the year-by-year
``contributionsCollection`` fetcher **and their own module-level cache**, so the
same data was pulled from GitHub up to three times per run.

This service fetches it **once** (one query per year of account history,
enriched to carry everything the callers need) and caches it on the instance.
The facade builds one ``ContributionsService`` and hands it to every analyzer.

Exposed views, each matching the aggregation the old code used:
  * ``all_user_commits()``        -> {repo_full_name: [{occurredAt, commitCount}]}
  * ``repo_stats(full_name)``     -> {monthly_contrib, weeks_with_commits, total_commits}
  * ``commit_depth_per_language()`` -> {language: total_commits}
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

from ssm.core.client import GitHubClient

__all__ = ["ContributionsService"]

# One enriched query feeds all three views:
#   nodes(occurredAt, commitCount) -> streaks / monthly / per-node totals
#   contributions.totalCount       -> per-repo commit depth
#   repository.primaryLanguage     -> grouping commit depth by language
_CONTRIBUTIONS_QUERY = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      commitContributionsByRepository(maxRepositories: 100) {
        repository {
          nameWithOwner
          primaryLanguage { name }
        }
        contributions(first: 100) {
          totalCount
          nodes { occurredAt commitCount }
        }
      }
    }
  }
}
"""

_USER_CREATED_AT_QUERY = """
query($login: String!) { user(login: $login) { createdAt } }
"""

_FALLBACK_JOIN_YEAR = 2015


class ContributionsService:
    """Fetches and caches a user's commit contributions across all repos."""

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        *,
        year_sleep: float = 0.3,
    ) -> None:
        self._client = client
        self._username = username
        self._year_sleep = year_sleep

        # Populated lazily by _ensure_fetched().
        self._commits_by_repo: Optional[Dict[str, List[dict]]] = None
        self._total_by_repo: Dict[str, int] = {}
        self._primary_language_by_repo: Dict[str, Optional[str]] = {}
        self._repo_stats_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------ public API

    def all_user_commits(self) -> Dict[str, List[dict]]:
        """Return {repo_full_name: [contribution nodes]} (cached)."""
        self._ensure_fetched()
        return self._commits_by_repo  # type: ignore[return-value]

    def repo_stats(self, repo_full_name: str) -> dict:
        """
        Return parsed stats for a single repo:
            {monthly_contrib: {YYYY-MM: n}, weeks_with_commits: {YYYY-WW}, total_commits: int}
        """
        if repo_full_name not in self._repo_stats_cache:
            nodes = self.all_user_commits().get(repo_full_name, [])
            self._repo_stats_cache[repo_full_name] = self._parse_nodes(nodes)
        return self._repo_stats_cache[repo_full_name]

    def commit_depth_per_language(self) -> Dict[str, int]:
        """Return {language: total_commits} grouped by each repo's primary language."""
        self._ensure_fetched()
        lang_commits: Dict[str, int] = defaultdict(int)
        for full_name, total in self._total_by_repo.items():
            lang = self._primary_language_by_repo.get(full_name)
            if lang and total:
                lang_commits[lang] += total
        return dict(lang_commits)

    def total_commits(self) -> int:
        """Total commit contributions across every repository (cached)."""
        self._ensure_fetched()
        return sum(self._total_by_repo.values())

    # ------------------------------------------------------------- internals

    def _ensure_fetched(self) -> None:
        if self._commits_by_repo is not None:
            return

        commits_by_repo: Dict[str, List[dict]] = defaultdict(list)
        total_by_repo: Dict[str, int] = defaultdict(int)

        join_year = self._join_year()
        current_year = datetime.now().year

        for year in range(join_year, current_year + 1):
            data = self._client.graphql(
                _CONTRIBUTIONS_QUERY,
                {
                    "login": self._username,
                    "from": f"{year}-01-01T00:00:00Z",
                    "to": f"{year}-12-31T23:59:59Z",
                },
            )
            if not data or not data.get("user"):
                continue

            collection = data["user"].get("contributionsCollection", {})
            for entry in collection.get("commitContributionsByRepository", []):
                repo = entry.get("repository") or {}
                full_name = repo.get("nameWithOwner", "")
                if not full_name:
                    continue

                contributions = entry.get("contributions") or {}
                commits_by_repo[full_name].extend(contributions.get("nodes", []))
                total_by_repo[full_name] += contributions.get("totalCount", 0)

                primary = (repo.get("primaryLanguage") or {}).get("name")
                if primary:
                    self._primary_language_by_repo[full_name] = primary

            time.sleep(self._year_sleep)

        self._commits_by_repo = dict(commits_by_repo)
        self._total_by_repo = dict(total_by_repo)

    def _join_year(self) -> int:
        data = self._client.graphql(
            _USER_CREATED_AT_QUERY, {"login": self._username}
        )
        created = (data or {}).get("user", {}).get("createdAt") if data else None
        if created:
            try:
                return datetime.fromisoformat(created.replace("Z", "+00:00")).year
            except ValueError:
                pass
        return _FALLBACK_JOIN_YEAR

    @staticmethod
    def _parse_nodes(nodes: List[dict]) -> dict:
        """
        Convert contribution nodes into monthly counts, week keys and a total.
        (Identical to the old ``_contributions_to_user_stats``.)
        """
        monthly_contrib: Dict[str, int] = defaultdict(int)
        weeks_with_commits: Set[str] = set()
        total_commits = 0

        for node in nodes:
            count = node.get("commitCount", 0)
            if count == 0:
                continue
            occurred = node.get("occurredAt", "")
            if not occurred:
                continue
            try:
                dt = datetime.fromisoformat(occurred.replace("Z", "+00:00"))
            except ValueError:
                continue
            monthly_contrib[dt.strftime("%Y-%m")] += count
            weeks_with_commits.add(dt.strftime("%Y-%U"))
            total_commits += count

        return {
            "monthly_contrib": dict(monthly_contrib),
            "weeks_with_commits": weeks_with_commits,
            "total_commits": total_commits,
        }

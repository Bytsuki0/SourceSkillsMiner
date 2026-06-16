"""
Commitment analyzer.

Ports ``WorkTypeAnalyzer.GitHubCommitmentAnalyzer``. Evaluates four
open-source commitment criteria entirely from data already cached by the shared
:class:`ContributionsService` plus a single follower-count query. The
``has_write_to_non_owned_repo`` key is kept (always False) for schema
compatibility with existing consumers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from ssm.analyzers.base import Analyzer
from ssm.core.client import GitHubClient
from ssm.services.contributions import ContributionsService

__all__ = ["CommitmentAnalyzer"]

_GQL_USER_PROFILE = """
query($login: String!) {
  user(login: $login) {
    createdAt
    followers { totalCount }
  }
}
"""


class CommitmentAnalyzer(Analyzer):
    """Scores long-term commitment from contribution streaks and reach."""

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        contributions: ContributionsService,
    ) -> None:
        super().__init__(client, username)
        self._contributions = contributions

    # ----------------------------------------------------------- criteria

    @staticmethod
    def _months_from_contributions(all_commits: Dict[str, list]) -> List[Tuple[int, int]]:
        months: Set[Tuple[int, int]] = set()
        for nodes in all_commits.values():
            for node in nodes:
                if not node.get("commitCount", 0):
                    continue
                occurred = node.get("occurredAt", "")
                if not occurred:
                    continue
                try:
                    dt = datetime.fromisoformat(occurred.replace("Z", "+00:00"))
                    months.add((dt.year, dt.month))
                except ValueError:
                    pass
        return sorted(months)

    @staticmethod
    def _max_consecutive_months(sorted_months: List[Tuple[int, int]]) -> int:
        if not sorted_months:
            return 0
        max_streak = current = 1
        for i in range(1, len(sorted_months)):
            y1, m1 = sorted_months[i - 1]
            y2, m2 = sorted_months[i]
            if (y2 * 12 + m2) - (y1 * 12 + m1) == 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1
        return max_streak

    @staticmethod
    def _check_50th_percentile_commits(all_commits: Dict[str, list]) -> bool:
        counts = [
            sum(n.get("commitCount", 0) for n in nodes)
            for nodes in all_commits.values()
            if any(n.get("commitCount", 0) for n in nodes)
        ]
        if not counts:
            return False
        median = sorted(counts)[len(counts) // 2]
        return median >= 50

    @staticmethod
    def _check_75th_percentile_followers(profile: dict) -> bool:
        return (profile.get("followers") or 0) >= 100

    def _fetch_user_profile(self) -> dict:
        data = self.client.graphql(_GQL_USER_PROFILE, {"login": self.username})
        if not data or not data.get("user"):
            return {}
        return {"followers": data["user"].get("followers", {}).get("totalCount", 0)}

    # ------------------------------------------------------------ analyze

    def analyze(self) -> dict:
        self.emit("Computing commitment score...", stage="commitment")
        all_commits = self._contributions.all_user_commits()

        sorted_months = self._months_from_contributions(all_commits)
        max_streak = self._max_consecutive_months(sorted_months)
        profile = self._fetch_user_profile()

        return {
            "summary": {
                "has_12_month_streak": max_streak >= 12,
                "has_6_month_streak": max_streak >= 6,
                "has_write_to_non_owned_repo": False,  # criterion removed; kept for schema compat
                "has_repo_at_50th_percentile_commits": self._check_50th_percentile_commits(all_commits),
                "at_75th_percentile_followers": self._check_75th_percentile_followers(profile),
            }
        }

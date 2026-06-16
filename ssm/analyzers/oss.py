"""
Open-source contribution analyzer (GraphQL).

Ports the parts of the old ``OSSanaliser`` module that the scorer consumes:
issue/PR resolution counts and a total commit count. The previous REST
``/search/issues`` queries and per-repo ``/contributors`` calls are replaced by
a single GraphQL counts query plus the shared :class:`ContributionsService`
(so the commit total reuses data already fetched for the other areas — no extra
round trips).
"""

from __future__ import annotations

from typing import Dict

from ssm.analyzers.base import Analyzer
from ssm.core.client import GitHubClient
from ssm.services.contributions import ContributionsService

__all__ = ["OSSAnalyzer"]

# A single query yields every count the OSS score needs. ``closedPRs`` includes
# MERGED so it matches the old REST ``state:closed`` semantics (a merged PR is
# also closed); ``mergedPRs`` is tracked separately for the merge rate.
_OSS_COUNTS_QUERY = """
query($login: String!) {
  user(login: $login) {
    openIssues:   issues(states: OPEN)            { totalCount }
    closedIssues: issues(states: CLOSED)          { totalCount }
    openPRs:      pullRequests(states: OPEN)       { totalCount }
    closedPRs:    pullRequests(states: [CLOSED, MERGED]) { totalCount }
    mergedPRs:    pullRequests(states: MERGED)     { totalCount }
  }
}
"""


class OSSAnalyzer(Analyzer):
    """Collects issue/PR resolution activity and commit volume."""

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        contributions: ContributionsService,
    ) -> None:
        super().__init__(client, username)
        self._contributions = contributions

    def issue_pr_counts(self) -> Dict[str, int]:
        data = self.client.graphql(_OSS_COUNTS_QUERY, {"login": self.username}) or {}
        user = data.get("user") or {}

        def count(alias: str) -> int:
            return (user.get(alias) or {}).get("totalCount", 0)

        return {
            "open_issues": count("openIssues"),
            "closed_issues": count("closedIssues"),
            "open_prs": count("openPRs"),
            "closed_prs": count("closedPRs"),
            "merged_prs": count("mergedPRs"),
        }

    def analyze(self) -> Dict:
        self.emit("Computing OSS score...", stage="oss")
        counts = self.issue_pr_counts()
        counts["total_commits"] = self._contributions.total_commits()
        return counts

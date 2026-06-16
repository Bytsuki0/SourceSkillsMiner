"""
Status analyzer (general activity).

Ports ``StatusAnaliser.GitHubStatsAnalyzerAllTime`` onto the shared services.
The dead ``AdvancedContributorAnalyzer`` (used only by that module's own
``__main__`` and superseded by :class:`~ssm.analyzers.commitment.CommitmentAnalyzer`)
is not carried over. The per-repo result keys are kept verbatim because the
language analyzer and the status scorer read them by name.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ssm.analyzers.base import Analyzer
from ssm.core.client import GitHubClient
from ssm.services.contributions import ContributionsService
from ssm.services.repositories import RepositoryService, estimate_lines

__all__ = ["StatusAnalyzer", "MAX_REPOS"]

# The original analyzer processed at most this many repositories.
MAX_REPOS = 15


class StatusAnalyzer(Analyzer):
    """Per-repo and aggregate commit/line statistics for a user."""

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        repos: RepositoryService,
        contributions: ContributionsService,
    ) -> None:
        super().__init__(client, username)
        self._repos = repos
        self._contributions = contributions

    @staticmethod
    def _weekly_streak(weeks: set) -> int:
        week_list = sorted(weeks)
        max_streak = current_streak = 0
        last_week = None
        for week in week_list:
            if last_week is None:
                current_streak = 1
            else:
                y1, w1 = map(int, last_week.split("-"))
                y2, w2 = map(int, week.split("-"))
                if (y2 == y1 and w2 == w1 + 1) or (y2 == y1 + 1 and w2 == 0 and w1 == 52):
                    current_streak += 1
                else:
                    current_streak = 1
            max_streak = max(max_streak, current_streak)
            last_week = week
        return max_streak

    def _commit_stats(self, repo_full_name: str) -> dict:
        stats = self._contributions.repo_stats(repo_full_name)
        return {
            "Contribuicoes_mensais": stats["monthly_contrib"],
            "Streak_contribuicoes_consecutivas": self._weekly_streak(
                stats["weeks_with_commits"]
            ),
            "Total_commits": stats["total_commits"],
        }

    def _process_repo(self, repo: dict) -> Tuple[str, dict]:
        name = repo["name"]
        owner = repo["owner"]["login"]
        return name, {
            "Repositoria": name,
            "eh_dono": owner.lower() == self.username.lower(),
            "Permissao_escritura": not repo.get("fork", False),
            "Linhas_trocas": estimate_lines(repo),
            "Linhas_trocas_metodo": "estimativa_tamanho_repo",
            **self._commit_stats(repo["full_name"]),
        }

    def analyze(self) -> Dict[str, dict]:
        """Return {repo_name: stats} for up to ``MAX_REPOS`` repositories."""
        self.emit("Computing status score...", stage="status")
        repos = self._repos.list_repos(limit=MAX_REPOS)
        results: Dict[str, dict] = {}
        for i, repo in enumerate(repos, 1):
            self.emit(
                f"[{i}/{len(repos)}] Processing {repo['full_name']}...",
                stage="status", fraction=i / len(repos) if repos else None,
            )
            try:
                name, data = self._process_repo(repo)
                results[name] = data
            except Exception as exc:
                results[repo["name"]] = {"error": str(exc)}
        return results

    def aggregate_stats(self, all_stats: Dict[str, dict]) -> dict:
        """Collapse per-repo stats into the totals the status scorer needs."""
        return {
            "Linhas_trocas": sum(v.get("Linhas_trocas", 0) for v in all_stats.values()),
            "Total_commits": sum(v.get("Total_commits", 0) for v in all_stats.values()),
            "Streak_contribuicoes_consecutivas": max(
                (v.get("Streak_contribuicoes_consecutivas", 0) for v in all_stats.values()),
                default=0,
            ),
        }

"""
Language usage analyzer.

Merges the two near-identical ``GitHubLanguageCommitAnalyzer`` classes (one in
StatusAnaliser, one in WorkTypeAnalyzer) into a single implementation. Commit
counts come from prefetched status data when available, otherwise from the
shared :class:`ContributionsService`; line counts are distributed across a
repo's languages proportionally to their byte share.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ssm.analyzers.base import Analyzer
from ssm.core.client import GitHubClient
from ssm.services.contributions import ContributionsService
from ssm.services.repositories import RepositoryService

__all__ = ["LanguageUsageAnalyzer"]


class LanguageUsageAnalyzer(Analyzer):
    """Distributes commit/line statistics across programming languages."""

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        repos: RepositoryService,
        contributions: ContributionsService,
        prefetched_stats: Optional[Dict[str, dict]] = None,
    ) -> None:
        super().__init__(client, username)
        self._repos = repos
        self._contributions = contributions

        # Cache keyed by repo name -> (lines, commits), seeded from status data.
        self._stats_cache: Dict[str, Optional[tuple]] = {}
        for repo_name, stats in (prefetched_stats or {}).items():
            self._stats_cache[repo_name] = (
                stats.get("Linhas_trocas", 0),
                stats.get("Total_commits", 0),
            )

    def _commit_stats(self, full_name: str) -> Optional[Tuple[int, int]]:
        short_name = full_name.split("/")[-1]
        if short_name in self._stats_cache:
            return self._stats_cache[short_name]
        if full_name in self._stats_cache:
            return self._stats_cache[full_name]

        commits = self._contributions.repo_stats(full_name)["total_commits"]
        result: Optional[Tuple[int, int]] = (0, commits) if commits else None
        self._stats_cache[full_name] = result
        return result

    def analyze_language_usage(self) -> Dict[str, List[int]]:
        language_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        repos = self._repos.list_repos()
        for i, repo in enumerate(repos, 1):
            full_name = repo["full_name"]
            self.emit(
                f"[{i}/{len(repos)}] Language analysis: {full_name}",
                stage="language", fraction=i / len(repos) if repos else None,
            )
            try:
                # Language byte-breakdown comes inline from RepositoryService (GraphQL).
                languages = repo.get("languages", {})
                stats = self._commit_stats(full_name)
                total_bytes = sum(languages.values())
                if not stats or not languages or total_bytes == 0:
                    continue
                lines_changed, commits = stats
                for lang, byte_count in languages.items():
                    proportion = byte_count / total_bytes
                    language_stats[lang][0] += int(lines_changed * proportion)
                    language_stats[lang][1] += int(commits * proportion)
            except Exception:
                pass
        return dict(language_stats)

    def analyze(self) -> Dict:
        """Return the language-usage block of the final report."""
        self.emit("Analyzing language usage...", stage="language")
        try:
            language_stats = self.analyze_language_usage()
            print(f"Found {len(language_stats)} languages used across repositories.")
            if not language_stats:
                return {
                    "error": None, "languages": {}, "language_count": 0,
                    "total_commits": 0, "total_lines": 0,
                }
            total_commits = sum(stats[1] for stats in language_stats.values())
            total_lines = sum(stats[0] for stats in language_stats.values())
            sorted_languages = sorted(
                language_stats.items(), key=lambda x: x[1][1], reverse=True
            )
            return {
                "error": None,
                "languages": language_stats,
                "language_count": len(language_stats),
                "total_commits": total_commits,
                "total_lines": total_lines,
                "top_5_languages": [
                    {"language": lang, "lines": stats[0], "commits": stats[1]}
                    for lang, stats in sorted_languages[:5]
                ],
            }
        except Exception as exc:
            return {
                "error": str(exc), "languages": {}, "language_count": 0,
                "total_commits": 0, "total_lines": 0,
            }

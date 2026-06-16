"""
ScoringFacade — the single entry point to the whole subsystem (Facade pattern).

Callers (the CLI, or any embedder) talk only to this class. It wires together
the configuration, the HTTP client, the shared services and every analyzer +
scorer, runs them in order, and assembles the final report. This is the OO
replacement for the old free-function ``ScoringSys.score_user``; the returned
dict is byte-for-byte compatible with what the classifier and frontend expect.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ssm.analyzers.adaptability import AdaptabilityAnalyzer
from ssm.analyzers.commitment import CommitmentAnalyzer
from ssm.analyzers.imports import ImportScanner
from ssm.analyzers.language import LanguageUsageAnalyzer
from ssm.analyzers.oss import OSSAnalyzer
from ssm.analyzers.sentiment import SentimentAnalyzer
from ssm.analyzers.status import MAX_REPOS, StatusAnalyzer
from ssm.core.client import GitHubClient
from ssm.core.config import Config
from ssm.core.events import Observer, Subject
from ssm.core.normalizers import to_01
from ssm.scoring.scorers import (
    AdaptabilityScorer,
    CommitmentScorer,
    OSSScorer,
    SentimentScorer,
    StatusScorer,
)
from ssm.services.contributions import ContributionsService
from ssm.services.repositories import RepositoryService

__all__ = ["ScoringFacade"]

_AVATAR_QUERY = "query($login: String!) { user(login: $login) { avatarUrl } }"


class ScoringFacade(Subject):
    """Orchestrates analyzers + scorers behind one ``analyze()`` call."""

    def __init__(
        self,
        config: Optional[Config] = None,
        observers: Optional[List[Observer]] = None,
    ) -> None:
        Subject.__init__(self)
        self._config = config or Config()
        self.attach_all(observers)

    def _wire(self, subject: Subject) -> Subject:
        """Propagate the facade's observers onto a built service/analyzer."""
        subject.attach_all(self.observers)
        return subject

    def analyze(
        self,
        username: Optional[str] = None,
        token: Optional[str] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
        repo_limit: Optional[int] = None,
        num_events_sentiment: int = 20000,
        include_import_scan: bool = True,
        import_scan_max_repos: int = 20,
        import_scan_max_files: int = 30,
    ) -> Dict:
        username = username or self._config.username
        token = token or self._config.token
        if not username:
            raise ValueError("username is required either as argument or in config.ini")

        normalized = self._config.normalized_weights(weights)

        # --- Build the subsystem (shared client + services) -----------------
        # Services are plain collaborators; only the observable analyzers (and
        # the facade itself) participate in the Observer chain.
        client = GitHubClient(token)
        repos = RepositoryService(client, username)
        contributions = ContributionsService(client, username)

        oss_analyzer = self._wire(OSSAnalyzer(client, username, contributions))
        status_analyzer = self._wire(StatusAnalyzer(client, username, repos, contributions))
        adaptability_analyzer = self._wire(
            AdaptabilityAnalyzer(client, username, contributions)
        )
        sentiment_analyzer = self._wire(
            SentimentAnalyzer(client, username, num_events=num_events_sentiment)
        )
        commitment_analyzer = self._wire(
            CommitmentAnalyzer(client, username, contributions)
        )

        # Repos used for sentiment (historically capped at MAX_REPOS, then by repo_limit)
        try:
            repo_full_list = repos.full_names(limit=MAX_REPOS)
            if repo_limit:
                repo_full_list = repo_full_list[:repo_limit]
        except Exception:
            repo_full_list = []

        # --- OSS ------------------------------------------------------------
        try:
            oss_res = OSSScorer().score(oss_analyzer.analyze())
        except Exception as exc:
            print(f"Error computing OSS score: {exc}")
            oss_res = {"score": 0.0, "details": {"error": str(exc)}}

        # --- Status (also produces raw stats reused by language analyzer) ---
        try:
            raw_stats: Dict = status_analyzer.analyze()
        except Exception as exc:
            print(f"Error computing status stats: {exc}")
            raw_stats = {}
        try:
            agg = status_analyzer.aggregate_stats(raw_stats)
        except Exception:
            agg = {"Linhas_trocas": 1, "Total_commits": 0, "Streak_contribuicoes_consecutivas": 0}
        status_res = StatusScorer().score(agg)

        # --- Adaptability ---------------------------------------------------
        try:
            adaptability_res = AdaptabilityScorer().score(adaptability_analyzer.analyze())
        except Exception as exc:
            print(f"Error computing adaptability score: {exc}")
            adaptability_res = {"score": 0.0, "details": {"error": str(exc), "metrics": {}}}

        # --- Sentiment ------------------------------------------------------
        try:
            sentiments = sentiment_analyzer.analyze(repo_full_list, num_events_sentiment)
        except Exception as exc:
            print(f"Error computing sentiment score: {exc}")
            sentiments = {}
        sentiment_res = SentimentScorer().score(sentiments)

        # --- Commitment -----------------------------------------------------
        try:
            commitment_res = CommitmentScorer().score(commitment_analyzer.analyze())
        except Exception as exc:
            commitment_res = {"score": 0.0, "details": {"error": str(exc), "criteria_met": {}}}

        areas = {
            "OSS": oss_res,
            "Status": status_res,
            "Adaptability": adaptability_res,
            "Sentiment": {**sentiment_res, "score": to_01(sentiment_res["score"])},
            "Commitment": commitment_res,
        }

        raw_final = sum(areas[k]["score"] * normalized[k] for k in areas)
        final = max(0.0, min(1.0, raw_final))

        # --- Supplementary data --------------------------------------------
        self.emit("\nCollecting supplementary data...", stage="supplementary")
        language_analyzer = self._wire(
            LanguageUsageAnalyzer(
                client, username, repos, contributions, prefetched_stats=raw_stats
            )
        )
        language_data = language_analyzer.analyze()

        if include_import_scan:
            import_scanner = self._wire(ImportScanner(client, username, repos))
            import_scan_data = import_scanner.analyze(
                max_repos=import_scan_max_repos,
                max_files_per_repo=import_scan_max_files,
            )
        else:
            self.emit("Skipping import scan.", stage="imports")
            import_scan_data = {"skipped": True, "reason": "Import scan skipped by user request."}

        self.emit("\nCollecting GitHub profile picture...", stage="avatar")
        avatar_url = self._fetch_avatar(client, username)

        self.emit("\nAnalysis complete!", stage="done")
        return {
            "username": username,
            "avatar_url": avatar_url,
            "areas": areas,
            "weights": normalized,
            "final_score": float(max(0.0, min(1.0, final))),
            "language_usage": language_data,
            "import_scan": import_scan_data,
        }

    @staticmethod
    def _fetch_avatar(client: GitHubClient, username: str) -> Optional[str]:
        data = client.graphql(_AVATAR_QUERY, {"login": username})
        user = (data or {}).get("user") or {}
        if user:
            return user.get("avatarUrl")
        print(f"Warning: Could not fetch avatar for {username}")
        return None

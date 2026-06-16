"""
Sentiment analyzer (GraphQL).

Ports ``SentimentalAnaliser``: collects discussion comments for a repo,
translates non-English text, and scores it with NLTK's VADER analyzer. The
translator and analyzer are built once per instance.

The old implementation read GitHub's repo-wide REST comment lists
(``/issues/comments``, ``/pulls/comments``, ``/comments``). GraphQL has no
equivalent flat repo-wide comment connection, so comments are now gathered by
traversing the repo's most recently updated issues and pull requests and
reading their comment threads. The downstream score only uses the *sign* of
each repo's average sentiment, which this sampling preserves.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import nltk
from langdetect import LangDetectException, detect
from nltk.sentiment import SentimentIntensityAnalyzer
from translate import Translator

from ssm.analyzers.base import Analyzer
from ssm.core.client import GitHubClient

__all__ = ["SentimentAnalyzer"]

# Per repo: pull comment bodies from the most recently updated issues and PRs.
_COMMENTS_QUERY = """
query($owner: String!, $name: String!, $threads: Int!, $perThread: Int!) {
  repository(owner: $owner, name: $name) {
    issues(first: $threads, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { comments(first: $perThread) { nodes { body } } }
    }
    pullRequests(first: $threads, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { comments(first: $perThread) { nodes { body } } }
    }
  }
}
"""

# GraphQL connection page size is capped at 100; keep sampling modest.
_MAX_THREADS = 50
_COMMENTS_PER_THREAD = 20


def _ensure_vader() -> None:
    """Download VADER's lexicon if it is not already available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download("vader_lexicon")


class SentimentAnalyzer(Analyzer):
    """Scores the tone of a user's discussion activity per repository."""

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        *,
        num_events: int = 20000,
    ) -> None:
        super().__init__(client, username)
        self.num_events = num_events
        _ensure_vader()
        self._translator = Translator(from_lang="pt", to_lang="en")
        self._sia = SentimentIntensityAnalyzer()

    @staticmethod
    def _is_english(text: str) -> bool:
        try:
            if len(text.strip()) < 5:
                return False
            return detect(text) == "en"
        except LangDetectException:
            return False

    def _to_english(self, text: str) -> str:
        return text if self._is_english(text) else self._translator.translate(text)

    def _fetch_comment_bodies(self, repo_full_name: str, threads: int) -> List[str]:
        owner, name = repo_full_name.split("/")
        data = self.client.graphql(
            _COMMENTS_QUERY,
            {
                "owner": owner,
                "name": name,
                "threads": threads,
                "perThread": _COMMENTS_PER_THREAD,
            },
        )
        repo = (data or {}).get("repository") or {}
        if not repo:
            print(f"Warning: could not fetch comments for {repo_full_name}")
            return []

        bodies: List[str] = []
        for collection in ("issues", "pullRequests"):
            for node in (repo.get(collection) or {}).get("nodes", []):
                for comment in (node.get("comments") or {}).get("nodes", []):
                    bodies.append(comment.get("body", "") or "")
        return bodies

    def repo_sentiment(self, repo_full_name: str, num_events: int) -> float:
        """Return the mean compound sentiment across a repo's comments."""
        threads = max(1, min(_MAX_THREADS, num_events))
        bodies = self._fetch_comment_bodies(repo_full_name, threads)

        scores = [
            self._sia.polarity_scores(self._to_english(body))["compound"]
            for body in bodies
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def analyze(
        self,
        repo_full_names: Optional[List[str]] = None,
        num_events: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return {repo_full_name: overall_sentiment} for repos with signal."""
        self.emit("Computing sentiment score...", stage="sentiment")
        events = self.num_events if num_events is None else num_events
        sentiments: Dict[str, float] = {}
        for full_name in repo_full_names or []:
            try:
                score = self.repo_sentiment(full_name, events)
                if score != 0.0:
                    sentiments[full_name] = score
            except Exception:
                pass
        return sentiments

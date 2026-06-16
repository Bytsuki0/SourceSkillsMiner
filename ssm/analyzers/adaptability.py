"""
Adaptability analyzer.

Ports ``Adaptabilityanalyzer.AdaptabilityAnalyzer``. Measures adaptability from
four GraphQL signals: polyglot breadth (depth-weighted), technology adoption
over time, domain (topic) entropy, and PR bounce-back resilience. Commit depth
per language now comes from the shared :class:`ContributionsService` instead of
a fourth private copy of the contributions fetcher.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

from ssm.analyzers.base import Analyzer
from ssm.core.client import GitHubClient
from ssm.core.normalizers import clamp01
from ssm.services.contributions import ContributionsService

__all__ = ["AdaptabilityAnalyzer"]

# Normalisation caps (tune to match your population)
_LANG_CAP = 8           # distinct primary languages -> score = 1
_TRANSITION_CAP = 5     # language-era transitions   -> score = 1
_ENTROPY_CAP = 3.0      # bits of topic entropy      -> score = 1
_RESILIENCE_CAP = 0.5   # fraction of bounced-back PRs -> score = 1

# Re-submission window (days) to count a closed PR as "recovered".
_BOUNCE_BACK_DAYS = 60

_GQL_REPOS_LANGUAGES = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    repositories(first: 100, after: $cursor,
                 ownerAffiliations: [OWNER],
                 orderBy: {field: CREATED_AT, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        createdAt
        primaryLanguage { name }
        languages(first: 20) { edges { size node { name } } }
      }
    }
  }
}
"""

_GQL_CONTRIBUTED_TOPICS = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    repositoriesContributedTo(
      first: 50, after: $cursor,
      includeUserRepositories: true,
      contributionTypes: [COMMIT, PULL_REQUEST]
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        repositoryTopics(first: 20) { nodes { topic { name } } }
      }
    }
  }
}
"""

_GQL_ALL_PRS = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    pullRequests(
      first: 100, after: $cursor,
      states: [OPEN, CLOSED, MERGED],
      orderBy: {field: CREATED_AT, direction: ASC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        createdAt
        closedAt
        mergedAt
        state
        repository { nameWithOwner }
      }
    }
  }
}
"""


def _shannon_entropy(counts: List[int]) -> float:
    """Shannon entropy in bits for a frequency distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


class AdaptabilityAnalyzer(Analyzer):
    """Measures adaptability from four GraphQL-sourced behavioural signals."""

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        contributions: ContributionsService,
    ) -> None:
        super().__init__(client, username)
        self._contributions = contributions

    # ---------------------------------------------------- sub-score 1: langs

    def _compute_language_diversity(
        self, repo_nodes: List[dict], lang_commits: Dict[str, int]
    ) -> Tuple[float, int]:
        primary_langs: Set[str] = set()
        for repo in repo_nodes:
            pl = (repo.get("primaryLanguage") or {}).get("name")
            if pl and lang_commits.get(pl, 0) > 0:
                primary_langs.add(pl)
        count = len(primary_langs)
        return clamp01(count / _LANG_CAP), count

    # ------------------------------------------------ sub-score 2: adoption

    def _compute_technology_adoption(self, repo_nodes: List[dict]) -> Tuple[float, int]:
        year_langs: Dict[int, Set[str]] = defaultdict(set)
        for repo in repo_nodes:
            pl = (repo.get("primaryLanguage") or {}).get("name")
            if not pl:
                continue
            created = repo.get("createdAt", "")
            try:
                year = datetime.fromisoformat(created.replace("Z", "+00:00")).year
            except ValueError:
                continue
            year_langs[year].add(pl)

        if not year_langs:
            return 0.0, 0

        transitions = 0
        seen: Set[str] = set()
        for year in sorted(year_langs):
            new_this_year = year_langs[year] - seen
            if seen and new_this_year:
                transitions += len(new_this_year)
            seen |= year_langs[year]

        return clamp01(transitions / _TRANSITION_CAP), transitions

    # ------------------------------------------------- sub-score 3: domains

    def _compute_domain_flexibility(self, contrib_nodes: List[dict]) -> Tuple[float, float]:
        topic_counts: Dict[str, int] = defaultdict(int)
        for repo in contrib_nodes:
            for t_node in repo.get("repositoryTopics", {}).get("nodes", []):
                topic = (t_node.get("topic") or {}).get("name")
                if topic:
                    topic_counts[topic] += 1
        if not topic_counts:
            return 0.0, 0.0
        entropy = _shannon_entropy(list(topic_counts.values()))
        return clamp01(entropy / _ENTROPY_CAP), round(entropy, 4)

    # ---------------------------------------------- sub-score 4: resilience

    def _compute_resilience(self, all_pr_nodes: List[dict]) -> Tuple[float, int, int]:
        repo_prs: Dict[str, List[dict]] = defaultdict(list)
        for pr in all_pr_nodes:
            repo = (pr.get("repository") or {}).get("nameWithOwner", "")
            if repo:
                repo_prs[repo].append(pr)

        closed_total = 0
        bounced_back = 0
        window = timedelta(days=_BOUNCE_BACK_DAYS)

        for prs in repo_prs.values():
            rejected = []
            for pr in prs:
                if pr.get("mergedAt"):
                    continue
                if pr.get("state") == "CLOSED" or (pr.get("closedAt") and not pr.get("mergedAt")):
                    closed_at_str = pr.get("closedAt")
                    if not closed_at_str:
                        continue
                    try:
                        rejected.append(
                            datetime.fromisoformat(closed_at_str.replace("Z", "+00:00"))
                        )
                        closed_total += 1
                    except ValueError:
                        continue

            open_dates = []
            for pr in prs:
                created_str = pr.get("createdAt")
                if not created_str:
                    continue
                try:
                    open_dates.append(
                        datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    )
                except ValueError:
                    continue
            open_dates_sorted = sorted(open_dates)

            for rejected_at in rejected:
                for open_dt in open_dates_sorted:
                    if open_dt > rejected_at and (open_dt - rejected_at) <= window:
                        bounced_back += 1
                        break

        if closed_total == 0:
            return 0.0, 0, 0
        return clamp01((bounced_back / closed_total) / _RESILIENCE_CAP), closed_total, bounced_back

    # ------------------------------------------------------------ analyze

    def analyze(self) -> dict:
        self.emit("Computing Adaptability score...", stage="adaptability")

        self.emit(f"Fetching repo language data for {self.username}...", stage="adaptability")
        repo_nodes = self.client.graphql_paginate(
            _GQL_REPOS_LANGUAGES, ["repositories"], self.username
        )

        self.emit("Fetching contributed-to topics...", stage="adaptability")
        contrib_nodes = self.client.graphql_paginate(
            _GQL_CONTRIBUTED_TOPICS, ["repositoriesContributedTo"], self.username
        )

        self.emit("Fetching PR history...", stage="adaptability")
        all_pr_nodes = self.client.graphql_paginate(
            _GQL_ALL_PRS, ["pullRequests"], self.username
        )

        self.emit("Fetching commit depth per language...", stage="adaptability")
        lang_commits = self._contributions.commit_depth_per_language()

        div_score, n_langs = self._compute_language_diversity(repo_nodes, lang_commits)
        adopt_score, n_transitions = self._compute_technology_adoption(repo_nodes)
        domain_score, entropy_bits = self._compute_domain_flexibility(contrib_nodes)
        resil_score, closed_total, bounced = self._compute_resilience(all_pr_nodes)

        sub_scores = [div_score, adopt_score, domain_score, resil_score]
        final = sum(sub_scores) / len(sub_scores)

        return {
            "score": round(final, 6),
            "details": {
                "language_diversity_score": round(div_score, 4),
                "technology_adoption_score": round(adopt_score, 4),
                "domain_flexibility_score": round(domain_score, 4),
                "resilience_score": round(resil_score, 4),
                "distinct_primary_languages": n_langs,
                "language_transitions": n_transitions,
                "topic_entropy_bits": entropy_bits,
                "closed_prs_total": closed_total,
                "bounced_back_prs": bounced,
            },
        }

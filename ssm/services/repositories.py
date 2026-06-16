"""
Repository listing service (GraphQL).

Replaces the ~6 near-identical ``_get_user_repositories`` REST loops that were
spread across the analyzer classes. One paginated GraphQL query fetches the
user's public owned repositories *together with* their language byte-breakdown,
so the language analyzer no longer needs a per-repo REST ``/languages`` call
either. Results are cached per instance and sliced by whichever caller needs a
cap.
"""

from __future__ import annotations

from typing import List, Optional

from ssm.core.client import GitHubClient

__all__ = ["RepositoryService", "estimate_lines"]

# Heuristic used to estimate touched lines from repo size (bytes -> lines).
_BYTES_PER_LINE = 40
_KB = 1024

# NAME/ASC mirrors the REST "list user repos" default sort (full_name asc), so
# the capped subsets (Status, import scan) stay close to the previous behaviour.
_REPOS_QUERY = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    repositories(first: 100, after: $cursor,
                 ownerAffiliations: [OWNER],
                 privacy: PUBLIC,
                 orderBy: {field: NAME, direction: ASC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        name
        nameWithOwner
        isFork
        diskUsage
        owner { login }
        primaryLanguage { name }
        languages(first: 20) { edges { size node { name } } }
      }
    }
  }
}
"""


def estimate_lines(repo: dict) -> int:
    """Estimate lines of code from the repo's reported size in KB."""
    return int(repo.get("size", 0) * _KB / _BYTES_PER_LINE)


class RepositoryService:
    """Fetches and caches a user's public repositories via GraphQL."""

    def __init__(self, client: GitHubClient, username: str) -> None:
        self._client = client
        self._username = username
        self._all_repos: Optional[List[dict]] = None

    def list_repos(self, limit: Optional[int] = None) -> List[dict]:
        """
        Return the user's repositories (cached).

        ``limit`` caps the number returned; callers that historically processed
        only the first N repos pass it, others omit it for the full list.
        """
        if self._all_repos is None:
            self._all_repos = self._fetch_all()
        if limit is not None:
            return self._all_repos[:limit]
        return self._all_repos

    def full_names(self, limit: Optional[int] = None) -> List[str]:
        """Convenience accessor for ``owner/repo`` strings."""
        return [r["full_name"] for r in self.list_repos(limit)]

    def _fetch_all(self) -> List[dict]:
        nodes = self._client.graphql_paginate(
            _REPOS_QUERY, ["repositories"], self._username
        )
        return [self._map_node(node) for node in nodes]

    @staticmethod
    def _map_node(node: dict) -> dict:
        """Project a GraphQL repository node onto the dict shape callers expect."""
        languages = {
            edge["node"]["name"]: edge.get("size", 0)
            for edge in (node.get("languages") or {}).get("edges", [])
            if edge.get("node")
        }
        return {
            "name": node.get("name"),
            "full_name": node.get("nameWithOwner"),
            "owner": {"login": (node.get("owner") or {}).get("login")},
            "fork": node.get("isFork", False),
            "size": node.get("diskUsage") or 0,
            "primary_language": (node.get("primaryLanguage") or {}).get("name"),
            "languages": languages,
        }

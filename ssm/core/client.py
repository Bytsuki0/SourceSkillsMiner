"""
A single GitHub GraphQL client.

Every old module built its own headers and ad-hoc ``requests`` calls; several
reimplemented GraphQL cursor pagination. This class is the one place that knows
how to talk to GitHub, and the whole pipeline now speaks GraphQL exclusively
(REST is no longer used anywhere in the package).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

__all__ = ["GitHubClient"]


class GitHubClient:
    """Thin, reusable wrapper around the GitHub GraphQL API."""

    GRAPHQL_URL = "https://api.github.com/graphql"

    def __init__(self, token: Optional[str], *, timeout: int = 30) -> None:
        self.token = token
        self.timeout = timeout

        self.headers: Dict[str, str] = {
            "User-Agent": "github-public-client",
            "Content-Type": "application/json",
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

        self._session = requests.Session()
        self._session.headers.update(self.headers)

    # --------------------------------------------------------------- GraphQL

    def graphql(self, query: str, variables: dict) -> Optional[dict]:
        """Execute a GraphQL query. Returns the ``data`` dict or None on error."""
        try:
            resp = self._session.post(
                self.GRAPHQL_URL,
                json={"query": query, "variables": variables},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            print(f"[GitHubClient/GraphQL] Network error: {exc}")
            return None

        if resp.status_code != 200:
            print(f"[GitHubClient/GraphQL] HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        body = resp.json()
        if "errors" in body:
            # Some GraphQL errors are non-fatal and still carry partial data.
            print(f"[GitHubClient/GraphQL] Query errors: {body['errors']}")
        return body.get("data")

    def graphql_paginate(
        self,
        query: str,
        path: List[str],
        login: str,
        *,
        max_pages: int = 30,
        sleep: float = 0.2,
    ) -> List[dict]:
        """
        Follow cursor pagination for a connection located at ``data.user.<path>``.

        ``path`` is the list of keys from ``data.user`` to the connection object,
        e.g. ``["repositories"]``. Returns the concatenated ``nodes``.
        """
        nodes: List[dict] = []
        cursor: Optional[str] = None

        for _ in range(max_pages):
            variables = {"login": login}
            if cursor:
                variables["cursor"] = cursor

            data = self.graphql(query, variables)
            if not data or not data.get("user"):
                break

            obj = data["user"]
            for key in path:
                obj = obj.get(key, {})

            nodes.extend(obj.get("nodes", []))

            page_info = obj.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            time.sleep(sleep)

        return nodes

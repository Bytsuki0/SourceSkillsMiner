"""
AdaptabilityAnalyzer.py

Measures a GitHub user's Adaptability dimension using four behavioural
proxies derived entirely from the GraphQL API:

  1. language_diversity_score
       Distinct primary languages across all owned repos, weighted by
       commit depth in each language (polyglot breadth with depth guard).
       Normalised to [0,1] against a cap of 8 languages.

  2. technology_adoption_score
       Year-over-year language shifts in repo creation order.
       Number of distinct "era transitions" (new primary language appearing
       in a later year than any previously seen) normalised to [0,1].

  3. domain_flexibility_score
       Shannon entropy of topic tags across all contributed repos.
       High entropy across domains (web, ml, devops, cli …) signals
       willingness to work in ambiguous and diverse contexts.

  4. resilience_score
       Fraction of closed (non-merged) PRs that the user re-submitted
       within 60 days as a new PR in the same repo.  Behavioural signal
       of bouncing back and adjusting approach after rejection.

Final score in [-1, +1]:
    mean of the four sub-scores, each individually mapped from [0,1] to
    [-1,1] via  score = 2x - 1.

References
----------
  Calefato et al. (2019). "What makes a successful OSS project contributor?"
  MSSAT (PMC 2024). "Adaptive decision-making styles."
  Waterloo/Functionize (2022). "Developer soft-skill proxies."
"""

import math
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GQL_URL = "https://api.github.com/graphql"

# Normalisation caps (tune these to match your population)
_LANG_CAP           = 8     # distinct primary languages → score=1
_TRANSITION_CAP     = 5     # language-era transitions   → score=1
_ENTROPY_CAP        = 3.0   # bits of topic entropy       → score=1
_RESILIENCE_CAP     = 0.5   # fraction of bounced-back PRs → score=1

# Re-submission window in days to count a closed PR as "recovered"
_BOUNCE_BACK_DAYS   = 60

# Module-level cache shared across instances in the same process
_adap_cache: Dict[str, object] = {}

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

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
        languages(first: 20) {
          edges {
            size
            node { name }
          }
        }
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
        repositoryTopics(first: 20) {
          nodes { topic { name } }
        }
      }
    }
  }
}
"""

_GQL_CLOSED_PRS = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    pullRequests(
      first: 100, after: $cursor,
      states: [CLOSED],
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        createdAt
        closedAt
        mergedAt
        repository { nameWithOwner }
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

_GQL_COMMIT_COUNTS = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      commitContributionsByRepository(maxRepositories: 100) {
        repository { nameWithOwner primaryLanguage { name } }
        contributions { totalCount }
      }
    }
  }
}
"""

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _gql(query: str, variables: dict, headers: dict) -> Optional[dict]:
    """Execute one GraphQL query. Returns the ``data`` dict or None."""
    try:
        resp = requests.post(
            _GQL_URL,
            headers={**headers, "Content-Type": "application/json"},
            json={"query": query, "variables": variables},
            timeout=30,
        )
    except Exception as exc:
        print(f"[AdaptabilityAnalyzer] Network error: {exc}")
        return None
    if resp.status_code != 200:
        print(f"[AdaptabilityAnalyzer] HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    body = resp.json()
    if "errors" in body:
        print(f"[AdaptabilityAnalyzer] GQL errors: {body['errors']}")
    return body.get("data")


def _paginate(query: str, path: List[str], headers: dict,
              login: str) -> List[dict]:
    """
    Follow GraphQL cursor pagination for a query that returns a connection
    at ``path`` inside ``data.user``.

    ``path`` is the list of keys to traverse from ``data.user`` to the
    connection object, e.g. ``["repositories"]``.
    """
    nodes: List[dict] = []
    cursor: Optional[str] = None

    for _ in range(30):          # hard cap: 30 pages × 100 items = 3 000
        variables = {"login": login}
        if cursor:
            variables["cursor"] = cursor

        data = _gql(query, variables, headers)
        if not data or not data.get("user"):
            break

        obj = data["user"]
        for key in path:
            obj = obj.get(key, {})

        page_nodes = obj.get("nodes", [])
        nodes.extend(page_nodes)

        page_info = obj.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        time.sleep(0.2)

    return nodes


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _to_signed(x01: float) -> float:
    """Map [0,1] → [-1,1]."""
    return 2.0 * x01 - 1.0


def _shannon_entropy(counts: List[int]) -> float:
    """Shannon entropy in bits for a frequency distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts if c > 0
    )

# ---------------------------------------------------------------------------
# Commit-depth per language (for weighted polyglot score)
# ---------------------------------------------------------------------------

def _fetch_commit_depth_per_language(
    username: str,
    headers:  dict,
) -> Dict[str, int]:
    """
    Return {language_name: total_commits} across all user repos via GraphQL
    contributionsCollection (year-by-year, same approach as WorkTypeAnalyzer).
    Cached under ("adap_lang_commits", username).
    """
    cache_key = ("adap_lang_commits", username)
    if cache_key in _adap_cache:
        return _adap_cache[cache_key]     # type: ignore

    # Determine join year
    data = _gql(
        "query($login:String!){user(login:$login){createdAt}}",
        {"login": username},
        headers,
    )
    join_year = 2015
    if data and data.get("user") and data["user"].get("createdAt"):
        join_year = datetime.fromisoformat(
            data["user"]["createdAt"].replace("Z", "+00:00")
        ).year

    lang_commits: Dict[str, int] = defaultdict(int)
    current_year = datetime.now().year

    for year in range(join_year, current_year + 1):
        d = _gql(
            _GQL_COMMIT_COUNTS,
            {
                "login": username,
                "from":  f"{year}-01-01T00:00:00Z",
                "to":    f"{year}-12-31T23:59:59Z",
            },
            headers,
        )
        if not d or not d.get("user"):
            continue
        cc = d["user"].get("contributionsCollection", {})
        for entry in cc.get("commitContributionsByRepository", []):
            pl = (entry.get("repository") or {}).get("primaryLanguage") or {}
            lang = pl.get("name")
            count = (entry.get("contributions") or {}).get("totalCount", 0)
            if lang and count:
                lang_commits[lang] += count
        time.sleep(0.25)

    result = dict(lang_commits)
    _adap_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# AdaptabilityAnalyzer
# ---------------------------------------------------------------------------

class AdaptabilityAnalyzer:
    """
    Measures adaptability from four GraphQL-sourced signals.

    Returns
    -------
    {
      "score": float in [0, 1],
      "details": {
        "language_diversity_score":   float [0,1],
        "technology_adoption_score":  float [0,1],
        "domain_flexibility_score":   float [0,1],
        "resilience_score":           float [0,1],
        "distinct_primary_languages": int,
        "language_transitions":       int,
        "topic_entropy_bits":         float,
        "closed_prs_total":           int,
        "bounced_back_prs":           int,
      }
    }
    """

    def __init__(self, username: str, token: str):
        self.username = username
        self.token    = token
        self.headers  = {
            "Authorization": f"token {token}",
            "Accept":        "application/vnd.github.v3+json",
        }

    # ------------------------------------------------------------------
    # Sub-score 1: Language diversity (polyglot breadth × depth)
    # ------------------------------------------------------------------

    def _compute_language_diversity(
        self,
        repo_nodes: List[dict],
        lang_commits: Dict[str, int],
    ) -> Tuple[float, int]:
        """
        Count distinct primary languages that the user has actually committed
        to (commit depth > 0).  Pure presence (forked repos, 0 commits) is
        excluded to guard against shallow exploration.

        Returns (normalised_score [0,1], distinct_count).
        """
        primary_langs: Set[str] = set()
        for repo in repo_nodes:
            pl = (repo.get("primaryLanguage") or {}).get("name")
            if pl and lang_commits.get(pl, 0) > 0:
                primary_langs.add(pl)

        count = len(primary_langs)
        score = _clamp01(count / _LANG_CAP)
        return score, count

    # ------------------------------------------------------------------
    # Sub-score 2: Technology adoption over time
    # ------------------------------------------------------------------

    def _compute_technology_adoption(
        self,
        repo_nodes: List[dict],
    ) -> Tuple[float, int]:
        """
        Count the number of "era transitions" — distinct primary languages
        first appearing in repos created in a later year than previously seen
        languages.  Repositories without a primary language are skipped.

        Returns (normalised_score [0,1], transition_count).
        """
        # Build a timeline: year → set of primary languages first seen that year
        year_langs: Dict[int, Set[str]] = defaultdict(set)
        for repo in repo_nodes:
            pl = (repo.get("primaryLanguage") or {}).get("name")
            if not pl:
                continue
            created = repo.get("createdAt", "")
            try:
                year = datetime.fromisoformat(
                    created.replace("Z", "+00:00")
                ).year
            except ValueError:
                continue
            year_langs[year].add(pl)

        if not year_langs:
            return 0.0, 0

        transitions = 0
        seen: Set[str] = set()
        for year in sorted(year_langs):
            new_this_year = year_langs[year] - seen
            if seen and new_this_year:      # a genuinely new language adopted
                transitions += len(new_this_year)
            seen |= year_langs[year]

        score = _clamp01(transitions / _TRANSITION_CAP)
        return score, transitions

    # ------------------------------------------------------------------
    # Sub-score 3: Domain flexibility (topic entropy)
    # ------------------------------------------------------------------

    def _compute_domain_flexibility(
        self,
        contrib_nodes: List[dict],
    ) -> Tuple[float, float]:
        """
        Compute Shannon entropy (bits) over topic-tag frequencies.

        Returns (normalised_score [0,1], entropy_bits).
        """
        topic_counts: Dict[str, int] = defaultdict(int)
        for repo in contrib_nodes:
            for t_node in repo.get("repositoryTopics", {}).get("nodes", []):
                topic = (t_node.get("topic") or {}).get("name")
                if topic:
                    topic_counts[topic] += 1

        if not topic_counts:
            return 0.0, 0.0

        entropy = _shannon_entropy(list(topic_counts.values()))
        score   = _clamp01(entropy / _ENTROPY_CAP)
        return score, round(entropy, 4)

    # ------------------------------------------------------------------
    # Sub-score 4: Resilience (PR bounce-back after rejection)
    # ------------------------------------------------------------------

    def _compute_resilience(
        self,
        all_pr_nodes: List[dict],
    ) -> Tuple[float, int, int]:
        """
        For each closed-but-not-merged PR, check whether the user opened
        another PR in the same repo within _BOUNCE_BACK_DAYS.

        Returns (normalised_score [0,1], total_closed_prs, bounced_back_count).
        """
        # Group all PRs by repo
        repo_prs: Dict[str, List[dict]] = defaultdict(list)
        for pr in all_pr_nodes:
            repo = (pr.get("repository") or {}).get("nameWithOwner", "")
            if repo:
                repo_prs[repo].append(pr)

        closed_total   = 0
        bounced_back   = 0
        window         = timedelta(days=_BOUNCE_BACK_DAYS)

        for repo, prs in repo_prs.items():
            # Closed without merge = rejected
            rejected = []
            for pr in prs:
                if pr.get("mergedAt"):          # merged — not rejected
                    continue
                if pr.get("state") == "CLOSED" or (
                    pr.get("closedAt") and not pr.get("mergedAt")
                ):
                    closed_at_str = pr.get("closedAt")
                    if not closed_at_str:
                        continue
                    try:
                        closed_at = datetime.fromisoformat(
                            closed_at_str.replace("Z", "+00:00")
                        )
                        rejected.append(closed_at)
                        closed_total += 1
                    except ValueError:
                        continue

            # All PR open dates in this repo
            open_dates = []
            for pr in prs:
                created_str = pr.get("createdAt")
                if not created_str:
                    continue
                try:
                    open_dates.append(
                        datetime.fromisoformat(
                            created_str.replace("Z", "+00:00")
                        )
                    )
                except ValueError:
                    continue
            open_dates_sorted = sorted(open_dates)

            # For each rejection, check if any subsequent PR opened within window
            for rejected_at in rejected:
                for open_dt in open_dates_sorted:
                    if open_dt > rejected_at and (open_dt - rejected_at) <= window:
                        bounced_back += 1
                        break

        if closed_total == 0:
            return 0.0, 0, 0

        fraction = bounced_back / closed_total
        score    = _clamp01(fraction / _RESILIENCE_CAP)
        return score, closed_total, bounced_back

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self) -> dict:
        """
        Run all four sub-analyses and return the combined adaptability result.
        """
        print(f"[AdaptabilityAnalyzer] Fetching repo language data for {self.username}…")
        repo_nodes = _paginate(
            _GQL_REPOS_LANGUAGES, ["repositories"], self.headers, self.username
        )

        print(f"[AdaptabilityAnalyzer] Fetching contributed-to topics…")
        contrib_nodes = _paginate(
            _GQL_CONTRIBUTED_TOPICS, ["repositoriesContributedTo"],
            self.headers, self.username,
        )

        print(f"[AdaptabilityAnalyzer] Fetching PR history…")
        all_pr_nodes = _paginate(
            _GQL_ALL_PRS, ["pullRequests"], self.headers, self.username
        )

        print(f"[AdaptabilityAnalyzer] Fetching commit depth per language…")
        lang_commits = _fetch_commit_depth_per_language(self.username, self.headers)

        # Sub-scores [0,1]
        div_score,   n_langs       = self._compute_language_diversity(repo_nodes, lang_commits)
        adopt_score, n_transitions = self._compute_technology_adoption(repo_nodes)
        domain_score, entropy_bits = self._compute_domain_flexibility(contrib_nodes)
        resil_score, closed_total, bounced = self._compute_resilience(all_pr_nodes)

        sub_scores = [div_score, adopt_score, domain_score, resil_score]
        mean01     = sum(sub_scores) / len(sub_scores)
        #final      = _to_signed(mean01)
        final      = mean01

        return {
            "score": round(final, 6),
            "details": {
                "language_diversity_score":   round(div_score,    4),
                "technology_adoption_score":  round(adopt_score,  4),
                "domain_flexibility_score":   round(domain_score, 4),
                "resilience_score":           round(resil_score,  4),
                "distinct_primary_languages": n_langs,
                "language_transitions":       n_transitions,
                "topic_entropy_bits":         entropy_bits,
                "closed_prs_total":           closed_total,
                "bounced_back_prs":           bounced,
            },
        }

    def get_results_as_json(self) -> str:
        return json.dumps(self.analyze(), indent=2)
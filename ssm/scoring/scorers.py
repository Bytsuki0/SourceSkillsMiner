"""
Per-area scorers.

Each scorer turns an analyzer's raw output into the ``{'score', 'details'}``
shape the report expects. This isolates the normalization math (previously the
``compute_*_score`` functions in ScoringSys) from data collection, so analyzers
stay about *gathering* and scorers stay about *grading*. The numbers are ported
verbatim.
"""

from __future__ import annotations

import statistics
from typing import Dict

from ssm.core.normalizers import linear_cap, safe_log_norm

__all__ = [
    "OSSScorer",
    "StatusScorer",
    "SentimentScorer",
    "CommitmentScorer",
    "AdaptabilityScorer",
]


class OSSScorer:
    """Issue-resolution, PR-merge and commit-activity rates."""

    def score(self, data: Dict) -> Dict:
        total_commits = data.get("total_commits", 0)
        if not isinstance(total_commits, (int, float)):
            total_commits = 0

        num_open_issues = data.get("open_issues", 0)
        num_open_prs = data.get("open_prs", 0)
        num_closed_issues = data.get("closed_issues", 0)
        # closed_prs already includes merged (matches old REST state:closed)
        num_closed_prs = data.get("closed_prs", 0)
        num_merged_prs = data.get("merged_prs", 0)
        if not isinstance(num_merged_prs, (int, float)):
            num_merged_prs = 0

        issue_total = num_open_issues + num_closed_issues
        pr_total = num_open_prs + num_closed_prs
        issue_resolution_rate = (num_closed_issues / issue_total) if issue_total > 0 else 0.0
        pr_merge_rate = (num_merged_prs / pr_total) if pr_total > 0 else 0.0
        commits_norm = safe_log_norm(total_commits, scale=200.0)

        subs = {
            "issue_resolution_rate": min(1.0, issue_resolution_rate),
            "pr_merge_rate": min(1.0, pr_merge_rate),
            "commits_activity": commits_norm,
        }

        if issue_total == 0 and pr_total == 0 and total_commits == 0:
            return {"score": 0.0, "details": subs}
        return {"score": statistics.mean(list(subs.values())), "details": subs}


class StatusScorer:
    """Lines-changed, commit and weekly-streak frequencies."""

    def score(self, agg: Dict) -> Dict:
        lines = agg.get("Linhas_trocas", 0)
        commits = agg.get("Total_commits", 0)
        streak = agg.get("Streak_contribuicoes_consecutivas", 0)

        subs = {
            "lines_frequency": safe_log_norm(lines, scale=50000.0),
            "commits_frequency": safe_log_norm(commits, scale=200.0),
            "week_streak": linear_cap(streak, cap=10),
        }

        if lines == 0 and commits == 0 and streak == 0:
            return {"score": 0.0, "details": subs}
        return {"score": statistics.mean(list(subs.values())), "details": subs}


class SentimentScorer:
    """Fraction of repos with net-positive discussion sentiment.

    Returns a signed [-1, 1]-style ratio; the facade maps it to [0, 1] when
    assembling the area, matching the original ``to_01`` step.
    """

    def score(self, sentiments: Dict[str, float]) -> Dict:
        upper = base = 0
        for value in sentiments.values():
            if value > 0:
                base += 1
                upper += 1
            if value < 0:
                base += 1
        ratio = upper / base if base > 0 else 0.0
        return {"score": ratio, "details": sentiments}


class CommitmentScorer:
    """Counts how many of the four commitment criteria are met."""

    REQUIRED_KEYS = [
        "has_12_month_streak",
        "has_6_month_streak",
        "has_repo_at_50th_percentile_commits",
        "at_75th_percentile_followers",
    ]

    def score(self, data: Dict) -> Dict:
        summary = data.get("summary")
        if summary is None:
            return {
                "score": 0.0,
                "details": {
                    "error": "Missing 'summary' section in analysis result.",
                    "criteria_met": {},
                },
            }

        total_points = 0
        criteria_met = {}
        for key in self.REQUIRED_KEYS:
            if key not in summary or not isinstance(summary[key], bool):
                criteria_met[key] = False
                continue
            criteria_met[key] = summary[key]
            total_points += int(summary[key])

        normalized_score = total_points / len(self.REQUIRED_KEYS)
        return {
            "score": normalized_score,
            "details": {
                "criteria_met": criteria_met,
                "total_points": total_points,
                "max_points": len(self.REQUIRED_KEYS),
                "percentage": normalized_score * 100,
            },
        }


class AdaptabilityScorer:
    """Reshapes the adaptability analyzer output into the report's detail keys."""

    _DETAIL_DEFAULTS = {
        "language_diversity_score": 0.0,
        "technology_adoption_score": 0.0,
        "domain_flexibility_score": 0.0,
        "resilience_score": 0.0,
        "distinct_primary_languages": 0,
        "language_transitions": 0,
        "topic_entropy_bits": 0.0,
        "closed_prs_total": 0,
        "bounced_back_prs": 0,
    }

    def score(self, data: Dict) -> Dict:
        details = data.get("details", {})
        return {
            "score": data.get("score", 0.0),
            "details": {
                key: details.get(key, default)
                for key, default in self._DETAIL_DEFAULTS.items()
            },
        }

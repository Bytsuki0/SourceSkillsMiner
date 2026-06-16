"""
SourceSkillsMiner (ssm)

An object-oriented GitHub profile scoring system.

The public entry point is :class:`ssm.scoring.facade.ScoringFacade`, which wires
the whole subsystem together (config, HTTP client, shared services, analyzers and
scorers) behind a single ``analyze()`` call. The command line lives in
:mod:`ssm.cli`.

Layout
------
    core/       cross-cutting building blocks (config, http, normalizers,
                events/Observer, serialization)
    services/   shared GitHub data sources reused by every analyzer
    analyzers/  one cohesive analyzer per scored area
    scoring/    per-area scorers + the orchestrating Facade
"""

__all__ = ["ScoringFacade"]


def __getattr__(name):
    # Lazy re-export so `from ssm import ScoringFacade` works without importing
    # the heavy analyzer stack at package-import time.
    if name == "ScoringFacade":
        from ssm.scoring.facade import ScoringFacade
        return ScoringFacade
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

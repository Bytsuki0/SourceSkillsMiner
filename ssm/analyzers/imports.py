"""
Import / package scanner (GraphQL).

Ports ``WorkTypeAnalyzer.GitHubImportScanner`` onto the GraphQL Git ``object``
API. Instead of one REST ``git/trees?recursive=1`` call followed by a REST
``contents`` call per file, it walks the repo tree breadth-first using
``HEAD:<path>`` tree expressions and reads matching source files' text inline
from the same query — so there is no separate per-file fetch.

Note: GraphQL has no recursive-tree flag, so the walk issues one query per
directory. Repo listing is delegated to :class:`RepositoryService`; the
language/pattern tables live in :mod:`ssm.analyzers.import_patterns`.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

from ssm.analyzers.base import Analyzer
from ssm.analyzers.import_patterns import (
    CODE_EXTENSIONS,
    EXTENSION_TO_LANGUAGE,
    GENERIC_IMPORT_KEYWORDS,
    IMPORT_PATTERNS,
    normalise_package,
)
from ssm.core.client import GitHubClient
from ssm.services.repositories import RepositoryService

__all__ = ["ImportScanner"]

# Lists a single directory's entries, with source-file contents inlined.
_TREE_QUERY = """
query($owner: String!, $name: String!, $expr: String!) {
  repository(owner: $owner, name: $name) {
    object(expression: $expr) {
      ... on Tree {
        entries {
          name
          path
          type
          object { ... on Blob { isBinary text } }
        }
      }
    }
  }
}
"""

# Safety cap on directories visited per repo so a deep monorepo can't run away.
_MAX_DIRS_PER_REPO = 200


class ImportScanner(Analyzer):
    """Scans repositories for the packages a user imports, grouped by language."""

    SKIP_DIRS = frozenset([
        'node_modules', 'vendor', 'dist', 'build', '.git',
        '__pycache__', '.venv', 'venv', 'env', 'target',
        'bin', 'obj', 'out', 'coverage', '.nyc_output',
    ])

    def __init__(
        self,
        client: GitHubClient,
        username: str,
        repos: RepositoryService,
    ) -> None:
        super().__init__(client, username)
        self._repos = repos

    def _scan_repo(self, full_name: str, max_files: int) -> List[Tuple[str, str, str]]:
        """
        Breadth-first walk of a repo's default branch via GraphQL.

        Returns a list of (path, language, content) for up to ``max_files``
        source files.
        """
        if "/" not in full_name:
            return []
        owner, name = full_name.split("/", 1)

        results: List[Tuple[str, str, str]] = []
        queue: List[str] = [""]          # directory paths relative to repo root
        dirs_visited = 0

        while queue and len(results) < max_files and dirs_visited < _MAX_DIRS_PER_REPO:
            dir_path = queue.pop(0)
            dirs_visited += 1
            expr = f"HEAD:{dir_path}" if dir_path else "HEAD:"

            data = self.client.graphql(
                _TREE_QUERY, {"owner": owner, "name": name, "expr": expr}
            )
            obj = ((data or {}).get("repository") or {}).get("object") or {}
            entries = obj.get("entries")
            if not entries:
                continue

            for entry in entries:
                etype = entry.get("type")
                epath = entry.get("path", "")
                ename = entry.get("name", "")

                if etype == "tree":
                    if ename in self.SKIP_DIRS:
                        continue
                    queue.append(epath)
                    continue

                if etype != "blob" or "." not in epath:
                    continue
                if any(skip in epath.split("/") for skip in self.SKIP_DIRS):
                    continue
                ext = "." + epath.rsplit(".", 1)[-1]
                language = EXTENSION_TO_LANGUAGE.get(ext)
                if ext not in CODE_EXTENSIONS or not language:
                    continue

                blob = entry.get("object") or {}
                if blob.get("isBinary"):
                    continue
                text = blob.get("text")
                if not text:
                    continue

                results.append((epath, language, text))
                if len(results) >= max_files:
                    break
            time.sleep(0.05)

        return results

    def _extract_imports(self, content: str, language: str) -> Set[str]:
        found: Set[str] = set()
        lines = content.splitlines()
        patterns = IMPORT_PATTERNS.get(language, [])
        for line in lines:
            for pattern in patterns:
                m = pattern.search(line)
                if m:
                    for part in m.group(1).split(','):
                        pkg = normalise_package(part, language)
                        if pkg:
                            found.add(pkg)
        for line in lines:
            if GENERIC_IMPORT_KEYWORDS.match(line):
                quoted = re.search(r"""['"]([^'"]+)['"]""", line)
                if quoted:
                    pkg = normalise_package(quoted.group(1), language)
                    if pkg:
                        found.add(pkg)
                else:
                    bare = re.search(
                        r"""
                        (?:import|require|include|using|use|open|alias|library|extern\s+crate)
                        \s+([^\s;(){},]+)
                        """,
                        line,
                        re.IGNORECASE | re.VERBOSE,
                    )
                    if bare:
                        pkg = normalise_package(bare.group(1), language)
                        if pkg:
                            found.add(pkg)
        return found

    def analyze_imports(
        self,
        max_repos: int = 100,
        max_files_per_repo: int = 100,
    ) -> dict:
        repos = self._repos.list_repos()[:max_repos]
        language_packages: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        language_file_count: Dict[str, int] = defaultdict(int)
        repo_summaries = []
        total_files = 0

        self.emit(f"Scanning {len(repos)} repositories for imports...", stage="imports")

        for idx, repo in enumerate(repos):
            full_name = repo['full_name']
            self.emit(
                f"  [{idx + 1}/{len(repos)}] {full_name}",
                stage="imports", fraction=(idx + 1) / len(repos) if repos else None,
            )

            repo_languages: Set[str] = set()
            files_in_repo = 0

            for path, language, content in self._scan_repo(full_name, max_files_per_repo):
                imports = self._extract_imports(content, language)
                if imports:
                    for pkg in imports:
                        language_packages[language][pkg] += 1
                    language_file_count[language] += 1
                    repo_languages.add(language)
                files_in_repo += 1
                total_files += 1

            if repo_languages:
                repo_summaries.append({
                    'repository':      full_name,
                    'files_analyzed':  files_in_repo,
                    'languages_found': sorted(repo_languages),
                })

        languages_output = {}
        for lang in sorted(language_packages.keys()):
            pkg_dict = language_packages[lang]
            sorted_pkgs = dict(
                sorted(pkg_dict.items(), key=lambda x: x[1], reverse=True)
            )
            languages_output[lang] = {
                'files_scanned': language_file_count[lang],
                'packages':      sorted_pkgs,
            }

        return {
            'username':             self.username,
            'analysis_date':        datetime.now().isoformat(),
            'total_repos_analyzed': len(repos),
            'total_files_analyzed': total_files,
            'languages':            languages_output,
            'repositories':         repo_summaries,
        }

    def analyze(self, max_repos: int = 20, max_files_per_repo: int = 30) -> dict:
        """Run the scan and shape it into the report's ``import_scan`` block."""
        try:
            import_data = self.analyze_imports(
                max_repos=max_repos, max_files_per_repo=max_files_per_repo
            )
            if not isinstance(import_data, dict):
                return {
                    'error': f'Invalid return type: {type(import_data)}',
                    'languages': {}, 'repositories': [],
                }
            return {'error': None, **import_data}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {
                'error': str(exc),
                'username': self.username,
                'analysis_date': None,
                'total_repos_analyzed': 0,
                'total_files_analyzed': 0,
                'languages': {},
                'repositories': [],
            }

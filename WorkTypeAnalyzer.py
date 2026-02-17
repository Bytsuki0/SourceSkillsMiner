import requests
import json
import time
import re
import base64
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Optional


# ---------------------------------------------------------------------------
# Language / commit distribution analyzer
# ---------------------------------------------------------------------------

class GitHubLanguageCommitAnalyzer:
    """
    Analyzes programming language usage across a user's GitHub repositories.
    Distributes commit and line-change statistics proportionally based on
    language byte counts in each repository.

    Accepts an optional `prefetched_stats` dict to avoid redundant API calls
    when GitHubStatsAnalyzerAllTime has already fetched contributor data.

    Expected format of prefetched_stats:
        {
            "repo_name": {
                "Linhas_trocas": <int>,
                "Total_commits": <int>,
                ...
            },
            ...
        }
    The keys must match repo['name'] (not full_name).
    """

    def __init__(
        self,
        username: str,
        token: str,
        prefetched_stats: Optional[Dict[str, dict]] = None,
    ):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
        }
        # Cache keyed by repo full_name → (lines_changed, commits).
        # Pre-populate from already-fetched data when provided.
        self._stats_cache: Dict[str, Optional[tuple]] = {}
        if prefetched_stats:
            for repo_name, stats in prefetched_stats.items():
                lines   = stats.get('Linhas_trocas', 0)
                commits = stats.get('Total_commits', 0)
                # We store under the full_name format used internally.
                # The caller must also pass the username so we can rebuild it,
                # OR we just key on repo short-name and resolve in the method.
                # Keying on short name here; _get_commit_stats receives full_name
                # but we split it back to short name for lookup.
                self._stats_cache[repo_name] = (lines, commits)

    def _get_user_repositories(self):
        repos, page = [], 1
        while True:
            url = f"{self.api_url}/users/{self.username}/repos?per_page=100&page={page}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            repos.extend(data)
            page += 1
        return repos

    def _get_repo_languages(self, full_name: str) -> dict:
        url = f"{self.api_url}/repos/{full_name}/languages"
        resp = requests.get(url, headers=self.headers)
        return resp.json() if resp.status_code == 200 else {}

    def _get_commit_stats(self, full_name: str) -> Optional[tuple]:
        """
        Returns (total_lines_changed, total_commits) for the authenticated user
        in the given repository.

        If the data was pre-fetched by GitHubStatsAnalyzerAllTime it is returned
        directly from the cache — no API call is made.
        """
        # Cache lookup: try both short name and full_name as keys
        short_name = full_name.split('/')[-1]
        if short_name in self._stats_cache:
            return self._stats_cache[short_name]
        if full_name in self._stats_cache:
            return self._stats_cache[full_name]

        # Cache miss — fetch from GitHub
        result = self._fetch_commit_stats_from_api(full_name)

        # Store in cache for any future call
        self._stats_cache[full_name] = result
        return result

    def _fetch_commit_stats_from_api(self, full_name: str) -> Optional[tuple]:
        """Fetches contributor stats from the GitHub API (used only on cache miss)."""
        url = f"{self.api_url}/repos/{full_name}/stats/contributors"
        for _ in range(10):
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 202:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data:
                return None
            for contributor in data:
                author = contributor.get('author')
                if author and author.get('login', '').lower() == self.username.lower():
                    total_lines = sum(
                        w.get('a', 0) + w.get('d', 0)
                        for w in contributor.get('weeks', [])
                    )
                    total_commits = sum(
                        w.get('c', 0) for w in contributor.get('weeks', [])
                    )
                    return total_lines, total_commits
        return None

    def analyze_language_usage(self) -> Dict[str, List[int]]:
        """
        Returns a dict mapping language names to [lines_changed, commits].
        """
        language_stats = defaultdict(lambda: [0, 0])
        for repo in self._get_user_repositories():
            full_name = repo['full_name']
            languages = self._get_repo_languages(full_name)
            commit_stats = self._get_commit_stats(full_name)
            if not commit_stats:
                continue
            lines_changed, commits = commit_stats
            total_bytes = sum(languages.values())
            if total_bytes == 0:
                continue
            for lang, byte_count in languages.items():
                proportion = byte_count / total_bytes
                language_stats[lang][0] += int(lines_changed * proportion)
                language_stats[lang][1] += int(commits * proportion)
        return dict(language_stats)

    def get_results_as_json(self) -> str:
        return json.dumps(self.analyze_language_usage(), indent=2)


# ---------------------------------------------------------------------------
# Import / package scanner (unchanged)
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    '.py':    'Python',
    '.js':    'JavaScript',
    '.ts':    'TypeScript',
    '.jsx':   'JavaScript (JSX)',
    '.tsx':   'TypeScript (TSX)',
    '.java':  'Java',
    '.go':    'Go',
    '.rs':    'Rust',
    '.cpp':   'C++',
    '.cc':    'C++',
    '.cxx':   'C++',
    '.c':     'C',
    '.h':     'C/C++ Header',
    '.hpp':   'C++ Header',
    '.cs':    'C#',
    '.php':   'PHP',
    '.rb':    'Ruby',
    '.swift': 'Swift',
    '.kt':    'Kotlin',
    '.scala': 'Scala',
    '.r':     'R',
    '.R':     'R',
    '.m':     'Objective-C / MATLAB',
    '.dart':  'Dart',
    '.lua':   'Lua',
    '.ex':    'Elixir',
    '.exs':   'Elixir',
    '.erl':   'Erlang',
    '.hrl':   'Erlang',
    '.hs':    'Haskell',
    '.ml':    'OCaml',
    '.mli':   'OCaml',
    '.jl':    'Julia',
    '.pl':    'Perl',
    '.pm':    'Perl',
}

CODE_EXTENSIONS: Set[str] = set(EXTENSION_TO_LANGUAGE.keys())

IMPORT_PATTERNS: Dict[str, List[re.Pattern]] = {
    'Python': [
        re.compile(r'^\s*import\s+([\w,\s]+)'),
        re.compile(r'^\s*from\s+([\w.]+)\s+import'),
    ],
    'JavaScript': [
        re.compile(r"""(?:import|export)[^'"]*['"]([^'"]+)['"]"""),
        re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
    ],
    'TypeScript': [
        re.compile(r"""(?:import|export)[^'"]*['"]([^'"]+)['"]"""),
        re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
    ],
    'JavaScript (JSX)': [
        re.compile(r"""(?:import|export)[^'"]*['"]([^'"]+)['"]"""),
        re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
    ],
    'TypeScript (TSX)': [
        re.compile(r"""(?:import|export)[^'"]*['"]([^'"]+)['"]"""),
        re.compile(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
    ],
    'Java': [
        re.compile(r'^\s*import\s+([\w.]+)\s*;'),
        re.compile(r'^\s*package\s+([\w.]+)\s*;'),
    ],
    'Go': [
        re.compile(r'^\s*import\s+"([^"]+)"'),
        re.compile(r'^\s*"([^"]+)"'),
    ],
    'Rust': [
        re.compile(r'^\s*use\s+([\w::<>]+)'),
        re.compile(r'^\s*extern\s+crate\s+([\w]+)'),
    ],
    'C': [
        re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]'),
    ],
    'C++': [
        re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]'),
        re.compile(r'^\s*using\s+namespace\s+([\w:]+)'),
        re.compile(r'^\s*import\s+<([^>]+)>'),
    ],
    'C/C++ Header': [
        re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]'),
    ],
    'C++ Header': [
        re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]'),
        re.compile(r'^\s*using\s+namespace\s+([\w:]+)'),
    ],
    'C#': [
        re.compile(r'^\s*using\s+([\w.]+)\s*;'),
    ],
    'PHP': [
        re.compile(r'^\s*use\s+([\w\\\\]+)\s*;'),
        re.compile(r"""(?:require|include)(?:_once)?\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
        re.compile(r"""(?:require|include)(?:_once)?\s+['"]([^'"]+)['"]"""),
    ],
    'Ruby': [
        re.compile(r"""^\s*require(?:_relative)?\s+['"]([^'"]+)['"]"""),
        re.compile(r"""^\s*gem\s+['"]([^'"]+)['"]"""),
    ],
    'Swift': [
        re.compile(r'^\s*import\s+([\w.]+)'),
    ],
    'Kotlin': [
        re.compile(r'^\s*import\s+([\w.*]+)'),
        re.compile(r'^\s*package\s+([\w.]+)'),
    ],
    'Scala': [
        re.compile(r'^\s*import\s+([\w._{},\s]+)'),
        re.compile(r'^\s*package\s+([\w.]+)'),
    ],
    'R': [
        re.compile(r"""(?:library|require)\s*\(\s*['"]?([\w.]+)['"]?\s*\)"""),
    ],
    'Dart': [
        re.compile(r"""^\s*import\s+['"]([^'"]+)['"]"""),
    ],
    'Lua': [
        re.compile(r"""^\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)"""),
        re.compile(r"""^\s*require\s+['"]([^'"]+)['"]"""),
    ],
    'Elixir': [
        re.compile(r'^\s*(?:import|alias|use|require)\s+([\w.]+)'),
    ],
    'Erlang': [
        re.compile(r'^\s*-include\s*\(\s*"([^"]+)"\s*\)'),
        re.compile(r'^\s*-include_lib\s*\(\s*"([^"]+)"\s*\)'),
    ],
    'Haskell': [
        re.compile(r'^\s*import\s+(?:qualified\s+)?([\w.]+)'),
    ],
    'OCaml': [
        re.compile(r'^\s*open\s+([\w.]+)'),
        re.compile(r'^\s*#require\s+"([^"]+)"'),
    ],
    'Julia': [
        re.compile(r'^\s*using\s+([\w,\s.]+)'),
        re.compile(r'^\s*import\s+([\w,\s.]+)'),
    ],
    'Perl': [
        re.compile(r'^\s*use\s+([\w:]+)'),
        re.compile(r'^\s*require\s+([\w:/"\']+)'),
    ],
    'Objective-C / MATLAB': [
        re.compile(r'^\s*#\s*import\s+[<"]([^>"]+)[>"]'),
        re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]'),
    ],
}

GENERIC_IMPORT_KEYWORDS = re.compile(
    r"""
    ^\s*
    (?:
        import   |
        from\s+\w |
        require  |
        include  |
        using    |
        use\s    |
        extern\s+crate |
        open\s   |
        alias\s  |
        library\s |
        \#\s*include |
        \#\s*import
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _normalise_package(raw: str, language: str) -> Optional[str]:
    raw = raw.strip().strip('"\'').split()[0]
    if raw.startswith('.') or raw.startswith('/'):
        return None
    if language in ('JavaScript', 'TypeScript', 'JavaScript (JSX)', 'TypeScript (TSX)'):
        if raw.startswith('@'):
            raw = raw.lstrip('@')
        raw = raw.split('/')[0]
    if language == 'Python':
        raw = raw.split('.')[0].split(',')[0].strip()
    if language in ('Java', 'Kotlin', 'Scala'):
        parts = raw.rstrip('*').rstrip('.').split('.')
        raw = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
    if language == 'Rust':
        raw = raw.split('::')[0]
    if language == 'Dart':
        if raw.startswith('package:'):
            raw = raw[len('package:'):].split('/')[0]
    return raw if raw else None


class GitHubImportScanner:
    SKIP_DIRS = frozenset([
        'node_modules', 'vendor', 'dist', 'build', '.git',
        '__pycache__', '.venv', 'venv', 'env', 'target',
        'bin', 'obj', 'out', 'coverage', '.nyc_output',
    ])

    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
        }

    def _get_user_repositories(self) -> List[dict]:
        repos, page = [], 1
        while True:
            url = (
                f"{self.api_url}/users/{self.username}/repos"
                f"?per_page=100&page={page}"
            )
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            if not isinstance(data, list) or not data:
                break
            repos.extend(data)
            page += 1
        return repos

    def _get_repo_tree(self, full_name: str) -> Optional[List[dict]]:
        for branch in ('main', 'master', 'develop'):
            url = (
                f"{self.api_url}/repos/{full_name}/git/trees"
                f"/{branch}?recursive=1"
            )
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                try:
                    return resp.json().get('tree', [])
                except (ValueError, json.JSONDecodeError):
                    continue
        return None

    def _get_file_content(self, full_name: str, file_path: str) -> Optional[str]:
        url = f"{self.api_url}/repos/{full_name}/contents/{file_path}"
        resp = requests.get(url, headers=self.headers)
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
            if data.get('encoding') == 'base64':
                return base64.b64decode(data['content']).decode('utf-8', errors='ignore')
        except Exception:
            pass
        return None

    def _extract_imports(self, content: str, language: str) -> Set[str]:
        found: Set[str] = set()
        lines = content.splitlines()
        patterns = IMPORT_PATTERNS.get(language, [])
        for line in lines:
            for pattern in patterns:
                m = pattern.search(line)
                if m:
                    raw = m.group(1)
                    for part in raw.split(','):
                        pkg = _normalise_package(part, language)
                        if pkg:
                            found.add(pkg)
        for line in lines:
            if GENERIC_IMPORT_KEYWORDS.match(line):
                quoted = re.search(r"""['"]([^'"]+)['"]""", line)
                if quoted:
                    pkg = _normalise_package(quoted.group(1), language)
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
                        pkg = _normalise_package(bare.group(1), language)
                        if pkg:
                            found.add(pkg)
        return found

    def analyze_imports(
        self,
        max_repos: int = 100,
        max_files_per_repo: int = 100,
    ) -> dict:
        repos = self._get_user_repositories()[:max_repos]
        language_packages: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        language_file_count: Dict[str, int] = defaultdict(int)
        repo_summaries = []
        total_files = 0

        print(f"Scanning {len(repos)} repositories for imports…")

        for idx, repo in enumerate(repos):
            full_name = repo['full_name']
            print(f"  [{idx + 1}/{len(repos)}] {full_name}")
            tree = self._get_repo_tree(full_name)
            if not tree:
                continue
            code_files = [
                item for item in tree
                if item['type'] == 'blob'
                and not any(skip in item['path'].split('/') for skip in self.SKIP_DIRS)
                and ('.' in item['path'])
                and ('.' + item['path'].rsplit('.', 1)[-1]) in CODE_EXTENSIONS
            ][:max_files_per_repo]

            repo_languages: Set[str] = set()
            files_in_repo = 0

            for file_item in code_files:
                path = file_item['path']
                ext = '.' + path.rsplit('.', 1)[-1]
                language = EXTENSION_TO_LANGUAGE.get(ext)
                if not language:
                    continue
                content = self._get_file_content(full_name, path)
                if not content:
                    continue
                imports = self._extract_imports(content, language)
                if imports:
                    for pkg in imports:
                        language_packages[language][pkg] += 1
                    language_file_count[language] += 1
                    repo_languages.add(language)
                files_in_repo += 1
                total_files += 1
                time.sleep(0.1)

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

    def get_results_as_json(self, **kwargs) -> str:
        return json.dumps(self.analyze_imports(**kwargs), indent=2)


# ---------------------------------------------------------------------------
# Entry point — stats from first class feed directly into second
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import configparser
    import sys
    import os

    # Allow importing GitHubStatsAnalyzerAllTime from the first file.
    # Adjust the path below if your file is located elsewhere.
    sys.path.insert(0, os.path.dirname(__file__))
    from StatusAnaliser import GitHubStatsAnalyzerAllTime

    config = configparser.ConfigParser()
    config.read('config.ini')
    username = config['github']['username']
    token    = config['github']['token']

    # --- Step 1: run the all-time stats (fetches contributor data once) ---
    print("=== ALL-TIME STATS ===\n")
    stats_analyzer = GitHubStatsAnalyzerAllTime(username, token)
    raw_stats = stats_analyzer.analyze_all()   # dict[repo_name → stats_dict]
    print(json.dumps(raw_stats, indent=2))

    # --- Step 2: pass those results as a cache into the language analyzer ---
    # raw_stats keys are short repo names; values already have Linhas_trocas
    # and Total_commits — exactly what the language analyzer needs.
    print("\n=== LANGUAGE USAGE ANALYSIS (using cached stats) ===\n")
    lang_analyzer = GitHubLanguageCommitAnalyzer(
        username,
        token,
        prefetched_stats=raw_stats,   # <-- cache injected here; no re-fetch
    )
    print(lang_analyzer.get_results_as_json())

    # --- Import / package scan (independent, no overlap) ---
    print("\n=== IMPORT & PACKAGE SCAN ===\n")
    import_scanner = GitHubImportScanner(username, token)
    print(import_scanner.get_results_as_json(max_repos=1000, max_files_per_repo=1000))
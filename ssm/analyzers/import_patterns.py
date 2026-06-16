"""
Static language/import data for :class:`~ssm.analyzers.imports.ImportScanner`.

Pure data + one normalization helper, separated from the scanner logic so the
scanner class stays focused on traversal and extraction.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

__all__ = [
    "EXTENSION_TO_LANGUAGE",
    "CODE_EXTENSIONS",
    "IMPORT_PATTERNS",
    "GENERIC_IMPORT_KEYWORDS",
    "normalise_package",
]

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


def normalise_package(raw: str, language: str) -> Optional[str]:
    """Reduce a raw import token to a comparable package name, or None."""
    raw = raw.strip().strip('"\'')
    parts = raw.split()
    if not parts:
        return None
    raw = parts[0]
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

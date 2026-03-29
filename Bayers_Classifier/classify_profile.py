"""
classify_profile.py

Loads the saved Naive Bayes model and classifies developer profiles.

Accepts three input modes:
  1. A SourceSkillsMiner JSON file (same format as json_to_csv.py reads)
  2. A CSV row file (same schema as features.csv)
  3. Interactive: pass --lang / --lib flags directly on the CLI

Output:
  Predicted category + confidence % for every class, sorted by probability.
  Result can also be saved as JSON with --output.

Usage
──────
    # Classify from a mined JSON
    python classify_profile.py --json albatrocity.json

    # Classify from a CSV row (first data row of a single-user CSV)
    python classify_profile.py --csv single_user.csv

    # Classify by passing features directly
    python classify_profile.py --lang Python "Jupyter Notebook" --lib numpy pandas sklearn torch

    # Save result to JSON
    python classify_profile.py --json albatrocity.json --output result.json

    # Use a custom model path
    python classify_profile.py --json profile.json --model models/developer_classifier.joblib
"""

import os
import sys
import csv
import json
import argparse
from typing import Dict, List, Optional, Tuple

import joblib

# ── Same column lists used during training ────────────────────────────────────
LANG_COLS = ['top1lang', 'top2lang', 'top3lang', 'top4lang', 'top5lang']
LIB_COLS  = ['top1lib',  'top2lib',  'top3lib',  'top4lib',  'top5lib',  'top6lib', 'top7lib']

DEFAULT_MODEL = os.path.join('models', 'developer_classifier.joblib')


# ===========================================================================
# Library validator (same rules as json_to_csv.py — keep in sync)
# ===========================================================================

import re

_LOCAL_EXTENSIONS = frozenset({
    '.rb', '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
    '.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx', '.cs', '.php',
    '.lua', '.ex', '.exs', '.erl', '.hrl', '.hs', '.ml', '.jl', '.pl',
    '.pm', '.swift', '.kt', '.scala', '.dart', '.r', '.html', '.htm',
    '.css', '.scss', '.sass', '.json', '.xml', '.yaml', '.yml', '.sh',
    '.bash', '.sql', '.proto', '.pdf', '.csv', '.txt', '.md',
})
_NOISE_WORDS = frozenset({
    'a','an','and','as','at','be','by','do','for','from','if','in','into',
    'is','it','not','of','on','or','per','the','to','up','us','via','we',
    'with','you','all','any','are','but','can','did','get','had','has',
    'have','her','him','his','how','its','let','may','more','my','new',
    'no','now','one','our','out','see','set','so','than','that','their',
    'them','then','there','they','this','those','too','type','use','used',
    'using','very','was','were','when','which','who','will','would','your',
    'indentation','backticks','favorites','undefined','true','false',
    'null','none','error','example','note','todo','fixme',
})
_CPP_KNOWN_STDLIB = frozenset({
    'algorithm','array','atomic','bitset','chrono','complex','deque',
    'exception','filesystem','forward_list','fstream','functional','future',
    'initializer_list','iostream','istream','iterator','limits','list',
    'locale','map','memory','mutex','numeric','optional','ostream','queue',
    'random','ratio','regex','set','sstream','stack','stdexcept','streambuf',
    'string','string_view','thread','tuple','type_traits','typeinfo',
    'unordered_map','unordered_set','utility','valarray','variant','vector',
    'cassert','cctype','cerrno','cfloat','cinttypes','climits','cmath',
    'csetjmp','csignal','cstdarg','cstddef','cstdint','cstdio','cstdlib',
    'cstring','ctime','cwchar','cwctype','dirent','dlfcn','fcntl','fnmatch',
    'glob','grp','netdb','poll','pthread','pwd','syslog','termios','unistd',
    'windows','tchar','shellapi','winsock2','winbase','windef','winuser',
})
_CPP_VALID_NS = frozenset({
    'boost','gtest','gmock','openssl','curl','zlib','fmt','spdlog',
    'nlohmann','eigen','opencv','tensorflow','torch','pybind11','cereal',
    'catch2','doctest','benchmark','absl','protobuf','grpc','sys','net',
    'netinet','arpa','linux','gl','glm','sdl','sfml','vulkan',
})
_RUBY_LOCAL = [re.compile(p) for p in [
    r'^test_helper$', r'^spec_helper$', r'^application$', r'^boot$',
    r'^test_help$', r'^helper$', r'^initializer$', r'^dispatcher$',
    r'^something_\w+$', r'^\w+_test$', r'^\w+_spec$',
    r'^(public_)?method_defined\?$', r'^:',
]]
_RE_INTERP   = re.compile(r'#\{|`\$\{|\$\{|<%|<\?')
_RE_HTML     = re.compile(r'<[a-zA-Z/][^>]{0,40}>')
_RE_VER      = re.compile(r'^\d+[\.\d]+$')
_RE_OP       = re.compile(r'^[=<>!&|^~+\-*/%]+$')
_RE_URL      = re.compile(r'^https?://')
_RE_NUM      = re.compile(r'^\d+$')
_RE_DOT      = re.compile(r'\.$')

def _is_valid_library(pkg: str, language: str) -> bool:
    p = pkg.strip()
    if not p: return False
    if _RE_INTERP.search(p): return False
    if _RE_HTML.search(p):   return False
    if _RE_URL.match(p):     return False
    if _RE_VER.match(p):     return False
    if _RE_OP.match(p):      return False
    if _RE_NUM.match(p):     return False
    if _RE_DOT.search(p):    return False
    if p.startswith(('./', '../', '/')): return False
    _, ext = os.path.splitext(p)
    if ext.lower() in _LOCAL_EXTENSIONS: return False
    if p.lower() in _NOISE_WORDS: return False
    ll = language.lower()
    if 'ruby' in ll:
        for pat in _RUBY_LOCAL:
            if pat.match(p): return False
    if any(x in ll for x in ('c++','cplusplus','c/','header',' c')):
        clean = p.strip('<>').rstrip('.h').rstrip('.hpp')
        segs  = clean.split('/')
        if len(segs) == 1: return clean.lower() in _CPP_KNOWN_STDLIB
        if segs[0].lower() not in _CPP_VALID_NS: return False
    if ll == 'c' and '++' not in ll and 'header' not in ll:
        clean = p.strip('<>').replace('.h','')
        segs  = clean.split('/')
        if len(segs) == 1: return clean.lower() in _CPP_KNOWN_STDLIB
        if segs[0].lower() not in _CPP_VALID_NS: return False
    if any(x in ll for x in ('javascript','typescript','jsx','tsx')):
        if p.startswith(('.','/')): return False
        if p.startswith('<') or p.endswith('>'): return False
    return True


# ===========================================================================
# Feature extraction helpers (mirror json_to_csv.py logic)
# ===========================================================================

def _normalise_token(s: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in s.lower()).strip('_')


def row_dict_to_token_string(row: Dict[str, str]) -> str:
    """Same transformation used during training."""
    tokens = []
    for col in LANG_COLS:
        val = row.get(col, '').strip()
        if val:
            tokens.append(f'LANG__{_normalise_token(val)}')
    for col in LIB_COLS:
        val = row.get(col, '').strip()
        if val:
            tokens.append(f'LIB__{_normalise_token(val)}')
    return ' '.join(tokens)


def json_profile_to_row(profile: dict) -> Dict[str, str]:
    """
    Convert a SourceSkillsMiner JSON profile into the same row dict
    format that json_to_csv.py would produce, applying the same
    top-5 language / top-7 library selection and noise filtering.
    """
    # ── Languages (ranked by lines) ───────────────────────────────────────
    lu = profile.get('language_usage', {})
    lang_lines: Dict[str, int] = {}
    if lu and not lu.get('error'):
        for lang, stats in lu.get('languages', {}).items():
            if isinstance(stats, (list, tuple)) and stats:
                lang_lines[lang] = int(stats[0])
    ranked_langs = sorted(lang_lines, key=lambda l: -lang_lines[l])[:5]

    # ── Libraries (ranked by occurrence, noise-filtered) ──────────────────
    imp = profile.get('import_scan', {})
    pkg_counts: List[Tuple[str, int]] = []
    if imp and not imp.get('error') and not imp.get('skipped'):
        seen: set = set()
        flat: List[Tuple[str, str, int]] = []
        for lang, ld in imp.get('languages', {}).items():
            for pkg, cnt in ld.get('packages', {}).items():
                if _is_valid_library(pkg, lang):
                    flat.append((lang, pkg, int(cnt)))
        flat.sort(key=lambda x: -x[2])
        for _, pkg, cnt in flat:
            if pkg not in seen:
                seen.add(pkg)
                pkg_counts.append((pkg, cnt))
    ranked_libs = [pkg for pkg, _ in pkg_counts[:7]]

    # ── Build row dict ────────────────────────────────────────────────────
    row: Dict[str, str] = {col: '' for col in LANG_COLS + LIB_COLS}
    row['name'] = profile.get('username', 'unknown')
    for i, lang in enumerate(ranked_langs):
        row[f'top{i+1}lang'] = lang
    for i, lib in enumerate(ranked_libs):
        row[f'top{i+1}lib'] = lib
    return row


# ===========================================================================
# Inference
# ===========================================================================

def load_model(model_path: str) -> dict:
    if not os.path.exists(model_path):
        sys.exit(
            f"ERROR: Model not found at '{model_path}'.\n"
            f"  Run train_classifier.py first to generate the model."
        )
    return joblib.load(model_path)


def classify(token_string: str, bundle: dict, name: str = 'profile') -> dict:
    """
    Run inference and return a structured result dict.

    Returns
    ───────
    {
        "name":           str,
        "prediction":     str,          ← top predicted category
        "confidence_pct": float,        ← confidence in % (0–100)
        "all_probabilities": [          ← sorted highest → lowest
            {"category": str, "probability_pct": float},
            ...
        ]
    }
    """
    pipeline = bundle['pipeline']
    classes  = bundle['classes']

    proba = pipeline.predict_proba([token_string])[0]
    top_idx = int(proba.argmax())

    all_probs = sorted(
        [{'category': cls, 'probability_pct': round(float(p) * 100, 2)}
         for cls, p in zip(classes, proba)],
        key=lambda x: -x['probability_pct']
    )

    return {
        'name':             name,
        'prediction':       classes[top_idx],
        'confidence_pct':   round(float(proba[top_idx]) * 100, 2),
        'all_probabilities': all_probs,
        'model_cv_accuracy': round(bundle.get('cv_mean', 0) * 100, 2),
    }


def print_result(result: dict) -> None:
    bar_width = 30
    print(f"\n{'='*55}")
    print(f"  Profile   : {result['name']}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence_pct']:.1f}%")
    print(f"  Model CV accuracy: {result['model_cv_accuracy']:.1f}%")
    print(f"\n  Full probability breakdown:")
    print(f"  {'Category':<35s}  {'Prob':>7s}  Bar")
    print(f"  {'-'*35}  {'-'*7}  {'-'*bar_width}")
    for entry in result['all_probabilities']:
        cat  = entry['category']
        prob = entry['probability_pct']
        bar  = '█' * int(prob / 100 * bar_width)
        print(f"  {cat:<35s}  {prob:6.1f}%  {bar}")
    print(f"{'='*55}\n")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='Classify a developer profile using the saved Naive Bayes model.'
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--json', metavar='FILE',
                     help='Path to a SourceSkillsMiner JSON profile file')
    src.add_argument('--csv',  metavar='FILE',
                     help='Path to a single-row CSV (same schema as features.csv)')
    src.add_argument('--lang', nargs='+', metavar='LANG',
                     help='Space-separated list of languages (use with --lib)')

    p.add_argument('--lib',    nargs='*', default=[], metavar='LIB',
                   help='Space-separated list of libraries (used with --lang)')
    p.add_argument('--model',  default=DEFAULT_MODEL,
                   help=f'Path to saved model. Default: {DEFAULT_MODEL}')
    p.add_argument('--output', default='',
                   help='Optional path to save the result as JSON')

    args = p.parse_args()

    bundle = load_model(args.model)

    # ── Build token string from the chosen input mode ─────────────────────
    if args.json:
        with open(args.json, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        row    = json_profile_to_row(profile)
        name   = profile.get('username', os.path.splitext(os.path.basename(args.json))[0])
        tokens = row_dict_to_token_string(row)

    elif args.csv:
        with open(args.csv, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        if not rows:
            sys.exit("ERROR: CSV is empty.")
        row    = rows[0]
        name   = row.get('name', 'profile')
        tokens = row_dict_to_token_string(row)

    else:  # --lang / --lib
        row = {col: '' for col in LANG_COLS + LIB_COLS}
        for i, lang in enumerate(args.lang[:5]):
            row[f'top{i+1}lang'] = lang
        for i, lib in enumerate(args.lib[:7]):
            row[f'top{i+1}lib'] = lib
        name   = 'manual_input'
        tokens = row_dict_to_token_string(row)

    if not tokens.strip():
        sys.exit("ERROR: No valid language or library tokens found in the input.")

    result = classify(tokens, bundle, name=name)
    print_result(result)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Result saved → {os.path.abspath(args.output)}")


if __name__ == '__main__':
    main()

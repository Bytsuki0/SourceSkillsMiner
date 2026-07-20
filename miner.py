#!/usr/bin/env python3
"""
miner.py — the friendly front door to SourceSkillsMiner.

Two ways to use it:

    python miner.py             interactive shell: type a GitHub username to
                                analyze it, `token` to set your token, `help`
                                for everything else.

    python miner.py octocat     one-shot CLI: analyze a single user and exit.
                                Flags: --token, --fast, --pdf, --no-classify,
                                --repo-limit N, --ascii, --no-color.

The shell wraps the existing pipeline unchanged:
    ssm.ScoringFacade                    → mines + scores  → json/<user>.json
    Bayers_Classifier/classify_profile   → developer class → json/<user>.classification.json
    backend/pdf_generator.build_pdf      → PDF report      → pdf/<user>.pdf

A GitHub personal access token is required (the pipeline speaks GraphQL, which
GitHub only serves authenticated). The shell looks for one in --token, the
GITHUB_TOKEN / GH_TOKEN / SSM_TOKEN env vars, config_main.ini, then config.ini,
and otherwise prompts for it — offering to remember it in config_main.ini
(which is gitignored).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from getpass import getpass
from typing import Dict, List, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(ROOT, "json")
PDF_DIR = os.path.join(ROOT, "pdf")
BACKEND_DIR = os.path.join(ROOT, "backend")
CLASS_SCRIPT = os.path.join(ROOT, "Bayers_Classifier", "classify_profile.py")
MODEL_PATH = os.path.join(ROOT, "Bayers_Classifier", "models", "developer_classifier.joblib")
CONFIG_MAIN = os.path.join(ROOT, "config_main.ini")
CONFIG_LOCAL = os.path.join(ROOT, "config.ini")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

USERNAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")
TOKEN_PREFIXES = ("ghp_", "github_pat_", "gho_", "ghu_", "ghs_")
PLACEHOLDER_MARKERS = ("your_", "xxxx", "ghp_...")

PANEL_W = 66  # inner width of result panels


# ── Terminal setup ───────────────────────────────────────────────────────────

def _reconfigure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _enable_ansi() -> bool:
    """Turn on VT escape processing on Windows consoles. True if ANSI is usable."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class Style:
    """ANSI palette; every attribute collapses to '' when colors are off."""

    NAMES = {
        "reset": "\x1b[0m", "bold": "\x1b[1m", "dim": "\x1b[2m",
        "red": "\x1b[31m", "green": "\x1b[32m", "yellow": "\x1b[33m",
        "blue": "\x1b[34m", "magenta": "\x1b[35m", "cyan": "\x1b[36m",
        "gray": "\x1b[90m", "white": "\x1b[97m",
    }

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        for name, code in self.NAMES.items():
            setattr(self, name, code if enabled else "")

    def wrap(self, code: str, text: str) -> str:
        return f"{code}{text}{self.reset}" if self.enabled else text


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def term_width(default: int = 100) -> int:
    return shutil.get_terminal_size((default, 30)).columns


# ── Small formatting helpers ─────────────────────────────────────────────────

def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s"


def mask_token(token: str) -> str:
    if len(token) <= 12:
        return token[:3] + "…"
    return f"{token[:7]}…{token[-4:]}"


def score_color(style: Style, value: float) -> str:
    if value >= 0.66:
        return style.green
    if value >= 0.33:
        return style.yellow
    return style.red


def score_bar(style: Style, value: Optional[float], width: int, unicode_ok: bool) -> str:
    if not isinstance(value, (int, float)):
        return style.wrap(style.gray, "no data".ljust(width))
    v = max(0.0, min(1.0, float(value)))
    filled = int(round(v * width))
    full_ch, empty_ch = ("█", "░") if unicode_ok else ("#", "-")
    return (
        score_color(style, v) + full_ch * filled
        + style.gray + empty_ch * (width - filled) + style.reset
    )


# ── Console UI (permanent lines + one live status line) ──────────────────────

class Ui:
    """
    All terminal output funnels through here so a transient status line can
    coexist with permanent lines. A tiny daemon thread animates the spinner
    while the (synchronous) pipeline blocks on network calls.
    """

    SPIN_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    SPIN_ASCII = "|/-\\"

    def __init__(self, style: Style, unicode_ok: bool) -> None:
        self.out = sys.stdout          # captured before any redirect_stdout
        self.style = style
        self.unicode_ok = unicode_ok
        self.live = self.out.isatty() and style.enabled
        self.lock = threading.RLock()
        self._status = ""
        self._status_shown = False
        self._spin_idx = 0
        self._spin_stop: Optional[threading.Event] = None
        self._spin_thread: Optional[threading.Thread] = None
        self.ok_mark = "✔" if unicode_ok else "+"
        self.fail_mark = "✘" if unicode_ok else "x"
        self.bullet = "·" if unicode_ok else "-"

    # -- primitives ----------------------------------------------------------

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _clear_status(self) -> None:
        if self._status_shown:
            self._write("\r\x1b[2K")
            self._status_shown = False

    def _draw_status(self) -> None:
        if not self.live or not self._status:
            return
        frames = self.SPIN_UNICODE if self.unicode_ok else self.SPIN_ASCII
        frame = frames[self._spin_idx % len(frames)]
        line = f"  {self.style.cyan}{frame}{self.style.reset} {self._status}"
        max_w = term_width() - 1
        if visible_len(line) > max_w:
            # crude but safe truncation: strip colors, cut, re-dim
            plain = _ANSI_RE.sub("", line)[: max_w - 1] + "…"
            line = plain
        self._write("\r\x1b[2K" + line)
        self._status_shown = True

    # -- public API ----------------------------------------------------------

    def line(self, text: str = "") -> None:
        """Print a permanent line above the status line."""
        with self.lock:
            self._clear_status()
            self._write(text + "\n")
            self._draw_status()

    def set_status(self, text: str) -> None:
        with self.lock:
            self._status = text
            self._draw_status()

    def clear_status(self) -> None:
        with self.lock:
            self._status = ""
            self._clear_status()

    def start_spinner(self) -> None:
        if not self.live or self._spin_thread:
            return
        self._spin_stop = threading.Event()

        def _tick() -> None:
            while not self._spin_stop.wait(0.09):
                with self.lock:
                    self._spin_idx += 1
                    self._draw_status()

        self._spin_thread = threading.Thread(target=_tick, daemon=True)
        self._spin_thread.start()

    def stop_spinner(self) -> None:
        if self._spin_stop:
            self._spin_stop.set()
        if self._spin_thread:
            self._spin_thread.join(timeout=1)
        self._spin_stop = None
        self._spin_thread = None
        self.clear_status()

    def prompt(self, text: str) -> str:
        with self.lock:
            self._clear_status()
            self._write(text)
        return input()


class _StdoutInterceptor(io.TextIOBase):
    """
    Receives everything the pipeline print()s directly (client errors, NLTK
    downloads, save messages) and re-emits each line as a dim log line above
    the status display. Full lines are kept in ``log`` for the `log` command.
    """

    def __init__(self, ui: Ui, log: List[str]) -> None:
        self.ui = ui
        self.log = log
        self._buf = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        self._buf += s
        while True:
            cut = -1
            for sep in ("\n", "\r"):
                idx = self._buf.find(sep)
                if idx != -1 and (cut == -1 or idx < cut):
                    cut = idx
            if cut == -1:
                break
            line, self._buf = self._buf[:cut], self._buf[cut + 1:]
            self._emit(line)
        return len(s)

    def _emit(self, line: str) -> None:
        line = line.rstrip()
        if not line.strip():
            return
        self.log.append(line)
        style = self.ui.style
        shown = line if len(line) <= 110 else line[:109] + "…"
        self.ui.line(f"    {style.gray}{shown}{style.reset}")

    def flush(self) -> None:  # pragma: no cover - interface completeness
        pass

    def isatty(self) -> bool:
        return False


# ── Progress observer (plugs into the ssm Observer chain) ────────────────────

STAGE_ORDER = ["oss", "status", "adaptability", "sentiment", "commitment",
               "language", "imports", "avatar"]
STAGE_LABELS = {
    "oss": "OSS engagement",
    "status": "Repository status",
    "adaptability": "Adaptability",
    "sentiment": "Sentiment",
    "commitment": "Commitment",
    "language": "Language usage",
    "imports": "Import scan",
    "avatar": "Profile picture",
}
TRANSIENT_STAGES = {"supplementary", "done"}


class ShellProgress:
    """Duck-typed ssm Observer: turns ProgressEvents into the live display."""

    def __init__(self, ui: Ui) -> None:
        self.ui = ui
        self.current: Optional[str] = None
        self.stage_start = 0.0
        self.last_msg = ""

    def update(self, event) -> None:  # event: ssm.core.events.ProgressEvent
        stage = (event.stage or "").strip()
        message = (event.message or "").strip()

        if stage in STAGE_LABELS and stage != self.current:
            self._finish_current()
            self.current = stage
            self.stage_start = time.monotonic()
            self.last_msg = ""

        if message:
            self.last_msg = message.splitlines()[-1]

        self._render()

    def _render(self) -> None:
        style = self.ui.style
        if self.current is None:
            if self.last_msg:
                self.ui.set_status(style.wrap(style.gray, self.last_msg))
            return
        label = STAGE_LABELS[self.current]
        pos = STAGE_ORDER.index(self.current) + 1
        text = (
            f"{style.bold}{label}{style.reset} "
            f"{style.gray}[{pos}/{len(STAGE_ORDER)}]{style.reset}"
        )
        if self.last_msg:
            text += f"  {style.gray}{self.bullet_msg()}{style.reset}"
        self.ui.set_status(text)

    def bullet_msg(self) -> str:
        return f"{self.ui.bullet} {self.last_msg}"

    def _finish_current(self) -> None:
        if self.current is None:
            return
        style = self.ui.style
        elapsed = fmt_elapsed(time.monotonic() - self.stage_start)
        label = STAGE_LABELS[self.current]
        self.ui.line(
            f"  {style.green}{self.ui.ok_mark}{style.reset} "
            f"{label:<20}{style.gray}{elapsed}{style.reset}"
        )

    def finish(self) -> None:
        self._finish_current()
        self.current = None
        self.ui.clear_status()


# ── Token handling ───────────────────────────────────────────────────────────

def _looks_real(token: Optional[str]) -> bool:
    if not token:
        return False
    lowered = token.strip().lower()
    return bool(lowered) and not any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _token_from_ini(path: str) -> Optional[str]:
    import configparser

    parser = configparser.ConfigParser()
    try:
        parser.read(path)
    except Exception:
        return None
    token = parser.get("github", "token", fallback=None)
    return token.strip() if token else None


def discover_token() -> Tuple[Optional[str], str]:
    """Return (token, human-readable source)."""
    for env in ("GITHUB_TOKEN", "GH_TOKEN", "SSM_TOKEN"):
        value = os.environ.get(env, "").strip()
        if _looks_real(value):
            return value, f"env {env}"
    for path, label in ((CONFIG_MAIN, "config_main.ini"), (CONFIG_LOCAL, "config.ini")):
        if os.path.exists(path):
            value = _token_from_ini(path)
            if _looks_real(value):
                return value, label
    return None, ""


def save_token(token: str) -> str:
    """Persist token to config_main.ini (gitignored), preserving other keys."""
    import configparser

    parser = configparser.ConfigParser()
    if os.path.exists(CONFIG_MAIN):
        parser.read(CONFIG_MAIN)
    if "github" not in parser:
        parser["github"] = {}
    parser["github"]["token"] = token
    with open(CONFIG_MAIN, "w", encoding="utf-8") as fh:
        parser.write(fh)
    return CONFIG_MAIN


def validate_token(token: str) -> Tuple[Optional[Dict], str]:
    """
    Ask GitHub who this token belongs to. Returns (info, client_chatter):
    info = {login, remaining, limit} on success, None otherwise.
    """
    from ssm.core.client import GitHubClient

    query = "query { viewer { login } rateLimit { remaining limit resetAt } }"
    chatter = io.StringIO()
    with redirect_stdout(chatter):
        data = GitHubClient(token, timeout=15).graphql(query, {})
    if data and data.get("viewer"):
        rate = data.get("rateLimit") or {}
        return (
            {
                "login": data["viewer"].get("login", "?"),
                "remaining": rate.get("remaining"),
                "limit": rate.get("limit"),
            },
            chatter.getvalue(),
        )
    return None, chatter.getvalue()


# ── Session state ────────────────────────────────────────────────────────────

@dataclass
class Session:
    token: Optional[str] = None
    token_source: str = ""
    viewer: Optional[Dict] = None
    fast: bool = False
    repo_limit: int = 25
    classify: bool = True
    last_log: List[str] = field(default_factory=list)


# ── Pipeline steps ───────────────────────────────────────────────────────────

def run_analysis(ui: Ui, session: Session, username: str) -> Optional[Tuple[Dict, str]]:
    """Mine + score one user. Returns (profile_dict, saved_json_path) or None."""
    style = ui.style

    ui.set_status(style.wrap(style.gray, "Loading analysis engine…"))
    ui.start_spinner()
    try:
        from ssm.core.config import Config
        from ssm.core.serialization import save_score_to_json
        from ssm.scoring.facade import ScoringFacade
    except Exception as exc:
        ui.stop_spinner()
        ui.line(f"  {style.red}{ui.fail_mark} Could not load the analysis engine:"
                f"{style.reset} {exc}")
        ui.line(f"  {style.gray}Try: pip install -r requirements.txt{style.reset}")
        return None
    # spinner keeps running into the analysis itself

    progress = ShellProgress(ui)
    interceptor = _StdoutInterceptor(ui, session.last_log)
    facade = ScoringFacade(Config(username=username, token=session.token),
                           observers=[progress])
    started = time.monotonic()
    try:
        with redirect_stdout(interceptor):
            profile = facade.analyze(
                username=username,
                token=session.token,
                repo_limit=session.repo_limit,
                include_import_scan=not session.fast,
            )
            saved_path = save_score_to_json(profile, username=username, base_dir=ROOT)
    except KeyboardInterrupt:
        ui.stop_spinner()
        ui.line(f"  {style.yellow}Aborted.{style.reset}")
        return None
    except Exception as exc:
        ui.stop_spinner()
        import traceback

        session.last_log.extend(traceback.format_exc().splitlines())
        ui.line(f"  {style.red}{ui.fail_mark} Analysis failed:{style.reset} {exc}")
        ui.line(f"  {style.gray}(type `log` to see the full output){style.reset}")
        return None

    progress.finish()
    ui.stop_spinner()
    ui.line(
        f"  {style.gray}Mined + scored in "
        f"{fmt_elapsed(time.monotonic() - started)}{style.reset}"
    )
    return profile, saved_path


def run_classifier(ui: Ui, session: Session, json_path: str,
                   username: str) -> Optional[Dict]:
    """Classify a saved profile via the Naive Bayes model. Offline + fast."""
    style = ui.style
    if not (os.path.exists(CLASS_SCRIPT) and os.path.exists(MODEL_PATH)):
        ui.line(f"  {style.gray}Classifier model not found — skipping.{style.reset}")
        return None

    out_path = os.path.join(JSON_DIR, f"{username}.classification.json")
    ui.set_status(style.wrap(style.gray, "Classifying developer profile…"))
    ui.start_spinner()
    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        proc = subprocess.run(
            [sys.executable, CLASS_SCRIPT,
             "--json", json_path, "--model", MODEL_PATH, "--output", out_path],
            capture_output=True, text=True, timeout=180, cwd=ROOT, env=env,
        )
    except Exception as exc:
        ui.stop_spinner()
        ui.line(f"  {style.yellow}Classifier could not run: {exc}{style.reset}")
        return None
    finally:
        ui.stop_spinner()

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    detail = (proc.stderr or proc.stdout or "").strip()
    if detail:
        session.last_log.extend(detail.splitlines())
    ui.line(f"  {style.yellow}Classifier produced no output"
            f"{' (see `log`)' if detail else ''}.{style.reset}")
    return None


def generate_pdf(ui: Ui, username: str, profile: Dict,
                 classification: Optional[Dict]) -> Optional[str]:
    style = ui.style
    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)
    try:
        from pdf_generator import build_pdf
    except ImportError as exc:
        ui.line(f"  {style.yellow}PDF needs ReportLab "
                f"(pip install reportlab): {exc}{style.reset}")
        return None

    ui.set_status(style.wrap(style.gray, "Rendering PDF report…"))
    ui.start_spinner()
    try:
        pdf_bytes = build_pdf(username, profile, classification or {})
    except Exception as exc:
        ui.stop_spinner()
        ui.line(f"  {style.red}PDF generation failed: {exc}{style.reset}")
        return None
    finally:
        ui.stop_spinner()

    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(PDF_DIR, f"{username}.pdf")
    with open(pdf_path, "wb") as fh:
        fh.write(pdf_bytes)
    return pdf_path


# ── Result rendering ─────────────────────────────────────────────────────────

def _panel_line(ui: Ui, content: str = "") -> None:
    style = ui.style
    v, h = ("│", "─") if ui.unicode_ok else ("|", "-")
    pad = PANEL_W - 2 - visible_len(content)
    ui.line(f"  {style.gray}{v}{style.reset} {content}{' ' * max(0, pad - 1)}"
            f"{style.gray}{v}{style.reset}")


def _panel_rule(ui: Ui, kind: str, title: str = "") -> None:
    style = ui.style
    if ui.unicode_ok:
        left, right = {"top": ("╭", "╮"), "mid": ("├", "┤"), "bot": ("╰", "╯")}[kind]
        h = "─"
    else:
        left, right, h = "+", "+", "-"
    if title:
        label = f" {title} "
        fill = PANEL_W - 2 - visible_len(label) - 2
        body = f"{h * 2}{label}{h * max(0, fill)}"
    else:
        body = h * (PANEL_W - 2)
    ui.line(f"  {style.gray}{left}{body}{right}{style.reset}")


def render_result(ui: Ui, profile: Dict, classification: Optional[Dict],
                  saved: Optional[List[str]] = None) -> None:
    style = ui.style
    username = profile.get("username", "?")
    final = profile.get("final_score")
    areas = profile.get("areas") or {}

    ui.line()
    _panel_rule(ui, "top", f"{style.bold}{style.cyan}{username}{style.reset}")

    # Final score headline
    if isinstance(final, (int, float)):
        color = score_color(style, final)
        headline = (
            f"{style.bold}Final score{style.reset}   "
            f"{color}{style.bold}{final:.2f}{style.reset} / 1.00   "
            f"{score_bar(style, final, 28, ui.unicode_ok)}"
        )
    else:
        headline = f"{style.bold}Final score{style.reset}   {style.gray}n/a{style.reset}"
    _panel_line(ui, headline)
    _panel_rule(ui, "mid")

    # Area bars
    for area in ("OSS", "Status", "Adaptability", "Sentiment", "Commitment"):
        info = areas.get(area) or {}
        value = info.get("score")
        val_txt = f"{value:.2f}" if isinstance(value, (int, float)) else " n/a"
        _panel_line(
            ui,
            f"{area:<13}{score_bar(style, value, 30, ui.unicode_ok)}  "
            f"{score_color(style, value if isinstance(value, (int, float)) else 0)}"
            f"{val_txt}{style.reset}",
        )

    commitment = (areas.get("Commitment") or {}).get("details") or {}
    if isinstance(commitment.get("total_points"), int):
        _panel_line(
            ui,
            f"{style.gray}{'':<13}{commitment['total_points']}/"
            f"{commitment.get('max_points', 4)} commitment criteria met{style.reset}",
        )

    # Languages
    lang_usage = profile.get("language_usage") or {}
    top_langs = lang_usage.get("top_5_languages") or []
    if top_langs:
        total_lines = lang_usage.get("total_lines") or sum(
            entry.get("lines", 0) for entry in top_langs
        ) or 1
        parts = []
        for entry in top_langs[:3]:
            pct = 100.0 * entry.get("lines", 0) / max(1, total_lines)
            parts.append(f"{style.cyan}{entry.get('language', '?')}{style.reset}"
                         f" {pct:.0f}%")
        _panel_rule(ui, "mid")
        _panel_line(ui, f"{style.bold}Languages{style.reset}    "
                        + f" {style.gray}{ui.bullet}{style.reset} ".join(parts))

    # Classification
    if classification and classification.get("prediction"):
        conf = classification.get("confidence_pct")
        conf_txt = f" ({conf:.1f}%)" if isinstance(conf, (int, float)) else ""
        _panel_line(ui, f"{style.bold}Profile type{style.reset} "
                        f"{style.magenta}{style.bold}"
                        f"{classification['prediction']}{style.reset}{conf_txt}")
        runners = [
            f"{entry.get('category')} {entry.get('probability_pct', 0):.1f}%"
            for entry in (classification.get("all_probabilities") or [])[1:3]
        ]
        if runners:
            _panel_line(ui, f"{style.gray}{'':<13}then "
                            + f" {ui.bullet} ".join(runners) + style.reset)
    elif classification is None:
        pass  # classifier skipped; keep the panel clean

    _panel_rule(ui, "bot")

    for path in saved or []:
        rel = os.path.relpath(path, ROOT)
        ui.line(f"  {style.gray}saved{style.reset}  {rel}")
    ui.line()


# ── Saved-profile helpers ────────────────────────────────────────────────────

def load_saved(username: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    profile_path = os.path.join(JSON_DIR, f"{username}.json")
    if not os.path.exists(profile_path):
        return None, None
    with open(profile_path, "r", encoding="utf-8") as fh:
        profile = json.load(fh)
    classification = None
    cls_path = os.path.join(JSON_DIR, f"{username}.classification.json")
    if os.path.exists(cls_path):
        try:
            with open(cls_path, "r", encoding="utf-8") as fh:
                classification = json.load(fh)
        except Exception:
            classification = None
    return profile, classification


def list_profiles(ui: Ui) -> None:
    style = ui.style
    if not os.path.isdir(JSON_DIR):
        ui.line(f"  {style.gray}No profiles analyzed yet.{style.reset}")
        return
    entries = []
    for name in sorted(os.listdir(JSON_DIR)):
        if not name.endswith(".json") or name.endswith(".classification.json"):
            continue
        path = os.path.join(JSON_DIR, name)
        username = name[:-5]
        score_txt, cls_txt = "?", ""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            final = data.get("final_score")
            if isinstance(final, (int, float)):
                score_txt = f"{final:.2f}"
        except Exception:
            pass
        cls_path = os.path.join(JSON_DIR, f"{username}.classification.json")
        if os.path.exists(cls_path):
            try:
                with open(cls_path, "r", encoding="utf-8") as fh:
                    cls_txt = json.load(fh).get("prediction", "")
            except Exception:
                pass
        modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
        entries.append((username, score_txt, cls_txt, modified))

    if not entries:
        ui.line(f"  {style.gray}No profiles analyzed yet — type a username to start.{style.reset}")
        return

    ui.line()
    ui.line(f"  {style.bold}{'PROFILE':<22}{'SCORE':<8}{'TYPE':<26}MODIFIED{style.reset}")
    for username, score_txt, cls_txt, modified in entries:
        ui.line(f"  {style.cyan}{username:<22}{style.reset}{score_txt:<8}"
                f"{cls_txt:<26}{style.gray}{modified}{style.reset}")
    ui.line()


# ── Interactive flows ────────────────────────────────────────────────────────

def token_flow(ui: Ui, session: Session, provided: Optional[str] = None,
               interactive: bool = True) -> bool:
    """Acquire + validate a token. Returns True when session has a valid one."""
    style = ui.style
    token = (provided or "").strip()

    if not token:
        if not interactive or not sys.stdin.isatty():
            ui.line(f"  {style.red}No GitHub token found.{style.reset} Pass --token, "
                    f"set GITHUB_TOKEN, or add it to config_main.ini.")
            return False
        ui.line()
        ui.line(f"  {style.bold}GitHub token needed{style.reset} — the miner talks to "
                f"GitHub's GraphQL API, which requires auth.")
        ui.line(f"  {style.gray}Create one at https://github.com/settings/tokens "
                f"(read-only public access is enough).{style.reset}")
        try:
            token = getpass("  Paste token (input hidden, Enter to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.line()
            return False
        if not token:
            ui.line(f"  {style.gray}Cancelled.{style.reset}")
            return False

    ui.set_status(style.wrap(style.gray, "Checking token with GitHub…"))
    ui.start_spinner()
    info, chatter = validate_token(token)
    ui.stop_spinner()

    if not info:
        hint = "network problem?" if "Network error" in chatter else "invalid or expired?"
        ui.line(f"  {style.red}{ui.fail_mark} GitHub rejected the token{style.reset} "
                f"{style.gray}({hint}){style.reset}")
        for line in chatter.strip().splitlines()[:2]:
            ui.line(f"    {style.gray}{line[:110]}{style.reset}")
        return False

    session.token = token
    session.viewer = info
    session.token_source = session.token_source or "entered now"
    remaining = info.get("remaining")
    limit = info.get("limit")
    rate = f" · {remaining:,}/{limit:,} API requests left" if remaining is not None else ""
    ui.line(f"  {style.green}{ui.ok_mark} Authenticated as "
            f"{style.bold}{info['login']}{style.reset}{style.gray}{rate}{style.reset}")

    if interactive and sys.stdin.isatty():
        try:
            answer = ui.prompt(f"  Remember this token in config_main.ini "
                               f"{style.gray}(gitignored){style.reset}? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("", "y", "yes"):
            path = save_token(token)
            session.token_source = "config_main.ini"
            ui.line(f"  {style.gray}Saved to {os.path.basename(path)} — future runs "
                    f"won't ask again.{style.reset}")
    return True


def ensure_token(ui: Ui, session: Session, interactive: bool) -> bool:
    if session.viewer:
        return True
    if session.token:
        if token_flow(ui, session, provided=session.token, interactive=False):
            return True
        session.token = None  # discovered token was bad; fall through to prompt
    return token_flow(ui, session, interactive=interactive)


def analyze_flow(ui: Ui, session: Session, username: str,
                 interactive: bool, make_pdf: bool = False) -> bool:
    style = ui.style
    if not ensure_token(ui, session, interactive):
        return False

    session.last_log.clear()
    mode = "quick (no import scan)" if session.fast else "full"
    ui.line()
    ui.line(f"  {style.bold}Mining {style.cyan}{username}{style.reset}"
            f"{style.gray}  {ui.bullet} {mode} analysis {ui.bullet} "
            f"repo limit {session.repo_limit}{style.reset}")
    ui.line(f"  {style.gray}This calls the GitHub API a lot — a big profile "
            f"can take several minutes.{style.reset}")
    ui.line()

    outcome = run_analysis(ui, session, username)
    if outcome is None:
        return False
    profile, json_path = outcome

    classification = None
    if session.classify:
        classification = run_classifier(ui, session, json_path, username)

    saved = [json_path]
    if classification:
        saved.append(os.path.join(JSON_DIR, f"{username}.classification.json"))

    render_result(ui, profile, classification, saved=saved)

    want_pdf = make_pdf
    if not want_pdf and interactive and sys.stdin.isatty():
        try:
            answer = ui.prompt(f"  Export a PDF report too? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        want_pdf = answer in ("y", "yes")
    if want_pdf:
        pdf_path = generate_pdf(ui, username, profile, classification)
        if pdf_path:
            ui.line(f"  {style.green}{ui.ok_mark}{style.reset} PDF "
                    f"{style.gray}{os.path.relpath(pdf_path, ROOT)}{style.reset}")
            ui.line()
    return True


def pdf_flow(ui: Ui, session: Session, username: str) -> None:
    style = ui.style
    profile, classification = load_saved(username)
    if profile is None:
        ui.line(f"  {style.yellow}No saved analysis for '{username}' — "
                f"type the username first to analyze it.{style.reset}")
        return
    if classification is None and session.classify:
        classification = run_classifier(
            ui, session, os.path.join(JSON_DIR, f"{username}.json"), username)
    pdf_path = generate_pdf(ui, username, profile, classification)
    if pdf_path:
        ui.line(f"  {style.green}{ui.ok_mark}{style.reset} PDF saved "
                f"{style.gray}{os.path.relpath(pdf_path, ROOT)}{style.reset}")


# ── Shell chrome ─────────────────────────────────────────────────────────────

def banner(ui: Ui, session: Session) -> None:
    style = ui.style
    ui.line()
    _panel_rule(ui, "top")
    _panel_line(ui, f"{style.bold}{style.cyan}SOURCE SKILLS MINER{style.reset}"
                    f"{style.gray}  — GitHub contributor profiling{style.reset}")
    _panel_line(ui, f"{style.gray}mine {ui.bullet} score {ui.bullet} classify "
                    f"{ui.bullet} report{style.reset}")
    _panel_rule(ui, "bot")

    if session.viewer:
        remaining = session.viewer.get("remaining")
        rate = f" · {remaining:,} requests left" if remaining is not None else ""
        ui.line(f"  token {style.green}{ui.ok_mark}{style.reset} "
                f"{style.bold}{session.viewer['login']}{style.reset}"
                f"{style.gray} ({session.token_source}){rate}{style.reset}")
    elif session.token:
        ui.line(f"  token {style.yellow}?{style.reset} found in "
                f"{session.token_source} {style.gray}(couldn't verify — "
                f"offline?){style.reset}")
    else:
        ui.line(f"  token {style.yellow}none{style.reset} "
                f"{style.gray}— type {style.reset}token{style.gray} or just paste a "
                f"ghp_… token; you'll also be asked on first run{style.reset}")

    ui.line()
    ui.line(f"  Type a {style.bold}GitHub username{style.reset} to analyze it, "
            f"or {style.cyan}help{style.reset} for commands.")
    ui.line()


def show_help(ui: Ui, session: Session) -> None:
    style = ui.style
    fast_state = "on" if session.fast else "off"

    def row(cmd: str, desc: str) -> None:
        ui.line(f"    {style.cyan}{cmd:<16}{style.reset}{desc}")

    ui.line()
    ui.line(f"  {style.bold}Commands{style.reset}")
    row("<username>", "analyze a GitHub profile (e.g.  octocat)")
    row("token", "set or replace your GitHub token")
    row("list", "profiles already analyzed")
    row("show <name>", "re-display a saved profile")
    row("pdf <name>", "export a PDF report for a saved profile")
    row("fast", f"toggle quick mode, skips import scan  [now: {fast_state}]")
    row("limit <n>", f"max repos scanned for sentiment  [now: {session.repo_limit}]")
    row("log", "raw pipeline output of the last run")
    row("clear", "clear the screen")
    row("exit", "leave the shell")
    ui.line()
    ui.line(f"  {style.gray}Tips: pasting a ghp_… token at the prompt sets it "
            f"directly; if a username{style.reset}")
    ui.line(f"  {style.gray}collides with a command (e.g. a user named 'log'), "
            f"use  run <name>.{style.reset}")
    ui.line()


def shell(ui: Ui, session: Session) -> None:
    style = ui.style
    banner(ui, session)

    while True:
        try:
            raw = ui.prompt(f"{style.cyan}{style.bold}ssm{style.reset} "
                            f"{style.cyan}{'❯' if ui.unicode_ok else '>'}{style.reset} ")
        except EOFError:
            ui.line()
            break
        except KeyboardInterrupt:
            ui.line(f"  {style.gray}^C — type exit to quit{style.reset}")
            continue

        # Piped input may carry a UTF-8 BOM (as U+FEFF, or as ï»¿ mojibake).
        raw = raw.lstrip("\ufeff\xef\xbb\xbf").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("exit", "quit", "q"):
            break
        elif cmd in ("help", "h", "?"):
            show_help(ui, session)
        elif cmd in ("clear", "cls"):
            os.system("cls" if os.name == "nt" else "clear")
            banner(ui, session)
        elif cmd == "token":
            session.viewer = None
            session.token_source = ""
            token_flow(ui, session)
        elif cmd in ("list", "ls"):
            list_profiles(ui)
        elif cmd == "show" and args:
            profile, classification = load_saved(args[0])
            if profile is None:
                ui.line(f"  {style.yellow}No saved analysis for "
                        f"'{args[0]}'.{style.reset}")
            else:
                render_result(ui, profile, classification)
        elif cmd == "pdf" and args:
            pdf_flow(ui, session, args[0])
        elif cmd == "fast":
            session.fast = not session.fast
            state = "ON — import scan skipped" if session.fast else "OFF — full analysis"
            ui.line(f"  quick mode {style.bold}{state}{style.reset}")
        elif cmd == "limit" and args:
            try:
                session.repo_limit = max(1, min(500, int(args[0])))
                ui.line(f"  repo limit set to {style.bold}"
                        f"{session.repo_limit}{style.reset}")
            except ValueError:
                ui.line(f"  {style.yellow}usage: limit <number>{style.reset}")
        elif cmd == "log":
            if not session.last_log:
                ui.line(f"  {style.gray}(no output captured yet){style.reset}")
            for line in session.last_log[-200:]:
                ui.line(f"  {style.gray}{line}{style.reset}")
        elif cmd in ("show", "pdf", "limit"):
            ui.line(f"  {style.yellow}usage: {cmd} <{'n' if cmd == 'limit' else 'username'}>"
                    f"{style.reset}")
        elif cmd in ("run", "mine", "analyze") and args:
            if USERNAME_RE.fullmatch(args[0]):
                analyze_flow(ui, session, args[0], interactive=True)
            else:
                ui.line(f"  {style.yellow}'{args[0]}' is not a valid GitHub "
                        f"username.{style.reset}")
        elif raw.startswith(TOKEN_PREFIXES):
            ui.line(f"  {style.gray}That looks like a token — checking it…"
                    f"{style.reset}")
            session.viewer = None
            session.token_source = ""
            token_flow(ui, session, provided=raw)
        elif USERNAME_RE.fullmatch(raw):
            analyze_flow(ui, session, raw, interactive=True)
        else:
            ui.line(f"  {style.yellow}'{raw}' isn't a command or a valid GitHub "
                    f"username — type help.{style.reset}")

    ui.line(f"  {style.gray}bye!{style.reset}")


# ── Entry point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miner",
        description="SourceSkillsMiner — interactive shell / one-shot CLI for "
                    "GitHub contributor profiling.",
        epilog="Run with no arguments to open the interactive shell.",
    )
    parser.add_argument("username", nargs="?",
                        help="GitHub username to analyze immediately (one-shot mode)")
    parser.add_argument("--token", "-t", help="GitHub personal access token")
    parser.add_argument("--fast", action="store_true",
                        help="skip the import/package scan (much faster)")
    parser.add_argument("--repo-limit", type=int, default=25, metavar="N",
                        help="max repositories for the sentiment scan (default: 25)")
    parser.add_argument("--pdf", action="store_true",
                        help="also export a PDF report (one-shot mode)")
    parser.add_argument("--no-classify", action="store_true",
                        help="skip the developer-type classifier")
    parser.add_argument("--ascii", action="store_true",
                        help="plain ASCII output (no unicode glyphs)")
    parser.add_argument("--no-color", action="store_true",
                        help="disable colors")
    return parser


def main() -> None:
    _reconfigure_stdio()
    args = build_parser().parse_args()

    ansi_ok = _enable_ansi()
    color = ansi_ok and not args.no_color and not os.environ.get("NO_COLOR") \
        and sys.stdout.isatty()
    unicode_ok = not args.ascii and (sys.stdout.isatty() or _can_encode("█─╭❯⠋"))

    style = Style(color)
    ui = Ui(style, unicode_ok)
    os.makedirs(JSON_DIR, exist_ok=True)

    session = Session(fast=args.fast, repo_limit=max(1, args.repo_limit),
                      classify=not args.no_classify)

    if args.token and _looks_real(args.token):
        session.token, session.token_source = args.token.strip(), "--token"
    else:
        session.token, session.token_source = discover_token()

    # Verify a discovered token up front so the banner can say who you are —
    # skipped silently on network trouble (the token may still work later).
    if session.token:
        ui.set_status(style.wrap(style.gray, "Checking saved token with GitHub…"))
        ui.start_spinner()
        info, _ = validate_token(session.token)
        ui.stop_spinner()
        session.viewer = info

    if args.username:
        if not USERNAME_RE.fullmatch(args.username):
            ui.line(f"'{args.username}' is not a valid GitHub username.")
            sys.exit(2)
        ok = analyze_flow(ui, session, args.username,
                          interactive=sys.stdin.isatty(), make_pdf=args.pdf)
        sys.exit(0 if ok else 1)

    shell(ui, session)


def _can_encode(sample: str) -> bool:
    try:
        sample.encode(sys.stdout.encoding or "utf-8")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(130)

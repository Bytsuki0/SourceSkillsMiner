"""
JSON output helpers.

``sanitize_floats`` and ``save_score_to_json`` were previously duplicated in
ScoringSys (and a third copy lives in backend/api.py, which is intentionally
left untouched so the subprocess contract stays stable).
"""

from __future__ import annotations

import json
import math
import os
import traceback
from typing import Dict

__all__ = ["sanitize_floats", "save_score_to_json"]


def sanitize_floats(obj):
    """
    Recursively replace float nan/inf with None so ``json.dump`` never raises.

    The scoring math can produce nan (0/0 ratios) or inf (log of huge numbers);
    these are not valid JSON, so the frontend would otherwise show a hard error.
    """
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def save_score_to_json(result: Dict, username: str, *, base_dir: str) -> str:
    """
    Write ``result`` to ``<json_dir>/<username>.json``.

    The destination is ``$SSM_OUTPUT_DIR`` when set (the backend uses this to
    pin output to a controlled path), otherwise ``<base_dir>/json``.
    """
    env_dir = os.environ.get("SSM_OUTPUT_DIR", "").strip()
    json_dir = env_dir if env_dir else os.path.join(base_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

    file_path = os.path.join(json_dir, f"{username}.json")
    print(f"\nSaving result for {username} -> {file_path}...")

    clean_result = sanitize_floats(result)
    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(clean_result, fh, indent=2, ensure_ascii=False)
        print("Saved successfully.")
        return file_path
    except Exception as exc:
        print(f"ERROR saving JSON to {file_path}: {exc}")
        traceback.print_exc()
        raise

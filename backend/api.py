"""
backend/api.py

Flask API that orchestrates:
  1. ScoringSys.py  — mines GitHub, scores the user, writes json/{username}.json
  2. Bayers_Classifier/classify_profile.py  — classifies the profile

Endpoints
─────────
  POST /api/analyze       { "username": "octocat" }
  GET  /api/status/<job>  Server-Sent Events stream (progress updates)
  GET  /api/health        Liveness check

Run with:
    python backend/api.py
    (from the project root — one level above this file)
"""

import os
import sys
import json
import uuid
import queue
import threading
import subprocess
import configparser
from typing import Optional

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ── Paths (all relative to the project root, one level up from this file) ───
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_SCRIPT  = os.path.join(ROOT, 'ScoringSys.py')
CLASS_SCRIPT  = os.path.join(ROOT, 'Bayers_Classifier', 'classify_profile.py')
MODEL_PATH    = os.path.join(ROOT, 'Bayers_Classifier', 'models', 'developer_classifier.joblib')
CONFIG_MAIN   = os.path.join(ROOT, 'config_main.ini')
JSON_DIR      = os.path.join(ROOT, 'json')
PYTHON        = sys.executable   # use the same venv Python that runs this file

app = Flask(__name__)
CORS(app)   # allow Node.js frontend on a different port

# ── Global error handlers — guarantee JSON for every error response ──────────
# Without these Flask returns plain HTML on 400/404/500, which breaks res.json()
# in the frontend.

@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': f'Bad request: {str(e)}'}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': f'Not found: {str(e)}'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': f'Method not allowed: {str(e)}'}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Job store: job_id -> {"status", "progress", "result", "error", "events"}
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _read_token() -> str:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_MAIN)
    token = cfg.get('github', 'token', fallback='').strip()
    if not token:
        raise RuntimeError(
            f"Could not read 'token' from {CONFIG_MAIN}. "
            "Make sure config_main.ini exists in the project root."
        )
    return token


def _push(job_id: str, msg: str, pct: int) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['progress'] = pct
            _jobs[job_id]['events'].put({'message': msg, 'pct': pct})


def _finish(job_id: str, result: Optional[dict], error: Optional[str]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id]['status']   = 'done' if result else 'error'
        _jobs[job_id]['result']   = result
        _jobs[job_id]['error']    = error
        _jobs[job_id]['events'].put({'done': True, 'error': error})


# ── Background worker ────────────────────────────────────────────────────────

def _run_analysis(job_id: str, username: str) -> None:
    try:
        token = _read_token()
    except Exception as exc:
        _finish(job_id, None, str(exc))
        return

    # ── 1. Write per-job config.ini into a temp working dir ──────────────
    import tempfile, shutil
    job_dir = tempfile.mkdtemp(prefix=f'ssm_{username}_')

    try:
        cfg_path = os.path.join(job_dir, 'config.ini')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(f'[github]\nusername = {username}\ntoken = {token}\n')

        _push(job_id, 'Config ready — starting GitHub mining…', 5)

        # ── 2. Run ScoringSys.py ─────────────────────────────────────────
        result = subprocess.run(
            [PYTHON, SCORE_SCRIPT],
            cwd=job_dir,
            capture_output=True,
            text=True,
            timeout=600,          # 10 min ceiling
        )

        if result.returncode != 0:
            err = (result.stderr or result.stdout or 'ScoringSys.py failed').strip()
            _finish(job_id, None, f'ScoringSys error: {err}')
            return

        _push(job_id, 'Scoring complete — running classifier…', 75)

        # ── 3. Find the written JSON ──────────────────────────────────────
        # ScoringSys saves to <project_root>/json/{username}.json
        profile_json_path = os.path.join(ROOT, 'json', f'{username}.json')

        # Fallback: check the job_dir too (in case cwd-relative save)
        if not os.path.exists(profile_json_path):
            alt = os.path.join(job_dir, 'json', f'{username}.json')
            if os.path.exists(alt):
                profile_json_path = alt

        if not os.path.exists(profile_json_path):
            _finish(job_id, None, f'JSON output not found for user {username}')
            return

        with open(profile_json_path, 'r', encoding='utf-8') as f:
            profile_data = json.load(f)

        # ── 4. Run classifier ─────────────────────────────────────────────
        cls_result = subprocess.run(
            [PYTHON, CLASS_SCRIPT,
             '--json', profile_json_path,
             '--model', MODEL_PATH,
             '--output', os.path.join(job_dir, 'classification.json')],
            capture_output=True,
            text=True,
            timeout=60,
        )

        classification = {}
        cls_json_path  = os.path.join(job_dir, 'classification.json')
        if os.path.exists(cls_json_path):
            with open(cls_json_path, 'r', encoding='utf-8') as f:
                classification = json.load(f)
        else:
            # Parse stdout as fallback
            classification = {'error': cls_result.stderr or 'Classifier produced no output'}

        _push(job_id, 'Classification complete — preparing response…', 95)

        combined = {
            'username':       username,
            'profile':        profile_data,
            'classification': classification,
        }

        _finish(job_id, combined, None)

    except subprocess.TimeoutExpired:
        _finish(job_id, None, 'Analysis timed out (>10 min). Profile may be too large.')
    except Exception as exc:
        _finish(job_id, None, str(exc))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'python': sys.version})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        body     = request.get_json(force=True, silent=True) or {}
        username = (body.get('username') or '').strip()
        if not username:
            return jsonify({'error': 'username is required'}), 400

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                'status':   'running',
                'username': username,
                'progress': 0,
                'result':   None,
                'error':    None,
                'events':   queue.Queue(),
            }

        thread = threading.Thread(target=_run_analysis, args=(job_id, username), daemon=True)
        thread.start()

        return jsonify({'job_id': job_id}), 202

    except Exception as exc:
        return jsonify({'error': f'Failed to start analysis: {str(exc)}'}), 500


@app.route('/api/status/<job_id>')
def status_stream(job_id: str):
    """Server-Sent Events stream for real-time progress."""

    with _jobs_lock:
        if job_id not in _jobs:
            return jsonify({'error': 'job not found'}), 404

    def event_generator():
        q = _jobs[job_id]['events']
        while True:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                yield 'data: {"heartbeat": true}\n\n'
                continue

            if event.get('done'):
                with _jobs_lock:
                    job = _jobs.get(job_id, {})
                if job.get('error'):
                    yield f'data: {json.dumps({"error": job["error"]})}\n\n'
                else:
                    payload = json.dumps({'done': True, 'result': job.get('result')})
                    yield f'data: {payload}\n\n'
                return
            else:
                yield f'data: {json.dumps(event)}\n\n'

    return Response(
        stream_with_context(event_generator()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':  'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/result/<job_id>')
def get_result(job_id: str):
    """Poll endpoint for clients that don't support SSE."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404
    return jsonify({
        'status':   job['status'],
        'progress': job['progress'],
        'result':   job['result'],
        'error':    job['error'],
    })


if __name__ == '__main__':
    os.makedirs(JSON_DIR, exist_ok=True)

    missing = [p for p in [SCORE_SCRIPT, CLASS_SCRIPT, MODEL_PATH] if not os.path.exists(p)]
    if missing:
        print("WARNING: The following required files were not found:")
        for m in missing:
            print(f"  {m}")
        print("Make sure to run from the project root and that all scripts are present.\n")

    print(f"Python   : {PYTHON}")
    print(f"Root     : {ROOT}")
    print(f"Scorer   : {SCORE_SCRIPT}")
    print(f"Classifier: {CLASS_SCRIPT}")
    print(f"Model    : {MODEL_PATH}")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
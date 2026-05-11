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
import io
import json
import math
import uuid
import queue
import threading
import subprocess
import configparser
import tempfile
import shutil
from typing import Optional

from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS

# Server-side PDF generator (ReportLab) — imported lazily so the rest of the
# API still starts up even if reportlab is not yet installed.
try:
    from pdf_generator import build_pdf as _build_pdf
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_SCRIPT = os.path.join(ROOT, 'ScoringSys.py')
CLASS_SCRIPT = os.path.join(ROOT, 'Bayers_Classifier', 'classify_profile.py')
MODEL_PATH   = os.path.join(ROOT, 'Bayers_Classifier', 'models', 'developer_classifier.joblib')
CONFIG_MAIN  = os.path.join(ROOT, 'config_main.ini')
JSON_DIR     = os.path.join(ROOT, 'json')
PDF_DIR      = os.path.join(ROOT, 'pdf')
PYTHON       = sys.executable

app = Flask(__name__)
CORS(app)

# ── Global JSON error handlers ────────────────────────────────────────────────

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

# Job store
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _sanitize_floats(obj):
    """
    Recursively walk a JSON-serializable structure and replace any
    float('nan'), float('inf'), or float('-inf') with None.

    Python's json module raises ValueError on these values because they
    are not valid JSON.  ScoringSys can produce them when:
      - A GitHub API call returns 0 totals (0/0 division → nan)
      - A log-normalisation receives a very large number (→ inf)
    Replacing with None means the frontend will display '—' instead
    of crashing the serialisation step.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _push(job_id: str, msg: str, pct: int) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['progress'] = pct
            _jobs[job_id]['events'].put({'message': msg, 'pct': pct})


def _finish(job_id: str, result: Optional[dict], error: Optional[str]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id]['status'] = 'done' if result else 'error'
        _jobs[job_id]['result'] = result
        _jobs[job_id]['error']  = error
        _jobs[job_id]['events'].put({'done': True, 'error': error})


# ── Background worker ─────────────────────────────────────────────────────────

def _run_analysis(job_id: str, username: str) -> None:
    try:
        token = _read_token()
    except Exception as exc:
        _finish(job_id, None, str(exc))
        return

    job_dir = tempfile.mkdtemp(prefix=f'ssm_{username}_')

    try:
        # ── 1. Write per-job config.ini ───────────────────────────────────
        cfg_path = os.path.join(job_dir, 'config.ini')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(f'[github]\nusername = {username}\ntoken = {token}\n')

        # We tell ScoringSys exactly where to write the JSON by creating
        # the json/ subdirectory inside job_dir and passing it via env var.
        # This removes all ambiguity about __file__-relative paths.
        out_json_dir = os.path.join(job_dir, 'json')
        os.makedirs(out_json_dir, exist_ok=True)
        out_json_path = os.path.join(out_json_dir, f'{username}.json')

        _push(job_id, 'Config ready — starting GitHub mining…', 5)

        # ── 2. Run ScoringSys.py ──────────────────────────────────────────
        # Set SSM_OUTPUT_DIR so ScoringSys writes to our controlled path.
        # Also capture all output so we can surface errors even on exit 0.
                # ── 2. Run ScoringSys.py exactly like its CLI main() ─────────────
        # We pass the same arguments main() would normally read from argparse,
        # but keep the execution isolated in a subprocess.
        # ── 2. Run ScoringSys.py ──────────────────────────────────────────
        env = os.environ.copy()
        env['SSM_OUTPUT_DIR'] = out_json_dir
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'

        score_cmd = [
            PYTHON, SCORE_SCRIPT,
            '--username', username,
            '--token', token,
            '--repo-limit', '200',
            '--import-max-repos', '20',
            '--import-max-files', '30',
            # '--skip-import-scan',   # uncomment if you want to skip that part
        ]

        score_proc = subprocess.run(
            score_cmd,
            cwd=job_dir,
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )

        combined_output = (score_proc.stdout or '') + (score_proc.stderr or '')
        print("\n=== ScoringSys OUTPUT ===")
        print(combined_output)
        print("================================\n")

        if score_proc.returncode != 0:
            snippet = combined_output.strip()[:8000] or f'ScoringSys.py exited with code {score_proc.returncode}'
            _finish(job_id, None, f'ScoringSys error (exit {score_proc.returncode}):\n{snippet}')
            return

        # ── 3. Locate the output JSON ─────────────────────────────────────
        candidates = [
            out_json_path,
            os.path.join(ROOT, 'json', f'{username}.json'),
            os.path.join(job_dir, 'json', f'{username}.json'),
        ]
        profile_json_path = next((p for p in candidates if os.path.exists(p)), None)

        if profile_json_path is None:
            detail = combined_output.strip()[:2000] if combined_output.strip() else 'No output captured from ScoringSys.'
            _finish(job_id, None,
                    f'ScoringSys completed (exit 0) but no JSON was saved for "{username}".\n\n'
                    f'Output:\n{detail}')
            return
        # ── 3. Locate the output JSON ─────────────────────────────────────
        # Priority 1: the path we asked ScoringSys to write to
        # Priority 2: the canonical project-level json/ dir (original behaviour)
        # Priority 3: cwd-relative fallback
        candidates = [
            out_json_path,
            os.path.join(ROOT, 'json', f'{username}.json'),
            os.path.join(job_dir, 'json', f'{username}.json'),
        ]
        profile_json_path = next((p for p in candidates if os.path.exists(p)), None)

        if profile_json_path is None:
            detail = combined_output.strip()[:2000] if combined_output.strip() else \
                'No output captured from ScoringSys.'

            print("\n=== ScoringSys DEBUG OUTPUT ===")
            print(combined_output)
            print("================================\n")

            _finish(job_id, None,
                f'No JSON generated for "{username}".\n\nOutput:\n{detail}')
            return

        # ── 4. Load and sanitize the JSON ─────────────────────────────────
        with open(profile_json_path, 'r', encoding='utf-8') as f:
            raw_profile = json.load(f)

        # Replace any nan/inf floats that would break JSON serialisation
        # in the SSE stream or when sending to the frontend.
        profile_data = _sanitize_floats(raw_profile)

        _push(job_id, 'Scoring complete — running classifier…', 75)
        
        # ── 5. Run classifier ─────────────────────────────────────────────
        cls_proc = subprocess.run(
            [
                PYTHON, CLASS_SCRIPT,
                '--json', profile_json_path,
                '--model', MODEL_PATH,
                '--output', os.path.join(job_dir, 'classification.json'),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        cls_json_path = os.path.join(job_dir, 'classification.json')
        if os.path.exists(cls_json_path):
            with open(cls_json_path, 'r', encoding='utf-8') as f:
                classification = _sanitize_floats(json.load(f))
        else:
            cls_out = (cls_proc.stderr or cls_proc.stdout or '').strip()
            classification = {'error': cls_out or 'Classifier produced no output'}

        _push(job_id, 'Classification complete — preparing response…', 95)
        # Persist JSON to project-level folder BEFORE deleting temp dir
        final_json_path = os.path.join(JSON_DIR, f'{username}.json')
        os.makedirs(JSON_DIR, exist_ok=True)

        shutil.copy2(profile_json_path, final_json_path)

        _finish(job_id, {
            'username':       username,
            'profile':        profile_data,
            'classification': classification,
            'json_path':      final_json_path  # optional but useful
        }, None)

    except subprocess.TimeoutExpired:
        _finish(job_id, None,
                'Analysis timed out (>20 min). '
                'The profile may be too large or GitHub is rate-limiting.')
    except Exception as exc:
        _finish(job_id, None, str(exc))
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'python': sys.version,
        'scripts': {
            'ScoringSys':  os.path.exists(SCORE_SCRIPT),
            'classifier':  os.path.exists(CLASS_SCRIPT),
            'model':       os.path.exists(MODEL_PATH),
            'config_main': os.path.exists(CONFIG_MAIN),
        }
    })


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

        thread = threading.Thread(
            target=_run_analysis, args=(job_id, username), daemon=True
        )
        thread.start()

        return jsonify({'job_id': job_id}), 202

    except Exception as exc:
        return jsonify({'error': f'Failed to start analysis: {str(exc)}'}), 500




@app.route('/api/status/<job_id>')
def status_stream(job_id: str):
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
            'Cache-Control':     'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/result/<job_id>')
def get_result(job_id: str):
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


@app.route('/api/save_pdf', methods=['POST'])
def save_pdf():
    username = request.form.get('username')
    if not username:
        return jsonify({'error': 'username is required'}), 400

    if 'pdf' not in request.files:
        return jsonify({'error': 'pdf file is missing'}), 400

    pdf_file = request.files['pdf']
    
    # Ensure the ./pdf directory exists
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(PDF_DIR, f"{username}.pdf")
    
    # Save the file
    pdf_file.save(pdf_path)
    return jsonify({'status': 'ok', 'pdf_path': pdf_path})


@app.route('/api/generate_pdf', methods=['POST'])
def generate_pdf():
    """
    Server-side PDF generation using ReportLab.
    Expects JSON body: { "username": "...", "profile": {...}, "classification": {...} }
    Returns the PDF as an attachment, and also saves it to PDF_DIR/<username>.pdf.
    """
    if not _PDF_AVAILABLE:
        return jsonify({'error': 'reportlab is not installed on this server. '
                                 'Run: pip install reportlab'}), 501

    body = request.get_json(force=True, silent=True) or {}
    username       = (body.get('username') or '').strip()
    profile_data   = body.get('profile')
    cls_data       = body.get('classification')

    if not username:
        return jsonify({'error': 'username is required'}), 400
    if not isinstance(profile_data, dict):
        return jsonify({'error': 'profile must be a JSON object'}), 400
    if cls_data is None:
        cls_data = {}

    try:
        pdf_bytes = _build_pdf(username, profile_data, cls_data)
    except Exception as exc:
        import traceback
        traceback.print_exc()          # ← add this line
        return jsonify({'error': f'PDF generation failed: {exc}'}), 500

    # Persist to ./pdf/<username>.pdf (mirrors save_pdf behaviour)
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(PDF_DIR, f'{username}.pdf')
    with open(pdf_path, 'wb') as fh:
        fh.write(pdf_bytes)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'{username}.pdf',
    )


if __name__ == '__main__':
    os.makedirs(JSON_DIR, exist_ok=True)

    missing = [p for p in [SCORE_SCRIPT, CLASS_SCRIPT, MODEL_PATH] if not os.path.exists(p)]
    if missing:
        print("WARNING: The following required files were not found:")
        for m in missing:
            print(f"  {m}")
        print()

    print(f"Python     : {PYTHON}")
    print(f"Root       : {ROOT}")
    print(f"Scorer     : {SCORE_SCRIPT}  ({'found' if os.path.exists(SCORE_SCRIPT) else 'MISSING'})")
    print(f"Classifier : {CLASS_SCRIPT}  ({'found' if os.path.exists(CLASS_SCRIPT) else 'MISSING'})")
    print(f"Model      : {MODEL_PATH}  ({'found' if os.path.exists(MODEL_PATH) else 'MISSING'})")
    print(f"Config     : {CONFIG_MAIN}  ({'found' if os.path.exists(CONFIG_MAIN) else 'MISSING'})")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
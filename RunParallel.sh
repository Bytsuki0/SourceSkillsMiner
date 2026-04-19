#!/usr/bin/env bash
# RunWorkType.sh
#
# Parallel runner for run_worktype.py.
#
# Reads up to 16 GitHub tokens from config_main.ini and cycles through
# them per-job so rate limits are distributed evenly across accounts.
# Each job runs in an isolated temp directory with its own config.ini,
# allowing MaxParallel instances to run simultaneously without collisions.
#
# Output: ./json_train/{username}.json for every processed user.
#
# Requirements: bash >= 4, mktemp, awk
#
# Usage:
#   chmod +x RunWorkType.sh
#   ./RunWorkType.sh
#
# Configuration: edit the variables in the "── Configuration ──" block below.

set -u
IFS=$'\n'

# ── Configuration ───────────────────────────────────────────────────────────
UsersFile="github_users.txt"
MainConfig="config_main.ini"
Script="run_worktype.py"
VenvPython="$HOME/Documents/Coding/venv/bin/python"   # ← adjust to your venv

# Maximum number of simultaneous run_worktype.py instances.
# Each instance makes GraphQL + REST calls; keep ≤ number of tokens to
# avoid hammering a single account's rate limit.
MaxParallel=2

# All output files land here regardless of each job's temp working dir.
OUTPUT_DIR="$(pwd)/json_train"
# ────────────────────────────────────────────────────────────────────────────

# ── Utility functions ────────────────────────────────────────────────────────
err()  { printf '%s\n' "$*" >&2; }
info() { printf '%s\n' "$*"; }

resolve_path() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$p" 2>/dev/null || readlink -f "$p" 2>/dev/null || printf '%s' "$p"
  else
    readlink -f "$p" 2>/dev/null || printf '%s' "$p"
  fi
}

sanitize() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

# ── Pre-flight checks ────────────────────────────────────────────────────────
if [[ ! -f "$UsersFile" ]];  then err "Error: $UsersFile not found.";  exit 1; fi
if [[ ! -f "$MainConfig" ]]; then err "Error: $MainConfig not found."; exit 1; fi
if [[ ! -f "$Script" ]];     then err "Error: $Script not found.";     exit 1; fi
if [[ ! -x "$VenvPython" && ! -f "$VenvPython" ]]; then
  err "Error: Python not found at $VenvPython"; exit 1
fi

# ── Read tokens from config_main.ini ────────────────────────────────────────
# Supports token, token_1 … token_15 (16 total).
# The script exits only if not even one token is found.
declare -a tokens=()

while IFS='=' read -r raw_key raw_value; do
  key="$(printf '%s' "$raw_key"     | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  val="$(printf '%s' "$raw_value"   | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  case "$key" in
    token)    tokens[0]="$val"  ;;
    token_1)  tokens[1]="$val"  ;;
    token_2)  tokens[2]="$val"  ;;
    token_3)  tokens[3]="$val"  ;;
    token_4)  tokens[4]="$val"  ;;
    token_5)  tokens[5]="$val"  ;;
    token_6)  tokens[6]="$val"  ;;
    token_7)  tokens[7]="$val"  ;;
    token_8)  tokens[8]="$val"  ;;
    token_9)  tokens[9]="$val"  ;;
    token_10) tokens[10]="$val" ;;
    token_11) tokens[11]="$val" ;;
    token_12) tokens[12]="$val" ;;
    token_13) tokens[13]="$val" ;;
    token_14) tokens[14]="$val" ;;
    token_15) tokens[15]="$val" ;;
  esac
done < <(
  awk '
    /^[[:space:]]*(token|token_[0-9]+)[[:space:]]*=/ { print }
  ' "$MainConfig"
)

if (( ${#tokens[@]} == 0 )); then
  err "Error: No tokens found in $MainConfig."; exit 1
fi

info "Loaded ${#tokens[@]} token(s) from $MainConfig."

# ── Read users ───────────────────────────────────────────────────────────────
mapfile -t users < <(
  awk '!/^[[:space:]]*($|#)/ {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "")
    if (length($0) > 0) print $0
  }' "$UsersFile"
)

if (( ${#users[@]} == 0 )); then
  info "No users found in $UsersFile."; exit 0
fi

# ── Skip users whose JSON already exists (resume-safe) ──────────────────────
declare -a pending_users=()
declare -i skipped=0

for username in "${users[@]}"; do
  safe="$(sanitize "$username")"
  if [[ -f "$OUTPUT_DIR/${safe}.json" ]]; then
    (( skipped++ )) || true
  else
    pending_users+=("$username")
  fi
done

if (( skipped > 0 )); then
  info "Skipping $skipped already-processed user(s) (JSON exists in $OUTPUT_DIR)."
fi

if (( ${#pending_users[@]} == 0 )); then
  info "All users already processed. Nothing to do."; exit 0
fi

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
AbsScript="$(resolve_path "$Script")"
ScriptDir="$(dirname "$AbsScript")"

info "Script        : $AbsScript"
info "Output dir    : $OUTPUT_DIR"
info "Users pending : ${#pending_users[@]}"
info "Max parallel  : $MaxParallel"
info "--------------------------------------------"

# ── Job-control structures ────────────────────────────────────────────────────
declare -A pids=()
declare -A jobdirs=()

reap_jobs() {
  local user pid exitcode dir
  for user in "${!pids[@]}"; do
    pid="${pids[$user]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      exitcode=$?
      dir="${jobdirs[$user]}"
      if [[ $exitcode -ne 0 ]]; then
        err "[$user] Exited with code $exitcode — temp dir kept for inspection: $dir"
      else
        info "[$user] Finished successfully."
        [[ -d "$dir" ]] && rm -rf "$dir"
      fi
      unset "pids[$user]"
      unset "jobdirs[$user]"
    fi
  done
}

wait_for_slot() {
  while (( ${#pids[@]} >= MaxParallel )); do
    sleep 0.5
    reap_jobs
  done
}

_cleanup_on_exit() {
  err "Interrupted — killing running jobs…"
  for user in "${!pids[@]}"; do
    pid="${pids[$user]}"
    kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
  done
  exit 1
}
trap _cleanup_on_exit INT TERM

# ── Dispatch ──────────────────────────────────────────────────────────────────
token_count="${#tokens[@]}"

for i in "${!pending_users[@]}"; do
  username="${pending_users[$i]}"
  wait_for_slot

  # Round-robin token cycling
  token_idx=$(( i % token_count ))
  token="${tokens[$token_idx]}"

  safe_user="$(sanitize "$username")"
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/rwt_${safe_user}.XXXX")"
  jobdirs["$username"]="$tmpdir"

  info "Starting: $username  [token $token_idx]"

  # Write isolated config.ini into the temp dir — run_worktype.py reads from cwd
  printf '[github]\nusername = %s\ntoken = %s\n' "$username" "$token" \
    > "$tmpdir/config.ini"

  (
    cd "$tmpdir" || exit 1

    # WTA_OUTPUT_DIR tells run_worktype.py exactly where to write its JSON file.
    # PYTHONPATH ensures WorkTypeAnalyzer.py is importable from its original location.
    export WTA_OUTPUT_DIR="$OUTPUT_DIR"
    export PYTHONPATH="$ScriptDir${PYTHONPATH:+:$PYTHONPATH}"

    "$VenvPython" "$AbsScript" >> "$tmpdir/out.log" 2>&1
  ) &
  pids["$username"]=$!
done

# ── Drain ─────────────────────────────────────────────────────────────────────
info "--------------------------------------------"
info "All jobs dispatched. Waiting for ${#pids[@]} remaining job(s)…"

while (( ${#pids[@]} > 0 )); do
  sleep 0.5
  reap_jobs
done

info "--------------------------------------------"
info "Done. JSON files written to: $OUTPUT_DIR"

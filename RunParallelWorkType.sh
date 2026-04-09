#!/usr/bin/env bash
# RunParallelWorkType.sh
#
# Parallel runner for WorkTypeAnalyzer.py.
# Mirrors RunParallel.sh exactly but targets WorkTypeAnalyzer instead of
# ScoringSys, and writes output to json_train/ (via WTA_OUTPUT_DIR env var).
#
# Requirements: bash >=4, mktemp, awk

set -u
IFS=$'\n'

# ── Configuration ──────────────────────────────────────────────────────────
UsersFile="github_users.txt"
MainConfig="config_main.ini"
Script="WorkTypeAnalyzer.py"
VenvPython="$HOME/Documents/Coding/venv/bin/python"   # adjust if needed

MaxParallel=2

# Output directory for json_train files.
# WorkTypeAnalyzer reads WTA_OUTPUT_DIR from the environment so every job
# writes to the SAME shared json_train/ folder regardless of its temp cwd.
OUTPUT_DIR="$(pwd)/json_train"

# ── Utility functions ───────────────────────────────────────────────────────
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

# ── Pre-flight checks ───────────────────────────────────────────────────────
if [[ ! -f "$UsersFile" ]];  then err "Error: $UsersFile not found.";  exit 1; fi
if [[ ! -f "$MainConfig" ]]; then err "Error: $MainConfig not found."; exit 1; fi
if [[ ! -f "$Script" ]];     then err "Error: $Script not found.";     exit 1; fi
if [[ ! -x "$VenvPython" && ! -f "$VenvPython" ]]; then
  err "Error: Python not found at $VenvPython"
  exit 1
fi

# ── Read token from config_main.ini ────────────────────────────────────────
# ── Read tokens from config_main.ini ───────────────────────────────────────
declare -a tokens=()

while IFS='=' read -r raw_key raw_value; do
  key="$(printf '%s' "$raw_key"   | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  value="$(printf '%s' "$raw_value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

  case "$key" in
    token)   tokens[0]="$value" ;;
    token_1) tokens[1]="$value" ;;
    token_2) tokens[2]="$value" ;;
    token_3) tokens[3]="$value" ;;
    token_4) tokens[4]="$value" ;;
    token_5) tokens[5]="$value" ;;
    token_6) tokens[6]="$value" ;;
    token_7) tokens[7]="$value" ;;
    token_8) tokens[8]="$value" ;;
    token_9) tokens[9]="$value" ;;
    token_10) tokens[10]="$value" ;;
    token_11) tokens[11]="$value" ;;
    token_12) tokens[12]="$value" ;;
    token_13) tokens[13]="$value" ;;
    token_14) tokens[14]="$value" ;;
    token_15) tokens[15]="$value" ;;
  esac
done < <(
  awk '
    /^[[:space:]]*(token|token_1|token_2|token_3|token_4|token_5|token_6|token_7|token_8|token_9|token_10|token_11|token_12|token_13|token_14|token_15)[[:space:]]*=/ {
      print
    }
  ' "$MainConfig"
)

if (( ${#tokens[@]} < 16 )); then
  err "Error: Could not read all tokens from $MainConfig."
  exit 1
fi

info "Tokens loaded from $MainConfig."
# ── Read users ──────────────────────────────────────────────────────────────
mapfile -t users < <(
  awk '!/^[[:space:]]*($|#)/ {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "")
    if (length($0) > 0) print $0
  }' "$UsersFile"
)

if (( ${#users[@]} == 0 )); then
  info "No users found in $UsersFile."
  exit 0
fi

# ── Ensure output directory exists ─────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
info "Output directory : $OUTPUT_DIR"
info "Found ${#users[@]} user(s). Running up to $MaxParallel in parallel."
info "--------------------------------------------"

AbsScript="$(resolve_path "$Script")"

# ── Job-control structures ──────────────────────────────────────────────────
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
        err "[$user] Script exited with code $exitcode"
        # Leave the temp dir for debugging on failure
      else
        info "[$user] Finished successfully."
        # Clean up temp dir on success
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
  err "Interrupted. Killing running jobs..."
  for user in "${!pids[@]}"; do
    pid="${pids[$user]}"
    kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
  done
  exit 1
}
trap _cleanup_on_exit INT TERM

# ── Dispatch ────────────────────────────────────────────────────────────────
for i in "${!users[@]}"; do
  username="${users[$i]}"
  wait_for_slot

  # Rotate tokens: 0 -> token, 1 -> token_1, 2 -> token_2, 3 -> token, ...
  token="${tokens[$(( i % ${#tokens[@]} ))]}"

  safe_user="$(sanitize "$username")"
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/wta_${safe_user}.XXXX")"
  jobdirs["$username"]="$tmpdir"

  info "Starting job for: $username using token index $(( i % ${#tokens[@]} ))"

  printf '[github]\nusername = %s\ntoken = %s\n' "$username" "$token" \
    > "$tmpdir/config.ini"

  (
    cd "$tmpdir" || exit 1
    export WTA_OUTPUT_DIR="$OUTPUT_DIR"
    "$VenvPython" "$AbsScript"
  ) &
  pids["$username"]=$!
done

# ── Drain ───────────────────────────────────────────────────────────────────
info "--------------------------------------------"
info "All jobs dispatched. Waiting for ${#pids[@]} remaining job(s)..."

while (( ${#pids[@]} > 0 )); do
  sleep 0.5
  reap_jobs
done

info "--------------------------------------------"
info "All users processed. JSON files written to: $OUTPUT_DIR"

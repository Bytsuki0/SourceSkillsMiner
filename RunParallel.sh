#!/usr/bin/env bash
# RunParallel.sh — cycles tokens (token → token_1 → token_2 → token …) per job

set -u
IFS=$'\n'

# -------- Configurações --------
UsersFile="github_users.txt"
MainConfig="config_main.ini"
Script="ScoringSys.py"
VenvPython="/home/tsuki/Documents/Coding/GitBlame/venv/bin/python"

MaxParallel=1

# -------- Funções utilitárias --------
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

# Read a key from an ini file.  Usage: read_ini <file> <key>
read_ini() {
  local file="$1" key="$2"
  grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" | head -n1 | awk -F'=' '{$1=""; sub(/^[[:space:]]+/,"",$0); print $0}' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# -------- Checks iniciais --------
if [[ ! -f "$UsersFile" ]];  then err "Error: $UsersFile not found.";  exit 1; fi
if [[ ! -f "$MainConfig" ]]; then err "Error: $MainConfig not found."; exit 1; fi
if [[ ! -f "$Script" ]];     then err "Error: $Script not found.";     exit 1; fi
if [[ ! -x "$VenvPython" ]]; then err "Error: Python not found or not executable at $VenvPython"; exit 1; fi

# -------- Ler os três tokens --------
# Keys in config_main.ini: token, token_1, token_2
TOKEN_0="$(read_ini "$MainConfig" "token")"
TOKEN_1="$(read_ini "$MainConfig" "token_1")"
TOKEN_2="$(read_ini "$MainConfig" "token_2")"

if [[ -z "$TOKEN_0" ]]; then err "Error: Could not read 'token' from $MainConfig.";   exit 1; fi
if [[ -z "$TOKEN_1" ]]; then err "Error: Could not read 'token_1' from $MainConfig."; exit 1; fi
if [[ -z "$TOKEN_2" ]]; then err "Error: Could not read 'token_2' from $MainConfig."; exit 1; fi

# Store them in an indexed array so we can cycle with modulo
TOKEN_POOL=("$TOKEN_0" "$TOKEN_1" "$TOKEN_2")
TOKEN_COUNT=${#TOKEN_POOL[@]}   # 3

info "Tokens loaded from $MainConfig (pool size: $TOKEN_COUNT)."

# -------- Ler usuários --------
mapfile -t users < <(awk '!/^[[:space:]]*($|#)/ { gsub(/^[[:space:]]+|[[:space:]]+$/,""); if (length($0)>0) print $0 }' "$UsersFile")

if (( ${#users[@]} == 0 )); then
  info "No users found in $UsersFile."
  exit 0
fi

info "Found ${#users[@]} user(s). Running up to $MaxParallel in parallel."
info "--------------------------------------------"

AbsScript="$(resolve_path "$Script")"

# -------- Controle de jobs --------
declare -A pids=()
declare -A jobdirs=()

pids_count() { echo "${#pids[@]}"; }

reap_jobs() {
  local user pid exitcode dir
  local -a active_users=("${!pids[@]}")
  for user in "${active_users[@]+"${active_users[@]}"}"; do
    pid="${pids[$user]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      exitcode=$?
      dir="${jobdirs[$user]}"
      if [[ $exitcode -ne 0 ]]; then
        err "[$user] Script exited with code $exitcode"
      else
        info "[$user] Finished successfully."
      fi
      if [[ -d "$dir" ]]; then rm -rf "$dir"; fi
      unset "pids[$user]"
      unset "jobdirs[$user]"
    fi
  done
}

wait_for_slot() {
  while (( $(pids_count) >= MaxParallel )); do
    sleep 0.5
    reap_jobs
  done
}

_cleanup_on_exit() {
  err "Interrupted. Killing running jobs..."
  local -a active_users=("${!pids[@]}")
  for user in "${active_users[@]+"${active_users[@]}"}"; do
    pid=${pids[$user]}
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
  done
  exit 1
}
trap _cleanup_on_exit INT TERM

# -------- Dispatch --------
job_index=0   # increments each dispatch; modulo TOKEN_COUNT selects the token

for username in "${users[@]}"; do
  wait_for_slot

  # Pick token for this job: 0→token, 1→token_1, 2→token_2, 3→token, …
  token_index=$(( job_index % TOKEN_COUNT ))
  current_token="${TOKEN_POOL[$token_index]}"

  safe_user=$(sanitize "$username")
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ssminer_${safe_user}.XXXX")"
  jobdirs["$username"]="$tmpdir"

  info "Starting job for: $username  [token_index=$token_index]"

  printf '[github]\nusername = %s\ntoken = %s\n' "$username" "$current_token" > "$tmpdir/config.ini"

  (
    cd "$tmpdir" || exit 1
    "$VenvPython" "$AbsScript"
  ) &
  pids["$username"]=$!

  (( job_index++ )) || true   # '|| true' prevents set -e from triggering on arithmetic 0
done

# -------- Drain --------
info "--------------------------------------------"
info "All jobs dispatched. Waiting for $(pids_count) remaining job(s)..."

while (( $(pids_count) > 0 )); do
  sleep 0.5
  reap_jobs
done

info "--------------------------------------------"
info "All users processed."
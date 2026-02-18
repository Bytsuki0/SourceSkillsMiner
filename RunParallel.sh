#!/usr/bin/env bash
# RunParallel.sh — versão bash do seu RunParallel.ps1
# Requisitos: bash (>=4 para arrays associativos), mktemp, awk, realpath/readlink (opcionais)

set -u
IFS=$'\n'

# -------- Configurações (equivalente às variáveis PowerShell) --------
UsersFile="github_users.txt"
MainConfig="config_main.ini"
Script="ScoringSys.py"
VenvPython="/home/tsuki/Documents/Coding/GitBlame/venv/bin/python"  # path to venv python binary

MaxParallel=1

# -------- Funções utilitárias --------
err() { printf '%s\n' "$*" >&2; }
info() { printf '%s\n' "$*"; }

# Safely resolve path to script
resolve_path() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "$p" 2>/dev/null || readlink -f "$p" 2>/dev/null || printf '%s' "$p"
  else
    readlink -f "$p" 2>/dev/null || printf '%s' "$p"
  fi
}

# Sanitize username for filenames
sanitize() {
  local s="$1"
  # keep alnum, dot, underscore, dash; replace others with underscore
  printf '%s' "$s" | sed 's/[^A-Za-z0-9._-]/_/g'
}

# -------- Checks iniciais (existência dos arquivos) --------
if [[ ! -f "$UsersFile" ]]; then err "Error: $UsersFile not found."; exit 1; fi
if [[ ! -f "$MainConfig" ]]; then err "Error: $MainConfig not found."; exit 1; fi
if [[ ! -f "$Script" ]]; then err "Error: $Script not found."; exit 1; fi
if [[ ! -x "$VenvPython" ]]; then
  err "Error: Python not found or not executable at $VenvPython"
  exit 1
fi

# -------- Ler token do config_main.ini --------
token_line=$(grep -E '^[[:space:]]*token[[:space:]]*=' "$MainConfig" | head -n1 || true)
if [[ -z "$token_line" ]]; then err "Error: Could not read token from $MainConfig."; exit 1; fi
token="${token_line#*=}"
# trim
token="$(printf '%s' "$token" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
info "Token loaded from $MainConfig."

# -------- Ler usuários (ignorar linhas vazias e comentários iniciados por #) --------
mapfile -t users < <(awk '!/^[[:space:]]*($|#)/ { gsub(/^[[:space:]]+|[[:space:]]+$/,""); if (length($0)>0) print $0 }' "$UsersFile")

if (( ${#users[@]} == 0 )); then
  info "No users found in $UsersFile."
  exit 0
fi

info "Found ${#users[@]} user(s). Running up to $MaxParallel in parallel."
info "--------------------------------------------"

AbsScript="$(resolve_path "$Script")"

# -------- Preparar estruturas para controle de jobs --------
declare -A pids=()      # pids[user]=pid
declare -A jobdirs=()   # jobdirs[user]=dir

# Safe count — avoids "unbound variable" with set -u on older bash
pids_count() { echo "${#pids[@]}"; }

# Reap finished jobs: check each pid; if finished, wait to get exitcode, report and cleanup
reap_jobs() {
  local user pid exitcode dir
  # Copy keys into a regular array first to avoid word-splitting on special chars
  local -a active_users=("${!pids[@]}")
  for user in "${active_users[@]+"${active_users[@]}"}"; do
    pid="${pids[$user]}"
    # if process does not exist, it's finished
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      exitcode=$?
      dir="${jobdirs[$user]}"
      if [[ $exitcode -ne 0 ]]; then
        err "[$user] Script exited with code $exitcode"
      else
        info "[$user] Finished successfully."
      fi
      # cleanup
      if [[ -d "$dir" ]]; then
        rm -rf "$dir"
      fi
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

# Trap para matar jobs filhos caso o script seja interrompido
_cleanup_on_exit() {
  err "Interrupted. Killing running jobs..."
  local -a active_users=("${!pids[@]}")
  for user in "${active_users[@]+"${active_users[@]}"}"; do
    pid=${pids[$user]}
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  exit 1
}
trap _cleanup_on_exit INT TERM

# -------- Dispatch (cria job por usuário) --------
for username in "${users[@]}"; do
  wait_for_slot

  safe_user=$(sanitize "$username")
  # use mktemp para evitar colisão; nome legível
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/ssminer_${safe_user}.XXXX")"
  jobdirs["$username"]="$tmpdir"

  info "Starting job for: $username"
  # criar config.ini com encoding UTF-8 (sem BOM)
  printf '[github]\nusername = %s\ntoken = %s\n' "$username" "$token" > "$tmpdir/config.ini"

  # executar em subshell no background; gravar saída em log se desejar
  (
    cd "$tmpdir" || exit 1
    # redirecionar stdout/stderr para arquivos de log por job (opcional)
    # "$VenvPython" "$AbsScript" > "out.log" 2> "err.log"
    "$VenvPython" "$AbsScript"
  ) &
  pid=$!
  pids["$username"]=$pid
done

# -------- Drain remaining jobs --------
info "--------------------------------------------"
info "All jobs dispatched. Waiting for $(pids_count) remaining job(s)..."

while (( $(pids_count) > 0 )); do
  sleep 0.5
  reap_jobs
done

info "--------------------------------------------"
info "All users processed."
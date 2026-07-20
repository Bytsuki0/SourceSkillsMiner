#!/usr/bin/env sh
# SourceSkillsMiner launcher — uses the project virtualenv when present.
HERE="$(cd "$(dirname "$0")" && pwd)"
for candidate in "$HERE/venv/bin/python" "$HERE/.venv/bin/python" "$HERE/win_venv/Scripts/python.exe"; do
    if [ -x "$candidate" ]; then
        exec "$candidate" "$HERE/miner.py" "$@"
    fi
done
exec python3 "$HERE/miner.py" "$@"

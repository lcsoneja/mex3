#!/usr/bin/env bash
# SONJ4 launcher (Linux / Raspberry Pi / macOS). Relative to this file.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$HERE/.venv/bin/python" ]; then
  exec "$HERE/.venv/bin/python" "$HERE/sonj.py" "$@"
else
  exec python3 "$HERE/sonj.py" "$@"
fi

#!/usr/bin/env bash
# One-command launcher for macOS / Linux.
#   ./run.sh          -> dashboard at http://localhost:8501
#   ./run.sh api      -> REST API at http://localhost:8000/docs
#   ./run.sh test     -> run the test suite
#   ./run.sh bench    -> re-run the ground-truth benchmark
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ is required. Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtual environment with $($PY --version)..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

STAMP=.venv/.installed
if [ ! -f "$STAMP" ] || [ requirements.txt -nt "$STAMP" ] || [ "${1:-}" = "test" ]; then
  echo "Installing dependencies (first run takes 1-2 minutes)..."
  python -m pip install --upgrade pip -q
  if [ "${1:-}" = "test" ]; then pip install -q -r requirements-dev.txt; else pip install -q -r requirements.txt; fi
  touch "$STAMP"
fi

case "${1:-app}" in
  api)   exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" ;;
  test)  exec python -m pytest ;;
  bench) exec python -m xaudit.cli benchmark ;;
  *)     URL="http://localhost:${PORT:-8501}"
         echo ""; echo "  Dashboard starting at $URL  (press Ctrl+C to stop)"; echo ""
         if [ -z "${NO_BROWSER:-}" ]; then
           ( sleep 4; if command -v open >/dev/null 2>&1; then open "$URL"; \
             elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1; fi ) &
         fi
         exec python -m streamlit run app/streamlit_app.py --server.port "${PORT:-8501}" ;;
esac

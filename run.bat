@echo off
REM One-command launcher for Windows. Usage: run.bat  |  run.bat api  |  run.bat test
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv ( %PY% -m venv .venv )
call .venv\Scripts\activate.bat
if not exist .venv\.installed (
  python -m pip install --upgrade pip -q
  pip install -q -r requirements-dev.txt
  echo ok> .venv\.installed
)
if "%1"=="api" ( python -m uvicorn api.main:app --port 8000 & goto :eof )
if "%1"=="test" ( python -m pytest & goto :eof )
start "" http://localhost:8501
python -m streamlit run app/streamlit_app.py --server.port 8501

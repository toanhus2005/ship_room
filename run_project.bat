@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    set "CONFIG=configs\toan_config.sample1.json"
) else (
    set "CONFIG=%~1"
)

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating virtual environment in .venv...
    py -3 -m venv .venv 2>nul
    if errorlevel 1 (
        python -m venv .venv
        if errorlevel 1 (
            echo [ERROR] Cannot create virtual environment. Install Python 3.10+ first.
            exit /b 1
        )
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Cannot activate .venv
    exit /b 1
)

echo [INFO] Installing dependencies...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

echo [INFO] Running pipeline with config: %CONFIG%
python -m src.pipeline_toan --config "%CONFIG%"
if errorlevel 1 (
    echo [ERROR] Pipeline execution failed.
    exit /b 1
)

for /f "usebackq delims=" %%S in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Get-Content -Raw '%CONFIG%' | ConvertFrom-Json).video.source } catch { '' }"`) do set "VIDEO_SOURCE=%%S"
if "%VIDEO_SOURCE%"=="" (
    set "VIDEO_SOURCE=data/sample1.mp4"
)

for /f "usebackq delims=" %%M in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Get-Content -Raw '%CONFIG%' | ConvertFrom-Json).detection.model_name } catch { '' }"`) do set "MODEL_NAME=%%M"
if "%MODEL_NAME%"=="" (
    set "MODEL_NAME=yolov8s.pt"
)

echo [INFO] Exporting person appearance timeline...
echo %VIDEO_SOURCE%| findstr /r "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    python -m src.module1.person_event_tour --video "%VIDEO_SOURCE%" --model "%MODEL_NAME%" --tracks-jsonl artifacts/events/person_tracks.jsonl --out artifacts/events/person_appearance_tour.json
    if errorlevel 1 (
        echo [WARN] Could not export person appearance timeline.
    ) else (
        echo [INFO] Exported: artifacts/events/person_appearance_tour.json
    )
) else (
    echo [WARN] Skip timeline export because source is a live camera index: %VIDEO_SOURCE%
)

echo [INFO] Starting live preview web server...
start "Ship Room Live Preview" ".venv\Scripts\python.exe" -m src.module2.live_preview_web --video "%VIDEO_SOURCE%"

echo [INFO] Opening browser at http://127.0.0.1:8787
timeout /t 2 >nul
start "" "http://127.0.0.1:8787"

echo [DONE] Pipeline completed. Live preview is running at http://127.0.0.1:8787
endlocal

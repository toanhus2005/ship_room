@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" (
    set "CONFIG=configs\toan_config.img_5812.json"
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

for /f "usebackq delims=" %%S in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Get-Content -Raw '%CONFIG%' | ConvertFrom-Json).video.source } catch { '' }"`) do set "VIDEO_SOURCE=%%S"
if "%VIDEO_SOURCE%"=="" (
    set "VIDEO_SOURCE=data/IMG_5812.MOV"
)

for /f "usebackq delims=" %%M in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Get-Content -Raw '%CONFIG%' | ConvertFrom-Json).detection.model_name } catch { '' }"`) do set "MODEL_NAME=%%M"
if "%MODEL_NAME%"=="" (
    set "MODEL_NAME=yolov8m.pt"
)

for /f "usebackq delims=" %%C in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "try { (Get-Content -Raw '%CONFIG%' | ConvertFrom-Json).detection.confidence_threshold } catch { '' }"`) do set "CONF_THRESHOLD=%%C"
if "%CONF_THRESHOLD%"=="" (
    set "CONF_THRESHOLD=0.4"
)

echo [INFO] Releasing port 8787 if occupied...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$conn = Get-NetTCPConnection -LocalPort 8787 -State Listen -ErrorAction SilentlyContinue; if ($conn) { $procIds = $conn | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($procId in $procIds) { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue } }"

echo [INFO] Starting live preview web server...
start "Ship Room Live Preview" ".venv\Scripts\python.exe" -m src.module2.live_preview_web --video "%VIDEO_SOURCE%" --model "%MODEL_NAME%" --conf "%CONF_THRESHOLD%"

echo [INFO] Opening browser at http://127.0.0.1:8787
timeout /t 2 >nul
start "" "http://127.0.0.1:8787"

echo [DONE] Preview started for %VIDEO_SOURCE%.
endlocal

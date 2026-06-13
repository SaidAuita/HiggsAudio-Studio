@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   Higgs Audio Studio - Установка
echo ========================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "TEMP=%SCRIPT_DIR%temp"
set "TMP=%SCRIPT_DIR%temp"

REM === Папки ===
if not exist "python" mkdir python
if not exist "downloads" mkdir downloads
if not exist "temp" mkdir temp
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "output" mkdir output
if not exist "voices" mkdir voices
if not exist "generations" mkdir generations

REM === [1] Выбор оборудования ===
echo.
echo Выберите ваше оборудование:
echo   1 - NVIDIA GTX 10xx Pascal            CUDA 11.8
echo   2 - NVIDIA RTX 20xx / 30xx            CUDA 12.6
echo   3 - NVIDIA RTX 40xx / 50xx            CUDA 12.8
echo   4 - CPU без GPU, очень медленно
echo.
set /p GPU_CHOICE="Введите номер (1-4): "

if "%GPU_CHOICE%"=="1" ( set "CUDA_VERSION=cu118" & set "CUDA_NAME=GTX 10xx Pascal / CUDA 11.8" )
if "%GPU_CHOICE%"=="2" ( set "CUDA_VERSION=cu126" & set "CUDA_NAME=RTX 20xx-30xx / CUDA 12.6" )
if "%GPU_CHOICE%"=="3" ( set "CUDA_VERSION=cu128" & set "CUDA_NAME=RTX 40xx-50xx / CUDA 12.8" )
if "%GPU_CHOICE%"=="4" ( set "CUDA_VERSION=cpu" & set "CUDA_NAME=CPU" )

if not defined CUDA_VERSION (
    echo Неверный выбор!
    pause
    exit /b 1
)
echo Выбрано: !CUDA_NAME!

REM === [2] Python embed 3.12 ===
set "PY_VER=3.12.8"
if exist "python\python.exe" (
    echo Python уже установлен, пропускаем...
) else (
    echo Скачиваю Python !PY_VER! embeddable...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/!PY_VER!/python-!PY_VER!-embed-amd64.zip' -OutFile 'downloads\python.zip'}"
    powershell -Command "& {Expand-Archive -Path 'downloads\python.zip' -DestinationPath 'python' -Force}"
)

REM === [3] _pth патч (включить site-packages, иначе pip-пакеты не найдутся) ===
cd python
if exist "python312._pth" (
    echo import site> python312._pth.new
    echo.>> python312._pth.new
    echo python312.zip>> python312._pth.new
    echo .>> python312._pth.new
    echo ..\Lib\site-packages>> python312._pth.new
    move /y python312._pth.new python312._pth >nul
)
cd ..

REM === [4] pip ===
if exist "python\Scripts\pip.exe" (
    echo pip уже установлен
) else (
    echo Устанавливаю pip...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'downloads\get-pip.py'}"
    python\python.exe downloads\get-pip.py --no-warn-script-location
)
python\python.exe -m pip install --upgrade pip setuptools wheel --no-warn-script-location

REM === [5] PyTorch ===
echo Устанавливаю PyTorch (!CUDA_VERSION!)...
if "%CUDA_VERSION%"=="cpu" (
    python\python.exe -m pip install torch torchaudio --no-warn-script-location
) else (
    python\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/%CUDA_VERSION% --no-warn-script-location
)

REM === [6] Зависимости проекта (transformers>=5.5, gradio, bitsandbytes и т.д.) ===
echo Устанавливаю зависимости...
python\python.exe -m pip install -r requirements.txt --no-warn-script-location

REM === Финализация ===
echo %CUDA_VERSION%> cuda_version.txt
echo.
echo ========================================
echo   Установка завершена! Запуск: run.bat
echo   Модели (TTS ~9.3 ГБ + LLM-режиссёр) скачаются при первом запуске
echo ========================================
pause

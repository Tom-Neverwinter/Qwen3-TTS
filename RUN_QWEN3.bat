@echo off
setlocal enabledelayedexpansion
title Qwen3-TTS Ultimate Workbench

echo =====================================
echo   Qwen3-TTS Ultimate Workbench
echo =====================================
echo:

echo [INFO] Checking for Python 3.12 virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Environment found. Starting application...
    echo:
    call venv\Scripts\activate.bat
    python app.py
) else (
    echo [WARN] Virtual environment not found!
    echo:
    echo Would you like to automatically install all dependencies?
    echo This will:
    echo   - Detect your GPU and CUDA version
    echo   - Install Python 3.12 (if needed)
    echo   - Install PyTorch with appropriate CUDA support
    echo   - Install Flash Attention (if compatible GPU detected)
    echo   - Install Qwen-TTS and all dependencies
    echo:
    choice /C YN /M "Run automatic installer"
    
    if errorlevel 2 (
        echo:
        echo Installation cancelled. To install manually, run:
        echo   powershell -ExecutionPolicy Bypass -File install.ps1
        echo:
        pause
        exit /b
    )
    
    echo:
    echo [INFO] Running automatic installer...
    echo:
    powershell -ExecutionPolicy Bypass -File install.ps1
    
    if %errorlevel% equ 0 (
        echo:
        echo [SUCCESS] Installation complete! Restarting application...
        echo:
        timeout /t 3 /nobreak >nul
        call "%~f0"
        exit /b
    ) else (
        echo:
        echo [ERROR] Installation failed. Please check the error messages above.
        echo:
        pause
        exit /b 1
    )
)

echo:
echo =====================================
echo   Workbench Closed
echo =====================================
pause

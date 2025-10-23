@echo off
echo F1Tenth Simulator GUI Launcher
echo ==============================
echo.

cd /d "%~dp0"

if exist "f1tenth_sim_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call f1tenth_sim_env\Scripts\activate.bat
    echo.
)

echo Starting GUI launcher...
python f1tenth_gui_launcher.py

pause
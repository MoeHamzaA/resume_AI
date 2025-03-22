@echo off
echo Downloading NLTK data...
python initialize.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to download NLTK data
    pause
    exit /b 1
)

echo Starting the application...
python self.py
pause 
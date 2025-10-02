@echo off
REM Startup script for production

echo 🚀 Starting Rooster Recognition System...

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "logs" mkdir "logs"
if not exist "uploads" mkdir "uploads"
if not exist "static\uploads" mkdir "static\uploads"

REM Set permissions (Windows doesn't need chmod)
echo ✅ Directories created

REM Start the application
echo 🌟 Starting Flask application...
if exist "gunicorn.conf.py" (
    gunicorn -c gunicorn.conf.py app:app
) else (
    python app.py
)

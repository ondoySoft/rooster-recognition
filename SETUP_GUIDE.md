# 🚀 Rooster Recognition System - Setup Guide

## Prerequisites Installation

### 1. Python 3.10 Installation
- **Download**: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
- **Installation Steps**:
  1. Run installer as Administrator
  2. ✅ Check "Add Python to PATH"
  3. Click "Install Now"
  4. Verify: `python --version` → Python 3.10.11

### 2. XAMPP Installation
- **Download**: https://www.apachefriends.org/download.html
- **Installation Steps**:
  1. Run installer as Administrator
  2. Select Apache + MySQL components
  3. Install to default location
  4. Start Apache and MySQL services

## Project Transfer & Setup

### 1. Copy Project Files
**IMPORTANT**: Copy the **ENTIRE** `source_code` folder including the `venv310` directory (~1.9GB)
```
📁 Copy complete source_code folder:
├── 📁 venv310/              # Virtual environment with ALL dependencies (~1.9GB)
├── 📄 app.py                # Main Flask application
├── 📁 flask_app/            # Templates and static files
├── 📁 dataset/              # Training images
├── 📁 local_model/          # Local training models
├── 📁 google_colab_models/  # Google Colab models
├── 📁 teachable_machine_models/  # Teachable Machine models
├── 📄 requirements.txt      # Dependencies list
├── 📄 start_app.bat        # Startup script
└── 📄 SETUP_GUIDE.md       # This guide
```

### 2. Database Setup
1. Open phpMyAdmin: http://localhost/phpmyadmin
2. Create new database: `rooster_db`
3. Import database schema: `database/rooster_db.sql`

### 3. Run Application
**EASY METHOD**: Use the provided startup script
```bash
# Double-click or run from command line
start_app.bat
```
*Note: The script will automatically check for Python 3.10, activate venv310, and install dependencies if needed*

**MANUAL METHOD**: 
```bash
# Navigate to project directory
cd path/to/source_code

# Activate existing virtual environment
venv310\Scripts\activate

# Run application
python app.py
```

## Verification

### ✅ Check Installation
- [ ] Python 3.10.11 installed
- [ ] XAMPP running (Apache + MySQL)
- [ ] Database `rooster_db` created
- [ ] Complete `source_code` folder copied (including `venv310/`)
- [ ] `start_app.bat` file present

### ✅ Test Application
- [ ] Run `start_app.bat` successfully
- [ ] Flask app starts: http://localhost:5000
- [ ] Upload page works: http://localhost:5000/upload
- [ ] Tools page accessible: http://localhost:5000/tools
- [ ] Admin login works: http://localhost:5000/login

## Troubleshooting

### Common Issues:
1. **Python not found**: Add Python to PATH
2. **Database connection failed**: Check XAMPP MySQL service
3. **Port 5000 in use**: Change Flask port in app.py
4. **Model loading errors**: Check model files exist
5. **Virtual environment not found**: Ensure `venv310/` folder was copied

### Support:
- Check Tools page: http://localhost:5000/tools
- View system requirements and compatibility
- Download links for all dependencies

## Quick Start Commands

```bash
# 1. EASIEST METHOD - Use startup script
start_app.bat

# 2. MANUAL METHOD - Command line
cd path/to/source_code
venv310\Scripts\activate
python app.py

# 3. Access web interface
# Open browser: http://localhost:5000
```

## Important Notes

### ✅ What's Included in venv310/:
- **TensorFlow 2.20.0** - AI/ML framework
- **Keras 3.11.3** - High-level neural network API
- **Flask 3.0.0** - Web framework
- **OpenCV 4.12.0** - Computer vision
- **SQLAlchemy 2.0.23** - Database ORM
- **NumPy 2.2.6** - Numerical computing
- **Scikit-learn 1.7.2** - Machine learning utilities
- **All dependencies** - Ready to use!

### 🚀 Transfer Benefits:
- **No internet required** during setup (if venv310 copied)
- **No dependency installation** needed (if venv310 copied)
- **Guaranteed compatibility** - exact same environment
- **Faster setup** - just copy and run
- **28,017 files** (~1.9GB) - complete environment

### 🛠️ Smart Startup Script (start_app.bat):
The `start_app.bat` script automatically:
1. **Checks** for Python 3.10 installation
2. **Creates** venv310 if it doesn't exist
3. **Activates** the virtual environment
4. **Installs** dependencies from requirements files
5. **Starts** the Flask application
6. **Opens** browser to http://localhost:5000

*This makes the script work even if venv310 wasn't copied!*

---
**Note**: This setup guide ensures compatibility with your current working environment. The `venv310` folder contains all dependencies and is portable across Windows machines with Python 3.10.

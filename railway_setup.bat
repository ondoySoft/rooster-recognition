@echo off
REM Railway Quick Setup Script for Rooster Recognition System
REM This script helps you deploy to Railway quickly

echo 🚂 Railway Quick Setup for Rooster Recognition System
echo.

echo 📋 Prerequisites Check:
echo 1. Do you have a Railway account? (https://railway.app)
echo 2. Do you have a GitHub account?
echo 3. Is your code pushed to GitHub?
echo.
pause

echo 🚀 Step 1: Railway Account Setup
echo 1. Go to https://railway.app
echo 2. Sign up with GitHub
echo 3. Create a new project
echo 4. Select "Deploy from GitHub repo"
echo.
pause

echo 📦 Step 2: GitHub Repository Setup
echo Your deployment package is ready in: rooster_recognition_deploy
echo.
echo To create GitHub repository:
echo 1. Go to https://github.com/new
echo 2. Create repository: rooster-recognition
echo 3. Upload the rooster_recognition_deploy folder contents
echo 4. Or use Git commands below:
echo.
echo Git Commands:
echo cd rooster_recognition_deploy
echo git init
echo git add .
echo git commit -m "Initial commit: Rooster Recognition System"
echo git branch -M main
echo git remote add origin https://github.com/YOUR_USERNAME/rooster-recognition.git
echo git push -u origin main
echo.
pause

echo 🔧 Step 3: Railway Configuration
echo 1. In Railway dashboard, select your project
echo 2. Go to "Variables" tab
echo 3. Add these environment variables:
echo.
echo FLASK_ENV=production
echo SECRET_KEY=your-super-secret-key-here
echo DATABASE_URL=mysql+pymysql://root:password@containers-us-west-xxx.railway.app:xxxx/railway
echo UPLOAD_FOLDER=uploads
echo MAX_CONTENT_LENGTH=16777216
echo.
pause

echo 🗄️ Step 4: Add Database Service
echo 1. In Railway project, click "New"
echo 2. Select "Database" → "MySQL"
echo 3. Railway will create database automatically
echo 4. Copy DATABASE_URL from database service
echo 5. Add DATABASE_URL to your app's environment variables
echo.
pause

echo 🚀 Step 5: Deploy
echo 1. Railway will automatically detect your Python app
echo 2. It will install dependencies from requirements.txt
echo 3. Build and deploy your application
echo 4. Your app will be available at: https://your-app.railway.app
echo.
pause

echo ✅ Step 6: Test Your Deployment
echo Test these URLs:
echo - Home: https://your-app.railway.app
echo - Upload: https://your-app.railway.app/upload
echo - Admin Login: https://your-app.railway.app/login
echo - Dashboard: https://your-app.railway.app/dashboard
echo.
pause

echo 🎉 Railway Deployment Complete!
echo.
echo 📚 Additional Resources:
echo - Railway Docs: https://docs.railway.app
echo - Python Deployment: https://docs.railway.app/deploy/python
echo - Database Setup: https://docs.railway.app/databases
echo - Railway Discord: https://discord.gg/railway
echo.
echo 🚂 Your Rooster Recognition System is now live on Railway!
echo.
pause

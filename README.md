# 🐓 Rooster Recognition System

A Flask web application that uses AI to identify rooster breeds from uploaded images. Built with TensorFlow, OpenCV, and MySQL.

## 🌟 Features

- **AI-Powered Breed Recognition**: Upload rooster images and get instant breed predictions with confidence scores
- **Modern Web Interface**: Clean, responsive design with Bootstrap 5
- **Database Storage**: All predictions are saved to MySQL database for future reference
- **RESTful API**: API endpoints for mobile app integration
- **File Upload Security**: Secure file handling with validation
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- XAMPP (for MySQL database)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd roster_identifiaction_system/source_code
   ```

2. **Start XAMPP**
   - Start Apache and MySQL services
   - Ensure MySQL is running on port 3306

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   python init_database.py
   ```

5. **Add your AI model** (Optional)
   - Place your `rooster_model.h5` file in the project root
   - If no model is provided, a dummy model will be created for demonstration

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open your browser**
   - Navigate to `http://localhost:5000`
   - Start uploading rooster images!

## 📁 Project Structure

```
roster_identifiaction_system/source_code/
├── app.py                          # Main Flask application
├── init_database.py                # Database initialization script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── rooster_model.h5               # AI model file (add your own)
└── flask_app/
    ├── templates/                  # HTML templates
    │   ├── base.html              # Base template
    │   ├── index.html             # Home page
    │   ├── upload.html            # Upload page
    │   ├── results.html           # Results page
    │   ├── records.html           # Records listing
    │   ├── 404.html               # Error page
    │   └── 500.html               # Server error page
    ├── static/                     # Static files
    │   ├── css/
    │   │   └── style.css          # Custom styles
    │   └── js/
    │       └── main.js            # Custom JavaScript
    └── uploads/                    # Uploaded images storage
```

## 🗄️ Database Schema

### Tables

1. **users**
   - `id` (Primary Key)
   - `username` (Unique)
   - `email` (Unique)
   - `password_hash`
   - `role` (admin/user)
   - `created_at`

2. **breeds**
   - `id` (Primary Key)
   - `name` (Unique)
   - `description`
   - `characteristics`
   - `created_at`

3. **rooster_records**
   - `id` (Primary Key)
   - `filename`
   - `original_filename`
   - `predicted_breed`
   - `confidence_score`
   - `user_id` (Foreign Key)
   - `breed_id` (Foreign Key)
   - `uploaded_at`

## 🔧 Configuration

### Database Connection
Update the connection string in `app.py`:
```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/rooster_db'
```

### File Upload Settings
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
```

## 🎯 API Endpoints

### Web Routes
- `GET /` - Home page
- `GET /upload` - Upload form
- `POST /upload` - Handle file upload
- `GET /results/<id>` - View prediction results
- `GET /records` - List all records

### API Routes
- `POST /api/predict` - Upload image and get prediction (JSON response)

### API Usage Example
```bash
curl -X POST -F "file=@rooster_image.jpg" http://localhost:5000/api/predict
```

## 🤖 AI Model Integration

### Model Requirements
- Input shape: `(224, 224, 3)`
- Output: Softmax probabilities for breed classes
- Format: TensorFlow/Keras `.h5` file

### Supported Breeds
- Rhode Island Red
- Leghorn
- Sussex
- Orpington
- Wyandotte
- Plymouth Rock
- Australorp
- Brahma

### Custom Model Integration
To use your own model:

1. Train your model with the supported breeds
2. Save as `rooster_model.h5`
3. Update the breed mapping in `app.py`:
   ```python
   breed_names = ['Your_Breed_1', 'Your_Breed_2', ...]
   ```

## 🔒 Security Features

- **File Validation**: Only image files are allowed
- **Secure Filenames**: Using `werkzeug.utils.secure_filename`
- **File Size Limits**: Maximum 16MB upload size
- **SQL Injection Protection**: Using SQLAlchemy ORM
- **XSS Protection**: Template auto-escaping enabled

## 🚀 Deployment

### Production Setup

1. **Environment Variables**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

2. **Database Configuration**
   - Update connection string with production credentials
   - Use environment variables for sensitive data

3. **Web Server**
   ```bash
   # Using Gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   
   # Using Waitress (Windows)
   waitress-serve --host=0.0.0.0 --port=5000 app:app
   ```

4. **Static Files**
   - Configure web server to serve static files
   - Consider using CDN for better performance

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest
```

### Manual Testing
1. Upload various rooster images
2. Check prediction accuracy
3. Verify database storage
4. Test error handling

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure XAMPP MySQL is running
   - Check connection string
   - Verify database exists

2. **Model Loading Error**
   - Check if `rooster_model.h5` exists
   - Verify model format compatibility
   - Check TensorFlow version

3. **File Upload Issues**
   - Check file size limits
   - Verify file permissions
   - Ensure upload directory exists

4. **Import Errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.11+)
   - Verify virtual environment activation

## 📈 Future Enhancements

- [ ] User authentication with Flask-Login
- [ ] Admin dashboard for user management
- [ ] Google Gemini API integration for breed descriptions
- [ ] Batch image processing
- [ ] Mobile app development
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Cloud storage integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- Flask community for the web framework
- Bootstrap team for the UI components
- OpenCV contributors for image processing

---

**Happy Rooster Recognition! 🐓✨**

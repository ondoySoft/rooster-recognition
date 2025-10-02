"""
Rooster Recognition System
A Flask web application for identifying rooster breeds using AI
"""

import os
import cv2
import numpy as np
import json
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
    UserMixin,
)
from datetime import datetime
import tensorflow as tf
from PIL import Image
import io
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

# Initialize Flask app
app = Flask(__name__, template_folder='flask_app/templates', static_folder='flask_app/static')

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/rooster_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'flask_app/uploads'
# Increase upload size to accommodate large .h5/ZIP models (e.g., Colab/TM)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256MB max file size
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours in seconds

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Login manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'basic'  # Changed from 'strong' to 'basic'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Global variable to store the loaded model
rooster_model = None

# Database Models
class User(db.Model, UserMixin):
    """User model for authentication and roles"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'admin' or 'user'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

class AuditLog(db.Model):
    """Audit log model for tracking user actions"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(64), nullable=False)  # CREATE, UPDATE, DELETE, LOGIN, LOGOUT, etc.
    entity_type = db.Column(db.String(64), nullable=False)  # breed, record, user, etc.
    entity_id = db.Column(db.Integer, nullable=True)  # ID of the affected entity
    payload = db.Column(db.JSON, nullable=True)  # Additional data (before/after values, etc.)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('audit_logs', lazy=True))
    
    def __repr__(self):
        return f'<AuditLog {self.action} {self.entity_type}:{self.entity_id} by {self.user_id}>'


class TrainingHistory(db.Model):
    """Training history model to track all training sessions"""
    __tablename__ = 'training_history'
    
    id = db.Column(db.Integer, primary_key=True)
    training_type = db.Column(db.Enum('local', 'google_colab', 'teachable_machine', name='training_type'), nullable=False)
    model_source_id = db.Column(db.Integer, db.ForeignKey('model_source.id'), nullable=False)
    description = db.Column(db.Text, nullable=True)
    accuracy_score = db.Column(db.Float, nullable=True)
    status = db.Column(db.Enum('pending', 'training', 'completed', 'accepted', 'rejected', 'failed', name='training_status'), nullable=False, default='pending')
    model_location = db.Column(db.String(255), nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    accepted_at = db.Column(db.DateTime, nullable=True)
    rejected_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('training_history', lazy=True))
    model_source = db.relationship('ModelSource', backref=db.backref('training_history', lazy=True))
    
    def __repr__(self):
        return f'<TrainingHistory {self.training_type}:{self.model_source_id} - {self.status}>'


class ModelSource(db.Model):
    """Model source model to track different AI models"""
    __tablename__ = 'model_source'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(50), unique=True, nullable=False)  # e.g., 'local_v1', 'colab_v1', 'tm_v1'
    model_source = db.Column(db.String(50), nullable=False)  # 'Local Training', 'Google Colab', 'Teachable Machine'
    description = db.Column(db.Text, nullable=True)  # Description of the model
    accuracy_score = db.Column(db.Float, nullable=True)  # Training accuracy score
    is_active = db.Column(db.Boolean, default=True)  # Whether this model is currently active
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelSource {self.model_id}: {self.model_source}>'


class Breed(db.Model):
    """Breed model to store rooster breed information"""
    __tablename__ = 'breeds'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    slug = db.Column(db.String(120), unique=True, nullable=True)
    description = db.Column(db.Text)
    characteristics = db.Column(db.Text)
    image_path = db.Column(db.String(255), nullable=True)
    is_deleted = db.Column(db.Boolean, default=False)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    updated_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<Breed {self.name}>'

class RoosterRecord(db.Model):
    """Model to store uploaded images and AI predictions"""
    __tablename__ = 'rooster_records'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    predicted_category = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)
    breed_id = db.Column(db.Integer, db.ForeignKey('breeds.id'), nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_correct = db.Column(db.SmallInteger, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    model_id = db.Column(db.Integer, nullable=True)
    model_source = db.Column(db.String(32), nullable=True)
    
    # Relationships
    breed = db.relationship('Breed', backref=db.backref('rooster_records', lazy=True))
    
    def __repr__(self):
        return f'<RoosterRecord {self.filename}>'

# Utility Functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_active_model_source():
    """Get the currently active model source from database"""
    try:
        # Ensure we always have an application context when querying
        if db.session is not None:
            try:
                return ModelSource.query.filter_by(is_active=True).first()
            except Exception:
                pass
        # Fallback: create a temporary app context for safe query
        with app.app_context():
            active_model = ModelSource.query.filter_by(is_active=True).first()
            return active_model
    except Exception as e:
        print(f"⚠️ Error getting active model: {e}")
        return None

def load_model():
    """Load the pre-trained TensorFlow model based on active model in database"""
    global rooster_model
    try:
        if rooster_model is None:
            # Get active model from database
            active_model = get_active_model_source()
            
            if active_model:
                print(f"🔄 Loading active model: {active_model.model_source} ({active_model.model_id})")
                
                # Load model based on active model source
                if active_model.model_source == 'Local Training':
                    model_loaded = load_local_model()
                elif active_model.model_source == 'Google Colab':
                    model_loaded = load_google_colab_model()
                elif active_model.model_source == 'Teachable Machine':
                    model_loaded = load_teachable_machine_model()
                else:
                    print(f"⚠️ Unknown model source: {active_model.model_source}, falling back to local model")
                    model_loaded = load_local_model()
                
                if not model_loaded:
                    print("❌ Failed to load active model, falling back to local model")
                    model_loaded = load_local_model()
                    
            else:
                print("⚠️ No active model found in database, using local model as fallback...")
                model_loaded = load_local_model()
            
            if not model_loaded:
                print("🔄 All model loading failed. Creating compatible model...")
                rooster_model = create_compatible_model()
                
        return rooster_model
        
    except Exception as e:
        print(f"❌ Error in load_model(): {e}")
        # Fallback to creating a compatible model
        rooster_model = create_compatible_model()
        return rooster_model

def load_local_model():
    """Load local training model"""
    global rooster_model
    try:
        print("🔄 Loading Local Training model...")
        
        # Check for models in local_model directory
        local_model_dir = 'local_model'
        model_path = os.path.join(local_model_dir, 'rooster_model.h5')
        savedmodel_dir = local_model_dir
        json_arch_path = os.path.join(local_model_dir, 'rooster_model_architecture.json')
        weights_path = os.path.join(local_model_dir, 'rooster_model.weights.h5')
        
        # Try H5 model first
        if os.path.exists(model_path):
            print("🔄 Found H5 model, attempting to load...")
            loading_strategies = [
                lambda: tf.keras.models.load_model(model_path, compile=False),
                lambda: tf.keras.models.load_model(model_path, compile=False, safe_mode=False),
                lambda: tf.keras.models.load_model(model_path, compile=False, custom_objects=None),
            ]
            
            for i, strategy in enumerate(loading_strategies, 1):
                try:
                    print(f"🔄 Trying strategy {i}...")
                    rooster_model = strategy()
                    test_input = tf.random.normal((1, 224, 224, 3))
                    test_output = rooster_model(test_input)
                    if test_output.shape == (1, 4):
                        print(f"✅ Local Training model loaded successfully!")
                        rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        return True
                except Exception as e:
                    print(f"❌ Strategy {i} failed: {str(e)[:100]}...")
                    continue
        
        # Try SavedModel directory
        if os.path.isdir(savedmodel_dir):
            print(f"🔄 Trying SavedModel directory...")
            try:
                rooster_model = tf.keras.models.load_model(savedmodel_dir, compile=False)
                _ = rooster_model(tf.random.normal((1, 224, 224, 3)))
                print("✅ Loaded SavedModel successfully!")
                rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return True
            except Exception as e:
                print(f"❌ SavedModel load failed: {str(e)[:120]}...")
        
        # Try JSON + weights
        if os.path.exists(json_arch_path) and os.path.exists(weights_path):
            try:
                print("🔄 Rebuilding from JSON + weights...")
                with open(json_arch_path, 'r') as f:
                    model_json = f.read()
                rooster_model = tf.keras.models.model_from_json(model_json)
                _ = rooster_model(tf.random.normal((1, 224, 224, 3)))
                rooster_model.load_weights(weights_path)
                print("✅ Loaded JSON+weights successfully!")
                rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return True
            except Exception as e:
                print(f"❌ JSON+weights failed: {str(e)[:120]}...")
        
        print("❌ Local Training model not found or failed to load")
        return False
        
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        return False

def load_google_colab_model():
    """Load Google Colab model"""
    global rooster_model
    try:
        print("🔄 Loading Google Colab model...")
        
        # Check for models in google_colab_models directory
        colab_model_dir = 'google_colab_models'
        model_path = os.path.join(colab_model_dir, 'rooster_model.h5')
        keras_path = os.path.join(colab_model_dir, 'rooster_model.keras')
        saved_model_dir = os.path.join(colab_model_dir, 'saved_model')
        
        # Try .h5 model first
        if os.path.exists(model_path):
            print("🔄 Found Google Colab H5 model...")
            loading_strategies = [
                lambda: tf.keras.models.load_model(model_path, compile=False),
                lambda: tf.keras.models.load_model(model_path, compile=False, safe_mode=False),
                lambda: tf.keras.models.load_model(model_path, compile=False, custom_objects=None),
            ]
            
            for i, strategy in enumerate(loading_strategies, 1):
                try:
                    print(f"🔄 Trying strategy {i}...")
                    rooster_model = strategy()
                    test_input = tf.random.normal((1, 224, 224, 3))
                    test_output = rooster_model(test_input)
                    if test_output.shape == (1, 4):
                        print(f"✅ Google Colab model loaded successfully!")
                        rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        return True
                except Exception as e:
                    print(f"❌ Strategy {i} failed: {str(e)[:100]}...")
                    continue
        
        # Try .keras (Keras v3) file
        if os.path.exists(keras_path):
            print("🔄 Found Google Colab .keras model...")
            try:
                rooster_model = tf.keras.models.load_model(keras_path, compile=False)
                _ = rooster_model(tf.random.normal((1, 224, 224, 3)))
                rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                print("✅ Google Colab .keras model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ .keras load failed: {str(e)[:120]}...")

        # Try SavedModel directory (Keras then raw TF fallback)
        if os.path.isdir(saved_model_dir):
            print("🔄 Found Google Colab SavedModel directory, attempting to load...")
            # Keras API first
            try:
                rooster_model = tf.keras.models.load_model(saved_model_dir, compile=False)
                _ = rooster_model(tf.random.normal((1, 224, 224, 3)))
                rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                print("✅ Google Colab (SavedModel via Keras) loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Colab SavedModel load via Keras failed: {str(e)[:120]}...")
            # Raw TF SavedModel
            try:
                loaded = tf.saved_model.load(saved_model_dir)
                infer = loaded.signatures.get('serving_default')
                if infer is None and len(loaded.signatures) > 0:
                    infer = list(loaded.signatures.values())[0]
                if infer is None:
                    raise ValueError('No serving signature found in SavedModel')

                class _ColabSavedModelWrapper:
                    def __init__(self, fn):
                        self._fn = fn
                    def predict(self, x):
                        x_tensor = tf.convert_to_tensor(x)
                        outputs = self._fn(x_tensor)
                        first = next(iter(outputs.values()))
                        return first.numpy()

                rooster_model = _ColabSavedModelWrapper(infer)
                _ = rooster_model.predict(np.random.randn(1, 224, 224, 3).astype(np.float32))
                print("✅ Google Colab (SavedModel via tf.saved_model.load) loaded successfully!")
                return True
            except Exception as e2:
                print(f"❌ Colab SavedModel raw load failed: {str(e2)[:120]}...")

        print("❌ Google Colab model not found")
        return False
        
    except Exception as e:
        print(f"❌ Error loading Google Colab model: {e}")
        return False

def load_teachable_machine_model():
    """Load Teachable Machine model"""
    global rooster_model
    try:
        print("🔄 Loading Teachable Machine model...")
        
        # Check for Teachable Machine model files
        tm_model_dir = 'teachable_machine_models'
        # Preferred (server-friendly) formats - check multiple possible H5 filenames
        h5_paths = [
            os.path.join(tm_model_dir, 'rooster_model.h5'),
            os.path.join(tm_model_dir, 'keras_model.h5'),
            os.path.join(tm_model_dir, 'model.h5'),
            os.path.join(tm_model_dir, 'teachable_machine_model.h5')
        ]
        saved_model_dir = os.path.join(tm_model_dir, 'saved_model')
        model_json_path = os.path.join(tm_model_dir, 'model.json')
        weights_bin_path = os.path.join(tm_model_dir, 'weights.bin')
        
        # 1) Try Keras H5 converted model (check all possible filenames)
        h5_path = None
        for path in h5_paths:
            if os.path.exists(path):
                h5_path = path
                break
        
        if h5_path:
            print(f"🔄 Found converted H5 model ({os.path.basename(h5_path)}), attempting to load...")
            strategies = [
                lambda: tf.keras.models.load_model(h5_path, compile=False),
                lambda: tf.keras.models.load_model(h5_path, compile=False, safe_mode=False),
                lambda: tf.keras.models.load_model(h5_path, compile=False, custom_objects=None),
            ]
            for i, strat in enumerate(strategies, 1):
                try:
                    print(f"🔄 TM/H5: Trying strategy {i}...")
                    rooster_model = strat()
                    # Probe
                    _ = rooster_model(tf.random.normal((1, 224, 224, 3)))
                    rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    print(f"✅ Teachable Machine ({os.path.basename(h5_path)}) loaded successfully!")
                    return True
                except Exception as e:
                    print(f"❌ TM/H5 strategy {i} failed: {str(e)[:120]}...")
            print("❌ Failed to load Teachable Machine H5 model")
        
        # 2) Try SavedModel directory
        if os.path.isdir(saved_model_dir):
            print("🔄 Found SavedModel directory, attempting to load...")
            # First try via Keras API (may fail on Keras 3)
            try:
                rooster_model = tf.keras.models.load_model(saved_model_dir, compile=False)
                _ = rooster_model(tf.random.normal((1, 224, 224, 3)))
                rooster_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                print("✅ Teachable Machine (SavedModel via Keras) loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ TM/SavedModel load via Keras failed: {str(e)[:120]}...")
            # Fallback: raw TensorFlow SavedModel loader
            try:
                loaded = tf.saved_model.load(saved_model_dir)
                infer = loaded.signatures.get('serving_default')
                if infer is None and len(loaded.signatures) > 0:
                    infer = list(loaded.signatures.values())[0]
                if infer is None:
                    raise ValueError('No serving signature found in SavedModel')

                class _TMSavedModelWrapper:
                    def __init__(self, fn):
                        self._fn = fn

                    def predict(self, x):
                        x_tensor = tf.convert_to_tensor(x)
                        outputs = self._fn(x_tensor)
                        first = next(iter(outputs.values()))
                        return first.numpy()

                rooster_model = _TMSavedModelWrapper(infer)
                # Probe
                _ = rooster_model.predict(np.random.randn(1, 224, 224, 3).astype(np.float32))
                print("✅ Teachable Machine (SavedModel via tf.saved_model.load) loaded successfully!")
                return True
            except Exception as e2:
                print(f"❌ TM/SavedModel raw load failed: {str(e2)[:120]}...")
        
        # 3) If only TF.js files exist, inform conversion is required
        if os.path.exists(model_json_path) and os.path.exists(weights_bin_path):
            print("🔄 Found Teachable Machine TF.js files (model.json + weights.bin)")
            print("⚠️ Conversion required: convert TF.js to Keras .h5 or SavedModel for server inference")
            return False
        
        print("❌ Teachable Machine model files not found")
        return False
        
    except Exception as e:
        print(f"❌ Error loading Teachable Machine model: {e}")
        return False

def create_compatible_model():
    """Create a compatible model that works with the current TensorFlow version"""
    from tensorflow.keras.applications import MobileNetV2
    
    # Use MobileNetV2 as base (same as the trained model)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create the same architecture as the trained model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 categories
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Created compatible model with MobileNetV2 architecture")
    return model

def create_dummy_model():
    """Create a dummy model for demonstration when the real model is not available"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 breeds
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_path):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (standard input size for many models)
        img = cv2.resize(img, (224, 224))
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Debug: print shape, dtype, and value range
        try:
            print(f"🧪 Preprocess debug -> shape: {img.shape}, dtype: {img.dtype}, min: {img.min():.4f}, max: {img.max():.4f}")
        except Exception:
            pass
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

def load_class_mapping():
    """Load class mapping based on active model in database"""
    try:
        # Get active model from database
        active_model = get_active_model_source()
        
        if active_model:
            print(f"🔄 Loading class mapping for active model: {active_model.model_source}")
            
            # Load class mapping based on active model source
            if active_model.model_source == 'Local Training':
                mapping_path = os.path.join('local_model', 'class_mapping.json')
            elif active_model.model_source == 'Google Colab':
                mapping_path = os.path.join('google_colab_models', 'class_mapping.json')
            elif active_model.model_source == 'Teachable Machine':
                mapping_path = os.path.join('teachable_machine_models', 'class_mapping.json')
            else:
                print(f"⚠️ Unknown model source: {active_model.model_source}, using local mapping")
                mapping_path = os.path.join('local_model', 'class_mapping.json')
            
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                print(f"✅ Loaded class mapping from {mapping_path}")
                return class_mapping
            else:
                print(f"⚠️ Class mapping not found at {mapping_path}, using defaults")
        else:
            print("⚠️ No active model found, checking local_model directory...")
            mapping_path = os.path.join('local_model', 'class_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    class_mapping = json.load(f)
                print(f"✅ Loaded class mapping from {mapping_path}")
                return class_mapping
        
        # Fallback to root directory
        if os.path.exists('class_mapping.json'):
            with open('class_mapping.json', 'r') as f:
                class_mapping = json.load(f)
            print(f"✅ Loaded class mapping from root directory")
            return class_mapping
        else:
            # Fallback to default categories
            print("⚠️ No class mapping file found, using defaults")
            return {
                '0': 'bantam',
                '1': 'dual_purpose', 
                '2': 'gamefowl',
                '3': 'other'
            }
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        return {
            '0': 'bantam',
            '1': 'dual_purpose', 
            '2': 'gamefowl',
            '3': 'other'
        }

def format_category_name(category: str) -> str:
    """Normalize category labels to a single canonical set.
    Ensures we don't create duplicates like 'other' vs 'Other Breeds'.
    Canonical labels:
      - Bantam
      - Dual Purpose
      - Gamefowl
      - Other
    """
    if not category:
        return 'Other'
    import re
    raw = str(category).strip()
    # Drop any numeric/index prefixes like "0 Bantam", "2_Gamefowl"
    raw = re.sub(r'^\s*\d+\s*[-_.:]*\s*', '', raw)
    key = raw.lower().replace(' ', '_')
    # Remove common suffixes like "_breeds"/"_breed"
    key = re.sub(r'_(breeds|breed)$', '', key)
    # Normalize common variants
    if key in {'other_breeds', 'others', 'other'}:
        return 'Other'
    if key in {'bantam', 'bantams'}:
        return 'Bantam'
    if key in {'dual_purpose', 'dual-purpose', 'dual'}:
        return 'Dual Purpose'
    if key in {'gamefowl', 'game_fowl', 'game-fowl'}:
        return 'Gamefowl'
    # Fallback
    return raw.title()


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None

def predict_breed(image_path):
    """Predict rooster breed using the loaded model"""
    try:
        model = load_model()
        
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        # Make prediction
        try:
            print(f"🧪 Predict debug -> input shape: {processed_img.shape}, dtype: {processed_img.dtype}")
        except Exception:
            pass
        predictions = model.predict(processed_img)
        try:
            print(f"🧪 Predict debug -> raw predictions shape: {predictions.shape}")
        except Exception:
            pass
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Load class mapping
        class_mapping = load_class_mapping()
        
        # Map class index to category name
        predicted_category = class_mapping.get(str(predicted_class), f"category_{predicted_class}")
        
        # Format category name for display
        formatted_category = format_category_name(predicted_category)
        
        return formatted_category, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Return dummy prediction for demonstration
        return "Unknown Breed", 0.5

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/tools')
def tools():
    """Tools and Technologies page"""
    return render_template('tools.html')


# ---------- Auth Routes ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form.get('username') or ''
        password = request.form.get('password') or ''
        user = User.query.filter((User.username==username_or_email)|(User.email==username_or_email)).first()
        if user and user.is_active and user.check_password(password):
            login_user(user, remember=True)  # Added remember=True for longer sessions
            # Log successful login
            log_audit_event('LOGIN', 'user', user.id, {'ip_address': request.remote_addr})
            flash('Logged in successfully.')
            next_url = request.args.get('next') or url_for('dashboard')
            return redirect(next_url)
        flash('Invalid credentials or inactive account.')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    # Log logout before logging out user
    log_audit_event('LOGOUT', 'user', current_user.id, {'ip_address': request.remote_addr})
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('login'))


def admin_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Administrator access required.')
            return redirect(url_for('login', next=request.path))
        return func(*args, **kwargs)
    return wrapper


@app.route('/dashboard')
@login_required
def dashboard():
    # KPIs
    total_records = db.session.query(func.count(RoosterRecord.id)).scalar() or 0
    total_breeds = db.session.query(func.count(Breed.id)).filter(Breed.is_deleted==False).scalar() or 0
    correct_count = db.session.query(func.count(RoosterRecord.id)).filter(RoosterRecord.is_correct==1).scalar() or 0
    incorrect_count = db.session.query(func.count(RoosterRecord.id)).filter(RoosterRecord.is_correct==0).scalar() or 0
    unchecked_count = db.session.query(func.count(RoosterRecord.id)).filter(RoosterRecord.is_correct==None).scalar() or 0
    # Overall accuracy against all records (unchecked counted as not yet correct)
    overall_denom = max(total_records, 1)
    accuracy_pct = round((correct_count / overall_denom) * 100.0, 2)

    # Distribution by predicted category
    cat_rows = db.session.query(RoosterRecord.predicted_category, func.count(RoosterRecord.id))\
        .group_by(RoosterRecord.predicted_category).all()
    cat_labels = [r[0] or 'Unknown' for r in cat_rows]
    cat_values = [int(r[1]) for r in cat_rows]

    # Correctness breakdown
    correctness_labels = ['Correct', 'Incorrect', 'Unchecked']
    correctness_values = [correct_count, incorrect_count, unchecked_count]

    # Get active model
    active_model = get_active_model_source()

    return render_template(
        'dashboard.html',
        total_records=total_records,
        total_breeds=total_breeds,
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        unchecked_count=unchecked_count,
        accuracy_pct=accuracy_pct,
        active_model=active_model,
        cat_labels=cat_labels,
        cat_values=cat_values,
        correctness_labels=correctness_labels,
        correctness_values=correctness_values,
    )


# ---------- Breeds CRUD (Admin) ----------

def slugify_value(value: str) -> str:
    try:
        from slugify import slugify  # python-slugify
        return slugify(value)
    except Exception:
        return value.lower().strip().replace(' ', '-')


@app.route('/admin/breeds')
@login_required
@admin_required
def breeds_list():
    breeds = Breed.query.filter_by(is_deleted=False).order_by(Breed.name.asc()).all()
    return render_template('admin/breeds_list.html', breeds=breeds)


@app.route('/admin/breeds/new', methods=['GET', 'POST'])
@login_required
@admin_required
def breeds_new():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        characteristics = request.form.get('characteristics', '').strip()
        if not name:
            flash('Name is required', 'breeds')
            return redirect(request.url)
        slug = slugify_value(name)
        # duplicate guard
        existing = Breed.query.filter(Breed.is_deleted==False).filter((Breed.name==name)|(Breed.slug==slug)).first()
        if existing:
            flash('Breed name already exists. Please choose another name.', 'breeds')
            return redirect(request.url)
        # image upload
        image_file = request.files.get('image')
        image_path = None
        if image_file and image_file.filename:
            img_name = secure_filename(image_file.filename)
            os.makedirs(os.path.join('flask_app', 'static', 'breeds'), exist_ok=True)
            save_path = os.path.join('flask_app', 'static', 'breeds', img_name)
            image_file.save(save_path)
            image_path = f"breeds/{img_name}"
        breed = Breed(
            name=name,
            slug=slug,
            description=description,
            characteristics=characteristics,
            image_path=image_path,
            created_by=current_user.id if current_user.is_authenticated else None,
        )
        try:
            db.session.add(breed)
            db.session.commit()
            # Log breed creation
            log_audit_event('CREATE', 'breed', breed.id, {
                'name': breed.name,
                'slug': breed.slug,
                'description': breed.description,
                'characteristics': breed.characteristics
            })
            flash('Breed created', 'breeds')
            return redirect(url_for('breeds_list'))
        except IntegrityError:
            db.session.rollback()
            flash('Duplicate breed name or slug. Please use a unique name.', 'breeds')
            return redirect(request.url)
    return render_template('admin/breeds_form.html', mode='new')


@app.route('/admin/breeds/<int:breed_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def breeds_edit(breed_id):
    breed = Breed.query.get_or_404(breed_id)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        characteristics = request.form.get('characteristics', '').strip()
        if not name:
            flash('Name is required', 'breeds')
            return redirect(request.url)
        new_slug = slugify_value(name)
        # duplicate guard excluding current id
        dup = Breed.query.filter(Breed.id!=breed.id, Breed.is_deleted==False).filter((Breed.name==name)|(Breed.slug==new_slug)).first()
        if dup:
            flash('Another breed with this name already exists.', 'breeds')
            return redirect(request.url)
        breed.name = name
        breed.slug = new_slug
        breed.description = description
        breed.characteristics = characteristics
        breed.updated_by = current_user.id if current_user.is_authenticated else None
        breed.updated_at = datetime.utcnow()
        image_file = request.files.get('image')
        if image_file and image_file.filename:
            img_name = secure_filename(image_file.filename)
            os.makedirs(os.path.join('flask_app', 'static', 'breeds'), exist_ok=True)
            save_path = os.path.join('flask_app', 'static', 'breeds', img_name)
            image_file.save(save_path)
            breed.image_path = f"breeds/{img_name}"
        try:
            db.session.commit()
            # Log breed update
            log_audit_event('UPDATE', 'breed', breed.id, {
                'name': breed.name,
                'slug': breed.slug,
                'description': breed.description,
                'characteristics': breed.characteristics,
                'image_path': breed.image_path
            })
            flash('Breed updated', 'breeds')
            return redirect(url_for('breeds_list'))
        except IntegrityError:
            db.session.rollback()
            flash('Duplicate breed name or slug.', 'breeds')
            return redirect(request.url)
    return render_template('admin/breeds_form.html', mode='edit', breed=breed)


@app.route('/admin/breeds/<int:breed_id>/delete', methods=['POST'])
@login_required
@admin_required
def breeds_delete(breed_id):
    breed = Breed.query.get_or_404(breed_id)
    breed.is_deleted = True
    breed.updated_by = current_user.id if current_user.is_authenticated else None
    breed.updated_at = datetime.utcnow()
    db.session.commit()
    # Log breed deletion
    log_audit_event('DELETE', 'breed', breed.id, {
        'name': breed.name,
        'slug': breed.slug,
        'description': breed.description,
        'characteristics': breed.characteristics
    })
    flash('Breed deleted', 'breeds')
    return redirect(url_for('breeds_list'))


# ---------- Admin Profile ----------
@app.route('/admin/profile')
@login_required
@admin_required
def admin_profile():
    """Admin profile page"""
    return render_template('admin/profile.html', user=current_user)

@app.route('/admin/profile/update', methods=['POST'])
@login_required
@admin_required
def admin_profile_update():
    """Update admin profile"""
    try:
        # Get form data
        email = request.form.get('email', '').strip()
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Update basic info
        if email:
            current_user.email = email
        
        # Handle password change if provided
        if new_password:
            if not current_password:
                flash('Current password is required to change password', 'profile')
                return redirect(url_for('admin_profile'))
            
            if not current_user.check_password(current_password):
                flash('Current password is incorrect', 'profile')
                return redirect(url_for('admin_profile'))
            
            if new_password != confirm_password:
                flash('New passwords do not match', 'profile')
                return redirect(url_for('admin_profile'))
            
            if len(new_password) < 6:
                flash('New password must be at least 6 characters', 'profile')
                return redirect(url_for('admin_profile'))
            
            current_user.set_password(new_password)
        
        # Save changes
        db.session.commit()
        # Log profile update
        log_audit_event('UPDATE', 'user', current_user.id, {
            'email': current_user.email,
            'password_changed': bool(new_password)
        })
        flash('Profile updated successfully!', 'profile')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating profile: {str(e)}', 'profile')
    
    return redirect(url_for('admin_profile'))


# ---------- Admin Reports ----------
@app.route('/admin/reports/test')
@login_required
@admin_required
def admin_reports_test():
    """Test route for reports"""
    return "Reports test page works!"

@app.route('/admin/reports')
@login_required
@admin_required
def admin_reports():
    """Main reports dashboard"""
    try:
        # Get quick stats for the dashboard
        total_predictions = RoosterRecord.query.count()
        
        # Calculate overall accuracy
        correct_predictions = RoosterRecord.query.filter_by(is_correct=1).count()
        overall_accuracy = round((correct_predictions / total_predictions * 100), 1) if total_predictions > 0 else 0
        
        # Get active models count
        active_models = ModelSource.query.filter_by(is_active=True).count()
        
        # Get total training sessions
        total_training = TrainingHistory.query.count()
        
        # Get current time for display
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        return render_template('admin/reports_main.html',
                             total_predictions=total_predictions,
                             overall_accuracy=overall_accuracy,
                             active_models=active_models,
                             total_training=total_training,
                             current_time=current_time)
    except Exception as e:
        flash(f'Error loading reports dashboard: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/admin/reports/prediction')
@login_required
@admin_required
def admin_reports_prediction():
    """Prediction Performance Analytics Report"""
    try:
        # Basic statistics
        total_predictions = RoosterRecord.query.count()
        correct_predictions = RoosterRecord.query.filter_by(is_correct=1).count()
        overall_accuracy = round((correct_predictions / total_predictions * 100), 1) if total_predictions > 0 else 0
        
        # Average confidence score
        avg_confidence_result = db.session.query(db.func.avg(RoosterRecord.confidence_score)).scalar()
        avg_confidence = round((avg_confidence_result * 100), 1) if avg_confidence_result else 0
        
        # Simplified model performance analysis
        model_performance = []
        model_labels = []
        model_accuracies = []
        
        # Get performance by model source (simplified)
        model_stats = db.session.query(
            RoosterRecord.model_source,
            db.func.count(RoosterRecord.id).label('total_predictions')
        ).filter(RoosterRecord.model_source.isnot(None)).group_by(RoosterRecord.model_source).all()
        
        for stat in model_stats:
            # Calculate accuracy for this model
            correct_for_model = RoosterRecord.query.filter_by(model_source=stat.model_source, is_correct=1).count()
            accuracy = round((correct_for_model / stat.total_predictions * 100), 1) if stat.total_predictions > 0 else 0
            
            model_performance.append({
                'model_source': stat.model_source,
                'total_predictions': stat.total_predictions,
                'correct_predictions': correct_for_model,
                'accuracy': accuracy,
                'avg_confidence': 0  # Simplified for now
            })
            model_labels.append(stat.model_source)
            model_accuracies.append(accuracy)
        
        # Simplified breed performance analysis
        breed_performance = []
        breed_labels = []
        breed_accuracies = []
        
        breed_stats = db.session.query(
            RoosterRecord.predicted_category,
            db.func.count(RoosterRecord.id).label('total_predictions')
        ).filter(RoosterRecord.predicted_category.isnot(None)).group_by(RoosterRecord.predicted_category).all()
        
        for stat in breed_stats:
            # Calculate accuracy for this breed
            correct_for_breed = RoosterRecord.query.filter_by(predicted_category=stat.predicted_category, is_correct=1).count()
            accuracy = round((correct_for_breed / stat.total_predictions * 100), 1) if stat.total_predictions > 0 else 0
            
            breed_performance.append({
                'predicted_category': stat.predicted_category,
                'total_predictions': stat.total_predictions,
                'correct_predictions': correct_for_breed,
                'accuracy': accuracy,
                'avg_confidence': 0  # Simplified for now
            })
            breed_labels.append(stat.predicted_category)
            breed_accuracies.append(accuracy)
        
        # Simplified confidence score distribution
        confidence_ranges = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        confidence_counts = []
        
        for i in range(5):
            lower = i * 0.2
            upper = (i + 1) * 0.2
            count = RoosterRecord.query.filter(
                RoosterRecord.confidence_score >= lower,
                RoosterRecord.confidence_score < upper
            ).count()
            confidence_counts.append(count)
        
        # Simplified performance trends (last 7 days)
        from datetime import datetime, timedelta
        trend_dates = []
        trend_accuracies = []
        
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            
            day_predictions = RoosterRecord.query.filter(
                db.func.date(RoosterRecord.uploaded_at) == date.date()
            ).count()
            
            day_correct = RoosterRecord.query.filter(
                db.func.date(RoosterRecord.uploaded_at) == date.date(),
                RoosterRecord.is_correct == 1
            ).count()
            
            day_accuracy = round((day_correct / day_predictions * 100), 1) if day_predictions > 0 else 0
            
            trend_dates.append(date.strftime('%m/%d'))
            trend_accuracies.append(day_accuracy)
        
        # Reverse to show chronological order
        trend_dates.reverse()
        trend_accuracies.reverse()
        
        return render_template('admin/reports_prediction.html',
                             total_predictions=total_predictions,
                             correct_predictions=correct_predictions,
                             overall_accuracy=overall_accuracy,
                             avg_confidence=avg_confidence,
                             model_performance=model_performance,
                             model_labels=model_labels,
                             model_accuracies=model_accuracies,
                             breed_performance=breed_performance,
                             breed_labels=breed_labels,
                             breed_accuracies=breed_accuracies,
                             confidence_ranges=confidence_ranges,
                             confidence_counts=confidence_counts,
                             trend_dates=trend_dates,
                             trend_accuracies=trend_accuracies)
        
    except Exception as e:
        flash(f'Error loading prediction analytics: {str(e)}', 'error')
        return redirect(url_for('admin_reports'))

@app.route('/admin/reports/usage')
@login_required
@admin_required
def admin_reports_usage():
    """Usage & Activity Reports"""
    try:
        # Basic statistics
        total_uploads = RoosterRecord.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        
        # Count model switches (ACTIVATE actions in audit logs)
        model_switches = AuditLog.query.filter_by(action='ACTIVATE').count()
        
        # Calculate average session time (simplified - time between LOGIN and LOGOUT)
        login_logs = AuditLog.query.filter_by(action='LOGIN').order_by(AuditLog.created_at.desc()).limit(10).all()
        logout_logs = AuditLog.query.filter_by(action='LOGOUT').order_by(AuditLog.created_at.desc()).limit(10).all()
        
        avg_session_time = "N/A"  # Simplified for now
        total_sessions = AuditLog.query.filter_by(action='LOGIN').count()
        
        # Upload trends (last 7 days)
        from datetime import datetime, timedelta
        trend_dates = []
        daily_uploads = []
        
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            day_uploads = RoosterRecord.query.filter(
                db.func.date(RoosterRecord.uploaded_at) == date.date()
            ).count()
            
            trend_dates.append(date.strftime('%m/%d'))
            daily_uploads.append(day_uploads)
        
        # Reverse to show chronological order
        trend_dates.reverse()
        daily_uploads.reverse()
        
        # Model usage distribution
        model_usage_stats = []
        model_usage_labels = []
        model_usage_data = []
        
        # Get usage by model source
        for model_source in ModelSource.query.all():
            prediction_count = RoosterRecord.query.filter_by(model_source=model_source.model_source).count()
            last_used = RoosterRecord.query.filter_by(model_source=model_source.model_source).order_by(RoosterRecord.uploaded_at.desc()).first()
            
            model_usage_stats.append({
                'model_source': model_source.model_source,
                'prediction_count': prediction_count,
                'last_used': last_used.uploaded_at if last_used else None,
                'is_active': model_source.is_active
            })
            model_usage_labels.append(model_source.model_source)
            model_usage_data.append(prediction_count)
        
        # Hourly activity pattern (0-23 hours)
        hourly_labels = [f"{i:02d}:00" for i in range(24)]
        hourly_data = []
        
        for hour in range(24):
            # Count audit logs by hour
            hour_count = AuditLog.query.filter(
                db.extract('hour', AuditLog.created_at) == hour
            ).count()
            hourly_data.append(hour_count)
        
        # Find peak hour
        peak_hour = f"{hourly_data.index(max(hourly_data)):02d}:00" if hourly_data else "N/A"
        
        # User activity timeline (last 7 days)
        activity_dates = []
        login_events = []
        
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            day_logins = AuditLog.query.filter(
                db.func.date(AuditLog.created_at) == date.date(),
                AuditLog.action == 'LOGIN'
            ).count()
            
            activity_dates.append(date.strftime('%m/%d'))
            login_events.append(day_logins)
        
        # Reverse to show chronological order
        activity_dates.reverse()
        login_events.reverse()
        
        # Find busiest day
        busiest_day = activity_dates[login_events.index(max(login_events))] if login_events else "N/A"
        
        # Calculate average daily uploads
        avg_daily_uploads = round(sum(daily_uploads) / len(daily_uploads), 1) if daily_uploads else 0
        
        # Recent activities (last 20)
        recent_activities = db.session.query(
            AuditLog.created_at,
            User.username,
            AuditLog.action,
            AuditLog.entity_type
        ).outerjoin(User, AuditLog.user_id == User.id).order_by(AuditLog.created_at.desc()).limit(20).all()
        
        return render_template('admin/reports_usage.html',
                             total_uploads=total_uploads,
                             active_users=active_users,
                             model_switches=model_switches,
                             avg_session_time=avg_session_time,
                             total_sessions=total_sessions,
                             trend_dates=trend_dates,
                             daily_uploads=daily_uploads,
                             model_usage_stats=model_usage_stats,
                             model_usage_labels=model_usage_labels,
                             model_usage_data=model_usage_data,
                             hourly_labels=hourly_labels,
                             hourly_data=hourly_data,
                             peak_hour=peak_hour,
                             activity_dates=activity_dates,
                             login_events=login_events,
                             busiest_day=busiest_day,
                             avg_daily_uploads=avg_daily_uploads,
                             recent_activities=recent_activities)
        
    except Exception as e:
        flash(f'Error loading usage reports: {str(e)}', 'error')
        return redirect(url_for('admin_reports'))

@app.route('/admin/reports/quality')
@login_required
@admin_required
def admin_reports_quality():
    """Data Quality & Validation Reports"""
    try:
        # Basic statistics
        total_records = RoosterRecord.query.count()
        validated_records = RoosterRecord.query.filter(RoosterRecord.is_correct.isnot(None)).count()
        pending_validation = total_records - validated_records
        
        # Calculate data quality score (combination of validation rate and accuracy)
        validation_rate = round((validated_records / total_records * 100), 1) if total_records > 0 else 0
        accuracy_rate = round((RoosterRecord.query.filter_by(is_correct=1).count() / validated_records * 100), 1) if validated_records > 0 else 0
        data_quality_score = round((validation_rate + accuracy_rate) / 2, 1)
        
        # Validation status distribution
        validation_labels = ['Correct', 'Incorrect', 'Pending']
        correct_count = RoosterRecord.query.filter_by(is_correct=1).count()
        incorrect_count = RoosterRecord.query.filter_by(is_correct=0).count()
        validation_data = [correct_count, incorrect_count, pending_validation]
        
        # Confidence vs Accuracy correlation
        scatter_data = []
        records_with_validation = RoosterRecord.query.filter(RoosterRecord.is_correct.isnot(None)).all()
        
        for record in records_with_validation:
            scatter_data.append({
                'x': float(record.confidence_score),
                'y': 1 if record.is_correct == 1 else 0
            })
        
        # Calculate correlation coefficient (simplified)
        if len(scatter_data) > 1:
            import statistics
            x_values = [point['x'] for point in scatter_data]
            y_values = [point['y'] for point in scatter_data]
            try:
                correlation_coefficient = round(statistics.correlation(x_values, y_values), 3)
            except:
                correlation_coefficient = 0.0
        else:
            correlation_coefficient = 0.0
        
        # Feedback analysis
        feedback_labels = ['Correct', 'Incorrect', 'No Feedback']
        feedback_data = [
            RoosterRecord.query.filter_by(is_correct=1).count(),
            RoosterRecord.query.filter_by(is_correct=0).count(),
            RoosterRecord.query.filter(RoosterRecord.is_correct.is_(None)).count()
        ]
        
        # Common misclassifications (predicted vs actual breed)
        misclassification_data = []
        misclassification_labels = []
        
        # Get records where prediction was wrong
        incorrect_records = RoosterRecord.query.filter_by(is_correct=0).all()
        misclassification_counts = {}
        
        for record in incorrect_records:
            if record.breed_id:
                # Get the actual breed name
                actual_breed = Breed.query.get(record.breed_id)
                if actual_breed:
                    key = f"{record.predicted_category} → {actual_breed.name}"
                    misclassification_counts[key] = misclassification_counts.get(key, 0) + 1
        
        # Sort by frequency and take top 5
        sorted_misclassifications = sorted(misclassification_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for misclass, count in sorted_misclassifications:
            misclassification_labels.append(misclass)
            misclassification_data.append(count)
        
        # Edge cases analysis (low confidence but correct, high confidence but wrong)
        edge_cases = []
        
        # Low confidence but correct
        low_conf_correct = RoosterRecord.query.filter(
            RoosterRecord.confidence_score < 0.5,
            RoosterRecord.is_correct == 1
        ).limit(5).all()
        
        # High confidence but wrong
        high_conf_wrong = RoosterRecord.query.filter(
            RoosterRecord.confidence_score > 0.8,
            RoosterRecord.is_correct == 0
        ).limit(5).all()
        
        edge_cases.extend(low_conf_correct)
        edge_cases.extend(high_conf_wrong)
        
        # Data distribution analysis
        data_distribution = []
        breeds = Breed.query.all()
        
        for breed in breeds:
            predicted_count = RoosterRecord.query.filter_by(predicted_category=breed.name).count()
            actual_count = RoosterRecord.query.filter_by(breed_id=breed.id).count()
            
            # Calculate balance (how well distributed the data is)
            total_for_category = max(predicted_count, actual_count, 1)
            balance = min(predicted_count, actual_count) / total_for_category
            
            # Calculate quality (accuracy for this breed)
            correct_for_breed = RoosterRecord.query.filter_by(breed_id=breed.id, is_correct=1).count()
            quality = correct_for_breed / actual_count if actual_count > 0 else 0
            
            data_distribution.append({
                'category': breed.name,
                'predicted_count': predicted_count,
                'actual_count': actual_count,
                'balance': balance,
                'quality': quality
            })
        
        # Additional quality metrics
        avg_confidence_validated = 0
        if validated_records > 0:
            avg_conf_result = db.session.query(db.func.avg(RoosterRecord.confidence_score)).filter(
                RoosterRecord.is_correct.isnot(None)
            ).scalar()
            avg_confidence_validated = round((avg_conf_result * 100), 1) if avg_conf_result else 0
        
        low_confidence_count = RoosterRecord.query.filter(RoosterRecord.confidence_score < 0.5).count()
        
        return render_template('admin/reports_quality.html',
                             total_records=total_records,
                             validated_records=validated_records,
                             pending_validation=pending_validation,
                             data_quality_score=data_quality_score,
                             validation_labels=validation_labels,
                             validation_data=validation_data,
                             scatter_data=scatter_data,
                             correlation_coefficient=correlation_coefficient,
                             feedback_labels=feedback_labels,
                             feedback_data=feedback_data,
                             misclassification_labels=misclassification_labels,
                             misclassification_data=misclassification_data,
                             edge_cases=edge_cases,
                             data_distribution=data_distribution,
                             validation_rate=validation_rate,
                             avg_confidence_validated=avg_confidence_validated,
                             low_confidence_count=low_confidence_count)
        
    except Exception as e:
        flash(f'Error loading quality reports: {str(e)}', 'error')
        return redirect(url_for('admin_reports'))

@app.route('/admin/reports/business')
@login_required
@admin_required
def admin_reports_business():
    """Business Intelligence Reports"""
    try:
        # Get breed popularity data
        breed_stats = db.session.query(
            RoosterRecord.predicted_category,
            db.func.count(RoosterRecord.id).label('count')
        ).filter(RoosterRecord.predicted_category.isnot(None)).group_by(RoosterRecord.predicted_category).all()
        
        # Find most popular breed
        most_popular_breed = max(breed_stats, key=lambda x: x.count).predicted_category if breed_stats else "N/A"
        total_breed_categories = len(breed_stats)
        
        # Calculate growth rate (simplified - comparing last 7 days vs previous 7 days)
        from datetime import datetime, timedelta
        current_week = RoosterRecord.query.filter(
            RoosterRecord.uploaded_at >= datetime.now() - timedelta(days=7)
        ).count()
        
        previous_week = RoosterRecord.query.filter(
            RoosterRecord.uploaded_at >= datetime.now() - timedelta(days=14),
            RoosterRecord.uploaded_at < datetime.now() - timedelta(days=7)
        ).count()
        
        growth_rate = round(((current_week - previous_week) / previous_week * 100), 1) if previous_week > 0 else 0
        
        # Calculate market share of most popular breed
        total_predictions = RoosterRecord.query.count()
        most_popular_count = max(breed_stats, key=lambda x: x.count).count if breed_stats else 0
        market_share = round((most_popular_count / total_predictions * 100), 1) if total_predictions > 0 else 0
        
        # Prepare data for charts
        breed_names = [stat.predicted_category for stat in breed_stats]
        breed_counts = [stat.count for stat in breed_stats]
        
        # Breed trends (last 7 days)
        trend_dates = []
        breed_trends = []
        
        # Get trend data for each breed
        colors = ['#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff']
        
        for i, breed_name in enumerate(breed_names[:5]):  # Limit to top 5 breeds
            trend_data = []
            for j in range(7):
                date = datetime.now() - timedelta(days=j)
                day_count = RoosterRecord.query.filter(
                    db.func.date(RoosterRecord.uploaded_at) == date.date(),
                    RoosterRecord.predicted_category == breed_name
                ).count()
                trend_data.append(day_count)
            
            trend_data.reverse()  # Show chronological order
            breed_trends.append({
                'name': breed_name,
                'data': trend_data,
                'color': colors[i % len(colors)]
            })
        
        # Generate trend dates
        for i in range(7):
            date = datetime.now() - timedelta(days=6-i)
            trend_dates.append(date.strftime('%m/%d'))
        
        # Seasonal analysis (last 12 months)
        seasonal_labels = []
        seasonal_data = []
        
        for i in range(12):
            month_start = datetime.now() - timedelta(days=30*i)
            month_end = month_start + timedelta(days=30)
            
            month_count = RoosterRecord.query.filter(
                RoosterRecord.uploaded_at >= month_start,
                RoosterRecord.uploaded_at < month_end
            ).count()
            
            seasonal_labels.append(month_start.strftime('%b'))
            seasonal_data.append(month_count)
        
        seasonal_labels.reverse()
        seasonal_data.reverse()
        
        # Breed rankings with performance metrics
        breed_rankings = []
        for stat in breed_stats:
            # Calculate accuracy for this breed
            breed_records = RoosterRecord.query.filter_by(predicted_category=stat.predicted_category)
            correct_count = breed_records.filter_by(is_correct=1).count()
            total_validated = breed_records.filter(RoosterRecord.is_correct.isnot(None)).count()
            accuracy = round((correct_count / total_validated * 100), 1) if total_validated > 0 else 0
            
            # Calculate popularity score (percentage of total predictions)
            popularity = round((stat.count / total_predictions * 100), 1) if total_predictions > 0 else 0
            
            breed_rankings.append({
                'name': stat.predicted_category,
                'prediction_count': stat.count,
                'accuracy': accuracy,
                'popularity': popularity
            })
        
        # Sort by popularity
        breed_rankings.sort(key=lambda x: x['popularity'], reverse=True)
        
        # Performance vs Popularity scatter data
        scatter_data = []
        for breed in breed_rankings:
            scatter_data.append({
                'x': breed['popularity'],
                'y': breed['accuracy']
            })
        
        # Market insights
        market_insights = [
            {
                'metric': 'Market Leader',
                'value': most_popular_breed,
                'trend': 'stable',
                'insight': f'Dominates {market_share}% of predictions'
            },
            {
                'metric': 'Growth Rate',
                'value': f'{growth_rate}%',
                'trend': 'up' if growth_rate > 0 else 'down',
                'insight': 'Weekly growth in prediction volume'
            },
            {
                'metric': 'Market Diversity',
                'value': f'{total_breed_categories} breeds',
                'trend': 'stable',
                'insight': 'Number of distinct breed categories'
            },
            {
                'metric': 'Avg Accuracy',
                'value': f'{round(sum(b["accuracy"] for b in breed_rankings) / len(breed_rankings), 1)}%',
                'trend': 'stable',
                'insight': 'Overall prediction accuracy across breeds'
            }
        ]
        
        # Strategic recommendations
        strategic_recommendations = []
        
        # High priority: Low accuracy breeds
        low_accuracy_breeds = [b for b in breed_rankings if b['accuracy'] < 70]
        if low_accuracy_breeds:
            strategic_recommendations.append({
                'title': 'Improve Low-Accuracy Breeds',
                'description': f'Focus training on {", ".join([b["name"] for b in low_accuracy_breeds[:2]])} to improve overall accuracy.',
                'priority': 'high'
            })
        
        # Medium priority: Market expansion
        if market_share > 50:
            strategic_recommendations.append({
                'title': 'Diversify Market Focus',
                'description': f'{most_popular_breed} dominates the market. Consider promoting other breeds.',
                'priority': 'medium'
            })
        
        # Low priority: Growth opportunities
        if growth_rate < 10:
            strategic_recommendations.append({
                'title': 'Accelerate Growth',
                'description': 'Current growth rate is moderate. Consider marketing initiatives to increase usage.',
                'priority': 'low'
            })
        
        return render_template('admin/reports_business.html',
                             most_popular_breed=most_popular_breed,
                             total_breed_categories=total_breed_categories,
                             growth_rate=growth_rate,
                             market_share=market_share,
                             breed_names=breed_names,
                             breed_counts=breed_counts,
                             trend_dates=trend_dates,
                             breed_trends=breed_trends,
                             seasonal_labels=seasonal_labels,
                             seasonal_data=seasonal_data,
                             breed_rankings=breed_rankings,
                             scatter_data=scatter_data,
                             market_insights=market_insights,
                             strategic_recommendations=strategic_recommendations)
        
    except Exception as e:
        flash(f'Error loading business intelligence reports: {str(e)}', 'error')
        return redirect(url_for('admin_reports'))


# ---------- Admin Model Training ----------
@app.route('/admin/teachable-machine/upload', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_teachable_machine_upload():
    """Upload Teachable Machine model"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'model_file' not in request.files:
                flash('No model file selected', 'error')
                return redirect(request.url)
            
            file = request.files['model_file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            # Check if it's a zip file
            if not file.filename.lower().endswith('.zip'):
                flash('Please upload a ZIP file from Teachable Machine', 'error')
                return redirect(request.url)
            
            # Get form data
            model_name = request.form.get('model_name', '').strip()
            description = request.form.get('description', '').strip()
            accuracy_score = request.form.get('accuracy_score', '').strip()
            
            if not model_name:
                flash('Model name is required', 'error')
                return redirect(request.url)
            
            # Convert accuracy to float
            try:
                accuracy = float(accuracy_score) if accuracy_score else None
            except ValueError:
                flash('Invalid accuracy score', 'error')
                return redirect(request.url)
            
            # Get the Teachable Machine model source ID
            tm_model_source = ModelSource.query.filter_by(model_source='Teachable Machine').first()
            if not tm_model_source:
                return jsonify({'success': False, 'message': 'Teachable Machine model source not found in database'})
            
            # Create training history record
            training_record = TrainingHistory(
                training_type='teachable_machine',
                model_source_id=tm_model_source.id,
                description=description or f'Teachable Machine model: {model_name}',
                accuracy_score=accuracy,
                status='pending',
                model_location='teachable_machine_models/temp',
                created_by=current_user.id,
                started_at=datetime.utcnow()
            )
            db.session.add(training_record)
            db.session.commit()
            training_id = training_record.id
            
            # Create directory for Teachable Machine models
            tm_dir = 'teachable_machine_models'
            os.makedirs(tm_dir, exist_ok=True)
            
            # Save uploaded file directly to teachable_machine_models directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(tm_dir, filename)
            file.save(file_path)
            
            # Extract and process the Teachable Machine model
            success = process_teachable_machine_model(file_path, tm_dir, training_id)
            
            if success:
                # Update training record
                training_record.status = 'completed'
                training_record.completed_at = datetime.utcnow()
                db.session.commit()
                
                flash(f'✅ Teachable Machine model "{model_name}" uploaded and processed successfully!<br><small class="text-muted">Status: Ready for use | Training ID: #{training_id}</small>', 'success')
                return redirect(url_for('admin_teachable_machine_upload'))
            else:
                # Mark as failed
                training_record.status = 'failed'
                db.session.commit()
                flash('❌ Failed to process Teachable Machine model.<br><small class="text-muted">Please check that you uploaded a valid SavedModel ZIP file from Teachable Machine.</small>', 'error')
                return redirect(request.url)
                
        except Exception as e:
            flash(f'❌ Error uploading Teachable Machine model: {str(e)}<br><small class="text-muted">Please try again or contact support if the problem persists.</small>', 'error')
            return redirect(request.url)
    
    return render_template('admin/teachable_machine_upload.html')

def process_teachable_machine_model(zip_path, output_dir, training_id):
    """Process uploaded Teachable Machine model.
    Supports Teachable Machine SavedModel ZIP:
      labels.txt
      model.savedmodel/ (contains saved_model.pb and variables/)
    Normalizes into our layout:
      teachable_machine_models/saved_model/{saved_model.pb, variables/}
      teachable_machine_models/labels.txt
      teachable_machine_models/class_mapping.json
    Returns True on success, False otherwise.
    """
    try:
        import zipfile
        import json
        import shutil
        
        # Create a unique temp extraction folder
        temp_extract = os.path.join(output_dir, f"_tm_extract_{training_id}")
        os.makedirs(temp_extract, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)

        # Detect SavedModel layout
        labels_src = os.path.join(temp_extract, 'labels.txt')
        model_savedmodel_src = os.path.join(temp_extract, 'model.savedmodel')
        saved_model_pb = os.path.join(model_savedmodel_src, 'saved_model.pb')
        variables_dir = os.path.join(model_savedmodel_src, 'variables')

        # Destination paths
        tm_root = 'teachable_machine_models'
        dest_saved_model_dir = os.path.join(tm_root, 'saved_model')
        os.makedirs(tm_root, exist_ok=True)

        if os.path.exists(saved_model_pb) and os.path.isdir(variables_dir):
            # Replace existing SavedModel safely
            if os.path.isdir(dest_saved_model_dir):
                try:
                    shutil.rmtree(dest_saved_model_dir)
                except Exception as e:
                    print(f"⚠️ Could not remove existing saved_model: {e}")
            os.makedirs(dest_saved_model_dir, exist_ok=True)

            # Copy SavedModel contents
            shutil.copy2(saved_model_pb, os.path.join(dest_saved_model_dir, 'saved_model.pb'))
            shutil.copytree(variables_dir, os.path.join(dest_saved_model_dir, 'variables'))

            # Copy labels.txt if present
            labels = []
            if os.path.exists(labels_src):
                try:
                    shutil.copy2(labels_src, os.path.join(tm_root, 'labels.txt'))
                    with open(labels_src, 'r', encoding='utf-8') as lf:
                        labels = [line.strip() for line in lf if line.strip()]
                except Exception as e:
                    print(f"⚠️ Could not process labels.txt: {e}")

            # Build class_mapping.json from labels (normalize)
            class_mapping = {}
            if labels:
                for i, label in enumerate(labels):
                    class_mapping[str(i)] = format_category_name(label)
            else:
                # Fallback default ordering
                class_mapping = {
                    '0': 'Bantam',
                    '1': 'Dual Purpose',
                    '2': 'Gamefowl',
                    '3': 'Other'
                }
            with open(os.path.join(tm_root, 'class_mapping.json'), 'w', encoding='utf-8') as f:
                json.dump(class_mapping, f)

            print("✅ Teachable Machine SavedModel normalized successfully")
            # Cleanup temp and zip
            try:
                shutil.rmtree(temp_extract)
                os.remove(zip_path)
            except Exception:
                pass
            return True

        # If not SavedModel, reject with clear message (we require SavedModel)
        print("❌ Teachable Machine ZIP does not contain model.savedmodel/saved_model.pb")
        try:
            shutil.rmtree(temp_extract)
        except Exception:
            pass
        return False
        
    except Exception as e:
        print(f"❌ Error processing Teachable Machine model: {e}")
        return False


# ---------- Admin Google Colab Upload ----------
@app.route('/admin/colab/upload', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_colab_upload():
    """Upload Google Colab model files (ZIP with SavedModel or individual .h5/.keras + .json)"""
    if request.method == 'POST':
        model_name = request.form.get('model_name', '').strip()
        description = request.form.get('description', '').strip()
        accuracy = request.form.get('accuracy', '').strip()
        
        if not model_name:
            return jsonify({'success': False, 'message': 'Model name is required'})
        
        try:
            accuracy_score = float(accuracy) if accuracy else None
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid accuracy score'})
        
        # Check if ZIP file is uploaded (preferred method)
        if 'zip_file' in request.files and request.files['zip_file'].filename != '':
            zip_file = request.files['zip_file']
            if not zip_file.filename.endswith('.zip'):
                return jsonify({'success': False, 'message': 'ZIP file must have .zip extension'})
            
            return process_google_colab_zip(zip_file, model_name, description, accuracy_score)
        
        # Fallback: individual files
        elif 'model_file' in request.files and 'mapping_file' in request.files:
            model_file = request.files['model_file']
            mapping_file = request.files['mapping_file']
            
            if model_file.filename == '' or mapping_file.filename == '':
                return jsonify({'success': False, 'message': 'No files selected'})
            
            # Validate file extensions
            if not (model_file.filename.endswith('.h5') or model_file.filename.endswith('.keras')):
                return jsonify({'success': False, 'message': 'Model file must be .h5 or .keras'})
            
            if not mapping_file.filename.endswith('.json'):
                return jsonify({'success': False, 'message': 'Mapping file must be .json'})
            
            return process_google_colab_files(model_file, mapping_file, model_name, description, accuracy_score)
        
        else:
            return jsonify({'success': False, 'message': 'Please upload either a ZIP file or both model and mapping files'})
    
    return render_template('admin/colab_upload.html')

def process_google_colab_zip(zip_file, model_name, description, accuracy_score):
    """Process Google Colab ZIP file and extract to proper structure"""
    try:
        import zipfile
        import shutil
        
        # Create google_colab_models directory if it doesn't exist
        colab_dir = 'google_colab_models'
        os.makedirs(colab_dir, exist_ok=True)
        
        # Save ZIP temporarily
        zip_filename = secure_filename(zip_file.filename)
        temp_zip_path = os.path.join(colab_dir, f'temp_{zip_filename}')
        zip_file.save(temp_zip_path)
        
        # Extract ZIP
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(colab_dir)
        
        # Remove temp ZIP
        os.remove(temp_zip_path)
        
        # Look for the nested structure: rooster_export/saved_model/
        nested_saved_model = os.path.join(colab_dir, 'rooster_export', 'saved_model')
        target_saved_model = os.path.join(colab_dir, 'saved_model')
        
        if os.path.exists(nested_saved_model):
            # Move nested saved_model to root level
            if os.path.exists(target_saved_model):
                shutil.rmtree(target_saved_model)
            shutil.move(nested_saved_model, target_saved_model)
            print(f"✅ Moved SavedModel from {nested_saved_model} to {target_saved_model}")
        
        # Look for class_mapping.json in rooster_export or root
        mapping_sources = [
            os.path.join(colab_dir, 'rooster_export', 'class_mapping.json'),
            os.path.join(colab_dir, 'class_mapping.json')
        ]
        
        target_mapping = os.path.join(colab_dir, 'class_mapping.json')
        for source in mapping_sources:
            if os.path.exists(source):
                if source != target_mapping:
                    if os.path.exists(target_mapping):
                        os.remove(target_mapping)
                    shutil.move(source, target_mapping)
                print(f"✅ Found class_mapping.json at {source}")
                break
        
        # Look for .keras file in rooster_export or root
        keras_sources = [
            os.path.join(colab_dir, 'rooster_export', 'rooster_model.keras'),
            os.path.join(colab_dir, 'rooster_model.keras')
        ]
        
        target_keras = os.path.join(colab_dir, 'rooster_model.keras')
        for source in keras_sources:
            if os.path.exists(source):
                if source != target_keras:
                    if os.path.exists(target_keras):
                        os.remove(target_keras)
                    shutil.move(source, target_keras)
                print(f"✅ Found rooster_model.keras at {source}")
                break
        
        # Clean up rooster_export directory if it exists
        rooster_export_dir = os.path.join(colab_dir, 'rooster_export')
        if os.path.exists(rooster_export_dir):
            shutil.rmtree(rooster_export_dir)
            print(f"✅ Cleaned up {rooster_export_dir}")
        
        # Get the Google Colab model source ID
        colab_model_source = ModelSource.query.filter_by(model_source='Google Colab').first()
        if not colab_model_source:
            return jsonify({'success': False, 'message': 'Google Colab model source not found in database'})
        
        # Create training history record
        training_record = TrainingHistory(
            training_type='google_colab',
            model_source_id=colab_model_source.id,
            description=description or f'Google Colab model: {model_name}',
            accuracy_score=accuracy_score,
            status='completed',
            model_location=os.path.join(colab_dir, 'saved_model'),
            created_by=current_user.id,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        db.session.add(training_record)
        db.session.commit()
        
        return jsonify({'success': True, 'message': f'Google Colab model "{model_name}" uploaded and processed successfully!'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error processing ZIP file: {str(e)}'})

def process_google_colab_files(model_file, mapping_file, model_name, description, accuracy_score):
    """Process individual Google Colab files (.h5/.keras + .json)"""
    try:
        # Create google_colab_models directory if it doesn't exist
        colab_dir = 'google_colab_models'
        os.makedirs(colab_dir, exist_ok=True)
        
        # Save files
        model_filename = secure_filename(model_file.filename)
        mapping_filename = secure_filename(mapping_file.filename)
        
        model_path = os.path.join(colab_dir, model_filename)
        mapping_path = os.path.join(colab_dir, mapping_filename)
        
        model_file.save(model_path)
        mapping_file.save(mapping_path)
        
        # Rename files for consistency
        if model_filename.endswith('.h5') and model_filename != 'rooster_model.h5':
            new_model_path = os.path.join(colab_dir, 'rooster_model.h5')
            if os.path.exists(new_model_path):
                os.remove(new_model_path)
            os.rename(model_path, new_model_path)
            model_path = new_model_path
        
        if model_filename.endswith('.keras') and model_filename != 'rooster_model.keras':
            new_model_path = os.path.join(colab_dir, 'rooster_model.keras')
            if os.path.exists(new_model_path):
                os.remove(new_model_path)
            os.rename(model_path, new_model_path)
            model_path = new_model_path
        
        if mapping_filename != 'class_mapping.json':
            new_mapping_path = os.path.join(colab_dir, 'class_mapping.json')
            if os.path.exists(new_mapping_path):
                os.remove(new_mapping_path)
            os.rename(mapping_path, new_mapping_path)
            mapping_path = new_mapping_path
        
        # Get the Google Colab model source ID
        colab_model_source = ModelSource.query.filter_by(model_source='Google Colab').first()
        if not colab_model_source:
            return jsonify({'success': False, 'message': 'Google Colab model source not found in database'})
        
        # Create training history record
        training_record = TrainingHistory(
            training_type='google_colab',
            model_source_id=colab_model_source.id,
            description=description or f'Google Colab model: {model_name}',
            accuracy_score=accuracy_score,
            status='completed',
            model_location=model_path,
            created_by=current_user.id,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        db.session.add(training_record)
        db.session.commit()
        
        return jsonify({'success': True, 'message': f'Google Colab model "{model_name}" uploaded successfully!'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Error saving model files: {str(e)}'})

@app.route('/admin/train-model')
@login_required
@admin_required
def admin_train_model():
    """Model training interface"""
    return render_template('admin/train_model.html')

@app.route('/admin/train-model/start', methods=['POST'])
@login_required
@admin_required
def admin_train_model_start():
    """Start model training"""
    try:
        # Check if dataset exists
        dataset_path = 'dataset'
        if not os.path.exists(dataset_path):
            return jsonify({'success': False, 'message': 'Dataset folder not found! Please ensure the dataset folder exists.'})
        
        # Start training in background
        import threading
        print("🧵 Starting training thread...")
        # Pass the initiating user's id into the background thread since
        # request context (and current_user) won't be available there
        initiator_user_id = current_user.id if current_user.is_authenticated else None
        training_thread = threading.Thread(target=run_training_process, args=(initiator_user_id,))
        training_thread.daemon = True
        training_thread.start()
        print("✅ Training thread started successfully!")
        
        return jsonify({'success': True, 'message': 'Model training started!'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error starting training: {str(e)}'})

@app.route('/admin/train-model/accept', methods=['POST'])
@login_required
@admin_required
def admin_train_model_accept():
    """Accept the trained model and move it to the main location"""
    try:
        temp_model_path = 'local_model/temp'
        main_model_path = 'local_model'
        
        # Check if temp model exists
        if not os.path.exists(temp_model_path):
            return jsonify({'success': False, 'message': 'No trained model found to accept'})
        
        # Read training ID from completion file
        completion_file = os.path.join('local_model', 'training_completed.txt')
        training_id = None
        if os.path.exists(completion_file):
            with open(completion_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('Training ID:'):
                    training_id = int(line.split(':')[1].strip())
                    break
        
        # Move model files from temp to main location
        import shutil
        
        # Remove existing model files
        for file in ['rooster_model.h5', 'class_mapping.json']:
            existing_file = os.path.join(main_model_path, file)
            if os.path.exists(existing_file):
                os.remove(existing_file)
        
        # Remove existing SavedModel directory if it exists
        if os.path.exists(main_model_path) and os.path.isdir(main_model_path):
            for item in os.listdir(main_model_path):
                item_path = os.path.join(main_model_path, item)
                if os.path.isdir(item_path) and item not in ['temp']:
                    shutil.rmtree(item_path)
        
        # Copy temp files to main location
        for item in os.listdir(temp_model_path):
            src = os.path.join(temp_model_path, item)
            dst = os.path.join(main_model_path, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # Update training history record to accepted
        if training_id:
            training_record = TrainingHistory.query.get(training_id)
            if training_record:
                training_record.status = 'accepted'
                training_record.accepted_at = datetime.utcnow()
                training_record.model_location = main_model_path
                db.session.commit()
                
                # Update model_source with new accuracy score
                local_model = ModelSource.query.filter_by(model_source='Local Training').first()
                if local_model:
                    local_model.accuracy_score = training_record.accuracy_score
                    local_model.updated_at = datetime.utcnow()
                    db.session.commit()
                    
                    # Log training acceptance
                    log_audit_event('ACCEPT_MODEL', 'model', local_model.id, {
                        'training_id': training_id,
                        'accuracy_score': float(training_record.accuracy_score),
                        'model_source_id': training_record.model_source_id
                    })
        
        # Update completion file status
        if os.path.exists(completion_file):
            with open(completion_file, 'r') as f:
                lines = f.readlines()
            
            # Update status line
            updated_lines = []
            for line in lines:
                if line.startswith('Status:'):
                    updated_lines.append('Status: accepted\n')
                else:
                    updated_lines.append(line)
            
            with open(completion_file, 'w') as f:
                f.writelines(updated_lines)
        
        # Clean up temporary files after successful acceptance
        try:
            # Remove temp directory
            if os.path.exists(temp_model_path):
                shutil.rmtree(temp_model_path)
                print(f"🗑️ Cleaned up temp directory: {temp_model_path}")
            
            # Remove completion file
            if os.path.exists(completion_file):
                os.remove(completion_file)
                print(f"🗑️ Cleaned up completion file: {completion_file}")
            
            # Remove progress file if it exists
            progress_file = os.path.join('local_model', 'training_progress.txt')
            if os.path.exists(progress_file):
                os.remove(progress_file)
                print(f"🗑️ Cleaned up progress file: {progress_file}")
                
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Error during cleanup: {cleanup_error}")
        
        return jsonify({'success': True, 'message': 'Model accepted and activated successfully!'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error accepting model: {str(e)}'})

@app.route('/admin/train-model/reject', methods=['POST'])
def admin_train_model_reject():
    """Reject the trained model and clean up temp files"""
    try:
        temp_model_path = 'local_model/temp'
        
        # Read training ID from completion file
        completion_file = os.path.join('local_model', 'training_completed.txt')
        training_id = None
        if os.path.exists(completion_file):
            with open(completion_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith('Training ID:'):
                    training_id = int(line.split(':')[1].strip())
                    break
        
        # Remove temp model files
        if os.path.exists(temp_model_path):
            import shutil
            shutil.rmtree(temp_model_path)
        
        # Update training history record to rejected
        if training_id:
            training_record = TrainingHistory.query.get(training_id)
            if training_record:
                training_record.status = 'rejected'
                training_record.rejected_at = datetime.utcnow()
                db.session.commit()
                
                # Log training rejection
                log_audit_event('REJECT_MODEL', 'model', None, {
                    'training_id': training_id,
                    'model_source_id': training_record.model_source_id,
                    'accuracy_score': float(training_record.accuracy_score) if training_record.accuracy_score else None
                })
        
        # Update completion file status
        if os.path.exists(completion_file):
            with open(completion_file, 'r') as f:
                lines = f.readlines()
            
            # Update status line
            updated_lines = []
            for line in lines:
                if line.startswith('Status:'):
                    updated_lines.append('Status: rejected\n')
                else:
                    updated_lines.append(line)
            
            with open(completion_file, 'w') as f:
                f.writelines(updated_lines)
        
        # Clean up temporary files after rejection
        try:
            # Remove temp directory
            if os.path.exists(temp_model_path):
                shutil.rmtree(temp_model_path)
                print(f"🗑️ Cleaned up temp directory: {temp_model_path}")
            
            # Remove completion file
            if os.path.exists(completion_file):
                os.remove(completion_file)
                print(f"🗑️ Cleaned up completion file: {completion_file}")
            
            # Remove progress file if it exists
            progress_file = os.path.join('local_model', 'training_progress.txt')
            if os.path.exists(progress_file):
                os.remove(progress_file)
                print(f"🗑️ Cleaned up progress file: {progress_file}")
                
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Error during cleanup: {cleanup_error}")
        
        return jsonify({'success': True, 'message': 'Model rejected and cleaned up successfully!'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error rejecting model: {str(e)}'})

@app.route('/admin/train-model/history')
def admin_train_model_history():
    """Get training history for the training page - LOCAL TRAINING ONLY"""
    try:
        # Get recent LOCAL training history only (last 10 records)
        training_history = TrainingHistory.query.filter_by(training_type='local').order_by(TrainingHistory.created_at.desc()).limit(10).all()
        
        history_data = []
        for record in training_history:
            history_data.append({
                'id': record.id,
                'training_type': record.training_type,
                'model_source': record.model_source.model_source if record.model_source else 'Unknown',
                'description': record.description,
                'accuracy_score': float(record.accuracy_score) if record.accuracy_score else None,
                'accuracy_percentage': f"{float(record.accuracy_score) * 100:.2f}%" if record.accuracy_score else None,
                'status': record.status,
                'started_at': record.started_at.isoformat() if record.started_at else None,
                'completed_at': record.completed_at.isoformat() if record.completed_at else None,
                'accepted_at': record.accepted_at.isoformat() if record.accepted_at else None,
                'rejected_at': record.rejected_at.isoformat() if record.rejected_at else None,
                'created_at': record.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'history': history_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting training history: {str(e)}'
        })

@app.route('/admin/model/reload', methods=['POST'])
@login_required
@admin_required
def admin_model_reload():
    """Reload the active model without restarting the app"""
    global rooster_model
    try:
        # Clear the cached model
        rooster_model = None
        
        # Reload the model based on current database setting
        active_model = get_active_model_source()
        if active_model:
            print(f"🔄 Reloading model: {active_model.model_source}")
            
            # Load model based on active model source
            if active_model.model_source == 'Local Training':
                model_loaded = load_local_model()
            elif active_model.model_source == 'Google Colab':
                model_loaded = load_google_colab_model()
            elif active_model.model_source == 'Teachable Machine':
                model_loaded = load_teachable_machine_model()
            else:
                model_loaded = load_local_model()
            
            if model_loaded:
                return jsonify({'success': True, 'message': f'Model reloaded: {active_model.model_source}'})
            else:
                return jsonify({'success': False, 'message': 'Failed to reload model'})
        else:
            return jsonify({'success': False, 'message': 'No active model found'})
            
    except Exception as e:
        print(f"❌ Error reloading model: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/active-model')
def api_active_model():
    """API endpoint to get active model information"""
    try:
        active_model = get_active_model_source()
        
        if active_model:
            return jsonify({
                'success': True,
                'model': {
                    'model_source': active_model.model_source,
                    'model_id': active_model.model_id,
                    'description': active_model.description,
                    'accuracy_score': float(active_model.accuracy_score) if active_model.accuracy_score else None,
                    'is_active': active_model.is_active,
                    'created_at': active_model.created_at.isoformat() if active_model.created_at else None,
                    'updated_at': active_model.updated_at.isoformat() if active_model.updated_at else None
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No active model found'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting active model: {str(e)}'
        })

@app.route('/admin/teachable-machine/history')
def admin_teachable_machine_history():
    """Get training history for Teachable Machine page - TEACHABLE MACHINE ONLY"""
    try:
        # Get recent TEACHABLE MACHINE training history only (last 10 records)
        training_history = TrainingHistory.query.filter_by(training_type='teachable_machine').order_by(TrainingHistory.created_at.desc()).limit(10).all()
        
        history_data = []
        for record in training_history:
            history_data.append({
                'id': record.id,
                'training_type': record.training_type,
                'model_source': record.model_source.model_source if record.model_source else 'Unknown',
                'status': record.status,
                'accuracy_score': float(record.accuracy_score) if record.accuracy_score else None,
                'accuracy_percentage': f"{float(record.accuracy_score) * 100:.2f}%" if record.accuracy_score else None,
                'started_at': record.started_at.isoformat() if record.started_at else None,
                'completed_at': record.completed_at.isoformat() if record.completed_at else None,
                'accepted_at': record.accepted_at.isoformat() if record.accepted_at else None,
                'rejected_at': record.rejected_at.isoformat() if record.rejected_at else None,
                'created_at': record.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'history': history_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting Teachable Machine history: {str(e)}'
        })

@app.route('/admin/colab/history')
def admin_colab_history():
    """Get training history for Google Colab page - GOOGLE COLAB ONLY"""
    try:
        # Get Google Colab model source ID
        colab_model_source = ModelSource.query.filter_by(model_source='Google Colab').first()
        if not colab_model_source:
            return jsonify({
                'success': False,
                'message': 'Google Colab model source not found'
            })
        
        # Get recent GOOGLE COLAB training history only (last 10 records)
        training_history = TrainingHistory.query.filter_by(
            model_source_id=colab_model_source.id
        ).order_by(TrainingHistory.created_at.desc()).limit(10).all()
        
        history_data = []
        for record in training_history:
            history_data.append({
                'id': record.id,
                'training_type': record.training_type,
                'model_source': record.model_source.model_source if record.model_source else 'Unknown',
                'status': record.status,
                'accuracy_score': float(record.accuracy_score) if record.accuracy_score else None,
                'accuracy_percentage': f"{float(record.accuracy_score) * 100:.2f}%" if record.accuracy_score else None,
                'started_at': record.started_at.isoformat() if record.started_at else None,
                'completed_at': record.completed_at.isoformat() if record.completed_at else None,
                'accepted_at': record.accepted_at.isoformat() if record.accepted_at else None,
                'rejected_at': record.rejected_at.isoformat() if record.rejected_at else None,
                'created_at': record.created_at.isoformat()
            })
        
        return jsonify({
            'success': True,
            'history': history_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting Google Colab training history: {str(e)}'
        })

@app.route('/admin/train-model/current-accuracy')
def admin_train_model_current_accuracy():
    """Get current active model accuracy score"""
    try:
        active_model = get_active_model_source()
        if active_model and active_model.accuracy_score is not None:
            return jsonify({
                'success': True,
                'accuracy': float(active_model.accuracy_score),
                'accuracy_percentage': f"{float(active_model.accuracy_score) * 100:.2f}%",
                'model_source': active_model.model_source,
                'model_id': active_model.model_id
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No active model with accuracy score found'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting current accuracy: {str(e)}'
        })

@app.route('/admin/train-model/progress')
def admin_train_model_progress():
    """Get training progress - public endpoint for training status"""
    try:
        # Check if training is complete by looking for the trained model files
        local_model_dir = 'local_model'
        model_h5_path = os.path.join(local_model_dir, 'rooster_model.h5')
        class_mapping_path = os.path.join(local_model_dir, 'class_mapping.json')
        completion_file = os.path.join(local_model_dir, 'training_completed.txt')
        
        # Check if training is currently running by looking for progress file
        progress_file = os.path.join(local_model_dir, 'training_progress.txt')
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    lines = f.readlines()
                
                status = 'training'
                progress = 0
                epoch = 0
                accuracy = 0.0
                
                for line in lines:
                    if line.startswith('status:'):
                        status = line.split(':')[1].strip()
                    elif line.startswith('progress:'):
                        progress = int(line.split(':')[1].strip())
                    elif line.startswith('epoch:'):
                        epoch = int(line.split(':')[1].strip())
                    elif line.startswith('accuracy:'):
                        accuracy = float(line.split(':')[1].strip())
                
                return jsonify({
                    'status': status,
                    'progress': progress,
                    'epoch': epoch,
                    'total_epochs': 25,
                    'accuracy': accuracy,
                    'dataset_size': 'Unknown'
                })
            except Exception as e:
                print(f"Error reading progress file: {e}")
        
        # Check if training completion marker exists
        if os.path.exists(completion_file):
            try:
                # Read completion details
                with open(completion_file, 'r') as f:
                    lines = f.readlines()
                
                # Extract details from completion file
                accuracy = 0.64  # default
                dataset_size = 'Unknown'
                status = 'completed'
                
                for line in lines:
                    if line.startswith('Accuracy:'):
                        accuracy = float(line.split(':')[1].strip())
                    elif line.startswith('Dataset size:'):
                        dataset_size = line.split(':')[1].strip()
                    elif line.startswith('Status:'):
                        status = line.split(':')[1].strip()
                
                # Clean up progress file if training is completed
                progress_file = os.path.join(local_model_dir, 'training_progress.txt')
                if os.path.exists(progress_file):
                    try:
                        os.remove(progress_file)
                    except:
                        pass
                
                return jsonify({
                    'status': status,  # 'completed', 'pending_approval', or 'rejected'
                    'progress': 100,
                    'epoch': 25,
                    'total_epochs': 25,
                    'accuracy': accuracy,
                    'dataset_size': dataset_size
                })
            except Exception as e:
                print(f"Error reading completion file: {e}")
        
        # Check if there's an active training session by looking for temp files
        temp_model_path = 'local_model/temp'
        if os.path.exists(temp_model_path):
            return jsonify({
                'status': 'training',
                'progress': 50,  # Assume halfway if temp files exist
                'epoch': 12,
                'total_epochs': 25,
                'accuracy': 0.75
            })
        
        # If no completion file and no temp files, no training is active
        return jsonify({
            'status': 'idle',
            'progress': 0,
            'epoch': 0,
            'total_epochs': 25,
            'accuracy': 0.0,
            'dataset_size': 'Unknown'
        })
    except Exception as e:
        print(f"Error in progress endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error getting progress: {str(e)}'
        })

def update_training_progress(status, progress, epoch, accuracy):
    """Update the training progress file"""
    try:
        progress_file = os.path.join('local_model', 'training_progress.txt')
        with open(progress_file, 'w') as f:
            f.write(f"status: {status}\n")
            f.write(f"progress: {progress}\n")
            f.write(f"epoch: {epoch}\n")
            f.write(f"accuracy: {accuracy}\n")
    except Exception as e:
        print(f"Error updating progress file: {e}")

def run_training_process(created_by_user_id: int | None = None):
    """Run the actual training process

    created_by_user_id: The id of the user who initiated training. This is
    passed from the request handler because Flask's current_user is not
    available in a background thread.
    """
    try:
        print("🚀 Starting model training...")
        
        # Get the Local Training model source ID
        local_model_source = ModelSource.query.filter_by(model_source='Local Training').first()
        if not local_model_source:
            print("❌ Local Training model source not found in database")
            return
        
        # Create training history record
        with app.app_context():
            training_record = TrainingHistory(
                training_type='local',
                model_source_id=local_model_source.id,
                description='MobileNetV2 model trained locally using dataset folder',
                status='training',
                model_location='local_model/temp',
                created_by=created_by_user_id,
                started_at=datetime.utcnow()
            )
            db.session.add(training_record)
            db.session.commit()
            training_id = training_record.id
            print(f"📝 Created training history record: {training_id}")
        
        # Create progress tracking file
        update_training_progress('training', 0, 0, 0.0)
        
        # Import training modules with error handling
        try:
            import tensorflow as tf
            print("✅ TensorFlow imported successfully")
        except ImportError as e:
            print(f"❌ TensorFlow import failed: {e}")
            return
            
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
            from tensorflow.keras.models import Model
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            print("✅ Keras modules imported successfully")
        except ImportError as e:
            print(f"❌ Keras import failed: {e}")
            return
            
        try:
            from sklearn.model_selection import train_test_split
            print("✅ Scikit-learn imported successfully")
        except ImportError as e:
            print(f"❌ Scikit-learn import failed: {e}")
            return
            
        import numpy as np
        import os
        import json
        print("✅ Standard libraries imported successfully")
        
        # Load and prepare dataset
        dataset_path = 'dataset'
        categories = ['bantam', 'dual_purpose', 'gamefowl', 'other']
        
        images = []
        labels = []
        
        print(f"📁 Loading dataset from {dataset_path}")
        for i, category in enumerate(categories):
            category_path = os.path.join(dataset_path, category)
            if os.path.exists(category_path):
                category_images = 0
                for filename in os.listdir(category_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(category_path, filename)
                        try:
                            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                            img_array = tf.keras.preprocessing.image.img_to_array(img)
                            images.append(img_array)
                            labels.append(i)
                            category_images += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {filename}: {e}")
                print(f"  📸 Loaded {category_images} images from {category}")
            else:
                print(f"⚠️ Category folder not found: {category_path}")
        
        if not images:
            print("❌ No images found in dataset!")
            return
        
        print(f"✅ Total images loaded: {len(images)}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Normalize images
        X = X / 255.0
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"📊 Training set: {len(X_train)} images, Test set: {len(X_test)} images")
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        # Create model
        print("🏗️ Creating MobileNetV2 model...")
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"🎯 Starting training with {len(X_train)} training images...")
        
        # Create custom callback for progress tracking
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                progress = int((epoch + 1) / 25 * 100)
                accuracy = logs.get('accuracy', 0.0)
                update_training_progress('training', progress, epoch + 1, accuracy)
                print(f"📊 Epoch {epoch + 1}/25 - Progress: {progress}% - Accuracy: {accuracy:.4f}")
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=25,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[ProgressCallback()]
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Training completed! Test accuracy: {test_accuracy:.4f}")
        
        # Update training history record with completion details
        with app.app_context():
            training_record = TrainingHistory.query.get(training_id)
            if training_record:
                training_record.status = 'completed'
                training_record.accuracy_score = float(test_accuracy)
                training_record.completed_at = datetime.utcnow()
                db.session.commit()
                print(f"📝 Updated training history record: {training_id}")
        
        # Update progress to completed
        update_training_progress('completed', 100, 25, test_accuracy)
        
        # Save model to temporary location (pending user approval)
        temp_model_path = 'local_model/temp'
        os.makedirs(temp_model_path, exist_ok=True)
        
        # Save in SavedModel format (Keras 3 compatible)
        print(f"💾 Saving model to temporary location: {temp_model_path}")
        try:
            # Try Keras 3 format first
            model.save(temp_model_path)
            print("✅ Model saved in SavedModel format to temp location")
        except Exception as e:
            print(f"⚠️ SavedModel format failed: {e}")
            # Fallback to H5 format
            try:
                model.save(os.path.join(temp_model_path, 'rooster_model.h5'))
                print("✅ Model saved in H5 format to temp location")
            except Exception as e2:
                print(f"❌ H5 format also failed: {e2}")
                raise e2
        
        # Save class mapping to temporary location
        class_mapping = {i: category for i, category in enumerate(categories)}
        with open(os.path.join(temp_model_path, 'class_mapping.json'), 'w') as f:
            json.dump(class_mapping, f)
        
        # Update progress to pending approval
        update_training_progress('pending_approval', 100, 25, test_accuracy)
        
        # Create a completion marker file for the frontend to detect
        completion_file = os.path.join('local_model', 'training_completed.txt')
        with open(completion_file, 'w') as f:
            f.write(f"Training completed at {datetime.utcnow().isoformat()}\n")
            f.write(f"Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Dataset size: {len(images)}\n")
            f.write(f"Status: pending_approval\n")
            f.write(f"Temp location: {temp_model_path}\n")
            f.write(f"Training ID: {training_id}\n")
        
        print("🎉 Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        # Clean up progress file on failure
        try:
            progress_file = os.path.join('local_model', 'training_progress.txt')
            if os.path.exists(progress_file):
                os.remove(progress_file)
        except:
            pass
        import traceback
        traceback.print_exc()


# ---------- Admin Model Management ----------
@app.route('/admin/models')
@login_required
@admin_required
def admin_models():
    """Admin model management page"""
    models = ModelSource.query.order_by(ModelSource.created_at.desc()).all()
    return render_template('admin/models.html', models=models)

@app.route('/admin/models/<int:model_id>/activate', methods=['POST'])
@login_required
@admin_required
def admin_model_activate(model_id):
    """Activate a specific model"""
    try:
        # Deactivate all models first
        ModelSource.query.update({'is_active': False})
        
        # Activate the selected model
        model = ModelSource.query.get_or_404(model_id)
        model.is_active = True
        model.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Log model activation
        log_audit_event('ACTIVATE', 'model', model.id, {
            'model_id': model.model_id,
            'model_source': model.model_source,
            'description': model.description
        })
        
        flash(f'Model "{model.model_source}" activated successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error activating model: {str(e)}', 'error')
    
    return redirect(url_for('admin_models'))

@app.route('/admin/models/<int:model_id>/deactivate', methods=['POST'])
@login_required
@admin_required
def admin_model_deactivate(model_id):
    """Deactivate a specific model"""
    try:
        model = ModelSource.query.get_or_404(model_id)
        model.is_active = False
        model.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Log model deactivation
        log_audit_event('DEACTIVATE', 'model', model.id, {
            'model_id': model.model_id,
            'model_source': model.model_source
        })
        
        flash(f'Model "{model.model_source}" deactivated successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deactivating model: {str(e)}', 'error')
    
    return redirect(url_for('admin_models'))



# ---------- Admin Audit Logs ----------
@app.route('/admin/audit-logs')
@login_required
@admin_required
def admin_audit_logs():
    """Admin audit logs page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    audit_logs = AuditLog.query.order_by(AuditLog.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/audit_logs.html', audit_logs=audit_logs)


# ---------- Admin Records (read-only list + view) ----------
@app.route('/admin/records')
@login_required
@admin_required
def admin_records():
    records = RoosterRecord.query.order_by(RoosterRecord.uploaded_at.desc()).all()
    return render_template('admin/records_list.html', records=records)


@app.route('/admin/records/<int:record_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_record_view(record_id: int):
    record = RoosterRecord.query.get_or_404(record_id)
    if request.method == 'POST':
        # Capture moderation fields
        is_correct_val = request.form.get('is_correct')
        notes = request.form.get('notes', '').strip() or None
        record.is_correct = 1 if is_correct_val == '1' else (0 if is_correct_val == '0' else None)
        record.notes = notes
        # Map selected breed_id directly
        breed_id_val = request.form.get('breed_id')
        if breed_id_val:
            try:
                record.breed_id = int(breed_id_val)
            except ValueError:
                record.breed_id = None
        else:
            record.breed_id = None
        try:
            db.session.commit()
            # Log record update
            log_audit_event('UPDATE', 'record', record.id, {
                'is_correct': record.is_correct,
                'breed_id': record.breed_id,
                'notes': record.notes,
                'predicted_category': record.predicted_category,
                'confidence_score': float(record.confidence_score) if record.confidence_score else None
            })
            flash('Record feedback saved successfully.', 'records')
        except Exception:
            db.session.rollback()
            flash('Failed to save feedback. Please try again.', 'records')
        return redirect(url_for('admin_record_view', record_id=record.id))
    # GET
    breeds = Breed.query.filter_by(is_deleted=False).order_by(Breed.name.asc()).all()
    return render_template('admin/record_view.html', record=record, breeds=breeds)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        # Check if file is allowed
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Add timestamp to make filename unique
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            
            # Save file to upload folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Make prediction
                predicted_breed, confidence = predict_breed(file_path)
                
                # Get active model information
                active_model = get_active_model_source()
                model_id = active_model.id if active_model else None
                model_source = active_model.model_source if active_model else None
                
                # Save record to database
                record = RoosterRecord(
                    filename=filename,
                    original_filename=file.filename,
                    predicted_category=predicted_breed,
                    confidence_score=confidence,
                    model_id=model_id,
                    model_source=model_source
                )
                
                db.session.add(record)
                db.session.commit()
                
                flash(f'File uploaded successfully! Predicted breed: {predicted_breed} (Confidence: {confidence:.2%})')
                return redirect(url_for('results', record_id=record.id))
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                # Clean up the uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/results/<int:record_id>')
def results(record_id):
    """Display prediction results for a specific record"""
    record = RoosterRecord.query.get_or_404(record_id)
    return render_template('results.html', record=record)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    # JSON for AJAX endpoints; HTML fallback for normal pages
    if request.path.startswith('/admin/colab/upload') or request.path.startswith('/admin/teachable-machine/upload'):
        return jsonify({'success': False, 'message': 'Upload failed: file is too large. Please ensure the ZIP/.h5 fits within the configured limit.'}), 413
    flash('Upload failed: file is too large. Please ensure it fits within the configured limit.', 'danger')
    return redirect(request.referrer or url_for('index'))

@app.route('/records')
def records():
    """Display all uploaded records"""
    page = request.args.get('page', 1, type=int)
    records = RoosterRecord.query.order_by(RoosterRecord.uploaded_at.desc()).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template('records.html', records=records)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction (for future mobile app integration)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            predicted_breed, confidence = predict_breed(file_path)
            
            # Get active model information
            active_model = get_active_model_source()
            model_id = active_model.id if active_model else None
            model_source = active_model.model_source if active_model else None
            
            # Save to database
            record = RoosterRecord(
                filename=filename,
                original_filename=file.filename,
                predicted_category=predicted_breed,
                confidence_score=confidence,
                model_id=model_id,
                model_source=model_source
            )
            db.session.add(record)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'predicted_breed': predicted_breed,
                'confidence': confidence,
                'record_id': record.id
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Database initialization
def log_audit_event(action: str, entity_type: str, entity_id: int = None, payload: dict = None):
    """Log an audit event"""
    try:
        audit_log = AuditLog(
            user_id=current_user.id if current_user.is_authenticated else None,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload
        )
        db.session.add(audit_log)
        db.session.commit()
    except Exception as e:
        print(f"⚠️ Failed to log audit event: {e}")
        db.session.rollback()


def create_tables():
    """Create database tables if they don't exist"""
    try:
        db.create_all()
        print("Database tables created successfully!")
        
        # Add some sample breeds if the table is empty
        if Breed.query.count() == 0:
            sample_breeds = [
                Breed(name='Rhode Island Red', description='A popular dual-purpose breed', characteristics='Hardy, good layers'),
                Breed(name='Leghorn', description='Excellent egg layers', characteristics='Active, white eggs'),
                Breed(name='Sussex', description='Calm and friendly breed', characteristics='Good for beginners'),
                Breed(name='Orpington', description='Large, docile birds', characteristics='Cold hardy, good mothers'),
                Breed(name='Wyandotte', description='Beautiful, productive breed', characteristics='Rose comb, good layers')
            ]
            for breed in sample_breeds:
                db.session.add(breed)
            
            db.session.commit()
            print("Sample breeds added to database!")
        
        # Add model sources if the table is empty
        if ModelSource.query.count() == 0:
            model_sources = [
                ModelSource(
                    model_id='local_v1',
                    model_source='Local Training',
                    description='MobileNetV2 model trained locally using train_local_mobilenet.py. Saved in SavedModel format for maximum compatibility.',
                    is_active=True
                ),
                ModelSource(
                    model_id='colab_v1',
                    model_source='Google Colab',
                    description='High-accuracy model trained in Google Colab with GPU acceleration. Exported as H5 format.',
                    is_active=False
                ),
                ModelSource(
                    model_id='tm_v1',
                    model_source='Teachable Machine',
                    description='Model created using Google Teachable Machine platform. Easy to create and deploy.',
                    is_active=False
                )
            ]
            for model_source in model_sources:
                db.session.add(model_source)
            
            db.session.commit()
            print("Model sources added to database!")
            
    except Exception as e:
        print(f"Error creating database tables: {e}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('upload_file'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize database tables
    with app.app_context():
        create_tables()
    
    # Load the model on startup
    print("Loading AI model...")
    load_model()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)

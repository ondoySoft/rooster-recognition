# Teachable Machine Models Directory

This directory is for models created using Google's Teachable Machine platform.

## Expected Files:
- `rooster_model.h5` - H5 format model
- `class_mapping.json` - Class mapping file
- `model.json` - Teachable Machine model definition (optional)
- `weights.bin` - Model weights (optional)

## Usage:
1. Create model in Teachable Machine (https://teachablemachine.withgoogle.com/)
2. Export the model files
3. Place them in this directory
4. The Flask app will detect and load them when Teachable Machine is selected

## Note:
Currently, the app prioritizes local models over Teachable Machine models.
Future dashboard feature will allow users to select which model to use.

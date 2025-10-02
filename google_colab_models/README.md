# Google Colab Models Directory

This directory is for models trained in Google Colab and exported to local setup.

## Expected Files:
- `rooster_model.h5` - H5 format model
- `class_mapping.json` - Class mapping file
- `rooster_model_80_plus.h5` - High accuracy model (optional)
- `class_mapping_80_plus.json` - High accuracy mapping (optional)

## Usage:
1. Train model in Google Colab using `Rooster_Training_Colab.ipynb`
2. Download the model files from Colab
3. Place them in this directory
4. The Flask app will automatically detect and load them

## Note:
Currently, the app prioritizes local models over Colab models.

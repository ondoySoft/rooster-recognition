# Local Model Directory

This directory contains models trained locally using `train_local_mobilenet.py`.

## Files:
- `train_local_mobilenet.py` - Local training script
- `saved_model.pb` - SavedModel format (preferred)
- `assets/` - Model assets
- `variables/` - Model variables
- `fingerprint.pb` - Model fingerprint

## Usage:
Run the training script from this directory:
```bash
python train_local_mobilenet.py
```

The trained model will be saved in SavedModel format and can be loaded by the Flask app.

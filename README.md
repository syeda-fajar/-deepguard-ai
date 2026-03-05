# Deepfake Detector
## Project Overview

This project implements a deepfake detection pipeline with a simple web UI. It provides preprocessing utilities, model loading, evaluation, and a Flask-based app for uploading videos/images to detect deepfakes.

## Features

- Web UI for uploading and viewing detection results
- Preprocessing utilities and Grad-CAM utilities for analysis
- Pretrained model included for inference
- Scripts for testing and evaluation

## Repository Structure

- `app.py` — Flask web application entrypoint
- `requirements.txt` — Python dependencies
- `models/` — model code and loader ([models/model_loader.py](models/model_loader.py))
- `saved_models/` — pretrained model weights ([saved_models/balanced_model.pth](saved_models/balanced_model.pth))
- `data/` — datasets used for evaluation (e.g., `data/test/fake`, `data/test/real`)
- `templates/` — HTML templates for the web UI
- `static/` — CSS and uploaded media
- `utils/` — helper modules (`utils/preprocessing.py`, `utils/analysis.py`, `utils/gradcam_utils.py`)
- `test_script.py` — quick test runner

## Requirements

Install Python 3.8+ (recommend 3.9 or 3.10) and the project dependencies:

```bash
python -m pip install -r requirements.txt
```

If you use a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick Start — Run the Web App

1. Ensure dependencies are installed.
2. Launch the app:

```bash
python app.py
```

3. Open your browser to the address shown in the terminal (usually `http://127.0.0.1:5000/`). Use the upload page to submit images/videos.

Relevant files:
- Web app: [app.py](app.py)
- Templates: [templates/index.html](templates/index.html), [templates/upload.html](templates/upload.html), [templates/result.html](templates/result.html)

## Inference / Batch Testing

Use `test_script.py` to run inference on the dataset under `data/test`.

```bash
python test_script.py --data-dir data/test --model saved_models/balanced_model.pth
```

Check `utils/analysis.py` for result aggregation and metrics computation.

## Model Details

- The model loader is implemented in [models/model_loader.py](models/model_loader.py).
- A pretrained weight file is included at [saved_models/balanced_model.pth](saved_models/balanced_model.pth).
- For training scripts or custom model architecture changes, update the loader in `models/model_loader.py` and the training loop accordingly.

## Data Format

- `data/test/fake/` — samples labeled as fake
- `data/test/real/` — samples labeled as real

Preprocessing steps (see `utils/preprocessing.py`): face extraction, resizing, normalization. Inspect that file to customize transforms.

## Grad-CAM and Analysis

- Grad-CAM utilities are in `utils/gradcam_utils.py` and `utils/analysis.py`. Use them to visualize model attention and to generate diagnostic figures.

## Training (outline)

This project focuses on inference, but to retrain:

1. Prepare training dataset with the same structure as `data/`.
2. Implement or reuse a training loop that constructs the model via `models/model_loader.py`.
3. Save checkpoints to `saved_models/`.

Example (pseudo-steps):

```bash
# train.py (not included) would:
# - load training data
# - build model via models/model_loader.py
# - run training epochs
# - save best model to saved_models/
```

## Tests

- Run `python test_script.py` to run the provided test script and evaluate performance on `data/test`.

## Common Tasks and Troubleshooting

- If dependencies fail to install, upgrade `pip` and retry: `python -m pip install --upgrade pip`.
- If the web app cannot find the model, confirm the path to `saved_models/balanced_model.pth` in `models/model_loader.py`.
- For GPU support, ensure `torch` was installed with CUDA support matching your GPU drivers.

## Contributing

If you'd like to contribute:

1. Open an issue describing the change.
2. Create a branch for your feature/fix.
3. Submit a pull request with tests or validation steps.


## Contact

For questions or help, open an issue or contact the repository owner.



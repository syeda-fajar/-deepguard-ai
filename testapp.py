from models.model_loader import load_model_from_checkpoint, predict_image
from PIL import Image

MODEL_PATH = "saved_models/balanced_model.pth"
DEVICE = "cpu"

# ✅ Load model
model = load_model_from_checkpoint(MODEL_PATH, device=DEVICE)
print("✅ Model loaded successfully!")

# ✅ Use your existing test image
img = Image.open("tests/sample1.jpg").convert("RGB")

# ✅ Run prediction
result = predict_image(model, img, device=DEVICE)
print("Prediction:", result)

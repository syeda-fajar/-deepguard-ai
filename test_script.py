# test_script.py
import os
from models.model_loader import load_model_from_checkpoint, predict_image
from utils.preprocessing import load_pil_from_path

def main():
    ckpt = "saved_models/balanced_model.pth"  # adjust if your checkpoint path is different
    sample = "tests/sample1.jpg"              # place a sample image at this path

    if not os.path.exists(ckpt):
        print(f"[test_script] Checkpoint not found at {ckpt}. Place checkpoint there or change the path.")
        return
    if not os.path.exists(sample):
        print(f"[test_script] Sample image not found at {sample}. Place a jpg there to test.")
        return

    model = load_model_from_checkpoint(ckpt, device='cpu', num_classes=2)
    img = load_pil_from_path(sample)
    idx, cls, conf, probs = predict_image(model, img, device='cpu', input_size=224)
    print("Prediction result:")
    print(f"  class_idx: {idx}")
    print(f"  class_name: {cls}")
    print(f"  confidence: {conf:.4f}")
    print(f"  probs: {probs}")

if __name__ == "__main__":
    main()

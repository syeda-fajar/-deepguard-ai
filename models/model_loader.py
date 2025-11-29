# models/model_loader.py
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeDetector(nn.Module):
    """
    Recreates the training model architecture used in your notebook:
    - torchvision EfficientNet-B0 base
    - custom classifier head:
        Dropout -> Linear(in_features,256) -> BatchNorm1d -> ReLU -> Dropout -> Linear(256,num_classes)
    - freeze first 2 feature blocks (same logic as notebook)
    """
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super().__init__()
        # Try to use the same weights enum; if not available, pass weights=None
        try:
            base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            base_model = models.efficientnet_b0(weights=None)

        # Freeze first two feature blocks (same as your notebook)
        freeze_count = 0
        for idx, (name, param) in enumerate(base_model.features.named_children()):
            if idx < 2:
                for p in param.parameters():
                    p.requires_grad = False
                freeze_count += 1
        # keep classifier features shape
        in_features = base_model.classifier[1].in_features

        base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(256, num_classes)
        )
        self.model = base_model

    def forward(self, x):
        return self.model(x)


# ---------- checkpoint helpers (robust loader) ----------
def _strip_module_prefix(state_dict):
    """Strip 'module.' from keys if present (for DataParallel checkpoints)."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k.replace("module.", "")] = v
        else:
            new_state[k] = v
    return new_state

def load_checkpoint_safe(checkpoint_path, map_location=DEVICE):
    """
    Attempts to load checkpoint with fallback (main -> backup). Returns the loaded object (dict or state_dict).
    This is simplified but robust for typical saved dicts containing 'model_state_dict'.
    """
    backup_path = checkpoint_path.replace(".pth", "_backup.pth")
    paths = [(checkpoint_path, "main"), (backup_path, "backup")]
    for path, desc in paths:
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=map_location)
                print(f"[loader] Loaded {desc} checkpoint from {path}")
                return ckpt
            except Exception as e:
                print(f"[loader] Failed to load {desc} checkpoint ({path}): {e}")
                continue
    print("[loader] No checkpoint found (main or backup).")
    return None


def build_model(num_classes=2, dropout_rate=0.4):
    return DeepfakeDetector(num_classes=num_classes, dropout_rate=dropout_rate)


def load_model_from_checkpoint(checkpoint_path, device=None, num_classes=2, dropout_rate=0.4):
    """
    Build model architecture and load weights from checkpoint_path.
    Handles:
      - checkpoint dicts with 'model_state_dict'
      - checkpoints that are raw state_dicts
      - 'module.' prefixes
    Returns model moved to device and set to eval().
    """
    device = device or DEVICE
    model = build_model(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    ckpt = load_checkpoint_safe(checkpoint_path, map_location=device)
    if ckpt is None:
        print("[load_model] No checkpoint loaded; returning freshly initialized model (not recommended).")
        model.eval()
        return model

    # If it's a dict containing 'model_state_dict', use that
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    # If ckpt itself looks like a state_dict (tensor values), assume that's fine
    elif isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        # fallback: maybe the checkpoint object is a nn.Module saved directly
        try:
            if hasattr(ckpt, 'state_dict'):
                model.load_state_dict(ckpt.state_dict())
                model.to(device).eval()
                return model
        except Exception:
            pass
        raise RuntimeError("Unrecognized checkpoint format. Inspect the checkpoint file.")

    # strip module prefixes if present
    state = _strip_module_prefix(state)

    # load (allow missing keys False but warn)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_model] Missing keys when loading state_dict: {len(missing)} (first 5): {missing[:5]}")
    if unexpected:
        print(f"[load_model] Unexpected keys in checkpoint: {len(unexpected)} (first 5): {unexpected[:5]}")

    model.to(device).eval()
    return model


# ---------- inference helper ----------
def get_val_transform(input_size=224, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std  = std  or [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def predict_image(model, pil_image, device=None, input_size=224, class_names=None):
    """
    Run forward pass on a PIL image and return:
      - pred_idx (int), class_name (str), confidence (float), probs (numpy array)
    class_names: list like ['fake','real'] (default uses ImageFolder alphabetical assumption)
    """
    device = device or DEVICE
    model.to(device).eval()

    # default class mapping: ImageFolder sorts alphabetically -> 'fake','real'
    if class_names is None:
        class_names = ['fake', 'real']

    tf = get_val_transform(input_size=input_size)
    x = tf(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs.max())
    class_name = class_names[pred_idx] if isinstance(class_names, (list,tuple)) and len(class_names) > pred_idx else str(pred_idx)
    return pred_idx, class_name, confidence, probs

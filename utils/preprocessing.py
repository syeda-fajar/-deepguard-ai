# utils/preprocessing.py
from PIL import Image
import io

def load_pil_from_bytes(b):
    """
    Convert uploaded bytes (e.g., from Streamlit file_uploader) to a PIL.Image (RGB).
    """
    return Image.open(io.BytesIO(b)).convert("RGB")

def load_pil_from_path(path):
    return Image.open(path).convert("RGB")

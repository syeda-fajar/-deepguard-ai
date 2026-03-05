from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import torch
from PIL import Image
import numpy as np
import cv2
from datetime import datetime
from werkzeug.utils import secure_filename

from models.model_loader import load_model_from_checkpoint, predict_image
from utils.gradcam_utils import GradCAM, overlay_heatmap
from utils.analysis import (
    calculate_image_quality_score, 
    detect_potential_artifacts,
    get_quality_label,
    calculate_heatmap_stats
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_models/balanced_model.pth"

print(" Loading model...")
try:
    model = load_model_from_checkpoint(MODEL_PATH, device=DEVICE)
    model.eval()
    print(f" Model loaded successfully on {DEVICE}!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(filepath):
    """Load and preprocess image for model input"""
    from torchvision import transforms
    
    image = Image.open(filepath).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(image).unsqueeze(0)
    return image, tensor

def save_visualization(image_array, heatmap, output_path):
    """Create and save overlayed heatmap visualization"""
    overlayed = overlay_heatmap(image_array, heatmap, alpha=0.4)
    cv2.imwrite(output_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
    return output_path



@app.route('/')
def home():
    """Landing page with upload form"""
    return render_template('index.html', device=str(DEVICE))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
   
    if model is None:
        flash('Model not loaded. Please check server logs.', 'danger')
        return redirect(url_for('home'))
   
    if 'file' not in request.files:
        flash('No file uploaded. Please select an image.', 'warning')
        return redirect(url_for('home'))
    
    file = request.files['file']
    
    
    if file.filename == '':
        flash('No file selected. Please choose an image.', 'warning')
        return redirect(url_for('home'))
    
  
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload PNG, JPG, or JPEG images only.', 'danger')
        return redirect(url_for('home'))
    
    try:
       
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        
        pil_image, img_tensor = preprocess_image(filepath)
        img_tensor = img_tensor.to(DEVICE)
        
        # ====== Prediction using model_loader ======
        pred_idx, class_name, confidence, probs = predict_image(
            model, pil_image, device=DEVICE, class_names=['fake', 'real']
        )
        
        # ====== Generate Grad-CAM ======
        gradcam = GradCAM(model)
        heatmap, _, _ = gradcam.generate_cam(img_tensor, class_idx=pred_idx)
        gradcam.remove_hooks()
        
        # ====== Create visualizations ======
        original_np = np.array(pil_image.resize((224, 224)))
        
        # Save heatmap visualization
        heatmap_filename = f"heatmap_{timestamp}.jpg"
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_colored)
        
        # Save overlay visualization
        overlay_filename = f"overlay_{timestamp}.jpg"
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
        save_visualization(original_np, heatmap, overlay_path)
        
        # ====== Analysis ======
        quality_metrics = calculate_image_quality_score(original_np)
        artifacts = detect_potential_artifacts(heatmap)
        quality_label, quality_badge = get_quality_label(confidence)
        heatmap_stats = calculate_heatmap_stats(heatmap)
        
        # ====== Prepare results ======
        results = {
            'prediction': class_name.capitalize(),
            'confidence': round(confidence * 100, 2),
            'fake_prob': round(float(probs[0]) * 100, 2),
            'real_prob': round(float(probs[1]) * 100, 2),
            'quality_label': quality_label,
            'quality_badge': quality_badge,
            'original_image': filename,
            'heatmap_image': heatmap_filename,
            'overlay_image': overlay_filename,
            'quality_metrics': quality_metrics,
            'artifacts': artifacts,
            'heatmap_stats': heatmap_stats,
            'timestamp': timestamp
        }
        
        return render_template('result.html', **results)
        
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'danger')
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('home'))

@app.route('/api/health')
def health_check():
    """API endpoint to check server health"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.', 'danger')
    return redirect(url_for('home'))

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404

# ====== Run Application ======
if __name__ == '__main__':
    app.run()
# utils/analysis.py
import numpy as np
import cv2

def calculate_image_quality_score(image_array):
    """Calculate comprehensive image quality metrics"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Contrast (standard deviation)
    contrast = gray.std()
    
    # Brightness
    brightness = gray.mean()
    
    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    return {
        'sharpness': float(sharpness),
        'contrast': float(contrast),
        'brightness': float(brightness),
        'edge_density': float(edge_density),
        'overall_score': float((sharpness/1000 + contrast/50 + edge_density*100) / 3)
    }

def detect_potential_artifacts(heatmap, threshold=0.7):
    """Detect potential deepfake artifacts based on attention patterns"""
    artifacts = []
    
    if heatmap is None or heatmap.size == 0:
        return ["⚠️ Unable to analyze heatmap"]
    
    # High attention on edges (common in GAN artifacts)
    edge_attention = (np.mean(heatmap[:3, :]) + np.mean(heatmap[-3:, :]) + 
                     np.mean(heatmap[:, :3]) + np.mean(heatmap[:, -3:])) / 4
    if edge_attention > threshold:
        artifacts.append("🔴 High edge attention detected")
    
    # Unusual attention patterns
    center_attention = np.mean(heatmap[heatmap.shape[0]//3:-heatmap.shape[0]//3, 
                                     heatmap.shape[1]//3:-heatmap.shape[1]//3])
    if center_attention < 0.2:
        artifacts.append("🟡 Low center attention (unusual)")
    
    # Asymmetric attention (faces should be roughly symmetric)
    left_half = np.mean(heatmap[:, :heatmap.shape[1]//2])
    right_half = np.mean(heatmap[:, heatmap.shape[1]//2:])
    asymmetry = abs(left_half - right_half) / max(left_half, right_half, 1e-8)
    if asymmetry > 0.3:
        artifacts.append("🟠 Asymmetric attention pattern")
    
    if not artifacts:
        artifacts.append("✅ No obvious artifacts detected")
    
    return artifacts

def get_quality_label(confidence):
    """Return quality label and color based on confidence"""
    if confidence >= 0.90:
        return "High Quality Detection", "success"
    elif confidence >= 0.70:
        return "Medium Quality Detection", "warning"
    else:
        return "Low Quality / Uncertain", "danger"

def calculate_heatmap_stats(heatmap):
    """Calculate statistical metrics from heatmap"""
    if heatmap is None or heatmap.size == 0:
        return {}
    
    return {
        'mean_activation': float(np.mean(heatmap)),
        'max_activation': float(np.max(heatmap)),
        'min_activation': float(np.min(heatmap)),
        'std_activation': float(np.std(heatmap)),
        'top_region': float(np.mean(heatmap[:heatmap.shape[0]//3, :])),
        'center_region': float(np.mean(heatmap[heatmap.shape[0]//3:2*heatmap.shape[0]//3, :])),
        'bottom_region': float(np.mean(heatmap[2*heatmap.shape[0]//3:, :]))
    }
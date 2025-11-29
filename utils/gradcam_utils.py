# utils/gradcam_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    Generates heatmaps showing which regions the model focuses on.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_maps = {}
        self.gradients = {}

        # Get the last convolutional layer
        if hasattr(model, "model"):  # For wrapped models
            self.target_layer = model.model.features[-1]
        else:
            self.target_layer = model.features[-1]

        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_feature_maps)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        """Hook to save feature maps during forward pass"""
        self.feature_maps["last_conv"] = output

    def _save_gradients(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass"""
        self.gradients["last_conv"] = grad_output[0]

    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Class Activation Map (CAM) for the input tensor.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            class_idx: Target class index (optional, uses predicted class if None)
            
        Returns:
            cam: Normalized heatmap (H, W) as numpy array
            class_idx: The class index used
            probs: Class probabilities as numpy array
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = int(np.argmax(probs))
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        # Check if hooks captured data
        if "last_conv" not in self.feature_maps or "last_conv" not in self.gradients:
            print("⚠️ Warning: Grad-CAM hooks did not capture feature maps or gradients")
            return np.zeros((7, 7)), class_idx, probs

        # Get feature maps and gradients
        feature_maps = self.feature_maps["last_conv"]
        gradients = self.gradients["last_conv"]

        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert to numpy and normalize
        cam = cam[0, 0].cpu().detach().numpy()
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, class_idx, probs

    def remove_hooks(self):
        """Remove registered hooks to free memory"""
        try:
            self.forward_hook.remove()
            self.backward_hook.remove()
        except Exception as e:
            print(f"Warning: Could not remove hooks: {e}")


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original RGB image (H, W, 3)
        heatmap: Normalized heatmap (H, W)
        alpha: Transparency factor (0-1)
        colormap: OpenCV colormap
        
    Returns:
        Overlayed image as uint8 RGB array
    """
    if heatmap is None or heatmap.size == 0 or heatmap.max() <= 0:
        return image
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlayed = (1 - alpha) * image + alpha * heatmap_colored
    
    return overlayed.astype(np.uint8)


def tensor_to_image_uint8(tensor, mean=None, std=None):
    """
    Convert normalized tensor back to uint8 image.
    
    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean (default ImageNet)
        std: Normalization std (default ImageNet)
        
    Returns:
        RGB image as uint8 numpy array (H, W, 3)
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to numpy and transpose to (H, W, C)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Convert to uint8
    img = (img * 255).astype(np.uint8)
    
    return img
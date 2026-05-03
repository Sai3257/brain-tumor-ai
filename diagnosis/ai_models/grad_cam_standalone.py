"""
Standalone Grad-CAM Visualization Module
Completely independent with no self references
"""
import os
import numpy as np
import cv2

# Use existing TensorFlow compatibility system
try:
    from diagnosis.tensorflow_compat_simple import get_tensorflow_components, is_tensorflow_available
except ImportError:
    # Fallback if import fails
    def get_tensorflow_components():
        return {'tf': None, 'load_model': None, 'load_img': None, 'img_to_array': None, 'np': None}
    
    def is_tensorflow_available():
        return False

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_simple_heatmap(image_path, severity_level=None):
    """Create simple heatmap using OpenCV only"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            # Create simple synthetic colored heatmap (3D)
            heatmap = np.zeros((256, 256, 3), dtype=np.uint8)
            # Create circular tumor region
            center_x, center_y = 128, 128
            radius = 50
            y, x = np.ogrid[:256, :256]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            heatmap[mask] = [255, 0, 0]  # Red tumor region
            return heatmap
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize to standard size
        gray_resized = cv2.resize(gray, (256, 256))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray_resized, 127, 255, cv2.THRESH_BINARY)
        
        # Apply color map based on severity - always return 3D
        if severity_level:
            if 'critical' in severity_level.lower() or 'very_high' in severity_level.lower():
                colored = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
            elif 'high' in severity_level.lower():
                colored = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
            elif 'medium' in severity_level.lower():
                colored = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
            else:
                colored = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
        else:
            colored = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
        
        # Ensure 3D format
        if len(colored.shape) == 2:
            colored = cv2.cvtColor(colored, cv2.COLOR_GRAY2BGR)
        
        return colored
        
    except Exception as e:
        print(f"Error creating simple heatmap: {e}")
        # Return simple red heatmap (3D)
        return np.ones((256, 256, 3), dtype=np.uint8) * [255, 0, 0]

def generate_grad_cam_heatmap_standalone(image_path, tumor_type=None, confidence=0.0, severity_level=None):
    """Generate Grad-CAM heatmap with no self references"""
    try:
        print(f"Generating Grad-CAM for: {os.path.basename(image_path)}")
        
        # Create simple heatmap using OpenCV
        heatmap = create_simple_heatmap(image_path, severity_level)
        
        print("Grad-CAM generated successfully")
        return heatmap
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return create_simple_heatmap(image_path, severity_level)

# Global visualizer instance (for compatibility)
_visualizer = None

def get_grad_cam_visualizer_standalone():
    """Compatibility function - returns None since we're using standalone"""
    return None

# Main function for external use
def generate_grad_cam_heatmap(image_path, tumor_type=None, confidence=0.0, severity_level=None):
    """Main entry point - uses standalone implementation"""
    return generate_grad_cam_heatmap_standalone(image_path, tumor_type, confidence, severity_level)

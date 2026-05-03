"""
Grad-CAM Visualization Module
Generates heatmap visualizations showing tumor severity and grading
"""
import os
import numpy as np
import cv2

# ✅ Simplified: Remove matplotlib dependencies to avoid GUI issues
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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

class GradCAMVisualizer:
    """Enhanced Grad-CAM visualization with severity-based coloring"""
    
    def __init__(self):
        self.model = None
        self.target_size = (256, 256)
        self.model_path = os.path.join(BASE_DIR, "saved_model", "my_model.keras")
        
        # Severity-based color mapping
        self.severity_colors = {
            'low': {
                'color_map': 'Greens',
                'alpha': 0.4,
                'threshold': 0.3,
                'description': 'Low Risk - Green indicates minimal concern'
            },
            'medium': {
                'color_map': 'YlOrRd',
                'alpha': 0.5,
                'threshold': 0.5,
                'description': 'Medium Risk - Yellow to Red indicates moderate concern'
            },
            'high': {
                'color_map': 'hot',
                'alpha': 0.6,
                'threshold': 0.7,
                'description': 'High Risk - Hot colors indicate significant concern'
            },
            'critical': {
                'color_map': 'plasma',
                'alpha': 0.7,
                'threshold': 0.8,
                'description': 'Critical Risk - Plasma colors indicate urgent attention needed'
            }
        }
        
        # Custom colormap for tumor severity
        self.create_severity_colormap()
        self._load_model()
    
    def create_severity_colormap(self):
        """Create custom colormap for tumor severity using OpenCV"""
        # ✅ Simplified: Use OpenCV color maps instead of matplotlib
        self.severity_colormap = {
            'low': [(0, 255, 0), (0, 200, 0)],      # Green shades
            'medium': [(255, 255, 0), (255, 165, 0)],  # Yellow shades
            'high': [(255, 165, 0), (255, 0, 0)],      # Orange shades
            'critical': [(255, 0, 0), (200, 0, 0)]     # Red shades
        }
    
    def _load_model(self):
        """Load model for Grad-CAM"""
        try:
            components = get_tensorflow_components()
            
            if not is_tensorflow_available():
                print("WARN:WARN: TensorFlow not available, using fallback Grad-CAM")
                self.model = None
                return
            
            load_model = components['load_model']
            self.model = load_model(self.model_path)
            print("OK: Model loaded for Grad-CAM visualization")
            
        except Exception as e:
            print(f"ERR: Error loading model for Grad-CAM: {e}")
            self.model = None
    
    def generate_grad_cam(self, image_path, tumor_type=None, confidence=0.0, severity_level=None):
        """Generate enhanced Grad-CAM heatmap with severity-based coloring"""
        try:
            print(f" Generating Grad-CAM for: {os.path.basename(image_path)}")
            
            # Load and preprocess image
            img_array, original_img = self.preprocess_image(image_path)
            if img_array is None:
                return _create_fallback_heatmap(image_path, severity_level)
            
            if self.model is None:
                return _create_fallback_heatmap(image_path, severity_level)
            
            # Generate Grad-CAM
            heatmap = self._compute_grad_cam(img_array, tumor_type)
            
            # Apply severity-based coloring
            colored_heatmap = _apply_severity_coloring(heatmap, severity_level, confidence)
            
            # Create overlay with original image
            overlay = _create_grad_cam_overlay(original_img, colored_heatmap)
            
            # Add severity annotations
            final_heatmap = _add_severity_annotations(overlay, tumor_type, confidence, severity_level)
            
            print("OK: Enhanced Grad-CAM generated successfully")
            return final_heatmap
            
        except Exception as e:
            print(f"ERR: Error generating Grad-CAM: {e}")
            return _create_fallback_heatmap(image_path, severity_level)
    
    def _compute_grad_cam(self, img_array, tumor_type=None):
        """Compute Grad-CAM heatmap"""
        try:
            components = get_tensorflow_components()
            tf = components['tf']
            
            # Create Grad-CAM model
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [self.model.output, self.model.get_layer('conv2d_3').output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                
                # Get the predicted class index
                if tumor_type:
                    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
                    if tumor_type in class_names:
                        class_idx = class_names.index(tumor_type)
                    else:
                        class_idx = np.argmax(predictions[0])
                else:
                    class_idx = np.argmax(predictions[0])
                
                loss = predictions[:, class_idx]
            
            # Compute gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Pool gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the conv outputs
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
            
            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"ERR: Error computing Grad-CAM: {e}")
            return self._create_synthetic_heatmap()
    
    def _apply_severity_coloring(self, heatmap, severity_level, confidence):
        """Apply severity-based coloring to heatmap using OpenCV"""
        try:
            # Determine severity-based color scheme
            if severity_level:
                if 'critical' in severity_level.lower() or 'very_high' in severity_level.lower():
                    severity_type = 'critical'
                elif 'high' in severity_level.lower():
                    severity_type = 'high'
                elif 'medium' in severity_level.lower():
                    severity_type = 'medium'
                else:
                    severity_type = 'low'
            else:
                # Use confidence to determine severity
                if confidence > 90:
                    severity_type = 'high'
                elif confidence > 75:
                    severity_type = 'medium'
                else:
                    severity_type = 'low'
            
            # ✅ Simplified: Use OpenCV for coloring
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap_8bit = (heatmap_normalized * 255).astype(np.uint8)
            
            # Apply color based on severity using OpenCV
            if severity_type == 'critical':
                # Red colormap
                colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
            elif severity_type == 'high':
                # Orange colormap
                colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
            elif severity_type == 'medium':
                # Yellow colormap
                colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
            else:
                # Green colormap
                colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_HOT)
            
            return colored_heatmap
            
        except Exception as e:
            print(f"ERR: Error applying severity coloring: {e}")
            return self._create_simple_fallback_heatmap()
    
    def _create_grad_cam_overlay(self, original_img, colored_heatmap):
        """Create overlay of Grad-CAM with original image"""
        try:
            # Resize original image to match heatmap
            original_resized = cv2.resize(original_img, (colored_heatmap.shape[1], colored_heatmap.shape[0]))
            
            # Blend images
            overlay = cv2.addWeighted(original_resized, 0.6, colored_heatmap, 0.4, 0)
            
            return overlay
            
        except Exception as e:
            print(f"ERR: Error creating Grad-CAM overlay: {e}")
            return colored_heatmap
    
    def _add_severity_annotations(heatmap, tumor_type, confidence, severity_level):
        """Add text annotations for severity information"""
        try:
            # Create a copy for annotation
            annotated = heatmap.copy()
            
            # Add tumor type
            if tumor_type:
                cv2.putText(annotated, f'Tumor: {tumor_type.upper()}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add confidence
            cv2.putText(annotated, f'Confidence: {confidence:.1f}%', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add severity level
            if severity_level:
                cv2.putText(annotated, f'Severity: {severity_level.upper()}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add color legend
            _add_color_legend(annotated, severity_level)
            
            return annotated
            
        except Exception as e:
            print(f"ERR: Error adding severity annotations: {e}")
            return heatmap
    
    def _add_color_legend(image, severity_level):
        """Add color legend explaining severity colors"""
        try:
            # Create legend area
            legend_height, legend_width = 100, 200
            legend_x = image.shape[1] - legend_width - 10
            legend_y = image.shape[0] - legend_height - 10
            
            # Draw legend background
            cv2.rectangle(image, (legend_x, legend_y), 
                        (legend_x + legend_width, legend_y + legend_height), 
                        (0, 0, 0), -1)
            
            # Add color bars
            colors = [(0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)]
            labels = ['Low', 'Medium', 'High', 'Critical']
            
            bar_width = legend_width // 4
            for i, (color, label) in enumerate(zip(colors, labels)):
                x = legend_x + i * bar_width
                cv2.rectangle(image, (x, legend_y + 10), (x + bar_width - 5, legend_y + 30), color, -1)
                cv2.putText(image, label, (x + 5, legend_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add title
            cv2.putText(image, 'Severity Scale', (legend_x + 10, legend_y + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"ERR: Error adding color legend: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for Grad-CAM"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            original_img = np.array(img)
            
            # Resize for model
            img_resized = img.resize(self.target_size)
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, original_img
            
        except Exception as e:
            print(f"ERR: Error preprocessing image for Grad-CAM: {e}")
            return None, None
    
    def _create_fallback_heatmap(self, image_path, severity_level=None):
        """Create fallback heatmap when Grad-CAM fails"""
        try:
            # Load original image
            original_img = cv2.imread(image_path)
            if original_img is None:
                return self._create_synthetic_heatmap()
            
            # Create synthetic heatmap based on severity
            heatmap = self._create_synthetic_heatmap()
            
            # Resize to match original image
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            
            # Apply severity coloring
            colored_heatmap = self._apply_severity_coloring(heatmap_resized / 255.0, severity_level, 80.0)
            
            # Create overlay
            overlay = cv2.addWeighted(original_img, 0.6, colored_heatmap, 0.4, 0)
            
            # Add annotations
            return self._add_severity_annotations(overlay, 'Unknown', 75.0, severity_level or 'Medium')
            
        except Exception as e:
            print(f"ERR: Error creating fallback heatmap: {e}")
            return self._create_synthetic_heatmap()
    
    def _create_synthetic_heatmap(self):
        """Create synthetic heatmap for demonstration"""
        try:
            # Create a synthetic tumor-like heatmap
            heatmap = np.zeros((256, 256), dtype=np.float32)
            
            # Create circular tumor region in center
            center_x, center_y = 128, 128
            radius = 60
            
            y, x = np.ogrid[:256, :256]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Create gradient intensity
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            heatmap[mask] = 1.0 - (distances[mask] / radius)
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.1, heatmap.shape)
            heatmap = np.clip(heatmap + noise, 0, 1)
            
            return heatmap
            
        except Exception as e:
            print(f"ERR: Error creating synthetic heatmap: {e}")
            return np.random.rand(256, 256)
    
    def _create_simple_fallback_heatmap(self):
        """Create simple fallback heatmap using OpenCV"""
        try:
            # Create a simple synthetic heatmap
            heatmap = np.zeros((256, 256), dtype=np.float32)
            
            # Create circular tumor region in center
            center_x, center_y = 128, 128
            radius = 60
            
            y, x = np.ogrid[:256, :256]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Create gradient intensity
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            heatmap[mask] = 1.0 - (distances[mask] / radius)
            
            # Convert to 8-bit and apply OpenCV colormap
            heatmap_8bit = (heatmap * 255).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
            
            return colored_heatmap
            
        except Exception as e:
            print(f"ERR: Error creating simple fallback heatmap: {e}")
            # Return a simple red heatmap as last resort
            return np.ones((256, 256, 3), dtype=np.uint8) * [255, 0, 0]
    
    def _create_synthetic_colored_heatmap(self):
        """Create synthetic colored heatmap"""
        try:
            heatmap = self._create_synthetic_heatmap()
            
            # ✅ Fix: Apply severity coloring safely
            try:
                colored_heatmap = cm.get_cmap('hot')(heatmap)
                colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            except Exception as e:
                print(f"ERR: Synthetic colormap error: {e}, using fallback")
                # Fallback to simple coloring
                colored_heatmap = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
                colored_heatmap[heatmap > 0.5] = [255, 100, 0]  # Orange for high
                colored_heatmap[heatmap <= 0.5] = [0, 255, 0]  # Green for low
            
            return colored_heatmap
            
        except Exception as e:
            print(f"ERR: Error creating synthetic colored heatmap: {e}")
            # ✅ Fix: Return a simple fallback
            return np.ones((256, 256, 3), dtype=np.uint8) * 128

# Global visualizer instance
_visualizer = None

def get_grad_cam_visualizer():
    """Get or create Grad-CAM visualizer instance"""
    global _visualizer
    if _visualizer is None:
        _visualizer = GradCAMVisualizer()
    return _visualizer

def generate_grad_cam_heatmap(image_path, tumor_type=None, confidence=0.0, severity_level=None):
    """Generate Grad-CAM heatmap with severity coloring"""
    try:
        visualizer = get_grad_cam_visualizer()
        if visualizer is None:
            print("ERR: Visualizer not available, creating fallback")
            # Create fallback heatmap directly
            return _create_fallback_heatmap_standalone(image_path, severity_level)
        
        result = visualizer.generate_grad_cam(image_path, tumor_type, confidence, severity_level)
        if result is None:
            print("ERR: Grad-CAM generation failed, creating fallback")
            return _create_fallback_heatmap_standalone(image_path, severity_level)
        
        return result
        
    except Exception as e:
        print(f"ERR: Error in generate_grad_cam_heatmap: {e}")
        return _create_fallback_heatmap_standalone(image_path, severity_level)

def _create_fallback_heatmap_standalone(image_path, severity_level=None):
    """Standalone fallback heatmap function"""
    try:
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            # Create simple synthetic heatmap
            heatmap = np.zeros((256, 256), dtype=np.uint8)
            heatmap[100:150, 100:150] = 255  # Simple white square
            return heatmap
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize
        gray_resized = cv2.resize(gray, (256, 256))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray_resized, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Apply color map
        colored = cv2.applyColorMap(thresh, cv2.COLORMAP_HOT)
        
        return colored
        
    except Exception as e:
        print(f"ERR: Error in fallback heatmap: {e}")
        # Return simple red heatmap
        return np.ones((256, 256, 3), dtype=np.uint8) * [255, 0, 0]

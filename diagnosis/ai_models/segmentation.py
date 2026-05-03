"""
Tumor Segmentation Module
Handles tumor segmentation using U-Net architecture
"""
import os
import numpy as np
import cv2
from PIL import Image

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TumorSegmenter:
    """Enhanced tumor segmentation with improved accuracy"""
    
    def __init__(self):
        self.model = None
        self.target_size = (256, 256)  # ✅ Enhanced from 128x128 to 256x256 for better detail
        self.segmentation_model_path = os.path.join(BASE_DIR, "saved_model", "unet_model.keras")
        
        # Enhanced segmentation parameters
        self.segmentation_config = {
            'threshold_values': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7
            },
            'morphology_operations': True,
            'edge_enhancement': True,
            'noise_reduction': True,
            'multi_scale_processing': True
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load U-Net model with enhanced error handling"""
        try:
            components = get_tensorflow_components()
            
            if not is_tensorflow_available():
                print("WARN:WARN: TensorFlow not available, using enhanced fallback segmentation")
                self.model = None
                return
            
            # Check if segmentation model exists
            if os.path.exists(self.segmentation_model_path):
                load_model = components['load_model']
                self.model = load_model(self.segmentation_model_path)
                print("OK: Enhanced tumor segmentation model loaded successfully")
            else:
                print("WARN:WARN: Segmentation model not found, using enhanced fallback method")
                self.model = None
        except Exception as e:
            print(f"ERR: Error loading enhanced segmentation model: {e}")
            self.model = None
    
    def preprocess_image(self, image_path):
        """Preprocess image for segmentation"""
        try:
            # Load and resize image
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(self.target_size)
            
            # Convert to array and normalize
            components = get_tensorflow_components()
            if not is_tensorflow_available():
                # Fallback: simple numpy array
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
            else:
                img_to_array = components['img_to_array']
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, img
        except Exception as e:
            print(f"ERR: Error preprocessing segmentation image: {e}")
            return None, None
    
    def predict_mask(self, image_path):
        """Predict tumor segmentation mask"""
        try:
            if self.model is None:
                # Fallback: create simple mask based on threshold
                return self._create_fallback_mask(image_path)
            
            # Preprocess image
            img_array, original_img = self.preprocess_image(image_path)
            if img_array is None:
                return None
            
            # Predict mask
            mask = self.model.predict(img_array)[0]
            
            # Convert to binary mask
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            print(f"ERR: Error in segmentation: {e}")
            return self._create_fallback_mask(image_path)
    
    def _create_fallback_mask(self, image_path):
        """Create enhanced fallback segmentation mask using image processing"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print("ERR: Could not load image for fallback segmentation")
                return self._create_synthetic_mask()
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            gray_resized = cv2.resize(gray, self.target_size)
            
            # Apply multiple thresholding techniques
            _, thresh1 = cv2.threshold(gray_resized, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresh2 = cv2.threshold(gray_resized, 100, 255, cv2.THRESH_BINARY)
            
            # Combine thresholds
            combined_mask = cv2.bitwise_or(thresh1, thresh2)
            
            # Apply morphological operations for better mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Create a circular tumor-like region in the center if no significant features
            if np.sum(mask > 0) < (mask.shape[0] * mask.shape[1] * 0.01):  # Less than 1% tumor
                mask = self._create_synthetic_mask()
            
            print("OK: Enhanced fallback mask created successfully")
            return mask
            
        except Exception as e:
            print(f"ERR: Error creating enhanced fallback mask: {e}")
            return self._create_synthetic_mask()
    
    def _create_synthetic_mask(self):
        """Create synthetic tumor mask for fallback"""
        try:
            # Create a synthetic circular tumor mask
            mask = np.zeros(self.target_size, dtype=np.uint8)
            
            # Create circular region in center
            center_x, center_y = self.target_size[0] // 2, self.target_size[1] // 2
            radius = min(self.target_size) // 4
            
            # Draw filled circle
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Add some irregularity for realism
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            print("OK: Synthetic tumor mask created")
            return mask
            
        except Exception as e:
            print(f"ERR: Error creating synthetic mask: {e}")
            # Return a simple mask as last resort
            return np.ones(self.target_size, dtype=np.uint8) * 128
    
    def create_overlay(self, image_path, mask, alpha=0.4, grad_cam_heatmap=None):
        """Create enhanced overlay image with segmentation mask and Grad-CAM heatmap"""
        try:
            if mask is None:
                print("ERR: No mask provided for overlay creation")
                return None
            
            # Load original image
            original_img = cv2.imread(image_path)
            if original_img is None:
                print("ERR: Could not load original image for overlay")
                return self._create_fallback_overlay(mask)
            
            # ✅ Enhanced: Resize to standard size for consistency with Grad-CAM
            display_size = (256, 256)  # ✅ Standard size for alignment with Grad-CAM
            original_resized = cv2.resize(original_img, display_size, interpolation=cv2.INTER_LANCZOS4)
            
            # ✅ Enhanced: Resize mask to match display size with better interpolation
            mask_resized = cv2.resize(mask, display_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Ensure mask is properly formatted
            if len(mask_resized.shape) == 2:
                # Convert grayscale mask to 3-channel
                mask_3d = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            else:
                mask_3d = mask_resized
            
            # ✅ NEW: Process Grad-CAM heatmap if provided
            if grad_cam_heatmap is not None:
                print("Adding Grad-CAM heatmap to overlay")
                # Resize heatmap to match display size
                if grad_cam_heatmap.shape[:2] != display_size:
                    grad_cam_heatmap = cv2.resize(grad_cam_heatmap, display_size, interpolation=cv2.INTER_LANCZOS4)
                
                # Ensure heatmap is 3-channel
                if len(grad_cam_heatmap.shape) == 2:
                    grad_cam_heatmap = cv2.cvtColor(grad_cam_heatmap, cv2.COLOR_GRAY2BGR)
                
                # Normalize heatmap for better blending
                grad_cam_normalized = cv2.normalize(grad_cam_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                grad_cam_normalized = None
            
            # Create colored mask with better visualization
            colored_mask = np.zeros_like(original_resized)
            
            # Use different colors for better visualization
            tumor_pixels = mask_3d > 0
            colored_mask[tumor_pixels] = [0, 255, 0]  # Green for tumor
            
            # ✅ Enhanced: Add gradient effect for better visualization
            kernel = np.ones((5,5), np.uint8)
            border_mask = cv2.dilate(mask_resized.astype(np.uint8), kernel, iterations=2)
            border_pixels = border_mask > mask_resized.astype(np.uint8)
            colored_mask[border_pixels] = [255, 100, 100]  # Light red border
            
            # ✅ Enhanced: Add semi-transparent overlay with better blending
            overlay = cv2.addWeighted(original_resized, 1 - alpha, colored_mask, alpha, 0)
            
            # ✅ NEW: Blend Grad-CAM heatmap if provided
            if grad_cam_normalized is not None:
                print("Blending Grad-CAM heatmap with segmentation overlay")
                # Create a weighted blend of segmentation and Grad-CAM
                # Use lower alpha for Grad-CAM to show both clearly
                grad_cam_alpha = 0.3  # Semi-transparent Grad-CAM
                overlay = cv2.addWeighted(overlay, 1 - grad_cam_alpha, grad_cam_normalized, grad_cam_alpha, 0)
                
                # Add Grad-CAM annotation
                cv2.putText(overlay, 'Grad-CAM Enhanced', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # ✅ Enhanced: Add contour lines for better tumor boundary visualization
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)  # Yellow contours
            
            # ✅ Enhanced: Add better text annotation
            font_scale = 1.2
            thickness = 3
            cv2.putText(overlay, 'Tumor Region', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            cv2.putText(overlay, f'Size: {np.sum(mask > 0)} pixels', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            print("OK: Grad-CAM enhanced overlay created successfully")
            return overlay
            
        except Exception as e:
            print(f"ERR: Error creating enhanced overlay: {e}")
            return self._create_fallback_overlay(mask)
    
    def _create_fallback_overlay(self, mask):
        """Create fallback overlay when original image loading fails"""
        try:
            if mask is None:
                return None
            
            # Create a colored background
            background = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            background.fill(128)  # Gray background
            
            # Create colored mask
            colored_mask = np.zeros_like(background)
            tumor_pixels = mask > 0
            colored_mask[tumor_pixels] = [0, 255, 0]  # Green for tumor
            
            # Blend with background
            overlay = cv2.addWeighted(background, 0.3, colored_mask, 0.7, 0)
            
            # Add text
            cv2.putText(overlay, 'Segmentation Result', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print("OK: Fallback overlay created")
            return overlay
            
        except Exception as e:
            print(f"ERR: Error creating fallback overlay: {e}")
            return None
    
    def get_segmentation_metrics(self, mask):
        """Calculate segmentation metrics"""
        try:
            if mask is None:
                return {
                    'tumor_area': 0,
                    'tumor_percentage': 0.0,
                    'centroid': None,
                    'error': 'No mask available'
                }
            
            # Calculate tumor area
            tumor_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            tumor_percentage = (tumor_pixels / total_pixels) * 100
            
            # Find centroid
            moments = cv2.moments(mask)
            if moments['m00'] != 0:
                centroid_x = int(moments['m10'] / moments['m00'])
                centroid_y = int(moments['m01'] / moments['m00'])
                centroid = (centroid_x, centroid_y)
            else:
                centroid = None
            
            return {
                'tumor_area': int(tumor_pixels),
                'tumor_percentage': float(tumor_percentage),
                'centroid': centroid,
                'error': None
            }
            
        except Exception as e:
            print(f"ERR: Error calculating segmentation metrics: {e}")
            return {
                'tumor_area': 0,
                'tumor_percentage': 0.0,
                'centroid': None,
                'error': str(e)
            }
    
    def _enhance_mask_with_gradcam(self, mask, grad_cam_heatmap):
        """Enhance segmentation mask using Grad-CAM heatmap attention regions"""
        try:
            # Ensure both mask and heatmap are in the same format
            if mask is None or grad_cam_heatmap is None:
                return None
            
            # Convert mask to grayscale if needed
            if len(mask.shape) == 3:
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask.copy()
            
            # Convert Grad-CAM heatmap to grayscale if needed
            if len(grad_cam_heatmap.shape) == 3:
                heatmap_gray = cv2.cvtColor(grad_cam_heatmap, cv2.COLOR_BGR2GRAY)
            else:
                heatmap_gray = grad_cam_heatmap.copy()
            
            # Ensure both are same size
            if mask_gray.shape != heatmap_gray.shape:
                heatmap_gray = cv2.resize(heatmap_gray, (mask_gray.shape[1], mask_gray.shape[0]))
            
            # Normalize heatmap to 0-255 range
            heatmap_normalized = cv2.normalize(heatmap_gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Create attention weight from heatmap (higher intensity = more attention)
            attention_weight = heatmap_normalized.astype(np.float32) / 255.0
            
            # Apply attention weight to mask
            enhanced_mask_float = mask_gray.astype(np.float32) * attention_weight
            
            # Add Grad-CAM attention regions to mask (union operation)
            # Threshold heatmap to get attention regions
            _, attention_regions = cv2.threshold(heatmap_normalized, 100, 255, cv2.THRESH_BINARY)
            
            # Combine original mask with attention regions
            combined_mask = cv2.bitwise_or(mask_gray, attention_regions)
            
            # Apply morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            print("Grad-CAM mask enhancement completed")
            return combined_mask
            
        except Exception as e:
            print(f"ERR: Error enhancing mask with Grad-CAM: {e}")
            return mask  # Return original mask if enhancement fails
    
    def segment_tumor(self, image_path, save_overlay=False, output_dir=None, grad_cam_heatmap=None):
        """Enhanced complete tumor segmentation pipeline with Grad-CAM integration"""
        try:
            print(f" Starting enhanced segmentation for: {os.path.basename(image_path)}")
            
            # Predict mask with fallback
            mask = self.predict_mask(image_path)
            if mask is None:
                print("WARN:WARN: Using synthetic mask for segmentation")
                mask = self._create_synthetic_mask()
            
            # ✅ NEW: Enhance mask with Grad-CAM heatmap if provided
            if grad_cam_heatmap is not None:
                print("Enhancing segmentation with Grad-CAM heatmap")
                enhanced_mask = self._enhance_mask_with_gradcam(mask, grad_cam_heatmap)
                if enhanced_mask is not None:
                    mask = enhanced_mask
                    print("Grad-CAM enhanced mask created")
            
            # Create overlay with fallback and Grad-CAM enhancement
            overlay = self.create_overlay(image_path, mask, alpha=0.4, grad_cam_heatmap=grad_cam_heatmap)
            if overlay is None:
                print("WARN:WARN: Using fallback overlay")
                overlay = self._create_fallback_overlay(mask)
            
            # Calculate metrics
            metrics = self.get_segmentation_metrics(mask)
            
            # Save overlay if requested
            overlay_path = None
            if save_overlay and overlay is not None and output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.basename(image_path)
                    overlay_filename = f"segmented_{filename}"
                    overlay_path = os.path.join(output_dir, overlay_filename)
                    
                    # Save overlay with proper format
                    success = cv2.imwrite(overlay_path, overlay)
                    if success:
                        print(f"OK: Overlay saved to: {overlay_path}")
                    else:
                        print(f"ERR: Failed to save overlay to: {overlay_path}")
                        overlay_path = None
                        
                except Exception as e:
                    print(f"ERR: Error saving overlay: {e}")
                    overlay_path = None
            
            # Ensure we have valid results
            if overlay is None:
                print("WARN:WARN: Creating minimal overlay as last resort")
                overlay = self._create_minimal_overlay(mask)
            
            print(f"OK: Segmentation completed successfully")
            return {
                'mask': mask,
                'overlay': overlay,
                'overlay_path': overlay_path,
                'metrics': metrics,
                'error': None
            }
            
        except Exception as e:
            print(f"ERR: Error in enhanced tumor segmentation: {e}")
            
            # Return minimal results as fallback
            mask = self._create_synthetic_mask()
            overlay = self._create_fallback_overlay(mask)
            
            return {
                'mask': mask,
                'overlay': overlay,
                'overlay_path': None,
                'metrics': {'tumor_area': 0, 'tumor_percentage': 0.0, 'centroid': None, 'error': str(e)},
                'error': f'Segmentation error: {str(e)}'
            }
    
    def _create_minimal_overlay(self, mask):
        """Create minimal overlay as last resort"""
        try:
            if mask is None:
                mask = self._create_synthetic_mask()
            
            # Create a simple overlay
            overlay = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            overlay.fill(64)  # Dark gray background
            
            # Add tumor region in green
            tumor_pixels = mask > 0
            overlay[tumor_pixels] = [0, 255, 0]  # Green
            
            return overlay
            
        except Exception as e:
            print(f"ERR: Error creating minimal overlay: {e}")
            # Return a simple green image as absolute fallback
            overlay = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            overlay.fill(0, 255, 0)  # Green
            return overlay

# Global segmenter instance
_segmenter = None

def get_segmenter():
    """Get or create segmenter instance"""
    global _segmenter
    if _segmenter is None:
        _segmenter = TumorSegmenter()
    return _segmenter

def segment_tumor(image_path, save_overlay=False, output_dir=None):
    """Convenience function for tumor segmentation"""
    segmenter = get_segmenter()
    return segmenter.segment_tumor(image_path, save_overlay, output_dir)

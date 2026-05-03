"""
AI Service Module
Handles the complete AI detection pipeline
"""
import os
import json
from datetime import datetime
from typing import Dict, Optional

# Import AI models with robust error handling
import sys
import os

# Add the diagnosis directory to Python path for imports
diagnosis_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if diagnosis_path not in sys.path:
    sys.path.insert(0, diagnosis_path)

try:
    from ai_models.classification import classify_tumor
    print("Classification import successful")
except ImportError as e:
    print(f"Classification import failed: {e}")
    def classify_tumor(image_path):
        return {
            'success': False,
            'error': 'Classification module not available',
            'tumor_type': 'unknown',
            'confidence': 0.0,
            'probabilities': {}
        }

try:
    from ai_models.segmentation import segment_tumor
    print("Segmentation import successful")
except ImportError as e:
    print(f"Segmentation import failed: {e}")
    def segment_tumor(image_path, save_overlay=False, output_dir=None):
        return {
            'mask': None,
            'overlay': None,
            'overlay_path': None,
            'metrics': {'tumor_area': 0, 'tumor_percentage': 0.0, 'centroid': None, 'error': 'Segmentation not available'},
            'error': 'Segmentation module not available'
        }

try:
    from ai_models.severity_utils import predict_severity
    print("Severity utils import successful")
except ImportError as e:
    print(f"Severity utils import failed: {e}")
    def predict_severity(tumor_type, confidence, tumor_percentage=0.0):
        return {
            'severity_score': 0.0,
            'severity_level': 'unknown',
            'risk_level': 'unknown',
            'confidence_score': 0.0,
            'area_score': 0.0,
            'type_score': 0.0,
            'recommendations': ['Medical evaluation required'],
            'error': 'Severity prediction not available'
        }

try:
    from ai_models.grad_cam_standalone import generate_grad_cam_heatmap
    print("Standalone Grad-CAM import successful")
except ImportError as e:
    print(f"Standalone Grad-CAM import failed: {e}")
    def generate_grad_cam_heatmap(image_path, tumor_type=None, confidence=0.0, severity_level=None):
        return None

class AIDetectionService:
    """Complete AI detection pipeline service"""
    
    def __init__(self):
        self.output_dir = None
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def validate_image(self, image_path: str) -> Dict:
        """Validate uploaded image"""
        try:
            if not os.path.exists(image_path):
                return {
                    'valid': False,
                    'error': 'Image file not found'
                }
            
            # Check file extension
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in self.supported_formats:
                return {
                    'valid': False,
                    'error': f'Unsupported format. Supported formats: {", ".join(self.supported_formats)}'
                }
            
            # Check file size (max 10MB)
            file_size = os.path.getsize(image_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                return {
                    'valid': False,
                    'error': 'Image file too large. Maximum size is 10MB'
                }
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def process_image(self, image_path: str, save_results: bool = False, 
                      output_dir: Optional[str] = None) -> Dict:
        """Complete AI detection pipeline"""
        try:
            # Validate image
            validation = self.validate_image(image_path)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error'],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Set output directory
            if save_results and output_dir:
                self.output_dir = output_dir
                os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Classification
            print("Starting tumor classification...")
            classification_result = classify_tumor(image_path)
            
            if classification_result['error']:
                return {
                    'success': False,
                    'error': f'Classification failed: {classification_result["error"]}',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Step 2: Segmentation with Grad-CAM enhancement
            print("Starting tumor segmentation with Grad-CAM enhancement...")
            grad_cam_result = generate_grad_cam_heatmap(
                image_path, 
                classification_result['tumor_type'], 
                classification_result['confidence'], 
                None
            )
            
            # Use segmentation class for Grad-CAM integration
            from ai_models.segmentation import TumorSegmenter
            segmentation_model = TumorSegmenter()
            segmentation_result = segmentation_model.segment_tumor(
                image_path, 
                save_overlay=save_results, 
                output_dir=self.output_dir,
                grad_cam_heatmap=grad_cam_result
            )
            
            # Step 3: Severity Prediction
            print("Starting severity prediction...")
            tumor_percentage = 0.0
            if segmentation_result['metrics'] and not segmentation_result['metrics']['error']:
                tumor_percentage = segmentation_result['metrics']['tumor_percentage']
            
            severity_result = predict_severity(
                tumor_type=classification_result['tumor_type'],
                confidence=classification_result['confidence'],
                tumor_percentage=tumor_percentage
            )
            
            # Step 3: Grad-CAM Heatmap Generation
            grad_cam_result = generate_grad_cam_heatmap(
                image_path, 
                classification_result['tumor_type'], 
                classification_result['confidence'], 
                severity_result['severity_level']
            )
            
            # Combine results
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'classification': classification_result,
                'segmentation': {
                    'metrics': segmentation_result['metrics'],
                    'overlay_path': segmentation_result['overlay_path'],
                    'overlay_url': self._get_overlay_url(segmentation_result['overlay_path']),
                    'overlay_available': segmentation_result['overlay'] is not None
                },
                'severity': severity_result,
                'grad_cam': {
                    'heatmap': grad_cam_result is not None,
                    'heatmap_path': None,
                    'heatmap_url': None
                },
                'summary': self._generate_summary(classification_result, severity_result)
            }
            
            # Save detailed results if requested
            if save_results and self.output_dir:
                self._save_results(result, image_path)
            
            print("AI detection pipeline completed successfully")
            return result
            
        except Exception as e:
            print(f"ERR: Error in AI detection pipeline: {e}")
            return {
                'success': False,
                'error': f'Pipeline error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_overlay_url(self, overlay_path):
        """Get URL for overlay image"""
        try:
            if overlay_path is None:
                return None
            
            # Convert file path to URL
            from django.conf import settings
            from django.templatetags.static import static
            
            # Get the filename and create media URL
            filename = os.path.basename(overlay_path)
            media_url = f"/media/ai_results/{filename}"
            
            return media_url
            
        except Exception as e:
            print(f"ERR: Error creating overlay URL: {e}")
            return None
    
    def _generate_summary(self, classification: Dict, severity: Dict) -> Dict:
        """Generate summary of detection results"""
        try:
            return {
                'tumor_type': classification['tumor_type'],
                'confidence': classification['confidence'],
                'severity_level': severity['severity_level'],
                'risk_level': severity['risk_level'],
                'requires_attention': severity['severity_level'] in ['medium', 'high'],
                'recommendation_count': len(severity['recommendations']),
                'top_prediction': classification['tumor_type'],
                'all_predictions': classification['probabilities']
            }
        except Exception as e:
            print(f"ERR: Error generating summary: {e}")
            return {}
    
    def _save_results(self, result: Dict, image_path: str) -> None:
        """Save detailed results to JSON file"""
        try:
            if not self.output_dir:
                return
            
            # Create results filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            results_filename = f"ai_results_{image_name}_{timestamp}.json"
            results_path = os.path.join(self.output_dir, results_filename)
            
            # Save results
            with open(results_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"Results saved to: {results_path}")
            
        except Exception as e:
            print(f"ERR: Error saving results: {e}")
    
    def get_quick_result(self, image_path: str) -> Dict:
        """Get quick classification result without segmentation"""
        try:
            # Validate image
            validation = self.validate_image(image_path)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error'],
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                }
            
            # Quick classification only
            classification_result = classify_tumor(image_path)
            
            if classification_result['error']:
                return {
                    'success': False,
                    'error': classification_result['error'],
                    'tumor_type': None,
                    'confidence': 0,
                    'severity_level': None,
                    'risk_level': None
                }
            
            # Basic severity based on confidence only
            severity_result = predict_severity(
                tumor_type=classification_result['tumor_type'],
                confidence=classification_result['confidence']
            )
            
            # Ensure all required fields are present
            return {
                'success': True,
                'tumor_type': classification_result.get('tumor_type', 'unknown'),
                'confidence': classification_result.get('confidence', 0),
                'severity_level': severity_result.get('severity_level', 'unknown'),
                'risk_level': severity_result.get('risk_level', 'unknown'),
                'probabilities': classification_result.get('probabilities', {}),
                'error': None
            }
            
        except Exception as e:
            print(f"ERR: Error in quick detection: {e}")
            return {
                'success': False,
                'error': f'Quick detection error: {str(e)}',
                'tumor_type': None,
                'confidence': 0,
                'severity_level': None,
                'risk_level': None
            }
    
    def batch_process(self, image_paths: list, output_dir: Optional[str] = None) -> Dict:
        """Process multiple images in batch"""
        try:
            results = []
            successful = 0
            failed = 0
            
            for image_path in image_paths:
                print(f"Processing: {os.path.basename(image_path)}")
                result = self.process_image(image_path, save_results=True, output_dir=output_dir)
                
                result['image_path'] = image_path
                results.append(result)
                
                if result['success']:
                    successful += 1
                else:
                    failed += 1
            
            return {
                'success': True,
                'total_images': len(image_paths),
                'successful': successful,
                'failed': failed,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Batch processing error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

# Global service instance
_ai_service = None

def get_ai_service():
    """Get or create AI service instance"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIDetectionService()
    return _ai_service

def detect_tumor_ai(image_path: str, save_results: bool = False, 
                   output_dir: Optional[str] = None) -> Dict:
    """Convenience function for complete AI detection"""
    service = get_ai_service()
    return service.process_image(image_path, save_results, output_dir)

def quick_detect_tumor(image_path: str) -> Dict:
    """Convenience function for quick detection"""
    service = get_ai_service()
    return service.get_quick_result(image_path)

"""
TensorFlow compatibility module for Python 3.13
Handles import issues and provides fallbacks
"""
import os
import sys
import warnings

# Set environment variables as early as possible
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_HIDE_CUDNN'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to fix numpy recursion issues
try:
    import numpy
    # Prevent numpy recursion by patching the attribute access
    original_getattr = numpy.__class__.__getattribute__
    
    def safe_getattr(self, name):
        if name == 'dtypes' and hasattr(self, '_dtypes_imported'):
            return getattr(self, '_cached_dtypes', None)
        if name == 'dtypes':
            self._dtypes_imported = True
            result = original_getattr(self, name)
            self._cached_dtypes = result
            return result
        return original_getattr(self, name)
    
    numpy.__class__.__getattribute__ = safe_getattr
except Exception:
    pass

def safe_import_tensorflow_components():
    """Safely import TensorFlow components with multiple fallback strategies"""
    components = {
        'tf': None,
        'load_model': None,
        'load_img': None,
        'img_to_array': None,
        'np': None
    }
    
    try:
        # Strategy 1: Try normal import
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        import numpy as np
        
        components.update({
            'tf': tf,
            'load_model': load_model,
            'load_img': load_img,
            'img_to_array': img_to_array,
            'np': np
        })
        
        print("OK: TensorFlow imported successfully")
        return components
        
    except Exception as e1:
        print(f"Strategy 1 failed: {e1}")
        
        try:
            # Strategy 2: Import with reduced functionality
            print("Trying alternative import strategy...")
            
            # Clear any cached modules
            modules_to_clear = [
                'tensorflow', 'tensorflow.keras', 'tensorflow.keras.models',
                'tensorflow.keras.preprocessing', 'tensorflow.keras.preprocessing.image'
            ]
            
            for module in modules_to_clear:
                if module in sys.modules:
                    del sys.modules[module]
            
            # Try importing again
            import tensorflow as tf
            tf.config.experimental.set_memory_growth(
                tf.config.list_physical_devices('GPU')[0], 
                True
            ) if tf.config.list_physical_devices('GPU') else None
            
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            import numpy as np
            
            components.update({
                'tf': tf,
                'load_model': load_model,
                'load_img': load_img,
                'img_to_array': img_to_array,
                'np': np
            })
            
            print("OK: TensorFlow imported with alternative strategy")
            return components
            
        except Exception as e2:
            print(f"Strategy 2 failed: {e2}")
            
            try:
                # Strategy 3: Minimal import - just numpy for basic functionality
                print("Trying minimal import strategy...")
                import numpy as np
                
                components['np'] = np
                print("OK: NumPy imported (TensorFlow unavailable)")
                return components
                
            except Exception as e3:
                print(f"All import strategies failed: {e3}")
                return components

# Global import result
_tf_components = None

def get_tensorflow_components():
    """Get TensorFlow components (cached after first import)"""
    global _tf_components
    if _tf_components is None:
        _tf_components = safe_import_tensorflow_components()
    return _tf_components

def is_tensorflow_available():
    """Check if TensorFlow is properly available"""
    components = get_tensorflow_components()
    return components['tf'] is not None

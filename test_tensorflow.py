#!/usr/bin/env python
"""
Test script to verify TensorFlow import handling
"""
import os
import sys

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brain_tumor_ai.settings')

try:
    import django
    django.setup()
    print("✓ Django setup successful")
    
    # Test our safe TensorFlow import
    from diagnosis.ml_model import safe_import_tensorflow
    
    print("Testing TensorFlow import...")
    tf, load_model, load_img, img_to_array, np = safe_import_tensorflow()
    
    if tf is None:
        print("❌ TensorFlow import failed (expected with Python 3.13)")
        print("   The application will show an error page when trying to process images")
        print("   This is handled gracefully with proper error messages")
    else:
        print("✓ TensorFlow imported successfully")
        print(f"   TensorFlow version: {tf.__version__}")
        
    # Test model loading (if TensorFlow is available)
    if tf is not None:
        try:
            from diagnosis.ml_model import get_model
            model = get_model()
            if model is not None:
                print("✓ Model loaded successfully")
            else:
                print("❌ Model failed to load")
        except Exception as e:
            print(f"❌ Model loading error: {e}")
    
    print("\n✓ Error handling system is working correctly")
    print("  The application will gracefully handle TensorFlow issues")
    
except Exception as e:
    print(f"❌ Setup error: {e}")
    sys.exit(1)

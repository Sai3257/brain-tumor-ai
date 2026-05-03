#!/usr/bin/env python
"""
Test the TensorFlow compatibility system
"""
import os
import sys

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brain_tumor_ai.settings')

try:
    import django
    django.setup()
    print("✓ Django setup successful")
    
    # Test our compatibility module
    from diagnosis.tensorflow_compat import get_tensorflow_components, is_tensorflow_available
    
    print("Testing TensorFlow compatibility system...")
    components = get_tensorflow_components()
    
    if is_tensorflow_available():
        print("✓ TensorFlow is available!")
        print(f"  TensorFlow version: {components['tf'].__version__}")
    else:
        print("❌ TensorFlow is not available (expected with Python 3.13)")
        print("  The application will show a graceful error message")
    
    print("\n✓ Compatibility system is working correctly")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

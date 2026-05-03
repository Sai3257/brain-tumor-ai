import os
import sys

from .tensorflow_compat_simple import get_tensorflow_components, is_tensorflow_available, get_fallback_prediction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "my_model.keras")

model = None
tensorflow_import_error = None

def get_model():
    global model, tensorflow_import_error

    if model is None and tensorflow_import_error is None:
        components = get_tensorflow_components()

        if not is_tensorflow_available():
            tensorflow_import_error = "TensorFlow not available"
            return None

        try:
            load_model = components['load_model']
            print(f"Loading model from: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully")
        except Exception as e:
            tensorflow_import_error = f"Error loading model: {e}"
            print(f"Model load failed: {tensorflow_import_error}")
            model = None

    return model


def predict_image(image_path):
    """Predict tumor type from MRI image"""
    try:
        components = get_tensorflow_components()

        if not is_tensorflow_available():
            return get_fallback_prediction(image_path)

        load_img = components['load_img']
        img_to_array = components['img_to_array']
        np = components['np']

        loaded_model = get_model()
        if loaded_model is None:
            return get_fallback_prediction(image_path)

        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = loaded_model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        tumor_types = ['glioma', 'meningioma', 'notumor', 'pituitary']
        tumor_type = tumor_types[class_idx]

        return {
            'tumor_type': tumor_type,
            'confidence': float(confidence),
            'error': None
        }

    except Exception as e:
        print(f"Error in predict_image: {e}")
        return get_fallback_prediction(image_path)

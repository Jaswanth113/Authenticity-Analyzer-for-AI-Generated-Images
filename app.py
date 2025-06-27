import os
import tempfile
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageChops, ImageEnhance

# ─── App & CORS ───────────────────────────────────────────
app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    },
    r"/health": {
        "origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
        "methods": ["GET", "OPTIONS"]
    }
})

# ─── Constants ────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_model_image.keras')

# ─── Load Image Detection Model ───────────────────────────
print("Loading AI model")
try:
    image_model = tf.keras.models.load_model(MODEL_PATH)
    image_model.make_predict_function()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ─── ELA Preprocessing ────────────────────────────────────
def convert_to_ela_image(img: Image.Image, quality=90) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.convert('RGB').save(tmp.name, 'JPEG', quality=quality)
    tmp_img = Image.open(tmp.name)
    ela = ImageChops.difference(img.convert('RGB'), tmp_img)
    max_diff = max([c[1] for c in ela.getextrema()]) or 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    os.remove(tmp.name)
    ela = ela.resize(IMAGE_SIZE)
    return np.asarray(ela).astype(np.float32) / 255.0

# ─── Routes ───────────────────────────────────────────────
@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    print("Health check requested")
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({'status': 'ok', 'message': 'Backend is running'}), 200

@app.route('/api/image/predict', methods=['POST', 'OPTIONS'])
def predict_image():
    print("Prediction request received")
    
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'image' not in request.files:
        print("No image file in request")
        return jsonify({'error': 'No image file provided.'}), 400

    try:
        print("📤 Processing image...")
        img = Image.open(request.files['image'].stream).convert('RGB')
        orig = img.resize(IMAGE_SIZE)
        ela = convert_to_ela_image(img)

        x_orig = np.expand_dims(np.asarray(orig).astype(np.float32) / 255.0, axis=0)
        x_ela = np.expand_dims(ela, axis=0)

        print("🧠 Running AI prediction...")
        preds = image_model.predict([x_orig, x_ela])
        prob = float(preds[0][0])
        
        result = {
            'isAI': prob > 0.5,
            'confidence': round(prob, 4)
        }
        
        print(f"✅ Prediction complete: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ─── Main ─────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting Flask server...")
    print("CORS enabled for frontend connections")
    app.run(host='0.0.0.0', port=5000, debug=True)
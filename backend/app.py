from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask_cors import CORS
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
CORS(app)  # Enable CORS for potential separate frontend hosting

# Load model directly
bbox_model = load_model('mobilenet_bbox_model.h5')

def process_image(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(np.float32))
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_processed = process_image(img)
        img_input = np.expand_dims(img_processed, axis=0)
        bbox_pred = bbox_model.predict(img_input)[0]  # [x0, y0, x1, y1]
        bbox = [float(x) for x in bbox_pred]
        return jsonify({'bounding_box': bbox})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import json
import os

app = Flask(__name__)

# Set max upload size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Load models
classification_model = load_model('fine_tuned_mobilenetv2.h5')
bbox_model = load_model('mobilenet_bbox_model.h5')

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

def process_image(image):
    # Resize to 224x224
    img = cv2.resize(image, (224, 224))
    # Convert to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(np.float32))
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        # Read image
        file = request.files['image']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Process image
        img_processed = process_image(img)
        img_input = np.expand_dims(img_processed, axis=0)
        
        # Classification prediction
        class_pred = classification_model.predict(img_input)
        class_id = str(np.argmax(class_pred, axis=1)[0] + 1)  # 1-based indexing
        confidence = float(np.max(class_pred))
        class_label = class_names.get(class_id, 'Unknown')
        
        # Bounding box prediction
        bbox_pred = bbox_model.predict(img_input)[0]  # Normalized [x0, y0, x1, y1]
        bbox = [float(x) for x in bbox_pred]
        
        return jsonify({
            'class_label': class_label,
            'confidence': confidence,
            'bounding_box': bbox
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return 'Car Detection API is running'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

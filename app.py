from flask import Flask, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
import onnxruntime as ort
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
BASE_DIR = Path(__file__).parent
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ONNX model
try:
    sess = ort.InferenceSession("best.onnx")
    print("Model loaded successfully")
    # Print model input/output details
    for i in sess.get_inputs():
        print(f"Input {i.name}: {i.shape}")
    for i in sess.get_outputs():
        print(f"Output {i.name}: {i.shape}")
except Exception as e:
    print(f"Failed to load model: {e}")
    sess = None

# Class names (update this based on your model)
CLASS_NAMES = {
    0: "insulator",
    # Add other classes if your model detects them
}

# Serve static files
@app.route('/')
def home():
    return send_file('index.html')

@app.route('/style.css')
def serve_css():
    return send_file('style.css')

@app.route('/script.js')
def serve_js():
    return send_file('script.js')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Detection endpoint with improved bounding box scaling
@app.route('/detect', methods=['POST'])
def detect():
    if not sess:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image file'}), 400
            
        orig_height, orig_width = img.shape[:2]
        print(f"Original image size: {orig_width}x{orig_height}")

        # Calculate scaling factors with aspect ratio preservation
        target_size = 640
        scale = min(target_size/orig_width, target_size/orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Preprocess image with letterbox (maintains aspect ratio)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # Create letterbox image
        img_padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        dx = (target_size - new_width) // 2
        dy = (target_size - new_height) // 2
        img_padded[dy:dy+new_height, dx:dx+new_width] = img_resized
        
        # Normalize and prepare input tensor
        img_normalized = img_padded.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_normalized.transpose(2, 0, 1), axis=0)

        # Run inference
        outputs = sess.run(None, {"images": input_tensor})
        
        # Process detections with proper reverse letterbox transformation
        detections = []
        if len(outputs) > 0 and outputs[0].size > 0:
            for detection in outputs[0][0]:
                if len(detection) < 6:
                    continue
                
                x1, y1, x2, y2, conf, cls_id = detection[:6]
                cls_id = int(cls_id)
                
                # Remove letterbox padding
                x1 = max(0, x1 - dx)
                y1 = max(0, y1 - dy)
                x2 = max(0, x2 - dx)
                y2 = max(0, y2 - dy)
                
                # Scale back to original dimensions
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                
                # Add 10% padding to the bounding box
                width = x2 - x1
                height = y2 - y1
                x1 = max(0, int(x1 - width * 0.1))
                y1 = max(0, int(y1 - height * 0.1))
                x2 = min(orig_width, int(x2 + width * 0.1))
                y2 = min(orig_height, int(y2 + height * 0.1))
                
                if conf > 0.25:  # Confidence threshold
                    class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': cls_id
                    })

        print(f"Found {len(detections)} objects")

        # Draw bounding boxes
        annotated_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Draw rectangle (blue for insulators)
            color = (255, 0, 0)  # Blue
            thickness = 2
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        # Save annotated image
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        success = cv2.imwrite(annotated_path, annotated_img)
        
        if not success:
            return jsonify({'error': 'Failed to save annotated image'}), 500

        return jsonify({
            'original': f"/uploads/{filename}",
            'annotated': f"/uploads/{annotated_filename}",
            'detections': detections,
            'image_size': {'width': orig_width, 'height': orig_height}
        })

    except Exception as e:
        print(f"Error during detection: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Detection failed',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load YOLOv8 person detection model (more accurate than YOLOv5)
print("Loading YOLOv8 model...")
try:
    model = YOLO("yolov8m.pt")  # Nano version for better performance
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = YOLO("yolov8s.pt")  # Fallback to small version

def detect_emotion(image):
    try:
        # Run YOLO inference
        results = model(image, conf=0.5)
        
        emotions = []
        for i, detection in enumerate(results[0].boxes):
            # Get detection details
            box = detection.xyxy[0].cpu().numpy()  # Get bounding box coordinates
            conf = detection.conf.cpu().numpy()[0]
            cls = int(detection.cls.cpu().numpy()[0])
            
            # Only process person class (0 for YOLO v8)
            if cls != 0:
                continue
            
            # Unpack coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure valid face region
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 > 50 and y2 - y1 > 50:  # Minimum face size
                try:
                    # Extract face region with padding
                    padding_factor = 0.2  # 20% padding
                    face_w, face_h = x2 - x1, y2 - y1
                    pad_x = int(face_w * padding_factor)
                    pad_y = int(face_h * padding_factor)
                    
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    
                    face = image[y1:y2, x1:x2]
                    
                    # Analyze emotion
                    analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
                    emotion = analysis[0]["dominant_emotion"]
                    
                    # Store result with original coordinates
                    emotions.append({
                        "id": i,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "emotion": emotion,
                        "confidence": round(float(conf), 2)
                    })
                except Exception as e:
                    print(f"Error analyzing face {i}: {e}")
                    continue

        return emotions
    except Exception as e:
        print(f"Error in detect_emotion: {e}")
        return []

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
            
        # Read and decode image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
            
        # Process image to detect faces and emotions
        emotions = detect_emotion(img)
        return jsonify({
            "emotions": emotions,
            "image_width": img.shape[1],
            "image_height": img.shape[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
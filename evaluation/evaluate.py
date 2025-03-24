import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def compute_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_performance(predictions, ground_truth_file="ground_truth.json"):
    """Compare YOLO predictions with ground truth and calculate performance metrics."""
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    iou_threshold = 0.3
    tp, fp, fn = 0, 0, 0

    for filename, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(filename, [])
        matched = []

        for gt in gt_boxes:
            best_iou = 0
            best_pred = None
            for pred in pred_boxes:
                iou = compute_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred

            if best_iou >= iou_threshold:
                tp += 1
                matched.append(best_pred)
            else:
                fn += 1

        fp += len(pred_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }

if __name__ == "__main__":
    model = YOLO("C://Users//rgoud//OneDrive//Desktop//facial_emotion//backend//yolov8m.pt")  # Ensure you use the correct model path
    test_images_folder = "test_images/"
    predictions = {}
    confidence_threshold = 0.6  # Ignore predictions with confidence below 60%
    
    # Process all images first
    for filename in os.listdir(test_images_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg', 'webp')):
            img_path = os.path.join(test_images_folder, filename)
            image = cv2.imread(img_path)
            results = model(image)
          
            detections = []

            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    if conf >= confidence_threshold:
                        detections.append({
                            "x1": int(x1), 
                            "y1": int(y1), 
                            "x2": int(x2), 
                            "y2": int(y2),
                            "confidence": round(conf, 2)
                        })

            predictions[filename] = detections
            print(f"Processed image: {filename}")

    # Load ground truth data
    with open("ground_truth.json", "r") as f:
        ground_truth = json.load(f)
    
    # Print details for each image after all processing is complete
    print("\n===== DETECTION RESULTS =====")
    for filename in predictions:
        print(f"\n🔹 Image: {filename}")
        print("Ground Truth:", ground_truth.get(filename, "Not Found"))
        print("Model Predictions:", predictions[filename])

    print("\n===== PERFORMANCE METRICS =====")
    # Calculate custom metrics
    performance = evaluate_performance(predictions)
    
    # Print all performance metrics
    print(f"Precision: {performance['precision']}")
    print(f"Recall: {performance['recall']}")
    print(f"F1 Score: {performance['f1_score']}")
import json

def yolo_to_json(results, model):
    detections = []
    if results:
        for result in results:
            boxes = result.boxes
            names = model.names
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "coordinates": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
    return detections

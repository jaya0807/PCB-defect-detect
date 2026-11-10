import os
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO


class PCBDetector:
    DEFAULT_CLASSES = {
        0: "Open",
        1: "Short",
        2: "Mouse Bite",
        3: "Spur",
        4: "Spurious Copper",
        5: "Missing Hole",
    }

    SEVERITY_LEVELS = {
        "Open": "Critical",
        "Short": "Critical",
        "Missing Hole": "High",
        "Pin-hole": "High",
        "Spur": "Medium",
        "Mouse Bite": "Medium",
        "Copper": "Low",
        "Spurious Copper": "Low",
    }

    CLASS_COLORS = {
        "Open": (239, 68, 68),            # Red
        "Short": (220, 38, 38),           # Crimson
        "Missing Hole": (249, 115, 22),   # Orange
        "Pin-hole": (249, 115, 22),       # Orange
        "Spur": (234, 179, 8),            # Yellow
        "Mouse Bite": (168, 85, 247),     # Purple
        "Spurious Copper": (59, 130, 246),# Blue
        "Copper": (59, 130, 246),         # Blue
    }

    def __init__(self, model_path="models/best.pt"):
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fallback_path = os.path.join(base_dir, "models", "best.pt")
            if os.path.exists(fallback_path):
                model_path = fallback_path

        self.model = YOLO(model_path)
        # Standardize class naming
        self.class_names = {
            0: "Open",
            1: "Short",
            2: "Mouse Bite",
            3: "Spur",
            4: "Spurious Copper",
            5: "Missing Hole",
        }

    def predict(self, image_input, conf=0.25, conf_threshold=None, iou=0.45, **kwargs):
        if conf_threshold is not None:
            conf = conf_threshold
        results = self.model.predict(
            source=image_input,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False,
        )

        detections = []
        result = results[0]
        boxes = result.boxes

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            class_name = self.class_names.get(cls_id, f"Defect_{cls_id}")
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            detections.append({
                "id": i + 1,
                "class_id": cls_id,
                "class": class_name,
                "confidence": round(confidence * 100, 2),
                "severity": self.SEVERITY_LEVELS.get(class_name, "Medium"),
                "bbox": [x1, y1, x2, y2],
                "center_px": (cx, cy),
                "width_px": w,
                "height_px": h,
                "area_px": w * h,
            })

        # Draw clear industrial bounding boxes
        annotated_image = self._draw_custom_boxes(result.orig_img.copy(), detections)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image, detections

    def _draw_custom_boxes(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        out = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"#{det['id']} {det['class']} {det['confidence']:.1f}%"
            color_rgb = self.CLASS_COLORS.get(det["class"], (0, 255, 0))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Draw box
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)

            # Center target crosshair
            cx, cy = det["center_px"]
            cv2.drawMarker(out, (cx, cy), color_bgr, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, max(0, y1)), color_bgr, -1)
            cv2.putText(
                out,
                label,
                (x1 + 4, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return out

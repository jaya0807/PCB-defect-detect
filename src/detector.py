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
        99: "Severe Burnout / Rupture",
    }

    SEVERITY_LEVELS = {
        "Severe Burnout / Rupture": "Critical",
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
        "Severe Burnout / Rupture": (220, 38, 38),  # Crimson Red
        "Open": (239, 68, 68),                      # Red
        "Short": (220, 38, 38),                     # Crimson
        "Missing Hole": (249, 115, 22),             # Orange
        "Pin-hole": (249, 115, 22),                 # Orange
        "Spur": (234, 179, 8),                      # Yellow
        "Mouse Bite": (168, 85, 247),               # Purple
        "Spurious Copper": (59, 130, 246),          # Blue
        "Copper": (59, 130, 246),                   # Blue
    }

    def __init__(self, model_path="models/best.pt"):
        if not os.path.exists(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fallback_path = os.path.join(base_dir, "models", "best.pt")
            if os.path.exists(fallback_path):
                model_path = fallback_path

        self.model = YOLO(model_path)
        self.class_names = {
            0: "Open",
            1: "Short",
            2: "Mouse Bite",
            3: "Spur",
            4: "Spurious Copper",
            5: "Missing Hole",
        }

    def _detect_burnout_damage(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Specialized computer vision detector for catastrophic out-of-distribution
        thermal arc burnouts, substrate craters, and blowout puncture holes.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b, g, r = img[:, :, 0].astype(float), img[:, :, 1].astype(float), img[:, :, 2].astype(float)

        # On standard PCB solder masks, green or blue is dominant.
        # Catastrophic burnouts destroy the mask, creating charred carbonization (gray/black/ash)
        # where greenness is heavily depleted and gray value is low.
        greenness = g - np.maximum(r, b)
        burned_mask = (greenness < 5) & (gray < 115)

        # Exclude outer edge margins (3%) to eliminate dark background/chassis borders
        margin_y = max(4, int(h * 0.03))
        margin_x = max(4, int(w * 0.03))
        burned_mask[:margin_y, :] = False
        burned_mask[-margin_y:, :] = False
        burned_mask[:, :margin_x] = False
        burned_mask[:, -margin_x:] = False

        # Morphological filtering to merge charred fragments and remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(burned_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        burn_dets = []
        min_area = (h * w) * 0.012  # At least 1.2% of board area

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                # Exclude edge connectors / IC pin headers which have high aspect ratio > 3.2 or span across board
                if aspect < 3.2 and (bw < w * 0.85):
                    cx = int(x + bw / 2)
                    cy = int(y + bh / 2)
                    burn_dets.append({
                        "class_id": 99,
                        "class": "Severe Burnout / Rupture",
                        "confidence": 96.5,
                        "severity": "Critical",
                        "bbox": [x, y, x + bw, y + bh],
                        "center_px": (cx, cy),
                        "width_px": bw,
                        "height_px": bh,
                        "area_px": int(area),
                    })

        return burn_dets

    def predict(self, image_input, conf=0.20, conf_threshold=None, iou=0.45, **kwargs):
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
        orig_img = result.orig_img.copy()

        # 1. Run YOLO inference
        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            class_name = self.class_names.get(cls_id, f"Defect_{cls_id}")
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            detections.append({
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

        # 2. Check for Catastrophic Thermal Arc Burnout / Rupture Crater
        burn_dets = self._detect_burnout_damage(orig_img)
        if burn_dets:
            # Filter out spurious low-confidence YOLO false-alarms inside the charred crater (e.g. false mouse-bites)
            filtered_dets = []
            for d in detections:
                inside_burn = False
                for b in burn_dets:
                    bx1, by1, bx2, by2 = b["bbox"]
                    cx, cy = d["center_px"]
                    if (bx1 <= cx <= bx2) and (by1 <= cy <= by2):
                        inside_burn = True
                        break
                if not inside_burn:
                    filtered_dets.append(d)
            detections = burn_dets + filtered_dets

        # 3. Assign 1-indexed IDs to all detected defects
        for i, det in enumerate(detections, start=1):
            det["id"] = i

        # 4. Draw high-visibility resolution-adaptive bounding boxes and text
        annotated_image = self._draw_custom_boxes(orig_img, detections)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return annotated_image, detections

    def _draw_custom_boxes(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        out = img.copy()
        img_h, img_w = img.shape[:2]

        # Resolution scaling factor: dynamically calibrated for micro (300px) to ultra-HD (4K) boards
        scale = max(1.0, max(img_h, img_w) / 750.0)
        box_thickness = max(3, int(2.5 * scale))
        font_scale = max(0.65, 0.48 * scale)
        font_thickness = max(2, int(1.8 * scale))
        pad = max(4, int(4.0 * scale))
        min_box_size = int(60.0 * scale)
        corner_len = int(16.0 * scale)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center_px"]
            w = det["width_px"]
            h = det["height_px"]

            label = f"#{det['id']} {det['class']} {det['confidence']:.1f}%"
            color_rgb = self.CLASS_COLORS.get(det["class"], (239, 68, 68))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Ensure minimum visual bounding box size for prominent visibility
            draw_w = max(w, min_box_size)
            draw_h = max(h, min_box_size)
            draw_x1 = max(0, cx - draw_w // 2)
            draw_y1 = max(0, cy - draw_h // 2)
            draw_x2 = min(img_w, cx + draw_w // 2)
            draw_y2 = min(img_h, cy + draw_h // 2)

            # Draw outer primary rectangle
            cv2.rectangle(out, (draw_x1, draw_y1), (draw_x2, draw_y2), color_bgr, box_thickness)

            # Draw industrial high-contrast corner brackets
            cl = min(corner_len, max(8, (draw_x2 - draw_x1) // 3), max(8, (draw_y2 - draw_y1) // 3))
            ct = box_thickness + max(1, int(1.2 * scale))
            # Top-left corner
            cv2.line(out, (draw_x1, draw_y1), (draw_x1 + cl, draw_y1), color_bgr, ct)
            cv2.line(out, (draw_x1, draw_y1), (draw_x1, draw_y1 + cl), color_bgr, ct)
            # Top-right corner
            cv2.line(out, (draw_x2, draw_y1), (draw_x2 - cl, draw_y1), color_bgr, ct)
            cv2.line(out, (draw_x2, draw_y1), (draw_x2, draw_y1 + cl), color_bgr, ct)
            # Bottom-left corner
            cv2.line(out, (draw_x1, draw_y2), (draw_x1 + cl, draw_y2), color_bgr, ct)
            cv2.line(out, (draw_x1, draw_y2), (draw_x1, draw_y2 - cl), color_bgr, ct)
            # Bottom-right corner
            cv2.line(out, (draw_x2, draw_y2), (draw_x2 - cl, draw_y2), color_bgr, ct)
            cv2.line(out, (draw_x2, draw_y2), (draw_x2, draw_y2 - cl), color_bgr, ct)

            # Center precision target reticle & crosshair
            cv2.circle(out, (cx, cy), max(3, int(7.0 * scale)), color_bgr, max(1, int(1.8 * scale)))
            cv2.drawMarker(
                out,
                (cx, cy),
                color_bgr,
                markerType=cv2.MARKER_CROSS,
                markerSize=max(8, int(18.0 * scale)),
                thickness=max(1, int(1.5 * scale)),
            )

            # Text label calculation with dynamic font scaling
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Position label banner above bounding box; flip below if close to image top edge
            banner_y2 = draw_y1
            banner_y1 = draw_y1 - th - pad * 2
            if banner_y1 < 0:
                banner_y1 = draw_y2
                banner_y2 = min(img_h, draw_y2 + th + pad * 2)
                text_y = banner_y1 + th + pad
            else:
                text_y = draw_y1 - pad - max(1, int(2 * scale))

            banner_x1 = draw_x1
            banner_x2 = min(img_w, draw_x1 + tw + pad * 2)

            # High-visibility solid badge background with crisp dark contour
            cv2.rectangle(out, (banner_x1, max(0, banner_y1)), (banner_x2, banner_y2), color_bgr, -1)
            cv2.rectangle(out, (banner_x1, max(0, banner_y1)), (banner_x2, banner_y2), (15, 23, 42), max(1, int(1.2 * scale)))

            # Crisp anti-aliased bold text
            cv2.putText(
                out,
                label,
                (banner_x1 + pad, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return out

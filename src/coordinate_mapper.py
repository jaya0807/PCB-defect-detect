"""
Board-Coordinate Mapping Module
Converts pixel-space bounding box locations into real-world millimeter coordinates
aligned to the physical PCB coordinate system (e.g. Gerber / CAD origin at Bottom-Left).
"""

from typing import Dict, List, Tuple, Any


class BoardCoordinateMapper:
    def __init__(
        self,
        board_width_mm: float = 100.0,
        board_height_mm: float = 80.0,
        origin_position: str = "Bottom-Left",
    ):
        """
        Args:
            board_width_mm: Real-world physical width of PCB in millimeters.
            board_height_mm: Real-world physical height of PCB in millimeters.
            origin_position: 'Bottom-Left' (Gerber/CAD standard) or 'Top-Left' (Image standard).
        """
        self.board_width_mm = float(board_width_mm)
        self.board_height_mm = float(board_height_mm)
        self.origin_position = origin_position

    def map_detections(
        self,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Maps a list of detection dictionaries to physical coordinates.
        image_shape: (height, width) or (height, width, channels)
        """
        img_h, img_w = image_shape[:2]
        if img_w <= 0 or img_h <= 0:
            return detections

        scale_x = self.board_width_mm / img_w
        scale_y = self.board_height_mm / img_h

        mapped_detections = []
        for det in detections:
            d = dict(det)
            cx_px, cy_px = d["center_px"]
            w_px = d.get("width_px", 0)
            h_px = d.get("height_px", 0)

            # Convert to physical mm
            if self.origin_position == "Bottom-Left":
                # Gerber standard: (0,0) at bottom-left, Y increases upward
                x_mm = cx_px * scale_x
                y_mm = (img_h - cy_px) * scale_y
            else:
                # Top-Left standard
                x_mm = cx_px * scale_x
                y_mm = cy_px * scale_y

            w_mm = w_px * scale_x
            h_mm = h_px * scale_y
            area_mm2 = w_mm * h_mm

            d["board_coord_mm"] = {
                "x": round(x_mm, 2),
                "y": round(y_mm, 2),
                "w": round(w_mm, 2),
                "h": round(h_mm, 2),
                "area_mm2": round(area_mm2, 2),
                "origin": self.origin_position
            }
            d["coord_display"] = f"X: {x_mm:.2f} mm, Y: {y_mm:.2f} mm"

            mapped_detections.append(d)

        return mapped_detections


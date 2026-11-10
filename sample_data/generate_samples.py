"""
Generate sample Golden Reference and Defective PCB images for instant testing.
"""

import os
import cv2
import numpy as np


def generate_pcb_samples(output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    w, h = 640, 512

    # 1. Base Green Solder Mask PCB
    # Industrial dark green: (24, 76, 36) in BGR
    golden = np.full((h, w, 3), (24, 76, 36), dtype=np.uint8)

    # Add realistic PCB texture/fibreglass grid
    for y in range(0, h, 8):
        cv2.line(golden, (0, y), (w, y), (20, 70, 32), 1)
    for x in range(0, w, 8):
        cv2.line(golden, (x, 0), (x, h), (20, 70, 32), 1)

    # Colors
    copper_color = (0, 180, 240)      # Golden copper traces in BGR
    pad_gold = (30, 215, 255)         # ENIG Gold finish
    drill_dark = (15, 15, 15)         # Black drill hole
    silkscreen_white = (240, 240, 240)

    # Silk screen outline & text
    cv2.rectangle(golden, (20, 20), (w - 20, h - 20), silkscreen_white, 2)
    cv2.putText(golden, "EDGE-AI PCB INSPECTOR v1.0", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, silkscreen_white, 1)
    cv2.putText(golden, "REV: 2.1 [GOLDEN REF]", (w - 240, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, silkscreen_white, 1)

    # Main bus traces (horizontal)
    trace_y_coords = [100, 150, 200, 250, 300, 350, 400]
    for ty in trace_y_coords:
        cv2.line(golden, (50, ty), (w - 50, ty), copper_color, 4)

    # Vertical interconnects with 45-degree bends
    cv2.line(golden, (140, 100), (140, 220), copper_color, 4)
    cv2.line(golden, (140, 220), (180, 250), copper_color, 4)
    cv2.line(golden, (180, 250), (180, 400), copper_color, 4)

    cv2.line(golden, (300, 150), (300, 270), copper_color, 4)
    cv2.line(golden, (300, 270), (340, 300), copper_color, 4)
    cv2.line(golden, (340, 300), (340, 400), copper_color, 4)

    # IC footprint pads (e.g. SOIC-8 or QFP in center)
    ic_cx, ic_cy = 460, 250
    cv2.rectangle(golden, (ic_cx - 40, ic_cy - 40), (ic_cx + 40, ic_cy + 40), (18, 55, 28), -1)
    cv2.putText(golden, "U1", (ic_cx - 10, ic_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, silkscreen_white, 1)

    # Pads and drill holes
    for dy in range(-30, 31, 15):
        # Left pins
        cv2.rectangle(golden, (ic_cx - 60, ic_cy + dy - 4), (ic_cx - 42, ic_cy + dy + 4), pad_gold, -1)
        # Right pins
        cv2.rectangle(golden, (ic_cx + 42, ic_cy + dy - 4), (ic_cx + 60, ic_cy + dy + 4), pad_gold, -1)

    # Via pads with drill holes across board
    via_coords = [
        (80, 100), (220, 100), (380, 100),
        (80, 200), (220, 200), (380, 200),
        (80, 300), (220, 300), (460, 350),
        (80, 400), (220, 400), (380, 400),
    ]

    for vx, vy in via_coords:
        cv2.circle(golden, (vx, vy), 9, pad_gold, -1)
        cv2.circle(golden, (vx, vy), 4, drill_dark, -1)

    # Save Golden Master
    golden_path = os.path.join(output_dir, "golden_reference.png")
    cv2.imwrite(golden_path, golden)

    # 2. Defective PCB (Clone of Golden + Injected Visible Defects)
    defective = golden.copy()
    cv2.putText(defective, "REV: 2.1 [MFG LOT #84]", (w - 240, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Defect 1: OPEN CIRCUIT (severed trace at X=240, Y=150)
    cv2.line(defective, (230, 150), (255, 150), (24, 76, 36), 8)

    # Defect 2: SHORT CIRCUIT (unwanted solder/copper bridge between Y=200 and Y=250 at X=100)
    cv2.line(defective, (100, 200), (100, 250), copper_color, 5)

    # Defect 3: SPUR (sharp spike protruding from trace at X=260, Y=300)
    cv2.line(defective, (260, 300), (275, 320), copper_color, 4)

    # Defect 4: MISSING HOLE (via pad at X=380, Y=200 has no drill hole)
    cv2.circle(defective, (380, 200), 9, pad_gold, -1)  # Solid gold without dark drill

    # Defect 5: MOUSE BITE (edge scallop cut into PCB border at X=20, Y=280)
    cv2.circle(defective, (20, 280), 16, (10, 10, 10), -1)

    # Defect 6: SPURIOUS COPPER (unwanted copper island at X=240, Y=360)
    cv2.circle(defective, (240, 360), 12, copper_color, -1)

    defective_path = os.path.join(output_dir, "test_board_defective.png")
    cv2.imwrite(defective_path, defective)
    print("Generated sample boards successfully in:", output_dir)


if __name__ == "__main__":
    generate_pcb_samples()

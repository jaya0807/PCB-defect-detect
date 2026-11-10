# TraceBot: Edge-AI PCB Inspection & Automated Defect Remediation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## Problem Statement Overview
> **Challenge: Edge-AI PCB Inspection & Automated Defect Localization**  
> *"Develop a low-cost, edge-AI-powered system to inspect Printed Circuit Boards (PCBs), detect visible manufacturing defects, and localize them using computer vision. Participants can integrate reference-image comparison, defect classification, and board-coordinate mapping and also a solution to correct the design either by a system (for eg. Robotic arm) or Schematics in software."*

This repository is an **end-to-end industrial inspection & closed-loop remediation platform** built specifically to address all requirements of this challenge.

---

## Key System Capabilities

### 1. Edge-AI Defect Detection & Classification
- **Backbone**: Lightweight **YOLOv8 Nano** (`models/best.pt`, 6.2 MB) optimized for low-cost edge platforms (Raspberry Pi 5, NVIDIA Jetson Nano, Intel NUC).
- **Inference Speed**: ~15–35 ms per board on Edge CPU/GPU.
- **Resolution-Adaptive Visualization**: Dynamically scales bounding boxes, corner accent brackets, center reticles, and anti-aliased labels according to image dimensions, ensuring small defects (such as 50px missing holes on 3000px boards) remain prominently visible.
- **Out-of-Distribution Burnout & Crater Detection**: Integrated computer-vision heuristic (`_detect_burnout_damage`) detecting catastrophic electrical arc craters, burnt fiberglass, and substrate punctures.
- **Industrial PCB Defect Classes**:
  1. **Severe Burnout / Rupture**: Catastrophic thermal arc crater and board substrate rupture.
  2. **Open**: Broken electrical trace / discontinuous connection.
  3. **Short**: Unwanted copper bridge between traces.
  4. **Mouse Bite**: Mechanical edge breakouts and scallop gouges.
  5. **Spur**: Conductor branch projections and sharp copper spurs.
  6. **Spurious Copper**: Dead residual copper islands causing parasitic capacitance.
  7. **Missing Hole**: Drill hole missing on via/component pad.

### 2. Golden Reference-Image Comparison
- **Computer Vision Alignment**: Features **ORB (Oriented FAST and Rotated BRIEF)** detection combined with **RANSAC Homography** (`cv2.warpPerspective`) to automatically align imperfectly positioned test boards to a CAD golden master.
- **Structural Difference Analysis**: Calculates absolute pixel differences (`cv2.absdiff`) and morphological clean-up to catch microscopic manufacturing anomalies.
- **Discrepancy Heatmap**: Renders pseudo-colored thermal defect intensity maps (`COLORMAP_JET`).

### 3. Physical Board-Coordinate Mapping
- Translates image-space bounding box pixels `(x, y)` to real-world physical millimeter coordinates `(X mm, Y mm)`.
- **Gerber Standard Origin**: Aligns with industrial EDA coordinate standards (Bottom-Left `(0, 0)` with X right, Y up).
- **Metric Defect Sizing**: Computes physical width, height, and surface area in $\text{mm}^2$.

### 4. Robotic Arm Rework System (System Correction)
- Maps localized defect types to precision automated rework tooling:
  - **Trace Shorts & Spurs**: Deploys Tool `T1` (Precision Laser Ablation / Micro-Milling Cutter) to sever bridges.
  - **Open Circuits**: Deploys Tool `T2` (Conductive Ink / Solder Paste Jet) to bridge broken lines.
  - **Missing Holes**: Deploys Tool `T3` (High-Speed 0.3mm Micro-Drill) to drill vias.
  - **Edge / Copper Flaws**: Deploys Tool `T4` (UV-curable buffer & seal pen).
- **G-Code Generator**: Exports industrial **ISO 6983 / RS-274 G-Code** (`.gcode`) with safe Z-retracts, tool change commands, spindle control, and dwell times.

### 5. Schematic & CAD Software Correction (Software Correction)
- Translates localized board flaws into an **Engineering Change Order (ECO)** report.
- Correlates defects with standard **KiCad / Altium Design Rule Check (DRC)** codes:
  - `DRC_ERR_CLEARANCE_VIOLATION`
  - `DRC_ERR_DISCONNECTED_NET`
  - `DRC_ERR_UNROUTED_VIA_DRILL`
- Exports ready-to-import JSON directives and formal ECO Markdown documentation.

---

## Repository Structure

```
pcb-defect-ai/
├── app.py                      # Main Streamlit Industrial Dashboard (TraceBot)
├── models/
│   └── best.pt                 # Fine-tuned YOLOv8 PCB Defect Detection Weights
├── src/
│   ├── __init__.py
│   ├── detector.py             # YOLOv8 Edge Inference & BBox Visualizer
│   ├── comparator.py           # Golden Template Registration & Heatmap Diff
│   ├── coordinate_mapper.py    # Pixel-to-Millimeter Board Calibration
│   ├── robotic_arm.py          # Robotic Rework Planner & G-Code Generator
│   └── schematic_report.py     # EDA Directives & Engineering Change Order (ECO)
├── sample_data/
│   ├── generate_samples.py     # Synthetic Golden & Defective PCB Generator
│   ├── golden_reference.png    # Defect-free Master PCB
│   └── test_board_defective.png# Test PCB with injected physical defects
├── requirements.txt            # Python Dependencies
├── .gitignore
└── README.md                   # System Documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/jaya0807/PCB-defect-detect.git
cd PCB-defect-detect
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Quick Test Workflow
1. Launch the Streamlit application.
2. Under **Step 1**, upload a PCB image from your desktop (or click the sample test helper).
3. Click **Run Inspection & Automated Localization**.
4. Explore the 6 inspection tabs:
   - **Tab 1:** View localized defect bounding boxes, confidence, and metrics.
   - **Tab 2:** Review the physical millimeter coordinate table and 2D board layout.
   - **Tab 3:** Review the Robotic toolpath sequence and download the `.gcode` file.
   - **Tab 4:** Review the Engineering Change Order (ECO) and download KiCad DRC JSON.
   - **Tab 5:** Inspect the Golden Reference alignment and anomaly heatmap.
   - **Tab 6:** Download the comprehensive PDF inspection report and CSV data.

---

## Defect Classification Table

| Defect Class | Severity | Industrial Consequence | Automated Remediation |
| :--- | :--- | :--- | :--- |
| **Severe Burnout / Rupture** | [CRITICAL] | Substrate carbonization / catastrophic arc crater | Laser substrate excision & power bus redesign |
| **Open Circuit** | [CRITICAL] | Signal disconnect / power failure | Conductive micro-jet dispenser / Jumper |
| **Short Circuit** | [CRITICAL] | Power-to-ground short / component burnout | Laser trace ablation / Micro-milling |
| **Missing Hole** | [HIGH] | Missing through-hole / unrouted layer via | High-speed 0.3mm CNC micro-drill |
| **Spur** | [MEDIUM] | High-frequency RF noise / potential short | Laser trimming & 45° miter layout directive |
| **Mouse Bite** | [MEDIUM] | Weakened mechanical edge during panelization | Edge routing keep-out increase (0.5mm) |
| **Spurious Copper** | [LOW] | Parasitic capacitance / stray RF antenna | Zone island removal & UV buffer |

---

## License
MIT License. Created for Edge-AI Hardware & Vision Inspection Challenges.

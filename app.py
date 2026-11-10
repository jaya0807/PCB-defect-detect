import base64
import hashlib
import io
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.detector import PCBDetector
from src.comparator import PCBComparator
from src.coordinate_mapper import BoardCoordinateMapper
from src.robotic_arm import RoboticReworkPlanner
from src.schematic_report import SchematicCorrectionEngine

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as PDFImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


APP_NAME = "PCBInspect AI"
MODEL_NAME = "YOLOv8 Edge"
DEVELOPER_NAME = "Edge-AI Inspection Team"
GITHUB_URL = "https://github.com/jaya0807/PCB-defect-detect"

st.set_page_config(
    page_title=f"{APP_NAME} | Edge-AI PCB Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_html(html: str, sidebar: bool = False) -> None:
    """Render HTML safely without tags leaking as code."""
    cleaned = dedent(html).strip()
    target = st.sidebar if sidebar else st
    if hasattr(target, "html"):
        target.html(cleaned)
    else:
        target.markdown(cleaned, unsafe_allow_html=True)


safe_html(
    """
    <style>
        :root {
            --bg: #07111f;
            --card: rgba(15, 23, 42, 0.97);
            --soft: #111d30;
            --text: #e8eef8;
            --muted: #9fb0c8;
            --border: #273850;
            --shadow: 0 12px 30px rgba(0, 0, 0, 0.30);
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(37,99,235,.10), transparent 26%),
                var(--bg);
        }

        .block-container {
            max-width: 1840px;
            padding-top: .8rem;
            padding-bottom: 1rem;
        }

        #MainMenu, footer, header { visibility: hidden; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2f62, #081b38);
            border-right: 1px solid rgba(255,255,255,.08);
        }

        [data-testid="stSidebar"] * { color: #f8fafc; }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label { color: #d8e5f7 !important; }

        .app-header {
            padding: 1.15rem 1.35rem;
            margin-bottom: .75rem;
            border-radius: 20px;
            color: white;
            background:
                radial-gradient(circle at 88% 10%, rgba(191,219,254,.35), transparent 21%),
                linear-gradient(115deg, #0b397d, #1d4ed8 55%, #2563eb);
            box-shadow: 0 16px 34px rgba(29,78,216,.24);
        }
        .header-title { margin: 0; font-size: 1.85rem; font-weight: 850; }
        .header-subtitle { margin-top: .2rem; color: #dbeafe; font-size: .88rem; }
        .badge {
            display: inline-block;
            padding: .3rem .6rem;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 750;
            background: rgba(255,255,255,.16);
            border: 1px solid rgba(255,255,255,.2);
            margin-right: .3rem;
        }
    </style>
    """
)


# -----------------------------------------------------------------------------
# Cached Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_detector():
    return PCBDetector()

@st.cache_resource
def load_comparator():
    return PCBComparator()

@st.cache_resource
def load_rework_planner():
    return RoboticReworkPlanner()

@st.cache_resource
def load_schematic_engine():
    return SchematicCorrectionEngine()


def get_sample_images():
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    ref_p = os.path.join(sample_dir, "golden_reference.png")
    test_p = os.path.join(sample_dir, "test_board_defective.png")
    ref_img = Image.open(ref_p).convert("RGB") if os.path.exists(ref_p) else None
    test_img = Image.open(test_p).convert("RGB") if os.path.exists(test_p) else None
    return ref_img, test_img


def format_file_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def image_to_png_bytes(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def calculate_health(defect_count, critical_count=0):
    if defect_count == 0:
        return 100, "Excellent (Passed)"
    if critical_count > 0:
        score = max(20, 75 - critical_count * 20 - defect_count * 5)
        return score, "Critical (Rework Required)"
    score = max(40, 95 - defect_count * 8)
    return score, "Needs Inspection"


def plot_physical_board(board_w_mm, board_h_mm, mapped_dets, origin_pos="Bottom-Left"):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor="#07111f")
    ax.set_facecolor("#0f172a")

    # Draw board outline
    ax.add_patch(plt.Rectangle((0, 0), board_w_mm, board_h_mm, fill=True, color="#064e3b", alpha=0.35, ec="#10b981", lw=2))

    ax.set_xlim(-5, board_w_mm + 5)
    ax.set_ylim(-5, board_h_mm + 5)
    ax.set_xlabel("Board X (mm)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Board Y (mm)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.2, color="#64748b")
    ax.set_title(f"Physical Board Localization (mm) - Origin: {origin_pos}", color="white", fontsize=11, fontweight="bold")

    # Origin marker
    ax.plot(0, 0, marker="o", color="#38bdf8", markersize=7)
    ax.text(2, 2, "ORIGIN (0,0)", color="#38bdf8", fontsize=8, fontweight="bold")

    col_map = {
        "Open": "#ef4444", "Short": "#dc2626", "Missing Hole": "#f97316",
        "Spur": "#eab308", "Mouse Bite": "#a855f7", "Spurious Copper": "#06b6d4"
    }

    for det in mapped_dets:
        c = det.get("board_coord_mm", {"x": 0, "y": 0})
        cls = det["class"]
        col = col_map.get(cls, "#f43f5e")

        ax.plot(c["x"], c["y"], marker="x", markersize=10, mew=2, color=col)
        ax.plot(c["x"], c["y"], marker="o", markersize=5, fillstyle="none", color=col, mew=1.5)
        ax.text(c["x"] + 1.2, c["y"] + 1.0, f"#{det['id']} {cls}", color="white", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.8, edgecolor=col))

    plt.tight_layout()
    return fig


def create_charts(counts):
    fig_bar, ax_bar = plt.subplots(figsize=(6.0, 2.8), facecolor="#0f172a")
    ax_bar.set_facecolor("#1e293b")
    labels = list(counts.keys())
    values = list(counts.values())
    palette = ["#ef4444", "#dc2626", "#f97316", "#eab308", "#a855f7", "#06b6d4"]

    bars = ax_bar.bar(labels, values, color=palette[:len(labels)], edgecolor="white", alpha=0.9)
    ax_bar.set_title("Defects by Category", color="white", fontsize=10, fontweight="bold")
    ax_bar.tick_params(colors="#94a3b8", labelsize=8)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.2, color="#64748b")
    for b in bars:
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, str(int(b.get_height())),
                    ha="center", color="white", fontsize=8, fontweight="bold")
    fig_bar.tight_layout()

    fig_pie, ax_pie = plt.subplots(figsize=(4.5, 2.8), facecolor="#0f172a")
    ax_pie.pie(
        values, labels=labels, autopct="%1.0f%%", startangle=90,
        colors=palette[:len(labels)],
        textprops={"color": "white", "fontsize": 8},
        wedgeprops={"edgecolor": "#0f172a", "linewidth": 1.2}
    )
    ax_pie.set_title("Defect Distribution", color="white", fontsize=10, fontweight="bold")
    fig_pie.tight_layout()

    return fig_bar, fig_pie


def build_pdf_report(result):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=32, leftMargin=32, topMargin=32, bottomMargin=32)
    styles = getSampleStyleSheet()

    story = [
        Paragraph(APP_NAME, ParagraphStyle("T", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#0f2f62"))),
        Paragraph("Edge-AI Automated PCB Inspection & Defect Localization Report", styles["Normal"]),
        Spacer(1, 10),
    ]

    summary_rows = [
        ["Date", result["timestamp"].strftime("%Y-%m-%d %H:%M"), "Model", MODEL_NAME],
        ["Resolution", result["resolution"], "Total Defects", str(result["total_defects"])],
        ["Processing Time", f"{result['elapsed']:.3f} s", "PCB Health", f"{result['health_score']}/100 - {result['status']}"],
    ]
    t = Table(summary_rows, colWidths=[1.3 * inch, 1.8 * inch, 1.3 * inch, 1.8 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Defect Table
    if not result["table"].empty:
        df = result["table"]
        headers = ["#", "Class", "Severity", "Confidence", "Board X (mm)", "Board Y (mm)"]
        cols = [c for c in headers if c in df.columns]
        rows = [cols] + df[cols].values.tolist()
        dt = Table(rows, repeatRows=1)
        dt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(dt)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1 class="header-title">🔬 {APP_NAME} · Edge-AI PCB Inspection</h1>
        <div class="header-subtitle">
            Autonomous defect detection, golden-reference comparison, coordinate localization, and closed-loop rework.
        </div>
        <div style="margin-top: .6rem;">
            <span class="badge">⚡ YOLOv8 Edge</span>
            <span class="badge">⚖️ Reference Comparison</span>
            <span class="badge">🗺️ Physical (mm) Mapping</span>
            <span class="badge">🤖 Robotic G-Code Rework</span>
            <span class="badge">📐 EDA Schematics ECO</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📏 Physical Board Calibration")
    board_w = st.number_input("Board Width (mm)", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
    board_h = st.number_input("Board Height (mm)", min_value=10.0, max_value=500.0, value=80.0, step=5.0)
    origin_pos = st.selectbox("EDA / CAD Origin", ["Bottom-Left", "Top-Left"], index=0, help="Gerber/CAD standard is Bottom-Left (0,0)")

    st.subheader("🎯 Detection Thresholds")
    conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.20, 0.05)
    iou_thresh = st.slider("NMS IoU Threshold", 0.1, 0.8, 0.45, 0.05)

    st.markdown("---")
    st.caption("🚀 **Hackathon Problem 7 Implementation**")
    st.markdown("""
    - ✅ **YOLOv8 Edge-AI Model**
    - ✅ **Golden Reference Comparison**
    - ✅ **Physical Coordinate Localization**
    - ✅ **Robotic Arm Rework G-Code**
    - ✅ **Schematic / CAD ECO Directives**
    """)


# -----------------------------------------------------------------------------
# Image Input & Session Management
# -----------------------------------------------------------------------------
if "inspection_result" not in st.session_state:
    st.session_state.inspection_result = None
if "golden_image" not in st.session_state:
    st.session_state.golden_image = None
if "test_image" not in st.session_state:
    st.session_state.test_image = None

mode_col1, mode_col2 = st.columns([2, 1])
with mode_col1:
    input_mode = st.radio(
        "Choose Inspection Input:",
        ["Upload Images", "Use Sample PCB Dataset (1-Click Test)", "Live Camera Capture"],
        horizontal=True,
    )

selected_file = None
source_name = "Uploaded Image"

if input_mode == "Use Sample PCB Dataset (1-Click Test)":
    ref_sample, test_sample = get_sample_images()
    st.session_state.golden_image = ref_sample
    st.session_state.test_image = test_sample
    source_name = "Sample PCB Dataset"
    st.info("Loaded synthetic Golden Reference and Defective Test PCB from sample repository.")
elif input_mode == "Upload Images":
    up1, up2 = st.columns(2)
    with up1:
        test_file = st.file_uploader("Upload Inspected PCB Image (Required)", type=["jpg", "jpeg", "png"])
        if test_file:
            st.session_state.test_image = Image.open(test_file).convert("RGB")
            source_name = test_file.name
    with up2:
        golden_file = st.file_uploader("Upload Golden Reference Master (Optional for Diff)", type=["jpg", "jpeg", "png"])
        if golden_file:
            st.session_state.golden_image = Image.open(golden_file).convert("RGB")
else:
    cam_file = st.camera_input("Capture PCB Image from Edge Camera")
    if cam_file:
        st.session_state.test_image = Image.open(cam_file).convert("RGB")
        source_name = "Live Camera"

start_inspection = st.button("🚀 Start Inspection & Localization", type="primary", use_container_width=True, disabled=st.session_state.test_image is None)

if start_inspection and st.session_state.test_image is not None:
    original_image = st.session_state.test_image
    test_img_np = np.array(original_image)
    width, height = original_image.size

    # Save to temp file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        original_image.save(tmp.name, format="JPEG", quality=95)
        tmp_path = tmp.name

    detector = load_detector()
    t_start = time.perf_counter()
    with st.spinner("Running YOLOv8 defect detection..."):
        annotated_image, detections = detector.predict(tmp_path, conf=conf_thresh, iou=iou_thresh)
    elapsed = time.perf_counter() - t_start

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Physical coordinate mapping
    mapper = BoardCoordinateMapper(board_w, board_h, origin_pos)
    mapped_detections = mapper.map_detections(detections, test_img_np.shape)

    # Golden Reference Comparison
    comparator_res = None
    if st.session_state.golden_image is not None:
        comparator = load_comparator()
        comparator_res = comparator.align_and_compare(np.array(st.session_state.golden_image), test_img_np)

    # Robotic Arm G-Code Plan
    planner = load_rework_planner()
    rework_plan = planner.generate_rework_plan(mapped_detections)
    gcode = planner.generate_gcode(rework_plan, operator_name=DEVELOPER_NAME)

    # Schematic ECO Report
    schematic_engine = load_schematic_engine()
    eco_markdown = schematic_engine.generate_eco_report(mapped_detections)
    kicad_json = schematic_engine.export_kicad_drc_json(mapped_detections)

    # Build DataFrame
    rows = []
    for d in mapped_detections:
        c = d["board_coord_mm"]
        rows.append({
            "#": d["id"],
            "Class": d["class"],
            "Severity": d["severity"],
            "Confidence": f"{d['confidence']:.1f}%",
            "Board X (mm)": c["x"],
            "Board Y (mm)": c["y"],
            "Width (mm)": c["w"],
            "Height (mm)": c["h"],
            "Area (mm²)": c["area_mm2"],
            "BBox [x1,y1,x2,y2]": str(d["bbox"])
        })
    df_table = pd.DataFrame(rows)

    critical_count = sum(1 for d in mapped_detections if d["severity"] == "Critical")
    health_score, status = calculate_health(len(mapped_detections), critical_count)

    result = {
        "filename": source_name,
        "source_name": source_name,
        "resolution": f"{width} × {height} px",
        "threshold": conf_thresh,
        "elapsed": elapsed,
        "timestamp": datetime.now(),
        "table": df_table,
        "total_defects": len(mapped_detections),
        "critical_count": critical_count,
        "health_score": health_score,
        "status": status,
        "original_bytes": image_to_png_bytes(original_image),
        "detection_bytes": image_to_png_bytes(annotated_image),
        "mapped_detections": mapped_detections,
        "comparator_res": comparator_res,
        "rework_plan": rework_plan,
        "gcode": gcode,
        "eco_markdown": eco_markdown,
        "kicad_json": kicad_json,
    }
    result["pdf_bytes"] = build_pdf_report(result)
    st.session_state.inspection_result = result


# -----------------------------------------------------------------------------
# Results Presentation
# -----------------------------------------------------------------------------
res = st.session_state.inspection_result

if res is None:
    st.info("👆 Please select an input option and click **'Start Inspection & Localization'**.")
else:
    # Top KPI Metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🔍 Total Defects", res["total_defects"])
    k2.metric("⚠️ Critical (Opens/Shorts)", res["critical_count"])
    k3.metric("⚡ Inference Latency", f"{res['elapsed'] * 1000:.1f} ms")
    k4.metric("🛡️ PCB Health Score", f"{res['health_score']}/100", res["status"])
    k5.metric("📏 Origin", origin_pos)

    st.progress(res["health_score"] / 100, text=f"Quality Index: {res['status']}")

    # Main Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🔍 AI Detection & Visuals",
        "⚖️ Golden Reference Comparison",
        "🗺️ Physical Board Map (mm)",
        "🤖 Robotic Arm Rework (G-Code)",
        "📐 Schematic & CAD Correction (ECO)",
        "⬇️ Download & PDF Report",
    ])

    # Tab 1: AI Detection & Visuals
    with t1:
        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("📷 Inspected Board (Raw)")
            st.image(res["original_bytes"], use_container_width=True)
        with c_right:
            st.subheader("🎯 Localized Defects (YOLOv8)")
            st.image(res["detection_bytes"], use_container_width=True)

        st.subheader("📋 Detected Defects Table")
        if not res["table"].empty:
            st.dataframe(res["table"], use_container_width=True, hide_index=True)
        else:
            st.success("🎉 No defects detected at current threshold. Board passed inspection!")

        # Charts
        if not res["table"].empty:
            st.markdown("---")
            cc1, cc2 = st.columns(2)
            counts = Counter(res["table"]["Class"])
            bchart, pchart = create_charts(counts)
            with cc1:
                st.pyplot(bchart, use_container_width=True)
                plt.close(bchart)
            with cc2:
                st.pyplot(pchart, use_container_width=True)
                plt.close(pchart)

    # Tab 2: Golden Reference Comparison
    with t2:
        st.subheader("⚖️ Template Alignment & Anomaly Difference Analysis")
        comp = res.get("comparator_res")
        if comp is not None:
            st.caption(f"Alignment: {'✅ ORB RANSAC Homography Successful' if comp['alignment_success'] else '⚠️ Scaled Fallback'} | Discrepancies: {comp['discrepancies_count']}")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**1. Golden Reference Master**")
                st.image(comp["reference_rgb"], use_container_width=True)
                st.markdown("**3. Binary Difference Mask**")
                st.image(comp["diff_mask_rgb"], use_container_width=True)
            with rc2:
                st.markdown("**2. Aligned Inspected Board**")
                st.image(comp["aligned_test_rgb"], use_container_width=True)
                st.markdown("**4. Anomaly Heatmap (JET)**")
                st.image(comp["heatmap_rgb"], use_container_width=True)
        else:
            st.warning("⚠️ No Golden Reference image provided for comparison. Upload one in sidebar or use the sample dataset.")

    # Tab 3: Physical Board Map (mm)
    with t3:
        st.subheader("🗺️ 2D Physical Board Coordinate Localization (mm)")
        st.markdown(f"Board Dimensions: **{board_w} mm × {board_h} mm** | Origin: **{origin_pos} (0,0)**")
        if res["mapped_detections"]:
            board_fig = plot_physical_board(board_w, board_h, res["mapped_detections"], origin_pos)
            st.pyplot(board_fig, use_container_width=True)
            plt.close(board_fig)
        else:
            st.info("No defect coordinates to map.")

    # Tab 4: Robotic Arm Rework (G-Code)
    with t4:
        st.subheader("🤖 Autonomous Robotic Arm Rework System")
        st.markdown("""
        Automatically maps detected flaws to CNC / Robotic tooling and generates standard **ISO 6983 G-Code**:
        - **Shorts & Spurs**: Precision Laser Ablation / Micro-Milling Cutter.
        - **Opens**: Micro-Dispenser with Conductive Epoxy / Solder Wire.
        - **Missing Holes**: High-Speed 0.3mm CNC Micro-Drill.
        """)
        if res["rework_plan"]:
            df_plan = pd.DataFrame(res["rework_plan"])
            st.markdown("#### 🛠️ Sequential Rework Operation Plan")
            st.dataframe(df_plan[["step", "defect_type", "severity", "tool_id", "tool_name", "action", "coord_x_mm", "coord_y_mm", "est_seconds"]], use_container_width=True, hide_index=True)

            gc1, gc2 = st.columns([3, 1])
            with gc1:
                st.markdown("#### 📜 CNC / Robotic G-Code Preview")
                st.code(res["gcode"], language="gcode")
            with gc2:
                st.markdown("#### 💾 Export Toolpath")
                st.download_button(
                    "⬇️ Download .GCODE",
                    data=res["gcode"],
                    file_name=f"pcb_rework_{datetime.now().strftime('%Y%m%d_%H%M')}.gcode",
                    mime="text/plain",
                    type="primary",
                    use_container_width=True
                )
                st.download_button(
                    "⬇️ Download Plan (CSV)",
                    data=df_plan.to_csv(index=False),
                    file_name="robotic_rework_plan.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.success("No defects detected. Zero rework operations needed.")

    # Tab 5: Schematic & CAD Correction (ECO)
    with t5:
        st.subheader("📐 Schematic & EDA Layout Design Correction")
        st.markdown("Translates physical defect coordinates into formal **Engineering Change Orders (ECO)** and **KiCad/Altium DRC directives**.")
        if res["mapped_detections"]:
            ec1, ec2 = st.columns([3, 1])
            with ec1:
                st.markdown("#### 📋 Engineering Change Order (ECO) Preview")
                st.markdown(res["eco_markdown"])
            with ec2:
                st.markdown("#### 💾 Export Directives")
                st.download_button(
                    "⬇️ Download ECO (.md)",
                    data=res["eco_markdown"],
                    file_name="PCB_Engineering_Change_Order.md",
                    mime="text/markdown",
                    type="primary",
                    use_container_width=True
                )
                st.download_button(
                    "⬇️ Download KiCad DRC JSON",
                    data=res["kicad_json"],
                    file_name="kicad_drc_violations.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.success("No design flaws detected. Board conforms to EDA Design Rules.")

    # Tab 6: Download Center & PDF
    with t6:
        st.subheader("⬇️ Download Center & Quality Reports")
        dc1, dc2, dc3 = st.columns(3)
        dc1.download_button(
            "📥 Download Detection Image",
            data=res["detection_bytes"],
            file_name="pcb_detection_result.png",
            mime="image/png",
            use_container_width=True,
        )
        dc2.download_button(
            "📄 Export CSV Results",
            data=res["table"].to_csv(index=False).encode("utf-8"),
            file_name="pcb_inspection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if res["pdf_bytes"] is not None:
            dc3.download_button(
                "🧾 Generate PDF Report",
                data=res["pdf_bytes"],
                file_name="pcb_inspection_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )


safe_html(
    f"""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #1e293b; padding-top: 1rem;">
        <b>{APP_NAME}</b> · Edge-AI Automated Industrial PCB Quality Inspection System · 
        <a href="{GITHUB_URL}" target="_blank" style="color: #60a5fa; text-decoration: none;">GitHub Repository</a>
    </div>
    """
)

"""
Schematic & CAD Software Correction Module
Generates Engineering Change Orders (ECO), KiCad DRC violation reports,
and layout redesign directives for software-side design correction.
"""

from typing import List, Dict, Any
from datetime import datetime
import json


class SchematicCorrectionEngine:
    CAD_RECOMMENDATIONS = {
        "Severe Burnout / Rupture": {
            "drc_code": "DRC_CRITICAL_ELECTRICAL_BURNOUT",
            "schematic_action": "Power Bus & Overcurrent Protection Redesign",
            "layout_directive": "Catastrophic thermal arc crater detected. Widen power delivery traces, insert crowbar/TVS protection, and reinforce FR-4 substrate isolation with thermal relief vias.",
            "kicad_layer": "All Layers (Substrate Puncture)",
            "risk": "Catastrophic power runaway, substrate carbonization short, and board destruction."
        },
        "Short": {
            "drc_code": "DRC_ERR_CLEARANCE_VIOLATION",
            "schematic_action": "Verify Net Separation & Increase Clearance",
            "layout_directive": "Increase minimum copper clearance rule from 0.15mm to 0.25mm. Check for acute trace angles (< 90 deg) creating acid traps.",
            "kicad_layer": "F.Cu / B.Cu",
            "risk": "Electrical bridge causing power-rail short or signal contention."
        },
        "Open": {
            "drc_code": "DRC_ERR_DISCONNECTED_NET",
            "schematic_action": "Check Netlist Continuity & Trace Gauge",
            "layout_directive": "Trace broken or over-etched. Increase minimum trace width (e.g., from 0.15mm to 0.25mm) or re-route away from high-stress flex zones.",
            "kicad_layer": "F.Cu / B.Cu",
            "risk": "Open circuit preventing power delivery or signal transmission."
        },
        "Missing Hole": {
            "drc_code": "DRC_ERR_UNROUTED_VIA_DRILL",
            "schematic_action": "Verify Through-Hole Component / Via Padstack",
            "layout_directive": "Excellon drill file missing drill coordinate. Ensure via padstack specifies drill diameter > 0.3mm and regenerate NC Drill output.",
            "kicad_layer": "Edge.Cuts / Drill",
            "risk": "No electrical connection between inner/outer layers or missing pin hole."
        },
        "Spur": {
            "drc_code": "DRC_WARN_ACUTE_ANGLE_SPUR",
            "schematic_action": "Smooth Copper Angles & Tear-drops",
            "layout_directive": "Acute trace branch or copper projection. Enable trace teardrops in KiCad/Altium layout and enforce 45-degree mitered corners.",
            "kicad_layer": "F.Cu",
            "risk": "High-frequency impedance discontinuity and potential micro-short."
        },
        "Mouse Bite": {
            "drc_code": "DRC_WARN_MECH_CLEARANCE",
            "schematic_action": "Adjust Mechanical Edge Keep-Out",
            "layout_directive": "Board routing edge indentation. Increase copper keep-out margin from board outline (Edge.Cuts) to at least 0.5mm.",
            "kicad_layer": "Edge.Cuts",
            "risk": "Mechanical stress failure during V-score panel depanelization."
        },
        "Spurious Copper": {
            "drc_code": "DRC_WARN_ISOLATED_COPPER_ISLAND",
            "schematic_action": "Remove Dead Copper Islands",
            "layout_directive": "Unconnected copper island detected. Enable 'Remove unused copper pour islands' in polygon fill settings.",
            "kicad_layer": "F.Cu / B.Cu Zone",
            "risk": "Parasitic capacitance and stray EMI antenna effects."
        },
    }

    def generate_eco_report(
        self,
        mapped_detections: List[Dict[str, Any]],
        board_name: str = "Main_Board_RevA",
        designer: str = "Hardware Engineering Team"
    ) -> str:
        """
        Generates a formal Engineering Change Order (ECO) in Markdown format.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# ENGINEERING CHANGE ORDER (ECO) REPORT",
            f"**Project / Board:** `{board_name}`  ",
            f"**Generated Date:** {now_str}  ",
            f"**Author / Inspector:** {designer} (Automated Edge-AI Inspection)  ",
            f"**Total Design Directives:** {len(mapped_detections)}",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "Visual inspection identified manufacturing anomalies that require schematic or PCB layout adjustments in EDA software (KiCad / Altium Designer / Eagle).",
            "",
            "| Item | Defect Type | Severity | Board Location (mm) | DRC Violation Code | Target Layer |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ]

        for i, det in enumerate(mapped_detections, 1):
            cls_name = det["class"]
            coords = det.get("board_coord_mm", {"x": 0.0, "y": 0.0})
            rec = self.CAD_RECOMMENDATIONS.get(cls_name, {})
            drc_code = rec.get("drc_code", "DRC_GENERIC")
            layer = rec.get("kicad_layer", "All Copper")
            lines.append(
                f"| #{i} | **{cls_name}** | {det.get('severity', 'Medium')} | `X={coords['x']:.2f}, Y={coords['y']:.2f}` | `{drc_code}` | `{layer}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Detailed Layout & Schematic Correction Directives",
            ""
        ])

        for i, det in enumerate(mapped_detections, 1):
            cls_name = det["class"]
            coords = det.get("board_coord_mm", {"x": 0.0, "y": 0.0})
            rec = self.CAD_RECOMMENDATIONS.get(cls_name, {})
            lines.extend([
                f"### Directive #{i}: {cls_name} at X: {coords['x']:.2f}mm, Y: {coords['y']:.2f}mm",
                f"- **Schematic Action:** {rec.get('schematic_action', 'Review affected net.')}",
                f"- **Layout Directive:** {rec.get('layout_directive', 'Adjust copper geometry.')}",
                f"- **Functional Risk:** {rec.get('risk', 'Unknown risk.')}",
                f"- **EDA Design Rule Check (DRC) Code:** `{rec.get('drc_code', 'DRC_VIOLATION')}`",
                ""
            ])

        lines.extend([
            "---",
            "*Approved for CAD Review and Layout Re-spin.*"
        ])

        return "\n".join(lines)

    def export_kicad_drc_json(
        self,
        mapped_detections: List[Dict[str, Any]],
        board_name: str = "Main_Board"
    ) -> str:
        """
        Exports a JSON payload compatible with EDA automated DRC log parsers.
        """
        violations = []
        for i, det in enumerate(mapped_detections, 1):
            cls_name = det["class"]
            coords = det.get("board_coord_mm", {"x": 0.0, "y": 0.0})
            rec = self.CAD_RECOMMENDATIONS.get(cls_name, {})
            violations.append({
                "violation_id": f"ECO-{i:03d}",
                "type": cls_name,
                "severity": det.get("severity", "Medium"),
                "coordinates_mm": {"x": coords["x"], "y": coords["y"]},
                "drc_code": rec.get("drc_code", "DRC_ERR"),
                "layer": rec.get("kicad_layer", "F.Cu"),
                "schematic_action": rec.get("schematic_action", ""),
                "layout_directive": rec.get("layout_directive", "")
            })

        payload = {
            "eda_report_version": "1.0",
            "board": board_name,
            "timestamp": datetime.now().isoformat(),
            "total_violations": len(violations),
            "violations": violations
        }
        return json.dumps(payload, indent=2)


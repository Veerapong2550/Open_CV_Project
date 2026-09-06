"""Optional export of completed measurements to an Excel workbook."""

from __future__ import annotations

from pathlib import Path


def save_measurement_to_excel(measurement: dict, output_path: str | Path) -> Path:
    """Save a stable result to a single ``Summary`` worksheet."""
    if not measurement:
        raise ValueError("A completed measurement is required before export")
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Excel export requires pandas and openpyxl") from exc
    path = Path(output_path).with_suffix(".xlsx")
    posture = measurement.get("posture", {})
    shoulders = measurement.get("shoulders", {})
    row = {
        "Measured At": measurement.get("measured_at"),
        "Input Height (cm)": measurement.get("input_height_cm"),
        "Calibration": measurement.get("calibration"),
        "Stable Samples": measurement.get("samples_used"),
        "Shoulder Width (cm; front view only)": measurement.get("shoulder_cm"),
        "Left Shoulder Length (cm; front view)": measurement.get("left_shoulder_cm"),
        "Right Shoulder Length (cm; front view)": measurement.get("right_shoulder_cm"),
        "Shoulder Difference (cm; front view)": measurement.get("shoulder_difference_cm"),
        "Left Shoulder Length (% shoulder span)": _percent(shoulders.get("left_shoulder_length_ratio")),
        "Right Shoulder Length (% shoulder span)": _percent(shoulders.get("right_shoulder_length_ratio")),
        "Shoulder Difference (% shoulder span)": _percent(shoulders.get("shoulder_length_difference_ratio")),
        "Shoulder Image Comparison": shoulders.get("shoulder_balance_label"),
        "Front Shoulder Samples": shoulders.get("samples_used"),
        "Front Shoulder Stable": shoulders.get("stable"),
        "Front Shoulder Confidence": shoulders.get("landmark_confidence"),
        "Shoulder Tilt (deg; front view)": shoulders.get("shoulder_tilt_deg"),
        "Posture Screening Level": posture.get("screening_level"),
        "Posture Screening Score (0-6)": posture.get("screening_score"),
        "Head-Shoulder Offset (% torso)": _percent(posture.get("head_shoulder_offset_ratio")),
        "Shoulder-Hip Offset (% torso)": _percent(posture.get("shoulder_hip_offset_ratio")),
        "Hip-Ankle Offset (% torso)": _percent(posture.get("hip_ankle_offset_ratio")),
        "Neck Inclination (deg)": posture.get("neck_inclination_deg"),
        "Trunk Inclination (deg)": posture.get("trunk_inclination_deg"),
        "Lower-body Inclination (deg)": posture.get("lower_body_inclination_deg"),
        "Ear-Shoulder-Hip Angle (deg)": posture.get("ear_shoulder_hip_angle"),
        "Head-Shoulder Offset (cm; ArUco only)": posture.get("head_shoulder_offset_cm"),
        "Shoulder-Hip Offset (cm; ArUco only)": posture.get("shoulder_hip_offset_cm"),
        "Landmark Confidence": posture.get("landmark_confidence"),
        "Frame Stability": posture.get("stable"),
    }
    pd.DataFrame([row]).to_excel(path, sheet_name="Summary", index=False)
    return path


def _percent(value: float | None) -> float | None:
    return None if value is None else value * 100.0

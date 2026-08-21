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
    row = {
        "Measured At": measurement["measured_at"],
        "Input Height (cm)": measurement["input_height_cm"],
        "Calibration": measurement["calibration"],
        "Shoulder Width (cm)": measurement["shoulder_cm"],
        "Left Shoulder Length (cm)": measurement["left_shoulder_cm"],
        "Right Shoulder Length (cm)": measurement["right_shoulder_cm"],
        "Stable Samples": measurement["samples_used"],
    }
    pd.DataFrame([row]).to_excel(path, sheet_name="Summary", index=False)
    return path

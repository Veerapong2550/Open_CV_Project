"""Backward-compatible launcher for the integrated body_measure package.

Run this file directly, or use ``python -m body_measure``.  The application
logic lives in ``body_measure``; this file intentionally contains no second
copy of the camera, MediaPipe, or calibration workflow.
"""

from body_measure.app import run
from body_measure.config import DEFAULT_HEIGHT_CM, DEFAULT_MARKER_CM
from body_measure.export import save_measurement_to_excel


def _prompt_float(prompt: str, default: float, minimum: float) -> float:
    try:
        value = input(f"{prompt} [{default:g}]: ").strip()
        result = default if not value else float(value)
        if result < minimum:
            raise ValueError
        return result
    except ValueError:
        print(f"Invalid value; using {default:g}.")
        return default


if __name__ == "__main__":
    height = _prompt_float("Your height (cm)", DEFAULT_HEIGHT_CM, 0.1)
    marker_size = _prompt_float("ArUco marker side (cm; 0 = ratios/angles only)", DEFAULT_MARKER_CM, 0.0)
    result = run(user_height_cm=height, marker_size_cm=marker_size)
    if result:
        print("Measurement result:", result)
        output = input("Excel output path (leave blank to skip): ").strip()
        if output:
            print(f"Saved: {save_measurement_to_excel(result, output)}")

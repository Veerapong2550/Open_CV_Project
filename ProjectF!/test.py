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
        posture = result.get("posture") or {}
        print("\n" + "=" * 55)
        print("  ผลการวิเคราะห์อาการหลังค่อมหลังงอ (Posture Screening)")
        print("=" * 55)
        print(f"  ระดับผลการตรวจ: {posture.get('screening_level', '-')}")
        print(f"  สรุปผล: {posture.get('screening_label', '-')}")
        print(f"  คะแนนความเสี่ยง (0-6): {posture.get('screening_score', 0)}")
        head_cm = f" ({posture.get('head_shoulder_offset_cm'):.1f} ซม.)" if posture.get('head_shoulder_offset_cm') else ""
        print(f"  การยื่นของศีรษะ (คอยื่น): {posture.get('head_shoulder_offset_ratio', 0) * 100:.1f}% ของลำตัว{head_cm}")
        shoulder_cm = f" ({posture.get('shoulder_hip_offset_cm'):.1f} ซม.)" if posture.get('shoulder_hip_offset_cm') else ""
        print(f"  แนวงุ้มของไหล่ (ไหล่ห่อ): {posture.get('shoulder_hip_offset_ratio', 0) * 100:.1f}% ของลำตัว{shoulder_cm}")
        print(f"  มุมเอียงของลำตัว (หลังค่อม): {posture.get('trunk_inclination_deg', 0):.1f}°")
        print(f"  มุมข้อต่อศีรษะ–ไหล่–สะโพก: {posture.get('ear_shoulder_hip_angle', 0):.1f}°")
        print("=" * 55 + "\n")

        output = input("Excel output path (leave blank to skip): ").strip()
        if output:
            print(f"Saved: {save_measurement_to_excel(result, output)}")

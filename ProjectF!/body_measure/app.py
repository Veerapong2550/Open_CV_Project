"""Camera application for a distance-tolerant side-profile posture screen."""

from __future__ import annotations

from datetime import datetime
import time

import cv2

from .calibration import get_pixel_scale
from .capture import save_full_body_capture
from .config import (CAMERA_INDEX, DEFAULT_HEIGHT_CM, DEFAULT_MARKER_CM, FRAME_HEIGHT,
                     FRAME_WIDTH, GESTURE_HOLD_SECONDS, MIN_ANALYSIS_BODY_HEIGHT_PX,
                     MIN_ANALYSIS_BODY_HEIGHT_RATIO, POSTURE_STABLE_SAMPLES,
                     POSTURE_VISIBILITY_THRESHOLD, SIDE_VIEW_MAX_WIDTH_TO_TORSO,
                     VISIBILITY_THRESHOLD, WINDOW_SIZE, WINDOW_TITLE)
from .posture import PostureSampleBuffer, analyse_posture_frame
from .state_machine import GestureStateMachine, State
from .ui import MeasurementUI, show_measurement_summary
from .utils import (draw_hand_landmarks, draw_pose_landmarks, draw_posture_guides,
                    draw_thai_text, is_peace_sign)
from .vision import VisionEngine


def _has_full_body(pose_result) -> bool:
    """Return whether enough of the body is present to save a useful capture."""
    if not pose_result.pose_landmarks:
        return False
    landmarks = pose_result.pose_landmarks[0]
    required = (0, 11, 12, 23, 24, 27, 28)
    if len(landmarks) <= max(required):
        return False
    return sum(landmarks[index].visibility >= VISIBILITY_THRESHOLD for index in required) >= 6


def _minimum_body_height(frame_height: int) -> float:
    """Protect measurement quality while retaining much more range than before."""
    return max(float(MIN_ANALYSIS_BODY_HEIGHT_PX), frame_height * MIN_ANALYSIS_BODY_HEIGHT_RATIO)


def _add_absolute_estimates(summary: dict) -> None:
    """Add optional cm estimates only when an ArUco scale was observed."""
    pixels_per_cm = summary.get("pixels_per_cm", 0.0)
    if pixels_per_cm <= 0:
        summary["head_shoulder_offset_cm"] = None
        summary["shoulder_hip_offset_cm"] = None
        summary["torso_cm"] = None
        return
    torso_cm = summary["torso_px"] / pixels_per_cm
    summary["torso_cm"] = torso_cm
    summary["head_shoulder_offset_cm"] = summary["head_shoulder_offset_ratio"] * torso_cm
    summary["shoulder_hip_offset_cm"] = summary["shoulder_hip_offset_ratio"] * torso_cm


def _quality_and_measurement(pose_result, frame, user_height_cm: float, marker_size_cm: float,
                             marker_scale: float | None, samples: PostureSampleBuffer):
    """Analyse one side-view frame and return status, a result, and live values."""
    height, width = frame.shape[:2]
    if not pose_result.pose_landmarks:
        samples.clear()
        return "ไม่พบร่างกาย — ยืนให้เห็นเต็มตัวและเพิ่มแสง", None, None

    metrics, reason = analyse_posture_frame(
        pose_result.pose_landmarks[0], width, height,
        min_visibility=POSTURE_VISIBILITY_THRESHOLD,
        side_view_max_width_to_torso=SIDE_VIEW_MAX_WIDTH_TO_TORSO,
    )
    if metrics is None:
        samples.clear()
        return reason, None, None

    if metrics["body_height_px"] < _minimum_body_height(height):
        samples.clear()
        minimum = int(_minimum_body_height(height))
        return (f"ตรวจพบแล้ว แต่ตัวแบบเล็กเกินไปสำหรับวิเคราะห์ ({metrics['body_height_px']:.0f}px; "
                f"ต้องอย่างน้อย {minimum}px) — ขยับเข้าใกล้เล็กน้อย"), None, None

    if metrics["view"] != "side":
        samples.clear()
        return reason, None, {
            "view": "front",
            "body_height_px": metrics["body_height_px"],
            "shoulder_tilt_deg": metrics["shoulder_tilt_deg"],
            "hip_tilt_deg": metrics["hip_tilt_deg"],
        }

    if marker_size_cm > 0 and marker_scale is None:
        samples.clear()
        return "วาง ArUco ID 0 ใกล้ระนาบลำตัว เพื่อแสดงค่าหน่วยเซนติเมตร", None, None

    # The screening itself is deliberately distance-independent.  A marker is
    # only used for optional centimetre estimates in the final report.
    metrics["pixels_per_cm"] = float(marker_scale or 0.0)
    draw_posture_guides(frame, metrics["points"])
    samples.add(metrics)
    live_values = {
        "view": "side",
        "head_ratio": metrics["head_shoulder_offset_ratio"],
        "shoulder_ratio": metrics["shoulder_hip_offset_ratio"],
        "trunk_angle": metrics["trunk_inclination_deg"],
        "body_height_px": metrics["body_height_px"],
        "progress": samples.progress,
    }
    if not samples.ready:
        return f"ยืนนิ่งในท่าด้านข้าง… {samples.samples_used}/{samples.required_samples} เฟรม", None, live_values

    summary = samples.result()
    if summary is None:
        return "กำลังรอข้อมูลท่าทาง", None, live_values
    if not summary["stable"]:
        samples.clear()
        return "ท่าทางเปลี่ยนระหว่างวัด — ยืนนิ่ง แล้วเริ่มใหม่", None, None

    _add_absolute_estimates(summary)
    calibration = "ArUco marker (ค่าระยะเป็นเซนติเมตร)" if marker_size_cm > 0 else (
        "ใช้สัดส่วนและมุมที่ไม่ขึ้นกับระยะกล้อง (ไม่มีค่าเซนติเมตร)"
    )
    result = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "input_height_cm": user_height_cm,
        "calibration": calibration,
        "samples_used": summary["samples_used"],
        "quality": (f"ความเชื่อมั่นจุดเฉลี่ย {summary['landmark_confidence']:.2f}; "
                    f"ความสูงร่างกาย {summary['body_height_px']:.0f}px"),
        "posture": summary,
        # Retain these keys so older export integrations do not fail.  A
        # profile image cannot give a non-foreshortened shoulder width.
        "shoulder_cm": None,
        "left_shoulder_cm": None,
        "right_shoulder_cm": None,
        "shoulder_difference_cm": None,
    }
    return "วิเคราะห์ท่าทางเสร็จแล้ว", result, live_values


def run(user_height_cm: float = DEFAULT_HEIGHT_CM, marker_size_cm: float = DEFAULT_MARKER_CM,
        video_source: int = CAMERA_INDEX):
    """Run the posture screen.

    Hold a peace sign for 1.5 seconds to begin.  Stand relaxed with your full
    body visible, turn sideways to the camera, and hold still while the app
    gathers stable frames.  A positive ``marker_size_cm`` enables optional
    centimetre estimates using ArUco ID 0; posture classification always uses
    distance-normalised ratios and angles.
    """
    if user_height_cm <= 0 or marker_size_cm < 0:
        raise ValueError("user_height_cm must be positive and marker_size_cm cannot be negative")
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera source {video_source}")

    ui = None
    vision = None
    last_measurement = None
    full_body_was_visible = False
    last_capture_path = None
    try:
        ui = MeasurementUI(WINDOW_TITLE, WINDOW_SIZE)
        vision = VisionEngine()
        state_machine = GestureStateMachine(GESTURE_HOLD_SECONDS)
        samples = PostureSampleBuffer(POSTURE_STABLE_SAMPLES)
        while ui.is_open:
            ok, camera_frame = cap.read()
            if not ok:
                break
            # Detect the marker before mirroring; a horizontally flipped
            # ArUco marker no longer decodes as the same ID.
            marker_scale, marker_corners = get_pixel_scale(camera_frame, marker_size_cm)
            frame = cv2.flip(camera_frame, 1)
            if marker_corners is not None:
                marker_corners = marker_corners.copy()
                marker_corners[:, 0] = frame.shape[1] - 1 - marker_corners[:, 0]
                cv2.polylines(frame, [marker_corners], True, (0, 255, 255), 2, cv2.LINE_AA)
            timestamp = time.monotonic()
            pose_result, hand_result = vision.process(frame, int(timestamp * 1000))

            if pose_result.pose_landmarks:
                draw_pose_landmarks(frame, pose_result.pose_landmarks[0])
            full_body_visible = _has_full_body(pose_result)
            if full_body_visible and not full_body_was_visible:
                last_capture_path = save_full_body_capture(frame)
            full_body_was_visible = full_body_visible

            peace = bool(hand_result.hand_landmarks and is_peace_sign(hand_result.hand_landmarks[0]))
            if hand_result.hand_landmarks:
                draw_hand_landmarks(frame, hand_result.hand_landmarks[0])

            state, progress, transitioned = state_machine.update(peace, timestamp)
            if transitioned and state is State.MEASURING:
                samples.clear()
            if state is State.EXIT:
                break
            if state is State.WAITING:
                status = ("ยืนให้เห็นเต็มตัว หันด้านข้าง แล้วชูสองนิ้วค้างเพื่อเริ่ม"
                          if pose_result.pose_landmarks else
                          "ไม่พบร่างกาย — ยืนในภาพเต็มตัว เพิ่มแสง หรือขยับใกล้ขึ้น")
                if peace:
                    status += f" ({progress * GESTURE_HOLD_SECONDS:.1f}/{GESTURE_HOLD_SECONDS:.1f} วินาที)"
            else:
                status, new_measurement, live_values = _quality_and_measurement(
                    pose_result, frame, user_height_cm, marker_size_cm, marker_scale, samples)
                if live_values:
                    if live_values["view"] == "side":
                        draw_thai_text(frame, f"ศีรษะ–ไหล่: {live_values['head_ratio'] * 100:.1f}% ของลำตัว", (30, 70), 23, (0, 255, 255))
                        draw_thai_text(frame, f"ไหล่–สะโพก: {live_values['shoulder_ratio'] * 100:.1f}% | เอียงลำตัว: {live_values['trunk_angle']:.1f}°", (30, 103), 21, (0, 255, 255))
                        draw_thai_text(frame, f"รายละเอียดร่างกาย: {live_values['body_height_px']:.0f}px | เก็บข้อมูล {live_values['progress'] * 100:.0f}%", (30, 133), 19, (0, 255, 255))
                    else:
                        draw_thai_text(frame, f"ภาพหน้า: ไหล่เอียง {live_values['shoulder_tilt_deg']:.1f}° | สะโพกเอียง {live_values['hip_tilt_deg']:.1f}°", (30, 70), 21, (0, 255, 255))
                if new_measurement:
                    if last_capture_path is not None:
                        new_measurement["capture_file"] = last_capture_path.name
                    last_measurement = new_measurement
                    show_measurement_summary(new_measurement, parent=ui.root)
                    samples.clear()
                    state_machine.reset()
                    full_body_was_visible = False
                    continue
            draw_thai_text(frame, status, (30, 30), 22, (255, 255, 255))
            ui.show_frame(frame)
    finally:
        cap.release()
        if vision is not None:
            vision.close()
        if ui is not None:
            ui.destroy()
    return last_measurement


if __name__ == "__main__":
    print(run())

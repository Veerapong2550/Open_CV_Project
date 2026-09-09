"""Two-view camera measurement: front shoulders plus side-profile posture."""

from __future__ import annotations

from datetime import datetime
from enum import Enum, auto
import time

import cv2

from .calibration import estimate_pixel_scale_from_height, get_pixel_scale
from .capture import save_full_body_capture
from .config import (CAMERA_INDEX, DEFAULT_HEIGHT_CM, DEFAULT_MARKER_CM, FRAME_HEIGHT,
                     FRAME_WIDTH, GESTURE_HOLD_SECONDS, MAX_CONSECUTIVE_CAMERA_READ_FAILURES,
                     MIN_ANALYSIS_BODY_HEIGHT_PX,
                     MIN_ANALYSIS_BODY_HEIGHT_RATIO, MIN_FRONT_SHOULDER_SPAN_PX,
                     POSTURE_STABLE_SAMPLES, POSTURE_VISIBILITY_THRESHOLD,
                     FRONT_SHOULDER_VISIBILITY_THRESHOLD, FRONT_VIEW_MIN_WIDTH_TO_TORSO,
                     SHOULDER_STABLE_SAMPLES, SIDE_VIEW_MAX_WIDTH_TO_TORSO,
                     VISIBILITY_THRESHOLD, WINDOW_SIZE, WINDOW_TITLE)
from .posture import (PostureSampleBuffer, ShoulderSampleBuffer, analyse_front_shoulder_frame,
                      analyse_posture_frame)
from .state_machine import GestureStateMachine, State
from .ui import (MeasurementUI, show_measurement_summary, show_phase_instruction,
                 show_startup_error)
from .utils import (PointSmoother, UNUSED_FACE_LANDMARKS, draw_hand_landmarks,
                    draw_pose_landmarks, draw_posture_guides,
                    draw_shoulder_measurement_guides, draw_thai_text, is_peace_sign)
from .vision import PoseFrameResult, PoseLandmark, VisionEngine


class MeasurementPhase(Enum):
    """The two camera views required for reliable combined reporting."""

    FRONT_SHOULDERS = auto()
    SIDE_POSTURE = auto()


def _draw_overlay(ui: MeasurementUI, frame, text: str, pos: tuple[int, int], size: int,
                  color: tuple[int, int, int]) -> None:
    """Keep labels the same visual size after a high-resolution frame is shrunk."""
    scale = ui.overlay_scale(frame)
    draw_thai_text(
        frame,
        text,
        (round(pos[0] * scale), round(pos[1] * scale)),
        max(1, round(size * scale)),
        color,
    )


def _open_camera(video_source: int | str):
    """Open a Windows webcam with a DirectShow fallback before giving up.

    Some webcams report an empty frame through the default backend while they
    work through DirectShow.  File/stream sources intentionally keep OpenCV's
    normal backend.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if isinstance(video_source, int) else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(video_source, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            return cap
        cap.release()
    return None


def _has_full_body(pose_result) -> bool:
    """Return whether every landmark required for an audit capture is clear."""
    if not pose_result.pose_landmarks:
        return False
    landmarks = pose_result.pose_landmarks[0]
    required = (0, 11, 12, 23, 24, 27, 28)
    if len(landmarks) <= max(required):
        return False
    return all(landmarks[index].visibility >= VISIBILITY_THRESHOLD for index in required)


def _has_side_body(pose_result) -> bool:
    """Return whether the key landmarks needed for side profile posture analysis are visible.

    In a side profile, joints on the far side (away from camera) are naturally occluded
    by the body. Therefore, we require at least one landmark from each bilateral pair
    (shoulder, hip, ankle, head/ear) to have sufficient visibility.
    """
    if not pose_result.pose_landmarks:
        return False
    landmarks = pose_result.pose_landmarks[0]
    if len(landmarks) <= 28:
        return False
    has_head = (
        landmarks[0].visibility >= POSTURE_VISIBILITY_THRESHOLD
        or landmarks[7].visibility >= POSTURE_VISIBILITY_THRESHOLD
        or landmarks[8].visibility >= POSTURE_VISIBILITY_THRESHOLD
    )
    has_shoulder = (
        landmarks[11].visibility >= POSTURE_VISIBILITY_THRESHOLD
        or landmarks[12].visibility >= POSTURE_VISIBILITY_THRESHOLD
    )
    has_hip = (
        landmarks[23].visibility >= POSTURE_VISIBILITY_THRESHOLD
        or landmarks[24].visibility >= POSTURE_VISIBILITY_THRESHOLD
    )
    has_ankle = (
        landmarks[27].visibility >= POSTURE_VISIBILITY_THRESHOLD
        or landmarks[28].visibility >= POSTURE_VISIBILITY_THRESHOLD
    )
    return bool(has_head and has_shoulder and has_hip and has_ankle)


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


def _add_absolute_shoulder_estimates(summary: dict) -> None:
    """Add cm values for front shoulder comparison only with ArUco calibration."""
    pixels_per_cm = summary.get("pixels_per_cm", 0.0)
    if pixels_per_cm <= 0:
        summary["left_shoulder_length_cm"] = None
        summary["right_shoulder_length_cm"] = None
        summary["shoulder_length_difference_cm"] = None
        summary["shoulder_span_cm"] = None
        return
    summary["left_shoulder_length_cm"] = summary["left_shoulder_length_px"] / pixels_per_cm
    summary["right_shoulder_length_cm"] = summary["right_shoulder_length_px"] / pixels_per_cm
    summary["shoulder_length_difference_cm"] = summary["shoulder_length_difference_px"] / pixels_per_cm
    summary["shoulder_span_cm"] = summary["shoulder_span_px"] / pixels_per_cm


def _quality_and_shoulder_measurement(pose_result, frame, user_height_cm: float,
                                      marker_size_cm: float, marker_scale: float | None,
                                      samples: ShoulderSampleBuffer):
    """Collect a stable left/right shoulder comparison from a front view."""
    height, width = frame.shape[:2]
    if not pose_result.pose_landmarks:
        samples.clear()
        return "ไม่พบร่างกาย — ยืนให้เห็นเต็มตัวและเพิ่มแสง", None, None

    metrics, reason = analyse_front_shoulder_frame(
        pose_result.pose_landmarks[0], width, height,
        min_visibility=FRONT_SHOULDER_VISIBILITY_THRESHOLD,
        front_view_min_width_to_torso=FRONT_VIEW_MIN_WIDTH_TO_TORSO,
        min_shoulder_span_px=MIN_FRONT_SHOULDER_SPAN_PX,
    )
    if metrics is None:
        samples.clear()
        return reason, None, None
    if not _has_full_body(pose_result):
        samples.clear()
        return "ให้เห็นศีรษะ ไหล่ สะโพก และข้อเท้าครบก่อนบันทึกภาพไหล่", None, None
    if metrics["body_height_px"] < _minimum_body_height(height):
        samples.clear()
        return "เห็นร่างกายเล็กเกินไปสำหรับเปรียบเทียบไหล่ — ขยับเข้าใกล้เล็กน้อย", None, None
    if marker_size_cm > 0 and marker_scale is None:
        return "วาง ArUco ID 0 ใกล้ระนาบไหล่ เพื่อแสดงความยาวเป็นเซนติเมตร", None, None

    scale = marker_scale
    if scale is None and user_height_cm > 0:
        nose_pt = metrics["points"].get("nose")
        ankle_pt = metrics["points"].get("ankle")
        if nose_pt is not None and ankle_pt is not None:
            scale = estimate_pixel_scale_from_height(nose_pt, ankle_pt, user_height_cm)
        else:
            pose = pose_result.pose_landmarks[0]
            nose_pt = (pose[0].x * width, pose[0].y * height)
            ankle_pt = (((pose[27].x + pose[28].x) / 2.0) * width, ((pose[27].y + pose[28].y) / 2.0) * height)
            scale = estimate_pixel_scale_from_height(nose_pt, ankle_pt, user_height_cm)

    metrics["pixels_per_cm"] = float(scale or 0.0)
    draw_shoulder_measurement_guides(frame, metrics["points"])
    samples.add(metrics)
    live_values = {
        "view": "front_shoulders",
        "left_ratio": metrics["left_shoulder_length_ratio"],
        "right_ratio": metrics["right_shoulder_length_ratio"],
        "difference_ratio": metrics["shoulder_length_difference_ratio"],
        "progress": samples.progress,
    }
    if not samples.ready:
        return (f"ขั้นที่ 1/2: ยืนนิ่งหน้าตรง… {samples.samples_used}/{samples.required_samples} เฟรม",
                None, live_values)

    summary = samples.result()
    if summary is None:
        return "กำลังรอข้อมูลไหล่", None, live_values
    if not summary["stable"]:
        samples.clear()
        return "ท่าไหล่เปลี่ยนระหว่างวัด — ยืนหน้าตรงนิ่ง ๆ แล้วเริ่มเก็บข้อมูลใหม่", None, None
    _add_absolute_shoulder_estimates(summary)
    return "บันทึกการเปรียบเทียบไหล่แล้ว — หันด้านข้างเพื่อวิเคราะห์ท่าทาง", summary, live_values


def _quality_and_posture_measurement(pose_result, frame, user_height_cm: float, marker_size_cm: float,
                                     marker_scale: float | None, samples: PostureSampleBuffer,
                                     shoulder_summary: dict | None = None):
    """Collect the side-view posture screen for kyphosis/slouching analysis."""
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
    if not _has_side_body(pose_result):
        samples.clear()
        return "ให้เห็นศีรษะ ไหล่ สะโพก และข้อเท้าชัดเจนในท่าด้านข้าง", None, None
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
        return "วาง ArUco ID 0 ใกล้ระนาบลำตัว เพื่อแสดงค่าหน่วยเซนติเมตร", None, None

    scale = marker_scale
    if scale is None and user_height_cm > 0:
        nose_pt = metrics["points"].get("nose") or metrics["points"].get("head")
        ankle_pt = metrics["points"].get("ankle")
        if nose_pt is not None and ankle_pt is not None:
            scale = estimate_pixel_scale_from_height(nose_pt, ankle_pt, user_height_cm)
        else:
            pose = pose_result.pose_landmarks[0]
            nose_pt = (pose[0].x * width, pose[0].y * height)
            ankle_pt = (((pose[27].x + pose[28].x) / 2.0) * width, ((pose[27].y + pose[28].y) / 2.0) * height)
            scale = estimate_pixel_scale_from_height(nose_pt, ankle_pt, user_height_cm)

    metrics["pixels_per_cm"] = float(scale or 0.0)
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
        return f"กำลังวิเคราะห์อาการหลังค่อมหลังงอ… {samples.samples_used}/{samples.required_samples} เฟรม", None, live_values

    summary = samples.result()
    if summary is None:
        return "กำลังรอข้อมูลท่าทาง", None, live_values
    if not summary["stable"]:
        samples.clear()
        return "ท่าทางเปลี่ยนระหว่างวัด — ยืนนิ่ง แล้วเริ่มเก็บข้อมูลใหม่", None, None

    _add_absolute_estimates(summary)
    if (marker_scale is not None and marker_scale > 0) or (marker_size_cm > 0 and summary.get("pixels_per_cm", 0) > 0):
        calibration = "ArUco marker (ค่าระยะเป็นเซนติเมตร)"
    elif user_height_cm > 0:
        calibration = f"ประเมินจากส่วนสูง ({user_height_cm:.0f} ซม.)"
    else:
        calibration = "ใช้สัดส่วนและมุมที่ไม่ขึ้นกับระยะกล้อง (ไม่มีค่าเซนติเมตร)"

    shoulder_samples_used = shoulder_summary["samples_used"] if shoulder_summary else 0
    quality = f"ความเชื่อมั่นท่าทางด้านข้าง {summary['landmark_confidence']:.2f}"
    if shoulder_summary:
        quality += f"; ไหล่ด้านหน้า {shoulder_summary['landmark_confidence']:.2f}"
    result = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "input_height_cm": user_height_cm,
        "calibration": calibration,
        "samples_used": summary["samples_used"],
        "total_samples_used": summary["samples_used"] + shoulder_samples_used,
        "quality": quality,
        "posture": summary,
        "shoulders": shoulder_summary,
        # Compatibility mirrors.
        "shoulder_cm": shoulder_summary.get("shoulder_span_cm") if shoulder_summary else None,
        "left_shoulder_cm": shoulder_summary.get("left_shoulder_length_cm") if shoulder_summary else None,
        "right_shoulder_cm": shoulder_summary.get("right_shoulder_length_cm") if shoulder_summary else None,
        "shoulder_difference_cm": shoulder_summary.get("shoulder_length_difference_cm") if shoulder_summary else None,
    }
    return "วิเคราะห์อาการหลังค่อมหลังงอเสร็จสิ้น", result, live_values


def run(user_height_cm: float = DEFAULT_HEIGHT_CM, marker_size_cm: float = DEFAULT_MARKER_CM,
        video_source: int | str = CAMERA_INDEX, measure_shoulders: bool = False):
    """Run the posture screen.

    Hold a peace sign for 1.5 seconds to begin.  Measures side profile posture
    for kyphosis/slouching screening.
    """
    if user_height_cm <= 0 or marker_size_cm < 0:
        raise ValueError("user_height_cm must be positive and marker_size_cm cannot be negative")
    cap = _open_camera(video_source)
    if cap is None:
        show_startup_error(
            f"ไม่สามารถเปิดกล้องหมายเลข {video_source} ได้\n\n"
            "ตรวจสอบว่ากล้องเชื่อมต่ออยู่ อนุญาตสิทธิ์ Camera ใน Windows แล้ว "
            "และปิดโปรแกรมอื่นที่กำลังใช้กล้อง"
        )
        return None

    ui = None
    vision = None
    last_measurement = None
    try:
        ui = MeasurementUI(WINDOW_TITLE, WINDOW_SIZE)
        vision = VisionEngine()
        smoother = PointSmoother(min_cutoff=0.3, beta=0.01)
        state_machine = GestureStateMachine(GESTURE_HOLD_SECONDS)
        shoulder_samples = ShoulderSampleBuffer(SHOULDER_STABLE_SAMPLES)
        posture_samples = PostureSampleBuffer(POSTURE_STABLE_SAMPLES)
        phase = MeasurementPhase.FRONT_SHOULDERS if measure_shoulders else MeasurementPhase.SIDE_POSTURE
        shoulder_summary = None
        front_capture_path = None
        camera_read_failures = 0
        while ui.is_open:
            ok, camera_frame = cap.read()
            if not ok or camera_frame is None or camera_frame.size == 0:
                camera_read_failures += 1
                # A brief empty frame is common with USB cameras.  Keeping the
                # loop alive prevents an in-progress measurement from ending
                # merely because one frame was dropped.
                if camera_read_failures < MAX_CONSECUTIVE_CAMERA_READ_FAILURES:
                    time.sleep(0.02)
                    continue
                show_startup_error(
                    "กล้องเปิดได้ แต่ไม่สามารถรับภาพต่อเนื่องได้\n\n"
                    "ลองถอด–เสียบกล้องใหม่ หรือปิดแอปอื่นที่กำลังใช้กล้อง",
                    parent=ui.root,
                )
                break
            camera_read_failures = 0
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
                smoothed_landmarks = []
                for idx, lm in enumerate(pose_result.pose_landmarks[0]):
                    if idx in UNUSED_FACE_LANDMARKS:
                        smoothed_landmarks.append(lm)
                    else:
                        sx, sy = smoother.update(f"lm_{idx}", timestamp, lm.x, lm.y)
                        smoothed_landmarks.append(
                            PoseLandmark(
                                x=sx,
                                y=sy,
                                z=lm.z,
                                visibility=lm.visibility,
                                presence=lm.presence,
                            )
                        )
                pose_result = PoseFrameResult(pose_landmarks=[smoothed_landmarks])
                draw_pose_landmarks(frame, pose_result.pose_landmarks[0])
            else:
                smoother.clear()

            peace = any(is_peace_sign(hand) for hand in hand_result.hand_landmarks)
            for hand in hand_result.hand_landmarks:
                draw_hand_landmarks(frame, hand)

            state, progress, transitioned = state_machine.update(peace, timestamp)
            if transitioned and state is State.MEASURING:
                shoulder_samples.clear()
                posture_samples.clear()
                smoother.clear()
                shoulder_summary = None
                phase = MeasurementPhase.FRONT_SHOULDERS if measure_shoulders else MeasurementPhase.SIDE_POSTURE
                front_capture_path = None
            if state is State.WAITING:
                if not pose_result.pose_landmarks:
                    status = "ไม่พบร่างกาย — ยืนในภาพเต็มตัว เพิ่มแสง หรือขยับใกล้ขึ้น"
                else:
                    landmarks = pose_result.pose_landmarks[0]
                    fh, fw = frame.shape[:2]
                    metrics, _ = analyse_posture_frame(landmarks, fw, fh)
                    if metrics and metrics.get("view") == "side":
                        status = "พร้อมวิเคราะห์อาการหลังค่อมหลังงอ — ชูสองนิ้วค้างเพื่อเริ่ม"
                    else:
                        status = "กรุณายืนหันด้านข้างลำตัว (ให้เห็นหู ไหล่ สะโพก ข้อเท้า) แล้วชูสองนิ้วค้างเพื่อเริ่ม"
                if peace:
                    status += f" ({progress * GESTURE_HOLD_SECONDS:.1f}/{GESTURE_HOLD_SECONDS:.1f} วินาที)"
            else:
                if phase is MeasurementPhase.FRONT_SHOULDERS:
                    status, completed_shoulders, live_values = _quality_and_shoulder_measurement(
                        pose_result, frame, user_height_cm, marker_size_cm, marker_scale, shoulder_samples)
                    new_measurement = None
                    if completed_shoulders is not None:
                        front_capture_path = save_full_body_capture(frame, "front_shoulders")
                        if front_capture_path is not None:
                            completed_shoulders["capture_file"] = front_capture_path.name
                        shoulder_summary = completed_shoulders
                        posture_samples.clear()
                        phase = MeasurementPhase.SIDE_POSTURE
                        show_phase_instruction(
                            "บันทึกภาพหน้าตรงแล้ว\n\n"
                            "หันลำตัวด้านข้างให้เห็นหู ไหล่ สะโพก และข้อเท้า "
                            "จากนั้นยืนนิ่งเพื่อวิเคราะห์อาการหลังค่อมหลังงอ",
                            parent=ui.root,
                        )
                        for _ in range(5):
                            cap.grab()
                else:
                    status, new_measurement, live_values = _quality_and_posture_measurement(
                        pose_result, frame, user_height_cm, marker_size_cm, marker_scale,
                        posture_samples, shoulder_summary)
                if live_values:
                    if live_values["view"] == "front_shoulders":
                        live_text = (
                            f"ไหล่ซ้าย: {live_values['left_ratio'] * 100:.1f}% ของช่วงไหล่ | "
                            f"ไหล่ขวา: {live_values['right_ratio'] * 100:.1f}%\n"
                            f"ส่วนต่างในภาพ: {live_values['difference_ratio'] * 100:.1f}% | "
                            f"เก็บข้อมูล {live_values['progress'] * 100:.0f}%"
                        )
                        _draw_overlay(ui, frame, live_text, (30, 70), 20, (0, 255, 255))
                    elif live_values["view"] == "side":
                        live_text = (
                            f"ศีรษะ–ไหล่: {live_values['head_ratio'] * 100:.1f}% ของลำตัว\n"
                            f"ไหล่–สะโพก: {live_values['shoulder_ratio'] * 100:.1f}% | "
                            f"เอียงลำตัว: {live_values['trunk_angle']:.1f}°\n"
                            f"รายละเอียดร่างกาย: {live_values['body_height_px']:.0f}px | "
                            f"เก็บข้อมูล {live_values['progress'] * 100:.0f}%"
                        )
                        _draw_overlay(ui, frame, live_text, (30, 70), 20, (0, 255, 255))
                    else:
                        _draw_overlay(ui, frame, f"ภาพหน้า: ไหล่เอียง {live_values['shoulder_tilt_deg']:.1f}° | สะโพกเอียง {live_values['hip_tilt_deg']:.1f}°", (30, 70), 21, (0, 255, 255))
                if new_measurement:
                    side_capture_path = save_full_body_capture(frame, "side_posture")
                    if front_capture_path is not None:
                        new_measurement["front_capture_file"] = front_capture_path.name
                    if side_capture_path is not None:
                        new_measurement["side_capture_file"] = side_capture_path.name
                        new_measurement["capture_file"] = side_capture_path.name
                    last_measurement = new_measurement
                    show_measurement_summary(new_measurement, parent=ui.root)
                    for _ in range(5):
                        cap.grab()
                    shoulder_samples.clear()
                    posture_samples.clear()
                    smoother.clear()
                    shoulder_summary = None
                    phase = MeasurementPhase.FRONT_SHOULDERS if measure_shoulders else MeasurementPhase.SIDE_POSTURE
                    state_machine.reset()
                    front_capture_path = None
                    continue
            if state is State.WAITING or phase is MeasurementPhase.FRONT_SHOULDERS:
                fh, fw = frame.shape[:2]
                cv2.line(frame, (int(fw * 0.40), int(fh * 0.05)), (int(fw * 0.40), int(fh * 0.95)), (100, 100, 100), 1, cv2.LINE_AA)
                cv2.line(frame, (int(fw * 0.60), int(fh * 0.05)), (int(fw * 0.60), int(fh * 0.95)), (100, 100, 100), 1, cv2.LINE_AA)
            _draw_overlay(ui, frame, status, (30, 30), 22, (255, 255, 255))
            ui.show_frame(frame)
    except Exception as error:
        # A desktop camera workflow must report every runtime failure visibly;
        # otherwise double-click launches can appear to close without a reason.
        show_startup_error(
            "ไม่สามารถเริ่มระบบตรวจจับได้\n\n"
            f"รายละเอียด: {error}\n\n"
            "ตรวจสอบว่าไฟล์โมเดล .task อยู่ในโฟลเดอร์โครงการ และติดตั้งไลบรารีตาม requirements.txt แล้ว",
            parent=ui.root if ui is not None else None,
        )
    finally:
        cap.release()
        if vision is not None:
            vision.close()
        if ui is not None:
            ui.destroy()
    return last_measurement


if __name__ == "__main__":
    print(run())

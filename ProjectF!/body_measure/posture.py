"""Distance-normalised posture screening built from MediaPipe pose landmarks.

This module deliberately calls its output a *screening* rather than a
diagnosis.  A conventional webcam has no direct thoracic-spine landmark, so
it can measure alignment patterns (ear--shoulder--hip--ankle) but cannot
confirm kyphosis or any other medical condition.  The metrics are designed
for a relaxed, side-on, full-body image with the camera roughly level with
the torso.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Iterable, Sequence

import numpy as np


# MediaPipe Pose landmark indexes used by the screening.  Keeping the values
# here makes the geometry testable without importing MediaPipe itself.
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _visibility(landmark: object) -> float:
    """Return a safe visibility value for MediaPipe-like landmark objects."""
    value = getattr(landmark, "visibility", 1.0)
    return float(1.0 if value is None else value)


def _point(landmark: object, width: int, height: int) -> tuple[float, float]:
    return float(getattr(landmark, "x")) * width, float(getattr(landmark, "y")) * height


def _centroid(landmarks: Sequence[object], indexes: Iterable[int], width: int, height: int,
              min_visibility: float) -> tuple[float, float] | None:
    points = [
        _point(landmarks[index], width, height)
        for index in indexes
        if index < len(landmarks) and _visibility(landmarks[index]) >= min_visibility
    ]
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _pair_span(landmarks: Sequence[object], first: int, second: int, width: int, height: int,
               min_visibility: float) -> float | None:
    if first >= len(landmarks) or second >= len(landmarks):
        return None
    if (_visibility(landmarks[first]) < min_visibility
            or _visibility(landmarks[second]) < min_visibility):
        return None
    return math.dist(_point(landmarks[first], width, height), _point(landmarks[second], width, height))


def _angle_from_vertical(top: tuple[float, float], bottom: tuple[float, float]) -> float:
    """Absolute angle in degrees from a vertical line through ``bottom``."""
    horizontal = abs(top[0] - bottom[0])
    vertical = abs(top[1] - bottom[1])
    return math.degrees(math.atan2(horizontal, max(vertical, 1e-6)))


def _joint_angle(first: tuple[float, float], vertex: tuple[float, float],
                 third: tuple[float, float]) -> float:
    """Return the 0--180 degree angle at ``vertex``."""
    first_vector = (first[0] - vertex[0], first[1] - vertex[1])
    third_vector = (third[0] - vertex[0], third[1] - vertex[1])
    first_length = math.hypot(*first_vector)
    third_length = math.hypot(*third_vector)
    if first_length <= 1e-6 or third_length <= 1e-6:
        return 0.0
    cosine = ((first_vector[0] * third_vector[0] + first_vector[1] * third_vector[1])
              / (first_length * third_length))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _segment_line_intersection(segment_start: tuple[float, float], segment_end: tuple[float, float],
                               line_start: tuple[float, float], line_end: tuple[float, float]) -> tuple[float, float] | None:
    """Intersect a shoulder segment with the estimated torso midline.

    Pose does not expose a literal neck landmark.  We use the intersection of
    the shoulder line and an upper-body midline as a repeatable reference for
    comparing the two sides.  An implausible intersection is rejected rather
    than silently replaced with a midpoint, which would falsely make a poor
    frame look symmetrical.
    """
    sx, sy = segment_start
    ex, ey = segment_end
    lx, ly = line_start
    rx, ry = line_end
    shoulder_x, shoulder_y = ex - sx, ey - sy
    line_x, line_y = rx - lx, ry - ly
    denominator = shoulder_x * line_y - shoulder_y * line_x
    if abs(denominator) < 1e-6:
        return None
    factor = ((lx - sx) * line_y - (ly - sy) * line_x) / denominator
    if factor < -0.05 or factor > 1.05:
        return None
    factor = max(0.0, min(1.0, factor))
    return sx + factor * shoulder_x, sy + factor * shoulder_y


def _band(value: float, caution: float, elevated: float) -> int:
    if value >= elevated:
        return 2
    if value >= caution:
        return 1
    return 0


def classify_profile(metrics: dict) -> dict:
    """Classify an aggregate profile-screening result without diagnosing it.

    The cut-offs intentionally flag an *alignment pattern*, not disease.  A
    clinician should assess anyone with pain, numbness, weakness, rapid
    change, or a persistent elevated result.
    """
    head_component = _band(metrics["head_shoulder_offset_ratio"], 0.07, 0.13)
    shoulder_component = _band(metrics["shoulder_hip_offset_ratio"], 0.07, 0.13)
    trunk_component = _band(metrics["trunk_inclination_deg"], 8.0, 15.0)
    score = head_component + shoulder_component + trunk_component

    # Ear and shoulder drifting to the same side of the hip is a more useful
    # rounded/forward alignment signal than either displacement by itself.
    chain_aligned = bool(metrics.get("same_side_chain", 0.0) >= 0.5)
    if score == 0:
        level = "neutral"
        label = "แนวศีรษะ ไหล่ และลำตัวอยู่ในช่วงค่อนข้างเป็นกลาง"
    elif score <= 2:
        level = "watch"
        label = "พบแนวโน้มการยื่นศีรษะ/ไหล่เล็กน้อย ควรติดตามท่ายืนและพักยืดเหยียด"
    else:
        level = "elevated"
        label = "พบรูปแบบการเยื้องของศีรษะ ไหล่ หรือช่วงอกค่อนข้างชัดเจน"
    if not chain_aligned and score:
        label += " (การเยื้องไม่ไปทิศเดียวกัน จึงควรตรวจซ้ำจากมุมกล้องที่ตรง)"
    return {
        "screening_level": level,
        "screening_score": score,
        "screening_label": label,
        "head_component": head_component,
        "shoulder_component": shoulder_component,
        "trunk_component": trunk_component,
    }


def classify_shoulder_balance(asymmetry_ratio: float, shoulder_span_px: float,
                              variability_ratio: float = 0.0) -> dict:
    """Describe left/right image-projection difference without clinical claims.

    A front camera cannot establish a skeletal shoulder asymmetry.  The
    tolerance is deliberately conservative: at least 7% of shoulder span,
    six pixels of landmark uncertainty, and twice the observed frame-to-frame
    variation.  The only actionable output is whether the image difference
    merits a repeat capture with the camera and body aligned.
    """
    tolerance_ratio = max(0.07, 6.0 / max(shoulder_span_px, 1.0), 2.0 * variability_ratio)
    if asymmetry_ratio <= tolerance_ratio:
        level = "within_tolerance"
        label = "ความต่างซ้าย–ขวาอยู่ในช่วงความคลาดเคลื่อนของภาพ"
    else:
        level = "recheck"
        label = "พบความต่างของระยะที่ฉายบนภาพ ควรวัดซ้ำโดยหันหน้าตรงและตั้งกล้องให้ตรง"
    return {
        "shoulder_balance_level": level,
        "shoulder_balance_label": label,
        "shoulder_difference_tolerance_ratio": tolerance_ratio,
    }


def analyse_posture_frame(landmarks: Sequence[object], frame_width: int, frame_height: int,
                          min_visibility: float = 0.42,
                          side_view_max_width_to_torso: float = 0.55) -> tuple[dict | None, str]:
    """Measure one frame and return ``(metrics, reason)``.

    Metrics are ratios of body segments or angles, so they stay meaningful as
    the subject moves farther from the camera.  Pixel size is still returned
    as a quality gate: below roughly 150 pixels of visible body height, a
    webcam does not contain enough information for a reliable screening.
    """
    if len(landmarks) <= RIGHT_ANKLE:
        return None, "ไม่พบจุดร่างกายครบถ้วน"

    ear = _centroid(landmarks, (LEFT_EAR, RIGHT_EAR), frame_width, frame_height, min_visibility)
    shoulder = _centroid(landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER), frame_width, frame_height,
                         min_visibility)
    hip = _centroid(landmarks, (LEFT_HIP, RIGHT_HIP), frame_width, frame_height, min_visibility)
    ankle = _centroid(landmarks, (LEFT_ANKLE, RIGHT_ANKLE), frame_width, frame_height, min_visibility)
    nose = _centroid(landmarks, (NOSE,), frame_width, frame_height, min_visibility)
    if not all((ear, shoulder, hip, ankle, nose)):
        return None, "ให้เห็นหู ไหล่ สะโพก และข้อเท้าชัดเจน"

    torso_px = math.dist(shoulder, hip)
    if torso_px < 20.0:
        return None, "ระยะลำตัวสั้นเกินไปสำหรับวิเคราะห์"

    shoulder_span = _pair_span(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER, frame_width, frame_height,
                                min_visibility)
    hip_span = _pair_span(landmarks, LEFT_HIP, RIGHT_HIP, frame_width, frame_height, min_visibility)
    if shoulder_span is None or hip_span is None:
        return None, "ให้เห็นไหล่และสะโพกทั้งสองข้าง"
    projected_width_ratio = (shoulder_span + hip_span) / (2.0 * torso_px)

    # Use the largest vertical extent of the landmarks that matter to the
    # screen; it is a direct proxy for available image detail at any distance.
    key_points = (ear, shoulder, hip, ankle)
    body_height_px = max(point[1] for point in key_points) - min(point[1] for point in key_points)
    confidences = []
    for index in (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                  LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE):
        if index < len(landmarks):
            confidences.append(_visibility(landmarks[index]))
    confidence = float(np.mean(confidences)) if confidences else 0.0

    common = {
        "body_height_px": float(body_height_px),
        "body_height_ratio": float(body_height_px / max(frame_height, 1)),
        "torso_px": float(torso_px),
        "view_width_to_torso_ratio": float(projected_width_ratio),
        "landmark_confidence": confidence,
        "points": {"ear": ear, "shoulder": shoulder, "hip": hip, "ankle": ankle, "nose": nose},
    }

    if projected_width_ratio > side_view_max_width_to_torso:
        shoulder_tilt = math.degrees(math.atan2(
            abs(_point(landmarks[RIGHT_SHOULDER], frame_width, frame_height)[1]
                - _point(landmarks[LEFT_SHOULDER], frame_width, frame_height)[1]),
            max(shoulder_span, 1e-6),
        ))
        hip_tilt = math.degrees(math.atan2(
            abs(_point(landmarks[RIGHT_HIP], frame_width, frame_height)[1]
                - _point(landmarks[LEFT_HIP], frame_width, frame_height)[1]),
            max(hip_span, 1e-6),
        ))
        common.update({
            "view": "front",
            "shoulder_tilt_deg": shoulder_tilt,
            "hip_tilt_deg": hip_tilt,
            "head_lateral_offset_ratio": abs(nose[0] - shoulder[0]) / max(shoulder_span, 1e-6),
        })
        return common, "ภาพด้านหน้าใช้ดูระดับไหล่ได้ แต่ยังประเมินหลังค่อมไม่ได้ — กรุณาหันด้านข้าง"

    ear_dx = ear[0] - shoulder[0]
    shoulder_dx = shoulder[0] - hip[0]
    # An exact zero is neutral rather than a direction conflict.
    same_side_chain = float(ear_dx * shoulder_dx >= 0.0)
    metrics = {
        **common,
        "view": "side",
        "head_shoulder_offset_ratio": abs(ear_dx) / torso_px,
        "shoulder_hip_offset_ratio": abs(shoulder_dx) / torso_px,
        "ear_hip_offset_ratio": abs(ear[0] - hip[0]) / torso_px,
        "hip_ankle_offset_ratio": abs(hip[0] - ankle[0]) / torso_px,
        "neck_inclination_deg": _angle_from_vertical(ear, shoulder),
        "trunk_inclination_deg": _angle_from_vertical(shoulder, hip),
        "lower_body_inclination_deg": _angle_from_vertical(hip, ankle),
        "ear_shoulder_hip_angle": _joint_angle(ear, shoulder, hip),
        "same_side_chain": same_side_chain,
    }
    metrics.update(classify_profile(metrics))
    return metrics, "พร้อมวิเคราะห์"


def analyse_front_shoulder_frame(landmarks: Sequence[object], frame_width: int, frame_height: int,
                                 min_visibility: float = 0.55,
                                 front_view_min_width_to_torso: float = 0.72,
                                 min_shoulder_span_px: float = 60.0) -> tuple[dict | None, str]:
    """Measure projected left/right shoulder lengths from a front-facing frame.

    Each length runs from a repeatable torso-midline proxy to the corresponding
    MediaPipe shoulder landmark.  The calculation is useful for comparing the
    two sides only in a front view.  It is intentionally not called a bone
    length and reports centimetres only when an ArUco marker calibrates the
    same image plane.
    """
    if len(landmarks) <= RIGHT_ANKLE:
        return None, "ไม่พบจุดร่างกายครบถ้วน"

    ears = _centroid(landmarks, (LEFT_EAR, RIGHT_EAR), frame_width, frame_height, min_visibility)
    left_shoulder = _centroid(landmarks, (LEFT_SHOULDER,), frame_width, frame_height, min_visibility)
    right_shoulder = _centroid(landmarks, (RIGHT_SHOULDER,), frame_width, frame_height, min_visibility)
    hip = _centroid(landmarks, (LEFT_HIP, RIGHT_HIP), frame_width, frame_height, min_visibility)
    ankle = _centroid(landmarks, (LEFT_ANKLE, RIGHT_ANKLE), frame_width, frame_height, min_visibility)
    if not all((ears, left_shoulder, right_shoulder, hip, ankle)):
        return None, "ให้เห็นหู ไหล่ สะโพก และข้อเท้าชัดเจน"

    shoulder_span_px = math.dist(left_shoulder, right_shoulder)
    torso_px = math.dist(((left_shoulder[0] + right_shoulder[0]) / 2.0,
                          (left_shoulder[1] + right_shoulder[1]) / 2.0), hip)
    hip_span_px = _pair_span(landmarks, LEFT_HIP, RIGHT_HIP, frame_width, frame_height,
                             min_visibility)
    if shoulder_span_px < min_shoulder_span_px or torso_px < 20.0 or hip_span_px is None:
        return None, "ภาพไหล่ยังเล็กเกินไปสำหรับเปรียบเทียบซ้าย–ขวา — ขยับเข้าใกล้เล็กน้อย"

    view_width_to_torso_ratio = (shoulder_span_px + hip_span_px) / (2.0 * torso_px)
    shoulder_width_to_torso_ratio = shoulder_span_px / torso_px
    hip_width_to_torso_ratio = hip_span_px / torso_px
    if (view_width_to_torso_ratio < front_view_min_width_to_torso
            or shoulder_width_to_torso_ratio < front_view_min_width_to_torso
            or hip_width_to_torso_ratio < front_view_min_width_to_torso * 0.75):
        return None, "หันหน้าตรงเข้ากล้องให้ไหล่ทั้งสองข้างเห็นชัด เพื่อวัดไหล่ซ้าย–ขวา"

    torso_midline_proxy = _segment_line_intersection(left_shoulder, right_shoulder, ears, hip)
    if torso_midline_proxy is None:
        return None, "แนวกึ่งกลางลำตัวยังไม่ชัด — ยืนตรงและหันหน้าตรงเข้ากล้อง"
    left_length_px = math.dist(left_shoulder, torso_midline_proxy)
    right_length_px = math.dist(right_shoulder, torso_midline_proxy)
    asymmetry_ratio = abs(left_length_px - right_length_px) / shoulder_span_px
    shoulder_tilt_deg = math.degrees(math.atan2(
        abs(right_shoulder[1] - left_shoulder[1]), max(shoulder_span_px, 1e-6),
    ))
    body_height_px = max(ears[1], ankle[1]) - min(ears[1], ankle[1])
    confidence_indexes = (LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                          LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE)
    confidence = float(np.mean([
        _visibility(landmarks[index]) for index in confidence_indexes
        if index < len(landmarks)
    ]))
    metrics = {
        "view": "front",
        "body_height_px": float(body_height_px),
        "body_height_ratio": float(body_height_px / max(frame_height, 1)),
        "torso_px": float(torso_px),
        "shoulder_span_px": float(shoulder_span_px),
        "view_width_to_torso_ratio": float(view_width_to_torso_ratio),
        "shoulder_width_to_torso_ratio": float(shoulder_width_to_torso_ratio),
        "hip_width_to_torso_ratio": float(hip_width_to_torso_ratio),
        "landmark_confidence": confidence,
        "left_shoulder_length_px": float(left_length_px),
        "right_shoulder_length_px": float(right_length_px),
        "left_shoulder_length_ratio": float(left_length_px / shoulder_span_px),
        "right_shoulder_length_ratio": float(right_length_px / shoulder_span_px),
        "shoulder_length_difference_px": float(abs(left_length_px - right_length_px)),
        "shoulder_length_difference_ratio": float(asymmetry_ratio),
        "shoulder_tilt_deg": float(shoulder_tilt_deg),
        "points": {
            "left_shoulder": left_shoulder,
            "torso_midline_proxy": torso_midline_proxy,
            "right_shoulder": right_shoulder,
        },
    }
    metrics.update(classify_shoulder_balance(asymmetry_ratio, shoulder_span_px))
    return metrics, "พร้อมวัดความยาวไหล่"


class ShoulderSampleBuffer:
    """Collect a stable, front-view comparison of left and right shoulders."""

    METRIC_KEYS = (
        "body_height_px",
        "body_height_ratio",
        "torso_px",
        "shoulder_span_px",
        "pixels_per_cm",
        "view_width_to_torso_ratio",
        "landmark_confidence",
        "left_shoulder_length_px",
        "right_shoulder_length_px",
        "left_shoulder_length_ratio",
        "right_shoulder_length_ratio",
        "shoulder_length_difference_px",
        "shoulder_length_difference_ratio",
        "shoulder_tilt_deg",
    )

    def __init__(self, required_samples: int = 18):
        if required_samples < 3:
            raise ValueError("required_samples must be at least 3")
        self.required_samples = required_samples
        self._samples: deque[dict] = deque(maxlen=required_samples)

    def clear(self) -> None:
        self._samples.clear()

    @property
    def samples_used(self) -> int:
        return len(self._samples)

    @property
    def progress(self) -> float:
        return min(1.0, self.samples_used / self.required_samples)

    @property
    def ready(self) -> bool:
        return self.samples_used >= self.required_samples

    def add(self, metrics: dict) -> None:
        if metrics.get("view") != "front":
            raise ValueError("Only front-view shoulder frames may be collected")
        self._samples.append({key: float(metrics[key]) for key in self.METRIC_KEYS})

    def result(self) -> dict | None:
        if not self.ready:
            return None
        values = {
            key: np.asarray([sample[key] for sample in self._samples], dtype=float)
            for key in self.METRIC_KEYS
        }
        summary = {key: float(np.median(value)) for key, value in values.items()}
        left_iqr = float(np.percentile(values["left_shoulder_length_ratio"], 75)
                         - np.percentile(values["left_shoulder_length_ratio"], 25))
        right_iqr = float(np.percentile(values["right_shoulder_length_ratio"], 75)
                          - np.percentile(values["right_shoulder_length_ratio"], 25))
        difference_iqr = float(np.percentile(values["shoulder_length_difference_ratio"], 75)
                               - np.percentile(values["shoulder_length_difference_ratio"], 25))
        tilt_iqr = float(np.percentile(values["shoulder_tilt_deg"], 75)
                         - np.percentile(values["shoulder_tilt_deg"], 25))
        stable = left_iqr <= 0.025 and right_iqr <= 0.025 and difference_iqr <= 0.020 and tilt_iqr <= 3.0
        summary.update(classify_shoulder_balance(
            summary["shoulder_length_difference_ratio"], summary["shoulder_span_px"], difference_iqr,
        ))
        summary.update({
            "view": "front",
            "samples_used": self.samples_used,
            "left_length_iqr": left_iqr,
            "right_length_iqr": right_iqr,
            "difference_iqr": difference_iqr,
            "tilt_iqr": tilt_iqr,
            "stable": stable,
        })
        return summary


class PostureSampleBuffer:
    """Collect robust medians from a continuous, side-view measurement run."""

    METRIC_KEYS = (
        "body_height_px",
        "body_height_ratio",
        "torso_px",
        "pixels_per_cm",
        "view_width_to_torso_ratio",
        "landmark_confidence",
        "head_shoulder_offset_ratio",
        "shoulder_hip_offset_ratio",
        "ear_hip_offset_ratio",
        "hip_ankle_offset_ratio",
        "neck_inclination_deg",
        "trunk_inclination_deg",
        "lower_body_inclination_deg",
        "ear_shoulder_hip_angle",
        "same_side_chain",
    )

    def __init__(self, required_samples: int = 24):
        if required_samples < 3:
            raise ValueError("required_samples must be at least 3")
        self.required_samples = required_samples
        self._samples: deque[dict] = deque(maxlen=required_samples)

    def clear(self) -> None:
        self._samples.clear()

    @property
    def samples_used(self) -> int:
        return len(self._samples)

    @property
    def progress(self) -> float:
        return min(1.0, self.samples_used / self.required_samples)

    @property
    def ready(self) -> bool:
        return self.samples_used >= self.required_samples

    def add(self, metrics: dict) -> None:
        if metrics.get("view") != "side":
            raise ValueError("Only side-view posture frames may be collected")
        self._samples.append({key: float(metrics[key]) for key in self.METRIC_KEYS})

    def result(self) -> dict | None:
        if not self.ready:
            return None
        values = {
            key: np.asarray([sample[key] for sample in self._samples], dtype=float)
            for key in self.METRIC_KEYS
        }
        summary = {key: float(np.median(value)) for key, value in values.items()}
        # IQR catches a user who turns or steps during the capture.  Keep the
        # values visible in the result so an operator can audit reliability.
        head_iqr = float(np.percentile(values["head_shoulder_offset_ratio"], 75)
                         - np.percentile(values["head_shoulder_offset_ratio"], 25))
        shoulder_iqr = float(np.percentile(values["shoulder_hip_offset_ratio"], 75)
                             - np.percentile(values["shoulder_hip_offset_ratio"], 25))
        trunk_iqr = float(np.percentile(values["trunk_inclination_deg"], 75)
                          - np.percentile(values["trunk_inclination_deg"], 25))
        stable = head_iqr <= 0.040 and shoulder_iqr <= 0.040 and trunk_iqr <= 4.0
        summary.update(classify_profile(summary))
        summary.update({
            "view": "side",
            "samples_used": self.samples_used,
            "head_offset_iqr": head_iqr,
            "shoulder_offset_iqr": shoulder_iqr,
            "trunk_angle_iqr": trunk_iqr,
            "stable": stable,
        })
        return summary

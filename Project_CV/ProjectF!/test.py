import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import math
import ctypes
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from collections import deque
from PIL import ImageFont, ImageDraw, Image

try:
    import pandas as pd
except ImportError:
    pd = None

# ----------------------------------------------------------------------
# 0. ฟังก์ชันวาดข้อความภาษาไทยด้วย PIL (cv2.putText ไม่รองรับอักษรไทย)
# ----------------------------------------------------------------------
FONT_PATH = r"C:\Windows\Fonts\tahoma.ttf"
FONT_BOLD_PATH = r"C:\Windows\Fonts\tahomabd.ttf"

_font_cache = {}


def _get_font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        path = FONT_BOLD_PATH if bold else FONT_PATH
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def put_thai_text(frame, text, pos, font_size=28, color=(255, 255, 255), bold=False):
    """วาดข้อความ (รองรับภาษาไทย) ลงบน OpenCV frame
    color เป็น BGR เหมือน OpenCV ปกติ"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size, bold)
    # แปลง BGR → RGB สำหรับ PIL
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)


def point_in_rect(x, y, rect):
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


# ----------------------------------------------------------------------
# 1. คลาส One Euro Filter (ตัวล็อกเส้นและตัวเลขนิ่งสนิท)
# ----------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return dt / (dt + tau)

    def __call__(self, t, x):
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        res_dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx = alpha_d * res_dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx)
        alpha = self._alpha(cutoff, dt)

        x_filtered = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_filtered
        self.dx_prev = dx
        self.t_prev = t

        return x_filtered

# ----------------------------------------------------------------------
# 2. ตั้งค่าระบบแลนด์มาร์ก (ทั้ง Pose และ Hand ใช้ Tasks API ตัวใหม่เหมือนกัน
#    เพราะ mediapipe เวอร์ชันนี้ไม่มี mp.solutions (Solutions API เก่า) ติดมาด้วยแล้ว)
# ----------------------------------------------------------------------
# --- โมเดล Pose (ร่างกาย) ---
pose_model_path = 'pose_landmarker_full.task'
if not os.path.exists(pose_model_path):
    print("กำลังดาวน์โหลด Pose Model... (ครั้งแรกอาจใช้เวลาสักครู่)")
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
    urllib.request.urlretrieve(url, pose_model_path)

# --- โมเดล Hand (มือ) ---
hand_model_path = 'hand_landmarker.task'
if not os.path.exists(hand_model_path):
    print("กำลังดาวน์โหลด Hand Model... (ครั้งแรกอาจใช้เวลาสักครู่)")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, hand_model_path)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=pose_model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,  # ตรวจจับแค่มือเดียวก็พอ
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

# โครงสร้างเส้นเชื่อมจุดมือ 21 จุด (เดิมเคยอยู่ใน mp.solutions.hands.HAND_CONNECTIONS
# แต่โมดูลนั้นไม่มีให้ใช้แล้ว เลยต้องกำหนดเองตามมาตรฐานเดิมของ MediaPipe)
HAND_CONNECTIONS = (
    (0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17),   # ฝ่ามือ
    (1, 2), (2, 3), (3, 4),                                # นิ้วโป้ง
    (5, 6), (6, 7), (7, 8),                                # นิ้วชี้
    (9, 10), (10, 11), (11, 12),                           # นิ้วกลาง
    (13, 14), (14, 15), (15, 16),                          # นิ้วนาง
    (17, 18), (18, 19), (19, 20),                          # นิ้วก้อย
)


def draw_hand_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# 3. รับค่าส่วนสูงเพื่อใช้ Calibration สเกลภาพ
# ----------------------------------------------------------------------
print("=" * 50)
try:
    USER_HEIGHT_CM = float(input("กรุณากรอกส่วนสูงจริงของคุณ (เซนติเมตร): "))
except ValueError:
    print("กรอกข้อมูลไม่ถูกต้อง ตั้งค่าเริ่มต้นเป็น 170 cm")
    USER_HEIGHT_CM = 170.0

# For a true cm scale, place an ArUco marker (DICT_4X4_50, id 0) flat on the
# same plane as the shoulders.  Set its printed side below (for example 10).
# Enter 0 only when no reference marker is available; that mode is an estimate.
try:
    REFERENCE_MARKER_CM = float(input("ขนาดด้านของ ArUco marker (cm, ใส่ 0 ถ้าไม่ใช้): "))
    if REFERENCE_MARKER_CM < 0:
        raise ValueError
except ValueError:
    REFERENCE_MARKER_CM = 0.0

if REFERENCE_MARKER_CM > 0 and not hasattr(cv2, "aruco"):
    print("OpenCV ArUco is unavailable; using height-based estimate instead.")
    REFERENCE_MARKER_CM = 0.0
print("=" * 50)

filters = {}


def filter_point(name, t, x, y):
    if name + "_x" not in filters:
        filters[name + "_x"] = OneEuroFilter(t, x, min_cutoff=0.3, beta=0.01)
        filters[name + "_y"] = OneEuroFilter(t, y, min_cutoff=0.3, beta=0.01)
    fx = filters[name + "_x"](t, x)
    fy = filters[name + "_y"](t, y)
    return fx, fy


def shoulder_center_from_body_axis(lx, ly, rx, ry, nx, ny, bx, by):
    shoulder_dx = rx - lx
    shoulder_dy = ry - ly
    body_dx = bx - nx
    body_dy = by - ny
    denom = shoulder_dx * body_dy - shoulder_dy * body_dx

    if abs(denom) < 1e-6:
        ratio = 0.5
    else:
        ratio = ((nx - lx) * body_dy - (ny - ly) * body_dx) / denom

    ratio = max(0.0, min(1.0, ratio))
    return lx + ratio * shoulder_dx, ly + ratio * shoulder_dy


def median_or_none(values):
    return float(np.median(values)) if values else None


class StableMeasurement:
    """Accept samples only after the subject is correctly positioned.

    Median aggregation rejects occasional landmark jumps much better than
    displaying a continually filtered value as though it were a measurement.
    """
    def __init__(self, required_samples=45):
        self.required_samples = required_samples
        self.shoulders = deque(maxlen=required_samples)
        self.lefts = deque(maxlen=required_samples)
        self.rights = deque(maxlen=required_samples)

    def clear(self):
        self.shoulders.clear()
        self.lefts.clear()
        self.rights.clear()

    def add(self, shoulder, left, right):
        self.shoulders.append(shoulder)
        self.lefts.append(left)
        self.rights.append(right)

    @property
    def progress(self):
        return len(self.shoulders) / self.required_samples

    @property
    def ready(self):
        return len(self.shoulders) >= self.required_samples

    def result(self):
        if not self.ready:
            return None
        return (
            median_or_none(self.shoulders),
            median_or_none(self.lefts),
            median_or_none(self.rights),
        )


def get_reference_scale(frame):
    """Return pixels/cm from a real-size marker, or None when unavailable."""
    if REFERENCE_MARKER_CM <= 0 or not hasattr(cv2, "aruco"):
        return None, None
    aruco = cv2.aruco
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(frame)
    except AttributeError:  # Compatible with older opencv-contrib builds.
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)
        corners, ids, _ = aruco.detectMarkers(frame, dictionary)

    if ids is None:
        return None, None
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) != 0:
            continue
        pts = marker_corners.reshape(4, 2)
        side_px = float(np.mean([
            np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)
        ]))
        if side_px > 1:
            return side_px / REFERENCE_MARKER_CM, pts.astype(int)
    return None, None


def draw_composition_guides(frame):
    """Draw a quiet, proportioned target area without obscuring landmarks."""
    h, w, _ = frame.shape
    left, right = int(w * 0.25), int(w * 0.75)
    top, bottom = int(h * 0.08), int(h * 0.92)
    shoulder_y = int(h * 0.34)
    centre_x = w // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (110, 175, 80), 2, cv2.LINE_AA)
    cv2.line(overlay, (centre_x, top), (centre_x, bottom), (105, 105, 105), 1, cv2.LINE_AA)
    cv2.line(overlay, (left, shoulder_y), (right, shoulder_y), (80, 160, 220), 1, cv2.LINE_AA)
    cv2.line(overlay, (left, bottom), (right, bottom), (90, 210, 90), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return left, top, right, bottom, shoulder_y


def build_measurement_rows(summary_data):
    if not summary_data:
        return []

    values = [
        ("Reference Height", summary_data.get("input_height_cm")),
        ("Shoulder Width", summary_data.get("shoulder_cm")),
        ("Left Shoulder Length", summary_data.get("left_shoulder_cm")),
        ("Right Shoulder Length", summary_data.get("right_shoulder_cm")),
    ]

    rows = []
    for name, value in values:
        rows.append({
            "Measurement": name,
            "Value (cm)": "" if value is None else round(float(value), 1),
        })
    return rows


def save_measurement_to_excel(summary_data):
    if not summary_data:
        messagebox.showwarning("No data", "No measurement data to save.")
        return

    if pd is None:
        messagebox.showerror(
            "Missing package",
            "pandas is not installed. Please install pandas and openpyxl to save Excel files.",
        )
        return

    default_name = "measurement_results_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".xlsx"
    file_path = filedialog.asksaveasfilename(
        title="Save measurement results",
        defaultextension=".xlsx",
        initialfile=default_name,
        filetypes=[("Excel Workbook", "*.xlsx")],
    )

    if not file_path:
        return

    try:
        summary_df = pd.DataFrame([{
            "Measured At": summary_data.get("measured_at", ""),
            "Input Height (cm)": round(float(summary_data.get("input_height_cm", 0)), 1),
            "Calibration": summary_data.get("calibration", "Estimated from height"),
            "Shoulder Width (cm)": round(float(summary_data.get("shoulder_cm", 0)), 1),
            "Left Shoulder Length (cm)": round(float(summary_data.get("left_shoulder_cm", 0)), 1),
            "Right Shoulder Length (cm)": round(float(summary_data.get("right_shoulder_cm", 0)), 1),
            "Stable Samples": summary_data.get("samples_used", 0),
        }])
        details_df = pd.DataFrame(build_measurement_rows(summary_data))

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            details_df.to_excel(writer, sheet_name="Measurements", index=False)

        messagebox.showinfo("Saved", "Saved Excel file successfully.")
    except Exception as exc:
        messagebox.showerror("Save failed", f"Could not save Excel file:\n{exc}")


def show_measurement_summary(summary_data):
    root = tk.Tk()
    root.title("Measurement Summary")
    root.configure(bg="#101418")
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda _event: close_summary_window())
    root.bind("<q>", lambda _event: close_summary_window())
    root.bind("<Q>", lambda _event: close_summary_window())
    root.bind("<F11>", lambda _event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    def close_summary_window():
        root.attributes("-fullscreen", False)
        root.quit()
        root.destroy()

    def minimize_summary_window():
        root.attributes("-fullscreen", False)
        root.geometry("960x540")
        root.update_idletasks()
        root.after(100, root.iconify)

    panel = tk.Frame(root, bg="#101418")
    panel.grid(row=0, column=0, sticky="nsew", padx=80, pady=60)
    panel.grid_columnconfigure(0, weight=1)

    title = tk.Label(
        panel,
        text="Measurement Summary",
        font=("Tahoma", 36, "bold"),
        fg="#f4f7fb",
        bg="#101418",
    )
    title.grid(row=0, column=0, sticky="w")

    subtitle_text = "Press Esc or Close to exit."
    if summary_data and summary_data.get("measured_at"):
        subtitle_text = "Measured at " + summary_data["measured_at"]
    if summary_data and summary_data.get("calibration"):
        subtitle_text += "  |  Calibration: " + summary_data["calibration"]

    subtitle = tk.Label(
        panel,
        text=subtitle_text,
        font=("Tahoma", 16),
        fg="#9aa8b8",
        bg="#101418",
    )
    subtitle.grid(row=1, column=0, sticky="w", pady=(8, 34))

    table = tk.Frame(panel, bg="#18202a", padx=28, pady=24)
    table.grid(row=2, column=0, sticky="ew")
    table.grid_columnconfigure(0, weight=1)
    table.grid_columnconfigure(1, weight=0)

    header_font = ("Tahoma", 18, "bold")
    body_font = ("Tahoma", 22)
    value_font = ("Tahoma", 26, "bold")

    tk.Label(table, text="Measurement", font=header_font, fg="#8fb3ff", bg="#18202a").grid(
        row=0, column=0, sticky="w", pady=(0, 16)
    )
    tk.Label(table, text="Value", font=header_font, fg="#8fb3ff", bg="#18202a").grid(
        row=0, column=1, sticky="e", pady=(0, 16)
    )

    rows = build_measurement_rows(summary_data)
    if not rows:
        tk.Label(
            table,
            text="No measurement data was captured.",
            font=body_font,
            fg="#f4f7fb",
            bg="#18202a",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=20)
    else:
        for index, row in enumerate(rows, start=1):
            value = row["Value (cm)"]
            value_text = "-" if value == "" else f"{value:.1f} cm"
            tk.Label(table, text=row["Measurement"], font=body_font, fg="#f4f7fb", bg="#18202a").grid(
                row=index, column=0, sticky="w", pady=12
            )
            tk.Label(table, text=value_text, font=value_font, fg="#7ee787", bg="#18202a").grid(
                row=index, column=1, sticky="e", pady=12, padx=(48, 0)
            )

    button_bar = tk.Frame(panel, bg="#101418")
    button_bar.grid(row=3, column=0, sticky="ew", pady=(34, 0))

    save_button = tk.Button(
        button_bar,
        text="Save Excel",
        command=lambda: save_measurement_to_excel(summary_data),
        font=("Tahoma", 18, "bold"),
        bg="#2f81f7",
        fg="#ffffff",
        activebackground="#1f6feb",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        padx=28,
        pady=12,
        state=tk.NORMAL if summary_data else tk.DISABLED,
    )
    save_button.pack(side=tk.LEFT)

    minimize_button = tk.Button(
        button_bar,
        text="Minimize",
        command=minimize_summary_window,
        font=("Tahoma", 18),
        bg="#30363d",
        fg="#f4f7fb",
        activebackground="#484f58",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        padx=28,
        pady=12,
    )
    minimize_button.pack(side=tk.LEFT, padx=(16, 0))

    close_button = tk.Button(
        button_bar,
        text="Close",
        command=close_summary_window,
        font=("Tahoma", 18),
        bg="#30363d",
        fg="#f4f7fb",
        activebackground="#484f58",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        padx=28,
        pady=12,
    )
    close_button.pack(side=tk.LEFT, padx=(16, 0))

    root.mainloop()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW_NAME = 'Real-time Precision Measurement'
WINDOWED_WIDTH = 960
WINDOWED_HEIGHT = 540
is_camera_fullscreen = True
camera_requested_exit = False
CAMERA_BUTTONS = {}
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def set_camera_fullscreen(enabled):
    global is_camera_fullscreen
    is_camera_fullscreen = enabled
    if enabled:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOWED_WIDTH, WINDOWED_HEIGHT)


def minimize_camera_window():
    set_camera_fullscreen(False)
    hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 6)


def draw_camera_controls(frame):
    h, w, _ = frame.shape
    min_rect = (w - 315, 15, w - 170, 58)
    exit_rect = (w - 155, 15, w - 35, 58)
    CAMERA_BUTTONS["minimize"] = min_rect
    CAMERA_BUTTONS["exit"] = exit_rect

    for rect, label, color in (
        (min_rect, "Minimize", (70, 130, 220)),
        (exit_rect, "Exit", (40, 40, 220)),
    ):
        left, top, right, bottom = rect
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (left + 14, top + 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)


def handle_camera_mouse(event, x, y, _flags, _param):
    global camera_requested_exit
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if point_in_rect(x, y, CAMERA_BUTTONS.get("exit", (0, 0, 0, 0))):
        camera_requested_exit = True
    elif point_in_rect(x, y, CAMERA_BUTTONS.get("minimize", (0, 0, 0, 0))):
        minimize_camera_window()


cv2.setMouseCallback(WINDOW_NAME, handle_camera_mouse)

smoothed_shoulder_cm = None
smoothed_left_shoulder_cm = None
smoothed_right_shoulder_cm = None
smoothed_height_cm = None
latest_measurement = None
VISIBILITY_THRESHOLD = 0.6
stable_measurement = StableMeasurement(required_samples=45)
marker_scale_history = deque(maxlen=20)
last_scale_px_per_cm = None
last_scale_source = "Height reference (estimated)"
quality_message = "Align yourself inside the frame"

# ตัวแปรสำหรับระบบ State Machine
# STATE: "WAITING"  = รอผู้ใช้ชู 2 นิ้ว (peace sign) เพื่อเริ่มวัด
#        "MEASURING" = กำลังวัดสัดส่วน
STATE = "WAITING"
gesture_start_time = None
HOLD_TIME = 1.5  # ต้องชู 2 นิ้วค้าง 1.5 วินาที


def is_peace_sign(hand_landmarks):
    """ตรวจจับท่า peace sign (ชู 2 นิ้ว: นิ้วชี้+นิ้วกลางขึ้น ส่วนนิ้วนาง+นิ้วก้อยพับ)
    ใช้ y-coordinate เทียบปลายนิ้ว (tip) กับข้อ PIP ซึ่งเสถียรกว่า distance จาก wrist
    ไม่ตรวจนิ้วโป้ง เพราะตอนทำ peace sign นิ้วโป้งมักพับไม่สนิท"""

    # ใน MediaPipe Hand, y น้อย = อยู่สูงในภาพ
    # นิ้วชี้: tip=8, pip=6  |  นิ้วกลาง: tip=12, pip=10
    # นิ้วนาง: tip=16, pip=14  |  นิ้วก้อย: tip=20, pip=18

    # นิ้วชี้ + นิ้วกลาง ต้องชูขึ้น (tip.y < pip.y)
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y

    # นิ้วนาง + นิ้วก้อย ต้องพับลง (tip.y > pip.y)
    ring_down = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_down = hand_landmarks[20].y > hand_landmarks[18].y

    return index_up and middle_up and ring_down and pinky_down


with PoseLandmarker.create_from_options(pose_options) as landmarker, \
     HandLandmarker.create_from_options(hand_options) as hand_landmarker:

    while cap.isOpened():
        success, camera_frame = cap.read()
        if not success:
            break

        # Detect the marker before mirroring.  A mirrored ArUco pattern may no
        # longer decode as the same ID, while the user-facing view stays mirrored.
        marker_scale, marker_corners = get_reference_scale(camera_frame)
        frame = cv2.flip(camera_frame, 1)
        h, w, _ = frame.shape
        t_now = time.time()
        analysis_frame = frame.copy()
        guide_left, guide_top, guide_right, guide_bottom, guide_shoulder_y = draw_composition_guides(frame)

        if marker_scale is not None:
            marker_scale_history.append(marker_scale)
            last_scale_px_per_cm = median_or_none(marker_scale_history)
            last_scale_source = "ArUco marker ID 0"
            marker_corners = marker_corners.copy()
            marker_corners[:, 0] = w - 1 - marker_corners[:, 0]
            cv2.polylines(frame, [marker_corners], True, (0, 255, 255), 2, cv2.LINE_AA)

        image_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(t_now * 1000)

        # =======================================================
        # ส่วนที่ 1: ตรวจจับมือ (Gesture Recognition)
        # =======================================================
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        is_peace = False

        if hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                # วาดเส้นโครงกระดูกมือบนหน้าจอ
                draw_hand_landmarks(frame, hand_landmarks, w, h)

                if is_peace_sign(hand_landmarks):
                    is_peace = True

        # =======================================================
        # ส่วนที่ 2: State Machine — ชู 2 นิ้ว (peace sign) สลับสถานะ
        # =======================================================
        should_exit = False

        if STATE == "WAITING":
            # ---- สถานะ: รอชู 2 นิ้วเพื่อเริ่มวัด ----
            if is_peace:
                if gesture_start_time is None:
                    gesture_start_time = t_now
                else:
                    elapsed = t_now - gesture_start_time
                    remain = max(0.0, HOLD_TIME - elapsed)

                    # แสดง progress bar
                    progress = min(1.0, elapsed / HOLD_TIME)
                    bar_w = 300
                    bar_x = w // 2 - bar_w // 2
                    bar_y = h // 2 + 60
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 30),
                                  (80, 80, 80), -1)
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + int(bar_w * progress), bar_y + 30),
                                  (0, 255, 100), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 30),
                                  (255, 255, 255), 2)

                    put_thai_text(frame, f"เริ่มวัดใน {remain:.1f} วินาที...",
                                  (bar_x - 20, bar_y - 40), font_size=26,
                                  color=(0, 255, 100), bold=True)

                    if remain <= 0:
                        STATE = "MEASURING"
                        gesture_start_time = None
                        # รีเซ็ตค่า filter ให้เริ่มวัดใหม่สด
                        filters.clear()
                        smoothed_shoulder_cm = None
                        smoothed_left_shoulder_cm = None
                        smoothed_right_shoulder_cm = None
                        smoothed_height_cm = None
                        stable_measurement.clear()
                        marker_scale_history.clear()
                        last_scale_px_per_cm = None
            else:
                gesture_start_time = None

            # แสดงข้อความแนะนำกลางจอ
            msg1 = "ยกมือชู 2 นิ้ว (peace sign) ค้างไว้ เพื่อเริ่มวัดสัดส่วน"
            msg2 = "กด 'q' เพื่อออก"
            put_thai_text(frame, msg1, (w // 2 - 250, h // 2 - 20),
                          font_size=28, color=(255, 255, 255), bold=True)
            put_thai_text(frame, msg2, (w // 2 - 80, h // 2 + 20),
                          font_size=22, color=(180, 180, 180))

        elif STATE == "MEASURING":
            # ---- สถานะ: กำลังวัดสัดส่วน ----
            # ตรวจจับชู 2 นิ้วอีกครั้ง → ปิดโปรแกรม
            if is_peace:
                if gesture_start_time is None:
                    gesture_start_time = t_now
                else:
                    elapsed = t_now - gesture_start_time
                    remain = max(0.0, HOLD_TIME - elapsed)

                    put_thai_text(frame, f"ปิดโปรแกรมใน {remain:.1f} วินาที",
                                  (w - 380, 15), font_size=26,
                                  color=(0, 0, 255), bold=True)

                    if remain <= 0:
                        should_exit = True
            else:
                gesture_start_time = None
                put_thai_text(frame, "ชู 2 นิ้วเพื่อปิดโปรแกรม", (w - 320, 15),
                              font_size=20, color=(200, 200, 200))

            # =======================================================
            # ส่วนที่ 3: ตรวจจับร่างกายและวัดสัดส่วน
            # =======================================================
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                l_sh = landmarks[11]
                r_sh = landmarks[12]
                nose = landmarks[0]
                l_an = landmarks[27]
                r_an = landmarks[28]

                points_visible = (getattr(l_sh, 'visibility', 1.0) > VISIBILITY_THRESHOLD and
                                   getattr(r_sh, 'visibility', 1.0) > VISIBILITY_THRESHOLD and
                                   getattr(l_an, 'visibility', 1.0) > VISIBILITY_THRESHOLD and
                                   getattr(r_an, 'visibility', 1.0) > VISIBILITY_THRESHOLD)

                if points_visible:
                    raw_lx, raw_ly = l_sh.x * w, l_sh.y * h
                    raw_rx, raw_ry = r_sh.x * w, r_sh.y * h
                    raw_nx, raw_ny = nose.x * w, nose.y * h

                    raw_bx = ((l_an.x + r_an.x) / 2.0) * w
                    raw_by = ((l_an.y + r_an.y) / 2.0) * h

                    # ส่งเข้าตัวกรองสัญญาณให้เส้นนิ่งสนิท
                    lx, ly = filter_point("left_shoulder", t_now, raw_lx, raw_ly)
                    rx, ry = filter_point("right_shoulder", t_now, raw_rx, raw_ry)
                    nx, ny = filter_point("nose", t_now, raw_nx, raw_ny)
                    bx, by = filter_point("base", t_now, raw_bx, raw_by)

                    neck_x, neck_y = shoulder_center_from_body_axis(lx, ly, rx, ry, nx, ny, bx, by)
                    shoulder_angle = abs(math.degrees(math.atan2(ry - ly, rx - lx)))
                    centered = abs(neck_x - w / 2) <= w * 0.10
                    shoulders_level = shoulder_angle <= 5.0
                    shoulder_in_guide = abs(neck_y - guide_shoulder_y) <= h * 0.14
                    full_body = guide_top <= ny and by <= guide_bottom

                    # A printed marker provides a real scale. The fallback keeps
                    # legacy behavior, but remains an estimate due to perspective.
                    if REFERENCE_MARKER_CM > 0:
                        pixels_per_cm = last_scale_px_per_cm if marker_scale is not None else None
                        calibration_ok = marker_scale is not None
                    else:
                        body_height_px = float(np.hypot(nx - bx, ny - by)) * 1.06
                        pixels_per_cm = body_height_px / USER_HEIGHT_CM
                        last_scale_source = "Height reference (estimated)"
                        calibration_ok = pixels_per_cm > 0

                    pose_is_stable = centered and shoulders_level and shoulder_in_guide and full_body
                    if not calibration_ok:
                        quality_message = "Show ArUco ID 0 beside your shoulders"
                    elif not centered:
                        quality_message = "Move to the centre line"
                    elif not shoulders_level:
                        quality_message = "Keep shoulders level"
                    elif not shoulder_in_guide:
                        quality_message = "Move until shoulders meet the blue line"
                    elif not full_body:
                        quality_message = "Keep head and feet inside the green frame"
                    else:
                        quality_message = "Hold still — collecting stable samples"

                    if pixels_per_cm and pixels_per_cm > 0:
                        shoulder_px = float(np.hypot(rx - lx, ry - ly))
                        raw_shoulder_cm = shoulder_px / pixels_per_cm
                        raw_left_shoulder_cm = float(np.hypot(lx - neck_x, ly - neck_y)) / pixels_per_cm
                        raw_right_shoulder_cm = float(np.hypot(rx - neck_x, ry - neck_y)) / pixels_per_cm
                        if pose_is_stable and calibration_ok:
                            stable_measurement.add(raw_shoulder_cm, raw_left_shoulder_cm, raw_right_shoulder_cm)
                            measured = stable_measurement.result()
                            if measured:
                                smoothed_shoulder_cm, smoothed_left_shoulder_cm, smoothed_right_shoulder_cm = measured
                                # Height is a calibration input, not a detected result.
                                smoothed_height_cm = None
                                latest_measurement = {
                                    "measured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "input_height_cm": USER_HEIGHT_CM,
                                    "shoulder_cm": smoothed_shoulder_cm,
                                    "left_shoulder_cm": smoothed_left_shoulder_cm,
                                    "right_shoulder_cm": smoothed_right_shoulder_cm,
                                    "samples_used": len(stable_measurement.shoulders),
                                    "calibration": last_scale_source,
                                }
                        else:
                            stable_measurement.clear()

                    p1 = (int(lx), int(ly))
                    p2 = (int(rx), int(ry))
                    center_point = (int(neck_x), int(neck_y))
                    cv2.circle(frame, p1, 8, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, p2, 8, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(frame, center_point, 6, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.line(frame, p1, p2, (255, 0, 0), 3, cv2.LINE_AA)
                    cv2.line(frame, p1, center_point, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, center_point, p2, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (int(nx), int(ny)), (int(bx), int(by)), (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    stable_measurement.clear()
                    quality_message = "Keep your full body visible"
                    put_thai_text(frame, "กรุณายืนให้เต็มเฟรม (หัวถึงเท้า)", (50, 120),
                                  font_size=24, color=(0, 165, 255))

            # แสดงผลลัพธ์การวัด
            # Show the acceptance state separately from recorded values so that
            # unstable frames never look like valid measurements.
            progress = stable_measurement.progress
            bar_x, bar_y, bar_w, bar_h = 34, h - 42, 320, 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (55, 55, 55), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (60, 210, 80), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (210, 210, 210), 1)
            cv2.putText(frame, quality_message, (bar_x, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Stable samples: {len(stable_measurement.shoulders)}/{stable_measurement.required_samples}",
                        (bar_x + bar_w + 14, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (120, 230, 140), 1, cv2.LINE_AA)
            calibration_label = "MARKER CALIBRATED" if REFERENCE_MARKER_CM > 0 else "HEIGHT-BASED ESTIMATE"
            calibration_color = (80, 230, 120) if REFERENCE_MARKER_CM > 0 else (0, 190, 255)
            cv2.putText(frame, calibration_label, (35, h - 78), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, calibration_color, 2, cv2.LINE_AA)

            if smoothed_height_cm is not None:
                put_thai_text(frame, f"ส่วนสูง: {smoothed_height_cm:.1f} ซม.", (50, 40),
                              font_size=30, color=(0, 255, 255), bold=True)

            if smoothed_shoulder_cm is not None:
                put_thai_text(frame, f"ความกว้างไหล่: {smoothed_shoulder_cm:.1f} ซม.", (50, 80),
                              font_size=28, color=(0, 255, 0), bold=True)

            if smoothed_left_shoulder_cm is not None and smoothed_right_shoulder_cm is not None:
                put_thai_text(frame, f"ไหล่ซ้าย: {smoothed_left_shoulder_cm:.1f} ซม.", (50, 120),
                              font_size=24, color=(255, 0, 255))
                put_thai_text(frame, f"ไหล่ขวา: {smoothed_right_shoulder_cm:.1f} ซม.", (50, 152),
                              font_size=24, color=(0, 165, 255))

        draw_camera_controls(frame)
        cv2.imshow(WINDOW_NAME, frame)
        # ป้องกันเหนียว: ยังคงสามารถกดปุ่ม 'q' บนคีย์บอร์ดเพื่อออกฉุกเฉินได้
        key = cv2.waitKeyEx(1)
        key_code = key & 0xFF if key != -1 else -1
        if key_code in (ord('m'), ord('M')):
            minimize_camera_window()
        if key_code in (ord('q'), ord('Q'), 27):
            camera_requested_exit = True

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            camera_requested_exit = True

        if should_exit or camera_requested_exit:
            break

cap.release()
cv2.destroyAllWindows()
show_measurement_summary(latest_measurement)

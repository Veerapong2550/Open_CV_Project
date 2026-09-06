"""Tkinter display owned by the application loop (no competing camera thread)."""

from __future__ import annotations

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


def _cm_text(value: float | None) -> str:
    return f"{value:.1f} ซม." if value is not None else "ไม่มี (ใช้ ArUco เพื่อแสดงเซนติเมตร)"


def _posture_summary_text(measurement: dict, posture: dict) -> str:
    """Format a readable, auditable report instead of a single opaque score."""
    reliability = "นิ่งและสม่ำเสมอ" if posture.get("stable") else "ความนิ่งไม่เพียงพอ"
    return (
        f"ผลคัดกรอง:  {posture.get('screening_label', 'ไม่มีผล') }\n"
        f"ระดับคัดกรอง:  {posture.get('screening_level', '-')}  |  คะแนน: {posture.get('screening_score', 0)}/6\n\n"
        "ตัวชี้วัดจากภาพด้านข้าง\n"
        f"• ระยะเยื้องศีรษะ–ไหล่:  {posture.get('head_shoulder_offset_ratio', 0) * 100:.1f}% ของลำตัว"
        f"  ({_cm_text(posture.get('head_shoulder_offset_cm'))})\n"
        f"• ระยะเยื้องไหล่–สะโพก:  {posture.get('shoulder_hip_offset_ratio', 0) * 100:.1f}% ของลำตัว"
        f"  ({_cm_text(posture.get('shoulder_hip_offset_cm'))})\n"
        f"• ระยะเยื้องสะโพก–ข้อเท้า:  {posture.get('hip_ankle_offset_ratio', 0) * 100:.1f}% ของลำตัว\n"
        f"• มุมเอียงคอ:  {posture.get('neck_inclination_deg', 0):.1f}°\n"
        f"• มุมเอียงลำตัว:  {posture.get('trunk_inclination_deg', 0):.1f}°\n"
        f"• มุมเอียงช่วงสะโพก–ข้อเท้า:  {posture.get('lower_body_inclination_deg', 0):.1f}°\n"
        f"• มุม หู–ไหล่–สะโพก:  {posture.get('ear_shoulder_hip_angle', 0):.1f}°\n"
        f"• ความยาวลำตัวในภาพ:  {_cm_text(posture.get('torso_cm'))}\n\n"
        "คุณภาพข้อมูล\n"
        f"• เฟรมที่ใช้: {posture.get('samples_used', measurement.get('samples_used', 0))} | {reliability}\n"
        f"• ความเชื่อมั่นจุดเฉลี่ย: {posture.get('landmark_confidence', 0):.2f}\n"
        f"• ความผันผวน (IQR): ศีรษะ {posture.get('head_offset_iqr', 0) * 100:.1f}%, "
        f"ไหล่ {posture.get('shoulder_offset_iqr', 0) * 100:.1f}%, "
        f"ลำตัว {posture.get('trunk_angle_iqr', 0):.1f}°\n\n"
        "ข้อควรทราบ: ผลนี้เป็นการคัดกรองแนวการจัดท่าจากกล้อง ไม่ใช่การวินิจฉัยหลังค่อมหรือโรคกระดูกสันหลัง "
        "หากมีอาการปวด ชา อ่อนแรง หรือกังวลกับผล ควรปรึกษาแพทย์/นักกายภาพบำบัด"
    )


def _legacy_summary_text(measurement: dict) -> str:
    """Keep prior callers usable while the app now produces posture reports."""
    def value(name: str) -> str:
        item = measurement.get(name)
        return f"{item:.1f} ซม." if isinstance(item, (int, float)) else "ไม่มี"

    return (
        f"ความกว้างไหล่: {value('shoulder_cm')}\n"
        f"ไหล่ซ้าย: {value('left_shoulder_cm')}\n"
        f"ไหล่ขวา: {value('right_shoulder_cm')}\n"
        f"ส่วนต่างซ้าย/ขวา: {value('shoulder_difference_cm')}\n\n"
        f"การปรับเทียบ: {measurement.get('calibration', 'ไม่มี')}\n"
        f"ตำแหน่ง: {measurement.get('quality', 'ไม่มี')}"
    )


def show_measurement_summary(measurement: dict, parent: tk.Misc | None = None) -> None:
    """Show a modal result screen, returning to the camera when it is closed."""
    root = tk.Tk() if parent is None else tk.Toplevel(parent)
    root.title("Posture Screening Summary")
    root.geometry("820x760")
    root.minsize(740, 620)
    root.configure(bg="#101418")
    if parent is not None:
        root.transient(parent)
        root.grab_set()
        root.attributes("-topmost", True)

    panel = tk.Frame(root, bg="#1b232c", padx=28, pady=24)
    panel.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
    posture = measurement.get("posture")
    heading = "ผลการคัดกรองท่าทาง" if posture else "สรุปผลการวัด"
    tk.Label(panel, text=heading, font=("Tahoma", 20, "bold"),
             fg="#f4f7fb", bg="#1b232c").pack(anchor=tk.W)
    tk.Label(panel, text=f"เวลา: {measurement.get('measured_at', '-')}", font=("Tahoma", 10),
             fg="#aebdca", bg="#1b232c").pack(anchor=tk.W, pady=(4, 14))

    summary_text = _posture_summary_text(measurement, posture) if posture else _legacy_summary_text(measurement)
    tk.Label(panel, text=summary_text, justify=tk.LEFT, anchor=tk.NW, wraplength=740,
             font=("Tahoma", 11), fg="#55d6be", bg="#1b232c").pack(fill=tk.BOTH, expand=True)
    footer = (f"การปรับเทียบ: {measurement.get('calibration', 'ไม่มี')}\n"
              f"ภาพเต็มตัว: {measurement.get('capture_file', 'ไม่ได้บันทึก')}")
    tk.Label(panel, text=footer, justify=tk.LEFT, anchor=tk.W, font=("Tahoma", 9),
             fg="#aebdca", bg="#1b232c").pack(fill=tk.X, pady=(8, 10))

    close = tk.Button(panel, text="วัดอีกครั้ง", command=root.destroy, font=("Tahoma", 11, "bold"),
                      bg="#2c7be5", fg="white", activebackground="#1f64c0", relief=tk.FLAT,
                      padx=28, pady=8, cursor="hand2")
    close.pack(anchor=tk.E)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    if parent is None:
        root.mainloop()
    else:
        root.update_idletasks()
        root.lift()
        root.focus_force()
        parent.wait_window(root)
        parent.lift()


class MeasurementUI:
    def __init__(self, window_title: str, window_size: tuple[int, int]):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry(f"{window_size[0]}x{window_size[1]}")
        self._open = True
        self.label = tk.Label(self.root, bg="#101418")
        self.label.pack(fill=tk.BOTH, expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("q", lambda _event: self.close())

    @property
    def is_open(self) -> bool:
        return self._open

    def show_frame(self, frame: np.ndarray) -> None:
        if not self._open:
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((self.root.winfo_width(), self.root.winfo_height()))
        photo = ImageTk.PhotoImage(image=image)
        self.label.configure(image=photo)
        self.label.image = photo
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        self._open = False

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

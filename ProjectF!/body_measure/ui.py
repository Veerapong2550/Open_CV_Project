"""Tkinter display owned by the application loop (no competing camera thread)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


BACKGROUND = "#101418"
PANEL = "#1b232c"
CARD = "#22303a"
MUTED = "#aebdca"
TEXT = "#f4f7fb"
ACCENT = "#55d6be"
CONTENT_WRAP = 620


def show_startup_error(message: str, parent: tk.Misc | None = None) -> None:
    """Make startup failures visible even when the camera window never opened."""
    temporary_root: tk.Tk | None = None
    try:
        if parent is None:
            temporary_root = tk.Tk()
            temporary_root.withdraw()
            parent = temporary_root
        messagebox.showerror("เปิดโปรแกรมไม่สำเร็จ", message, parent=parent)
    except tk.TclError:
        # A terminal still receives a useful error on headless machines or if
        # Windows cannot initialise Tk; previously this failure was silent for
        # people launching the program by double-clicking.
        print(f"เปิดโปรแกรมไม่สำเร็จ: {message}")
    finally:
        if temporary_root is not None:
            try:
                temporary_root.destroy()
            except tk.TclError:
                pass


def _status_style(level: str) -> dict[str, str]:
    """Give every screening level a label as well as a distinct colour."""
    styles = {
        "neutral": {
            "accent": "#36c98d",
            "background": "#173c32",
            "badge": "ผล: ค่อนข้างสมดุล",
            "headline": "ไม่พบแนวโน้มเด่นจากการวัดครั้งนี้",
            "next_step": "รักษาท่ายืนที่ผ่อนคลาย และพักเปลี่ยนอิริยาบถเป็นระยะ",
        },
        "watch": {
            "accent": "#f5c451",
            "background": "#443719",
            "badge": "ผล: ควรติดตาม",
            "headline": "พบแนวโน้มเล็กน้อยที่ควรสังเกต",
            "next_step": "ลองปรับระดับจอหรือเก้าอี้ ยืดเหยียดเบา ๆ แล้ววัดซ้ำในท่าผ่อนคลาย",
        },
        "elevated": {
            "accent": "#ff7d72",
            "background": "#492629",
            "badge": "ผล: ควรวัดซ้ำ",
            "headline": "พบแนวโน้มการเยื้องค่อนข้างชัดเจน",
            "next_step": "ตรวจซ้ำโดยตั้งกล้องให้ตรง; หากผลซ้ำร่วมกับปวด ชา หรืออ่อนแรง ควรปรึกษาผู้เชี่ยวชาญ",
        },
    }
    return styles.get(level, {
        "accent": "#8ca0b3",
        "background": "#28333d",
        "badge": "ผล: ต้องตรวจซ้ำ",
        "headline": "ข้อมูลยังไม่เพียงพอสำหรับสรุป",
        "next_step": "ตรวจซ้ำโดยให้เห็นร่างกายเต็มตัวและยืนนิ่ง",
    })


def _metric_style(component: float | int | None) -> tuple[str, str]:
    """Translate a technical 0/1/2 component into a plain-language badge."""
    if component is None or component <= 0:
        return "อยู่ในช่วงต่ำ", "#36c98d"
    if component < 2:
        return "ควรติดตาม", "#f5c451"
    return "เด่นชัด", "#ff7d72"


def _distance_text(value: float | None) -> str | None:
    return f"ประมาณ {value:.1f} ซม." if value is not None else None


def _unit_text(posture: dict) -> str:
    if posture.get("torso_cm") is not None:
        return "หน่วย: % ของลำตัว • องศา • เซนติเมตรจาก ArUco"
    return "หน่วย: % ของลำตัว และองศา"


def _section_title(parent: tk.Misc, text: str) -> None:
    tk.Label(parent, text=text, font=("Tahoma", 12, "bold"), fg=TEXT, bg=PANEL).pack(
        anchor=tk.W, pady=(16, 7)
    )


def _metric_card(parent: tk.Misc, column: int, *, title: str, value: str, subtitle: str,
                 component: float | int | None) -> None:
    """Draw one compact, readable primary metric card."""
    hint, colour = _metric_style(component)
    card = tk.Frame(parent, bg=CARD, padx=14, pady=12)
    card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5,
                                                            0 if column == 2 else 5))
    tk.Label(card, text=title, font=("Tahoma", 10, "bold"), fg=TEXT, bg=CARD).pack(anchor=tk.W)
    tk.Label(card, text=value, font=("Tahoma", 22, "bold"), fg=ACCENT, bg=CARD).pack(anchor=tk.W, pady=(5, 0))
    tk.Label(card, text=subtitle, font=("Tahoma", 8), fg=MUTED, bg=CARD,
             wraplength=190, justify=tk.LEFT).pack(anchor=tk.W, pady=(1, 7))
    tk.Label(card, text=hint, font=("Tahoma", 8, "bold"), fg=colour, bg=CARD).pack(anchor=tk.W)


def _quality_text(measurement: dict, posture: dict) -> tuple[str, str]:
    frames = posture.get("samples_used", measurement.get("samples_used", 0))
    stable = posture.get("stable", False)
    confidence = posture.get("landmark_confidence", 0.0)
    if stable and confidence >= 0.65:
        return f"คุณภาพดี • {frames} เฟรม • ความเชื่อมั่น {confidence:.2f}", "#36c98d"
    if stable:
        return f"ใช้ได้ • {frames} เฟรม • ความเชื่อมั่น {confidence:.2f}", "#f5c451"
    return f"ควรตรวจซ้ำ • {frames} เฟรม • ความนิ่งไม่พอ", "#ff7d72"


def _shoulder_status_style(level: str) -> tuple[str, str]:
    """Style a camera-image comparison without turning it into a diagnosis."""
    if level == "within_tolerance":
        return "ความต่างในภาพอยู่ในช่วงคลาดเคลื่อน", "#36c98d"
    return "แนะนำวัดซ้ำเพื่อยืนยันความต่างในภาพ", "#f5c451"


def _shoulder_length_text(shoulders: dict, *, cm_key: str, ratio_key: str) -> str:
    value_cm = shoulders.get(cm_key)
    if value_cm is not None:
        return f"{value_cm:.1f} ซม."
    return f"{shoulders.get(ratio_key, 0) * 100:.1f}%"


def _build_front_shoulder_section(parent: tk.Misc, shoulders: dict | None) -> None:
    """Show left/right shoulder data last, separate from side-view posture data."""
    _section_title(parent, "ภาพด้านหน้า: เปรียบเทียบระยะไหล่ซ้าย–ขวา")
    if not shoulders:
        tk.Label(parent, text="ไม่มีข้อมูลไหล่ด้านหน้าจากการวัดครั้งนี้", font=("Tahoma", 10),
                 fg=MUTED, bg=PANEL).pack(anchor=tk.W)
        return

    status_text, status_colour = _shoulder_status_style(str(shoulders.get("shoulder_balance_level", "")))
    card = tk.Frame(parent, bg=CARD, padx=14, pady=13)
    card.pack(fill=tk.X)
    tk.Label(card, text=status_text, font=("Tahoma", 10, "bold"), fg=status_colour,
             bg=CARD).pack(anchor=tk.W)
    tk.Label(card, text="ระยะจากแนวกึ่งกลางลำตัว (ประมาณ) ถึงหัวไหล่แต่ละข้าง • ซ้าย/ขวาตามภาพที่แสดง",
             font=("Tahoma", 9), fg=MUTED, bg=CARD, wraplength=CONTENT_WRAP,
             justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 10))

    values = tk.Frame(card, bg=CARD)
    values.pack(fill=tk.X)
    for column in range(2):
        values.grid_columnconfigure(column, weight=1, uniform="shoulder")
    left = _shoulder_length_text(
        shoulders, cm_key="left_shoulder_length_cm", ratio_key="left_shoulder_length_ratio",
    )
    right = _shoulder_length_text(
        shoulders, cm_key="right_shoulder_length_cm", ratio_key="right_shoulder_length_ratio",
    )
    for column, side, value, colour in (
        (0, "ไหล่ซ้าย", left, "#36c98d"),
        (1, "ไหล่ขวา", right, "#f5c451"),
    ):
        item = tk.Frame(values, bg="#1b2831", padx=12, pady=10)
        item.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5,
                                                               0 if column == 1 else 5))
        tk.Label(item, text=side, font=("Tahoma", 10, "bold"), fg=TEXT, bg="#1b2831").pack(anchor=tk.W)
        tk.Label(item, text=value, font=("Tahoma", 21, "bold"), fg=colour, bg="#1b2831").pack(anchor=tk.W, pady=(3, 0))
        if shoulders.get("left_shoulder_length_cm") is None:
            tk.Label(item, text="สัดส่วนของช่วงไหล่", font=("Tahoma", 8), fg=MUTED,
                     bg="#1b2831").pack(anchor=tk.W)

    difference_ratio = shoulders.get("shoulder_length_difference_ratio", 0) * 100
    difference_cm = shoulders.get("shoulder_length_difference_cm")
    difference = f"{difference_ratio:.1f}% ของช่วงไหล่"
    if difference_cm is not None:
        difference = f"{difference_cm:.1f} ซม. • {difference}"
    tk.Label(card, text=f"ส่วนต่างในภาพ: {difference}", font=("Tahoma", 10, "bold"),
             fg=TEXT, bg=CARD).pack(anchor=tk.W, pady=(11, 1))
    tk.Label(card, text=shoulders.get("shoulder_balance_label", "ควรวัดซ้ำเพื่อเปรียบเทียบ"),
             font=("Tahoma", 9), fg=MUTED, bg=CARD, wraplength=CONTENT_WRAP,
             justify=tk.LEFT).pack(anchor=tk.W)
    frames = shoulders.get("samples_used", 0)
    confidence = shoulders.get("landmark_confidence", 0.0)
    stable = "ผ่านการตรวจความนิ่ง" if shoulders.get("stable") else "ความนิ่งไม่เพียงพอ"
    tk.Label(card, text=f"คุณภาพภาพหน้า: {stable} • {frames} เฟรม • ความเชื่อมั่น {confidence:.2f}",
             font=("Tahoma", 8), fg="#8fa6b7", bg=CARD).pack(anchor=tk.W, pady=(8, 0))
    if shoulders.get("left_shoulder_length_cm") is None:
        tk.Label(card, text="ยังไม่มีค่าเซนติเมตร — ใช้ ArUco ที่ระนาบไหล่เพื่อปรับเทียบ",
                 font=("Tahoma", 8), fg="#8fa6b7", bg=CARD).pack(anchor=tk.W, pady=(2, 0))
    tk.Label(parent, text=("หมายเหตุ: เป็นระยะที่ฉายบนภาพเพื่อเปรียบเทียบสองข้าง "
                           "ไม่ใช่ความยาวกระดูกหรือการวินิจฉัย"),
             font=("Tahoma", 8), fg="#7f92a3", bg=PANEL, wraplength=CONTENT_WRAP,
             justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))


def _build_posture_summary(panel: tk.Misc, measurement: dict, posture: dict) -> None:
    """Build an information hierarchy for a non-technical reader first."""
    style = _status_style(str(posture.get("screening_level", "")))
    quality, quality_colour = _quality_text(measurement, posture)

    tk.Label(panel, text="ผลคัดกรองแนวการจัดท่าทาง", font=("Tahoma", 20, "bold"),
             fg=TEXT, bg=PANEL).pack(anchor=tk.W)
    tk.Label(panel, text=f"ภาพด้านข้าง  •  {measurement.get('measured_at', '-')}",
             font=("Tahoma", 10), fg=MUTED, bg=PANEL).pack(anchor=tk.W, pady=(3, 12))

    # The hero card puts the conclusion and the next action before the raw numbers.
    hero = tk.Frame(panel, bg=style["background"], padx=18, pady=15,
                    highlightbackground=style["accent"], highlightthickness=1)
    hero.pack(fill=tk.X)
    tk.Label(hero, text=style["badge"], font=("Tahoma", 10, "bold"), fg=style["accent"],
             bg=style["background"]).pack(anchor=tk.W)
    tk.Label(hero, text=style["headline"], font=("Tahoma", 17, "bold"), fg=TEXT,
             bg=style["background"], wraplength=CONTENT_WRAP, justify=tk.LEFT).pack(anchor=tk.W, pady=(3, 4))
    tk.Label(hero, text=posture.get("screening_label", "ผลคัดกรองจากแนวหู ไหล่ และสะโพก"),
             font=("Tahoma", 10), fg="#d5e2df", bg=style["background"], wraplength=CONTENT_WRAP,
             justify=tk.LEFT).pack(anchor=tk.W)
    tk.Label(hero, text=quality, font=("Tahoma", 9, "bold"), fg=quality_colour,
             bg=style["background"]).pack(anchor=tk.W, pady=(10, 0))

    action = tk.Frame(panel, bg="#1e2a34", padx=14, pady=10)
    action.pack(fill=tk.X, pady=(10, 0))
    tk.Label(action, text="สิ่งที่ควรทำต่อ", font=("Tahoma", 10, "bold"), fg=TEXT,
             bg="#1e2a34").pack(anchor=tk.W)
    tk.Label(action, text=style["next_step"], font=("Tahoma", 10), fg=MUTED,
             bg="#1e2a34", wraplength=CONTENT_WRAP, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

    _section_title(panel, "ตัวชี้วัดหลัก")
    tk.Label(panel, text=_unit_text(posture), font=("Tahoma", 9), fg=MUTED, bg=PANEL).pack(anchor=tk.W, pady=(0, 7))
    metrics = tk.Frame(panel, bg=PANEL)
    metrics.pack(fill=tk.X)
    for column in range(3):
        metrics.grid_columnconfigure(column, weight=1, uniform="metric")

    head_value = f"{posture.get('head_shoulder_offset_ratio', 0) * 100:.1f}%"
    head_cm = _distance_text(posture.get("head_shoulder_offset_cm"))
    _metric_card(
        metrics, 0, title="ศีรษะเทียบไหล่", value=head_value,
        subtitle=("ระยะเยื้องศีรษะจากแนวไหล่" + (f" • {head_cm}" if head_cm else "")),
        component=posture.get("head_component"),
    )
    shoulder_value = f"{posture.get('shoulder_hip_offset_ratio', 0) * 100:.1f}%"
    shoulder_cm = _distance_text(posture.get("shoulder_hip_offset_cm"))
    _metric_card(
        metrics, 1, title="ไหล่เทียบสะโพก", value=shoulder_value,
        subtitle=("แนวไหล่เยื้องจากแนวสะโพก" + (f" • {shoulder_cm}" if shoulder_cm else "")),
        component=posture.get("shoulder_component"),
    )
    _metric_card(
        metrics, 2, title="ลำตัวเอียง", value=f"{posture.get('trunk_inclination_deg', 0):.1f}°",
        subtitle="มุมเอียงของช่วงไหล่ถึงสะโพก",
        component=posture.get("trunk_component"),
    )

    _section_title(panel, "ความน่าเชื่อถือของผล")
    reliability = tk.Frame(panel, bg=CARD, padx=14, pady=10)
    reliability.pack(fill=tk.X)
    stable_text = "ผ่านการตรวจความนิ่ง" if posture.get("stable") else "ควรวัดใหม่ เนื่องจากขยับระหว่างวัด"
    tk.Label(reliability, text=stable_text, font=("Tahoma", 10, "bold"), fg=quality_colour,
             bg=CARD).grid(row=0, column=0, sticky=tk.W)
    tk.Label(reliability, text=f"ใช้ข้อมูล {posture.get('samples_used', measurement.get('samples_used', 0))} เฟรม",
             font=("Tahoma", 10), fg=TEXT, bg=CARD).grid(row=0, column=1, sticky=tk.W, padx=(30, 0))
    tk.Label(reliability, text=f"ความเชื่อมั่นจุด {posture.get('landmark_confidence', 0):.2f}",
             font=("Tahoma", 10), fg=TEXT, bg=CARD).grid(row=0, column=2, sticky=tk.W, padx=(30, 0))
    for column in range(3):
        reliability.grid_columnconfigure(column, weight=1)

    # Advanced values are available but stay out of the first-read path.
    detail_button = tk.Button(panel, text="ดูรายละเอียดตัวเลข ▾", font=("Tahoma", 9, "bold"),
                              bg="#2b3b47", fg=TEXT, activebackground="#3a4e5e",
                              activeforeground=TEXT, relief=tk.FLAT, padx=12, pady=6, cursor="hand2")
    detail_button.pack(anchor=tk.W, pady=(12, 0))
    details = tk.Frame(panel, bg="#172029", padx=14, pady=11)
    details_text = (
        f"มุมคอ {posture.get('neck_inclination_deg', 0):.1f}°  •  "
        f"มุมหู–ไหล่–สะโพก {posture.get('ear_shoulder_hip_angle', 0):.1f}°\n"
        f"สะโพก–ข้อเท้า {posture.get('hip_ankle_offset_ratio', 0) * 100:.1f}% ของลำตัว  •  "
        f"มุมช่วงล่าง {posture.get('lower_body_inclination_deg', 0):.1f}°\n"
        f"ความผันผวนระหว่างวัด: ศีรษะ {posture.get('head_offset_iqr', 0) * 100:.1f}%  •  "
        f"ไหล่ {posture.get('shoulder_offset_iqr', 0) * 100:.1f}%  •  "
        f"ลำตัว {posture.get('trunk_angle_iqr', 0):.1f}°"
    )
    tk.Label(details, text=details_text, font=("Tahoma", 9), fg=MUTED, bg="#172029",
             justify=tk.LEFT, wraplength=CONTENT_WRAP).pack(anchor=tk.W)

    # Keep the requested bilateral shoulder comparison at the end of the
    # readable report, separate from the side-view posture metrics.
    shoulder_section = tk.Frame(panel, bg=PANEL)
    shoulder_section.pack(fill=tk.X, pady=(16, 0))
    _build_front_shoulder_section(shoulder_section, measurement.get("shoulders"))

    footer = tk.Frame(panel, bg=PANEL)
    footer.pack(fill=tk.X, pady=(14, 0))
    tk.Label(footer, text=f"การปรับเทียบ: {measurement.get('calibration', 'ไม่มี')}",
             font=("Tahoma", 9), fg=MUTED, bg=PANEL).pack(anchor=tk.W)
    front_capture = measurement.get("front_capture_file")
    side_capture = measurement.get("side_capture_file", measurement.get("capture_file"))
    if front_capture:
        tk.Label(footer, text=f"ภาพที่บันทึกเมื่อวัดไหล่หน้าตรง: {front_capture}",
                 font=("Tahoma", 9), fg=MUTED, bg=PANEL).pack(anchor=tk.W, pady=(2, 0))
    if side_capture:
        tk.Label(footer, text=f"ภาพที่บันทึกเมื่อวัดท่าด้านข้าง: {side_capture}",
                 font=("Tahoma", 9), fg=MUTED, bg=PANEL).pack(anchor=tk.W, pady=(2, 0))
    tk.Label(footer, text=("ผลนี้ใช้คัดกรองแนวการจัดท่าจากกล้อง ไม่ใช่การวินิจฉัยโรค "
                           "หากมีปวด ชา อ่อนแรง หรือกังวลกับผล ควรปรึกษาแพทย์/นักกายภาพบำบัด"),
             font=("Tahoma", 8), fg="#7f92a3", bg=PANEL, wraplength=CONTENT_WRAP,
             justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

    details_visible = False

    def toggle_details() -> None:
        nonlocal details_visible
        details_visible = not details_visible
        if details_visible:
            details.pack(fill=tk.X, pady=(7, 0), before=shoulder_section)
            detail_button.configure(text="ซ่อนรายละเอียดตัวเลข ▴")
        else:
            details.pack_forget()
            detail_button.configure(text="ดูรายละเอียดตัวเลข ▾")

    detail_button.configure(command=toggle_details)


def _build_legacy_summary(panel: tk.Misc, measurement: dict) -> None:
    """Keep prior callers usable while using the same visual hierarchy."""
    def value(name: str) -> str:
        item = measurement.get(name)
        return f"{item:.1f} ซม." if isinstance(item, (int, float)) else "ไม่มี"

    tk.Label(panel, text="สรุปผลการวัด", font=("Tahoma", 20, "bold"), fg=TEXT, bg=PANEL).pack(anchor=tk.W)
    tk.Label(panel, text=f"เวลา: {measurement.get('measured_at', '-')}", font=("Tahoma", 10),
             fg=MUTED, bg=PANEL).pack(anchor=tk.W, pady=(3, 16))
    card = tk.Frame(panel, bg=CARD, padx=18, pady=16)
    card.pack(fill=tk.X)
    entries = (
        ("ความกว้างไหล่", value("shoulder_cm")),
        ("ไหล่ซ้าย", value("left_shoulder_cm")),
        ("ไหล่ขวา", value("right_shoulder_cm")),
        ("ส่วนต่างซ้าย/ขวา", value("shoulder_difference_cm")),
    )
    for row, (label, result) in enumerate(entries):
        tk.Label(card, text=label, font=("Tahoma", 11), fg=MUTED, bg=CARD).grid(row=row, column=0, sticky=tk.W, pady=3)
        tk.Label(card, text=result, font=("Tahoma", 12, "bold"), fg=ACCENT, bg=CARD).grid(row=row, column=1, sticky=tk.E, padx=(36, 0), pady=3)
    card.grid_columnconfigure(1, weight=1)
    tk.Label(panel, text=f"การปรับเทียบ: {measurement.get('calibration', 'ไม่มี')}",
             font=("Tahoma", 9), fg=MUTED, bg=PANEL).pack(anchor=tk.W, pady=(14, 0))


def _scrollable_panel(root: tk.Misc) -> tk.Frame:
    """Return a responsive vertical panel so text is never clipped on small DPI screens."""
    shell = tk.Frame(root, bg=BACKGROUND)
    shell.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
    canvas = tk.Canvas(shell, bg=BACKGROUND, highlightthickness=0, borderwidth=0)
    scrollbar = tk.Scrollbar(shell, orient=tk.VERTICAL, command=canvas.yview)
    panel = tk.Frame(canvas, bg=PANEL, padx=24, pady=22)
    panel_window = canvas.create_window((0, 0), window=panel, anchor=tk.NW)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    panel.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(panel_window, width=event.width))
    canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120), "units"))
    return panel


def show_measurement_summary(measurement: dict, parent: tk.Misc | None = None) -> None:
    """Show a modal result screen, returning to the camera when it is closed."""
    root = tk.Tk() if parent is None else tk.Toplevel(parent)
    root.title("ผลคัดกรองท่าทาง")
    root.geometry("900x780")
    root.minsize(760, 620)
    root.configure(bg=BACKGROUND)
    if parent is not None:
        root.transient(parent)
        root.grab_set()
        root.attributes("-topmost", True)

    panel = _scrollable_panel(root)
    posture = measurement.get("posture")
    if posture:
        _build_posture_summary(panel, measurement, posture)
    else:
        _build_legacy_summary(panel, measurement)

    close = tk.Button(panel, text="วัดอีกครั้ง", command=root.destroy, font=("Tahoma", 11, "bold"),
                      bg="#2c7be5", fg="white", activebackground="#1f64c0", relief=tk.FLAT,
                      padx=28, pady=8, cursor="hand2")
    close.pack(anchor=tk.E, pady=(16, 0))
    root.bind("<Escape>", lambda _event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    if parent is None:
        root.mainloop()
    else:
        root.update_idletasks()
        root.lift()
        root.focus_force()
        try:
            parent.wait_window(root)
            # The camera window may have been closed while this modal was
            # visible; lifting a destroyed Tk window would otherwise raise a
            # TclError and terminate the capture loop unexpectedly.
            if parent.winfo_exists():
                parent.lift()
        except tk.TclError:
            pass


class MeasurementUI:
    def __init__(self, window_title: str, window_size: tuple[int, int]):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry(f"{window_size[0]}x{window_size[1]}")
        self._open = True
        self.label = tk.Label(self.root, bg=BACKGROUND)
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
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((max(self.root.winfo_width(), 1), max(self.root.winfo_height(), 1)))
            photo = ImageTk.PhotoImage(image=image)
            self.label.configure(image=photo)
            self.label.image = photo
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            # Treat a window closed by Windows as a normal user-close rather
            # than propagating a GUI exception out of the camera loop.
            self._open = False

    def close(self) -> None:
        self._open = False

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

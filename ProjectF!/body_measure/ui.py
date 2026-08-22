"""Tkinter display owned by the application loop (no competing camera thread)."""

from __future__ import annotations

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


def show_measurement_summary(measurement: dict, parent: tk.Misc | None = None) -> None:
    """Show a modal result screen, returning to the camera when it is closed."""
    root = tk.Tk() if parent is None else tk.Toplevel(parent)
    root.title("Measurement Summary")
    root.geometry("700x560")
    root.resizable(False, False)
    root.configure(bg="#101418")
    if parent is not None:
        root.transient(parent)
        root.grab_set()
        root.attributes("-topmost", True)

    panel = tk.Frame(root, bg="#1b232c", padx=32, pady=28)
    panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    tk.Label(panel, text="Measurement complete", font=("Segoe UI", 20, "bold"),
             fg="#f4f7fb", bg="#1b232c").pack(anchor=tk.W)
    tk.Label(panel, text=f"Measured at: {measurement['measured_at']}", font=("Segoe UI", 10),
             fg="#aebdca", bg="#1b232c").pack(anchor=tk.W, pady=(4, 22))

    # A single multi-line label avoids geometry conflicts that could hide the
    # value column in a small or high-DPI display.
    summary_text = (
        f"Shoulder width:  {measurement.get('shoulder_cm', 0.0):.1f} cm\n"
        f"Left shoulder:     {measurement.get('left_shoulder_cm', 0.0):.1f} cm\n"
        f"Right shoulder:   {measurement.get('right_shoulder_cm', 0.0):.1f} cm\n"
        f"Left/right difference:  {measurement.get('shoulder_difference_cm', 0.0):.1f} cm\n"
        f"Input height:       {measurement.get('input_height_cm', 0.0):.1f} cm\n\n"
        f"Calibration:  {measurement.get('calibration', 'Not available')}\n"
        f"Position:       {measurement.get('quality', 'Not available')}\n"
        f"Full-body photo:  {measurement.get('capture_file', 'Not saved')}"
    )
    tk.Label(panel, text=summary_text, justify=tk.LEFT, anchor=tk.W,
             font=("Consolas", 14, "bold"), fg="#55d6be", bg="#1b232c").pack(fill=tk.X, pady=(0, 8))

    close = tk.Button(panel, text="Measure again", command=root.destroy, font=("Segoe UI", 11, "bold"),
                      bg="#2c7be5", fg="white", activebackground="#1f64c0", relief=tk.FLAT,
                      padx=28, pady=8, cursor="hand2")
    close.pack(anchor=tk.E, pady=(28, 0))
    root.bind("<Escape>", lambda _event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    if parent is None:
        root.mainloop()
    else:
        root.update_idletasks()
        # Make the result screen unmissable even when the live-camera window
        # was the active foreground window.
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

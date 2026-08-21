"""Tkinter display owned by the application loop (no competing camera thread)."""

from __future__ import annotations

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


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

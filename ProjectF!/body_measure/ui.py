"""
Tkinter based UI wrapper for the Real‑time Precision Body Measurement System.
Creates a window, embeds the OpenCV video feed, and shows measurement results.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import threading
import numpy as np

class MeasurementUI:
    def __init__(self, video_source=0, window_title="Precision Body Measurement", window_size=(960, 540)):
        self.root = tk.Tk()
        self.root.title(window_title)
        self.root.geometry(f"{window_size[0]}x{window_size[1]}")
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.label = ttk.Label(self.root)
        self.label.pack(fill=tk.BOTH, expand=True)
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw Thai text overlay onto the frame using Pillow.
        This can be extended to draw landmarks, guidelines, etc.
        """
        # Convert BGR to RGB for Pillow
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            # Use a generic font that supports Thai; replace with a specific .ttf if desired
            font = ImageFont.truetype("arial.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 10), "วัดสัดส่วนร่างกาย", fill="white", font=font)
        # Convert back to BGR for OpenCV display
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _update_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = self._draw_overlay(frame)
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.label.configure(image=img)
            self.label.image = img
            # Allow Tkinter event handling
            self.root.update_idletasks()
            self.root.update()

    def stop(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

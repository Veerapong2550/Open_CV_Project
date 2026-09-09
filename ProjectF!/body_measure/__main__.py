"""Run the integrated body measurement application as a module."""

from __future__ import annotations


def _show_launch_error(message: str) -> None:
    """Show missing-runtime errors to people who start the module by clicking."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("เปิดโปรแกรมไม่สำเร็จ", message, parent=root)
        root.destroy()
    except Exception:
        print(message)


def main() -> int:
    try:
        import argparse
        from .config import CAMERA_INDEX, DEFAULT_HEIGHT_CM, DEFAULT_MARKER_CM
        from .app import run

        parser = argparse.ArgumentParser(description="Precision Body Measurement System")
        parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_CM,
                            help=f"User height in cm for fallback calibration (default: {DEFAULT_HEIGHT_CM})")
        parser.add_argument("--marker-size", type=float, default=DEFAULT_MARKER_CM,
                            help=f"ArUco marker side length in cm (default: {DEFAULT_MARKER_CM})")
        parser.add_argument("--camera", default=str(CAMERA_INDEX),
                            help=f"Camera index or video file/stream path (default: {CAMERA_INDEX})")
        args, _ = parser.parse_known_args()

        camera_source: int | str = int(args.camera) if str(args.camera).isdigit() else args.camera
        run(user_height_cm=args.height, marker_size_cm=args.marker_size, video_source=camera_source)
        return 0
    except (ImportError, OSError) as error:
        _show_launch_error(
            "ยังติดตั้งไลบรารีที่จำเป็นไม่ครบ\n\n"
            f"รายละเอียด: {error}\n\n"
            "เปิด Command Prompt ในโฟลเดอร์โครงการ แล้วรัน:\n"
            "py -3 -m pip install -r requirements.txt"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

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
        from .app import run
    except ModuleNotFoundError as error:
        _show_launch_error(
            "ยังติดตั้งไลบรารีที่จำเป็นไม่ครบ\n\n"
            f"รายละเอียด: {error}\n\n"
            "เปิด Command Prompt ในโฟลเดอร์โครงการ แล้วรัน:\n"
            "py -3 -m pip install -r requirements.txt"
        )
        return 1
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

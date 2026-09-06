"""Top-level package for the real-time posture screening application."""


def run(*args, **kwargs):
    """Lazily load the camera runtime while keeping geometry tools importable."""
    from .app import run as camera_run
    return camera_run(*args, **kwargs)

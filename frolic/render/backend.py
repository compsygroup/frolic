"""Offscreen OpenGL backend selection for pyrender."""

from __future__ import annotations

import os
from typing import Any

from frolic3d.log import get_logger

log = get_logger(__name__)

HINT = (
    "install the render extra (pip install 'frolic3d[render]') and an EGL (GPU) or OSMesa (CPU) OpenGL; "
    "see README, Troubleshooting"
)


def _try(platform: str) -> bool:
    os.environ["PYOPENGL_PLATFORM"] = platform
    try:
        import pyrender

        r = pyrender.OffscreenRenderer(8, 8)
        r.delete()
        return True
    except Exception as exc:
        log.debug("%s backend unavailable: %s", platform, exc)
        return False


def select_backend(preferred: str = "auto") -> str:
    """Set PYOPENGL_PLATFORM to a working backend and return its name."""
    if "PYOPENGL_PLATFORM" in os.environ and preferred == "auto" and _try(os.environ["PYOPENGL_PLATFORM"]):
        return os.environ["PYOPENGL_PLATFORM"]
    candidates = ["egl", "osmesa"] if preferred == "auto" else [preferred]
    for c in candidates:
        if _try(c):
            log.info("render backend: %s", c)
            return c
    raise RuntimeError(f"no offscreen OpenGL backend among {candidates}: {HINT}")


def pyrender_module() -> Any:
    import pyrender

    return pyrender

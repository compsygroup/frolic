from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname).1s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("frolic3d") else f"frolic3d.{name}")


def configure(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger("frolic3d")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)

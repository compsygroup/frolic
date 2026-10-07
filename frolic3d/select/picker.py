"""Interactive / headless pickers that return the ids to keep."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from frolic3d.log import get_logger

log = get_logger(__name__)


def _parse_ids(text: str, ids: list[str]) -> list[str]:
    out = []
    for tok in text.replace(",", " ").split():
        tok = tok.strip()
        if tok in ids:
            out.append(tok)
        elif tok.isdigit() and int(tok) < len(ids):
            out.append(ids[int(tok)])
        elif tok:
            log.warning("unknown id %r ignored", tok)
    return list(dict.fromkeys(out))


def pick_file(path: Path, ids: list[str]) -> list[str]:
    lines = [ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines()]
    return _parse_ids(" ".join(ln for ln in lines if ln), ids)


def pick_cli(ids: list[str], people: list[dict[str, Any]], stream=None) -> list[str]:
    info = {p["id"]: p for p in people}
    stream = stream or sys.stdin
    print("tracked people (index  id  stature  duration  views):", file=sys.stderr)
    for k, pid in enumerate(ids):
        p = info.get(pid, {})
        st = p.get("stature_m")
        print(
            f"  {k:2d}  {pid:6s}  {st:.2f} m  {p.get('duration_s', 0):6.1f} s  {p.get('median_views', 0):.0f}"
            if st
            else f"  {k:2d}  {pid}",
            file=sys.stderr,
        )
    print("ids to keep (comma separated; empty = all): ", end="", file=sys.stderr, flush=True)
    line = stream.readline()
    sel = _parse_ids(line, ids)
    return sel if sel else list(ids)


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) or sys.platform in ("win32", "darwin")


def pick_opencv(sheet_jpg: Path, ids: list[str], row_h: int = 220) -> list[str] | None:
    """Click rows to toggle, press Enter to accept. None when no window can be shown."""
    import cv2

    if not has_display():
        return None
    img = cv2.imread(str(sheet_jpg))
    if img is None:
        return None
    state = {pid: True for pid in ids}

    def draw() -> np.ndarray:
        d = img.copy()
        for k, pid in enumerate(ids):
            col = (0, 200, 0) if state[pid] else (0, 0, 220)
            cv2.rectangle(d, (0, k * row_h), (d.shape[1] - 1, (k + 1) * row_h - 1), col, 4)
        return d

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and y // row_h < len(ids):
            pid = ids[y // row_h]
            state[pid] = not state[pid]
            cv2.imshow("frolic select", draw())

    try:
        cv2.namedWindow("frolic select", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("frolic select", on_mouse)
        cv2.imshow("frolic select", draw())
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k in (13, 10, 27, ord("q")):
                break
        cv2.destroyAllWindows()
    except cv2.error:
        return None
    return [pid for pid in ids if state[pid]]


def pick_matplotlib(sheet_jpg: Path, ids: list[str], row_h: int = 220) -> list[str] | None:
    try:
        import matplotlib

        if not has_display():
            return None
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    import cv2

    img = cv2.imread(str(sheet_jpg))
    if img is None:
        return None
    state = {pid: True for pid in ids}
    fig, ax = plt.subplots(figsize=(12, 1.2 * len(ids) + 1))
    ax.imshow(img[..., ::-1])
    ax.set_title("click a row to toggle; close the window to accept")
    rects = []
    for k in range(len(ids)):
        r = matplotlib.patches.Rectangle((0, k * row_h), img.shape[1], row_h, fill=False, lw=3, color="g")
        ax.add_patch(r)
        rects.append(r)

    def on_click(ev):
        if ev.ydata is None:
            return
        k = int(ev.ydata // row_h)
        if k < len(ids):
            state[ids[k]] = not state[ids[k]]
            rects[k].set_color("g" if state[ids[k]] else "r")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return [pid for pid in ids if state[pid]]


def pick(
    mode: str, ids: list[str], people: list[dict[str, Any]], sheet_jpg: Path, selection_file: Path | None
) -> tuple[list[str], str]:
    """Dispatch by picker mode with graceful headless fallbacks; returns (ids, mode actually used)."""
    if mode == "file" or (selection_file is not None and Path(selection_file).exists()):
        if selection_file is None or not Path(selection_file).exists():
            raise FileNotFoundError("select.picker = file needs select.selection_file")
        return pick_file(Path(selection_file), ids), "file"
    if mode in ("opencv", "matplotlib"):
        sel = pick_opencv(sheet_jpg, ids) if mode == "opencv" else pick_matplotlib(sheet_jpg, ids)
        if sel is not None:
            return sel, mode
        log.warning("no display for the %s picker: falling back to the terminal", mode)
    if sys.stdin is not None and sys.stdin.isatty():
        return pick_cli(ids, people), "cli"
    log.warning(
        "no terminal: keeping every person; write select.selection_file (one id per line) and rerun with --force"
    )
    return list(ids), "all"

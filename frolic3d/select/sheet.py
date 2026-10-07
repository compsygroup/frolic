"""Contact sheets of the tracked people (numbered tiles per identity) as JPG and self-contained HTML."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from frolic3d.io.detections import Detections
from frolic3d.io.tracks import Tracks
from frolic3d.io.video import FrameReader
from frolic3d.log import get_logger

log = get_logger(__name__)
TILE = (150, 220)


def pick_frames(tr: Tracks, p: int, n: int) -> list[tuple[int, int]]:
    """(track frame, camera index) tiles spread over the person's span, from the camera with the largest box."""
    idx = np.flatnonzero(tr.has_3d[p])
    if len(idx) == 0:
        return []
    want = idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    out = []
    for i in want:
        views = np.flatnonzero(tr.support[p, i] >= 0)
        if len(views) == 0:
            continue
        out.append((int(i), int(views[len(views) // 2])))
    return out


def crop_tile(img: np.ndarray, box: np.ndarray, pad: int = 20) -> np.ndarray:
    b = np.round(box).astype(int)
    x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad)
    c = img[y0 : b[3] + pad, x0 : b[2] + pad]
    if c.size == 0:
        return np.zeros((TILE[1], TILE[0], 3), np.uint8)
    return cv2.resize(c, TILE)


def person_tiles(
    tr: Tracks, p: int, dets: dict[str, Detections], videos: dict[str, Any], n: int, readers: dict[str, FrameReader]
) -> list[np.ndarray]:
    tiles = []
    for i, v in pick_frames(tr, p, n):
        cam = tr.cameras[v]
        if cam not in videos["videos"]:
            continue
        if cam not in readers:
            readers[cam] = FrameReader(videos["videos"][cam])
        f = tr.source_frame(i) + int(videos.get("frame_offset", {}).get(cam, 0))
        img = readers[cam].read(f)
        r = int(tr.support[p, i, v])
        if img is None or r < 0:
            continue
        tile = crop_tile(img, dets[cam].box[r])
        cv2.putText(
            tile, f"{i / tr.fps:.0f}s {cam}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(tile)
    while len(tiles) < n:
        tiles.append(np.zeros((TILE[1], TILE[0], 3), np.uint8))
    return tiles


def _label_tile(k: int, pid: str, info: dict[str, Any]) -> np.ndarray:
    lab = np.zeros((TILE[1], 130, 3), np.uint8)
    cv2.putText(lab, f"[{k}] {pid}", (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    st = info.get("stature_m")
    cv2.putText(
        lab, f"{st:.2f} m" if st else "? m", (4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        lab,
        f"{info.get('duration_s', 0):.0f} s",
        (4, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        lab,
        f"{info.get('median_views', 0):.0f} views",
        (4, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return lab


def contact_sheet(
    tr: Tracks,
    dets: dict[str, Detections],
    videos: dict[str, Any],
    people: list[dict[str, Any]],
    out_jpg: Path,
    out_html: Path,
    n: int = 8,
) -> dict[str, list[np.ndarray]]:
    readers: dict[str, FrameReader] = {}
    info = {p["id"]: p for p in people}
    rows, per_person = [], {}
    for k, pid in enumerate(tr.ids):
        tiles = person_tiles(tr, tr.index(pid), dets, videos, n, readers)
        per_person[pid] = tiles
        rows.append(np.hstack([_label_tile(k, pid, info.get(pid, {})), *tiles]))
    for r in readers.values():
        r.close()
    if rows:
        cv2.imwrite(str(out_jpg), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    write_html(tr, people, per_person, out_html)
    log.info("contact sheet: %d people -> %s, %s", len(rows), out_jpg, out_html)
    return per_person


def write_html(tr: Tracks, people: list[dict[str, Any]], tiles: dict[str, list[np.ndarray]], out_html: Path) -> None:
    info = {p["id"]: p for p in people}
    style = "body{font-family:sans-serif;background:#222;color:#eee}td{padding:4px}img{margin:1px}"
    parts = [
        f"<html><head><meta charset='utf-8'><title>FROLIC 3D people</title><style>{style}</style></head><body>",
        "<h2>Tracked people</h2><p>Write the ids to keep (one per line) into the selection file, "
        "or pass them to <code>frolic select</code>.</p><table>",
        "<tr><th>#</th><th>id</th><th>stature</th><th>duration</th><th>views</th><th>tiles</th></tr>",
    ]
    for k, pid in enumerate(tr.ids):
        p = info.get(pid, {})
        imgs = []
        for t in tiles.get(pid, []):
            ok, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                imgs.append(f"<img src='data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}'/>")
        st = p.get("stature_m")
        stat = f"{st:.2f} m" if st else "?"
        cells = [
            str(k),
            html.escape(pid),
            stat,
            f"{p.get('duration_s', 0):.1f} s",
            f"{p.get('median_views', 0):.0f}",
            "".join(imgs),
        ]
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    parts.append("</table></body></html>")
    out_html.write_text("\n".join(parts))

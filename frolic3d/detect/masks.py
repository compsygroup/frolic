"""Static masks (polygons from the config) and static-cell suppression (posters, screens, mirror reflections)."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from frolic3d.io.detections import Detections


def polygon_mask(polys: list[list[list[float]]], width: int, height: int) -> np.ndarray | None:
    if not polys:
        return None
    m = np.zeros((height, width), np.uint8)
    for poly in polys:
        pts = np.round(np.asarray(poly, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def box_mask_fraction(box: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the box area covered by the mask."""
    h, w = mask.shape
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())


def masked_rows(det: Detections, polys: list[list[list[float]]], overlap: float) -> np.ndarray:
    """Bool (M,) rows whose box lies inside a mask polygon by more than `overlap`."""
    out = np.zeros(len(det), bool)
    mask = polygon_mask(polys, det.width, det.height)
    if mask is None:
        return out
    for i, b in enumerate(det.box):
        out[i] = box_mask_fraction(b, mask) > overlap
    return out


def static_rows(
    det: Detections, occupancy: float, spread: float, cell_px: int = 60, iou_thr: float = 0.5
) -> np.ndarray:
    """Rows on a cell occupied in > `occupancy` of frames whose keypoints barely move (std / box height < spread)."""
    out = np.zeros(len(det), bool)
    if len(det) == 0:
        return out
    n_frames = max(1, len(det.frames))
    cell = np.stack(
        [(det.box[:, 0] + det.box[:, 2]) // (2 * cell_px), (det.box[:, 1] + det.box[:, 3]) // (2 * cell_px)], 1
    ).astype(int)
    rows_in: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, c in enumerate(map(tuple, cell)):
        rows_in[c].append(i)
    static_boxes = []
    for rows in rows_in.values():
        nf = len(np.unique(det.frame[rows]))
        if nf / n_frames <= occupancy:
            continue
        B = det.box[rows]
        K = det.kps[rows]
        h = max(1.0, float(np.median(B[:, 3] - B[:, 1])))
        if float(np.median(K.std(0).mean(1))) / h < spread:
            static_boxes.append(np.median(B, axis=0))
    if not static_boxes:
        return out
    from frolic3d.geometry import box_iou

    for i, b in enumerate(det.box):
        out[i] = any(box_iou(b, sb) > iou_thr for sb in static_boxes)
    return out

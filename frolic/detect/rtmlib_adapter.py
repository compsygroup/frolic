"""YOLOX + RTMPose (Halpe-26) through the rtmlib package (ONNX Runtime)."""

from __future__ import annotations

import numpy as np

from frolic.config import DetectConfig
from frolic.detect.adapter import DetectorOutput, empty_output
from frolic.log import get_logger

log = get_logger(__name__)

_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DET_MODELS: dict[str, tuple[str, tuple[int, int]]] = {
    "yolox_tiny": (_BASE + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", (416, 416)),
    "yolox_s": (_BASE + "yolox_s_8xb8-300e_humanart-3ef259a7.zip", (640, 640)),
    "yolox_m": (_BASE + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip", (640, 640)),
}
POSE_MODELS: dict[str, tuple[str, tuple[int, int]]] = {
    "rtmpose_x_halpe26": (
        _BASE + "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        (288, 384),
    ),
    "rtmpose_l_halpe26": (
        _BASE + "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230606.zip",
        (192, 256),
    ),
    "rtmpose_m_halpe26": (
        _BASE + "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        (192, 256),
    ),
}
INSTALL_HINT = "install the detector extra: pip install 'frolic[detect]' (rtmlib + onnxruntime-gpu)"


class RtmlibDetector:
    def __init__(self, cfg: DetectConfig, device: str = "cuda") -> None:
        try:
            from rtmlib import YOLOX, RTMPose
        except ImportError as e:
            raise ImportError(f"rtmlib is not available ({e}); {INSTALL_HINT}") from e
        if cfg.det_model not in DET_MODELS or cfg.pose_model not in POSE_MODELS:
            raise ValueError(f"unknown model; detectors {list(DET_MODELS)}, pose {list(POSE_MODELS)}")
        dev = "cuda" if device.startswith("cuda") else "cpu"
        durl, dsize = DET_MODELS[cfg.det_model]
        purl, psize = POSE_MODELS[cfg.pose_model]
        self.det = YOLOX(onnx_model=durl, model_input_size=dsize, backend="onnxruntime", device=dev)
        self.det.score_thr = cfg.det_threshold
        self.det.nms_thr = cfg.nms_threshold
        self.pose = RTMPose(onnx_model=purl, model_input_size=psize, backend="onnxruntime", device=dev)
        self.min_box = cfg.min_box_px
        log.info("rtmlib: %s + %s on %s", cfg.det_model, cfg.pose_model, dev)

    def __call__(self, img_bgr: np.ndarray) -> DetectorOutput:
        out = self.det(img_bgr)
        boxes = np.asarray(out[0] if isinstance(out, tuple) else out, np.float32).reshape(-1, 4)
        scores = (
            np.asarray(out[1], np.float32).ravel()
            if isinstance(out, tuple) and len(out) > 1
            else np.ones(len(boxes), np.float32)
        )
        keep = (boxes[:, 2] - boxes[:, 0] >= self.min_box) & (boxes[:, 3] - boxes[:, 1] >= self.min_box)
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) == 0:
            return empty_output()
        kps, ks = self.pose(img_bgr, boxes.tolist())
        return boxes, scores, np.asarray(kps, np.float32).reshape(-1, 26, 2), np.asarray(ks, np.float32).reshape(-1, 26)

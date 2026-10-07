"""Camera calibration file: `cameras.json` (schema `frolic3d.cameras/1`) and converters from other formats."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frolic3d.geometry import Camera

SCHEMA = "frolic3d.cameras/1"


@dataclass
class CameraSet:
    cameras: list[Camera]
    units: str = "m"
    floor: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self):
        return iter(self.cameras)

    def __getitem__(self, key: int | str) -> Camera:
        if isinstance(key, int):
            return self.cameras[key]
        return self.by_name[key]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cameras]

    @property
    def by_name(self) -> dict[str, Camera]:
        return {c.name: c for c in self.cameras}

    @property
    def centers(self) -> np.ndarray:
        return np.stack([c.center for c in self.cameras])

    def subset(self, names: list[str]) -> CameraSet:
        return CameraSet([self.by_name[n] for n in names], self.units, dict(self.floor), dict(self.source))

    def transformed(self, R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> CameraSet:
        return CameraSet(
            [c.transformed(R, t, scale) for c in self.cameras], self.units, dict(self.floor), dict(self.source)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "units": self.units,
            "world": {"up": "z", **self.floor},
            "source": self.source,
            "cameras": [
                {
                    "name": c.name,
                    "width": int(c.width),
                    "height": int(c.height),
                    "K": c.K.tolist(),
                    "dist": c.dist.tolist(),
                    "R": c.R.tolist(),
                    "t": c.t.tolist(),
                }
                for c in self.cameras
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraSet:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} file (schema={d.get('schema')!r})")
        cams = [
            Camera(c["name"], c["width"], c["height"], c["K"], c.get("dist", []), c["R"], c["t"]) for c in d["cameras"]
        ]
        world = dict(d.get("world", {}))
        world.pop("up", None)
        return cls(cams, d.get("units", "m"), world, dict(d.get("source", {})))


def save_cameras(cams: CameraSet, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cams.to_dict(), fh, indent=1)


def load_cameras(path: Path | str, fmt: str = "frolic") -> CameraSet:
    path = Path(path)
    if fmt == "frolic":
        with open(path) as fh:
            return CameraSet.from_dict(json.load(fh))
    if fmt == "gui":
        return from_gui_json(path)
    if fmt == "anipose":
        return from_anipose_toml(path)
    if fmt == "opencv":
        return from_opencv_yaml(path)
    raise ValueError(f"unknown calibration format {fmt!r}")


GUI_MM_THRESHOLD = 5.0  # a square_length above this is taken as millimetres


def _board_scale(board: dict[str, Any]) -> float:
    """Factor from the board's length unit to metres (the GUI accepts any unit; mm is the common one)."""
    unit = str(board.get("units", "")).lower()
    if unit in ("m", "metre", "meter", "metres", "meters"):
        return 1.0
    if unit in ("mm", "millimetre", "millimeter", "millimetres", "millimeters"):
        return 1e-3
    return 1e-3 if float(board.get("square_length", 1.0)) > GUI_MM_THRESHOLD else 1.0


def _pose(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = T[:3, :3], T[:3, 3]
    return out


def from_gui_json(path: Path) -> CameraSet:
    """`multi_camera_extrinsics` JSON of the ChArUco calibrator GUI (per-camera intrinsics JSON referenced inside).

    The world frame is the reference camera; `T_reference_to_camera` maps reference-camera coordinates into each camera.
    """
    path = Path(path)
    with open(path) as fh:
        d = json.load(fh)
    if d.get("mode") != "multi_camera_extrinsics":
        raise ValueError("expected a multi_camera_extrinsics JSON")
    scale = _board_scale(d.get("board", {}))
    cams = []
    for c in d["cameras"]:
        ip = Path(c["intrinsics_path"])
        if not ip.is_absolute():
            ip = path.parent / ip
        with open(ip) as fh:
            intr = json.load(fh)
        T = np.array(c["T_reference_to_camera"], float)
        size = intr.get("image_size", {})
        cams.append(
            Camera(
                c["camera_name"],
                int(size.get("width", 0)),
                int(size.get("height", 0)),
                np.array(intr["camera_matrix"], float),
                np.array(intr["distortion_coefficients"], float),
                T[:3, :3],
                T[:3, 3] * scale,
            )
        )
    return CameraSet(cams, "m", {}, {"format": "gui", "path": str(path), "reference": d.get("reference_camera")})


def gui_intrinsics_dict(
    cam: Camera, board: dict[str, Any], rms_px: float = 0.0, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "camera_name": cam.name,
        "mode": "intrinsics",
        "board": board,
        "image_size": {"width": cam.width, "height": cam.height},
        "reprojection_error": float(rms_px),
        "camera_matrix": cam.K.tolist(),
        "distortion_coefficients": cam.dist.tolist(),
        **(extra or {}),
    }


def to_gui_json(
    cams: CameraSet,
    out_dir: Path,
    board: dict[str, Any],
    reference: str | None = None,
    rms_px: dict[str, float] | None = None,
) -> Path:
    """Write per-camera intrinsics JSON + a multi_camera_extrinsics JSON in the ChArUco calibrator GUI format.

    Translations are written in the board's unit (metres unless `board` says millimetres); the reference camera is the
    GUI's world, so `T_reference_to_camera = T_world_to_cam @ inv(T_world_to_ref)`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = cams[reference or cams.names[0]]
    unit = 1.0 / _board_scale(board)
    T_world_to_ref = _pose(np.c_[ref.R, ref.t * unit])
    rms_px = rms_px or {}
    entries = []
    for c in cams:
        ipath = out_dir / f"{c.name}_intrinsics.json"
        with open(ipath, "w") as fh:
            json.dump(gui_intrinsics_dict(c, board, rms_px.get(c.name, 0.0)), fh, indent=2)
        T_world_to_cam = _pose(np.c_[c.R, c.t * unit])
        T_ref_to_cam = T_world_to_cam @ np.linalg.inv(T_world_to_ref)
        entries.append(
            {
                "camera_name": c.name,
                "intrinsics_path": str(ipath),
                "T_camera_to_reference": np.linalg.inv(T_ref_to_cam).tolist(),
                "T_reference_to_camera": T_ref_to_cam.tolist(),
                "reference_camera": ref.name,
            }
        )
    payload = {"mode": "multi_camera_extrinsics", "reference_camera": ref.name, "board": board, "cameras": entries}
    mpath = out_dir / "multi_camera_extrinsics.json"
    with open(mpath, "w") as fh:
        json.dump(payload, fh, indent=2)
    return mpath


def from_anipose_toml(path: Path) -> CameraSet:
    """aniposelib calibration.toml: [cam_N] with name, size, matrix, distortions, rotation (rvec), translation."""
    import cv2

    try:
        import tomllib
    except ModuleNotFoundError:  # python < 3.11
        import tomli as tomllib

    with open(path, "rb") as fh:
        d = tomllib.load(fh)
    cams = []
    for key in sorted(k for k in d if k.startswith("cam")):
        c = d[key]
        R = cv2.Rodrigues(np.array(c["rotation"], float))[0]
        size = c.get("size", [0, 0])
        cams.append(
            Camera(
                c.get("name", key),
                int(size[0]),
                int(size[1]),
                np.array(c["matrix"]),
                np.array(c.get("distortions", [])),
                R,
                np.array(c["translation"]),
            )
        )
    return CameraSet(cams, "m", {}, {"format": "anipose", "path": str(path)})


def from_opencv_yaml(path: Path) -> CameraSet:
    """A YAML/JSON with a list `cameras: [{name, width, height, K, dist, R, t}]` (OpenCV conventions)."""
    import yaml

    with open(path) as fh:
        d = yaml.safe_load(fh)
    cams = [Camera(c["name"], c["width"], c["height"], c["K"], c.get("dist", []), c["R"], c["t"]) for c in d["cameras"]]
    return CameraSet(cams, d.get("units", "m"), {}, {"format": "opencv", "path": str(path)})

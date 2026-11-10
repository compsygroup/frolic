"""The one YAML configuration (validated with pydantic). `frolic init` writes a commented template."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InputConfig(Strict):
    video_dir: Path = Field(Path("data/videos"), description="folder with one video per camera")
    pattern: str = Field("*.mp4", description="glob for the camera videos; sorted names define the camera order")
    camera_names: list[str] | None = Field(None, description="override camera names (one per matched video)")
    fps: float | None = Field(None, description="override the container frame rate")
    synced: bool = Field(True, description="videos start at the same instant (else run `frolic sync`)")
    start_s: float = Field(0.0, description="process from this time (after sync) ...")
    duration_s: float = Field(0.0, description="... for this long; 0 = to the end of the common overlap")
    work_dir: Path = Field(Path("work"), description="all stage outputs go here")


class SyncConfig(Strict):
    sample_rate: int = 8000
    max_audio_s: float = Field(150.0, description="seconds of audio used from each file (0 = whole file)")
    band_hz: tuple[float, float] = (300.0, 3000.0)
    envelope_ms: float = 10.0
    min_overlap_s: float = 15.0
    fine_window_ms: float = Field(80.0, description="GCC-PHAT search half-window around the coarse lag")
    max_closure_ms: float = Field(40.0, description="triangle-closure tolerance for a good sync")
    reference: str | None = Field(None, description="reference camera name (default: first)")
    trim: bool = Field(True, description="write trimmed, aligned copies into work/sync/videos")
    manual_offsets_s: dict[str, float] | None = Field(None, description="skip audio: camera -> offset in s")


class BoardConfig(Strict):
    squares_x: int = 8
    squares_y: int = 6
    square_length: float = Field(0.095, description="metres")
    marker_length: float = Field(0.071, description="metres")
    dictionary: str = "DICT_4X4_100"
    legacy_pattern: bool = True
    video_dir: Path | None = Field(None, description="board videos, same camera names as the input videos")
    pattern: str = "*.mp4"
    step: int = Field(3, description="detect the board in every step-th frame")
    min_corners: int = 12
    n_intrinsic_frames: int = 60
    max_ba_frames: int = 600
    fix_aspect_ratio: bool = True
    fix_k3: bool = True
    intrinsics_json: dict[str, Path] | None = Field(None, description="camera -> GUI-format intrinsics JSON to reuse")


class BootstrapConfig(Strict):
    focal_guess: float | None = Field(None, description="initial focal in px (default: search over fov_range_deg)")
    fov_range_deg: tuple[float, float] = (35.0, 110.0)
    fov_steps: int = 12
    min_pair_frames: int = 40
    max_frames: int = 400
    ransac_px: float = 4.0
    refine_intrinsics: bool = True
    person: Literal["most_seen", "tallest", "shortest"] = Field(
        "most_seen", description="which bootstrap person (grouped by clothing colour) has calib.reference_height_m"
    )


class CalibConfig(Strict):
    mode: Literal["board", "bootstrap", "given"] = "given"
    given: Path | None = Field(None, description="cameras.json (FROLIC schema) or a converted format")
    given_format: Literal["frolic", "gui", "anipose", "opencv"] = "frolic"
    board: BoardConfig = BoardConfig()
    bootstrap: BootstrapConfig = BootstrapConfig()
    reference_height_m: float | None = Field(
        None, description="bootstrap: stature (m) of the person chosen by calib.bootstrap.person, for metric scale"
    )
    level_floor: bool = Field(
        True, description="rotate the world onto the standing-feet plane; a given calibration that level_floor levelled on another clip is only z-shifted"
    )
    level_min_spread_m: float = Field(
        0.25,
        description="smallest horizontal std (m) the floor points need across their narrowest direction; "
        "below it the plane tilt is unconstrained (feet on a walkway line) and the orientation is kept",
    )
    ankle_height_m: float = Field(0.07, description="ankle joint height above the floor when standing")


class DetectConfig(Strict):
    detector: Literal["rtmlib", "mock"] = "rtmlib"
    mock_dir: Path | None = Field(None, description="detector = mock: folder with ground-truth <camera>.npz detections")
    det_model: str = "yolox_tiny"
    pose_model: str = "rtmpose_x_halpe26"
    det_threshold: float = 0.3
    nms_threshold: float = 0.65
    step: int = Field(1, description="run every step-th source frame")
    min_box_px: int = 24
    masks: dict[str, list[list[list[float]]]] = Field(
        default_factory=dict,
        description="camera -> list of polygons [[x, y], ...] in pixels; detections inside are dropped",
    )
    mask_overlap: float = Field(0.6, description="box fraction inside a mask that drops the detection")
    static_occupancy: float = Field(
        0.6, description="suppress cells occupied in > this fraction of frames (posters, mirrors)"
    )
    static_spread: float = Field(0.01, description="... whose keypoints barely move (std / box height)")
    noisy_occupancy: float = Field(
        0.45, description="also suppress cells occupied in > this fraction of frames by weak detections (0 = off)"
    )
    noisy_kp: float = Field(0.55, description="... mean keypoint score below this ...")
    noisy_box: float = Field(0.60, description="... and median box score below this")


class TrackConfig(Strict):
    fps: float = Field(15.0, description="track grid rate; source frames are subsampled to the nearest integer stride")
    min_views: int = 2
    min_score: float = 0.3
    reproj_px: float = 30.0
    max_people: int = Field(0, description="0 = estimate from the data")
    lost_frames: int = 60
    max_jump_m: float = 0.5
    room_bounds_m: list[float] | None = Field(
        None, description="[x0, x1, y0, y1] in the world frame; points outside are dropped"
    )
    appearance: bool = Field(True, description="torso colour and body size in the assignment cost")
    reid: Literal["osnet", "none"] = "osnet"
    reid_weights: Path | None = Field(None, description="OSNet checkpoint; see THIRD_PARTY.md")
    reid_every_s: float = 1.0
    identity_join: float = Field(0.25, description="centred-embedding cosine to join a segment to an identity")
    identity_merge: float = Field(0.25, description="cosine to merge identities that never co-occur")
    identity_cut: float = Field(0.2, description="cut a segment when its embedding drifts below this cosine")
    min_identity_frac: float = Field(0.03, description="identities carrying fewer frames than this are dropped")
    bridge_s: float = Field(3.0, description="gaps up to this long are interpolated inside an identity")
    max_speed_mps: float = 1.2
    teleport_mps: float = Field(
        3.5,
        description="identity: a short run entered and left faster than this (m/s) is another person's cluster and is "
        "dropped; keep above a running child's speed (2-3 m/s), max_speed_mps only widens the association gate",
    )
    max_pelvis_z_m: float = Field(1.15, description="identities floating above this are mirror ghosts")
    merge: list[list[str]] = Field(default_factory=list, description="identity ids to force-merge, e.g. [['p1', 'p3']]")
    split: dict[str, list[float]] = Field(default_factory=dict, description="identity -> times (s) at which to split")
    drop: list[str] = Field(default_factory=list)
    refine: bool = True
    refine_smooth: float = 200.0
    refine_length: float = 2.0
    euro_min_hz: float = 0.3
    euro_beta: float = 5.0


class SelectConfig(Strict):
    mode: Literal["auto", "interactive", "config"] = "auto"
    min_views: int = Field(2, description="auto: a person must be triangulated from >= this many views somewhere")
    min_track_s: float = 2.0
    min_height_m: float | None = None
    max_height_m: float | None = None
    children_only: bool = False
    child_height_m: float = 1.3
    max_people: int = 0
    ids: list[str] = Field(default_factory=list, description="config/interactive: identity ids to keep")
    labels: dict[str, str] = Field(default_factory=dict, description="identity id -> display name")
    picker: Literal["cli", "opencv", "matplotlib", "file"] = "cli"
    selection_file: Path | None = Field(None, description="headless: text file with one id per line")
    sheet_tiles: int = 8
    spans_min_views: int = Field(0, description="> 0: keep only spans where the person is seen by >= k views")
    spans_min_s: float = 1.0
    spans_bridge_s: float = 0.2


class SingleViewConfig(Strict):
    enabled: bool = True
    min_torso: float = 0.5
    min_box: float = 0.5
    min_kp: float = 0.5
    edge_px: float = 40.0
    gate_m: float = 0.35
    gate_max_m: float = 1.2
    max_prior_s: float = 6.0
    claim_m: float = 0.35
    multi_m: float = 0.15
    vmax_mps: float = 1.5
    sigma_prior_m: float = 0.10
    smooth_frames: float = 1.5
    room_margin_m: float = 0.10
    colour_margin: float = 0.5
    colour_abs: float = 3.0
    reid_margin: float = 0.02
    score: float = 0.25


class Sam3dConfig(Strict):
    enabled: bool = True
    backend: Literal["sam3d", "mock"] = "sam3d"
    root: Path | None = Field(None, description="SAM 3D Body checkout (or set FROLIC_SAM3D_ROOT)")
    checkpoint: str = "checkpoints/sam-3d-body-dinov3"
    gate_px: float = 80.0
    iters: int = 600
    w_prior: float = 2.0
    w_anchor: float = 0.5
    gap_vmax: float = Field(1.5, description="m/s: anchor tolerance growth for frames without a tracked mid-hip")
    w_acc: float = 20.0
    w_floor: float = 200.0
    sigma: float = 0.04


class BiomechConfig(Strict):
    enabled: bool = True
    limits: bool = True
    collision: bool = True
    foot: bool = True
    velocity: bool = True
    arm_collision: bool = False
    foot_collision: bool = True
    w_limits: float = 0.5
    w_collision: float = 0.5
    w_foot_heading: float = 0.3
    w_foot_yaw: float = 0.1
    w_floor: float = 0.3
    w_skate: float = 0.5
    w_velocity: float = 1.0
    collision_radius_scale: float = 0.85
    child_age_hi: float = Field(0.45, description="Anny age slider below which paediatric ranges blend in")
    shape_pass: bool = Field(True, description="shape from an unconstrained pass, then constrained pose only")
    ik_init: bool = True


class PersonConfig(Strict):
    label: str | None = None
    age_years: float | None = None
    height_m: float | None = None
    sex: Literal["male", "female"] | None = None
    weight_kg: float | None = None


class FitConfig(Strict):
    backend: Literal["anny", "mock"] = "anny"
    iters: int = 800
    fps: float | None = Field(None, description="defaults to track.fps")
    device: str = "cuda"
    anny_cache: Path | None = Field(None, description="Anny asset cache (or set ANNY_CACHE_DIR)")
    w_ours: float = 0.5
    w_sam: float = 1.0
    w_surface: float = 0.5
    w_hand: float = 0.5
    fit_fingers: bool = True
    dense: bool = True
    dense_n: int = 400
    w_dense: float = 0.5
    dense_max_d: float = 0.06
    dense_build: bool = True
    w_build_prior: float = Field(
        0.0,
        description="pull of the weight and muscle sliders (logit) toward the population middle when no mass is given; "
        "0 = off. Without it weight absorbs joint-spacing mismatches (adults 0.97, 3 y 0.96, 2026-11-09)",
    )
    segments: bool = True
    w_segment: float = 0.002
    w_length: float = 5.0
    segment_gate: float = 0.10
    ankle_bias: bool = True
    ankle_bias_max: float = Field(0.06, description="cap (m) on the shin-target ankle bias")
    w_smooth: float = 2.0
    smooth_v0: float = 0.4
    smooth_floor: float = 0.25
    sam_euro_min_hz: float = 0.5
    sam_euro_beta: float = 5.0
    fade_s: float = 0.5
    sam_jump_m: float = 0.12
    sam_body_gate_m: float = 0.25
    sam_gate_m: float = 0.15
    sv_weight: float = 0.3
    sv_bridge_s: float = 0.5
    root_continuity: bool = True
    density_kg_m3: float = 1010.0
    biomech: BiomechConfig = BiomechConfig()
    write_vertices: bool = True


class ExportConfig(Strict):
    max_gap_s: float = Field(1.0, description="holes between observations up to this long stay visible")
    observed_rule: bool = Field(True, description="hide a person only on frames no camera observes them")
    vertices_dtype: Literal["float16", "float32"] = "float16"
    full_rate_joints: bool = Field(False, description="also write joints interpolated to the source frame rate")


class RenderConfig(Strict):
    what: list[Literal["overlay", "grid", "room", "trajectories"]] = ["overlay", "room"]
    fps: int = 30
    crf: int = 18
    tile_width: int = 960
    grid_cols: int = Field(0, description="grid: cameras per row (0 = near-square tiling)")
    grid_tile_width: int = Field(0, description="grid: tile width in px (0 = tile_width)")
    room_width: int = 1920
    room_height: int = 1080
    alpha: float = 0.6
    pan_deg: float = 24.0
    room_zoom: float = Field(1.0, description="room/trajectories: camera distance divided by this (> 1 = closer)")
    elevation_deg: float = 22.0
    trail_s: float = 3.0
    footprints_s: float = Field(
        0.0, description="room: mark the floor where each foot lands and fade the mark out over this many seconds (0 = off)"
    )
    footprints_hold_s: float = Field(
        0.0, description="room: keep each footprint at full opacity for this long before it starts fading"
    )
    supersample: int = 1
    colours: dict[str, list[int]] = Field(default_factory=dict, description="person id -> RGB")
    start_s: float = 0.0
    duration_s: float = 0.0
    backend: Literal["egl", "osmesa", "auto"] = "auto"


class SlurmConfig(Strict):
    account: str | None = None
    partition_cpu: str = "cpu"
    partition_gpu: str = "gpu"
    gres_gpu: str = "gpu:1"
    cpus: int = 8
    mem: str = "32G"
    time_cpu: str = "04:00:00"
    time_gpu: str = "12:00:00"
    setup: list[str] = Field(
        default_factory=list, description="shell lines run before each stage (module load, conda activate)"
    )
    extra_sbatch: list[str] = Field(default_factory=list)


class ComputeConfig(Strict):
    device: str = "cuda"
    threads: int = 8
    stage_python: dict[str, Path] = Field(
        default_factory=dict, description="stage -> python interpreter (run a stage in a different environment)"
    )
    slurm: SlurmConfig = SlurmConfig()


class FrolicConfig(Strict):
    input: InputConfig = InputConfig()
    sync: SyncConfig = SyncConfig()
    calib: CalibConfig = CalibConfig()
    detect: DetectConfig = DetectConfig()
    track: TrackConfig = TrackConfig()
    select: SelectConfig = SelectConfig()
    single_view: SingleViewConfig = SingleViewConfig()
    sam3d: Sam3dConfig = Sam3dConfig()
    fit: FitConfig = FitConfig()
    export: ExportConfig = ExportConfig()
    render: RenderConfig = RenderConfig()
    compute: ComputeConfig = ComputeConfig()
    people: dict[str, PersonConfig] = Field(default_factory=dict, description="optional per-person attributes by id")

    @field_validator("people")
    @classmethod
    def _check_people(cls, v: dict[str, PersonConfig]) -> dict[str, PersonConfig]:
        for pid, p in v.items():
            if p.height_m is not None and not 0.3 < p.height_m < 2.5:
                raise ValueError(f"people.{pid}.height_m out of range")
            if p.age_years is not None and not 0 <= p.age_years < 120:
                raise ValueError(f"people.{pid}.age_years out of range")
        return v

    @model_validator(mode="after")
    def _check_modes(self) -> FrolicConfig:
        if self.calib.mode == "given" and self.calib.given is None:
            raise ValueError("calib.mode = given needs calib.given")
        if self.calib.mode == "bootstrap" and self.calib.reference_height_m is None:
            raise ValueError("calib.mode = bootstrap needs calib.reference_height_m for metric scale")
        if self.select.mode == "config" and not self.select.ids:
            raise ValueError("select.mode = config needs select.ids")
        if self.track.min_views < 2:
            raise ValueError("track.min_views must be >= 2 (single-view frames come from the single_view stage)")
        return self

    @property
    def work_dir(self) -> Path:
        return self.input.work_dir

    @property
    def fit_fps(self) -> float:
        return self.fit.fps or self.track.fps


def load_config(path: Path | str) -> FrolicConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    cfg = FrolicConfig.model_validate(data)
    base = Path(path).resolve().parent
    if not cfg.input.video_dir.is_absolute():
        cfg.input.video_dir = (base / cfg.input.video_dir).resolve()
    if not cfg.input.work_dir.is_absolute():
        cfg.input.work_dir = (base / cfg.input.work_dir).resolve()
    for owner, attr in (
        (cfg.calib, "given"),
        (cfg.calib.board, "video_dir"),
        (cfg.detect, "mock_dir"),
        (cfg.select, "selection_file"),
        (cfg.track, "reid_weights"),
        (cfg.sam3d, "root"),
        (cfg.fit, "anny_cache"),
    ):
        p = getattr(owner, attr)
        if p is not None and not Path(p).is_absolute():
            setattr(owner, attr, (base / p).resolve())
    return cfg


def save_config(cfg: FrolicConfig, path: Path | str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)


def _comment(model: type[BaseModel], indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = "  " * indent
    for name, f in model.model_fields.items():
        ann = f.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            lines.append(f"{pad}{name}:")
            lines += _comment(ann, indent + 1)
            continue
        default = f.get_default(call_default_factory=True)
        if isinstance(default, BaseModel):
            default = default.model_dump(mode="json")
        elif isinstance(default, Path):
            default = str(default)
        elif isinstance(default, tuple):
            default = list(default)
        val = yaml.safe_dump(default, default_flow_style=True).replace("\n...", "").strip()
        desc = f"  # {f.description}" if f.description else ""
        lines.append(f"{pad}{name}: {val}{desc}")
    return lines


TEMPLATE_OVERRIDES = {
    "  mode: given": "  mode: given  # board | bootstrap | given",
    "  given: null": "  given: calib/cameras.json",
}


def template() -> str:
    """A commented YAML template with every option at its default (and a valid `calib.given`)."""
    head = [
        "# FROLIC 3D configuration. Every key is optional; missing keys take the defaults shown here.",
        "# Paths are relative to this file. See docs/config.md for the meaning of each section.",
        "",
    ]
    body = _comment(FrolicConfig)
    out = []
    for ln in body:
        key = ln.split("#")[0].rstrip()
        out.append(TEMPLATE_OVERRIDES.get(key, ln))
    return "\n".join(head + out) + "\n"

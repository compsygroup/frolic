# Configuration reference

Generated from `frolic3d/config.py` by `docs/gen_config.py`.
`frolic init` writes a commented template with these defaults.

Paths are resolved relative to the configuration file. Unknown keys are rejected.

## `input`

Where the videos are and where the work goes.

| key | type | default | description |
|---|---|---|---|
| `input.video_dir` | `Path` | `data/videos` | folder with one video per camera |
| `input.pattern` | `str` | `*.mp4` | glob for the camera videos; sorted names define the camera order |
| `input.camera_names` | `list[str] or None` | `None` | override camera names (one per matched video) |
| `input.fps` | `float or None` | `None` | override the container frame rate |
| `input.synced` | `bool` | `True` | videos start at the same instant (else run `frolic sync`) |
| `input.start_s` | `float` | `0.0` | process from this time (after sync) ... |
| `input.duration_s` | `float` | `0.0` | ... for this long; 0 = to the end of the common overlap |
| `input.work_dir` | `Path` | `work` | all stage outputs go here |

## `sync`

Audio-based time alignment (`frolic sync`); ignored when `input.synced` is true.

| key | type | default | description |
|---|---|---|---|
| `sync.sample_rate` | `int` | `8000` |  |
| `sync.max_audio_s` | `float` | `150.0` | seconds of audio used from each file (0 = whole file) |
| `sync.band_hz` | `tuple[float, float]` | `(300.0, 3000.0)` |  |
| `sync.envelope_ms` | `float` | `10.0` |  |
| `sync.min_overlap_s` | `float` | `15.0` |  |
| `sync.fine_window_ms` | `float` | `80.0` | GCC-PHAT search half-window around the coarse lag |
| `sync.max_closure_ms` | `float` | `40.0` | triangle-closure tolerance for a good sync |
| `sync.reference` | `str or None` | `None` | reference camera name (default: first) |
| `sync.trim` | `bool` | `True` | write trimmed, aligned copies into work/sync/videos |
| `sync.manual_offsets_s` | `dict[str, float] or None` | `None` | skip audio: camera -> offset in s |

## `calib`

Camera calibration mode and floor levelling.

| key | type | default | description |
|---|---|---|---|
| `calib.mode` | `'board' or 'bootstrap' or 'given'` | `given` |  |
| `calib.given` | `pathlib.Path or None` | `None` | cameras.json (FROLIC schema) or a converted format |
| `calib.given_format` | `'frolic' or 'gui' or 'anipose' or 'opencv'` | `frolic` |  |

#### `calib.board`

| `calib.board.squares_x` | `int` | `8` |  |
| `calib.board.squares_y` | `int` | `6` |  |
| `calib.board.square_length` | `float` | `0.095` | metres |
| `calib.board.marker_length` | `float` | `0.071` | metres |
| `calib.board.dictionary` | `str` | `DICT_4X4_100` |  |
| `calib.board.legacy_pattern` | `bool` | `True` |  |
| `calib.board.video_dir` | `pathlib.Path or None` | `None` | board videos, same camera names as the input videos |
| `calib.board.pattern` | `str` | `*.mp4` |  |
| `calib.board.step` | `int` | `3` | detect the board in every step-th frame |
| `calib.board.min_corners` | `int` | `12` |  |
| `calib.board.n_intrinsic_frames` | `int` | `60` |  |
| `calib.board.max_ba_frames` | `int` | `600` |  |
| `calib.board.fix_aspect_ratio` | `bool` | `True` |  |
| `calib.board.fix_k3` | `bool` | `True` |  |
| `calib.board.intrinsics_json` | `dict[str, pathlib.Path] or None` | `None` | camera -> GUI-format intrinsics JSON to reuse |

#### `calib.bootstrap`

| `calib.bootstrap.focal_guess` | `float or None` | `None` | initial focal in px (default: search over fov_range_deg) |
| `calib.bootstrap.fov_range_deg` | `tuple[float, float]` | `(35.0, 110.0)` |  |
| `calib.bootstrap.fov_steps` | `int` | `12` |  |
| `calib.bootstrap.min_pair_frames` | `int` | `40` |  |
| `calib.bootstrap.max_frames` | `int` | `400` |  |
| `calib.bootstrap.ransac_px` | `float` | `4.0` |  |
| `calib.bootstrap.refine_intrinsics` | `bool` | `True` |  |
| `calib.bootstrap.person` | `'most_seen' or 'tallest' or 'shortest'` | `most_seen` | which bootstrap person (grouped by clothing colour) has calib.reference_height_m |
| `calib.reference_height_m` | `float or None` | `None` | bootstrap: stature (m) of the person chosen by calib.bootstrap.person, for metric scale |
| `calib.level_floor` | `bool` | `True` | rotate the world onto the standing-feet plane; a given calibration that level_floor levelled on another clip is only z-shifted |
| `calib.level_min_spread_m` | `float` | `0.25` | smallest horizontal std (m) the floor points need across their narrowest direction; below it the plane tilt is unconstrained (feet on a walkway line) and the orientation is kept |
| `calib.ankle_height_m` | `float` | `0.07` | ankle joint height above the floor when standing |

## `detect`

2D person detection and pose per camera.

| key | type | default | description |
|---|---|---|---|
| `detect.detector` | `'rtmlib' or 'mock'` | `rtmlib` |  |
| `detect.mock_dir` | `pathlib.Path or None` | `None` | detector = mock: folder with ground-truth <camera>.npz detections |
| `detect.det_model` | `str` | `yolox_tiny` |  |
| `detect.pose_model` | `str` | `rtmpose_x_halpe26` |  |
| `detect.det_threshold` | `float` | `0.3` |  |
| `detect.nms_threshold` | `float` | `0.65` |  |
| `detect.step` | `int` | `1` | run every step-th source frame |
| `detect.min_box_px` | `int` | `24` |  |
| `detect.masks` | `dict[str, list[list[list[float]]]]` | `{}` | camera -> list of polygons [[x, y], ...] in pixels; detections inside are dropped |
| `detect.mask_overlap` | `float` | `0.6` | box fraction inside a mask that drops the detection |
| `detect.static_occupancy` | `float` | `0.6` | suppress cells occupied in > this fraction of frames (posters, mirrors) |
| `detect.static_spread` | `float` | `0.01` | ... whose keypoints barely move (std / box height) |
| `detect.noisy_occupancy` | `float` | `0.45` | also suppress cells occupied in > this fraction of frames by weak detections (0 = off) |
| `detect.noisy_kp` | `float` | `0.55` | ... mean keypoint score below this ... |
| `detect.noisy_box` | `float` | `0.6` | ... and median box score below this |

## `track`

Cross-view association, identities, refinement.

| key | type | default | description |
|---|---|---|---|
| `track.fps` | `float` | `15.0` | track grid rate; source frames are subsampled to the nearest integer stride |
| `track.min_views` | `int` | `2` |  |
| `track.min_score` | `float` | `0.3` |  |
| `track.reproj_px` | `float` | `30.0` |  |
| `track.max_people` | `int` | `0` | 0 = estimate from the data |
| `track.lost_frames` | `int` | `60` |  |
| `track.max_jump_m` | `float` | `0.5` |  |
| `track.room_bounds_m` | `list[float] or None` | `None` | [x0, x1, y0, y1] in the world frame; points outside are dropped |
| `track.appearance` | `bool` | `True` | torso colour and body size in the assignment cost |
| `track.reid` | `'osnet' or 'none'` | `osnet` |  |
| `track.reid_weights` | `pathlib.Path or None` | `None` | OSNet checkpoint; see THIRD_PARTY.md |
| `track.reid_every_s` | `float` | `1.0` |  |
| `track.identity_join` | `float` | `0.25` | centred-embedding cosine to join a segment to an identity |
| `track.identity_merge` | `float` | `0.25` | cosine to merge identities that never co-occur |
| `track.identity_cut` | `float` | `0.2` | cut a segment when its embedding drifts below this cosine |
| `track.min_identity_frac` | `float` | `0.03` | identities carrying fewer frames than this are dropped |
| `track.bridge_s` | `float` | `3.0` | gaps up to this long are interpolated inside an identity |
| `track.max_speed_mps` | `float` | `1.2` |  |
| `track.teleport_mps` | `float` | `3.5` | identity: a short run entered and left faster than this (m/s) is another person's cluster and is dropped; keep above a running child's speed (2-3 m/s), max_speed_mps only widens the association gate |
| `track.max_pelvis_z_m` | `float` | `1.15` | identities floating above this are mirror ghosts |
| `track.merge` | `list[list[str]]` | `[]` | identity ids to force-merge, e.g. [['p1', 'p3']] |
| `track.split` | `dict[str, list[float]]` | `{}` | identity -> times (s) at which to split |
| `track.drop` | `list[str]` | `[]` |  |
| `track.refine` | `bool` | `True` |  |
| `track.refine_smooth` | `float` | `200.0` |  |
| `track.refine_length` | `float` | `2.0` |  |
| `track.euro_min_hz` | `float` | `0.3` |  |
| `track.euro_beta` | `float` | `5.0` |  |

## `select`

Which people to reconstruct and how they are chosen.

| key | type | default | description |
|---|---|---|---|
| `select.mode` | `'auto' or 'interactive' or 'config'` | `auto` |  |
| `select.min_views` | `int` | `2` | auto: a person must be triangulated from >= this many views somewhere |
| `select.min_track_s` | `float` | `2.0` |  |
| `select.min_height_m` | `float or None` | `None` |  |
| `select.max_height_m` | `float or None` | `None` |  |
| `select.children_only` | `bool` | `False` |  |
| `select.child_height_m` | `float` | `1.3` |  |
| `select.max_people` | `int` | `0` |  |
| `select.ids` | `list[str]` | `[]` | config/interactive: identity ids to keep |
| `select.labels` | `dict[str, str]` | `{}` | identity id -> display name |
| `select.picker` | `'cli' or 'opencv' or 'matplotlib' or 'file'` | `cli` |  |
| `select.selection_file` | `pathlib.Path or None` | `None` | headless: text file with one id per line |
| `select.sheet_tiles` | `int` | `8` |  |
| `select.spans_min_views` | `int` | `0` | > 0: keep only spans where the person is seen by >= k views |
| `select.spans_min_s` | `float` | `1.0` |  |
| `select.spans_bridge_s` | `float` | `0.2` |  |

## `single_view`

Recovery of frames seen by one camera only.

| key | type | default | description |
|---|---|---|---|
| `single_view.enabled` | `bool` | `True` |  |
| `single_view.min_torso` | `float` | `0.5` |  |
| `single_view.min_box` | `float` | `0.5` |  |
| `single_view.min_kp` | `float` | `0.5` |  |
| `single_view.edge_px` | `float` | `40.0` |  |
| `single_view.gate_m` | `float` | `0.35` |  |
| `single_view.gate_max_m` | `float` | `1.2` |  |
| `single_view.max_prior_s` | `float` | `6.0` |  |
| `single_view.claim_m` | `float` | `0.35` |  |
| `single_view.multi_m` | `float` | `0.15` |  |
| `single_view.vmax_mps` | `float` | `1.5` |  |
| `single_view.sigma_prior_m` | `float` | `0.1` |  |
| `single_view.smooth_frames` | `float` | `1.5` |  |
| `single_view.room_margin_m` | `float` | `0.1` |  |
| `single_view.colour_margin` | `float` | `0.5` |  |
| `single_view.colour_abs` | `float` | `3.0` |  |
| `single_view.reid_margin` | `float` | `0.02` |  |
| `single_view.score` | `float` | `0.25` |  |

## `sam3d`

SAM 3D Body per view and the multi-view MHR fit.

| key | type | default | description |
|---|---|---|---|
| `sam3d.enabled` | `bool` | `True` |  |
| `sam3d.backend` | `'sam3d' or 'mock'` | `sam3d` |  |
| `sam3d.root` | `pathlib.Path or None` | `None` | SAM 3D Body checkout (or set FROLIC_SAM3D_ROOT) |
| `sam3d.checkpoint` | `str` | `checkpoints/sam-3d-body-dinov3` |  |
| `sam3d.gate_px` | `float` | `80.0` |  |
| `sam3d.iters` | `int` | `600` |  |
| `sam3d.w_prior` | `float` | `2.0` |  |
| `sam3d.w_anchor` | `float` | `0.5` |  |
| `sam3d.gap_vmax` | `float` | `1.5` | m/s: anchor tolerance growth for frames without a tracked mid-hip |
| `sam3d.w_acc` | `float` | `20.0` |  |
| `sam3d.w_floor` | `float` | `200.0` |  |
| `sam3d.sigma` | `float` | `0.04` |  |

## `fit`

Anny body fit (shape, pose, biomechanics, root continuity, single-view weighting).

| key | type | default | description |
|---|---|---|---|
| `fit.backend` | `'anny' or 'mock'` | `anny` |  |
| `fit.iters` | `int` | `800` |  |
| `fit.fps` | `float or None` | `None` | defaults to track.fps |
| `fit.device` | `str` | `cuda` |  |
| `fit.anny_cache` | `pathlib.Path or None` | `None` | Anny asset cache (or set ANNY_CACHE_DIR) |
| `fit.w_ours` | `float` | `0.5` |  |
| `fit.w_sam` | `float` | `1.0` |  |
| `fit.w_surface` | `float` | `0.5` |  |
| `fit.w_hand` | `float` | `0.5` |  |
| `fit.fit_fingers` | `bool` | `True` |  |
| `fit.dense` | `bool` | `True` |  |
| `fit.dense_n` | `int` | `400` |  |
| `fit.w_dense` | `float` | `0.5` |  |
| `fit.dense_max_d` | `float` | `0.06` |  |
| `fit.dense_build` | `bool` | `True` |  |
| `fit.w_build_prior` | `float` | `0.0` | pull of the weight and muscle sliders (logit) toward the population middle when no mass is given; 0 = off. Without it weight absorbs joint-spacing mismatches (adults 0.97, 3 y 0.96, 2026-11-09) |
| `fit.segments` | `bool` | `True` |  |
| `fit.w_segment` | `float` | `0.002` |  |
| `fit.w_length` | `float` | `5.0` |  |
| `fit.segment_gate` | `float` | `0.1` |  |
| `fit.ankle_bias` | `bool` | `True` |  |
| `fit.ankle_bias_max` | `float` | `0.06` | cap (m) on the shin-target ankle bias |
| `fit.w_smooth` | `float` | `2.0` |  |
| `fit.smooth_v0` | `float` | `0.4` |  |
| `fit.smooth_floor` | `float` | `0.25` |  |
| `fit.sam_euro_min_hz` | `float` | `0.5` |  |
| `fit.sam_euro_beta` | `float` | `5.0` |  |
| `fit.fade_s` | `float` | `0.5` |  |
| `fit.sam_jump_m` | `float` | `0.12` |  |
| `fit.sam_body_gate_m` | `float` | `0.25` |  |
| `fit.sam_gate_m` | `float` | `0.15` |  |
| `fit.sv_weight` | `float` | `0.3` |  |
| `fit.sv_bridge_s` | `float` | `0.5` |  |
| `fit.root_continuity` | `bool` | `True` |  |
| `fit.density_kg_m3` | `float` | `1010.0` |  |

#### `fit.biomech`

| `fit.biomech.enabled` | `bool` | `True` |  |
| `fit.biomech.limits` | `bool` | `True` |  |
| `fit.biomech.collision` | `bool` | `True` |  |
| `fit.biomech.foot` | `bool` | `True` |  |
| `fit.biomech.velocity` | `bool` | `True` |  |
| `fit.biomech.arm_collision` | `bool` | `False` |  |
| `fit.biomech.foot_collision` | `bool` | `True` |  |
| `fit.biomech.w_limits` | `float` | `0.5` |  |
| `fit.biomech.w_collision` | `float` | `0.5` |  |
| `fit.biomech.w_foot_heading` | `float` | `0.3` |  |
| `fit.biomech.w_foot_yaw` | `float` | `0.1` |  |
| `fit.biomech.w_floor` | `float` | `0.3` |  |
| `fit.biomech.w_skate` | `float` | `0.5` |  |
| `fit.biomech.w_velocity` | `float` | `1.0` |  |
| `fit.biomech.collision_radius_scale` | `float` | `0.85` |  |
| `fit.biomech.child_age_hi` | `float` | `0.45` | Anny age slider below which paediatric ranges blend in |
| `fit.biomech.shape_pass` | `bool` | `True` | shape from an unconstrained pass, then constrained pose only |
| `fit.biomech.ik_init` | `bool` | `True` |  |
| `fit.write_vertices` | `bool` | `True` |  |

## `export`

Visibility rule and output formats.

| key | type | default | description |
|---|---|---|---|
| `export.max_gap_s` | `float` | `1.0` | holes between observations up to this long stay visible |
| `export.observed_rule` | `bool` | `True` | hide a person only on frames no camera observes them |
| `export.vertices_dtype` | `'float16' or 'float32'` | `float16` |  |
| `export.full_rate_joints` | `bool` | `False` | also write joints interpolated to the source frame rate |

## `render`

Video rendering.

| key | type | default | description |
|---|---|---|---|
| `render.what` | `list['overlay' or 'grid' or 'room' or 'trajectories']` | `['overlay', 'room']` |  |
| `render.fps` | `int` | `30` |  |
| `render.crf` | `int` | `18` |  |
| `render.tile_width` | `int` | `960` |  |
| `render.grid_cols` | `int` | `0` | grid: cameras per row (0 = near-square tiling) |
| `render.grid_tile_width` | `int` | `0` | grid: tile width in px (0 = tile_width) |
| `render.room_width` | `int` | `1920` |  |
| `render.room_height` | `int` | `1080` |  |
| `render.alpha` | `float` | `0.6` |  |
| `render.pan_deg` | `float` | `24.0` |  |
| `render.room_zoom` | `float` | `1.0` | room/trajectories: camera distance divided by this (> 1 = closer) |
| `render.elevation_deg` | `float` | `22.0` |  |
| `render.trail_s` | `float` | `3.0` |  |
| `render.footprints_s` | `float` | `0.0` | room: mark the floor where each foot lands and fade the mark out over this many seconds (0 = off) |
| `render.footprints_hold_s` | `float` | `0.0` | room: keep each footprint at full opacity for this long before it starts fading |
| `render.supersample` | `int` | `1` |  |
| `render.colours` | `dict[str, list[int]]` | `{}` | person id -> RGB |
| `render.start_s` | `float` | `0.0` |  |
| `render.duration_s` | `float` | `0.0` |  |
| `render.backend` | `'egl' or 'osmesa' or 'auto'` | `auto` |  |

## `compute`

Device, threads, per-stage interpreters, Slurm.

| key | type | default | description |
|---|---|---|---|
| `compute.device` | `str` | `cuda` |  |
| `compute.threads` | `int` | `8` |  |
| `compute.stage_python` | `dict[str, Path]` | `{}` | stage -> python interpreter (run a stage in a different environment) |

#### `compute.slurm`

| `compute.slurm.account` | `str or None` | `None` |  |
| `compute.slurm.partition_cpu` | `str` | `cpu` |  |
| `compute.slurm.partition_gpu` | `str` | `gpu` |  |
| `compute.slurm.gres_gpu` | `str` | `gpu:1` |  |
| `compute.slurm.cpus` | `int` | `8` |  |
| `compute.slurm.mem` | `str` | `32G` |  |
| `compute.slurm.time_cpu` | `str` | `04:00:00` |  |
| `compute.slurm.time_gpu` | `str` | `12:00:00` |  |
| `compute.slurm.setup` | `list[str]` | `[]` | shell lines run before each stage (module load, conda activate) |
| `compute.slurm.extra_sbatch` | `list[str]` | `[]` |  |

## `people`

Optional per-person attributes keyed by identity id (`p0`, `p1`, ...).

| key | type | default | description |
|---|---|---|---|
| `people` | `dict[str, PersonConfig]` | `{}` | optional per-person attributes by id |

Each entry of `people` has:

| key | type | default | description |
|---|---|---|---|
| `people.<id>.label` | `str or None` | `None` |  |
| `people.<id>.age_years` | `float or None` | `None` |  |
| `people.<id>.height_m` | `float or None` | `None` |  |
| `people.<id>.sex` | `'male' or 'female' or null` | `None` |  |
| `people.<id>.weight_kg` | `float or None` | `None` |  |

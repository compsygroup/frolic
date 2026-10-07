# Output schema

Everything FROLIC 3D writes is NumPy `.npz`/`.npy`, JSON or CSV; no pickles. World frame: metres, z up, floor at z = 0
after levelling. Time: the track grid (`track.fps`, default 15 Hz); `source_frames` maps every grid frame to the aligned
source frame index (`frame + frame_offset[camera]` is the file frame of that camera, see `work/videos.json`).

## `work/export/<id>.npz` (one per person)

| key | shape / type | meaning |
|---|---|---|
| `vertices` | `(N, V, 3)` float16 (or float32) | body mesh per grid frame, metres; rows are NaN when the person is hidden |
| `faces` | `(F, 3)` int32 | triangle indices (Anny, SMPL-X topology: V = 10475, F = 20908) |
| `joints` | `(N, 17, 3)` float32 | fitted body joints in the FROLIC layout (`joint_names`); NaN when hidden |
| `joint_names` | `(17,)` str | `pelvis, hip_l, hip_r, knee_l, knee_r, ankle_l, ankle_r, toe_l, toe_r, neck, head, shoulder_l, shoulder_r, elbow_l, elbow_r, wrist_l, wrist_r` |
| `frame_times` | `(N,)` float64 | seconds on the track grid, from the first track frame |
| `source_frames` | `(N,)` int32 | aligned source frame of each grid frame |
| `phenotype`, `phenotype_labels` | `(6,)` | Anny sliders in [0, 1]: gender (0 male, 1 female), age, muscle, weight, height, proportions |
| `stature_m` | scalar | rest-pose height of the fitted body |
| `single_view` | `(N,)` bool | frame reconstructed from one camera (single-view lift) or bridged next to one |
| `observed` | `(N,)` bool | frame shown (some camera observes the person; see visibility rule below) |
| `label` | str | display name from `select.labels` / `people.<id>.label`, else the id |

`<id>_joints_full.npz` (when `export.full_rate_joints`): `joints (M, 17, 3)`, `frame_times (M,)`, `fps` at the source
frame rate, linear interpolation inside visible runs, NaN elsewhere.

## Visibility rule (`export.observed_rule`, `export.max_gap_s`)

A person is hidden only on frames where no camera observes them: triangulated frames, single-view frames and merely
observed frames (a confident detection matched to the person but no 3D estimate) all count as observations. Gaps between
observations up to `max_gap_s` are shown (the fit interpolates). Observed frames without any 3D estimate are shown only
when they lie between 3D frames at most 3 s apart. With `observed_rule: false` only triangulated frames count.

## `work/export/meta.json`

```json
{"schema": "frolic3d.export/1", "fps": 15.0, "n_frames": 1350, "source_fps": 60.0,
 "people": [{"id": "p0", "label": "adult", "stature_m": 1.72, "frames_visible": 1340}],
 "body_model": "Anny (SMPL-X topology)", "joint_names": [...], "world": {"up": "z", "floor_z": 0.0, "units": "m"}}
```

## `work/export/cameras.json`

Schema `frolic3d.cameras/1`: `cameras: [{name, width, height, K (3x3), dist (k1 k2 p1 p2 k3), R (3x3), t (3)}]` with
`x_cam = R X + t` (OpenCV conventions, pixels of the distorted image, K of the undistorted image), `units`, `world`.
This is the levelled calibration used by the fit, so meshes project into the raw videos with `cv2.projectPoints`.

## `work/export/metrics.json` and `summary.csv`

`{"schema": "frolic3d.metrics/1", "fps", "per_person": {id: {...}}, "global": {...}}`. Per person:

| key | meaning |
|---|---|
| `coverage_pct`, `frames_visible`, `n_frames` | fraction of grid frames with a mesh |
| `box_iou_median`, `box_iou_ge_0.5_pct`, `n_box_pairs` | projected mesh bounding box vs the best detector box (one-to-one assignment across people per camera and frame), over frames x cameras where the projection lies at least 50 % inside the image |
| `kp2d_err_median` | median 2D error of projected joints vs the assigned detection's keypoints (score > 0.5), as a fraction of the box height |
| `jitter_acc_median_ms2`, `jitter_acc_p90_ms2` | |acceleration| of the mesh centroid |
| `flip_frames`, `flip_times_s`, `vertex_jump_max_cm` | frames where part of the mesh jumps > 10 cm and > 2x the SAM 3D Body surface step (or > 4x the median vertex step when no reference exists) |
| `root_spin_events`, `root_spin_times_s` | root rotation > 150 deg within 1.5 s with a net heading change below 40 % of it |
| `lowest_vertex_median_cm`, `floor_below_5cm_pct`, `stature_p95_m` | floor contact and height sanity |
| `biomech`, `biomech_source` | joint-limit / capsule-penetration frame counts reported by the fit (see docs/biomech_sources.md) |
| `stature_m`, `label` | |

`summary.csv` has one row per person with the headline columns above.

## `work/fit/<id>.npz` (intermediate, kept)

`phenotype`, `phenotype_labels`, `rootrot (N, 3)`, `roottr (N, 3)`, `rotvecs (N, B, 3)` local bone rotations of
`posed_bones`, `joints`, `target` (track joints the fit was pulled to), `have`, `single_view`, `bridged`, `observed`,
`frame0`, `fps`, `stature`, `seg_labels`/`seg_values` (per-person Anny segment-length corrections), `biomech_report`
(JSON string). `<id>_verts.npy` float16 `(N, 10475, 3)`, `faces.npy`.

## `work/videos.json`, detections, tracks

See `docs/CONTRACTS.md` for the intermediate files (`work/detect/<camera>.npz`, `work/track/tracks.npz`, ...).

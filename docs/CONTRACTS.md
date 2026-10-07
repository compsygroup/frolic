# Internal data contracts (read before implementing a stage)

All stages read and write only the canonical files below, through the classes in `frolic3d/io/`.
No pickles. World frame: metres, z up, floor at z = 0 after levelling.

## work/videos.json (written by `sync`)
```
{"cameras": ["cam1", ...],                 # order used everywhere (= CameraSet order)
 "videos": {"cam1": "/abs/path.mp4", ...}, # aligned videos (trimmed copies, or the originals)
 "frame_offset": {"cam1": 0, ...},         # source frame of aligned frame 0 in that camera's file
 "fps": 30.0, "width": {..}, "height": {..},
 "n_frames": 1800,                          # aligned frames available in every camera
 "offsets_s": {"cam1": 0.0, ...}, "sync": {...quality report...}}
```
Aligned frame index `f` of camera `c` is file frame `f + frame_offset[c]`. Every later stage uses aligned indices.

## work/calib/cameras.json (`calibrate`) -> `CameraSet` (`frolic3d/io/calib.py`)
Schema `frolic3d.cameras/1`. `Camera(name, width, height, K, dist, R, t)` with x_cam = R X + t, OpenCV distortion.
`world.levelled` true when the floor is z = 0. `work/track/cameras.json` is the levelled copy written by `track`.

## work/detect/<cam>.npz + .json (`detect`) -> `Detections`
Flat rows: `frame` (aligned index), `box` (x1 y1 x2 y2, distorted pixels), `box_score`, `kps` (26, 2) Halpe-26 distorted
pixels, `kps_score` (26,), `colour` (HSV torso median, NaN if none). `Detections.at(frame)` gives the row indices.
`detect.detector = mock` copies `detect.mock_dir/<cam>.npz` (ground truth) instead of running models.

## Tracks (`frolic3d/io/tracks.py`) -> `Tracks`
`ids` (P), `xyz` (P, N, 17, 3) NaN where missing, `score` (P, N, 17), `support` (P, N, V) detection row per camera or -1,
`fps`, `stride` (aligned source frames per track frame), `frame0`, `single_view` (P, N), `observed` (P, N), `labels`.
Track frame `i` = aligned frame `frame0 + i * stride`. Joint layout: `frolic3d/skeleton.py` (JOINT_NAMES, HALPE26, ...).
- `work/track/tracks_raw.npz`: association output, one id per raw track slot (`t0`, `t1`, ...).
- `work/track/tracks.npz`: identities after clustering/merging, refined, floor-levelled (ids `p0`, `p1`, ...). Also
  `work/track/identities.json` (per id: frames, median stature, median views, segments) for the selection sheet.
- `work/select/tracks.npz`: the selected people, same ids, `labels` filled from config; `work/select/selection.json`.
- `work/singleview/tracks.npz`: selected tracks with lifted frames (`single_view` true, `observed` true) + `svfill.json`.
`WorkDir.fit_input_tracks` returns the single-view file when it exists, else the selected one.

## work/sam3d/ (`sam3d`)
`prompts.npz`: per person boxes/keypoints in the undistorted image per camera and track frame:
`ids`, `cameras`, `box` (P, V, N, 4), `kps` (P, V, N, 26, 2), `ks` (P, V, N, 26), `P3` (P, N, 17, 3), `frame0`, `stride`.
`views/<cam>.npz`: per-view SAM 3D Body outputs for every (person, frame) with a box: `pid` (M,), `frame` (M,) and the
model arrays stacked along M (`pred_keypoints_3d` (M, 70, 3), `body_pose_params`, `hand_pose_params`, `shape_params`,
`scale_params`, `pred_keypoints_2d`, ...). `mhr/<pid>.npz`: multi-view fit (`kp` (N, 70, 3) world, `have` (N,), params,
`stature`). `mhr/<pid>_verts.npy`: posed MHR vertices (N, V, 3) float16 for the dense surface term.
`sam3d.backend = mock` builds `kp` straight from the tracks (joints mapped through MHR70_BODY) for tests.

## work/fit/ (`fit`)
`<pid>.npz`: `phenotype` (6), `phenotype_labels`, `rootrot` (N, 3), `roottr` (N, 3), `rotvecs` (N, B, 3), `posed_bones`,
`joints` (N, 17, 3), `target` (N, 17, 3), `have`, `single_view`, `bridged`, `observed`, `frame0`, `fps`, `stature`,
`seg_labels`, `seg_values`, `biomech_report` (json string). `<pid>_verts.npy` float16 (N, 10475, 3). `faces.npy`.
`fit.backend = mock` writes the same schema from a stick-figure body (tests without Anny).

## work/export/ (`export`): see docs/output_schema.md
`<pid>.npz`: `vertices` float16 (N, V, 3) NaN rows when hidden, `faces` int32, `joints` float32 (N, 17, 3),
`joint_names`, `frame_times` float64, `source_frames` int32 (aligned), `phenotype`, `phenotype_labels`, `stature_m`,
`single_view`, `observed`, `label`. `meta.json`, `cameras.json`, `metrics.json`, `summary.csv`.

## Stage functions
Each `frolic3d/<stage>/stage.py` exposes `run(cfg: FrolicConfig, work: WorkDir) -> dict` (info for the done marker).
Heavy models live behind adapters with a `mock` backend; stages must run end to end on the synthetic scene with mocks.
Logging through `frolic3d.log.get_logger(__name__)`, never print.

## Stage order
`sync -> detect -> calibrate -> track -> select -> singleview -> sam3d -> fit -> export -> render`.
`detect` runs before `calibrate` because the calibration-free bootstrap is built from the 2D keypoints; the board and
given modes ignore the detections. `calibrate` in `bootstrap` mode may use a ground-truth-free subset of frames where
one person is visible in every camera.

# Architecture

FROLIC 3D is a staged pipeline. Every stage is a function `run(cfg, work) -> dict` in `frolic3d/<stage>/stage.py`,
reads only canonical files in the work directory and writes a done marker when it finishes
(`docs/CONTRACTS.md` lists every file). The command line (`frolic3d/cli.py`) is a thin wrapper around
`frolic3d/pipeline.py`, which owns the stage order, the markers, `--force` / `--from`, and the per-stage interpreter
hand-off.

```
videos ──sync──> videos.json ──detect──> detect/<cam>.npz
                                  │
       calibrate (board | bootstrap | given) ──> calib/cameras.json
                                  │
                                track ──> track/tracks.npz (+ levelled cameras)
                                  │
                               select ──> select/tracks.npz (ids, labels)
                                  │
                            singleview ──> singleview/tracks.npz
                                  │
                                sam3d ──> sam3d/mhr/<id>.npz, <id>_verts.npy
                                  │
                                  fit ──> fit/<id>.npz, <id>_verts.npy, faces.npy
                                  │
                               export ──> export/<id>.npz, meta.json, metrics.json, summary.csv
                                  │
                               render ──> render/*.mp4
```

## Packages

| package | what it does |
|---|---|
| `config.py` | one pydantic model for the whole YAML; `frolic init` writes the commented template |
| `skeleton.py` | the 17-joint layout and its mappings to Halpe-26, SAM 3D Body (MHR 70) and Anny bones |
| `geometry.py` | pinhole camera, undistortion, DLT triangulation, rays, Kabsch / Umeyama, plane fitting and levelling |
| `io/` | `cameras.json`, per-camera `Detections`, `Tracks`, video probing / reading / writing, `WorkDir` (markers, canonical paths) |
| `sync/` | onset-envelope cross-correlation + GCC-PHAT fine lag, triangle-closure check, trimming to the common overlap |
| `calib/` | `board.py` (ChArUco: intrinsics with pose-diverse frame selection and sanity checks, IPPE board poses, chained extrinsics, multi-frame bundle adjustment, metric scale from the square length), `bootstrap.py` (person-based calibration-free SfM + bundle adjustment, metric scale from a known stature), `floor.py` (z shift from feet, plane levelling), converters |
| `detect/` | detector / pose adapter (`rtmlib` YOLOX + RTMPose halpe26; `mock` for tests), static masks, polygon masks, torso colour |
| `track/` | `associate.py` (per-frame greedy RANSAC clustering across views on pelvis + neck, Hungarian linking over time with appearance and body size), `identity.py` (segment-level identity assembly with cannot-link constraints and re-ID embeddings, merge / split / drop overrides), `refine.py` (joint multi-view refinement: robust reprojection + smoothness + limb lengths, consensus weighting, One-Euro), `reid.py` (OSNet adapter) |
| `select/` | auto filters (views, duration, height, children only), contact sheet (JPG + HTML), pickers (CLI, OpenCV, matplotlib, text file), visibility spans |
| `singleview/` | the single-view lift: candidate detections for frames without 3D, depth fused from track prior / pelvis height / limb lengths / feet on floor / a second camera, joint depths from SAM 3D Body or the nearest 3D pose, smoothing and rejection |
| `sam3d/` | SAM 3D Body adapter (`mock` for tests), prompts from the tracks, per-view inference, multi-view MHR fit |
| `fit/` | `anny_model.py` (Anny adapter: skeleton function, landmark and dense surface sampling), `shape.py` (segment-length targets, ankle bias, phenotype from age / sex / height / mass), `root.py` (continuous quaternion root init, delta parameterisation, geodesic acceleration), `biomech.py` (joint limits, capsule self-collision, foot rules, angular velocity), `filters.py` (One-Euro, soft fades, jump gates), `fitter.py` (stage 1 shape + pose from the tracks, stage 2 pose from SAM 3D Body with gates, single-view weighting and bridging) |
| `export/` | visibility rule, per-person npz, meta, full-rate joints |
| `metrics/` | coverage, detector reprojection (box IoU, 2D keypoint error), jitter, flips, root spins, floor, biomechanics |
| `render/` | backend selection (EGL / OSMesa), overlay with the lens model applied to the render, grid, room view, trajectories |
| `slurm.py` | one sbatch job per stage with `afterok` dependencies |
| `synthetic.py` | stick-figure scenes with known cameras, used by the tests and the example |

## Adapters and mocks

Heavy third-party models sit behind small adapters with a documented interface and a `mock` backend:

| adapter | real backend | mock |
|---|---|---|
| `detect.adapter.PoseDetector` | rtmlib YOLOX + RTMPose (ONNX) | copies ground-truth detections from `detect.mock_dir` |
| `track.reid` | torchreid OSNet | colour-only identity features (`track.reid: none`) |
| `sam3d.adapter` | SAM 3D Body (external checkout + checkpoint) | MHR keypoints taken from the tracks |
| `fit.anny_model` | Anny (`anny` package + asset cache) | stick-figure body with the same output schema |
| `render.backend` | pyrender on EGL / OSMesa | tests skip when no backend exists |

Model weights are never vendored (see THIRD_PARTY.md and README "Install").

## What was generalised

| in the research code | in FROLIC 3D |
|---|---|
| six fixed cameras named by their room position, 1080p60 | any number of cameras (>= 2), any resolution and frame rate; cameras are named after the video files, the track grid rate is a config value |
| people were fixed household roles decided by room-zone heuristics and manual overrides | people are stable identities `p0, p1, ...` assembled from appearance (re-ID) and geometry; optional labels and attributes (age, height, sex, weight) in the config; merge / split / drop overrides by id |
| person selection done by hand from contact sheets | `select` stage: auto filters, interactive picker (CLI / OpenCV / matplotlib), headless sheet + text file, or ids in the config |
| calibration from a site-specific board script or an external toolbox | three modes in one stage: ChArUco board videos, person-based calibration-free bootstrap with metric scale from a known stature, or a given file (own schema, calibrator-GUI JSON, anipose TOML, OpenCV YAML) |
| mirror masked by painting frames | polygon masks per camera in the config, applied to detections |
| three conda environments and shell drivers with hard-coded paths | one package; stages can still run in different interpreters via `compute.stage_python` |
| a ladder of fitter versions | one fitter with the final behaviour, every term a config switch |
| videos synced by hand or by a notebook | `sync` stage (audio onset envelope + GCC-PHAT) with a quality report and trimming |
| GPU locks and `taskset` wrappers | `--device`, `--threads`, Slurm submission with dependencies |

## Frames and time

Every stage after `sync` uses the aligned frame index. The track grid subsamples it by an integer stride
(`stride = round(source_fps / track.fps)`), so grid frame `i` is aligned frame `frame0 + i * stride`. The export keeps
both (`frame_times`, `source_frames`) and the overlay renderer reads the raw frames through `videos.json`'s per-camera
`frame_offset`.

# Third-party software and models

FROLIC does not vendor any model weights. Every model is loaded through an adapter from a location the user installs
(README, "Install"). The table lists what the pipeline uses, under which licence, and whether that licence restricts
redistribution or commercial use. Flags are our reading of the licence texts as of the date of writing; they are not
legal advice. Check the linked licences before redistributing or deploying commercially.

| component | used for | licence | redistribution / commercial use |
|---|---|---|---|
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (Meta) + checkpoint | per-view 3D body estimate, MHR body model | SAM License (Meta) | **Flag.** Redistribution of the materials or derivative works only under the same agreement with a copy of it; acceptable-use terms apply; the checkpoint must be obtained from Meta by every user. Not an OSI licence. |
| [Anny](https://github.com/naver/anny) (NAVER) | parametric body model (all ages, SMPL-X topology) | Apache 2.0 (code and the model files in the `anny` package, per the package LICENSE) | No restriction beyond attribution and the Apache notice. Verify the asset cache files you download carry the same licence. |
| [rtmlib](https://github.com/Tau-J/rtmlib) | YOLOX + RTMPose ONNX inference | Apache 2.0 | No restriction. |
| [RTMPose / MMPose](https://github.com/open-mmlab/mmpose) (`rtmpose-x body7 halpe26` ONNX weights) | 2D pose | Apache 2.0 | No restriction on the code and weights; note that the body7 training mix includes datasets with research-only terms (e.g. Human-Art, COCO-WholeBody derivatives). **Flag for commercial deployment: check the dataset terms.** |
| [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (`yolox_tiny humanart` ONNX weights) | person detection | Apache 2.0 | No restriction on the code; the Human-Art fine-tune data is research-only. **Flag as above.** |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | ONNX inference | MIT | No restriction. |
| [torchreid](https://github.com/KaiyangZhou/deep-person-reid) (OSNet) | appearance re-identification | MIT (code) | Code: no restriction. **Flag:** the published OSNet weights are trained on MSMT17 / Market-1501, whose terms are research-only; commercial use needs weights trained on suitably licensed data. |
| [aniposelib](https://github.com/lambdaloop/aniposelib) | multi-view triangulation and the joint track refinement (`optim_points`) | BSD-2-Clause | No restriction. Optional extra (`track`); FROLIC falls back to its own DLT when absent. |
| [roma](https://github.com/naver/roma) (NAVER) | rotation utilities in torch | BSD-3-Clause | No restriction. |
| [PyTorch](https://pytorch.org) | optimisation | BSD-3-Clause | No restriction. |
| [OpenCV](https://opencv.org) (`opencv-python`) | video, undistortion, ChArUco | Apache 2.0 | No restriction (the `contrib` ArUco module is included in OpenCV >= 4.7 main). |
| [NumPy](https://numpy.org), [SciPy](https://scipy.org) | numerics | BSD-3-Clause | No restriction. |
| [pyrender](https://github.com/mmatl/pyrender), [trimesh](https://github.com/mikedh/trimesh) | rendering | MIT | No restriction. |
| [PyOpenGL](https://pyopengl.sourceforge.net) | OpenGL bindings | BSD-style | No restriction. |
| [Pillow](https://python-pillow.org), [matplotlib](https://matplotlib.org) (DejaVu fonts) | labels, legends, pickers | HPND (Pillow), PSF-based (matplotlib), Bitstream Vera / DejaVu (fonts) | No restriction. |
| [pydantic](https://docs.pydantic.dev), [typer](https://typer.tiangolo.com), [PyYAML](https://pyyaml.org), [soundfile](https://github.com/bastibe/python-soundfile) | config, CLI, audio I/O | MIT / MIT / MIT / BSD-3 | No restriction. |
| [FFmpeg](https://ffmpeg.org) (binary, called as a subprocess) | decoding, trimming, encoding | LGPL 2.1+ / GPL depending on the build | Not linked, called as an external program; the user's build governs. |

## Not dependencies, but related

| component | relation | licence | note |
|---|---|---|---|
| [Kineo](https://github.com/liris-xr/kineo) (Ecole Centrale de Lyon et al.) | the research pipeline used Kineo's calibration-free bootstrap; FROLIC reimplements a person-based bootstrap (`frolic/calib/bootstrap.py`) and does not import Kineo | non-commercial research licence | **Flag:** do not add Kineo as a dependency or copy code from it without accepting its non-commercial terms for the whole distribution. |
| [ChArUco calibrator GUI](https://github.com/nairg1/charuco-calibrator-gui) | FROLIC's board calibration reads and writes its intrinsics / multi-camera JSON formats and uses the same board model (8 x 6, legacy pattern, DICT_4X4_100 by default); `frolic/calib/board.py` is an independent implementation whose design follows that tool's OpenCV >= 4.7 object-API port, intrinsics plausibility safeguards, constrained re-solve for degenerate views and multi-frame extrinsics (credited here); FROLIC adds video input, pose-diverse frame sampling and a multi-camera bundle adjustment | PolyForm Noncommercial 1.0.0 | **Flag:** copying code from that repository into FROLIC would make the combined work noncommercial unless its author relicenses it. The formats themselves are not copyrightable interfaces; crediting the tool is a courtesy. |

## Choosing a licence for FROLIC (recommendation only, not decided)

`LICENSE` is a placeholder. Points to weigh:

- The FROLIC code itself depends only on permissively licensed libraries (Apache 2.0, BSD, MIT), so a permissive
  licence such as **Apache-2.0** (patent grant, explicit contribution terms) or **BSD-3-Clause** (shortest) is
  possible for the code.
- A permissive licence on the code does not make a deployment commercial-ready: the SAM 3D Body checkpoint, the
  Human-Art-tuned detector weights and the MSMT17-trained OSNet weights carry their own restrictions. Document them
  (this file) and keep the adapters so that users can swap models.
- A noncommercial licence (PolyForm Noncommercial, CC BY-NC) would match the status of the research models and the
  calibrator GUI, at the cost of excluding commercial contributors and of GitHub / packaging friction (not OSI).
- Whatever the choice, keep the citation block in the README and the NOTICE-style attributions in this file.

"""Write the synthetic example scene (videos, cameras, ground-truth detections) next to this file:

    python examples/synthetic/make_scene.py [--frames 90] [--cams 4]

Then `frolic run -c examples/synthetic/frolic.yaml` runs every stage with mocked models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from frolic3d.synthetic import make_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--cams", type=int, default=4)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("data"))
    a = ap.parse_args()
    info = make_scene(a.out, n_cams=a.cams, n_frames=a.frames, fps=a.fps, videos=True)
    print(f"wrote {a.out}: {info['n_cams']} cameras, {info['n_frames']} frames, {len(info['people'])} people")


if __name__ == "__main__":
    main()

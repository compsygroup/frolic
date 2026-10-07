"""Generate docs/config.md from the pydantic configuration model: python docs/gen_config.py"""

from __future__ import annotations

from pathlib import Path
from typing import get_args, get_origin

from pydantic import BaseModel

from frolic3d.config import FrolicConfig

SECTION_NOTES = {
    "input": "Where the videos are and where the work goes.",
    "sync": "Audio-based time alignment (`frolic sync`); ignored when `input.synced` is true.",
    "calib": "Camera calibration mode and floor levelling.",
    "detect": "2D person detection and pose per camera.",
    "track": "Cross-view association, identities, refinement.",
    "select": "Which people to reconstruct and how they are chosen.",
    "single_view": "Recovery of frames seen by one camera only.",
    "sam3d": "SAM 3D Body per view and the multi-view MHR fit.",
    "fit": "Anny body fit (shape, pose, biomechanics, root continuity, single-view weighting).",
    "export": "Visibility rule and output formats.",
    "render": "Video rendering.",
    "compute": "Device, threads, per-stage interpreters, Slurm.",
    "people": "Optional per-person attributes keyed by identity id (`p0`, `p1`, ...).",
}


def type_name(ann) -> str:
    origin = get_origin(ann)
    if origin is None:
        return getattr(ann, "__name__", str(ann)).replace("NoneType", "null")
    args = get_args(ann)
    if str(origin) == "typing.Literal":
        return " | ".join(repr(a) for a in args)
    if origin in (list, dict, tuple):
        return f"{origin.__name__}[{', '.join(type_name(a) for a in args)}]"
    if str(origin) in ("types.UnionType", "typing.Union"):
        return " | ".join(type_name(a) for a in args)
    return str(ann)


def rows(model: type[BaseModel], prefix: str) -> list[str]:
    out = []
    for name, f in model.model_fields.items():
        ann = f.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out.append(f"\n#### `{prefix}{name}`\n")
            out += rows(ann, f"{prefix}{name}.")
            continue
        default = f.get_default(call_default_factory=True)
        if isinstance(default, BaseModel):
            default = default.model_dump(mode="json")
        d = str(default).replace("|", "\\|")
        out.append(f"| `{prefix}{name}` | `{type_name(ann).replace('|', 'or')}` | `{d}` | {f.description or ''} |")
    return out


def main() -> None:
    lines = [
        "# Configuration reference",
        "",
        "Generated from `frolic3d/config.py` by `docs/gen_config.py`.",
        "`frolic init` writes a commented template with these defaults.",
        "",
        "Paths are resolved relative to the configuration file. Unknown keys are rejected.",
        "",
    ]
    for name, f in FrolicConfig.model_fields.items():
        ann = f.annotation
        lines += [f"## `{name}`", "", SECTION_NOTES.get(name, ""), ""]
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            lines += ["| key | type | default | description |", "|---|---|---|---|"]
            lines += rows(ann, f"{name}.")
        else:
            lines += [
                "| key | type | default | description |",
                "|---|---|---|---|",
                f"| `{name}` | `{type_name(ann).replace('|', 'or')}` "
                f"| `{f.get_default(call_default_factory=True)}` | {f.description or ''} |",
            ]
            sub = get_args(ann)[-1] if get_args(ann) else None
            if isinstance(sub, type) and issubclass(sub, BaseModel):
                lines += [
                    "",
                    f"Each entry of `{name}` has:",
                    "",
                    "| key | type | default | description |",
                    "|---|---|---|---|",
                ] + rows(sub, f"{name}.<id>.")
        lines.append("")
    Path(__file__).with_name("config.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

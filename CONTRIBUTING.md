# Contributing

## Setup

```bash
conda env create -f environment.yml && conda activate frolic3d   # or: pip install -e ".[dev]"
pre-commit install
```

## Tests and lint

```bash
make test        # pytest, GPU tests skipped unless FROLIC_TEST_CLIP is set
make cov         # coverage report
make lint        # ruff check + ruff format --check
make format      # apply ruff
```

The test suite runs on CPU with mocked models (synthetic scenes from `frolic3d/synthetic.py`). Rendering tests skip
when no offscreen OpenGL backend is available. Tests that need the real models are marked `gpu` and opt in through
environment variables (`FROLIC_TEST_CLIP`, `FROLIC_TEST_BOARD_DIR`); never add data or videos to the repository.

## Style

- Python >= 3.10, type hints everywhere, small functions, `logging` (never `print`).
- Comments only where the reason is not obvious from the code; no narrated steps, no emoji.
- No clinical data, subject identifiers, site hostnames or absolute local paths anywhere in the repository.
- Every stage reads and writes only the files in `docs/CONTRACTS.md`; new intermediate files go there first.
- Heavy models stay behind adapters with a `mock` backend so the pipeline keeps running end to end in the tests.
- Configuration options go into `frolic3d/config.py` with a description; regenerate `docs/config.md` with
  `python docs/gen_config.py`.

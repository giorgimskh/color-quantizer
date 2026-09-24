# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
color-quantizer: a Python CLI that reduces an image to k colors with k-means implemented from scratch in NumPy. Learning + portfolio project. Full design and milestones: `docs/design.md`.

## Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                    # all tests
pytest tests/test_image_io.py::test_name  # single test
quantize <input> -k <int> -o <output> [--seed N] [--max-iter N] [--palette]
```

## Architecture (`src/color_quantizer/`)
`image_io` (file <-> (H, W, 3) uint8 <-> (N, 3)) -> `kmeans` (pure NumPy engine) -> `quantizer` (sample ~50k pixels, fit, map all pixels once, rebuild image) -> `cli` (argparse; `quantize` entry point in pyproject).

K-means rules: squared Euclidean distance (no sqrt), fully vectorized (no per-pixel Python loops), k-means++ init, stop on no assignment change / centroid shift < tol / max-iter, re-seed an empty cluster at the point farthest from its nearest centroid, all randomness via `numpy.random.default_rng(seed)`.

## Constraints
- Dependencies: numpy, Pillow (image I/O only); dev: pytest. NO scikit-learn, scipy.cluster, or other ML/clustering libraries.
- CLI only: no web API, server, or GUI.
- Type hints and docstrings on all public functions.

## Workflow rules
- Complete one milestone at a time; run tests before calling it done.
- All five milestones are complete. The k-means algorithm (`kmeans.py`) was originally reserved for the user to write; it was implemented by Claude at the user's explicit request. Change it only when asked.
- Commit after each milestone with a clear message.

# color-quantizer — Design

A Python command-line tool that reduces an image to k colors using k-means clustering implemented from scratch. Learning + portfolio project, built in small, reviewable steps.

## Design
- Python 3.11+, src layout, configured with pyproject.toml:
  ```
  src/color_quantizer/
    image_io.py   : load/save images, convert to/from (H, W, 3) and (N, 3) arrays
    kmeans.py     : k-means engine (pure NumPy)
    quantizer.py  : full pipeline (sample -> fit -> map all pixels -> rebuild image)
    cli.py        : command-line interface (argparse)
  tests/          : pytest
  ```
- Allowed dependencies: numpy, Pillow (image I/O only). Dev: pytest.
  NO scikit-learn, scipy.cluster, or other ML/clustering libraries.
- This is a CLI-only project. No web API, server, or GUI.
- CLI usage: `quantize <input> [-k <int>] -o <output> [--seed <int>] [--max-iter <int>] [--palette]`
  (exposed via a `[project.scripts]` entry point; if `-k` is omitted, the user is prompted for it)

### K-means decisions
- pixels are points in RGB space; distance = squared Euclidean (no sqrt)
- distances computed with vectorized NumPy operations, no per-pixel Python loops
- k-means++ initialization
- stop when no assignments change, centroid shift < tolerance, or max-iter reached
- empty cluster: re-seed at the point farthest from its nearest centroid
- train on a random sample of pixels (default 50,000), then map all pixels once
- use `numpy.random.default_rng(seed)` so runs are reproducible

## Milestones
1. Project skeleton + image load/save utilities
2. K-means with random init + unit tests on tiny known datasets
3. k-means++ init + empty-cluster handling
4. Pixel sampling + full quantization pipeline
5. CLI argument parsing, --palette output, error handling

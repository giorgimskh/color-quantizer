# color-quantizer

Reduce an image to *k* colors using k-means clustering implemented from scratch in NumPy (no ML libraries).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage (work in progress)
```bash
quantize input.jpg -k 8 -o output.png [--seed 0] [--max-iter 100] [--palette]
```

See [docs/design.md](docs/design.md) for the design and milestones.

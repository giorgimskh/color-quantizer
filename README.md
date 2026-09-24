# color-quantizer

Reduce an image to *k* colors using k-means clustering implemented from scratch in NumPy — no scikit-learn or other ML libraries.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage
```bash
quantize input.jpg [-k 8] -o output.png [--seed 0] [--max-iter 100] [--palette]
```

| Option | Meaning |
| --- | --- |
| `-k`, `--colors` | number of colors (≥ 1); if omitted, you are asked for it after start |
| `-o`, `--output` | output image path; format from the extension (required) |
| `--seed` | random seed for reproducible results |
| `--max-iter` | maximum k-means iterations (default 100) |
| `--no-compare` | skip saving `<output>_compare.png` (original on the left, reconstructed on the right; saved by default) |
| `--show` | open the side-by-side comparison in your image viewer |
| `--palette` | also save `<output>_palette.png` (swatch strip, most-used color first) and print hex codes |

Example:
```
$ quantize photo.jpg -k 4 -o photo_4.png --palette --seed 0
Saved 4-color image to photo_4.png
Saved comparison (original | reconstructed) to photo_4_compare.png
Saved palette to photo_4_palette.png
#243e2e
#25c063
...
```

## How it works
1. Load the image as an (H, W, 3) array and flatten it to (N, 3) RGB points.
2. Train k-means on a random sample of up to 50,000 pixels:
   k-means++ initialization, vectorized squared-Euclidean distances, Lloyd
   iterations until assignments stop changing, centroids move less than a
   tolerance, or `--max-iter` is reached. Empty clusters are re-seeded at the
   point farthest from its nearest centroid.
3. Map every pixel to its nearest centroid once and set it to that centroid's
   color rounded to integers, rebuilding the image. A side-by-side comparison
   with the original is saved next to it.

A 6-megapixel image takes about 1–2 seconds.

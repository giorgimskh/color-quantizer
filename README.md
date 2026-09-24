# color-quantizer

Reduce an image to *k* colors using k-means clustering implemented from scratch in NumPy — no scikit-learn or other ML libraries.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage

### Guided (interactive)
Run `quantize` with no arguments and answer the questions:
```
$ quantize
Image path: ~/Pictures/cherry.png
Loaded cherry.png: 1920x1080 pixels, 93,795 distinct colors
How many colors (k)? 8
Output file [cherry_k8.png]:
Save palette image too? [y/N] y
Open result in image viewer? [Y/n]
Training k-means (k=8) on 50,000 of 2,073,600 pixels...
  converged after 41 iterations
Mapping all 2,073,600 pixels to their nearest color...
Saved 8-color image to cherry_k8.png
Saved comparison (original | reconstructed) to cherry_k8_compare.png
Saved palette to cherry_k8_palette.png
Done in 0.3s
  Colors:      93,795 -> 8
  Color error: 5.4 (RMS, 0-255 scale)
  Palette (most used first):
    ████ #010204   87.8%
    ████ #bba0bc    2.0%
    ...

Try another k? (Enter to quit) 24
...
Try another k? (Enter to quit)
Bye!
```
Tips: you can drag an image into the terminal to paste its path; press Enter
to accept the default shown in `[brackets]`; Ctrl+C quits.

### With arguments
Anything you pass is not asked again; with input, `-k` and `-o` all given,
it runs once without questions (good for scripts):
```bash
quantize input.jpg -k 8 -o output.png [--seed 0] [--palette] [--show]
```

| Option | Meaning |
| --- | --- |
| `input` | image to quantize (asked if omitted) |
| `-k`, `--colors` | number of colors, ≥ 1 (asked if omitted) |
| `-o`, `--output` | output image path; format from the extension (asked if omitted, default `<input>_k<k>.png`) |
| `--seed` | random seed for reproducible results |
| `--max-iter` | maximum k-means iterations (default 100) |
| `--palette` | also save `<output>_palette.png` (swatch strip, most-used color first) |
| `--no-compare` | skip saving `<output>_compare.png` (original left, reconstructed right; saved by default) |
| `--show` | open the side-by-side comparison in your image viewer |
| `-i`, `--interactive` | after each result, offer to try another k |
| `-q`, `--quiet` | print only saved files and errors (no progress or stats) |

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

# color-quantizer

A command-line tool that reduces an image to **k colors** using **k-means clustering implemented from scratch** in NumPy. It uses no scikit-learn, SciPy clustering or other ML libraries; Pillow is used only to read and write image files.

## Example

| Original (96,085 colors) | k = 4 | k = 8 |
| :---: | :---: | :---: |
| ![Original](assets/autumn_original.jpg) | ![4 colors](assets/autumn_4colors.png) | ![8 colors](assets/autumn_8colors.png) |
| | ![4-color palette](assets/autumn_4colors_palette.png) | ![8-color palette](assets/autumn_8colors_palette.png) |

With **k = 4** the scene is reduced to four browns, from dark trunks to light sky, giving a flat poster look.
With **k = 8** there are more steps of shading, so the stairs, the stone path and the bushes regain their depth.
Each color is the *average* of a cluster, so small bright areas (the red leaves) blend into the surrounding warm tones. A larger k keeps more of them.

<sub>Generated with `quantize assets/autumn_original.jpg -k 4 -o assets/autumn_4colors.png --seed 0 --palette --no-compare` (and `-k 8`). Wallpaper from wallhaven.cc.</sub>

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/giorgimskh/color-quantizer.git
cd color-quantizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

In each new terminal, activate the environment again (`source .venv/bin/activate`) before using `quantize`.

## Usage

### Interactive

Run `quantize` with no arguments. It explains what it does, then asks for everything it needs:

```text
$ quantize
+-------------------------------------------------------+
|  color-quantizer 0.1.0 - reduce an image to k colors  |
+-------------------------------------------------------+

What it does:
  Groups similar colors of your image into k clusters with k-means
  and repaints every pixel with its cluster's average color.
  ...

Image path: ~/Pictures/autumn.png
Loaded autumn.png: 1920x1080 pixels, 197,414 distinct colors
How many colors (k)? 8

Options (yes/no options switch when picked; the others ask for a value):
  1. Random seed             (now: random)
  2. Max k-means iterations  (now: 100)
  3. Save palette image      (now: no)
  4. Save comparison image   (now: yes)
  5. Open result in viewer   (now: yes)
Enter numbers separated by spaces (e.g. 1 3), or press Enter to keep: 1 3
Random seed (whole number >= 0, Enter = random): 42
Options now:
  - Random seed             (now: 42)
  - Max k-means iterations  (now: 100)
  - Save palette image      (now: yes)
  - Save comparison image   (now: yes)
  - Open result in viewer   (now: yes)
Training k-means (k=8) on 50,000 of 2,073,600 pixels...
  converged after 22 iterations
Mapping all 2,073,600 pixels to their nearest color...
Saved 8-color image to output/autumn_k8.png
Saved comparison (original | reconstructed) to output/autumn_k8_compare.png
Saved palette to output/autumn_k8_palette.png
Done in 0.3s
  Colors:      197,414 -> 8
  Color error: 13.5 (RMS, 0-255 scale)
  Palette (most used first):
    #783b26   20.8%
    #73523e   15.3%
    ...

What next?
  1. Try a different number of colors (k) on this image
  2. Choose another image
  3. Change options (seed, iterations, palette, comparison, viewer)
  4. Quit
Choose 1-4: 1
How many colors (k)? [8] 16
...
Choose 1-4: 4

Result files from this session:
  1. output/autumn_k8.png
  2. output/autumn_k8_compare.png
  3. output/autumn_k8_palette.png
  4. output/autumn_k16.png
  5. output/autumn_k16_compare.png
  6. output/autumn_k16_palette.png
Numbers of files to KEEP (e.g. 1 3), a = keep all, Enter = delete all: 1 5
Kept 2 file(s) in saved/:
  saved/autumn_k8.png
  saved/autumn_k16_compare.png
Deleted 4 result file(s) from output/ - next start is fresh.
Bye!
```

**Options list** (shown right after you enter k): pick any number of options at once, e.g. `1 3`.
The yes/no options (3-5) switch on or off when picked. Seed and iterations ask for a value.
Press Enter to keep everything as it is.

**What next? menu** (shown after each result):

| Choose | To |
| --- | --- |
| `1` | run again on the same image with a different k (Enter keeps the current k) |
| `2` | choose another image |
| `3` | change options, then return to the menu |
| `4` | quit, choosing which result files to keep |

Tips: drag an image into the terminal to paste its path, press Enter to accept a `[default]`, and use Ctrl+C to quit at any time.

### One command

Give the image, `-k` and `-o` and it runs once with no questions, which is useful in scripts:

```bash
quantize photo.jpg -k 8 -o photo_8.png
quantize photo.jpg -k 4 -o poster.png --palette --show --seed 42
```

| Option | Meaning |
| --- | --- |
| `input` | image to quantize (asked if omitted) |
| `-k`, `--colors` | number of colors, ≥ 1 (asked if omitted) |
| `-o`, `--output` | output image path to keep; format from the extension (without it, results are temporary) |
| `--seed` | random seed (≥ 0); the same seed gives the same result |
| `--max-iter` | maximum k-means iterations (default 100) |
| `--palette` | also save the colors as a swatch strip |
| `--no-compare` | don't save the side-by-side comparison |
| `--show` | open the comparison in your image viewer |
| `-i`, `--interactive` | show the options list and the What next? menu even when all arguments are given |
| `-q`, `--quiet` | print only saved files and errors |

### Output files

| File | Content |
| --- | --- |
| `<name>_k<k>.png` | the image with k colors |
| `<name>_k<k>_compare.png` | original (left) and result (right) side by side |
| `<name>_k<k>_palette.png` | the k colors, most used first (option 3 / `--palette`) |

**Interactive results are temporary.** While the program runs they are saved in `output/`
and opened in your image viewer. When you quit (menu 4 or Ctrl+C at a question), the
program lists every result file with a number and asks which to keep:

- type the numbers of the files to keep, e.g. `1 5`. They are moved to `saved/`
  (an existing name gets `_1`, `_2`, ... so nothing is overwritten)
- `a` keeps all, and Enter deletes all

Everything not kept is deleted and `output/` is removed, so each start is fresh. If the
program stops because of an error or Ctrl+C during processing, all results are deleted
without asking. Other files in `output/` are never touched.

You can also keep a result by giving an output path on the command line. Files at that path are never deleted:

```bash
quantize photo.jpg -k 8 -o keep.png
```

## How it works

Every pixel is a point in 3-D RGB space; similar colors are close together.

1. **Sample.** Pick a random sample of up to 50,000 pixels to train on (fast even for large photos).
2. **Cluster (k-means).**
   - Choose k starting centers with **k-means++**, which spreads them out.
   - Repeat until nothing changes, the centers barely move, or `--max-iter` is reached:
     - assign each pixel to the nearest center (squared Euclidean distance, computed for all pixels at once with NumPy)
     - move each center to the mean of its pixels
   - If a cluster ends up empty, restart it at the pixel farthest from its nearest center.
3. **Reconstruct.** Assign *every* pixel of the full image to its nearest center once and paint it with that center's color, rounded to whole numbers.

A 1920×1080 image takes well under a second.

## Project structure

```text
src/color_quantizer/
├── image_io.py     load/save images; image <-> pixel-array conversions
├── kmeans.py       k-means engine: k-means++, Lloyd iterations, empty clusters
├── quantizer.py    pipeline: sample -> fit -> map all pixels -> rebuild image
├── interactive.py  prompts for values not given on the command line
└── cli.py          the `quantize` command (arguments, output files, stats)
tests/              pytest tests for each module
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

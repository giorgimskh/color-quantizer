# color-quantizer

A command-line tool that reduces an image to **k colors** using **k-means clustering implemented from scratch** in NumPy. It uses no scikit-learn, SciPy clustering or other ML libraries; Pillow is used only to read and write image files.

- **K-means written from scratch.** It uses k-means++ initialization, vectorized distances and empty-cluster handling.
- **Guided interactive mode.** Numbered menus ask for everything, so you don't need to remember any flags.
- **Pick several options at once**, e.g. `1 3` for a fixed seed plus a palette image.
- **Temporary results.** When you quit, you choose which files to keep, so every start is fresh.
- **One-command mode** runs without any questions, which suits scripts.
- **Reproducible.** The same `--seed` gives the same result.

## Example

| Original (96,085 colors) | k = 4 | k = 8 |
| :---: | :---: | :---: |
| ![Original](assets/autumn_original.jpg) | ![4 colors](assets/autumn_4colors.png) | ![8 colors](assets/autumn_8colors.png) |
| | ![4-color palette](assets/autumn_4colors_palette.png) | ![8-color palette](assets/autumn_8colors_palette.png) |

With **k = 4** the scene is reduced to four browns, from dark trunks to light sky, giving a flat poster look.
With **k = 8** there are more steps of shading, so the stairs, the stone path and the bushes regain their depth.
Each color is the *average* of a cluster, so small bright areas (the red leaves) blend into the surrounding warm tones. A larger k keeps more of them.

<sub>Generated with `quantize assets/autumn_original.jpg -k 4 -o assets/autumn_4colors.png --seed 0 --palette --no-compare` (and `-k 8`). Wallpaper from wallhaven.cc.</sub>

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/giorgimskh/color-quantizer.git
cd color-quantizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
quantize
```

In every new terminal, run `source .venv/bin/activate` again before using `quantize`.

## Interactive mode, step by step

Run `quantize` with no arguments. It first prints a short explanation of what it does and what it will ask. Then:

### 1. Image and number of colors

```text
Image path: ~/Pictures/autumn.png
Loaded autumn.png: 1920x1080 pixels, 197,414 distinct colors
How many colors (k)? 8
```

You can drag an image into the terminal to paste its path. A wrong path or an invalid k is asked again.

### 2. Options: pick any number at once

```text
Options (yes/no options switch when picked; the others ask for a value):
  1. Random seed             (now: random)
  2. Max k-means iterations  (now: 100)
  3. Save palette image      (now: no)
  4. Save comparison image   (now: yes)
  5. Open result in viewer   (now: yes)
Enter numbers separated by spaces (e.g. 1 3), or press Enter to keep: 1 3
Random seed (whole number >= 0, Enter = random): 42
```

- **Pick several at once**, separated by spaces or commas, e.g. `1 3`. Enter keeps everything as it is.
- **Options 3–5 are yes/no**: picking one switches it on or off.
- **Options 1 and 2 ask for a value.**
- **The updated list is printed afterwards**, so you can check the settings.

### 3. The result

```text
Training k-means (k=8) on 50,000 of 2,073,600 pixels...
  converged after 22 iterations
Mapping all 2,073,600 pixels to their nearest color...
Saved 8-color image to output/autumn_k8.png
Saved comparison (original | reconstructed) to output/autumn_k8_compare.png
Saved palette to output/autumn_k8_palette.png
Done in 0.4s
  Colors:      197,414 -> 8
  Color error: 13.5 (RMS, 0-255 scale)
  Palette (most used first):
    #783b26   20.8%
    #73523e   15.3%
    ...
```

The comparison (original on the left, result on the right) opens in your image viewer. In a color terminal, each palette line also shows a swatch of that color.

### 4. What next?

```text
What next?
  1. Try a different number of colors (k) on this image
  2. Choose another image
  3. Change options (seed, iterations, palette, comparison, viewer)
  4. Quit
Choose 1-4: 1
How many colors (k)? [8] 16
```

| Choose | To |
| --- | --- |
| `1` | run again on the same image with a different k (Enter keeps the current k) |
| `2` | choose another image |
| `3` | change options, then return to this menu |
| `4` | quit, choosing which result files to keep |

### 5. Quit and keep what you like

```text
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

Type the numbers of the files to keep. `a` keeps all of them, and Enter deletes all of them. Ctrl+C at any question after a result also takes you here.

## One-command mode

Give the image, `-k` and `-o` and it runs once with no questions:

```bash
quantize photo.jpg -k 8 -o photo_8.png
quantize photo.jpg -k 4 -o poster.png --palette --show --seed 42
```

Anything you leave out is asked for interactively, e.g. `quantize photo.jpg` asks only for k and the options.

| Option | Meaning |
| --- | --- |
| `input` | image to quantize (asked if omitted) |
| `-k`, `--colors` | number of colors, ≥ 1 (asked if omitted) |
| `-o`, `--output` | output path to keep; format from the extension (without it, results are temporary) |
| `--seed` | random seed (≥ 0); the same seed gives the same result |
| `--max-iter` | maximum k-means iterations (default 100) |
| `--palette` | also save the colors as a swatch strip |
| `--no-compare` | don't save the side-by-side comparison |
| `--show` | open the comparison in your image viewer |
| `-i`, `--interactive` | show the options list and the What next? menu even when all arguments are given |
| `-q`, `--quiet` | print only saved files and errors |
| `--version` | print the version |

## Where results go

| Location | What's there | Deleted? |
| --- | --- | --- |
| `output/` | results of the current interactive session | yes, when the program ends (except the files you keep) |
| `saved/` | the files you chose to keep at quit | never. An existing name gets `_1`, `_2`, … instead of being overwritten |
| the `-o` path | the result of a one-command run | never |

File names:

| File | Content |
| --- | --- |
| `<name>_k<k>.png` | the image with k colors |
| `<name>_k<k>_compare.png` | original (left) and result (right) side by side |
| `<name>_k<k>_palette.png` | the k colors as swatches, most used first |

If the program stops because of an error or Ctrl+C while an image is being processed, all of that session's results are deleted without asking. Files the program didn't create are never touched.

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
├── interactive.py  prompts and numbered menus
└── cli.py          the `quantize` command: arguments, sessions, output files, stats
tests/              pytest tests for each module
assets/             example images used in this README
```

## Development

```bash
pip install -e ".[dev]"
pytest                         # all 97 tests
pytest tests/test_kmeans.py    # one file
pytest -k keep                 # tests whose name contains "keep"
```

## Troubleshooting

- **`quantize: command not found`.** Activate the environment first with `source .venv/bin/activate`, from the project folder.
- **No image viewer opens.** Check that option 5 is `yes`. You can also open the files in `output/` yourself while the program is still running.
- **My results disappeared.** Interactive results are temporary. Keep them by typing their numbers when you quit, or use `-o` to write a result that is never deleted.

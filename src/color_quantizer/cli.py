"""Command-line interface for color-quantizer.

Anything not given as an argument is asked for interactively. When anything
was asked (or ``-i`` is given), the CLI keeps offering to try another k.
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from color_quantizer import __version__
from color_quantizer.image_io import load_image, save_image
from color_quantizer.interactive import (
    Cancelled,
    ask_image,
    ask_k,
    ask_output,
    ask_yes_no,
    parse_k,
)
from color_quantizer.quantizer import (
    color_error,
    count_colors,
    palette_image,
    quantize_with_stats,
    side_by_side,
)


def _positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        return parse_k(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``quantize`` command."""
    parser = argparse.ArgumentParser(
        prog="quantize",
        description=(
            "Reduce an image to k colors using k-means clustering. "
            "Anything you leave out is asked for interactively; "
            "run `quantize` with no arguments for a guided session."
        ),
    )
    parser.add_argument("input", type=Path, nargs="?", help="input image path")
    parser.add_argument(
        "-k", "--colors", type=_positive_int, default=None, help="number of colors"
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="output image path")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--max-iter", type=_positive_int, default=100, help="max k-means iterations (default: 100)"
    )
    parser.add_argument(
        "--palette",
        action="store_true",
        help="also save the palette as <output>_palette.png",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="do not save the side-by-side <output>_compare.png (original | reconstructed)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the original and reconstructed images side by side in an image viewer",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="after each result, offer to try another k",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only print saved files and errors"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def palette_path(output: Path) -> Path:
    """Path of the palette image saved next to ``output``."""
    return output.with_name(f"{output.stem}_palette.png")


def compare_path(output: Path) -> Path:
    """Path of the side-by-side comparison image saved next to ``output``."""
    return output.with_name(f"{output.stem}_compare.png")


def default_output(input_path: Path, k: int) -> Path:
    """Default output name: ``<input stem>_k<k>.png`` in the current directory."""
    return Path(f"{input_path.stem}_k{k}.png")


def _use_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def format_palette(palette: np.ndarray, counts: np.ndarray) -> list[str]:
    """One line per color: swatch (in color terminals), hex code, share of pixels."""
    total = counts.sum()
    color = _use_color()
    lines = []
    for (r, g, b), n in zip(palette, counts):
        swatch = f"\033[48;2;{r};{g};{b}m    \033[0m " if color else ""
        lines.append(f"    {swatch}#{r:02x}{g:02x}{b:02x}  {100 * n / total:5.1f}%")
    return lines


class Session:
    """One CLI run: holds the loaded image and options, quantizes for each k."""

    def __init__(self, args: argparse.Namespace, image: np.ndarray, input_path: Path):
        self.args = args
        self.image = image
        self.input_path = input_path
        self.n_pixels = image.shape[0] * image.shape[1]
        self.n_colors = count_colors(image)

    def say(self, message: str = "") -> None:
        """Print progress/stats unless --quiet."""
        if not self.args.quiet:
            print(message, flush=True)

    def clamp_k(self, k: int) -> int:
        """Limit k to the number of pixels, with a warning."""
        if k > self.n_pixels:
            print(
                f"warning: image has only {self.n_pixels} pixels; using k={self.n_pixels}",
                file=sys.stderr,
            )
            return self.n_pixels
        return k

    def run(self, k: int, output: Path, save_palette: bool, show: bool) -> None:
        """Quantize to k colors, save the results and print a summary."""
        args = self.args
        start = time.perf_counter()
        result = quantize_with_stats(
            self.image, k, seed=args.seed, max_iter=args.max_iter, progress=self.say
        )
        elapsed = time.perf_counter() - start

        save_image(result.image, output)
        print(f"Saved {k}-color image to {output}")
        comparison = side_by_side(self.image, result.image)
        if not args.no_compare:
            path = compare_path(output)
            save_image(comparison, path)
            print(f"Saved comparison (original | reconstructed) to {path}")
        if save_palette:
            path = palette_path(output)
            save_image(palette_image(result.palette), path)
            print(f"Saved palette to {path}")

        self.say(f"Done in {elapsed:.1f}s")
        self.say(f"  Colors:      {self.n_colors:,} -> {k}")
        self.say(f"  Color error: {color_error(self.image, result.image):.1f} (RMS, 0-255 scale)")
        self.say("  Palette (most used first):")
        for line in format_palette(result.palette, result.counts):
            self.say(line)

        if show:
            Image.fromarray(comparison).show(title="original | reconstructed")


def run(args: argparse.Namespace) -> int:
    """Gather missing values interactively, then quantize (possibly repeatedly)."""
    wizard = args.input is None or args.output is None
    interactive = wizard or args.colors is None or args.interactive

    if args.input is None:
        input_path, image = ask_image()
    else:
        input_path, image = args.input, load_image(args.input)
    session = Session(args, image, input_path)
    session.say(
        f"Loaded {input_path.name}: {image.shape[1]}x{image.shape[0]} pixels, "
        f"{session.n_colors:,} distinct colors"
    )

    k = args.colors if args.colors is not None else ask_k()
    k = session.clamp_k(k)
    output = args.output if args.output is not None else ask_output(default_output(input_path, k))
    output_is_default = output == default_output(input_path, k)
    save_palette = args.palette or (wizard and ask_yes_no("Save palette image too?", default=False))
    show = args.show or (interactive and ask_yes_no("Open result in image viewer?", default=True))

    base_output = output
    while True:
        session.run(k, output, save_palette, show)
        if not interactive:
            return 0
        try:
            next_k = ask_k("\nTry another k? (Enter to quit) ", allow_empty=True)
        except Cancelled:
            next_k = None
        if next_k is None:
            print("Bye!")
            return 0
        k = session.clamp_k(next_k)
        if output_is_default:
            output = default_output(input_path, k)
        else:
            output = base_output.with_name(f"{base_output.stem}_k{k}{base_output.suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``quantize`` command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Cancelled:
        print("\nerror: cancelled, no answer given", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

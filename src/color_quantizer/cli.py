"""Command-line interface for color-quantizer.

Anything not given as an argument is asked for interactively. In an
interactive run the user also picks options from a numbered list after
entering k, and after each result a numbered menu offers another k, another
image, changing options, or quitting.
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from color_quantizer import __version__
from color_quantizer.image_io import load_image, save_image
from color_quantizer.interactive import (
    Cancelled,
    ask_image,
    ask_int,
    ask_k,
    ask_menu,
    ask_multi,
    ask_output,
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


def _non_negative_int(value: str) -> int:
    """argparse type: an integer >= 0 (NumPy seeds cannot be negative)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


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
    parser.add_argument("--seed", type=_non_negative_int, default=None, help="random seed (>= 0)")
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
        help="ask for options and show the what-next menu even when all arguments are given",
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


@dataclass
class Options:
    """Settings that can be changed from the options menu."""

    seed: int | None
    max_iter: int
    palette: bool
    compare: bool
    show: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Options":
        return cls(args.seed, args.max_iter, args.palette, not args.no_compare, args.show)

    def items(self) -> list[str]:
        """Menu lines showing each option's current value."""
        yes_no = {True: "yes", False: "no"}
        seed = "random" if self.seed is None else str(self.seed)
        return [
            f"Random seed             (now: {seed})",
            f"Max k-means iterations  (now: {self.max_iter})",
            f"Save palette image      (now: {yes_no[self.palette]})",
            f"Save comparison image   (now: {yes_no[self.compare]})",
            f"Open result in viewer   (now: {yes_no[self.show]})",
        ]


def choose_options(options: Options) -> Options:
    """Let the user change any number of options from a numbered list."""
    choices = ask_multi(
        "\nOptions (yes/no options switch when picked; the others ask for a value):",
        options.items(),
    )
    options = replace(options)
    for choice in choices:
        if choice == 1:
            options.seed = ask_int("Random seed (whole number >= 0, Enter = random): ", 0, None)
        elif choice == 2:
            options.max_iter = ask_int(f"Max iterations [{options.max_iter}]: ", 1, options.max_iter)
        elif choice == 3:
            options.palette = not options.palette
        elif choice == 4:
            options.compare = not options.compare
        elif choice == 5:
            options.show = not options.show
    if choices:
        print("Options now:")
        for line in options.items():
            print(f"  - {line}")
    return options


NEXT_MENU = [
    "Try a different number of colors (k) on this image",
    "Choose another image",
    "Change options (seed, iterations, palette, comparison, viewer)",
    "Quit",
]


def build_intro(args: argparse.Namespace) -> str:
    """Explanation printed before the questions of an interactive run.

    Lists only the questions that will actually be asked, given ``args``.
    """
    questions = []
    if args.input is None:
        questions.append("Image path - you can drag the image into this terminal")
    if args.colors is None:
        questions.append("Number of colors k - e.g. 4 (bold poster look), 8, 16, 32 (subtle)")
    questions.append(
        "Options - pick any by number (seed, iterations, palette, comparison, viewer)"
    )
    if args.output is None:
        questions.append("Output file - press Enter to keep the suggested name")

    title = f"color-quantizer {__version__} - reduce an image to k colors"
    lines = [
        "+" + "-" * (len(title) + 4) + "+",
        f"|  {title}  |",
        "+" + "-" * (len(title) + 4) + "+",
        "",
        "What it does:",
        "  Groups similar colors of your image into k clusters with k-means",
        "  and repaints every pixel with its cluster's average color.",
        "  A small k gives a bold poster look; a large k stays close to the original.",
        "",
    ]
    if questions:
        lines.append("You will be asked:")
        lines += [f"  {i}. {q}" for i, q in enumerate(questions, 1)]
        lines.append("")
    lines.append("Files saved:")
    lines.append("  <output>.png           the image with k colors")
    lines.append("  <output>_compare.png   original and result side by side (option 4)")
    lines.append("  <output>_palette.png   the k colors as swatches (option 3)")
    lines += [
        "",
        "After each result a numbered menu lets you try another k, choose another",
        "image, change options, or quit. Ctrl+C quits at any time.",
        "All options, for use without questions: quantize --help",
        "",
    ]
    return "\n".join(lines)


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

    def describe(self) -> None:
        """Print the loaded image's size and color count."""
        self.say(
            f"Loaded {self.input_path.name}: {self.image.shape[1]}x{self.image.shape[0]} "
            f"pixels, {self.n_colors:,} distinct colors"
        )

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

    def run(self, k: int, output: Path, options: Options) -> None:
        """Quantize to k colors, save the results and print a summary."""
        start = time.perf_counter()
        result = quantize_with_stats(
            self.image, k, seed=options.seed, max_iter=options.max_iter, progress=self.say
        )
        elapsed = time.perf_counter() - start

        save_image(result.image, output)
        print(f"Saved {k}-color image to {output}")
        comparison = side_by_side(self.image, result.image)
        if options.compare:
            path = compare_path(output)
            save_image(comparison, path)
            print(f"Saved comparison (original | reconstructed) to {path}")
        if options.palette:
            path = palette_path(output)
            save_image(palette_image(result.palette), path)
            print(f"Saved palette to {path}")

        self.say(f"Done in {elapsed:.1f}s")
        self.say(f"  Colors:      {self.n_colors:,} -> {k}")
        self.say(f"  Color error: {color_error(self.image, result.image):.1f} (RMS, 0-255 scale)")
        self.say("  Palette (most used first):")
        for line in format_palette(result.palette, result.counts):
            self.say(line)

        if options.show:
            Image.fromarray(comparison).show(title="original | reconstructed")


def run(args: argparse.Namespace) -> int:
    """Gather missing values interactively, then quantize (possibly repeatedly)."""
    interactive = (
        args.input is None or args.output is None or args.colors is None or args.interactive
    )
    if interactive and not args.quiet:
        print(build_intro(args), flush=True)

    if args.input is None:
        input_path, image = ask_image()
    else:
        input_path, image = args.input, load_image(args.input)
    session = Session(args, image, input_path)
    session.describe()

    k = args.colors if args.colors is not None else ask_k()
    k = session.clamp_k(k)
    options = Options.from_args(args)
    if interactive:
        options = choose_options(options)
    output = args.output if args.output is not None else ask_output(default_output(input_path, k))
    output_is_default = output == default_output(input_path, k)
    base_output = output

    while True:
        session.run(k, output, options)
        if not interactive:
            return 0
        try:
            while True:
                choice = ask_menu("\nWhat next?", NEXT_MENU)
                if choice != 3:
                    break
                options = choose_options(options)
            if choice == 1:
                k = ask_k(f"How many colors (k)? [{k}] ", default=k)
            elif choice == 2:
                input_path, image = ask_image()
                session = Session(args, image, input_path)
                session.describe()
                k = ask_k(f"How many colors (k)? [{k}] ", default=k)
                # Name outputs after the new image so earlier results are kept.
                output_is_default = True
        except Cancelled:
            choice = 4
        if choice == 4:
            print("Bye!")
            return 0
        k = session.clamp_k(k)
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

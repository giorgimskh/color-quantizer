"""Command-line interface for color-quantizer."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from color_quantizer import __version__
from color_quantizer.image_io import load_image, save_image
from color_quantizer.quantizer import palette_image, quantize, side_by_side


def _positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``quantize`` command."""
    parser = argparse.ArgumentParser(
        prog="quantize",
        description="Reduce an image to k colors using k-means clustering.",
    )
    parser.add_argument("input", type=Path, help="input image path")
    parser.add_argument(
        "-k",
        "--colors",
        type=_positive_int,
        default=None,
        help="number of colors (asked interactively if omitted)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="output image path"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--max-iter", type=_positive_int, default=100, help="max k-means iterations (default: 100)"
    )
    parser.add_argument(
        "--palette",
        action="store_true",
        help="also save the palette as <output>_palette.png and print hex colors",
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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def prompt_for_k() -> int:
    """Ask on stdin for the number of colors until a valid integer >= 1 is given.

    Raises:
        EOFError: If stdin is closed before a valid answer.
    """
    while True:
        answer = input("How many colors (k)? ").strip()
        try:
            return _positive_int(answer)
        except argparse.ArgumentTypeError as exc:
            print(f"Please enter a whole number >= 1 ({exc}).", file=sys.stderr)


def palette_path(output: Path) -> Path:
    """Path of the palette image saved next to ``output``."""
    return output.with_name(f"{output.stem}_palette.png")


def compare_path(output: Path) -> Path:
    """Path of the side-by-side comparison image saved next to ``output``."""
    return output.with_name(f"{output.stem}_compare.png")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``quantize`` command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        image = load_image(args.input)
        n_pixels = image.shape[0] * image.shape[1]
        k = args.colors
        if k is None:
            try:
                k = prompt_for_k()
            except (EOFError, KeyboardInterrupt):
                print("\nerror: no number of colors given", file=sys.stderr)
                return 1
        if k > n_pixels:
            print(
                f"warning: image has only {n_pixels} pixels; using k={n_pixels}",
                file=sys.stderr,
            )
            k = n_pixels

        quantized, palette = quantize(image, k, seed=args.seed, max_iter=args.max_iter)
        save_image(quantized, args.output)
        print(f"Saved {k}-color image to {args.output}")

        comparison = side_by_side(image, quantized)
        if not args.no_compare:
            path = compare_path(args.output)
            save_image(comparison, path)
            print(f"Saved comparison (original | reconstructed) to {path}")
        if args.show:
            Image.fromarray(comparison).show(title="original | reconstructed")

        if args.palette:
            path = palette_path(args.output)
            save_image(palette_image(palette), path)
            print(f"Saved palette to {path}")
            for r, g, b in palette:
                print(f"#{r:02x}{g:02x}{b:02x}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

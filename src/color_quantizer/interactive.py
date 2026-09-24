"""Interactive prompts used by the CLI when values are not given as arguments.

Every prompt re-asks on invalid input. Ctrl+C or end of input (Ctrl+D) raises
:class:`Cancelled`.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from color_quantizer.image_io import load_image


class Cancelled(Exception):
    """The user pressed Ctrl+C or closed input while being asked a question."""


def ask(prompt: str) -> str:
    """Read one stripped line from the user.

    Raises:
        Cancelled: On Ctrl+C or end of input.
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise Cancelled from None


def _complain(message: str) -> None:
    print(f"  {message}", file=sys.stderr)


def ask_yes_no(question: str, default: bool) -> bool:
    """Ask a yes/no question; Enter picks ``default``."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = ask(f"{question} {hint} ").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        _complain("Please answer y or n.")


def parse_k(text: str) -> int:
    """Parse a number of colors (an integer >= 1).

    Raises:
        ValueError: With a user-facing message if ``text`` is not valid.
    """
    try:
        k = int(text)
    except ValueError:
        raise ValueError(f"{text!r} is not a whole number") from None
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return k


def ask_k(prompt: str = "How many colors (k)? ", allow_empty: bool = False) -> int | None:
    """Ask for k until a valid value is given.

    Args:
        prompt: Question shown to the user.
        allow_empty: If True, an empty answer returns None (used to quit).
    """
    while True:
        answer = ask(prompt)
        if not answer and allow_empty:
            return None
        try:
            return parse_k(answer)
        except ValueError as exc:
            _complain(f"{exc}. Please enter a whole number >= 1.")


def clean_path(text: str) -> Path:
    """Turn typed or drag-and-dropped text into a path (strips quotes, expands ~)."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return Path(text).expanduser()


def ask_image() -> tuple[Path, np.ndarray]:
    """Ask for an image path until one loads successfully."""
    while True:
        text = ask("Image path: ")
        if not text:
            _complain("Please enter the path to an image file.")
            continue
        path = clean_path(text)
        try:
            return path, load_image(path)
        except (OSError, ValueError) as exc:
            _complain(f"{exc}. Try again.")


def ask_output(default: Path) -> Path:
    """Ask for the output path; Enter accepts ``default``. The format must be known."""
    extensions = Image.registered_extensions()
    while True:
        text = ask(f"Output file [{default}]: ")
        path = clean_path(text) if text else default
        if path.suffix.lower() in extensions:
            return path
        _complain(f"Unknown image format {path.suffix or '(none)'!r}; use e.g. .png or .jpg.")

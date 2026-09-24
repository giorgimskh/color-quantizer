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


def ask_k(prompt: str = "How many colors (k)? ", default: int | None = None) -> int:
    """Ask for k until a valid value is given.

    Args:
        prompt: Question shown to the user.
        default: Value returned for an empty answer; if None, an answer is required.
    """
    while True:
        answer = ask(prompt)
        if not answer and default is not None:
            return default
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


NEXT_PROMPT = "\nNext: type a number for a new k, i to change the image, or Enter to quit: "


def ask_next() -> int | str | None:
    """Ask what to do after a result.

    Returns:
        A new k (int), ``"image"`` to switch to another image, or None to quit.
    """
    while True:
        answer = ask(NEXT_PROMPT).lower()
        if answer in ("", "q", "quit"):
            return None
        if answer in ("i", "image"):
            return "image"
        try:
            return parse_k(answer)
        except ValueError:
            _complain(f"{answer!r} is not a choice. Type a number (e.g. 16), i, or press Enter.")

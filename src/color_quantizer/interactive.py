"""Interactive prompts used by the CLI when values are not given as arguments.

Every prompt re-asks on invalid input. Ctrl+C or end of input (Ctrl+D) raises
:class:`Cancelled`.
"""

import sys
from pathlib import Path

import numpy as np
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


def ask_int(prompt: str, minimum: int, empty: int | None) -> int | None:
    """Ask for a whole number >= ``minimum``; an empty answer returns ``empty``."""
    while True:
        answer = ask(prompt)
        if not answer:
            return empty
        try:
            value = int(answer)
        except ValueError:
            _complain(f"{answer!r} is not a whole number.")
            continue
        if value >= minimum:
            return value
        _complain(f"Please enter a number >= {minimum}.")


def _print_items(title: str, items: list[str]) -> None:
    print(title)
    for number, item in enumerate(items, 1):
        print(f"  {number}. {item}")


def ask_menu(title: str, items: list[str]) -> int:
    """Show a numbered menu and return the chosen item's number (1-based)."""
    _print_items(title, items)
    while True:
        answer = ask(f"Choose 1-{len(items)}: ")
        if answer.isdigit() and 1 <= int(answer) <= len(items):
            return int(answer)
        _complain(f"Please enter a number from 1 to {len(items)}.")


def parse_choices(text: str, count: int) -> list[int]:
    """Parse numbers like ``"1 3"`` or ``"1,3"`` into a list of distinct choices.

    Raises:
        ValueError: If any entry is not a number from 1 to ``count``.
    """
    choices: list[int] = []
    for part in text.replace(",", " ").split():
        if not (part.isdigit() and 1 <= int(part) <= count):
            raise ValueError(f"{part!r} is not an option (1-{count})")
        if int(part) not in choices:
            choices.append(int(part))
    return choices


def ask_multi(title: str, items: list[str]) -> list[int]:
    """Show a numbered list; the user picks any number of items (Enter picks none)."""
    _print_items(title, items)
    while True:
        answer = ask("Enter numbers separated by spaces (e.g. 1 3), or press Enter to keep: ")
        try:
            return parse_choices(answer, len(items))
        except ValueError as exc:
            _complain(f"{exc}. Try again.")


def ask_keep(files: list[Path]) -> list[int]:
    """List files with numbers and ask which to keep.

    Returns:
        The 1-based numbers of the files to keep: all of them for ``a``,
        none for an empty answer.
    """
    _print_items("\nResult files from this session:", [str(f) for f in files])
    while True:
        answer = ask("Numbers of files to KEEP (e.g. 1 3), a = keep all, Enter = delete all: ")
        if answer.lower() in ("a", "all"):
            return list(range(1, len(files) + 1))
        try:
            return parse_choices(answer, len(files))
        except ValueError as exc:
            _complain(f"{exc}. Try again.")

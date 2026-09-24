"""Image I/O and conversions between image files, (H, W, 3) arrays and (N, 3) pixel arrays."""

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB array.

    Any mode Pillow can read (RGBA, L, P, ...) is converted to RGB.

    Args:
        path: Path to the image file.

    Returns:
        Array of shape (H, W, 3) with dtype uint8.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a readable image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Not a readable image: {path}") from exc


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save an RGB array to an image file; the format is inferred from the extension.

    Parent directories are created if needed.

    Args:
        image: Array of shape (H, W, 3) with dtype uint8.
        path: Destination path.

    Raises:
        ValueError: If ``image`` has the wrong shape or dtype.
    """
    _check_image(image)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {image.dtype}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def image_to_pixels(image: np.ndarray) -> np.ndarray:
    """Flatten an (H, W, 3) image into an (H*W, 3) pixel array in row-major order.

    Raises:
        ValueError: If ``image`` is not of shape (H, W, 3).
    """
    _check_image(image)
    return image.reshape(-1, 3)


def pixels_to_image(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Reshape an (H*W, 3) pixel array back into an (H, W, 3) uint8 image.

    Float values (e.g. centroids) are rounded and clipped to [0, 255].

    Raises:
        ValueError: If ``pixels`` is not (N, 3) or N != height * width.
    """
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError(f"Expected pixels of shape (N, 3), got {pixels.shape}")
    if pixels.shape[0] != height * width:
        raise ValueError(
            f"Cannot reshape {pixels.shape[0]} pixels into {height}x{width} image"
        )
    if pixels.dtype != np.uint8:
        pixels = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
    return pixels.reshape(height, width, 3)


def _check_image(image: np.ndarray) -> None:
    """Raise ValueError unless ``image`` has shape (H, W, 3)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image of shape (H, W, 3), got {image.shape}")

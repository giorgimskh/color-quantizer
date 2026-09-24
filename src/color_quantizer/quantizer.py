"""Quantization pipeline: sample -> fit -> map all pixels -> rebuild image."""

import numpy as np

from color_quantizer.image_io import image_to_pixels, pixels_to_image
from color_quantizer.kmeans import SeedLike, assign, kmeans

DEFAULT_SAMPLE_SIZE = 50_000
# Pixels mapped per chunk; bounds the (chunk, k) distance matrix in memory.
_ASSIGN_CHUNK = 500_000


def sample_pixels(
    pixels: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Random sample of ``n`` pixels without replacement (all pixels if N <= n).

    Args:
        pixels: (N, 3) array.
        n: Sample size (>= 1).
        rng: Random generator.

    Returns:
        (min(N, n), 3) array.
    """
    if n < 1:
        raise ValueError(f"Sample size must be >= 1, got {n}")
    if len(pixels) <= n:
        return pixels
    return pixels[rng.choice(len(pixels), size=n, replace=False)]


def map_pixels(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest-centroid label for every pixel, computed in chunks to bound memory."""
    labels = np.empty(len(pixels), dtype=np.intp)
    for start in range(0, len(pixels), _ASSIGN_CHUNK):
        stop = start + _ASSIGN_CHUNK
        labels[start:stop] = assign(pixels[start:stop], centroids)
    return labels


def quantize(
    image: np.ndarray,
    k: int,
    *,
    seed: SeedLike = None,
    max_iter: int = 100,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an image to at most ``k`` colors.

    K-means is trained on a random sample of pixels, then every pixel is
    mapped once to its nearest centroid.

    Args:
        image: (H, W, 3) uint8 image.
        k: Number of colors.
        seed: Seed or Generator for reproducible results.
        max_iter: Maximum k-means iterations.
        sample_size: Number of pixels used for training.

    Returns:
        ``(quantized, palette)``: the (H, W, 3) uint8 quantized image and the
        (k, 3) uint8 palette, ordered by how many pixels use each color
        (most frequent first).

    Raises:
        ValueError: On invalid image shape or parameters.
    """
    height, width = image.shape[:2]
    pixels = image_to_pixels(image)
    rng = np.random.default_rng(seed)

    sample = sample_pixels(pixels, sample_size, rng)
    result = kmeans(sample, k, max_iter=max_iter, seed=rng)

    labels = map_pixels(pixels, result.centroids)
    palette = np.clip(np.rint(result.centroids), 0, 255).astype(np.uint8)

    # Sort palette by frequency so --palette output is meaningful.
    order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(k)
    palette = palette[order]
    labels = rank[labels]

    return pixels_to_image(palette[labels], height, width), palette


def palette_image(palette: np.ndarray, swatch: int = 50) -> np.ndarray:
    """Render a palette as a horizontal strip of square swatches.

    Args:
        palette: (k, 3) uint8 colors.
        swatch: Side length of each swatch in pixels.

    Returns:
        (swatch, k * swatch, 3) uint8 image.
    """
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"Expected palette of shape (k, 3), got {palette.shape}")
    strip = np.repeat(palette[None, :, :].astype(np.uint8), swatch, axis=1)
    return np.repeat(strip, swatch, axis=0)

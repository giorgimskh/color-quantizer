"""Quantization pipeline: sample -> fit -> map all pixels -> rebuild image."""

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class QuantizeResult:
    """Output of :func:`quantize_with_stats`.

    Attributes:
        image: (H, W, 3) uint8 reconstructed image.
        palette: (k, 3) uint8 colors, most frequent first.
        counts: (k,) number of pixels using each palette color.
        n_iter: K-means iterations performed.
        converged: Whether k-means met a stopping criterion before max_iter.
        n_sampled: Number of pixels k-means was trained on.
    """

    image: np.ndarray
    palette: np.ndarray
    counts: np.ndarray
    n_iter: int
    converged: bool
    n_sampled: int


def quantize_with_stats(
    image: np.ndarray,
    k: int,
    *,
    seed: SeedLike = None,
    max_iter: int = 100,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    progress: Callable[[str], None] | None = None,
) -> QuantizeResult:
    """Reduce an image to at most ``k`` colors and report how it went.

    K-means is trained on a random sample of pixels, then every pixel is
    mapped once to its nearest centroid, whose color is rounded to integers.

    Args:
        image: (H, W, 3) uint8 image.
        k: Number of colors.
        seed: Seed or Generator for reproducible results.
        max_iter: Maximum k-means iterations.
        sample_size: Number of pixels used for training.
        progress: Optional callback receiving a message as each stage starts
            and finishes.

    Returns:
        A :class:`QuantizeResult`.

    Raises:
        ValueError: On invalid image shape or parameters.
    """
    report = progress or (lambda message: None)
    height, width = image.shape[:2]
    pixels = image_to_pixels(image)
    rng = np.random.default_rng(seed)

    sample = sample_pixels(pixels, sample_size, rng)
    report(f"Training k-means (k={k}) on {len(sample):,} of {len(pixels):,} pixels...")
    result = kmeans(sample, k, max_iter=max_iter, seed=rng)
    if result.converged:
        report(f"  converged after {result.n_iter} iterations")
    else:
        report(f"  stopped at the {result.n_iter}-iteration limit (not fully converged)")

    report(f"Mapping all {len(pixels):,} pixels to their nearest color...")
    labels = map_pixels(pixels, result.centroids)
    palette = np.clip(np.rint(result.centroids), 0, 255).astype(np.uint8)

    # Sort palette by frequency so palette output is meaningful.
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(k)
    palette = palette[order]
    labels = rank[labels]

    return QuantizeResult(
        image=pixels_to_image(palette[labels], height, width),
        palette=palette,
        counts=counts[order],
        n_iter=result.n_iter,
        converged=result.converged,
        n_sampled=len(sample),
    )


def quantize(
    image: np.ndarray,
    k: int,
    *,
    seed: SeedLike = None,
    max_iter: int = 100,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an image to at most ``k`` colors.

    Returns:
        ``(quantized, palette)``: the (H, W, 3) uint8 reconstructed image and
        the (k, 3) uint8 palette, most frequent color first. See
        :func:`quantize_with_stats` for details.
    """
    result = quantize_with_stats(
        image, k, seed=seed, max_iter=max_iter, sample_size=sample_size
    )
    return result.image, result.palette


def count_colors(image: np.ndarray) -> int:
    """Number of distinct RGB colors in an (H, W, 3) uint8 image."""
    px = image_to_pixels(image).astype(np.uint32)
    codes = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
    return int(np.unique(codes).size)


def color_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root-mean-square difference between two images, on the 0-255 scale."""
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


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


def side_by_side(
    original: np.ndarray, reconstructed: np.ndarray, gap: int = 10
) -> np.ndarray:
    """Place the original (left) and reconstructed (right) images next to each other.

    Args:
        original: (H, W, 3) uint8 image.
        reconstructed: (H, W, 3) uint8 image of the same shape.
        gap: Width in pixels of the white separator strip.

    Returns:
        (H, 2 * W + gap, 3) uint8 image.

    Raises:
        ValueError: If the two images have different shapes.
    """
    if original.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: {original.shape} vs {reconstructed.shape}"
        )
    separator = np.full((original.shape[0], gap, 3), 255, dtype=np.uint8)
    return np.hstack([original, separator, reconstructed]).astype(np.uint8)

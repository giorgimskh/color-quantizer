"""Tests for color_quantizer.quantizer."""

import numpy as np
import pytest

from color_quantizer.quantizer import palette_image, quantize, sample_pixels


def gradient(h: int = 40, w: int = 60) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.stack([xs * 255 // (w - 1), ys * 255 // (h - 1), np.full_like(xs, 128)], axis=2)
    return img.astype(np.uint8)


def unique_colors(img: np.ndarray) -> int:
    return len(np.unique(img.reshape(-1, 3), axis=0))


def test_output_shape_and_color_count():
    img = gradient()
    out, palette = quantize(img, 5, seed=0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert palette.shape == (5, 3) and palette.dtype == np.uint8
    assert unique_colors(out) <= 5
    assert {tuple(c) for c in out.reshape(-1, 3)} <= {tuple(c) for c in palette}


def test_image_with_exactly_k_colors_is_reproduced():
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [20, 20, 20]], np.uint8)
    idx = np.random.default_rng(0).integers(0, 4, size=(30, 30))
    img = colors[idx]
    out, _ = quantize(img, 4, seed=1)
    np.testing.assert_array_equal(out, img)


def test_reproducible_with_seed():
    img = gradient()
    a, pa = quantize(img, 6, seed=123)
    b, pb = quantize(img, 6, seed=123)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(pa, pb)


def test_small_sample_size_still_maps_all_pixels():
    img = gradient()
    out, _ = quantize(img, 4, seed=0, sample_size=100)
    assert out.shape == img.shape
    assert unique_colors(out) <= 4


def test_palette_sorted_by_frequency():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:2] = [255, 255, 255]  # 20% white, 80% black
    _, palette = quantize(img, 2, seed=0)
    np.testing.assert_array_equal(palette, [[0, 0, 0], [255, 255, 255]])


def test_sample_pixels_caps_and_is_subset():
    rng = np.random.default_rng(0)
    pixels = np.arange(300).reshape(100, 3)
    s = sample_pixels(pixels, 10, rng)
    assert s.shape == (10, 3)
    assert len({tuple(r) for r in s}) == 10
    assert sample_pixels(pixels, 1000, rng) is pixels
    with pytest.raises(ValueError):
        sample_pixels(pixels, 0, rng)


def test_palette_image():
    palette = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)
    strip = palette_image(palette, swatch=3)
    assert strip.shape == (3, 6, 3)
    np.testing.assert_array_equal(strip[0, 0], [1, 2, 3])
    np.testing.assert_array_equal(strip[2, 5], [4, 5, 6])

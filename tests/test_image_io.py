"""Tests for color_quantizer.image_io."""

import numpy as np
import pytest
from PIL import Image

from color_quantizer.image_io import (
    image_to_pixels,
    load_image,
    pixels_to_image,
    save_image,
)


@pytest.fixture
def rgb_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(4, 5, 3), dtype=np.uint8)


def test_save_load_roundtrip(tmp_path, rgb_image):
    path = tmp_path / "sub" / "img.png"
    save_image(rgb_image, path)
    loaded = load_image(path)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, rgb_image)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_load_converts_to_rgb(tmp_path, mode):
    path = tmp_path / f"img_{mode}.png"
    Image.new(mode, (3, 2)).save(path)
    loaded = load_image(path)
    assert loaded.shape == (2, 3, 3)
    assert loaded.dtype == np.uint8


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_load_non_image(tmp_path):
    path = tmp_path / "bad.png"
    path.write_text("not an image")
    with pytest.raises(ValueError):
        load_image(path)


def test_save_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        save_image(np.zeros((2, 2), dtype=np.uint8), tmp_path / "a.png")
    with pytest.raises(ValueError):
        save_image(np.zeros((2, 2, 3), dtype=np.float64), tmp_path / "a.png")


def test_pixels_roundtrip_row_major(rgb_image):
    pixels = image_to_pixels(rgb_image)
    assert pixels.shape == (20, 3)
    np.testing.assert_array_equal(pixels[1], rgb_image[0, 1])
    np.testing.assert_array_equal(pixels[5], rgb_image[1, 0])
    np.testing.assert_array_equal(pixels_to_image(pixels, 4, 5), rgb_image)


def test_image_to_pixels_bad_shape():
    with pytest.raises(ValueError):
        image_to_pixels(np.zeros((2, 2, 4), dtype=np.uint8))


def test_pixels_to_image_bad_shape():
    with pytest.raises(ValueError):
        pixels_to_image(np.zeros((6, 3), dtype=np.uint8), 2, 2)
    with pytest.raises(ValueError):
        pixels_to_image(np.zeros((4, 2), dtype=np.uint8), 2, 2)


def test_pixels_to_image_float_rounded_and_clipped():
    pixels = np.array([[-5.0, 0.4, 0.6], [254.6, 300.0, 127.5]])
    img = pixels_to_image(pixels, 1, 2)
    assert img.dtype == np.uint8
    np.testing.assert_array_equal(img[0], [[0, 0, 1], [255, 255, 128]])

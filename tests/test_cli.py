"""Tests for color_quantizer.cli."""

import numpy as np
import pytest

from color_quantizer.cli import main
from color_quantizer.image_io import load_image, save_image


@pytest.fixture
def input_image(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "in.png"
    save_image(rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8), path)
    return path


def test_quantize_writes_output(tmp_path, input_image, capsys):
    out = tmp_path / "out" / "q.png"
    assert main([str(input_image), "-k", "4", "-o", str(out), "--seed", "0"]) == 0
    img = load_image(out)
    assert img.shape == (20, 30, 3)
    assert len(np.unique(img.reshape(-1, 3), axis=0)) <= 4
    assert "Saved 4-color image" in capsys.readouterr().out


def test_palette_flag(tmp_path, input_image, capsys):
    out = tmp_path / "q.png"
    assert main([str(input_image), "-k", "3", "-o", str(out), "--palette", "--seed", "1"]) == 0
    strip = load_image(tmp_path / "q_palette.png")
    assert strip.shape == (50, 150, 3)
    hex_lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("#")]
    assert len(hex_lines) == 3


def test_seed_makes_output_reproducible(tmp_path, input_image):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for out in (a, b):
        main([str(input_image), "-k", "5", "-o", str(out), "--seed", "7"])
    np.testing.assert_array_equal(load_image(a), load_image(b))


def test_missing_input_returns_error(tmp_path, capsys):
    code = main([str(tmp_path / "nope.png"), "-k", "4", "-o", str(tmp_path / "o.png")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_non_image_input_returns_error(tmp_path, capsys):
    bad = tmp_path / "bad.png"
    bad.write_text("hello")
    assert main([str(bad), "-k", "2", "-o", str(tmp_path / "o.png")]) == 1
    assert "error:" in capsys.readouterr().err


def test_k_larger_than_pixel_count_is_clamped(tmp_path, capsys):
    tiny = tmp_path / "tiny.png"
    save_image(np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8), tiny)
    assert main([str(tiny), "-k", "10", "-o", str(tmp_path / "o.png")]) == 0
    assert "warning" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["in.png", "-k", "0", "-o", "o.png"],
        ["in.png", "-k", "abc", "-o", "o.png"],
        ["in.png", "-o", "o.png"],
        ["in.png", "-k", "4"],
        ["in.png", "-k", "4", "-o", "o.png", "--max-iter", "0"],
    ],
)
def test_bad_arguments_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2

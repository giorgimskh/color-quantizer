"""Tests for color_quantizer.cli."""

from pathlib import Path

import numpy as np
import pytest

from color_quantizer.cli import main
from color_quantizer.image_io import load_image, save_image
from color_quantizer.interactive import clean_path, parse_k


@pytest.fixture
def input_image(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "in.png"
    save_image(rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8), path)
    return path


@pytest.fixture(autouse=True)
def viewer(monkeypatch):
    """Never open a real image viewer; record what would have been shown."""
    shown = []
    monkeypatch.setattr("PIL.Image.Image.show", lambda self, title=None: shown.append(self.size))
    return shown


@pytest.fixture
def no_input(monkeypatch):
    """Fail if the CLI asks anything."""

    def fail(prompt):
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    monkeypatch.setattr("builtins.input", fail)


@pytest.fixture
def answers(monkeypatch):
    """Feed scripted answers to input(); records the prompts that were shown."""
    prompts = []

    def feed(*replies):
        it = iter(replies)

        def fake_input(prompt):
            prompts.append(prompt)
            try:
                reply = next(it)
            except StopIteration:
                raise AssertionError(f"no scripted answer for prompt {prompt!r}") from None
            if isinstance(reply, BaseException):
                raise reply
            return reply

        monkeypatch.setattr("builtins.input", fake_input)
        return prompts

    return feed


def n_colors(path: Path) -> int:
    return len(np.unique(load_image(path).reshape(-1, 3), axis=0))


# --- Non-interactive use (all arguments given) ---------------------------------


def test_quantize_writes_output(tmp_path, input_image, capsys, no_input):
    out = tmp_path / "out" / "q.png"
    assert main([str(input_image), "-k", "4", "-o", str(out), "--seed", "0"]) == 0
    assert load_image(out).shape == (20, 30, 3)
    assert n_colors(out) <= 4
    assert "Saved 4-color image" in capsys.readouterr().out


def test_progress_and_stats_printed(tmp_path, input_image, capsys, no_input):
    main([str(input_image), "-k", "3", "-o", str(tmp_path / "q.png"), "--seed", "0"])
    out = capsys.readouterr().out
    assert "Loaded in.png: 30x20 pixels, 600 distinct colors" in out
    assert "Training k-means (k=3) on 600 of 600 pixels" in out
    assert "Mapping all 600 pixels" in out
    assert "Colors:      600 -> 3" in out
    assert "Color error:" in out
    palette_lines = [line for line in out.splitlines() if line.strip().startswith("#")]
    assert len(palette_lines) == 3
    shares = [float(line.split()[-1].rstrip("%")) for line in palette_lines]
    assert sum(shares) == pytest.approx(100, abs=0.2)
    assert shares == sorted(shares, reverse=True)


def test_quiet_hides_progress(tmp_path, input_image, capsys, no_input):
    main([str(input_image), "-k", "3", "-o", str(tmp_path / "q.png"), "-q"])
    out = capsys.readouterr().out
    assert "Saved 3-color image" in out
    assert "Training" not in out and "Palette" not in out


def test_palette_flag(tmp_path, input_image, no_input):
    out = tmp_path / "q.png"
    assert main([str(input_image), "-k", "3", "-o", str(out), "--palette", "--seed", "1"]) == 0
    assert load_image(tmp_path / "q_palette.png").shape == (50, 150, 3)


def test_seed_makes_output_reproducible(tmp_path, input_image, no_input):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for out in (a, b):
        main([str(input_image), "-k", "5", "-o", str(out), "--seed", "7"])
    np.testing.assert_array_equal(load_image(a), load_image(b))


def test_comparison_saved_by_default(tmp_path, input_image, no_input):
    out = tmp_path / "q.png"
    assert main([str(input_image), "-k", "4", "-o", str(out), "--seed", "0"]) == 0
    combo = load_image(tmp_path / "q_compare.png")
    assert combo.shape == (20, 2 * 30 + 10, 3)
    np.testing.assert_array_equal(combo[:, :30], load_image(input_image))
    np.testing.assert_array_equal(combo[:, 40:], load_image(out))


def test_no_compare_flag(tmp_path, input_image, no_input):
    out = tmp_path / "q.png"
    assert main([str(input_image), "-k", "4", "-o", str(out), "--no-compare"]) == 0
    assert out.exists()
    assert not (tmp_path / "q_compare.png").exists()


def test_show_opens_viewer(tmp_path, input_image, viewer, no_input):
    out = tmp_path / "q.png"
    assert main([str(input_image), "-k", "3", "-o", str(out), "--show", "--no-compare"]) == 0
    assert viewer == [(70, 20)]


def test_missing_input_returns_error(tmp_path, capsys, no_input):
    code = main([str(tmp_path / "nope.png"), "-k", "4", "-o", str(tmp_path / "o.png")])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_non_image_input_returns_error(tmp_path, capsys, no_input):
    bad = tmp_path / "bad.png"
    bad.write_text("hello")
    assert main([str(bad), "-k", "2", "-o", str(tmp_path / "o.png")]) == 1
    assert "error:" in capsys.readouterr().err


def test_k_larger_than_pixel_count_is_clamped(tmp_path, capsys, no_input):
    tiny = tmp_path / "tiny.png"
    save_image(np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8), tiny)
    assert main([str(tiny), "-k", "10", "-o", str(tmp_path / "o.png")]) == 0
    assert "warning" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["in.png", "-k", "0", "-o", "o.png"],
        ["in.png", "-k", "abc", "-o", "o.png"],
        ["in.png", "-k", "4", "-o", "o.png", "--max-iter", "0"],
    ],
)
def test_bad_arguments_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


# --- Interactive use -----------------------------------------------------------


def test_k_prompted_when_omitted(tmp_path, input_image, answers, capsys):
    out = tmp_path / "q.png"
    answers("abc", "0", " 3 ", "n", "")  # two bad k's, k=3, no viewer, quit
    assert main([str(input_image), "-o", str(out), "--seed", "0"]) == 0
    assert n_colors(out) <= 3
    captured = capsys.readouterr()
    assert captured.err.count("Please enter a whole number") == 2
    assert "Saved 3-color image" in captured.out
    assert "Bye!" in captured.out


def test_full_wizard_with_try_again_loop(tmp_path, input_image, answers, capsys, monkeypatch, viewer):
    monkeypatch.chdir(tmp_path)
    prompts = answers(
        str(tmp_path / "missing.png"),  # bad path -> asked again
        f"'{input_image}'",  # quoted, as when drag-and-dropped
        "5",  # k
        "",  # accept default output name
        "y",  # save palette
        "",  # open viewer (default yes)
        "2",  # try another k
        "",  # quit
    )
    assert main([]) == 0
    assert prompts[:2] == ["Image path: ", "Image path: "]
    assert "Output file [in_k5.png]: " in prompts
    for name in ["in_k5.png", "in_k5_compare.png", "in_k5_palette.png",
                 "in_k2.png", "in_k2_compare.png", "in_k2_palette.png"]:
        assert (tmp_path / name).exists(), name
    assert n_colors(tmp_path / "in_k2.png") <= 2
    assert len(viewer) == 2  # opened after each round
    captured = capsys.readouterr()
    assert "Try again" in captured.err
    assert "Bye!" in captured.out


def test_wizard_rejects_unknown_output_format(tmp_path, input_image, answers, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers(str(input_image), "2", "result.xyz", "result.png", "n", "n", "")
    assert main([]) == 0
    assert (tmp_path / "result.png").exists()
    assert "Unknown image format" in capsys.readouterr().err


def test_interactive_flag_names_extra_outputs_after_custom_output(tmp_path, input_image, answers):
    out = tmp_path / "art.png"
    prompts = answers("n", "2", "")  # no viewer, then k=2, then quit
    assert main([str(input_image), "-k", "4", "-o", str(out), "-i", "--no-compare"]) == 0
    assert out.exists() and (tmp_path / "art_k2.png").exists()
    assert "Try another k? (Enter to quit) " in prompts[1]


def test_cancel_at_first_question_returns_error(tmp_path, input_image, answers, capsys):
    answers(EOFError())
    assert main([str(input_image), "-o", str(tmp_path / "q.png")]) == 1
    assert "cancelled" in capsys.readouterr().err


def test_ctrl_c_at_try_again_quits_cleanly(tmp_path, input_image, answers, capsys):
    answers("n", KeyboardInterrupt())
    assert main([str(input_image), "-k", "2", "-o", str(tmp_path / "q.png"), "-i"]) == 0
    assert "Bye!" in capsys.readouterr().out


def test_missing_input_fails_before_prompt(tmp_path, no_input):
    assert main([str(tmp_path / "nope.png"), "-o", str(tmp_path / "o.png")]) == 1


# --- Helpers -------------------------------------------------------------------


def test_clean_path():
    assert clean_path("'/a b/c.png'") == Path("/a b/c.png")
    assert clean_path('"x.png"') == Path("x.png")
    assert clean_path("~/p.png") == Path.home() / "p.png"


@pytest.mark.parametrize("text", ["0", "-2", "abc", "", "1.5"])
def test_parse_k_rejects(text):
    with pytest.raises(ValueError):
        parse_k(text)


# --- Intro text ----------------------------------------------------------------


def test_intro_printed_before_first_question(tmp_path, input_image, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    seen_before_first_prompt = []
    replies = iter([str(input_image), "2", "", "n", "n", ""])

    def fake_input(prompt):
        if not seen_before_first_prompt:
            seen_before_first_prompt.append(capsys.readouterr().out)
        return next(replies)

    monkeypatch.setattr("builtins.input", fake_input)
    assert main([]) == 0
    intro = seen_before_first_prompt[0]
    assert intro.startswith("+---")
    assert "What it does:" in intro
    for question in ["1. Image path", "2. Number of colors k", "3. Output file",
                     "4. Save the palette", "5. Open the result"]:
        assert question in intro
    assert "<output>_compare.png" in intro


def test_intro_lists_only_questions_that_will_be_asked(tmp_path, input_image, answers, capsys):
    answers("3", "")  # k=3, then quit (viewer not asked because of --show)
    assert main([str(input_image), "-o", str(tmp_path / "q.png"), "--show", "--no-compare"]) == 0
    out = capsys.readouterr().out
    assert "1. Number of colors k" in out
    assert "Image path" not in out and "Open the result" not in out
    assert "_compare.png" not in out.split("Loaded")[0]


def test_no_intro_when_quiet(tmp_path, input_image, answers, capsys):
    answers("3", "n", "")
    main([str(input_image), "-o", str(tmp_path / "q.png"), "-q"])
    assert "What it does" not in capsys.readouterr().out


def test_no_intro_when_all_arguments_given(tmp_path, input_image, capsys, no_input):
    main([str(input_image), "-k", "3", "-o", str(tmp_path / "q.png")])
    assert "What it does" not in capsys.readouterr().out

"""Tests for color_quantizer.cli."""

from pathlib import Path

import numpy as np
import pytest

from color_quantizer.cli import main
from color_quantizer.image_io import load_image, save_image
from color_quantizer.interactive import clean_path, parse_choices, parse_k


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
            if callable(reply):
                reply = reply()
            return reply

        monkeypatch.setattr("builtins.input", fake_input)
        return prompts

    return feed


def png_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.png")) if folder.exists() else []


def snapshot(folder: Path, store: dict, reply: str):
    """An answer that records the PNGs in ``folder`` at prompt time, then replies."""

    def take():
        store["files"] = png_files(folder)
        return reply

    return take


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
# Answer order in an interactive run: [image path], [k], options list, then the
# "What next?" menu after each result. Results are saved in output/ and deleted
# at exit, except files at a path given with -o.

KEEP = ""  # Enter: keep options / accept default
QUIT = "4"  # "What next?" menu: Quit
DELETE_ALL = ""  # keep list at quit: Enter deletes all


def test_k_prompted_when_omitted(tmp_path, input_image, answers, capsys):
    out = tmp_path / "q.png"
    answers("abc", "0", " 3 ", KEEP, QUIT, DELETE_ALL)  # two bad k's, k=3
    assert main([str(input_image), "-o", str(out), "--seed", "0"]) == 0
    assert n_colors(out) <= 3
    captured = capsys.readouterr()
    assert captured.err.count("Please enter a whole number") == 2
    assert "Saved 3-color image" in captured.out
    assert "Bye!" in captured.out


def test_full_wizard(tmp_path, input_image, answers, capsys, monkeypatch, viewer):
    monkeypatch.chdir(tmp_path)
    during = {}
    prompts = answers(
        str(tmp_path / "missing.png"),  # bad path -> asked again
        f"'{input_image}'",  # quoted, as when drag-and-dropped
        "5",  # k
        "3",  # options: palette on (viewer is on by default)
        "1", "2",  # What next -> another k: 2
        snapshot(tmp_path / "output", during, QUIT), DELETE_ALL,
    )
    assert main([]) == 0
    assert prompts[:2] == ["Image path: ", "Image path: "]
    assert not any(p.startswith("Output file") for p in prompts)
    assert during["files"] == sorted(
        f"in_k{k}{suffix}.png" for k in (5, 2) for suffix in ("", "_compare", "_palette")
    )
    assert not (tmp_path / "output").exists()  # everything deleted at exit
    assert len(viewer) == 2  # opened after each round
    captured = capsys.readouterr()
    assert "Try again" in captured.err
    assert "Deleted 6 result file(s) from output/" in captured.out
    assert "Bye!" in captured.out


def test_options_list_shown_after_k(tmp_path, input_image, answers, capsys):
    prompts = answers("3", KEEP, QUIT, DELETE_ALL)
    main([str(input_image), "-o", str(tmp_path / "q.png")])
    out = capsys.readouterr().out
    for line in ["1. Random seed", "2. Max k-means iterations", "3. Save palette image",
                 "4. Save comparison image", "5. Open result in viewer"]:
        assert line in out
    assert prompts[0] == "How many colors (k)? "
    assert prompts[1].startswith("Enter numbers separated by spaces")


def test_options_multiple_choices_with_values(tmp_path, input_image, answers, capsys):
    out = tmp_path / "q.png"
    prompts = answers(
        "3",
        "1 2 4 9",  # 9 is invalid -> asked again
        "1, 2, 4, 3",  # commas also work
        "-5", "7",  # seed: negative rejected, then 7
        "abc", "25",  # max iterations
        QUIT,
    )
    assert main([str(input_image), "-o", str(out)]) == 0
    assert "Random seed (whole number >= 0, Enter = random): " in prompts
    assert "Max iterations [100]: " in prompts
    assert out.exists()
    assert not (tmp_path / "q_compare.png").exists()  # option 4 switched off
    assert (tmp_path / "q_palette.png").exists()  # option 3 switched on
    captured = capsys.readouterr()
    assert "'9' is not an option (1-5)" in captured.err
    assert "Random seed             (now: 7)" in captured.out
    assert "Max k-means iterations  (now: 25)" in captured.out


def test_options_seed_makes_rounds_reproducible(tmp_path, input_image, answers):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for out in (a, b):
        answers("4", "1", "11", QUIT, DELETE_ALL)
        main([str(input_image), "-o", str(out)])
    np.testing.assert_array_equal(load_image(a), load_image(b))


def test_cli_flags_preset_options(tmp_path, input_image, answers, capsys):
    answers("3", KEEP, QUIT, DELETE_ALL)
    main([str(input_image), "-o", str(tmp_path / "q.png"), "--seed", "5", "--palette", "--no-compare"])
    out = capsys.readouterr().out
    assert "(now: 5)" in out
    assert "Save palette image      (now: yes)" in out
    assert "Save comparison image   (now: no)" in out


def test_what_next_menu_is_numbered(tmp_path, input_image, answers, capsys):
    prompts = answers("3", KEEP, "0", "abc", QUIT, DELETE_ALL)
    assert main([str(input_image), "-o", str(tmp_path / "q.png")]) == 0
    captured = capsys.readouterr()
    for line in ["What next?", "1. Try a different number of colors (k)",
                 "2. Choose another image", "3. Change options", "4. Quit"]:
        assert line in captured.out
    assert prompts[-1] == "Choose 1-4: "
    assert captured.err.count("Please enter a number from 1 to 4") == 2


def test_what_next_change_options_then_run(tmp_path, input_image, answers):
    out = tmp_path / "q.png"
    during = {}
    answers(
        "4", KEEP,  # first round: k=4, default options
        "3", "3",  # What next -> change options -> palette on
        "1", KEEP,  # What next -> another k, Enter keeps k=4
        snapshot(tmp_path, during, QUIT), DELETE_ALL,
    )
    assert main([str(input_image), "-o", str(out)]) == 0
    assert {"q_k4.png", "q_k4_palette.png"} <= set(during["files"])
    assert "q_palette.png" not in during["files"]
    assert out.exists() and (tmp_path / "q_compare.png").exists()  # -o files kept
    assert not (tmp_path / "q_k4.png").exists()  # derived files deleted


def test_interactive_flag_names_extra_outputs_after_custom_output(tmp_path, input_image, answers):
    out = tmp_path / "art.png"
    during = {}
    prompts = answers(KEEP, "1", "2", snapshot(tmp_path, during, QUIT), DELETE_ALL)
    assert main([str(input_image), "-k", "4", "-o", str(out), "-i", "--no-compare"]) == 0
    assert {"art.png", "art_k2.png"} <= set(during["files"])
    assert out.exists() and not (tmp_path / "art_k2.png").exists()
    assert prompts[0].startswith("Enter numbers separated by spaces")  # options still offered


def test_cancel_at_first_question_returns_error(tmp_path, input_image, answers, capsys):
    answers(EOFError())
    assert main([str(input_image), "-o", str(tmp_path / "q.png")]) == 1
    assert "cancelled" in capsys.readouterr().err


def test_ctrl_c_at_menu_quits_cleanly(tmp_path, input_image, answers, capsys):
    answers(KEEP, KeyboardInterrupt())
    assert main([str(input_image), "-k", "2", "-o", str(tmp_path / "q.png"), "-i"]) == 0
    assert "Bye!" in capsys.readouterr().out


def test_missing_input_fails_before_prompt(tmp_path, no_input):
    assert main([str(tmp_path / "nope.png"), "-o", str(tmp_path / "o.png")]) == 1


def test_negative_seed_argument_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["in.png", "-k", "2", "-o", "o.png", "--seed", "-1"])
    assert exc.value.code == 2


# --- Changing the image between rounds -----------------------------------------


@pytest.fixture
def second_image(tmp_path):
    path = tmp_path / "other.png"
    img = np.zeros((10, 12, 3), np.uint8)
    img[:, 6:] = [250, 10, 10]
    save_image(img, path)
    return path


def test_change_image_after_a_result(tmp_path, input_image, second_image, answers, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    during = {}
    prompts = answers(
        str(input_image), "3", KEEP,  # first round: in.png, k=3
        "2",  # What next -> choose another image
        "nope.png",  # bad path -> asked again
        str(second_image),
        KEEP,  # keep k=3
        "1", "2",  # What next -> another k: 2
        snapshot(tmp_path / "output", during, QUIT), DELETE_ALL,
    )
    assert main([]) == 0
    assert "How many colors (k)? [3] " in prompts
    assert {"in_k3.png", "other_k3.png", "other_k2.png"} <= set(during["files"])
    assert "in_k2.png" not in during["files"]
    assert not (tmp_path / "output").exists()
    out = capsys.readouterr().out
    assert "Loaded other.png: 12x10 pixels, 2 distinct colors" in out
    assert "Bye!" in out


def test_change_image_result_colors(tmp_path, second_image, answers, monkeypatch):
    monkeypatch.chdir(tmp_path)
    grabbed = {}

    def grab():
        grabbed["img"] = load_image(tmp_path / "output" / "other_k2.png")
        return QUIT

    answers(str(second_image), "2", KEEP, grab, DELETE_ALL)
    assert main([]) == 0
    np.testing.assert_array_equal(grabbed["img"][0, [0, 11]], [[0, 0, 0], [250, 10, 10]])


def test_change_image_after_custom_output_uses_default_names(
    tmp_path, input_image, second_image, answers, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    during = {}
    answers(KEEP, "2", str(second_image), "4", snapshot(tmp_path / "output", during, QUIT), DELETE_ALL)
    assert main([str(input_image), "-k", "2", "-o", "art.png", "-i", "--no-compare"]) == 0
    assert "other_k4.png" in during["files"]
    assert (tmp_path / "art.png").exists()
    assert not (tmp_path / "output").exists()


def test_ctrl_c_while_changing_image_quits_cleanly(tmp_path, input_image, answers, capsys):
    answers(KEEP, "2", KeyboardInterrupt())
    assert main([str(input_image), "-k", "2", "-o", str(tmp_path / "q.png"), "-i"]) == 0
    assert "Bye!" in capsys.readouterr().out


# --- Intro text ----------------------------------------------------------------


def test_intro_printed_before_first_question(tmp_path, input_image, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    seen_before_first_prompt = []
    replies = iter([str(input_image), "2", KEEP, QUIT, DELETE_ALL])

    def fake_input(prompt):
        if not seen_before_first_prompt:
            seen_before_first_prompt.append(capsys.readouterr().out)
        return next(replies)

    monkeypatch.setattr("builtins.input", fake_input)
    assert main([]) == 0
    intro = seen_before_first_prompt[0]
    assert intro.startswith("+---")
    assert "What it does:" in intro
    for question in ["1. Image path", "2. Number of colors k", "3. Options"]:
        assert question in intro
    assert "Output file" not in intro
    assert "you choose which results to keep" in intro
    assert "the rest are deleted" in intro
    assert "numbered menu" in intro


def test_intro_lists_only_questions_that_will_be_asked(tmp_path, input_image, answers, capsys):
    answers("3", KEEP, QUIT, DELETE_ALL)
    assert main([str(input_image), "-o", str(tmp_path / "q.png")]) == 0
    intro = capsys.readouterr().out.split("Loaded")[0]
    assert "1. Number of colors k" in intro and "2. Options" in intro
    assert "Image path" not in intro and "Output file" not in intro


def test_no_intro_when_quiet(tmp_path, input_image, answers, capsys):
    answers("3", KEEP, QUIT, DELETE_ALL)
    main([str(input_image), "-o", str(tmp_path / "q.png"), "-q"])
    assert "What it does" not in capsys.readouterr().out


def test_no_intro_when_all_arguments_given(tmp_path, input_image, capsys, no_input):
    main([str(input_image), "-k", "3", "-o", str(tmp_path / "q.png")])
    assert "What it does" not in capsys.readouterr().out


# --- Helpers -------------------------------------------------------------------


def test_clean_path():
    assert clean_path("'/a b/c.png'") == Path("/a b/c.png")
    assert clean_path('"x.png"') == Path("x.png")
    assert clean_path("~/p.png") == Path.home() / "p.png"


@pytest.mark.parametrize("text", ["0", "-2", "abc", "", "1.5"])
def test_parse_k_rejects(text):
    with pytest.raises(ValueError):
        parse_k(text)


def test_parse_choices():
    assert parse_choices("1 3", 5) == [1, 3]
    assert parse_choices(" 3,1 , 3 ", 5) == [3, 1]
    assert parse_choices("", 5) == []
    for bad in ["0", "6", "x", "1 -2", "1.5"]:
        with pytest.raises(ValueError):
            parse_choices(bad, 5)


# --- Temporary results ---------------------------------------------------------


def test_unrelated_files_in_output_folder_survive(tmp_path, input_image, answers, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    mine = tmp_path / "output" / "mine.txt"
    mine.write_text("keep me")
    answers(str(input_image), "2", KEEP, QUIT, DELETE_ALL)
    assert main([]) == 0
    assert mine.read_text() == "keep me"
    assert png_files(tmp_path / "output") == []


def test_cleanup_on_ctrl_c_at_menu(tmp_path, input_image, answers, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    answers(str(input_image), "2", KEEP, KeyboardInterrupt(), DELETE_ALL)
    assert main([]) == 0
    assert not (tmp_path / "output").exists()
    assert "Deleted 2 result file(s)" in capsys.readouterr().out


@pytest.mark.parametrize("error, code", [(ValueError("boom"), 1), (KeyboardInterrupt(), 130)])
def test_cleanup_when_second_round_fails(tmp_path, input_image, answers, monkeypatch, capsys, error, code):
    monkeypatch.chdir(tmp_path)
    import color_quantizer.cli as cli

    real = cli.quantize_with_stats
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise error
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "quantize_with_stats", flaky)
    answers(str(input_image), "3", KEEP, "1", "2")
    assert main([]) == code
    assert not (tmp_path / "output").exists()
    err = capsys.readouterr().err
    assert ("error: boom" in err) if code == 1 else ("Interrupted." in err)


def test_quiet_cleanup_prints_nothing(tmp_path, input_image, answers, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    answers(str(input_image), "2", KEEP, QUIT, DELETE_ALL)
    main(["-q"])
    assert "Deleted" not in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


# --- Choosing which results to keep at quit ------------------------------------


@pytest.fixture
def two_rounds(tmp_path, input_image, answers, monkeypatch):
    """Run a wizard session with k=3 then k=2 (default options), then answer the keep list."""
    monkeypatch.chdir(tmp_path)

    def go(*keep_answers):
        prompts = answers(str(input_image), "3", KEEP, "1", "2", QUIT, *keep_answers)
        code = main([])
        return code, prompts

    return go


def test_keep_list_shows_files_in_order(two_rounds, capsys):
    code, prompts = two_rounds(DELETE_ALL)
    assert code == 0
    out = capsys.readouterr().out
    listing = out.split("Result files from this session:")[1]
    assert [line.strip() for line in listing.splitlines()[1:5]] == [
        "1. output/in_k3.png", "2. output/in_k3_compare.png",
        "3. output/in_k2.png", "4. output/in_k2_compare.png",
    ]
    assert prompts[-1].startswith("Numbers of files to KEEP")
    assert out.rstrip().endswith("Bye!")


def test_keep_chosen_files_moved_to_saved(two_rounds, tmp_path, capsys):
    code, _ = two_rounds("3 1")
    assert code == 0
    assert png_files(tmp_path / "saved") == ["in_k2.png", "in_k3.png"]
    assert not (tmp_path / "output").exists()
    out = capsys.readouterr().out
    assert "Kept 2 file(s) in saved/:" in out
    assert "Deleted 2 result file(s)" in out


def test_keep_all(two_rounds, tmp_path, capsys):
    two_rounds("a")
    assert len(png_files(tmp_path / "saved")) == 4
    assert not (tmp_path / "output").exists()
    assert "Deleted" not in capsys.readouterr().out


def test_keep_none_with_enter(two_rounds, tmp_path):
    two_rounds(DELETE_ALL)
    assert not (tmp_path / "saved").exists()
    assert not (tmp_path / "output").exists()


def test_keep_list_rejects_invalid_numbers(two_rounds, tmp_path, capsys):
    two_rounds("9", "x 1", "2")
    assert png_files(tmp_path / "saved") == ["in_k3_compare.png"]
    assert capsys.readouterr().err.count("is not an option (1-4)") == 2


def test_ctrl_c_at_keep_list_deletes_all(two_rounds, tmp_path):
    code, _ = two_rounds(KeyboardInterrupt())
    assert code == 0
    assert not (tmp_path / "saved").exists()
    assert not (tmp_path / "output").exists()


def test_ctrl_c_at_menu_still_offers_keep_list(tmp_path, input_image, answers, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers(str(input_image), "2", KEEP, KeyboardInterrupt(), "1")
    assert main([]) == 0
    assert png_files(tmp_path / "saved") == ["in_k2.png"]


def test_saved_name_clash_gets_suffix(two_rounds, tmp_path):
    (tmp_path / "saved").mkdir()
    existing = tmp_path / "saved" / "in_k3.png"
    existing.write_bytes(b"old")
    two_rounds("1")
    assert existing.read_bytes() == b"old"
    assert (tmp_path / "saved" / "in_k3_1.png").exists()


def test_o_files_not_listed(tmp_path, input_image, answers, capsys):
    out = tmp_path / "art.png"
    answers(KEEP, "1", "2", QUIT, "a")
    assert main([str(input_image), "-k", "4", "-o", str(out), "-i", "--no-compare"]) == 0
    listing = capsys.readouterr().out.split("Result files from this session:")[1]
    assert "art_k2.png" in listing and "art.png" not in listing
    assert out.exists()

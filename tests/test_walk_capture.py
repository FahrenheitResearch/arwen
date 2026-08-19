"""The walk-capture harness measures the run it actually performed.

A UX harness that gets its own arithmetic wrong is worse than no
harness: it produces a confident, plausible, wrong picture of the
product, and the wrongness is invisible because nobody re-walks the walk
by hand to check.  So every promise the report makes is held here
against a real child process -- the tests run commands and read what
came back, they do not mock ``subprocess``.

The four surfaces under gate:

* capture correctness -- exit codes, both streams, wall time, and the
  files a step created under the walk root;
* ``expect_failure`` semantics in both directions, including the one
  that matters most: a step declared as a failure that *succeeds* is
  friction, because the walk's description of the product has gone
  stale;
* the report renders, is self-contained, and states time-to-first-plot;
* no shell gets between the walk script and the child, so a Windows path
  containing ``&`` arrives as text rather than as a second command.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import walk_capture as wc  # noqa: E402

PY = sys.executable


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _walk_document(root: Path, steps: list[dict], **extra) -> dict:
    document = {
        "name": "gate walk",
        "root": str(root),
        "steps": steps,
    }
    document.update(extra)
    return document


def _run(tmp_path: Path, steps: list[dict], **extra) -> dict:
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    walk = wc.build_walk(_walk_document(root, steps, **extra),
                         script_path=tmp_path / "walk.toml")
    return wc.run_walk(walk, tmp_path / "capture")


# ---------------------------------------------------------------- tokenizing

@pytest.mark.parametrize("command,expected", [
    ("gpuwm doctor", ["gpuwm", "doctor"]),
    (r'py --out=C:\walks\deep\a.toml', ["py", r"--out=C:\walks\deep\a.toml"]),
    (r'py --out="C:\Program Files\a b\d.toml"',
     ["py", r"--out=C:\Program Files\a b\d.toml"]),
    (r'py x.py "C:\Program Files\a b\x.txt"',
     ["py", "x.py", r"C:\Program Files\a b\x.txt"]),
    (r'py -c "print(1)"', ["py", "-c", "print(1)"]),
    ("  spaced   out  ", ["spaced", "out"]),
])
def test_tokenize_keeps_windows_paths_intact(command, expected):
    assert wc.tokenize_command(command) == expected


def test_tokenize_refuses_an_unbalanced_quote():
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.tokenize_command('py -c "print(1)')
    assert "one argument" in str(excinfo.value)


def test_batch_target_with_a_metacharacter_is_refused():
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.refuse_batch_metacharacters(
            ["setup.cmd", r"C:\walks\a & b\x.txt"])
    message = str(excinfo.value)
    assert "cmd.exe" in message and "re-parses" in message


def test_batch_target_without_metacharacters_is_allowed():
    wc.refuse_batch_metacharacters(["setup.cmd", r"C:\walks\plain\x.txt"])


def test_a_windows_path_with_shell_characters_reaches_the_child_verbatim(
        tmp_path):
    """The one that would be a silent disaster if it broke.

    ``a & b`` in a directory name is legal on Windows.  Routed through a
    shell it is two commands; routed through argv it is eight
    characters.  The child here prints its own argv, so the assertion is
    on what the process received, not on what the harness intended.
    """

    root = tmp_path / "root"
    root.mkdir()
    echo = _write(root / "echo_argv.py",
                  "import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    hostile = str(tmp_path / "a & b" / "x.txt")
    document = _run(tmp_path, [{
        "id": "argv",
        "intent": "A path with shell metacharacters is just text.",
        "command": '"{}" "{}" "{}"'.format(PY, echo, hostile),
    }])
    step = document["steps"][0]
    assert step["exit_code"] == 0, step["stderr_excerpt"]
    assert json.loads(step["stdout_excerpt"]) == [hostile]
    assert "&" not in step["stderr_excerpt"]


# ------------------------------------------------------------------- capture

def test_capture_records_exit_code_streams_time_and_new_files(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    script = _write(root / "work.py", "\n".join([
        "import pathlib, sys, time",
        "sys.stdout.write('made the thing\\n')",
        "sys.stderr.write('a warning\\n')",
        "pathlib.Path('made.txt').write_text('x', encoding='utf-8')",
        "time.sleep(0.2)",
        "sys.exit(0)",
        "",
    ]))
    document = _run(tmp_path, [{
        "id": "make",
        "intent": "Produce a file the way a real command does.",
        "command": '"{}" "{}"'.format(PY, script),
    }])

    step = document["steps"][0]
    assert step["exit_code"] == 0
    assert step["status"] == "ok"
    assert "made the thing" in step["stdout_excerpt"]
    assert "a warning" in step["stderr_excerpt"]
    assert step["seconds"] >= 0.2
    assert "made.txt" in step["files_created"]
    assert step["files_created_count"] >= 1
    assert step["first_output_seconds"] is not None

    # The streams are on disk in full, not only excerpted in the record.
    assert "made the thing" in Path(step["stdout_path"]).read_text(
        encoding="utf-8")
    assert "a warning" in Path(step["stderr_path"]).read_text(encoding="utf-8")
    assert document["status"] == "clean"


def test_the_capture_json_is_the_record_the_report_reloads(tmp_path):
    document = _run(tmp_path, [{
        "id": "hello",
        "command": '"{}" -c "print(1)"'.format(PY),
    }])
    reloaded = wc.load_capture(tmp_path / "capture")
    assert reloaded["steps"][0]["id"] == "hello"
    assert reloaded["schema"] == wc.SCHEMA_VERSION
    assert reloaded == document


def test_full_output_is_kept_on_disk_when_the_report_excerpt_is_bounded(
        tmp_path):
    document = _run(
        tmp_path,
        [{
            "id": "chatty",
            "command": '"{}" -c "print(\'y\' * 200000)"'.format(PY),
        }],
        settings={"output_bytes_in_report": 4000},
    )
    step = document["steps"][0]
    assert step["stdout_truncated"] is True
    assert "omitted from the middle" in step["stdout_excerpt"]
    assert len(step["stdout_excerpt"]) < 20000
    assert Path(step["stdout_path"]).stat().st_size >= 200000


def test_a_short_stream_is_reproduced_exactly_in_the_excerpt(tmp_path):
    """Head and tail overlap below the limit; the stitch must be lossless."""

    payload = "".join("line {}\n".format(n) for n in range(200))
    script_body = "import sys; sys.stdout.write({!r})".format(payload)
    document = _run(
        tmp_path,
        [{"id": "exact", "argv": [PY, "-c", script_body]}],
        settings={"output_bytes_in_report": 8000},
    )
    step = document["steps"][0]
    assert step["stdout_truncated"] is False
    assert step["stdout_excerpt"].replace("\r\n", "\n") == payload


def test_a_command_that_cannot_start_is_recorded_not_raised(tmp_path):
    document = _run(tmp_path, [{
        "id": "missing",
        "command": "definitely-not-an-installed-program-9f3c --help",
        "allow_failure": True,
    }])
    step = document["steps"][0]
    assert step["status"] == "failed"
    assert step["exit_code"] is None
    assert any(flag["code"] == "not-found" for flag in step["flags"])


def test_stdin_is_closed_so_an_interactive_prompt_cannot_hang_the_walk(
        tmp_path):
    document = _run(tmp_path, [{
        "id": "prompt",
        "argv": [PY, "-c", "input('name? ')"],
        "expect_failure": True,
    }])
    step = document["steps"][0]
    assert step["status"] == "expected-failure"
    assert step["seconds"] < 30


# --------------------------------------------------------- expect_failure

def test_expected_failure_is_not_friction_and_the_walk_continues(tmp_path):
    document = _run(tmp_path, [
        {
            "id": "typo",
            "intent": "The mistake a first-time user makes.",
            "argv": [PY, "-c", "import sys; sys.exit(3)"],
            "expect_failure": True,
        },
        {"id": "after", "argv": [PY, "-c", "print('still going')"]},
    ])
    first, second = document["steps"]
    assert first["exit_code"] == 3
    assert first["status"] == "expected-failure"
    assert first["flags"] == []
    assert second["status"] == "ok"
    assert document["status"] == "clean"
    assert document["aborted"] is False


def test_expect_exit_code_pins_the_exact_code(tmp_path):
    document = _run(tmp_path, [{
        "id": "pinned",
        "argv": [PY, "-c", "import sys; sys.exit(2)"],
        "expect_exit_code": 2,
    }])
    assert document["steps"][0]["status"] == "ok"

    other = _run(tmp_path / "b", [{
        "id": "pinned",
        "argv": [PY, "-c", "import sys; sys.exit(5)"],
        "expect_exit_code": 2,
        "allow_failure": True,
    }])
    step = other["steps"][0]
    assert step["status"] == "failed"
    assert any(flag["code"] == "nonzero" for flag in step["flags"])


def test_a_step_that_was_supposed_to_fail_and_did_not_is_friction(tmp_path):
    document = _run(tmp_path, [{
        "id": "stale",
        "argv": [PY, "-c", "print('it works now')"],
        "expect_failure": True,
        "allow_failure": True,
    }])
    step = document["steps"][0]
    assert step["status"] == "failed"
    codes = {flag["code"] for flag in step["flags"]}
    assert "unexpected-success" in codes
    assert "out of date" in "".join(f["detail"] for f in step["flags"])


def test_an_unexpected_failure_stops_the_walk_and_marks_the_rest_skipped(
        tmp_path):
    document = _run(tmp_path, [
        {"id": "boom", "argv": [PY, "-c", "import sys; sys.exit(1)"]},
        {"id": "never", "argv": [PY, "-c", "print('unreachable')"]},
    ])
    first, second = document["steps"]
    assert first["status"] == "failed"
    assert second["status"] == "skipped"
    assert second["seconds"] == 0.0
    assert document["aborted"] is True
    assert document["status"] == "blocked"
    assert any("stop_on_unexpected_failure" in note
               for note in document["notes"])


def test_keep_going_records_the_whole_experience_after_a_failure(tmp_path):
    document = _run(
        tmp_path,
        [
            {"id": "boom", "argv": [PY, "-c", "import sys; sys.exit(1)"]},
            {"id": "after", "argv": [PY, "-c", "print('kept going')"]},
        ],
        settings={"stop_on_unexpected_failure": False},
    )
    assert [s["status"] for s in document["steps"]] == ["failed", "ok"]
    assert "kept going" in document["steps"][1]["stdout_excerpt"]


def test_allow_failure_lets_one_step_fail_without_ending_the_walk(tmp_path):
    document = _run(tmp_path, [
        {"id": "gap", "argv": [PY, "-c", "import sys; sys.exit(1)"],
         "allow_failure": True},
        {"id": "after", "argv": [PY, "-c", "print('ok')"]},
    ])
    assert [s["status"] for s in document["steps"]] == ["failed", "ok"]
    assert document["aborted"] is False


def test_a_timeout_is_recorded_and_the_child_is_killed(tmp_path):
    document = _run(tmp_path, [{
        "id": "hang",
        "argv": [PY, "-c", "import time; time.sleep(120)"],
        "timeout_seconds": 2,
        "allow_failure": True,
    }])
    step = document["steps"][0]
    assert step["timed_out"] is True
    assert step["status"] == "timeout"
    assert step["seconds"] < 60
    assert any(flag["code"] == "timeout" for flag in step["flags"])


# -------------------------------------------------------------- friction

def test_slow_and_silent_steps_are_flagged_with_the_measured_number(tmp_path):
    document = _run(
        tmp_path,
        [{"id": "quiet", "argv": [PY, "-c", "import time; time.sleep(1.2)"]}],
        settings={"slow_seconds": 0.5, "quiet_seconds": 0.5},
    )
    step = document["steps"][0]
    codes = {flag["code"] for flag in step["flags"]}
    assert codes == {"slow", "silent"}
    assert step["max_silence_seconds"] >= 1.2
    assert document["status"] == "friction"


def test_a_gap_between_two_prints_counts_as_silence(tmp_path):
    document = _run(
        tmp_path,
        [{"id": "gappy", "argv": [
            PY, "-u", "-c",
            "import time; print('start', flush=True); time.sleep(1.0); "
            "print('end', flush=True)"]}],
        settings={"slow_seconds": 60, "quiet_seconds": 0.6},
    )
    step = document["steps"][0]
    assert step["first_output_seconds"] < 0.6
    assert step["max_silence_seconds"] >= 1.0
    assert any(flag["code"] == "silent" for flag in step["flags"])
    assert "gap in output" in "".join(f["label"] for f in step["flags"])


def test_a_traceback_on_stderr_is_flagged_as_a_message_for_us(tmp_path):
    document = _run(tmp_path, [{
        "id": "crash",
        "argv": [PY, "-c", "raise ValueError('boom')"],
        "expect_failure": True,
    }])
    step = document["steps"][0]
    assert step["status"] == "expected-failure"
    codes = {flag["code"] for flag in step["flags"]}
    assert "traceback" in codes


# ---------------------------------------------------------------- reports

def test_time_to_first_plot_is_measured_from_a_png_the_walk_wrote(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    maker = _write(root / "make_png.py", "\n".join([
        "import pathlib, time",
        "time.sleep(0.4)",
        "pathlib.Path('plots').mkdir(exist_ok=True)",
        "pathlib.Path('plots/first.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')",
        "print('rendered 1 frame')",
        "",
    ]))
    document = _run(tmp_path, [
        {"id": "warmup", "argv": [PY, "-c", "import time; time.sleep(0.3)"]},
        {"id": "render", "intent": "Make a picture.",
         "argv": [PY, str(maker)]},
    ])
    assert document["time_to_first_plot_seconds"] is not None
    assert document["time_to_first_plot_seconds"] >= 0.3
    assert document["plots"][0]["path"] == "plots/first.png"
    assert document["plots"][0]["step"] == "render"

    page = wc.render_html(document)
    assert "time to first plot" in page
    summary = wc.render_markdown(document)
    assert "time to first plot" in summary


def test_a_walk_with_no_plots_says_so_rather_than_inventing_a_number(
        tmp_path):
    document = _run(tmp_path, [{"id": "none", "argv": [PY, "-c", "pass"]}])
    assert document["time_to_first_plot_seconds"] is None
    assert "this walk produced no image files" in wc.render_html(document)
    assert "no image files" in wc.render_markdown(document)


# The measured defect (N25, UX persona walks 2026-08-18): a walk that
# produced zero weather images reported "time to first plot: 13.4 s",
# because `pip install` unpacked scipy's test PNGs and matplotlib's
# toolbar icons into a venv under the walk root and the scan counted
# them.  A harness that reports a plot the user never saw is worse than
# one that reports none, so both directions are held here against a real
# venv on disk -- `python -m venv` writes the genuine pyvenv.cfg and
# site-packages layout the rule keys on.

def _seed_installed_package_images(root: Path, venv_name: str = ".venv"):
    """Create a real venv under ``root`` and the payload pip unpacks."""

    target = root / venv_name
    subprocess.run([PY, "-m", "venv", "--without-pip", str(target)],
                   check=True, capture_output=True)
    site = next(p for p in target.rglob("site-packages") if p.is_dir())
    return site


def test_images_unpacked_into_an_installed_package_are_not_plots(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    site = _seed_installed_package_images(root)
    unpack = _write(root / "unpack.py", "\n".join([
        "import pathlib, sys",
        "site = pathlib.Path(sys.argv[1])",
        # matplotlib ships toolbar icons; scipy ships test-fixture PNGs.
        "for rel in ('plotlib/mpl-data/images/home.png',",
        "            'sciencekit/ndimage/tests/data/label.png'):",
        "    out = site / rel",
        "    out.parent.mkdir(parents=True, exist_ok=True)",
        "    out.write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')",
        "print('installed 2 packages')",
        "",
    ]))
    document = _run(tmp_path, [
        {"id": "install", "intent": "Install the wheel.",
         "argv": [PY, str(unpack), str(site)]},
    ])

    assert document["plots"] == []
    assert document["time_to_first_plot_seconds"] is None
    assert "this walk produced no image files" in wc.render_html(document)

    # The files are still tracked -- only the plot metric ignores them.
    created = document["steps"][0]["files_created_count"]
    assert created >= 2


def test_a_real_plot_beats_installed_package_images_to_the_metric(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    site = _seed_installed_package_images(root)
    unpack = _write(root / "unpack.py", "\n".join([
        "import pathlib, sys",
        "out = pathlib.Path(sys.argv[1]) / 'plotlib/mpl-data/images/back.png'",
        "out.parent.mkdir(parents=True, exist_ok=True)",
        "out.write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')",
        "",
    ]))
    maker = _write(root / "make_png.py", "\n".join([
        "import pathlib, time",
        "time.sleep(0.4)",
        "pathlib.Path('png').mkdir(exist_ok=True)",
        "pathlib.Path('png/first.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')",
        "print('rendered 1 frame')",
        "",
    ]))
    document = _run(tmp_path, [
        {"id": "install", "argv": [PY, str(unpack), str(site)]},
        {"id": "render", "argv": [PY, str(maker)]},
    ])

    assert [plot["path"] for plot in document["plots"]] == ["png/first.png"]
    assert document["plots"][0]["step"] == "render"
    assert document["time_to_first_plot_seconds"] >= 0.4


def test_site_packages_outside_a_venv_is_still_installed_payload(tmp_path):
    """`pip install --target` writes site-packages with no pyvenv.cfg."""

    root = tmp_path / "root"
    root.mkdir()
    unpack = _write(root / "unpack.py", "\n".join([
        "import pathlib",
        "for rel in ('vendor/site-packages/kit/img/a.png',",
        "            'vendor/lib/python3.12/dist-packages/kit/b.png'):",
        "    out = pathlib.Path(rel)",
        "    out.parent.mkdir(parents=True, exist_ok=True)",
        "    out.write_bytes(b'\\x89PNG\\r\\n\\x1a\\n')",
        "",
    ]))
    document = _run(tmp_path, [
        {"id": "install", "argv": [PY, str(unpack)]}])

    assert document["plots"] == []
    assert document["time_to_first_plot_seconds"] is None


def test_the_html_report_carries_the_walk_and_is_self_contained(tmp_path):
    document = _run(
        tmp_path,
        [
            {"id": "good", "intent": "Check the install.",
             "argv": [PY, "-c", "print('all clear')"]},
            {"id": "typo", "intent": "Mistype the flag.",
             "argv": [PY, "-c", "import sys; sys.exit(2)"],
             "expect_failure": True},
        ],
        persona="someone who has never installed this before",
        description="The first ten minutes.",
    )
    page = wc.render_html(document)

    assert "<title>gate walk</title>" in page
    assert "someone who has never installed this before" in page
    assert "Check the install." in page and "Mistype the flag." in page
    assert "all clear" in page
    assert "<details>" in page  # output is collapsible
    assert "Timeline" in page and "Friction" in page

    # Self-contained: nothing is fetched from anywhere at open time.
    for forbidden in ("http://", "https://", "<script", "@import", "src="):
        assert forbidden not in page


def test_report_output_is_html_escaped(tmp_path):
    document = _run(tmp_path, [{
        "id": "markup",
        "argv": [PY, "-c", "print('<script>alert(1)</script>')"],
    }])
    page = wc.render_html(document)
    assert "&lt;script&gt;" in page
    assert "<script>alert(1)</script>" not in page


def test_the_markdown_summary_names_every_step_and_the_friction(tmp_path):
    document = _run(
        tmp_path,
        [{"id": "slowpoke", "argv": [PY, "-c", "import time; time.sleep(1.0)"]}],
        settings={"slow_seconds": 0.2, "quiet_seconds": 900},
    )
    summary = wc.render_markdown(document)
    assert "| 1 | `slowpoke` |" in summary
    assert "## Friction" in summary
    assert "time the user spends waiting" in summary


def test_write_reports_then_re_render_from_the_capture_alone(tmp_path):
    _run(tmp_path, [{"id": "one", "argv": [PY, "-c", "print('hi')"]}])
    capture = tmp_path / "capture"
    rc = wc.main(["report", str(capture)])
    assert rc == 0
    assert (capture / "report.html").is_file()
    assert (capture / "report.md").is_file()
    assert "hi" in (capture / "report.html").read_text(encoding="utf-8")


def test_the_writer_does_not_translate_newlines(tmp_path):
    """What is rendered is what is written, byte for byte.

    A captured Windows child prints ``hi\\r\\n``, and that CRLF is part
    of what the user saw, so it belongs in the report verbatim.  What
    must not happen is the *writer* adding carriage returns of its own
    to the page's own structure -- on a CRLF-translating open() every
    embedded ``\\r\\n`` in the captured output becomes ``\\r\\r\\n`` and
    the report grows a blank line between every line the user saw.
    """

    document = _run(tmp_path, [{"id": "one", "argv": [PY, "-c", "print('hi')"]}])
    wc.write_reports(document, html_path=tmp_path / "r.html",
                     md_path=tmp_path / "r.md")
    assert (tmp_path / "r.html").read_bytes() == \
        wc.render_html(document).encode("utf-8")
    assert (tmp_path / "r.md").read_bytes() == \
        wc.render_markdown(document).encode("utf-8")
    assert b"\r\r\n" not in (tmp_path / "r.html").read_bytes()
    assert b"\r\n" not in (tmp_path / "capture" / "capture.json").read_bytes()


# --------------------------------------------------------------- refusals

def test_an_unknown_key_is_refused_by_name(tmp_path):
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.build_walk({"name": "x", "stpes": []})
    message = str(excinfo.value)
    assert "'stpes'" in message
    assert "silently ignored" in message


def test_a_step_with_both_command_and_argv_is_refused(tmp_path):
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.build_walk({"name": "x", "steps": [
            {"id": "a", "command": "one", "argv": ["two"]}]})
    assert "only one of them can be what actually runs" in str(excinfo.value)


def test_duplicate_step_ids_are_refused_because_logs_would_collide(tmp_path):
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.build_walk({"name": "x", "steps": [
            {"id": "a", "command": "one"}, {"id": "a", "command": "two"}]})
    assert "overwrite" in str(excinfo.value)


def test_a_walk_with_no_steps_is_refused(tmp_path):
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.build_walk({"name": "x", "steps": []})
    assert "measures nothing" in str(excinfo.value)


def test_a_capture_directory_inside_the_walk_root_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    walk = wc.build_walk(_walk_document(root, [
        {"id": "a", "argv": [PY, "-c", "pass"]}]),
        script_path=tmp_path / "walk.toml")
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.run_walk(walk, root / "capture")
    message = str(excinfo.value)
    assert "measuring itself" in message


def test_a_bad_command_in_a_later_step_is_refused_before_step_one_runs(
        tmp_path):
    """An authoring mistake must not cost five real commands first."""

    root = tmp_path / "root"
    root.mkdir()
    marker = root / "ran.txt"
    walk = wc.build_walk(_walk_document(root, [
        {"id": "first", "argv": [
            PY, "-c", "open(r'{}', 'w').close()".format(str(marker))]},
        {"id": "second", "command": 'py -c "print(1)'},
    ]), script_path=tmp_path / "walk.toml")
    with pytest.raises(wc.WalkRefusal):
        wc.run_walk(walk, tmp_path / "capture")
    assert not marker.exists()


def test_a_missing_step_cwd_is_refused_by_name(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    walk = wc.build_walk(_walk_document(root, [
        {"id": "a", "argv": [PY, "-c", "pass"], "cwd": "nope"}]),
        script_path=tmp_path / "walk.toml")
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.run_walk(walk, tmp_path / "capture")
    assert "does not exist" in str(excinfo.value)


def test_an_unreadable_capture_schema_is_refused(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "capture.json").write_text(
        json.dumps({"schema": 999}), encoding="utf-8")
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.load_capture(capture)
    assert "mislabel" in str(excinfo.value)


def test_an_unknown_walk_script_suffix_is_refused(tmp_path):
    script = _write(tmp_path / "walk.yaml", "name: x\n")
    with pytest.raises(wc.WalkRefusal) as excinfo:
        wc.load_walk(script)
    assert ".json or .toml" in str(excinfo.value)


# ------------------------------------------------------- script formats

def test_toml_and_json_walk_scripts_describe_the_same_walk(tmp_path):
    toml_script = _write(tmp_path / "walk.toml", "\n".join([
        'name = "two formats"',
        'persona = "a researcher"',
        'root = "{walk_dir}/root"',
        "",
        "[settings]",
        "slow_seconds = 30.0",
        "",
        "[[steps]]",
        'id = "one"',
        'intent = "check"',
        'command = "echo hi"',
        "",
    ]))
    json_script = _write(tmp_path / "walk.json", json.dumps({
        "name": "two formats",
        "persona": "a researcher",
        "root": "{walk_dir}/root",
        "settings": {"slow_seconds": 30.0},
        "steps": [{"id": "one", "intent": "check", "command": "echo hi"}],
    }))
    from_toml = wc.load_walk(toml_script)
    from_json = wc.load_walk(json_script)
    assert from_toml.name == from_json.name
    assert from_toml.persona == from_json.persona
    assert from_toml.root == from_json.root
    assert from_toml.settings.slow_seconds == from_json.settings.slow_seconds
    assert [s.command for s in from_toml.steps] == \
           [s.command for s in from_json.steps]


def test_placeholders_expand_in_commands_env_and_root(tmp_path):
    script = _write(tmp_path / "walk.json", json.dumps({
        "name": "tokens",
        "root": "{walk_dir}/root",
        "env": {"SOMEWHERE": "{root}/data"},
        "steps": [{"id": "one", "command": "{python} -c \"pass\""}],
    }))
    walk = wc.load_walk(script)
    assert walk.root == Path(os.path.normpath(str(tmp_path / "root")))
    assert walk.env["SOMEWHERE"] == str(walk.root) + "/data"
    assert walk.steps[0].command.startswith(sys.executable)


def test_walk_level_env_reaches_the_child_and_env_remove_takes_it_away(
        tmp_path):
    document = _run(
        tmp_path,
        [
            {"id": "present", "argv": [
                PY, "-c",
                "import os; print(os.environ.get('WALK_PROBE', 'absent'))"]},
            {"id": "overridden", "argv": [
                PY, "-c",
                "import os; print(os.environ.get('WALK_PROBE', 'absent'))"],
             "env": {"WALK_PROBE": "per-step"}},
        ],
        env={"WALK_PROBE": "walk-level"},
    )
    assert "walk-level" in document["steps"][0]["stdout_excerpt"]
    assert "per-step" in document["steps"][1]["stdout_excerpt"]


def test_env_remove_hides_a_variable_from_every_step(tmp_path):
    os.environ["WALK_CAPTURE_REMOVE_ME"] = "set"
    try:
        document = _run(
            tmp_path,
            [{"id": "gone", "argv": [
                PY, "-c",
                "import os; print(os.environ.get("
                "'WALK_CAPTURE_REMOVE_ME', 'absent'))"]}],
            env_remove=["WALK_CAPTURE_REMOVE_ME"],
        )
    finally:
        os.environ.pop("WALK_CAPTURE_REMOVE_ME", None)
    assert "absent" in document["steps"][0]["stdout_excerpt"]


# ------------------------------------------------------------- bootstrap

@pytest.mark.slow
def test_a_fresh_venv_bootstrap_is_a_real_step_and_later_steps_use_it(
        tmp_path):
    """The bootstrap is not configuration -- it is a timed, captured step.

    And the step after it must run the venv's interpreter, which on
    Windows only happens because the harness resolves a bare executable
    name against the step's own PATH.
    """

    document = _run(tmp_path, [{
        "id": "which-python",
        "intent": "Whose python am I running?",
        "command": 'python -c "import sys; print(sys.prefix)"',
    }], venv={"path": ".venv", "system_site_packages": True})

    bootstrap, probe = document["steps"]
    assert bootstrap["id"] == "bootstrap-venv"
    assert bootstrap["kind"] == "bootstrap"
    assert bootstrap["status"] == "ok"
    assert bootstrap["seconds"] > 0
    assert probe["status"] == "ok"
    prefix = probe["stdout_excerpt"].strip()
    assert Path(prefix).resolve() == (tmp_path / "root" / ".venv").resolve()
    assert any("first on PATH" in note for note in document["notes"])


# ------------------------------------------------------------------- CLI

def test_validate_prints_the_plan_without_running_anything(tmp_path, capsys):
    root = tmp_path / "root"
    root.mkdir()
    marker = root / "ran.txt"
    script = _write(tmp_path / "walk.json", json.dumps({
        "name": "plan only",
        "root": str(root),
        "steps": [
            {"id": "side-effect", "intent": "must not happen",
             "argv": [PY, "-c",
                      "open(r'{}', 'w').close()".format(str(marker))]},
            {"id": "expected", "command": "gpuwm nope",
             "expect_failure": True},
        ],
    }))
    assert wc.main(["validate", str(script)]) == 0
    printed = capsys.readouterr().out
    assert "plan only" in printed
    assert "must not happen" in printed
    assert "expects failure" in printed
    assert not marker.exists()


def test_run_through_the_cli_writes_a_capture_and_both_reports(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    script = _write(tmp_path / "walk.json", json.dumps({
        "name": "cli walk",
        "root": str(root),
        "steps": [{"id": "one", "argv": [PY, "-c", "print('through the cli')"]}],
    }))
    capture = tmp_path / "capture"
    rc = wc.main(["run", str(script), "--out", str(capture),
                  "--label", "gate"])
    assert rc == 0
    assert (capture / "capture.json").is_file()
    assert (capture / "report.html").is_file()
    assert (capture / "report.md").is_file()
    document = json.loads((capture / "capture.json").read_text(
        encoding="utf-8"))
    assert document["label"] == "gate"
    assert document["status"] == "clean"


def test_the_cli_exit_code_separates_clean_friction_and_refusal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    friction = _write(tmp_path / "friction.json", json.dumps({
        "name": "friction walk",
        "root": str(root),
        "settings": {"stop_on_unexpected_failure": False},
        "steps": [{"id": "boom", "argv": [PY, "-c", "import sys; sys.exit(1)"]}],
    }))
    assert wc.main(["run", str(friction), "--out",
                    str(tmp_path / "c1")]) == 1

    broken = _write(tmp_path / "broken.json", json.dumps({"name": "no steps"}))
    assert wc.main(["run", str(broken), "--out", str(tmp_path / "c2")]) == 2


def test_the_module_runs_as_a_script(tmp_path):
    """Verify against the artifact: the door people actually type."""

    root = tmp_path / "root"
    root.mkdir()
    script = _write(tmp_path / "walk.json", json.dumps({
        "name": "subprocess walk",
        "root": str(root),
        "steps": [{"id": "one", "argv": [PY, "-c", "print('real door')"]}],
    }))
    completed = subprocess.run(
        [PY, "-m", "tools.walk_capture", "run", str(script),
         "--out", str(tmp_path / "capture")],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    assert "outcome: clean" in completed.stdout
    assert (tmp_path / "capture" / "report.html").is_file()

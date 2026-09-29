"""The native renderer's warning lines reach whoever ran the render.

The bridges read the engine's stderr for its ``FAILED`` rows and used to
drop everything else, warnings included, on a clean exit and a failed
one alike.  Progress and machine rows stay quiet.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import rustwx, rustwx_lanes

WARNING = ('warning: the left subtitle does not fit the plot width and was '
           'drawn as "Init 09/27 12Z | +000:15 | Vali..."')
GEOREF = ("WARNING georef manifest out/render-georef.json is unreadable "
          "(EOF); starting a fresh record")


def _engine(monkeypatch, *, returncode, stderr):
    response = SimpleNamespace(
        returncode=returncode,
        stdout="RENDERED 2m_temperature out/2m_temperature.png\n",
        stderr=stderr)
    monkeypatch.setattr(rustwx.subprocess, "run",
                        lambda *args, **kwargs: response)
    monkeypatch.setattr(rustwx, "fit_series_command",
                        lambda command, count, env: (command, None, env))


def _render(series: bool):
    call = rustwx.run_renderer_series if series else rustwx.run_renderer
    inputs = ([Path("wrfout_a"), Path("wrfout_b")] if series
              else Path("wrfout_a"))
    return call(Path("rw_wrfbatch"), inputs, store_root=Path("store"),
                out_dir=Path("out"), products="2m_temperature",
                frames="all", width=1200, height=900)


@pytest.mark.parametrize("series", [False, True])
@pytest.mark.parametrize("returncode", [0, 1])
def test_native_warnings_reach_the_callers_stderr(monkeypatch, capsys,
                                                  series, returncode):
    failure = "FAILED total_qpf wrfout_a: grid is missing\n" if returncode else ""
    _engine(monkeypatch, returncode=returncode,
            stderr=f"IMPORT_NOTE\tuvmet ok\n{WARNING}\n{GEOREF}\n{failure}")

    written, failures, skipped = _render(series)

    captured = capsys.readouterr()
    assert captured.err == f"{WARNING}\n{GEOREF}\n"
    assert captured.out == ""
    assert written == [Path("out/2m_temperature.png")]
    assert len(failures) == returncode
    assert skipped == []


def test_progress_and_machine_rows_stay_quiet(monkeypatch, capsys):
    _engine(monkeypatch, returncode=0,
            stderr="IMPORT_NOTE\tdone\nGENERIC_STYLE\twrf_x\tgeneric fill\n"
                   "TIMING total=5\n")

    _render(False)

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_the_ensemble_lane_relays_the_same_warnings(capsys):
    written, failures, skipped = rustwx_lanes._read_events(
        "RENDERED mean out/mean.png\n",
        f"MEMBERS n=2\n{WARNING}\nFAILED pmm no members\n",
        "ensemble")

    assert capsys.readouterr().err == f"{WARNING}\n"
    assert written == [Path("out/mean.png")]
    assert failures == ["ensemble: pmm no members"]
    assert skipped == []


#: The engine's import note for a 1792x1024x55 frame on a host with 6 GiB
#: available, whose isobaric volumes need more than that.
VOLUME_NOTE = (
    "WRF 3-D pressure-volume products omitted; retained 41 independently "
    "available 2-D products: WRF volume path requires 7853834240 known "
    "owned bytes for 55 levels x 1835008 cells, exceeding the "
    "6442450944-byte host memory ceiling (this process has 6442450944 bytes "
    "available now; no host is held below 4294967296 bytes, 4 GiB), so the "
    "volumes are not built rather than risk the host killing the import "
    "with every picture of the frame")


@pytest.mark.parametrize("series", [False, True])
def test_an_omitted_pressure_volume_is_a_named_skip(monkeypatch, capsys,
                                                    series):
    """The pressure-level products a large frame loses reach the summary.

    A 1132x906x55 frame drew 43 pictures against 67 on an 880x704x55
    frame under the fixed 4 GiB ceiling, and the render summary named
    nothing, because the engine said so only in an import note.  The note stays off the terminal like every
    import note; the products it removed become a skip row carrying it.
    """
    _engine(monkeypatch, returncode=0,
            stderr=f"IMPORT_NOTE\tuvmet ok\nIMPORT_NOTE\t{VOLUME_NOTE}\n")

    written, failures, skipped = _render(series)

    captured = capsys.readouterr()
    assert captured.err == captured.out == ""
    assert written == [Path("out/2m_temperature.png")]
    assert failures == []
    frame = "wrfout_b" if series else "wrfout_a"
    assert skipped == [(rustwx.PRESSURE_LEVEL_FAMILY,
                        f"{frame}: {VOLUME_NOTE}")]


#: The engine's import note when ``build_iso_volumes`` panics: the process
#: layer isolates the panic (``isolate_panics``) into the same omission.
PANIC_NOTE = (
    "WRF 3-D pressure-volume products omitted; retained 41 independently "
    "available 2-D products: panicked computing isobaric volumes: index "
    "out of bounds: the len is 55 but the index is 55")


def test_the_matched_note_is_the_engines_own_wording():
    """The bridge matches the words the engine writes.

    Every omission is a skip, whatever stopped the volumes, except a
    frame that stores no 3-D pressure: it has no pressure-level products
    to draw, so its omission note is not a skip.
    """
    root = Path(__file__).resolve().parents[1] / "tools" / "rustwx"
    src = root / "crates" / "rw-wrfbatch" / "src"
    note = (src / "wrf_process.rs").read_text(encoding="utf-8")
    ceiling = (src / "wrf_volumes.rs").read_text(encoding="utf-8")
    absent = (root / "vendor" / "crates-io" / "wrf-core" / "src"
              / "error.rs").read_text(encoding="utf-8")
    assert f'"{rustwx.PRESSURE_VOLUME_OMITTED_NOTE}; retained ' in note
    assert f"}}-byte {rustwx.PRESSURE_VOLUME_CEILING_CLAUSE} ({{host}}; " in ceiling
    assert 'Err(format!("panicked computing {what}: {message}"))' in note
    assert (f'"{rustwx.PRESSURE_VOLUME_ABSENT_FIELD_CLAUSE} in WRF file: '
            in absent)
    assert rustwx.import_note_skip(VOLUME_NOTE, [Path("f")]) == (
        rustwx.PRESSURE_LEVEL_FAMILY, f"f: {VOLUME_NOTE}")
    assert rustwx.import_note_skip("uvmet ok", [Path("f")]) is None
    no_pressure = (
        "WRF 3-D pressure-volume products omitted; retained 19 "
        "independently available 2-D products: read WRF pressure (sounding "
        "field 1/5): variable not found in WRF file: P")
    assert rustwx.import_note_skip(no_pressure, [Path("f")]) is None


def test_a_panic_in_the_volume_builder_is_a_named_skip():
    """A panic loses the pressure-level products exactly as the ceiling does.

    The bridge matched the ceiling's words, so the same omission caused by
    a panic in ``build_iso_volumes`` drew fewer pictures with nothing in
    the summary to say which.
    """
    assert rustwx.import_note_skip(PANIC_NOTE, [Path("f")]) == (
        rustwx.PRESSURE_LEVEL_FAMILY, f"f: {PANIC_NOTE}")


def test_a_series_files_one_row_per_distinct_omission(monkeypatch, capsys):
    """Every frame of a series writes the note; the summary gets one row.

    The note names no frame, so each frame's copy was filed against the
    last input: a series of N frames over the ceiling recorded N identical
    rows.
    """
    _engine(monkeypatch, returncode=0,
            stderr=(f"IMPORT_NOTE\t{VOLUME_NOTE}\nIMPORT_NOTE\tuvmet ok\n"
                    f"IMPORT_NOTE\t{VOLUME_NOTE}\nIMPORT_NOTE\t{PANIC_NOTE}\n"))

    _written, failures, skipped = _render(True)

    assert capsys.readouterr().err == ""
    assert failures == []
    assert skipped == [
        (rustwx.PRESSURE_LEVEL_FAMILY, f"wrfout_b: {VOLUME_NOTE}"),
        (rustwx.PRESSURE_LEVEL_FAMILY, f"wrfout_b: {PANIC_NOTE}")]


def test_the_skip_note_gives_each_product_its_own_reason():
    """The render's skip note names why each product was skipped.

    Its one line, the line ``gpuwm go`` relays, said every skipped product
    lacked its declared input fields or time window, including a frame too
    large for the pressure-level volume, which carries every field.
    """
    from gpuwm import explain
    from gpuwm.render import skip_notice

    window = "F000: 1-h QPF requires forecast hour >= 1"
    skipped = [(rustwx.PRESSURE_LEVEL_FAMILY, f"wrfout_b: {VOLUME_NOTE}"),
               ("qpf_1h", f"wrfout_a: {window}"),
               ("qpf_1h", "wrfout_b: a later reason")]
    notice = skip_notice(skipped, drawn={"qpf_1h"},
                         sources=("wrfout_a", "wrfout_b"))
    note = explain.render(notice, explain=False, command="gpuwm render")
    first = note.splitlines()[0]
    assert first.startswith("note: render skipped 3 product render(s)")
    assert "do not carry their declared input fields" not in first
    assert (f"{rustwx.PRESSURE_LEVEL_FAMILY}: {VOLUME_NOTE}"
            in first), first
    assert f"qpf_1h: {window}" in first, first
    assert "a later reason" not in first
    # The line gives the reason, not the file it was recorded against;
    # the file stays with every per-item row behind --explain.
    assert "wrfout_a" not in note and "wrfout_b" not in note, note
    explained = explain.render(notice, explain=True, command="gpuwm render")
    for product, detail in skipped:
        assert f"skipped {product}: {detail}" in explained, explained


def test_the_skip_note_takes_off_only_a_prefix_that_is_an_input():
    """Only the invocation's own inputs come off a reason; the rest stays.

    A series verdict opens with its span, the matplotlib engine with the
    frame index, and a storeless term's sentence with no file at all.
    """
    from gpuwm.render import _skip_reason

    sources = ("runs/a/wrfout_d01_0", "runs/a/wrfout_d01_2")
    assert _skip_reason(
        "runs/a/wrfout_d01_0 to wrfout_d01_2 (3 frames): no 6-h window",
        sources) == "no 6-h window"
    assert _skip_reason("runs/a/wrfout_d01_2[1] carries no REFL_10CM, T2",
                        sources) == "the file carries no REFL_10CM, T2"
    assert _skip_reason("runs/a/wrfout_d01_2: F001: a reason.",
                        sources) == "F001: a reason"
    elsewhere = "runs/b/wrfout_d01_0: F000: a reason"
    assert _skip_reason(elsewhere, sources) == elsewhere
    storeless = "a vertical section is cut along a LINE: add --section"
    assert _skip_reason(storeless, sources) == storeless
    assert _skip_reason("   ", sources) == "no reason given"

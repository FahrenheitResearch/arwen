"""``gpuwm render --engine rust`` and ``--pair`` -- CPU-only tests.

The rust-engine tests run against the real vendored renderer binary
(``tools/rustwx``) and honestly skip when it is not built; nothing here
mocks the engine.  The fixture wrfout is written by the project's own
``WrfoutWriter`` with the production global-attribute profile
(``wrf_global_attrs``: Lambert projection, START_DATE, domain
topology), because that is what ``gpuwm run`` writes and what the rust
importer's fail-closed preflight requires.  Its second frame is on the
half hour -- every rust-engine test doubles as the exact-time
(sub-hourly) regression test that the source renderer refused.

PNG dimensions are read from the IHDR chunk directly so the assertions
add no image-library dependency.
"""

from __future__ import annotations

import datetime
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.cli as cli
from gpuwm import render_layout, run_stamp, rustwx
from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs
from gpuwm.render import parse_products_rust, parse_size

_NZ, _NY, _NX = 4, 12, 16
_STAMPS = ("1974-04-03_18:00:00", "1974-04-03_18:30:00")
_LEADS = ("lead_000h00m00s", "lead_000h30m00s")

RENDERER = rustwx.find_renderer()


def _renderer_gate() -> tuple[bool, str]:
    """(usable, why not) for THIS tree's renderer -- contract, not stat().

    Task #106.  The gate used to be ``find_renderer() is None``, which is
    a question about a filename.  The resolution ladder's last rung is
    ``~/.gpuwm/bridges``, so on any box that has ever installed a bundle
    the answer was yes -- and the binary answering it was whichever build
    happened to be staged there.  A stale one turned this suite red on a
    correct tree (catalog 153 against 168, zero generic rows) and would
    just as readily have turned a broken tree green, because none of
    these tests can tell which engine drew the plots.

    So the gate asks the contract question instead, and it asks it
    through :func:`gpuwm.render.renderer_refusal` -- the one function the
    render path itself reads, covering both the
    :func:`gpuwm.rustwx.probe_renderer` ``--abi`` handshake ``gpuwm
    doctor`` reports and the provenance clause that catches a sibling
    checkout answering the same contract correctly.  Reading the
    product's own gate rather than re-deriving it is the point: a gate
    the suite defines for itself can admit a renderer the render path
    then refuses, and the suite would report that as a failure of the
    code under test.  A foreign engine SKIPS with its mismatch named
    rather than silently standing in for this tree's.
    """

    if RENDERER is None:
        return False, ("rust renderer not built (cd tools/rustwx && cargo "
                       "build --release --locked --offline)")
    from gpuwm.render import renderer_refusal

    reason = renderer_refusal(RENDERER)
    if reason is None:
        return True, ""
    return False, (
        f"the renderer resolved at {RENDERER} is not this tree's: "
        f"{reason}.  These tests assert THIS tree's product catalog, so "
        "a foreign engine is skipped, not substituted")


_RENDERER_USABLE, _RENDERER_SKIP_REASON = _renderer_gate()
needs_renderer = pytest.mark.skipif(
    not _RENDERER_USABLE, reason=_RENDERER_SKIP_REASON)


def _checkout_build() -> Path | None:
    """THIS checkout's own rw_wrfbatch, or None if it is not built.

    Deliberately not :func:`gpuwm.rustwx.find_renderer`, whose ladder
    ends at ``~/.gpuwm/bridges``: the marker test below is about drift
    between this tree's Rust and this tree's Python, and a binary
    installed from some other release cannot answer that question either
    way.  Whoever has a stale bundle staged learns it from ``gpuwm
    doctor``, which now reports it with the rebuild remedy; they do not
    learn it from a red in a suite they cannot fix by editing code.
    """

    from gpuwm.bridges import executable_name

    built = (rustwx.crate_dir() / "target" / "release"
             / executable_name(rustwx.RENDERER_NAME))
    return built if built.is_file() else None


def test_the_pinned_abi_marker_is_the_built_renderer_s_own_answer():
    """The Python constant against the Rust writer, not against itself.

    A marker both halves read from one Python string proves nothing.
    This runs the binary THIS checkout built and compares its stdout to
    :data:`gpuwm.rustwx.RENDERER_ABI_MARKER` byte for byte, so editing
    the contract on one side and not the other is red rather than a
    silent skip of the whole rust lane.

    Skipped only when this checkout has no build at all -- and that skip
    cannot hide a mismatch, because there is nothing of this tree's to
    mismatch with.
    """

    built = _checkout_build()
    if built is None:
        pytest.skip("this checkout's rust renderer is not built (cd "
                    "tools/rustwx && cargo build --release --locked "
                    "--offline)")
    import subprocess

    probe = subprocess.run([str(built), "--abi"], capture_output=True,
                           text=True, errors="replace", timeout=60)
    assert probe.returncode == 0, (
        f"{built} --abi exited {probe.returncode}: "
        f"{(probe.stderr or '').strip()!r} -- this build predates the "
        "renderer contract handshake; rebuild it from this checkout")
    assert probe.stdout.strip() == rustwx.RENDERER_ABI_MARKER, (
        f"{built} answers a different render contract than "
        "gpuwm.rustwx.RENDERER_ABI_MARKER pins")


def test_every_bundled_rustwx_binary_pins_a_contract_marker():
    """The renderer was the odd one out; nothing may be the odd one again.

    ``rw_fetch`` and ``rw_nexrad`` have always pinned an exact ``--abi``
    line in their Python wrapper.  ``rw_wrfbatch`` did not, which is how
    two builds with different md5s both reported ``verified``.  This
    holds the three together structurally, so the next binary added to
    the workspace cannot arrive without a contract to check it against.
    """

    from gpuwm import rustwx_fetch
    from gpuwm.obs import nexrad

    markers = {
        "rw_fetch": rustwx_fetch.FETCH_ABI_MARKER,
        "rw_nexrad": nexrad.NEXRAD_ABI_MARKER,
        "rw_wrfbatch": rustwx.RENDERER_ABI_MARKER,
    }
    for name, marker in markers.items():
        assert isinstance(marker, str) and marker.strip(), name
        assert "\t" in marker, (
            f"{name}'s marker is not a tab-separated contract line: "
            f"{marker!r}")
    assert len(set(markers.values())) == len(markers), (
        "two bundled binaries pin the SAME marker, so one of them would "
        f"verify against the other's contract: {markers}")


# ---- the probe's decision table -------------------------------------------
#
# Against the real binary above; here against the two answers a stale
# build gives, which cannot be produced without shipping a stale binary.
# What is stubbed is the subprocess layer, never the renderer's meaning:
# these assert what probe_renderer DECIDES, given each observable.

def _stub_probe(monkeypatch, *, abi_returncode, abi_stdout):
    from gpuwm import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))

    def run(command, **_kwargs):
        if command[1] == "--help":
            return SimpleNamespace(
                returncode=0, stdout="usage: rw_wrfbatch --store-root DIR",
                stderr="")
        assert command[1] == "--abi", command
        return SimpleNamespace(returncode=abi_returncode, stdout=abi_stdout,
                               stderr="")

    monkeypatch.setattr(rustwx.subprocess, "run", run)


def test_a_binary_that_launches_but_predates_the_handshake_is_refused(
        monkeypatch, tmp_path):
    """`unknown option --abi`, exit 2: every build older than the contract."""

    _stub_probe(monkeypatch, abi_returncode=2, abi_stdout="")
    ok, evidence = rustwx.probe_renderer(tmp_path / "rw_wrfbatch.exe")
    assert ok is False
    assert "--abi does not match the render contract" in evidence
    assert "REBUILD" in evidence


def test_a_binary_answering_another_contract_is_refused(monkeypatch,
                                                        tmp_path):
    """A build that answers --abi, with somebody else's grammar."""

    _stub_probe(monkeypatch, abi_returncode=0,
                abi_stdout="gpuwm-rw-wrfbatch-catalog-v0\tPRODUCT\tslug\n")
    ok, evidence = rustwx.probe_renderer(tmp_path / "rw_wrfbatch.exe")
    assert ok is False
    assert "gpuwm-rw-wrfbatch-catalog-v0" in evidence


def test_the_probe_accepts_the_contract_it_pins(monkeypatch, tmp_path):
    """The negative test: the guard does NOT fire on the right answer."""

    _stub_probe(monkeypatch, abi_returncode=0,
                abi_stdout=rustwx.RENDERER_ABI_MARKER + "\n")
    ok, evidence = rustwx.probe_renderer(tmp_path / "rw_wrfbatch.exe")
    assert ok is True, evidence
    assert "--abi matches the render contract" in evidence


def _frame(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    return {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": rng.uniform(-20.0, 65.0,
                                 (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "Q2": rng.uniform(0.004, 0.012, (_NY, _NX)).astype(np.float32),
        "PSFC": rng.uniform(96000.0, 98000.0,
                            (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.zeros((_NY, _NX), np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


def _write_wrfout(path, stamps, *, grid_id=2, dx=1000.0):
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=grid_id,
        parent_id=max(grid_id - 1, 1), i_parent_start=5, j_parent_start=5,
        parent_grid_ratio=3, dt=6.0)
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=dx, dy=dx,
                      global_attrs=attrs) as writer:
        for index, stamp in enumerate(stamps):
            writer.write_frame(stamp, _frame(seed=7 + index))
    return path


@pytest.fixture(scope="module")
def wrfout(tmp_path_factory):
    """Two frames, the second on the half hour (exact-time axis)."""

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust") /
        "wrfout_d02_1974-04-03_18-00-00.nc", _STAMPS)


@pytest.fixture(scope="module")
def wrfout_hourly(tmp_path_factory):
    """Two WHOLE-hour frames -- the windowed lane's admissible axis."""

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust-hourly") /
        "wrfout_d02_1974-04-03_18-00-00.nc",
        ("1974-04-03_18:00:00", "1974-04-03_19:00:00"))


@pytest.fixture(scope="module")
def wrfout_hourly_d03(tmp_path_factory):
    """The SAME init and the same whole-hour frames, one nest down.

    Sub-kilometre spacing (333 m), so its resolution token exercises the
    integer-metre branch while its lead/valid times are identical to
    ``wrfout_hourly``'s -- everything the old filename carried.
    """

    return _write_wrfout(
        tmp_path_factory.mktemp("render-rust-hourly-d03") /
        "wrfout_d03_1974-04-03_18-00-00.nc",
        ("1974-04-03_18:00:00", "1974-04-03_19:00:00"),
        grid_id=3, dx=333.3333)


def _png_size(path) -> tuple[int, int]:
    """(width, height) from the PNG IHDR chunk; no image library.

    Read through ``render_layout.fs_path`` for the same reason the
    placement seam writes through it: ``--out`` plus a run folder plus
    ``<domain>/<product>/<valid-day>`` plus an exact-time product name
    passes Windows' MAX_PATH from an ordinary temp root, and a reader
    that is less long-path aware than the writer reports a correctly
    filed frame as missing.
    """

    data = Path(render_layout.fs_path(path)).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


@needs_renderer
def test_rust_engine_renders_every_frame_including_half_hourly(
        wrfout, tmp_path, capsys):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--out", str(out)])
    assert rc == 0
    assert "render: engine rust" in capsys.readouterr().out
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert produced, "the rust engine wrote no PNGs"
    # Both frames rendered -- the :30 frame is the sub-hourly lead the
    # source renderer refused (exact-time ordinal axis).
    for lead in _LEADS:
        matching = [name for name in produced if lead in name]
        assert matching, (lead, produced)
    # 'all' renders the catalog the fixture's fields prove out, which
    # must include the reflectivity composite at minimum.
    assert any("composite_reflectivity" in name for name in produced)
    for png in out.rglob("*.png"):
        assert Path(render_layout.fs_path(png)).stat().st_size > 5_000, (
            png.name)
        width, height = _png_size(png)
        assert (width, height) == (1200, 900), png.name


@needs_renderer
def test_rust_engine_timeidx_selects_the_half_hourly_frame(
        wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl", "--timeidx", "1",
                   "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.rglob("*.png")]
    assert len(produced) == 1, produced
    assert "composite_reflectivity" in produced[0]
    assert _LEADS[1] in produced[0], produced
    assert _LEADS[0] not in produced[0]


@needs_renderer
def test_rust_engine_maps_shared_product_names(wrfout, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl,t2", "--timeidx", "0",
                   "--size", "800x600", "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert len(produced) == 2, produced
    assert any("composite_reflectivity" in name for name in produced)
    assert any("2m_temperature" in name for name in produced)
    for png in out.rglob("*.png"):
        assert _png_size(png) == (800, 600)


@needs_renderer
def test_wind10_renders_the_wind_not_a_pressure_chart(wrfout, tmp_path):
    """``--products wind10`` must draw the wind.

    It did not.  ``RUST_PRODUCT_ALIASES["wind10"]`` pointed at
    ``mslp_10m_winds`` -- a mean-sea-level PRESSURE analysis with 10 m
    barbs over it -- under a source comment calling that "the catalog's
    standalone surface-wind product".  Asking the tuned engine for wind
    returned pressure, so during a wildfire an operator went to the
    matplotlib fallback (whose own ``wind10`` IS a wind map) to get one.

    A test comparing ``RUST_PRODUCT_ALIASES["wind10"]`` to a hardcoded
    string would have passed on the broken code, which is why this one
    goes to the artifact instead: render it, and separately ask the
    renderer's own catalog what that product FILLS with.
    """

    from gpuwm import render as render_module

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "wind10", "--timeidx", "0",
                   "--size", "800x600", "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert len(produced) == 1, produced
    assert "10m_wind_speed_and_direction" in produced[0], produced
    assert "mslp" not in produced[0], (
        "wind10 must not return a mean-sea-level pressure chart")

    rows, _summary = rustwx.list_products(
        RENDERER, wrfout, store_root=tmp_path / "store")
    detail = {slug: text for slug, _kind, _status, text in rows}
    status = {slug: state for slug, _kind, state, _text in rows}

    # What each shared name actually DRAWS, in the renderer's own words.
    # A renderable row states its fill as "[fill: <selector>]"; a row the
    # store cannot serve names the same selector as the field it lacks.
    # Either way the answer to "what is this a chart OF?" is in the row.
    for alias, selector_key in (
            render_module.RUST_PRODUCT_ALIAS_FILL_SELECTORS.items()):
        slug = render_module.RUST_PRODUCT_ALIASES[alias]
        assert slug in detail, f"{alias} -> {slug} is not in the catalog"
        assert selector_key in detail[slug], (
            f"--products {alias} draws {detail[slug]!r}, which is not "
            f"a chart of {selector_key}")

    # The wind one must additionally be renderable from this store --
    # three instantaneous 10 m planes, one frame, no neighbours.
    wind_slug = render_module.RUST_PRODUCT_ALIASES["wind10"]
    assert status[wind_slug] == "renderable", detail[wind_slug]

    # And the product it must never be confused with again: whatever
    # `mslp_10m_winds` fills with, it is pressure, and it is not wind10.
    assert "pressure_reduced_to_mean_sea_level" in detail["mslp_10m_winds"]
    assert wind_slug != "mslp_10m_winds"


# ---------------------------------------------------------------------------
# Domain + resolution tokens: two domains at one lead must not collide
# ---------------------------------------------------------------------------

@needs_renderer
def test_two_domains_at_one_lead_do_not_overwrite_each_other(
        wrfout_hourly, wrfout_hourly_d03, tmp_path):
    """The data-loss bug this token exists to prevent.

    Two nests of one run share model, init cycle, and forecast hour.  On
    the whole-hour axis nothing else distinguished them, so the second
    domain's PNG landed on the first domain's path and one of the two
    forecasts vanished with no error, no warning, and an exit code of 0.
    """

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout_hourly), str(wrfout_hourly_d03),
                   "--engine", "rust", "--products", "refl",
                   "--timeidx", "0", "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert len(produced) == 2, (
        "two domains at one lead collapsed to one file", produced)
    assert any("_d02-1km_" in name for name in produced), produced
    assert any("_d03-333m_" in name for name in produced), produced
    # Both survive as real images, not one truncated overwrite.
    for png in out.rglob("*.png"):
        assert png.stat().st_size > 5_000, png.name


@needs_renderer
def test_domain_and_resolution_token_on_the_exact_time_axis(
        wrfout, tmp_path):
    """The sub-hourly axis carries the same token as the whole-hour one."""

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert len(produced) == 2, produced
    for name in produced:
        assert "_d02-1km_" in name, name
        assert "native_grid" not in name, name


@needs_renderer
def test_unknown_domain_identity_keeps_the_native_grid_token(
        tmp_path):
    """No GRID_ID, no wrfout_dNN name: the token degrades, never guesses.

    A file whose domain identity cannot be established keeps the
    renderer's generic ``native_grid`` slug rather than being labelled
    ``d01`` on no evidence.
    """

    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(grid, datetime.datetime(1974, 4, 3, 18),
                             dt=6.0)
    assert "GRID_ID" not in attrs
    path = tmp_path / "some_model_output.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0,
                      dy=1000.0, global_attrs=attrs) as writer:
        writer.write_frame(_STAMPS[0], _frame(seed=21))
    out = tmp_path / "png"
    rc = cli.main(["render", str(path), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.rglob("*.png")]
    assert produced, "the anonymous-domain file rendered nothing"
    for name in produced:
        assert "native_grid" in name, name


def test_the_engines_skip_line_is_read_and_is_not_a_failure(monkeypatch,
                                                            tmp_path):
    """`rw_wrfbatch` has always emitted three verdicts; two were read.

    A product whose stored fields are absent is `SKIPPED <slug>
    <reason>` on stdout and is NOT counted in `summary.failed`, so the
    process still exits 0.  Reading only RENDERED/FAILED made that
    verdict invisible to every caller, which is why a render could
    quietly draw one image fewer than the catalog and say nothing.

    Third arm is the negative control: FAILED is still a failure.
    """

    import subprocess

    class Result:
        returncode = 0
        stdout = "\n".join((
            "RENDERED t2 /tmp/png/t2.png",
            "SKIPPED refl store carries no REFL_10CM",
            ""))
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    written, failures, skipped = rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d01_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600)
    assert [p.name for p in written] == ["t2.png"]
    assert failures == []
    assert len(skipped) == 1
    product, detail = skipped[0]
    assert product == "refl"
    assert "store carries no REFL_10CM" in detail

    class Failed:
        returncode = 1
        stdout = "RENDERED t2 /tmp/png/t2.png\n"
        stderr = "FAILED refl contour engine panicked\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Failed())
    _written, failures, skipped = rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d01_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600)
    assert skipped == []
    assert len(failures) == 1
    assert "contour engine panicked" in failures[0]


def test_source_label_reaches_the_renderer_invocation(monkeypatch,
                                                      tmp_path):
    """A locally imported run is not a GDEX fetch, and a stock-WRF file
    is not ours: both are one flag away from being labelled honestly."""

    import subprocess

    from gpuwm import render as render_module

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def spy(command, **kwargs):
        seen.append([str(part) for part in command])
        return Result()

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr("gpuwm.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    monkeypatch.setattr("gpuwm.rustwx.probe_renderer",
                        lambda path: (True, "stubbed"))
    # A stub under tmp_path belongs to no tree, so the bridge gate would
    # refuse it -- correctly.  Declaring it through the override is the
    # documented way to say "this binary is the one I mean", and is what
    # a person pointing gpuwm at a hand-built renderer does.
    monkeypatch.setenv(rustwx.RENDERER_ENV,
                       str((tmp_path / "rw_wrfbatch.exe").resolve()))

    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="composite_reflectivity",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600))
    assert seen, "no renderer invocation was made"
    command = seen[-1]
    assert "--source-label" in command
    # The default label is the brand plus the EXECUTING version, and it
    # is asked for rather than transcribed so the assertion survives a
    # release cut.
    assert (command[command.index("--source-label") + 1]
            == render_module.default_source_label())
    assert command[command.index("--source-label") + 1].startswith("ArWen ")

    seen.clear()
    render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="composite_reflectivity",
        timeidx=0, outdir=tmp_path / "png", size=(800, 600),
        source_label="WRF-ARW 4.6.1")
    command = seen[-1]
    assert command[command.index("--source-label") + 1] == "WRF-ARW 4.6.1"


def test_engine_outputs_are_rebranded_to_the_product_prefix(monkeypatch,
                                                            tmp_path,
                                                            capsys):
    """rustwx_* out of the engine becomes arwen_* out of `gpuwm render`.

    The vendored engine hardcodes ``rustwx_`` into every output
    filename (rustwx-products' format strings); the product shipping
    those files is ArWen, and the field run's first question about its
    own pictures was what "rustwx" meant.  The rename happens at the
    one seam every wheel render flows through, so the engine binary
    stays byte-identical to its campaign builds.  Filenames only:
    a name the engine did not brand passes through untouched, and a
    RENDERED line naming a file that is not on disk keeps its reported
    path rather than failing the render over branding.
    """

    import subprocess

    from gpuwm import render as render_module

    outdir = tmp_path / "png"
    branded = "rustwx_wrf_19740403_12z_f003_d02-3km_sbcape.png"
    ghost = "rustwx_wrf_19740403_12z_f003_d02-3km_ghost.png"

    def fake_renderer(command, **kwargs):
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / branded).write_bytes(b"\x89PNG")
        (out / "unbranded_extra.png").write_bytes(b"\x89PNG")

        class Result:
            returncode = 0
            stdout = (f"RENDERED sbcape {out / branded}\n"
                      f"RENDERED extra {out / 'unbranded_extra.png'}\n"
                      f"RENDERED ghost {out / ghost}\n")
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_renderer)
    monkeypatch.setattr("gpuwm.rustwx.find_renderer",
                        lambda: tmp_path / "rw_wrfbatch.exe")
    # The renderer gate, satisfied both ways, exactly as the source-label
    # test above does it: the CONTRACT half is stubbed because this
    # tmp_path stub is a name and not an executable, and the PROVENANCE
    # half is declared because a stub under tmp_path belongs to no tree
    # and the bridge gate is right to refuse one nobody named.  What this
    # test pins is the rebranding of the engine's output, not which
    # engine is admitted.
    monkeypatch.setattr("gpuwm.rustwx.probe_renderer",
                        lambda path: (True, "stubbed"))
    monkeypatch.setenv(rustwx.RENDERER_ENV,
                       str((tmp_path / "rw_wrfbatch.exe").resolve()))

    written, failures, _skipped = render_module.render_wrfouts_rust(
        [tmp_path / "wrfout_d02_x.nc"], products="sbcape", timeidx=0,
        outdir=outdir, size=(800, 600))
    assert failures == []
    assert sorted(p.name for p in written) == [
        "arwen_wrf_19740403_12z_f003_d02-3km_sbcape.png",
        ghost,
        "unbranded_extra.png",
    ]
    # On disk: the branded file moved and nothing else changed.  Since
    # 2.5.0 it moves TWICE -- rebranded, then filed under the render
    # layout (gpuwm.render_layout) -- and a name the engine's grammar
    # does not produce is left exactly where the engine put it, because
    # nothing here knows its product or its valid time.
    assert (outdir / "d02-3km" / "sbcape" / "1974-04-03"
            / "arwen_wrf_19740403_12z_f003_d02-3km_sbcape.png").is_file()
    assert not (outdir / branded).exists()
    assert (outdir / "unbranded_extra.png").is_file()
    # The per-file console lines name what is actually on disk.
    transcript = capsys.readouterr().out
    assert "arwen_wrf_19740403_12z_f003_d02-3km_sbcape.png" in transcript
    assert branded not in transcript


@needs_renderer
def test_rust_engine_unknown_slug_fails_loudly(wrfout, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "definitely_not_a_product",
                   "--out", str(tmp_path / "png")])
    # rc 1 is `gpuwm render`'s, not the renderer's.  rw_wrfbatch exits 2 on a
    # bad command line -- matching what matplotlib's engine costs for the same
    # typo -- but gpuwm.rustwx collects any nonzero exit as a render failure
    # and gpuwm.cli reports failures as 1, so the two engines agree here
    # whichever one runs.  The pin is on the CLI's contract, so it is
    # unchanged by the renderer's exit code.
    assert rc == 1
    assert not list((tmp_path / "png").rglob("*.png"))
    # ...and the reason reaches the user.  gpuwm.rustwx surfaces the LAST
    # non-empty stderr line, which the renderer used to make the usage line:
    # every arg mistake reported the same unactionable sentence.
    reported = capsys.readouterr()
    transcript = reported.out + reported.err
    assert "definitely_not_a_product" in transcript, transcript
    assert "--list-products" in transcript, transcript
    assert not transcript.rstrip().endswith("wrfout..."), (
        "the usage line is back as the reported cause", transcript)


def test_engine_rust_unbuilt_is_a_documented_refusal(
        wrfout, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: None)
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--out", str(tmp_path / "png")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not built" in err
    assert "cargo build --release --locked --offline" in err


def test_engine_auto_refuses_when_the_renderer_is_not_built(
        wrfout, tmp_path, monkeypatch, capsys):
    """`auto` used to degrade here; the render law says it may not.

    Weather-field product plots come from ``rw_wrfbatch`` (CLAUDE.md,
    Drew 2026-08-06), and the one permitted fallback --
    ``da_nowcast_render.py`` -- draws none of this door's products.  So
    with nothing staged the answer is a refusal naming the artifact and
    the staging remedy, at a nonzero exit; drawing five matplotlib
    weather fields and reporting success is the defect (audit F7).
    """

    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: None)
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--products", "t2",
                   "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "rw_wrfbatch" in err and "fetch-bridges" in err
    assert not out.exists() or not list(out.rglob("*.png"))


def test_engine_matplotlib_is_reachable_only_by_name(
        wrfout, tmp_path, capsys):
    """And says WORKAROUND every time it draws."""

    pytest.importorskip(
        "wrf", reason="the matplotlib workaround needs the render extra")
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "matplotlib",
                   "--products", "t2", "--out", str(out)])
    assert rc == 0
    printed = capsys.readouterr()
    assert "render: engine matplotlib" in printed.out
    assert "WORKAROUND:" in printed.err
    assert list(out.rglob("t2_*.png"))


# --------------------------------------------------------------------------
# Task #106, the half `--engine auto` did not cover: an EXPLICIT engine
# request must pass the same contract, or the one caller who pinned
# `--engine rust` to be sure of the real renderer is the one caller who
# still gets a foreign one.
# --------------------------------------------------------------------------

def _stage_foreign_renderer(monkeypatch, tmp_path):
    """A renderer that resolves and fails the ``--abi`` contract.

    The subprocess handshake itself is proved against real binaries --
    ``test_the_pinned_abi_marker_is_the_built_renderer_s_own_answer``
    runs this checkout's build, and a foreign build was run by hand for
    the incident.  What is pinned here is what the CALLERS do with a
    failing probe, which must be assertable on a clean checkout with no
    build staged, since that is where a regression would land.
    """

    staged = tmp_path / "bridges" / rustwx.executable_name(
        rustwx.RENDERER_NAME)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(b"MZ not this tree's build")
    evidence = ("launches, but --abi does not match the render contract "
                "this gpuwm expects (exit 2 with no --abi line) -- it is a "
                "build from another checkout")
    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: staged)
    monkeypatch.setattr("gpuwm.rustwx.probe_renderer",
                        lambda path: (False, evidence))
    return staged, evidence


def test_engine_rust_refuses_a_renderer_that_fails_the_contract(
        wrfout, tmp_path, monkeypatch, capsys):
    staged, _ = _stage_foreign_renderer(monkeypatch, tmp_path)
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "t2", "--out", str(out)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--engine rust" in err
    assert str(staged) in err
    assert "--abi does not match the render contract" in err
    # An explicit request is a statement about which engine must draw:
    # no silent substitution, and no fallback either.
    assert not out.exists() or not list(out.rglob("*.png"))
    # ... and the exit is named (1.8.8 refusal sweep).  Refusing to
    # choose for the caller is not the same as refusing to tell them
    # what the choices are.  `--engine auto` is no longer one of those
    # choices: it refuses this same renderer now, because degrading to
    # matplotlib weather fields is what the render law forbids.
    assert "cargo build --release" in err
    assert "gpuwm fetch-bridges" in err
    assert "--engine matplotlib to take the named workaround" in err
    assert "fall back" not in err


def test_a_missing_renderer_override_names_the_three_ways_out(
        tmp_path, monkeypatch, capsys):
    """The other dead end on the same door: the variable names nothing.

    ``GPUWM_RW_WRFBATCH`` pointing at a path that does not exist used to
    refuse with the variable, the path, and nothing else -- no
    suggestion to repoint it, to unset it and take the vendored ladder,
    or to ask for the fallback engine.  A reader who inherited that
    variable from a shell profile had a diagnosis and no instruction.
    """

    missing = tmp_path / "nonexistent-renderer.exe"
    monkeypatch.setenv(rustwx.RENDERER_ENV, str(missing))
    assert cli.main(["render", "--list-products", "--engine", "rust"]) == 2

    err = capsys.readouterr().err
    assert "names a missing file" in err
    assert str(missing) in err
    assert "Point it at a built rw_wrfbatch binary" in err
    assert f"unset {rustwx.RENDERER_ENV}" in err
    assert "--engine matplotlib" in err


def test_engine_auto_refuses_the_same_renderer_engine_rust_refuses(
        wrfout, tmp_path, monkeypatch, capsys):
    """Same staged renderer, two request forms, ONE outcome now.

    This is the pair the original defect was invisible to: with a stale
    bridge staged, ``auto`` fell back naming the mismatch while ``rust``
    ran the stale build without a word.  The contract check closed the
    second half; the render law closes the first, because "fell back"
    meant matplotlib drew the weather fields.  What must stay asserted
    is that both forms still NAME the mismatch -- a refusal that hides
    which binary was rejected is the outage the probe exists to prevent.
    """

    _stage_foreign_renderer(monkeypatch, tmp_path)
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "auto",
                   "--products", "t2", "--out", str(out)])
    assert rc == 2
    printed = capsys.readouterr()
    assert "--abi does not match the render contract" in (
        printed.out + printed.err)
    assert not out.exists() or not list(out.rglob("*.png"))


def test_engine_rust_accepts_the_renderer_that_passes_the_contract(
        tmp_path, monkeypatch):
    """The other direction: a passing probe still resolves to rust.

    A refusal that fires on everything is not a contract check, it is an
    outage.  Both request forms take the rust path on a probe that
    passes, and neither consults anything else to decide.
    """

    from gpuwm.render import _resolve_engine

    staged = tmp_path / rustwx.executable_name(rustwx.RENDERER_NAME)
    staged.write_bytes(b"MZ this tree's build")
    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: staged)
    # The renderer gate has TWO clauses since 1.8.8 and this test is
    # about the first one, so the second is satisfied here the same way
    # the rebranding sibling satisfies it: the CONTRACT half is stubbed
    # passing, and the PROVENANCE half is stubbed silent because a
    # binary under tmp_path belongs to no tree and the bridge gate is
    # right to refuse one nobody named.  Left unstubbed, whether this
    # test passes depends on how the RUNNER was installed -- a wheel or
    # a non-git source root makes bridge_tree_match unanswerable and the
    # test green, an editable checkout makes the same tmp_path binary
    # foreign and the test red -- which is a property of the box, not of
    # the code under test.
    monkeypatch.setattr("gpuwm.rustwx.probe_renderer",
                        lambda path: (True, "--abi matches"))
    monkeypatch.setattr("gpuwm.provenance_gate.renderer_bridge_refusal",
                        lambda bridge, **kwargs: None)
    assert _resolve_engine("rust") == ("rust", str(staged))
    assert _resolve_engine("auto") == ("rust", str(staged))

    monkeypatch.setattr("gpuwm.rustwx.probe_renderer",
                        lambda path: (False, "stale"))
    # Both request forms refuse a failing contract now, and both name
    # the evidence.  `auto` used to answer ("matplotlib", ...) here.
    with pytest.raises(RuntimeError, match="stale"):
        _resolve_engine("auto")
    with pytest.raises(RuntimeError, match="stale"):
        _resolve_engine("rust")


def test_the_product_catalog_refuses_an_engine_it_cannot_verify(
        tmp_path, monkeypatch, capsys):
    """``--list-products`` with no wrfout resolves the engine of its own.

    It is the one entry point that reaches ``_resolve_engine`` outside
    the render path's try, so the refusal has to be caught here too or a
    documented refusal arrives as a traceback.
    """

    _stage_foreign_renderer(monkeypatch, tmp_path)
    assert cli.main(["render", "--list-products", "--engine", "rust"]) == 2
    assert "--abi does not match the render contract" in (
        capsys.readouterr().err)


def test_both_engine_resolvers_treat_an_explicit_request_the_same(
        tmp_path, monkeypatch):
    """The renderer was the odd one out here too; hold the pair together.

    ``gpuwm fetch`` has always resolved ``--engine rust`` by probing and
    then refusing an explicit request while degrading an automatic one.
    The render resolver skipped the probe for an explicit request, which
    is the whole defect.  Asserting both in one place is what makes the
    next divergence a red rather than a discovery in the field.

    The two doors diverge on ONE point deliberately, and it is a law and
    not an oversight: ``fetch --engine auto`` still degrades to the
    Python engine, because moving bytes on Python is a workaround, while
    ``render --engine auto`` refuses, because DRAWING a weather field on
    matplotlib is not permitted at all (the render law, CLAUDE.md, Drew
    2026-08-06).
    """

    from gpuwm import fetch as fetch_module
    from gpuwm.render import _resolve_engine

    # render: staged binary, failing contract.  Both forms refuse.
    _stage_foreign_renderer(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError):
        _resolve_engine("rust")
    with pytest.raises(RuntimeError):
        _resolve_engine("auto")

    # fetch: staged backbone, failing contract, same two answers.
    backbone = tmp_path / "rw_fetch"
    backbone.write_bytes(b"MZ not this tree's build")
    monkeypatch.setattr("gpuwm.rustwx_fetch.find_fetch_bin",
                        lambda: backbone)
    monkeypatch.setattr("gpuwm.rustwx_fetch.probe_fetch_bin",
                        lambda path: (False, "abi mismatch"))
    with pytest.raises(ValueError):
        fetch_module.resolve_fetch_engine("rust", progress=lambda *a: None)
    assert fetch_module.resolve_fetch_engine(
        "auto", progress=lambda *a: None)[0] == "python"


def test_doctor_reports_a_renderer_that_fails_the_contract(
        tmp_path, monkeypatch):
    """The doctor branch for a foreign build, without staging one.

    ``test_doctor_reports_the_rust_renderer`` covers this outcome only
    when a foreign build happens to be resolvable, which on a clean
    checkout it is not -- so the branch that reports a stale bridge went
    unexecuted in the ordinary battery, which is exactly where a
    regression would be caught.
    """

    from gpuwm.doctor import _rust_renderer_check

    staged, evidence = _stage_foreign_renderer(monkeypatch, tmp_path)
    check = _rust_renderer_check()
    assert check.status == "missing"
    assert check.blocking is False
    assert str(staged) in check.detail
    assert evidence in check.detail
    assert check.remedy and "cargo build" in check.remedy


def test_parse_products_rust_mapping():
    # String equality only.  This test PASSED while `wind10` returned a
    # pressure chart, which is why it is not the guard against that bug
    # -- `test_wind10_renders_the_wind_not_a_pressure_chart` is, by
    # going to the rendered artifact and to the renderer's own statement
    # of what each product fills with.
    assert parse_products_rust("all") == "all"
    assert parse_products_rust("refl,t2,wind10,precip") == (
        "composite_reflectivity,2m_temperature,"
        "10m_wind_speed_and_direction,total_qpf")
    # Raw catalog slugs pass through for the renderer's own validation.
    assert parse_products_rust("sbcape,refl") == (
        "sbcape,composite_reflectivity")
    assert parse_products_rust("refl,refl") == "composite_reflectivity"
    with pytest.raises(ValueError, match="no products"):
        parse_products_rust(",")


def test_parse_size():
    assert parse_size("1200x900") == (1200, 900)
    assert parse_size("800X600") == (800, 600)
    for bad in ("1200", "axb", "100x100", "1200x"):
        with pytest.raises(ValueError):
            parse_size(bad)


# ---------------------------------------------------------------------------
# --list-products: the catalog with per-file availability
# ---------------------------------------------------------------------------

@needs_renderer
def test_list_products_reports_the_full_catalog(wrfout, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # The complete catalog is enumerated, not just what renders.
    # 168 = 152 + the standalone 10 m wind chart + this fixture's 15
    # generic ``var:`` rows (stored 2-D planes no named product claims;
    # the generic family is store-dependent, so the count is the
    # FIXTURE's, not the build's).
    assert "total=168" in out
    assert "renderable" in out and "excluded" in out
    # The generic rows are part of the catalog, not a side channel: every
    # stored plane without a named product renders as ``var:<name>``.
    generic_rows = [line for line in out.splitlines()
                    if " generic " in line and " var:" in line]
    assert len(generic_rows) == 15, out
    assert all("renderable" in line for line in generic_rows), generic_rows
    # The fixture's fields prove out the reflectivity composite ...
    assert any("composite_reflectivity" in line and "renderable" in line
               for line in out.splitlines())
    # ... and its exact-time (half-hourly) axis excludes fixed-hour
    # windowed accumulations with the reason spelled out.
    assert any("qpf_total" in line and "exact-time" in line
               for line in out.splitlines())
    # Every non-renderable row carries a reason string.
    for line in out.splitlines():
        if line.lstrip().startswith("missing-fields"):
            assert "not stored:" in line, line


@needs_renderer
def test_terrain_product_is_renderable_from_a_wrfout(wrfout, tmp_path, capsys):
    """Orography is in every wrfout; the catalog must offer it as a product.

    The negative control this pins is a silent skip: before the recipe
    existed the terrain plane sat in the store as a browse-only raw field
    and no catalog row mentioned it at all, so "there is no terrain frame"
    and "the terrain frame failed" were indistinguishable.
    """
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if "terrain_height" in line]
    assert rows, out
    assert all("renderable" in row for row in rows), rows


@needs_renderer
def test_list_products_whole_hour_axis_serves_windowed(
        wrfout_hourly, tmp_path, capsys):
    rc = cli.main(["render", str(wrfout_hourly), "--engine", "rust",
                   "--list-products", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # Run-total QPF folds the wrfout lane's apcp plane; the 1 h wind max
    # realizes through the documented hypot(u_10m, v_10m) fallback.
    assert any("qpf_total" in line and "renderable" in line
               for line in out.splitlines()), out
    assert any("10m_wind_1h_max" in line and "renderable" in line
               for line in out.splitlines()), out
    # 24/48 h windows stay blocked with their window arithmetic.
    assert any("2m_temp_0_24h_max" in line and "blocked" in line
               for line in out.splitlines()), out


@needs_renderer
def test_list_products_rejects_identity_gated_rows(
        wrfout, wrfout_hourly, tmp_path, capsys):
    """Architectural guard: availability rows must justify themselves
    with FIELDS, never with the source model's identity.  A 'gated'
    status, a 'no fetch plan' reason, or any 'for model ...' wording is
    a regression -- a product that cannot render always names the
    fields it is missing."""
    allowed = {"renderable", "missing-fields", "blocked", "excluded"}
    for path in (wrfout, wrfout_hourly):
        rc = cli.main(["render", str(path), "--engine", "rust",
                       "--list-products", "--out", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        rows = [line for line in out.splitlines()
                if line.startswith("  ") and line.split()]
        assert rows, out
        for line in rows:
            status = line.split()[0]
            assert status in allowed, line
            low = line.lower()
            assert "gated" not in low, line
            assert "fetch plan" not in low, line
            assert "for model" not in low, line
        assert "gated=" not in out
        # The families the old model-identity gate hid (smoke, simulated
        # IR, categorical precip) now carry honest field reasons.
        for slug in ("smoke_column", "simulated_ir_satellite",
                     "precipitation_type"):
            assert any(slug in line and "missing-fields" in line
                       and "not stored:" in line for line in rows), \
                (slug, out)


@needs_renderer
def test_windowed_products_render_on_whole_hour_stores(
        wrfout_hourly, tmp_path):
    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout_hourly), "--engine", "rust",
                   "--products", "qpf_total", "--out", str(out)])
    assert rc == 0
    produced = [p.name for p in out.rglob("*.png")]
    assert any("qpf_total" in name for name in produced), produced


def test_list_products_matplotlib_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("gpuwm.rustwx.find_renderer", lambda: None)
    path = tmp_path / "wrfout_d01_1974-04-03_18-00-00.nc"
    fields = _frame(seed=5)
    del fields["REFL_10CM"]
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0) \
            as writer:
        writer.write_frame(_STAMPS[0], fields)
    # `--engine matplotlib` by name: `auto` no longer resolves to this
    # engine, so its catalog is reachable only where a caller asked for
    # it (render law, audit F7).
    rc = cli.main(["render", str(path), "--list-products",
                   "--engine", "matplotlib", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "engine matplotlib" in out
    # The identity-gate guard holds on this engine too: reasons are
    # variable presence, never a model identity.
    assert "gated" not in out
    lines = out.splitlines()
    assert any("refl" in line and "missing-fields" in line
               and "REFL_10CM" in line for line in lines), out
    for product in ("t2", "wind10", "precip"):
        assert any(f" {product} " in f" {line} " and "renderable" in line
                   for line in lines), (product, out)


# ---------------------------------------------------------------------------
# --pair: compose two runs' rendered PNGs into comparison sheets
# ---------------------------------------------------------------------------

def _tiny_png(path, *, width=32, height=24) -> None:
    PIL = pytest.importorskip(
        "PIL", reason="--pair needs Pillow (render extra)")
    from PIL import Image

    Image.new("RGB", (width, height), "#336699").save(path)


def _pair_dirs(tmp_path):
    left = tmp_path / "run-a"
    right = tmp_path / "run-b"
    left.mkdir()
    right.mkdir()
    # Rust-engine naming: domain token + slug after the _fNNN_ lead
    # marker is the pair key, so differing run prefixes still pair.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f003_d02-3km_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_15z_f006_d02-3km_sbcape.png")
    # matplotlib naming pairs by identical stems.
    _tiny_png(left / "t2_d02-3km_1974-04-03_18-00-00.png")
    _tiny_png(right / "t2_d02-3km_1974-04-03_18-00-00.png")
    # Unmatched product must not produce a sheet.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f003_d02-3km_mlcin.png")
    return left, right


def test_pair_composes_common_products(tmp_path, capsys):
    left, right = _pair_dirs(tmp_path)
    case = tmp_path / "pairs"
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(case)])
    assert rc == 0
    # A pair compose is an invocation of the render door like any other,
    # so it claims its own run folder under --out (gpuwm.run_stamp) and
    # two composes into one directory cannot overwrite each other.
    out = run_stamp.latest(case)
    assert out is not None and run_stamp.is_run_folder(out)
    sheets = sorted(p.name for p in out.glob("*.png"))
    assert sheets == ["d02-3km_sbcape-pair.png",
                      "t2_d02-3km_1974-04-03_18-00-00-pair.png"]
    manifest = (out / "manifest.tsv").read_text(encoding="utf-8")
    assert manifest.startswith("product\tleft\tright\tpair\n")
    assert "sbcape" in manifest
    for sheet in out.glob("*.png"):
        assert sheet.stat().st_size > 500
        width, height = _png_size(sheet)
        assert width > 1_000 and height > 100, sheet.name


def test_pair_keys_keep_the_domain_so_nests_do_not_cross_pair(tmp_path):
    """Two nests of one run in one directory must not pair with each
    other -- a 3 km panel beside a 333 m panel compares nothing."""

    from gpuwm.pair_compose import product_name

    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_d02-3km_sbcape")
    ) == "d02-3km_sbcape"
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_d03-333m_sbcape")
    ) == "d03-333m_sbcape"
    # The pre-token naming still keys the same way it always did.
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f003_native_grid_sbcape")
    ) == "native_grid_sbcape"
    # The exact-time suffix rides along, as before.
    assert product_name(
        Path("rustwx_wrf_19740403_12z_f000_d02-1km_sbcape_valid_"
             "19740403_183000z_lead_000h30m00s")
    ) == "d02-1km_sbcape_valid_19740403_183000z_lead_000h30m00s"
    # A matplotlib name has no lead marker; the stem is the key.
    assert product_name(
        Path("t2_d02-1km_1974-04-03_18-00-00")
    ) == "t2_d02-1km_1974-04-03_18-00-00"
    # The shipped brand spelling keys exactly like the engine's own...
    assert product_name(
        Path("arwen_wrf_19740403_12z_f003_d02-3km_sbcape")
    ) == "d02-3km_sbcape"
    assert product_name(
        Path("arwen_wrf_19740403_12z_f000_d02-1km_sbcape_valid_"
             "19740403_183000z_lead_000h30m00s")
    ) == "d02-1km_sbcape_valid_19740403_183000z_lead_000h30m00s"
    # ...so a directory rendered before the arwen_ output rebrand still
    # pairs against one rendered after it.
    assert product_name(
        Path("arwen_wrf_19740403_12z_f003_d02-3km_sbcape")
    ) == product_name(
        Path("rustwx_wrf_19740403_15z_f006_d02-3km_sbcape"))


def test_pair_matches_one_domain_per_side_across_domains(tmp_path, capsys):
    """DOWNSCALE.md's own compare -- parent dir vs child dir -- pairs.

    The two sides of that documented command are DIFFERENT domains by
    definition (a 12 km parent against its 3 km offline child), and the
    domain-bearing pairing key made them share no key at all: walked
    2026-08-17, the doc's exact command refused with "no matching
    product PNGs".  When each side's stripped key is unambiguous, the
    same product pairs across the domain tokens.
    """
    left = tmp_path / "parent"
    right = tmp_path / "child"
    left.mkdir()
    right.mkdir()
    _tiny_png(left / "arwen_wrf_19740403_12z_f002_d01-12km_sbcape.png")
    _tiny_png(right / "arwen_wrf_19740403_12z_f002_d02-3km_sbcape.png")
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 0
    sheets = list(run_stamp.latest(tmp_path / "pairs").glob("*-pair.png"))
    assert len(sheets) == 1
    capsys.readouterr()


def test_pair_never_cross_pairs_an_ambiguous_side(tmp_path, capsys):
    """Two nests of one run in one directory still never cross-pair."""
    from gpuwm.pair_compose import compose_pairs

    left = tmp_path / "a"
    right = tmp_path / "b"
    left.mkdir()
    right.mkdir()
    # Left carries TWO nests of the same product: the stripped key is
    # ambiguous, so neither may pair with the right side's single nest.
    _tiny_png(left / "rustwx_wrf_19740403_12z_f000_d02-3km_sbcape.png")
    _tiny_png(left / "rustwx_wrf_19740403_12z_f000_d03-333m_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_12z_f000_d04-111m_sbcape.png")
    with pytest.raises(ValueError, match="no matching product"):
        compose_pairs(left, right, tmp_path / "pairs", title="t")


def test_pair_with_no_common_products_is_exit_2(tmp_path, capsys):
    left = tmp_path / "a"
    right = tmp_path / "b"
    left.mkdir()
    right.mkdir()
    _tiny_png(left / "rustwx_wrf_19740403_12z_f000_d02-1km_sbcape.png")
    _tiny_png(right / "rustwx_wrf_19740403_12z_f000_d02-1km_mucape.png")
    rc = cli.main(["render", "--pair", str(left), str(right),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 2
    assert "no matching product PNGs" in capsys.readouterr().err


def test_pair_rejects_wrfout_arguments(tmp_path, capsys):
    rc = cli.main(["render", "some_wrfout.nc", "--pair",
                   str(tmp_path), str(tmp_path),
                   "--out", str(tmp_path / "pairs")])
    assert rc == 2
    assert "do not combine" in capsys.readouterr().err


def test_render_without_wrfouts_or_pair_is_exit_2(capsys):
    rc = cli.main(["render"])
    assert rc == 2
    assert "WRFOUT" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# doctor integration
# ---------------------------------------------------------------------------

def test_doctor_reports_the_rust_renderer():
    from gpuwm.doctor import _rust_renderer_check

    check = _rust_renderer_check()
    assert check.name.startswith("renderer rw_wrfbatch")
    if RENDERER is None:
        # Not built: an info line with the build one-liner, never a gap
        # (matplotlib remains the documented fallback).
        assert check.status in ("info", "missing")
        assert check.remedy and "cargo build" in check.remedy
        return
    assert str(RENDERER) in check.detail
    if _RENDERER_USABLE:
        assert check.status == "verified"
        assert "basemap" in check.detail
        assert "--abi matches the render contract" in check.detail
    else:
        # Task #106.  This branch used to be unreachable, because the
        # condition above was `RENDERER is not None` -- the file existing.
        # A resolved binary from another checkout was therefore REQUIRED
        # by this test to report `verified`, which is the defect written
        # down as an assertion.  A renderer that resolves but is not this
        # tree's is reported, with the remedy, and does not block:
        # matplotlib is still the documented fallback.
        assert check.status == "missing"
        assert check.blocking is False
        assert check.remedy and "cargo build" in check.remedy


def test_doctor_env_override_naming_missing_file_is_hard(monkeypatch):
    from gpuwm.doctor import _rust_renderer_check

    monkeypatch.setenv(rustwx.RENDERER_ENV, r"C:\nonexistent\rw.exe")
    check = _rust_renderer_check()
    assert check.status == "missing"
    assert "names a missing file" in check.detail

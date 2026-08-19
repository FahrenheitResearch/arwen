"""The observation remap runs on Rust by default, and matches the scipy
reference bit for bit.

`gpuwm/verify/obs/regrid.py` was the last scipy site on a shipped data
path: every remap plan the observation battery builds went through
`scipy.spatial.cKDTree` with no second engine and no fallback, and
"regrid/transform" is named verbatim in Drew's Python boundary.  This
file is the shipping-path half of that port -- the crate's own goldens
(`tools/rustwx/crates/obs-regrid/golden/`) prove the arithmetic, and
these prove the ROUTE: that a bare call lands on Rust, that the estate
can supply the library a refusal names, and that the fallback says what
it costs.

Every parity assertion here compares IEEE bit patterns rather than a
tolerance, because that is the contract the port was written to and a
tolerance would hide exactly the drift this file exists to catch.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm import bridge_assets, bridges  # noqa: E402
from gpuwm import obs_regrid_bridge as regrid_bridge  # noqa: E402
from gpuwm.verify.obs import regrid  # noqa: E402


def _bridge_available() -> bool:
    return regrid_bridge.unavailable_reason() is None


requires_bridge = pytest.mark.skipif(
    not _bridge_available(),
    reason="the obs-regrid library is not built or staged on this box; "
           "build it with `cd tools/rustwx && cargo build --release "
           "-p obs-regrid --offline`")


# ---------------------------------------------------------------------------
# grids: one real pair off the staged case, one synthetic pair always present
# ---------------------------------------------------------------------------

#: The gitignored object cache the observation battery stages into,
#: under the checkout root.  Written repo-relative rather than as one
#: box's absolute path: the release snapshot builder's machine-path scan
#: fails the build on an absolute developer path in a tracked file, and
#: the hardcoded spelling also meant the real-bytes rows below could
#: never run anywhere else.  ``GPUWM_OBSBATTERY_CACHE`` points at a cache
#: outside this checkout -- a worktree borrowing the main clone's staged
#: case -- and the rows skip, saying so, when neither has the bytes.
CACHE = Path(os.environ.get("GPUWM_OBSBATTERY_CACHE")
             or REPO_ROOT / "cache" / "obsbattery")
MODEL_GRID = CACHE / "battery" / "model_grid_case20240521.npz"
OBS_GRID_PACK = CACHE / "packs" / "mrms_grid.obspack"
OBS_FIELD_PACK = CACHE / "packs" / "mrms_20240521T210037.obspack"


def _synthetic_pair(seed: int = 17):
    """A curvilinear source and a rotated destination, deterministic.

    Not a stand-in for the real case -- the crate's goldens carry that --
    but a pair every box has, so the route assertions below are not
    silently skipped everywhere the archive is absent.
    """
    rng = np.random.default_rng(seed)
    j, i = np.meshgrid(np.arange(37.0), np.arange(43.0), indexing="ij")
    source_latitude = 35.0 + 0.04 * j + 0.004 * np.sin(0.3 * i)
    source_longitude = -100.0 + 0.05 * i + 0.004 * np.cos(0.2 * j)
    j, i = np.meshgrid(np.arange(29.0), np.arange(31.0), indexing="ij")
    destination_latitude = 35.3 + 0.045 * j - 0.003 * i
    destination_longitude = -99.6 + 0.055 * i + 0.003 * j
    values = rng.normal(size=source_latitude.shape) * 12.0 + 20.0
    valid = rng.random(source_latitude.shape) > 0.25
    return (source_latitude, source_longitude, destination_latitude,
            destination_longitude, values, valid)


def _real_pair():
    for path in (MODEL_GRID, OBS_GRID_PACK, OBS_FIELD_PACK):
        if not path.is_file():
            pytest.skip(f"the staged case bytes are absent: {path}")
    from gpuwm.obs import obspack

    grid = np.load(MODEL_GRID)
    model_latitude = np.asarray(grid["latitude"], dtype=np.float64)[::4, ::4]
    model_longitude = np.asarray(grid["longitude"], dtype=np.float64)[::4, ::4]
    geometry = obspack.read_pack(OBS_GRID_PACK)
    field = obspack.read_pack(OBS_FIELD_PACK)
    obs_latitude = np.asarray(geometry.arrays["latitude"], dtype=np.float64)
    obs_longitude = np.asarray(geometry.arrays["longitude"], dtype=np.float64)
    values = np.asarray(field.arrays["values"], dtype=np.float64)
    valid = np.asarray(field.arrays["valid"]).astype(bool)

    rows = np.flatnonzero(
        (obs_latitude[:, 0] >= model_latitude.min() - 0.25)
        & (obs_latitude[:, 0] <= model_latitude.max() + 0.25))
    cols = np.flatnonzero(
        (obs_longitude[0, :] >= model_longitude.min() - 0.25)
        & (obs_longitude[0, :] <= model_longitude.max() + 0.25))
    rows = slice(int(rows[0]), int(rows[-1]) + 1, 12)
    cols = slice(int(cols[0]), int(cols[-1]) + 1, 12)
    return (obs_latitude[rows, cols], obs_longitude[rows, cols],
            model_latitude, model_longitude,
            values[rows, cols], valid[rows, cols])


def _both_engines(pair, method: str, max_distance_m: float, monkeypatch):
    (source_latitude, source_longitude, destination_latitude,
     destination_longitude, values, valid) = pair

    monkeypatch.delenv(regrid_bridge.OBSREGRID_PYTHON_ENV, raising=False)
    rust_plan = regrid.build_plan(
        source_latitude=source_latitude, source_longitude=source_longitude,
        destination_latitude=destination_latitude,
        destination_longitude=destination_longitude, method=method,
        max_distance_m=max_distance_m)
    rust_values, rust_valid = regrid.apply_plan(rust_plan, values, valid)

    monkeypatch.setenv(regrid_bridge.OBSREGRID_PYTHON_ENV, "1")
    python_plan = regrid.build_plan(
        source_latitude=source_latitude, source_longitude=source_longitude,
        destination_latitude=destination_latitude,
        destination_longitude=destination_longitude, method=method,
        max_distance_m=max_distance_m)
    python_values, python_valid = regrid.apply_plan(
        python_plan, values, valid)
    return ((rust_plan, rust_values, rust_valid),
            (python_plan, python_values, python_valid))


def _assert_bit_identical(rust, python, label):
    rust_plan, rust_values, rust_valid = rust
    python_plan, python_values, python_valid = python
    assert np.array_equal(rust_plan.source_index, python_plan.source_index), (
        f"{label}: the integer mapping differs between the engines")
    assert np.array_equal(rust_plan.reachable, python_plan.reachable), (
        f"{label}: the reachability mask differs between the engines")
    assert (np.float64(rust_plan.max_used_distance_m).tobytes()
            == np.float64(python_plan.max_used_distance_m).tobytes()), (
        f"{label}: max_used_distance_m differs in its bits")
    assert np.array_equal(rust_valid, python_valid), (
        f"{label}: the remapped validity differs between the engines")
    assert rust_values.tobytes() == python_values.tobytes(), (
        f"{label}: the remapped values differ in their bits")


# ---------------------------------------------------------------------------
# 1. parity, on real bytes and on a pair every box has
# ---------------------------------------------------------------------------

@requires_bridge
@pytest.mark.parametrize("method", regrid.METHODS)
def test_the_engines_agree_bit_for_bit_on_a_synthetic_curvilinear_pair(
        method, monkeypatch):
    rust, python = _both_engines(
        _synthetic_pair(), method, 6_000.0, monkeypatch)
    _assert_bit_identical(rust, python, f"synthetic/{method}")
    # A parity claim over a plan that reaches nothing measures nothing.
    assert int(np.count_nonzero(rust[0].reachable)) > 0


@requires_bridge
@pytest.mark.parametrize("method", regrid.METHODS)
def test_the_engines_agree_bit_for_bit_on_the_real_staged_case(
        method, monkeypatch):
    rust, python = _both_engines(_real_pair(), method, 12_000.0, monkeypatch)
    _assert_bit_identical(rust, python, f"real/{method}")
    assert int(np.count_nonzero(rust[0].reachable)) > 1000


@requires_bridge
def test_the_engines_agree_where_the_bound_leaves_the_domain_unreachable(
        monkeypatch):
    """The branch a loose bound never reaches."""
    rust, python = _both_engines(
        _synthetic_pair(), regrid.NEAREST, 400.0, monkeypatch)
    _assert_bit_identical(rust, python, "tight-bound")
    unreachable = int(rust[0].reachable.size
                      - np.count_nonzero(rust[0].reachable))
    assert unreachable > 0, (
        "the tight-bound case no longer leaves any destination cell "
        "unreachable, so it stopped testing the bound")


@requires_bridge
def test_the_engines_agree_through_the_single_field_front_door(monkeypatch):
    """`regrid_field` is a shipped entry point of its own."""
    pair = _synthetic_pair(seed=5)
    monkeypatch.delenv(regrid_bridge.OBSREGRID_PYTHON_ENV, raising=False)
    rust_values, rust_valid, rust_plan = regrid.regrid_field(
        source_latitude=pair[0], source_longitude=pair[1],
        destination_latitude=pair[2], destination_longitude=pair[3],
        values=pair[4], valid=pair[5], method=regrid.CELL_AVERAGE,
        max_distance_m=6_000.0)
    monkeypatch.setenv(regrid_bridge.OBSREGRID_PYTHON_ENV, "1")
    python_values, python_valid, python_plan = regrid.regrid_field(
        source_latitude=pair[0], source_longitude=pair[1],
        destination_latitude=pair[2], destination_longitude=pair[3],
        values=pair[4], valid=pair[5], method=regrid.CELL_AVERAGE,
        max_distance_m=6_000.0)
    _assert_bit_identical(
        (rust_plan, rust_values, rust_valid),
        (python_plan, python_values, python_valid), "regrid_field")


# ---------------------------------------------------------------------------
# 2. the ROUTE: a bare call lands on Rust
# ---------------------------------------------------------------------------

@requires_bridge
def test_a_bare_build_plan_calls_the_rust_engine(monkeypatch):
    """FAILS before the port: nothing called the bridge at all.

    Asserted by observing the call rather than by trusting the docstring:
    the seam's own entry point is replaced with one that records.
    """
    monkeypatch.delenv(regrid_bridge.OBSREGRID_PYTHON_ENV, raising=False)
    calls: list[str] = []
    real = regrid_bridge.build_plan

    def recording(**kwargs):
        calls.append(kwargs["method"])
        return real(**kwargs)

    monkeypatch.setattr(regrid_bridge, "build_plan", recording)
    pair = _synthetic_pair()
    regrid.build_plan(
        source_latitude=pair[0], source_longitude=pair[1],
        destination_latitude=pair[2], destination_longitude=pair[3],
        method=regrid.NEAREST, max_distance_m=6_000.0)
    assert calls == [regrid.NEAREST], (
        "a bare build_plan did not reach the Rust engine")


@requires_bridge
def test_a_bare_apply_plan_calls_the_rust_engine(monkeypatch):
    monkeypatch.delenv(regrid_bridge.OBSREGRID_PYTHON_ENV, raising=False)
    pair = _synthetic_pair()
    plan = regrid.build_plan(
        source_latitude=pair[0], source_longitude=pair[1],
        destination_latitude=pair[2], destination_longitude=pair[3],
        method=regrid.NEAREST, max_distance_m=6_000.0)
    calls: list[str] = []
    real = regrid_bridge.apply_plan

    def recording(**kwargs):
        calls.append(kwargs["method"])
        return real(**kwargs)

    monkeypatch.setattr(regrid_bridge, "apply_plan", recording)
    regrid.apply_plan(plan, pair[4], pair[5])
    assert calls == [regrid.NEAREST], (
        "a bare apply_plan did not reach the Rust engine")


def test_the_scipy_body_is_no_longer_reachable_without_the_opt_out():
    """The port is not "the Rust exists somewhere".

    `_tree` is the only door onto `scipy.spatial.cKDTree` in this module,
    and it must be reached only from the fallback body -- not from
    `build_plan`, which is the default path.
    """
    source = (REPO_ROOT / "gpuwm" / "verify" / "obs" / "regrid.py").read_text(
        encoding="utf-8")
    body = source.split("def build_plan(")[1].split("def _build_plan_python(")[0]
    assert "_tree(" not in body, (
        "build_plan reaches the scipy tree directly; the default path must "
        "route through the Rust bridge and reach scipy only through the "
        "announced fallback")
    assert "regrid_bridge.route(" in body, (
        "build_plan does not consult the bridge route at all")


# ---------------------------------------------------------------------------
# 3. the fallback is announced, and names what it costs
# ---------------------------------------------------------------------------

def test_the_opt_out_prints_a_workaround_naming_the_breakage(
        monkeypatch, capsys):
    monkeypatch.setenv(regrid_bridge.OBSREGRID_PYTHON_ENV, "1")
    monkeypatch.setattr(regrid_bridge, "_REPORTED_FALLBACKS", set())
    pair = _synthetic_pair()
    regrid.build_plan(
        source_latitude=pair[0], source_longitude=pair[1],
        destination_latitude=pair[2], destination_longitude=pair[3],
        method=regrid.NEAREST, max_distance_m=6_000.0)
    printed = capsys.readouterr().out
    assert "WORKAROUND" in printed, (
        "a run on the Python engine said nothing; fixed-means-default "
        "requires the fallback to report itself")
    # The gate law: the announcement must name the concrete breakage.
    assert "traversal order" in printed and "lowest source index" in printed


def test_the_workaround_line_is_printed_once_per_operation(
        monkeypatch, capsys):
    monkeypatch.setenv(regrid_bridge.OBSREGRID_PYTHON_ENV, "1")
    monkeypatch.setattr(regrid_bridge, "_REPORTED_FALLBACKS", set())
    pair = _synthetic_pair()
    for _ in range(3):
        regrid.build_plan(
            source_latitude=pair[0], source_longitude=pair[1],
            destination_latitude=pair[2], destination_longitude=pair[3],
            method=regrid.NEAREST, max_distance_m=6_000.0)
    printed = capsys.readouterr().out
    assert printed.count("WORKAROUND") == 1, (
        "the workaround line repeated per call; a line printed on every "
        "remap is a line readers learn to ignore")


# ---------------------------------------------------------------------------
# 4. the estate: a refusal may not name a remedy that cannot supply it
# ---------------------------------------------------------------------------

def test_the_library_is_a_bundled_artifact():
    """FAILS before the fix: `gpuwm fetch-bridges` could not stage it.

    The bridge's own refusal leads with `gpuwm fetch-bridges`.  If the
    bundle table does not carry `obs_regrid`, that command stages a
    complete bundle and the refusal repeats verbatim -- the
    `rw_mpas_convert` failure, a third time.
    """
    names = {artifact.name for artifact in bridge_assets.BUNDLED_ARTIFACTS}
    assert "obs_regrid" in names


def test_the_bundle_entry_uses_the_bridges_own_env_var_and_crate():
    artifact, = [a for a in bridge_assets.BUNDLED_ARTIFACTS
                 if a.name == "obs_regrid"]
    assert artifact.env_var == regrid_bridge.OBSREGRID_BRIDGE_ENV
    assert artifact.kind == "library"
    assert artifact.crate == bridges.RUSTWX_CRATE_RELATIVE


def test_the_library_abi_handshake_matches_the_module():
    symbol, version = bridge_assets.library_abi_for("obs_regrid")
    assert symbol == "gpuwm_obsregrid_abi_version"
    assert version == regrid_bridge.OBSREGRID_ABI


def test_the_contract_marker_matches_the_module():
    assert bridges.BRIDGE_ABI_MARKERS["obs_regrid"] == regrid_bridge.ABI_MARKER


def test_the_crate_injects_the_source_revision():
    """A bridge artifact the release cut cannot pin is a bridge that
    cannot ship."""
    script = (REPO_ROOT / "tools" / "rustwx" / "crates" / "obs-regrid"
              / "build.rs")
    assert script.is_file()
    assert "rustc-env=GPUWM_BRIDGE_SOURCE_REV" in script.read_text(
        encoding="utf-8")
    capi = (REPO_ROOT / "tools" / "rustwx" / "crates" / "obs-regrid" / "src"
            / "capi.rs").read_text(encoding="utf-8")
    assert "SOURCE_REV_STAMP" in capi
    assert 'env!("GPUWM_BRIDGE_SOURCE_REV")' in capi


@requires_bridge
def test_the_built_library_carries_a_source_revision_stamp():
    """Read out of the binary as bytes, never by executing it."""
    path = regrid_bridge.resolve_obsregrid_bridge()
    assert bridge_assets.SOURCE_REV_MARKER in path.read_bytes(), (
        f"{path} carries no GPUWM_BRIDGE_SOURCE_REV stamp, so "
        "`tools/build_bridge_bundle.py pin` would refuse it")


# ---------------------------------------------------------------------------
# 5. refusals name the breakage
# ---------------------------------------------------------------------------

@requires_bridge
def test_a_bound_that_is_not_a_distance_is_refused_by_the_engine():
    pair = _synthetic_pair()
    with pytest.raises(Exception) as caught:
        regrid.build_plan(
            source_latitude=pair[0], source_longitude=pair[1],
            destination_latitude=pair[2], destination_longitude=pair[3],
            method=regrid.NEAREST, max_distance_m=0.0)
    assert "positive and finite" in str(caught.value)


@requires_bridge
def test_a_field_that_does_not_match_the_plan_is_refused(monkeypatch):
    monkeypatch.delenv(regrid_bridge.OBSREGRID_PYTHON_ENV, raising=False)
    pair = _synthetic_pair()
    plan = regrid.build_plan(
        source_latitude=pair[0], source_longitude=pair[1],
        destination_latitude=pair[2], destination_longitude=pair[3],
        method=regrid.NEAREST, max_distance_m=6_000.0)
    with pytest.raises(ValueError, match="does not match the plan"):
        regrid.apply_plan(plan, pair[4][:-1], pair[5][:-1])


def test_the_missing_library_refusal_names_a_command_that_can_supply_it(
        monkeypatch, tmp_path):
    monkeypatch.setenv(regrid_bridge.OBSREGRID_BRIDGE_ENV,
                       str(tmp_path / "absent.dll"))
    with pytest.raises(FileNotFoundError) as caught:
        regrid_bridge.resolve_obsregrid_bridge()
    assert "absent.dll" in str(caught.value)


def test_the_search_ladder_refusal_lists_every_path_and_the_remedies(
        monkeypatch):
    monkeypatch.delenv(regrid_bridge.OBSREGRID_BRIDGE_ENV, raising=False)
    monkeypatch.setattr(regrid_bridge, "library_candidates", lambda: ())
    with pytest.raises(FileNotFoundError) as caught:
        regrid_bridge.resolve_obsregrid_bridge()
    message = str(caught.value)
    assert "gpuwm fetch-bridges" in message
    assert "cargo build --release -p obs-regrid" in message

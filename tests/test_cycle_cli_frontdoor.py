"""``gpuwm cycle`` must reach the capability, not just describe it.

Everything the cycle has ever demonstrated -- a real anchor per boundary,
the armed clock triple, the three-hash ingestion gate on both arms, a
child placed on a real echo -- ran through ``tools/cycle_demo_*.py``.  The
product door reached NONE of it: ``_engine_for`` handed the supervisor a
closure that returned two integers, so the CLI cycled a clock over an
empty root and wrote a receipt that looked exactly like a run.

These tests are the door's own reachability proof.  Every one of them is
BEHAVIOURAL -- a missing flag fails as a wrong answer, never as an
ImportError -- and every one is red on revert.
"""

import json

import numpy as np
import pytest

from gpuwm.cli import build_parser
from gpuwm.cycle.cli import cycle_main
from gpuwm.cycle.contracts import CycleRefusal

NY = NX = 48


def _frame(path, k, *, blob_j=24, blob_i=24):
    """One replay frame in the shape ``tools/cycle_mpas_leg.py`` reads.

    A flat npz of prognostic fields, ``time_seconds`` and the derived
    diagnostics beside them -- the OTHER producer's real on-disk format,
    not a shape invented here.
    """
    yy, xx = np.mgrid[0:NY, 0:NX]
    blob = np.exp(-(((yy - (blob_j + k)) ** 2
                     + (xx - (blob_i + k)) ** 2) / (2 * 4.0 ** 2)))
    rho = 1.10 - 0.02 * blob
    theta = 300.0 + 12.0 * blob
    rho_theta = rho * theta
    np.savez(
        path,
        rho=rho, rho_theta=rho_theta, rho_u=rho * (8.0 + 14.0 * blob),
        rho_w=rho * 3.5 * blob, scalars=rho * 0.012 * blob,
        time_seconds=np.asarray(float(k) * 960.0),
        exner=np.power(np.maximum(rho_theta, 1e-12) * (287.0 / 100000.0),
                       287.0 / (1004.5 - 287.0)),
        composite_reflectivity=(14.0 + 52.0 * blob))
    return path


def _series(tmp_path, n=4):
    directory = tmp_path / "state"
    directory.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        _frame(directory / f"frame_{k:03d}.npz", k)
    return str(directory / "*.npz")


def _increment(path, *, scale):
    """An increment in PROGNOSTIC space, keyed by prognostic field name."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    blob = np.exp(-(((yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 3.0 ** 2)))
    np.savez(path, rho_u=(scale * blob))
    return str(path)


def _args(tmp_path, argv, *, root=None):
    parser = build_parser()
    root = root if root is not None else tmp_path / "run"
    base = ["cycle", "--root", str(root), "--epoch-anchor",
            "2026-08-14T18:00:00Z", "--parent-kind", "replay",
            "--cycle-seconds", "960"]
    return parser.parse_args(base + argv)


def _receipt(root, cycle):
    return json.loads((root / f"cycle_{cycle:03d}" / "RECEIPT.json")
                      .read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. the door writes a REAL anchor and fires the REAL gate
# ---------------------------------------------------------------------------
def test_the_front_door_writes_an_anchor_per_boundary(tmp_path):
    """A cycle that leaves no anchor never cycled anything.

    Red on revert with the tick-only closure: it returns
    ``{"parent_ticks": ..., "anchor_ticks": ...}`` and touches no disk, so
    ``anchors/`` never exists and the receipt's gate block is null forever.
    """
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "2", "--parent-state",
                            _series(tmp_path), "--parent-mesh-id", "t48x48"])
    assert cycle_main(args) == 0

    anchors = sorted((root / "anchors").glob("*"))
    assert anchors, "the front door wrote no anchor at any boundary"
    assert len(anchors) == 2, f"one anchor per boundary, got {anchors}"


def test_the_front_door_fires_the_three_hash_gate_on_both_arms(tmp_path):
    """Null arm flat with the hash unmoved; applied arm moves it.

    This is the evidence the demo scripts produced and the door could not.
    A nonzero delta alone proves nothing, so both arms are asserted: the
    null arm must leave the state hash EXACTLY where it was, and the
    applied arm must move it while naming nonzero cells.
    """
    root = tmp_path / "run"
    args = _args(tmp_path, [
        "--cycles", "3", "--parent-state", _series(tmp_path),
        "--parent-mesh-id", "t48x48",
        "--analysis-increment", "1=null",
        "--analysis-increment", f"2={_increment(tmp_path / 'inc2.npz', scale=3.5)}",
    ])
    assert cycle_main(args) == 0

    null_arm = _receipt(root, 2)["ingestion"]
    assert null_arm is not None, (
        "cycle 2 evidences the increment applied at cycle 1; a null block "
        "here means the gate never ran inside the supervisor")
    assert int(null_arm["increment_nonzero_cells"]) == 0
    assert null_arm["background_sha256"] == null_arm["analysis_sha256"], (
        "NULL ARM: a zero increment must be bit-stable through the anchor")

    applied = _receipt(root, 3)["ingestion"]
    assert applied is not None
    assert int(applied["increment_nonzero_cells"]) > 0, (
        "APPLIED ARM: an exact-zero cell count means the experiment never "
        "ran, not that the increment was small")
    assert applied["background_sha256"] != applied["analysis_sha256"], (
        "APPLIED ARM: the state hash must MOVE when an increment lands")
    # verify_ingestion spells this "fields"; the supervisor's halt path
    # read "increment_fields", which only a hand-built fixture supplies.
    assert "rho_u" in (applied.get("fields") or ())
    from gpuwm.cycle.supervisor import _increment_fields
    assert _increment_fields(applied) == ["rho_u"], (
        "the ANALYSIS_NOT_INGESTED halt must name the fields the REAL "
        "producer reports, not the ones a fixture invented")


def test_the_arming_triple_is_armed_at_every_boundary(tmp_path):
    root = tmp_path / "run"
    args = _args(tmp_path, ["--cycles", "3", "--parent-state",
                            _series(tmp_path), "--parent-mesh-id", "t48x48"])
    assert cycle_main(args) == 0
    for cycle in (1, 2, 3):
        arming = _receipt(root, cycle)["arming"]
        assert arming["armed"] is True
        assert arming["parent_ticks"] == arming["expected_ticks"]
        assert arming["anchor_ticks"] == arming["expected_ticks"]


def test_a_run_with_no_parent_state_refuses_instead_of_receipting_nothing(
        tmp_path):
    """No state is not a cycle.  It must refuse, not write a healthy receipt."""
    args = _args(tmp_path, ["--cycles", "1"])
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    assert "--parent-state" in str(excinfo.value)
    assert not (tmp_path / "run" / "cycle_ledger.jsonl").exists()


# ---------------------------------------------------------------------------
# 2. placement resolves THROUGH the CLI
# ---------------------------------------------------------------------------
def test_placement_resolves_through_the_cli(tmp_path):
    """``_resolve_placement_provider`` never passed ``parent_geometry``.

    Red on revert as a REFUSAL from inside the provider -- "placement needs
    the parent's geometry" -- which is a wrong answer at the front door,
    not an import failure.
    """
    root = tmp_path / "run"
    geo = tmp_path / "geo.npz"
    lat = np.linspace(35.0, 35.0 + 0.027 * (NY - 1), NY)[:, None] * np.ones(
        (1, NX))
    lon = np.ones((NY, 1)) * np.linspace(-97.5, -97.5 + 0.033 * (NX - 1),
                                         NX)[None, :]
    np.savez(geo, XLAT=lat, XLONG=lon)

    args = _args(tmp_path, [
        "--cycles", "1", "--parent-state", _series(tmp_path),
        "--parent-mesh-id", "t48x48", "--child-slots", "1",
        "--child-dt-seconds", "30",
        "--placement-provider", "tracker",
        "--placement-tracker-field", "composite_reflectivity",
        "--parent-geo-file", str(geo), "--parent-dx-m", "3000",
        "--retire-below-strength", "35",
        "--child-nx", "9", "--child-ny", "9", "--child-dx-m", "1000",
    ])
    assert cycle_main(args) == 0
    placements = _receipt(root, 1)["placements"]
    assert placements, "the tracker found no placement through the CLI"
    assert any(item.get("state") in ("PLANNED", "LIVE")
               for item in placements), placements


def test_placement_asks_for_a_geometry_it_can_name(tmp_path):
    args = _args(tmp_path, [
        "--cycles", "1", "--parent-state", _series(tmp_path),
        "--parent-mesh-id", "t48x48", "--child-slots", "1",
        "--placement-provider", "tracker", "--retire-below-strength", "35"])
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    message = str(excinfo.value)
    assert "--parent-geo-file" in message or "--placement-obs-file" in message


# ---------------------------------------------------------------------------
# 3. every flag a refusal names must EXIST, and every flag must be documented
# ---------------------------------------------------------------------------
def _cycle_option_strings():
    import argparse as _argparse

    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, _argparse._SubParsersAction):
            return {name for opt in action.choices["cycle"]._actions
                    for name in opt.option_strings}
    raise AssertionError("gpuwm has no subparsers")


def test_every_flag_named_in_cycle_code_exists_on_the_parser():
    """A refusal that tells a user to pass a flag that does not exist is
    worse than no refusal at all.

    Red on revert: ``--placement-obs-file`` is named by
    ``gpuwm.cycle.engine``'s obs refusal and by the docs, and was absent
    from argparse entirely.
    """
    import re
    from pathlib import Path

    import gpuwm.cycle as cycle_pkg

    options = _cycle_option_strings()
    source_dir = Path(cycle_pkg.__file__).parent
    named = set()
    for path in sorted(source_dir.glob("*.py")):
        for flag in re.findall(r"--[a-z][a-z0-9-]{2,}", path.read_text(
                encoding="utf-8")):
            named.add(flag)
    # Flags owned by a different front door are not this parser's to carry.
    #
    # ``--anchor``, ``--steps`` and ``--out`` appear in
    # ``gpuwm/cycle/mpas_bridge.py`` because that module BUILDS the argv
    # for ``mpas_cycle_bridge.worker``, a separate program with its own
    # parser.  They are the worker's flags, not the door's, and adding
    # them here would advertise options ``gpuwm cycle`` does not take.
    # ``--port-root`` and ``--port-config`` DID become real door flags
    # when the model parent adapter landed, so they are no longer
    # excused -- this test is what keeps that honest.
    named -= {"--backend", "--history", "--cycle-index",
              "--consistency-threshold", "--plan", "--out",
              "--anchor", "--steps"}
    missing = sorted(named - options)
    assert not missing, (
        f"gpuwm/cycle names these flags but argparse does not define them: "
        f"{missing}")


def test_every_cycle_flag_is_documented_in_the_spine_doc():
    """The reverse direction: a flag nobody documented is unreachable too."""
    from pathlib import Path

    import gpuwm

    doc = (Path(gpuwm.__file__).parents[1] / "docs" / "cycle-spine.md"
           ).read_text(encoding="utf-8")
    long_flags = {name for name in _cycle_option_strings()
                  if name.startswith("--")} - {"--help", "--explain"}
    undocumented = sorted(name for name in long_flags if name not in doc)
    assert not undocumented, (
        f"undocumented cycle flags: {undocumented}")


# ---------------------------------------------------------------------------
# 4. the observation field is the consumer's choice, and it is CHECKED
# ---------------------------------------------------------------------------
def test_the_obs_placement_field_defaults_to_the_files_observation_variable():
    """``gpuwm-obs.radar-grid.v1`` ships z_obs, z_max and z_mean side by side
    and states the choice between them is the consumer's.  The default is
    the file's own observation variable, ``z_obs`` -- the one every DA lane
    reads and the one the real three-volume series was placed against.
    """
    parser = build_parser()
    args = parser.parse_args(["cycle", "--root", ".", "--epoch-anchor",
                              "2026-08-14T18:00:00Z", "--cycle-seconds",
                              "960", "--cycles", "1", "--parent-kind",
                              "replay"])
    assert args.placement_obs_field == "z_obs"


def test_an_unknown_obs_field_is_refused_naming_what_the_file_carries(
        tmp_path):
    """Root cause, not an alias: the plane lookup must say what IS there."""
    from gpuwm.cycle.placement import _document_plane

    document = {"schema": "gpuwm-obs.radar-grid.v1",
                "variables": {"z_obs": np.zeros((2, 4, 4)),
                              "z_max": np.zeros((2, 4, 4))}}
    with pytest.raises(CycleRefusal) as excinfo:
        _document_plane(document, "z_maximum", "<doc>")
    assert excinfo.value.observed["available"] == ["z_max", "z_obs"]


# ---------------------------------------------------------------------------
# 5. the unwired parent kind refuses SPECIFICALLY
# ---------------------------------------------------------------------------
def test_the_adapter_symbol_the_refusal_named_actually_exists_now(tmp_path):
    """The refusal named a symbol to land; this asserts it landed.

    While the closed-loop lane was unmerged this test asserted the
    OPPOSITE -- that the door refused and named ``adapter_module`` /
    ``adapter_symbol`` so the adapter lane knew what to build.  Both
    halves then merged with zero textual conflicts and the door STILL
    refused, because the capability shipped as
    ``gpuwm.cycle.mpas_bridge`` and nothing ever defined the name the
    door looked up.  This test is the tripwire for that regression: if
    the symbol goes missing again, the door silently returns to refusing
    a feature the tree can perform.
    """
    import importlib

    from gpuwm.cycle import cli

    module = importlib.import_module(cli._ENGINE_MODULE)
    assert hasattr(module, cli._PARENT_BACKEND_SYMBOL), (
        f"{cli._ENGINE_MODULE}.{cli._PARENT_BACKEND_SYMBOL} is the symbol "
        f"the door imports for a model parent kind; without it "
        f"--parent-kind mpas-cuda refuses a capability this tree has")

    # And with the adapter present, the door gets PAST the adapter check
    # and refuses on the port plumbing instead -- proof the lookup
    # succeeded rather than the refusal merely changing wording.
    args = _args(tmp_path, ["--cycles", "1", "--parent-kind", "mpas-cuda",
                            "--parent-state", _series(tmp_path),
                            "--parent-mesh-id", "t48x48"])
    with pytest.raises(CycleRefusal) as excinfo:
        cycle_main(args)
    observed = excinfo.value.observed
    assert observed["parent_kind"] == "mpas-cuda"
    assert "adapter_symbol" not in observed
    assert observed["missing"] == ["--port-root", "--port-config",
                                   "--port-steps"]
    assert not (tmp_path / "run" / "cycle_ledger.jsonl").exists()

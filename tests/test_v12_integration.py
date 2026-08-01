"""v1.2 assembly seams: the places where two lanes have to agree.

EXPERIMENTAL, like everything the six v1.2 lanes shipped.  Each lane's own
suite proves that lane correct in isolation; nothing in those suites can
prove that the ensemble engine's manifest is the manifest the product side
reads, or that the radar lane's file is the file the filter ingests.  These
tests exercise one lane's real writer against another lane's real reader
with no fixture in between, because a fixture is exactly where two lanes
can agree with a third thing and still disagree with each other.

CPU-only by construction: no CuPy import anywhere in this file.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from gpuwm.da import enprod, perturb
from gpuwm.ensemble.config import load_ensemble_config
from gpuwm.ensemble.engine import run_ensemble
from gpuwm.ensemble.manifest import (
    ENSEMBLE_MANIFEST_NAME, MEMBER_STATUSES,
)
from gpuwm.ensemble.member import (
    MemberOutcome, diagnostics_sha256, refresh_diagnostics, run_member,
)
from gpuwm.ensemble.perturbation import resolve_perturbation
from gpuwm.ensemble.state_sha import (checkpoint_state_sha256,
                                      live_state_sha256,
                                      serialized_state_attrs)
from gpuwm.ensemble.wrfout_inventory import (WRFOUT_INVENTORY_KEY,
                                             member_inventory)

BASE_TOML = """
[experiment]
name = "v12_integration"
start_time = 1999-05-03T12:00:00
run_seconds = 120.0
restart_interval_s = 60.0

[projection]
map_proj = "lambert"
ref_lat = 39.6848
ref_lon = -83.9297
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -83.9297

[shared]
nz = 4
ztop = 20000.0
p_top = 10000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 16
ny = 12
time_step = 60
dx = 3000.0
history_interval_s = 60.0
"""

_NX, _NY, _NZ, _DX = 16, 12, 4, 3000.0
_STAMPS = ("1970-01-01_18:00:00", "1970-01-01_19:00:00")


def _write_configs(tmp_path, *, n_members=2, perturbation="none"):
    (tmp_path / "base.toml").write_text(BASE_TOML, encoding="utf-8")
    overlay = tmp_path / "ensemble.toml"
    overlay.write_text(
        "[ensemble]\n"
        'base_config = "base.toml"\n'
        f"n_members = {n_members}\n"
        "base_seed = 20260730\n"
        f'perturbation = "{perturbation}"\n',
        encoding="utf-8")
    return overlay


def _wrfout_runner(*, base_config, member_dir, index, seed, perturbation,
                   perturbation_options, run_seconds=None, **_):
    """A member kernel that writes real wrfouts and a real MemberOutcome.

    The engine's manifest writer, seed derivation, and status machine are
    the shipped ones; only the integration is stood in for, because the
    seam under test is the manifest, not the dycore.
    """

    from gpuwm.io.wrfout import WrfoutWriter

    member_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed % (2 ** 32))
    yy, xx = np.mgrid[0:_NY, 0:_NX].astype(float)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    y0 = _NY / 2.0 + rng.normal(0.0, _NY / 12.0)
    x0 = _NX / 2.0 + rng.normal(0.0, _NX / 12.0)
    path = member_dir / f"wrfout_d02_{_STAMPS[0].replace(':', '-')}.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=_DX, dy=_DX,
                      global_attrs={"GRID_ID": 2}) as writer:
        for step, stamp in enumerate(_STAMPS):
            blob = np.exp(-(((yy - y0 - 1.5 * step) ** 2
                             + (xx - x0 - 1.5 * step) ** 2)
                            / (2.0 * (min(_NY, _NX) / 6.0) ** 2)))
            surface = 60.0 * blob + rng.normal(0.0, 2.0, (_NY, _NX))
            column = np.stack(
                [surface * w for w in (1.0, 0.9, 0.7, 0.4)][:_NZ])
            writer.write_frame(stamp, {
                "T": np.zeros((_NZ, _NY, _NX), np.float32),
                "MU": np.zeros((_NY, _NX), np.float32),
                "REFL_10CM": column.astype(np.float32),
                "UP_HELI_MAX": (180.0 * blob ** 2).astype(np.float32),
                "T2": (292.0 + 4.0 * blob).astype(np.float32),
                "U10": rng.normal(0.0, 5.0, (_NY, _NX)).astype(np.float32),
                "V10": rng.normal(0.0, 5.0, (_NY, _NX)).astype(np.float32),
                "RAINC": (5.0 * blob).astype(np.float32),
                "RAINNC": (20.0 * blob).astype(np.float32),
                "XLAT": lat.astype(np.float32),
                "XLONG": lon.astype(np.float32),
                "HGT": np.zeros((_NY, _NX), np.float32),
                "SINALPHA": np.zeros((_NY, _NX), np.float32),
                "COSALPHA": np.ones((_NY, _NX), np.float32),
            })
    sha = f"{(seed * 2654435761 + index) % (16 ** 64):064x}"
    return MemberOutcome(
        index=index, seed=seed, member_dir=member_dir,
        initial_state_sha256=sha, final_state_sha256=sha,
        wall_seconds=1.0, sim_seconds=float(run_seconds or 120),
        wrfout_count=1, last_checkpoint=None, perturbation={"report": None},
        # The real member kernel's own inventory function, over the files
        # this runner actually wrote: the seam under test is that what the
        # engine RECORDS is what the product side can BIND.
        wrfout_inventory=member_inventory([path], member_dir=member_dir))


# ---------------------------------------------------------------------------
# Seam 1 -- gpuwm-ensemble-manifest.v1: the engine writes it, enprod reads it
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine_written_root(tmp_path):
    pytest.importorskip(
        "netCDF4", reason="the wrfout writer needs netCDF4")
    overlay = _write_configs(tmp_path, n_members=3)
    root = tmp_path / "ens"
    result = run_ensemble(load_ensemble_config(overlay), root,
                          runner=_wrfout_runner)
    assert result.status == "COMPLETE"
    return root


def test_enprod_reads_the_engines_own_manifest(engine_written_root):
    """The seam, stated as one assertion.

    Before this was fixed the engine wrote ``status: "DONE"`` and enprod
    accepted only ``"complete"``, so a real ensemble was refused whole --
    a three-member ensemble reported as "3 of 3 declared member(s) are
    unusable".  Neither lane's own suite could see it: each tested
    against its own idea of the manifest.
    """

    manifest = enprod.load_manifest(engine_written_root)
    assert manifest.n_members == 3
    assert [member.number for member in manifest.members] == [0, 1, 2]
    assert [member.directory.name for member in manifest.members] == [
        "member_000", "member_001", "member_002"]
    # Every member directory the reader resolved actually exists; the
    # reader refuses otherwise, so this pins that it resolved them from
    # the manifest rather than from its own naming guess.
    assert all(member.directory.is_dir() for member in manifest.members)
    # The seed survives the round trip -- it is the reproduction handle.
    assert len({member.seed for member in manifest.members}) == 3


def test_enprod_reads_the_writers_directory_key_not_its_own_guess(tmp_path):
    """``member_dir`` is honoured, not fallen through.

    The reader's fallback is ``member_{NNN:03d}``, which agrees with the
    writer today.  If the writer's key were ignored, a manifest naming a
    different directory would be read from the wrong place instead of
    refused -- so give it one and require the reader to follow it.
    """

    root = tmp_path / "ens"
    (root / "elsewhere_07").mkdir(parents=True)
    (root / enprod.MANIFEST_FILENAME).write_text(json.dumps({
        "schema": enprod.MANIFEST_SCHEMA,
        "n_members": 1,
        "members": [{"index": 7, "member_dir": "elsewhere_07",
                     "seed": 11, "status": "DONE"}],
    }), encoding="utf-8")
    manifest = enprod.load_manifest(root)
    assert manifest.members[0].directory.name == "elsewhere_07"


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "FAILED"])
def test_enprod_still_refuses_every_unfinished_writer_status(tmp_path,
                                                             capsys,
                                                             status):
    """Closing the seam must not widen the gate.

    ``DONE`` became acceptable; the writer's other three statuses are
    exactly the members that must never be averaged, and none of them
    is -- with the operator told, by name, which ``--accept-status``
    would admit it.

    Warn-not-block split those two facts across two lines rather than
    dropping either.  An unaccepted member is now DROPPED with one
    warning that names its status and the exact flag; the refusal fires
    when dropping leaves nothing to average.  Both halves are pinned
    here, because the gate is the conjunction: a warning without the
    refusal would average an unfinished member, and a refusal without
    the warning would hide which status caused it.
    """

    assert status in MEMBER_STATUSES
    root = tmp_path / "ens"
    (root / "member_000").mkdir(parents=True)
    (root / enprod.MANIFEST_FILENAME).write_text(json.dumps({
        "schema": enprod.MANIFEST_SCHEMA,
        "n_members": 1,
        "members": [{"index": 0, "member_dir": "member_000",
                     "seed": 1, "status": status}],
    }), encoding="utf-8")
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        enprod.load_manifest(root)
    # Nothing was averaged: the sole member was dropped, and dropping it
    # left an ensemble of none.
    assert "no usable member remains" in str(caught.value)
    assert "skipped 1 by status" in str(caught.value)
    # And the operator is told which status, and the flag that admits it.
    warnings = [line for line in capsys.readouterr().err.splitlines()
                if line.startswith("warning:")]
    assert len(warnings) == 1, warnings
    assert status in warnings[0]
    assert "--accept-status" in warnings[0]


def test_the_engines_manifest_carries_the_experimental_stamp(
        engine_written_root):
    document = json.loads(
        (engine_written_root / ENSEMBLE_MANIFEST_NAME).read_text("utf-8"))
    assert document["experimental"] is True
    assert document["stability"] == "experimental"
    assert document["status"] == "COMPLETE"


# ---------------------------------------------------------------------------
# Seam 1b -- the frame inventory: the engine records it, enprod binds it
# ---------------------------------------------------------------------------


def _fail_one_member(root, index):
    """Mark a finished member FAILED, leaving its inventory intact.

    The state an operator meets when a member crashed after writing
    history and reaches for ``--accept-status``.
    """
    path = root / ENSEMBLE_MANIFEST_NAME
    document = json.loads(path.read_text("utf-8"))
    document["status"] = "FAILED"
    document["members"][index]["status"] = "FAILED"
    document["members"][index]["error"] = {"type": "RuntimeError",
                                           "message": "device fell over"}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _override_report(root):
    wrf = pytest.importorskip("wrf", reason="the frame indexer needs wrf")
    manifest = enprod.load_manifest(root, accept_status=("DONE", "FAILED"))
    indexed, _ = enprod.index_ensemble(manifest, wrf=wrf, domain=None)
    return enprod.verify_override_inventory(manifest, indexed)


def test_enprod_binds_the_engines_own_frame_inventory(engine_written_root):
    """The E-03 seam: the writer's inventory is the reader's evidence.

    ``wrfout_count`` was the whole of what the manifest carried and the
    whole of what the override checked, so one stale file replacing one
    real file was indistinguishable from the run's own output.  The
    engine now records path/domain/valid times/frame indices/size/sha256
    per file; this is the reader binding against exactly that, with no
    fixture in between.
    """

    _fail_one_member(engine_written_root, 1)
    report = _override_report(engine_written_root)
    assert report["overridden"] is True
    assert report["verified"] == [1]
    assert report["detail"][0]["declared_wrfout_files"] == 1
    # Two valid times in the file the runner wrote, both bound.
    assert report["detail"][0]["bound_frames"] == 2
    assert report["detail"][0]["declared_frames"] == 2


def test_stale_bytes_under_the_engines_own_inventory_are_refused(
        engine_written_root):
    """The report's probe, end to end across the two real lanes."""

    document = _fail_one_member(engine_written_root, 1)
    declared = document["members"][1][WRFOUT_INVENTORY_KEY][0]
    target = engine_written_root / "member_001" / declared["path"]
    with open(target, "ab") as stream:
        stream.write(b"\0" * 32)
    with pytest.raises(enprod.EnsembleRefusal, match="not the same bytes"):
        _override_report(engine_written_root)


def _mutate_declared_inventory(root, index, mutate):
    """Apply ``mutate`` to one member's REAL declared inventory list.

    The inventory being altered is the one
    ``gpuwm.ensemble.wrfout_inventory.member_inventory`` wrote over the
    files the runner actually produced, so each probe below changes
    exactly one thing about a genuine record and leaves the rest of the
    manifest -- and every byte on disk -- as the writer left it.  A
    hand-built inventory could not make that claim.
    """
    path = root / ENSEMBLE_MANIFEST_NAME
    document = json.loads(path.read_text("utf-8"))
    record = document["members"][index]
    inventory = mutate([dict(entry) for entry in record[
        WRFOUT_INVENTORY_KEY]])
    record[WRFOUT_INVENTORY_KEY] = inventory
    record["wrfout_count"] = len(inventory)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _drop_contract_and_relabel_domain(inventory):
    del inventory[0]["contract"]
    inventory[0]["domain"] = "d99"
    return inventory


def _size_as_a_sentence(inventory):
    inventory[0]["size_bytes"] = "same as whatever is on disk"
    return inventory


def _declare_the_same_file_twice(inventory):
    return [inventory[0], dict(inventory[0])]


def _add_a_phantom_frame(inventory):
    inventory[0]["frames"] = list(inventory[0]["frames"]) + [
        {"index": 99, "valid_time": "2099-01-01_00:00:00"}]
    return inventory


@pytest.mark.parametrize("mutate, expected", [
    (_drop_contract_and_relabel_domain, "contract"),
    (_size_as_a_sentence, "size_bytes"),
    (_declare_the_same_file_twice, "more than once"),
    (_add_a_phantom_frame, "did not resolve"),
])
def test_a_malformed_inventory_does_not_verify(engine_written_root,
                                               mutate, expected):
    """The four inventories re-verification found still VERIFIED.

    Each was reported ``verified=[1]`` by a verifier that stamps
    ``WRFOUT_INVENTORY_CONTRACT`` on its own report:

    * no ``contract`` key at all and ``domain="d99"`` on a d02 file --
      the version label was never compared and the domain never checked
      against the filename it came from;
    * ``size_bytes="same as whatever is on disk"`` -- the size check ran
      only ``if isinstance(declared_size, int)``, so a malformed value
      did not fail the check, it DISABLED it;
    * the same record declared twice with ``wrfout_count`` adjusted to
      two -- the verifier builds ``{path: record}`` and a dict keeps the
      last one, so one of the two claims was never examined;
    * one real frame plus a phantom at index 99 in 2099 --
      ``declared_frames=2, bound_frames=1``, because only "every found
      frame is declared" was ever proved, never the reverse.

    Each is now a refusal.  The bytes on disk are untouched in every
    case, which is the point: these are inventories that bind nothing,
    not stale files, and the verifier's attestation was stronger than
    what it checked.
    """

    _fail_one_member(engine_written_root, 1)
    _mutate_declared_inventory(engine_written_root, 1, mutate)
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        _override_report(engine_written_root)
    message = str(caught.value)
    assert "member 1" in message
    assert expected in message


@pytest.mark.parametrize("digest", [
    "0" * 63,                             # one hex digit short
    "0" * 65,                             # one too many
    "not-a-digest",
    None,
])
def test_an_inventory_digest_that_is_not_a_digest_is_refused(
        engine_written_root, digest):
    """The field the whole binding rests on has a shape.

    A ``sha256`` that cannot be a sha256 used to be compared to the real
    one and reported as a hash mismatch -- honest, but it says the bytes
    are stale when the truth is that nothing was ever declared about
    them.  Four values because a length check alone passes ``"not a
    digest but 64 characters long ................................."``.
    """

    _fail_one_member(engine_written_root, 1)
    _mutate_declared_inventory(
        engine_written_root, 1,
        lambda inventory: [{**inventory[0], "sha256": digest}])
    with pytest.raises(enprod.EnsembleRefusal, match="64 lowercase hex"):
        _override_report(engine_written_root)


@pytest.mark.parametrize("frames", [
    [{"index": -1, "valid_time": _STAMPS[0]}],
    [{"index": 0, "valid_time": _STAMPS[0]},
     {"index": 0, "valid_time": _STAMPS[1]}],
    [{"index": True, "valid_time": _STAMPS[0]}],
    [{"index": 0, "valid_time": _STAMPS[0], "note": "extra"}],
])
def test_an_inventory_frame_record_that_is_not_one_is_refused(
        engine_written_root, frames):
    """A frame index is a non-negative, unique position in a file.

    ``True`` is included because ``isinstance(True, int)`` is True in
    Python and a bool smuggled into an index would compare equal to
    frame 1.
    """

    _fail_one_member(engine_written_root, 1)
    _mutate_declared_inventory(
        engine_written_root, 1,
        lambda inventory: [{**inventory[0], "frames": frames}])
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        _override_report(engine_written_root)
    assert "member 1" in str(caught.value)


@pytest.mark.parametrize("path_value", [
    "member_001\\wrfout_d02_1970-01-01_18-00-00.nc",
    "./wrfout_d02_1970-01-01_18-00-00.nc",
])
def test_a_non_canonical_inventory_path_is_refused(engine_written_root,
                                                   path_value):
    """One file, one spelling.

    Every check downstream keys on the path string, so a Windows-spelled
    or dot-prefixed duplicate of the canonical one is a second entry for
    the same bytes as far as any of them can tell.
    """

    _fail_one_member(engine_written_root, 1)
    _mutate_declared_inventory(
        engine_written_root, 1,
        lambda inventory: [{**inventory[0], "path": path_value}])
    with pytest.raises(enprod.EnsembleRefusal) as caught:
        _override_report(engine_written_root)
    assert "canonical" in str(caught.value) or "relative path" in str(
        caught.value)


def test_an_unaltered_engine_inventory_still_verifies(engine_written_root):
    """The control the refusals above are only meaningful against.

    Every probe in this section changes ONE field of the record the real
    writer produced.  This is that record, unchanged, still verifying --
    so none of the refusals above can be explained by a schema check that
    simply rejects the writer's own output.
    """

    _fail_one_member(engine_written_root, 1)
    report = _override_report(engine_written_root)
    assert report["verified"] == [1]
    assert report["detail"][0]["declared_frames"] == 2
    assert report["detail"][0]["bound_frames"] == 2


def test_the_engine_records_an_inventory_for_every_finished_member(
        engine_written_root):
    document = json.loads(
        (engine_written_root / ENSEMBLE_MANIFEST_NAME).read_text("utf-8"))
    for record in document["members"]:
        inventory = record[WRFOUT_INVENTORY_KEY]
        assert isinstance(inventory, list) and len(inventory) == 1
        assert record["wrfout_count"] == len(inventory)
        entry = inventory[0]
        assert entry["domain"] == "d02"
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] > 0
        assert [frame["index"] for frame in entry["frames"]] == [0, 1]
        assert [frame["valid_time"] for frame in entry["frames"]] \
            == list(_STAMPS)
        # Relative and POSIX-spelled: the inventory must survive the
        # ensemble being moved or archived.
        assert not entry["path"].startswith("/")
        assert "\\" not in entry["path"]


# ---------------------------------------------------------------------------
# Seam 2 -- the perturbation caller post-condition
# ---------------------------------------------------------------------------


def numpy_pseudo_state(nz=8, ny=16, nx=20, dz=1150.0):
    """A host state carrying exactly what both sides of the seam read.

    ``gpuwm.core.state.init_at_rest`` allocates on the device whenever one
    is visible, which would make every test below a ``gpu`` test for no
    reason: the seam is an ordering question, not a kernel question.  This
    namespace carries the mass/geopotential/coefficient set
    ``_update_diagnostics_numpy`` reads and the ``thp``/``qv``/``u``/``v``/
    ``p`` set ``gpuwm.da.perturb`` reads, and nothing else, so a field
    either side starts depending on shows up as an AttributeError here
    rather than as a silent divergence in a run.
    """

    f32 = np.float32
    z_w = np.arange(nz + 1, dtype=np.float64) * dz
    return types.SimpleNamespace(
        thp=np.zeros((nz, ny, nx), f32),
        qv=np.full((nz, ny, nx), 8.0e-3, f32),
        u=np.zeros((nz, ny, nx + 1), f32),
        v=np.zeros((nz, ny + 1, nx), f32),
        w=np.zeros((nz + 1, ny, nx), f32),
        mub2d=np.full((ny, nx), 90000.0, f32),
        mup=np.zeros((ny, nx), f32),
        thb=np.full((nz,), 300.0, f32),
        phb=((9.81 * z_w).astype(f32)[:, None, None]
             * np.ones((1, ny, nx), f32)),
        php=np.zeros((nz + 1, ny, nx), f32),
        alb=np.zeros((nz,), f32),
        rdnw=np.full((nz,), -float(nz), f32),
        c1h=np.ones((nz,), f32), c2h=np.zeros((nz,), f32),
        c3h=np.ones((nz,), f32), c4h=np.zeros((nz,), f32),
        c3f=np.ones((nz + 1,), f32), c4f=np.zeros((nz + 1,), f32),
        p_top=f32(10000.0),
        p=np.zeros((nz, ny, nx), f32),
        al=np.zeros((nz, ny, nx), f32),
        alt=np.zeros((nz, ny, nx), f32))


_PERTURB_OPTIONS = {
    # 7 km sits inside gpuwm.da.perturb's admission band on this 16x20
    # grid of 3 km cells: at or above 2 grid spacings (6 km) and at or
    # below span/(2*pi) (48/6.283 = 7.64 km), which is where the
    # documented spectral peak k=1/L is actually resolved.  12 km was
    # admitted under the old quarter-span limit and produced a
    # domain-wide offset whose peak fell below the fundamental.
    "fields": [{"name": "t", "amplitude": 0.5, "length_scale_km": 7.0}],
    "rim_width": 2,
}


def test_the_perturbation_stales_the_diagnostics_and_the_refresh_fixes_it():
    """The post-condition is load-bearing, demonstrated both ways.

    ``gpuwm.da.perturb`` says in its own provenance that ``p``/``al``/
    ``alt`` are stale afterwards.  Show that they are -- byte-identical
    across a theta perturbation -- and that the engine's
    ``refresh_diagnostics`` moves them.  A test that only asserted the
    refresh runs would pass against a refresh that did nothing.
    """

    from gpuwm.core.diagnostics import update_diagnostics

    state = numpy_pseudo_state()
    update_diagnostics(state, 1)
    before = state.p.copy()
    assert np.all(before > 0.0)

    provenance = perturb.apply_perturbations(
        state, 12345,
        perturb.PerturbationConfig.from_mapping(
            {"dx_km": 3.0, "dy_km": 3.0, **_PERTURB_OPTIONS}))
    assert np.abs(state.thp).max() > 0.0
    assert any("update_diagnostics" in line
               for line in provenance["post_conditions"])
    # Stale: the perturbation moved theta and did not touch pressure.
    assert np.array_equal(before, state.p)

    receipt = refresh_diagnostics(state, hypsometric_opt=1)
    assert receipt["ran"] is True and receipt["moved"] is True
    assert receipt["sha256_before"] != receipt["sha256_after"]
    assert not np.array_equal(before, state.p)
    assert np.all(np.isfinite(state.p))


def test_refreshing_an_untouched_state_is_the_identity():
    """Idempotence, so 'moved' means what the manifest says it means."""

    from gpuwm.core.diagnostics import update_diagnostics

    state = numpy_pseudo_state()
    update_diagnostics(state, 1)
    settled = diagnostics_sha256(state)
    receipt = refresh_diagnostics(state, hypsometric_opt=1)
    assert receipt["moved"] is False
    assert receipt["sha256_after"] == settled


def test_the_engine_supplies_the_grid_spacing_the_perturb_lane_requires():
    """``apply_perturbations`` takes a config; the engine has a TOML table.

    The engine passed ``dict(perturbation_options)`` straight through, and
    ``apply_perturbations`` refuses anything that is not a
    ``PerturbationConfig`` -- so ``perturbation = "gpuwm.da.perturb"``
    raised ``TypeError`` on the first member.  Neither lane's suite could
    see it: the engine only ever resolved the hook, and the perturb lane
    only ever built the config itself.
    """

    from gpuwm.core.diagnostics import update_diagnostics

    state = numpy_pseudo_state()
    update_diagnostics(state, 1)
    hook = resolve_perturbation("gpuwm.da.perturb")
    report = hook(state, 4242, dict(_PERTURB_OPTIONS),
                  grid_spacing_km=(3.0, 3.0))
    assert report["schema"] == "gpuwm.da.perturb/provenance/v1"
    # The engine's dx reached the module, not a default.
    assert report["grid_spacing_km"] == {"dx": 3.0, "dy": 3.0}
    assert np.abs(state.thp).max() > 0.0


def test_an_overlay_that_states_the_grid_spacing_is_refused():
    """Two sources for dx is a wrong perturbation that still runs."""

    state = numpy_pseudo_state()
    hook = resolve_perturbation("gpuwm.da.perturb")
    with pytest.raises(ValueError, match="dx_km"):
        hook(state, 1, {"dx_km": 1.0, **_PERTURB_OPTIONS},
             grid_spacing_km=(3.0, 3.0))


def test_the_da_lane_hook_refuses_to_run_without_a_grid():
    state = numpy_pseudo_state()
    hook = resolve_perturbation("gpuwm.da.perturb")
    with pytest.raises(ValueError, match="grid spacing"):
        hook(state, 1, dict(_PERTURB_OPTIONS))


@pytest.fixture()
def stubbed_runtime(monkeypatch, tmp_path):
    """``run_member`` with everything but the seam replaced.

    The dycore, the ingest, and the I/O are stubbed; the perturbation
    resolution, the grid-spacing supply, the post-condition refresh, and
    the manifest detail are the shipped code.  That is the point -- the
    seam sits between preparation and integration and can be tested
    without either.
    """

    from gpuwm import case_data, runtime
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.ingest import preflight
    from gpuwm.io import wrfout as wrfout_io

    state = numpy_pseudo_state()
    update_diagnostics(state, 1)

    run = types.SimpleNamespace(dx=3000.0, dy=3000.0, hypsometric_opt=1)
    dc = types.SimpleNamespace(run=run, history_interval_s=60.0, grid_id=1)
    exp = types.SimpleNamespace(domains=[dc], run_seconds=120.0,
                                start_time=None, restart_interval_s=60.0,
                                domain=lambda gid: dc)
    data = types.SimpleNamespace(output_title="t", output_domain=1)
    prepared = types.SimpleNamespace(
        initial_result=types.SimpleNamespace(state=state))

    def integrate(outdir, prep, **kwargs):
        # A member the engine did not watch move is a refusal, so the
        # stub has to actually advance the state it was handed.
        prep.initial_result.state.thp[...] += np.float32(0.25)
        return types.SimpleNamespace(completed_seconds=120.0,
                                     wrfout_paths=[])

    monkeypatch.setattr(case_data, "load_experiment_case",
                        lambda path: (exp, data))
    monkeypatch.setattr(preflight, "build_input_catalog", lambda d: {})
    monkeypatch.setattr(wrfout_io, "quarantine_orphan_wrfouts",
                        lambda outdir: None)
    monkeypatch.setattr(runtime, "single_domain", lambda e: dc)
    monkeypatch.setattr(runtime, "forcing_snapshots", lambda d, cat: {})
    monkeypatch.setattr(runtime, "forcing_schedule",
                        lambda e, d, snaps: None)
    monkeypatch.setattr(runtime, "prepare_experiment_case",
                        lambda e, d, **kw: prepared)
    monkeypatch.setattr(runtime, "integrate_prepared_case", integrate)
    return state, tmp_path


def test_run_member_honours_the_post_condition_and_records_it(
        stubbed_runtime):
    """The seam, end to end through the engine's own member path."""

    state, tmp_path = stubbed_runtime
    settled = diagnostics_sha256(state)
    outcome = run_member(
        base_config=tmp_path / "base.toml", member_dir=tmp_path / "m0",
        index=0, seed=4242, perturbation="gpuwm.da.perturb",
        perturbation_options=dict(_PERTURB_OPTIONS))

    post = outcome.perturbation["post_conditions"]
    assert outcome.perturbation["provenance"] == "gpuwm.da.perturb"
    assert outcome.perturbation["changed_state"] is True
    assert post["ran"] is True
    assert post["moved"] is True, (
        "a theta perturbation must move p/al/alt; 'moved: false' here "
        "means the refresh did not reach the state being integrated")
    assert post["hypsometric_opt"] == 1
    assert post["sha256_before"] == settled
    assert post["contract"].endswith("update_diagnostics")
    # The receipt is JSON-serialisable, because it goes into the manifest.
    json.dumps(outcome.perturbation)
    assert live_state_sha256(state) == outcome.final_state_sha256


def test_run_member_skips_the_refresh_when_nothing_was_perturbed(
        stubbed_runtime):
    """``perturbation = none`` leaves the prepared diagnostics alone."""

    state, tmp_path = stubbed_runtime
    before = diagnostics_sha256(state)
    outcome = run_member(
        base_config=tmp_path / "base.toml", member_dir=tmp_path / "m0",
        index=0, seed=1, perturbation="none")
    post = outcome.perturbation["post_conditions"]
    assert outcome.perturbation["changed_state"] is False
    assert post["ran"] is False
    assert "byte-identical" in post["reason"]
    assert diagnostics_sha256(state) == before


# ---------------------------------------------------------------------------
# Seam 2b -- F-06: on a restart leg the guard's baseline is the RESTORED
#            object, hash and clock
# ---------------------------------------------------------------------------


def _stub_runtime(monkeypatch, integrate, *, state=None):
    """``run_member``'s runtime with a caller-supplied integrator.

    The same stubbing ``stubbed_runtime`` does, with the integration step
    under the test's control: F-06 is entirely about what the engine
    concludes from an integrator that did not do what it said.
    """

    from gpuwm import case_data, runtime
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.ingest import preflight
    from gpuwm.io import wrfout as wrfout_io

    if state is None:
        state = numpy_pseudo_state()
        update_diagnostics(state, 1)
    state.elapsed_seconds = 0.0

    run = types.SimpleNamespace(dx=3000.0, dy=3000.0, hypsometric_opt=1)
    dc = types.SimpleNamespace(run=run, history_interval_s=60.0, grid_id=1)
    exp = types.SimpleNamespace(domains=[dc], run_seconds=120.0,
                                start_time=None, restart_interval_s=60.0,
                                domain=lambda gid: dc)
    data = types.SimpleNamespace(output_title="t", output_domain=1)
    prepared = types.SimpleNamespace(
        initial_result=types.SimpleNamespace(state=state))

    monkeypatch.setattr(case_data, "load_experiment_case",
                        lambda path: (exp, data))
    monkeypatch.setattr(preflight, "build_input_catalog", lambda d: {})
    monkeypatch.setattr(wrfout_io, "quarantine_orphan_wrfouts",
                        lambda outdir: None)
    monkeypatch.setattr(runtime, "single_domain", lambda e: dc)
    monkeypatch.setattr(runtime, "forcing_snapshots", lambda d, cat: {})
    monkeypatch.setattr(runtime, "forcing_schedule",
                        lambda e, d, snaps: None)
    monkeypatch.setattr(runtime, "prepare_experiment_case",
                        lambda e, d, **kw: prepared)
    monkeypatch.setattr(runtime, "integrate_prepared_case", integrate)
    return state


def _write_restart_checkpoint(path, state, *, elapsed_seconds, offset=1.0):
    """A checkpoint whose state differs from the prepared one.

    ``offset`` displaces every serialised array, so the checkpoint's hash
    and the prepared state's hash cannot coincide -- which is what makes
    "the engine hashed the prepared state" detectable at all.
    """

    payload = {}
    for name in serialized_state_attrs():
        value = getattr(state, name, None)
        if value is None:
            continue
        payload[f"state/{name}"] = np.asarray(value) + np.asarray(
            offset, dtype=np.asarray(value).dtype)
    payload["meta/elapsed_seconds"] = np.asarray(float(elapsed_seconds))
    np.savez(path, **payload)
    return path


def _restore_into(state, checkpoint):
    """What ``restore_restart`` does: overwrite the state from the file."""

    with np.load(checkpoint, allow_pickle=False) as data:
        for name in serialized_state_attrs():
            key = f"state/{name}"
            if key in data.files and getattr(state, name, None) is not None:
                getattr(state, name)[...] = data[key]


def test_a_no_op_integrator_on_a_restart_leg_is_refused(monkeypatch,
                                                        tmp_path):
    """The probe that reopened F-06, promoted to a permanent test.

    A no-op integrator returned a SUCCESSFUL outcome claiming a 60 s
    advance: the guard compared the untouched PREPARED state's hash
    against the checkpoint's, found them different -- of course they are,
    the restore is what would have made them equal -- and concluded the
    runner had advanced the object.  Nothing had restored anything.
    """

    def noop(outdir, prep, **kwargs):
        return types.SimpleNamespace(completed_seconds=60.0, wrfout_paths=[])

    state = _stub_runtime(monkeypatch, noop)
    checkpoint = _write_restart_checkpoint(
        tmp_path / "gpuwmrst_000.npz", state, elapsed_seconds=60.0)
    with pytest.raises(RuntimeError) as caught:
        run_member(base_config=tmp_path / "base.toml",
                   member_dir=tmp_path / "m0", index=0, seed=1,
                   perturbation="none", restart=checkpoint)
    message = str(caught.value)
    assert "PRE-RESTORE" in message
    assert "did not happen" in message


def test_a_restart_that_restores_but_does_not_integrate_is_refused(
        monkeypatch, tmp_path):
    """The clock baseline, taken from the RESTORED object.

    This integrator does restore -- so the hash check alone cannot see
    the problem -- and then advances nothing.  The prepared state's clock
    was 0 and the restored clock is 60, so the old baseline read that as
    a 60 s advance and recorded the member.  Measured against the
    checkpoint's own clock, which is what the restored object carries,
    nothing moved.
    """

    checkpoint_path = tmp_path / "gpuwmrst_000.npz"

    def restore_only(outdir, prep, **kwargs):
        _restore_into(prep.initial_result.state, checkpoint_path)
        prep.initial_result.state.elapsed_seconds = 60.0
        return types.SimpleNamespace(completed_seconds=60.0, wrfout_paths=[])

    state = _stub_runtime(monkeypatch, restore_only)
    _write_restart_checkpoint(checkpoint_path, state, elapsed_seconds=60.0)
    with pytest.raises(RuntimeError) as caught:
        run_member(base_config=tmp_path / "base.toml",
                   member_dir=tmp_path / "m0", index=0, seed=1,
                   perturbation="none", restart=checkpoint_path)
    message = str(caught.value)
    assert "byte-identical" in message
    assert "clock did not advance past the checkpoint" in message


def test_a_restart_that_restores_and_advances_is_accepted(monkeypatch,
                                                          tmp_path):
    """The positive control: a real leg must still be recorded.

    Without it the two refusals above could be satisfied by a guard that
    refuses everything, which is not a fix.
    """

    checkpoint_path = tmp_path / "gpuwmrst_000.npz"

    def restore_and_advance(outdir, prep, **kwargs):
        target = prep.initial_result.state
        _restore_into(target, checkpoint_path)
        target.thp[...] += np.float32(0.25)
        target.elapsed_seconds = 120.0
        return types.SimpleNamespace(completed_seconds=60.0, wrfout_paths=[])

    state = _stub_runtime(monkeypatch, restore_and_advance)
    _write_restart_checkpoint(checkpoint_path, state, elapsed_seconds=60.0)
    outcome = run_member(base_config=tmp_path / "base.toml",
                         member_dir=tmp_path / "m0", index=0, seed=1,
                         perturbation="none", restart=checkpoint_path)
    assert outcome.initial_state_sha256 == checkpoint_state_sha256(
        checkpoint_path), (
        "a restart leg's initial state is the checkpoint it restored")
    assert outcome.final_state_sha256 == live_state_sha256(state)
    assert outcome.sim_seconds == 60.0
    # And the leg was not perturbed, because a restart leg never is.
    assert outcome.perturbation["applied"] is False


def test_an_exact_steady_state_restart_leg_is_not_falsely_refused(
        monkeypatch, tmp_path):
    """A leg whose state is unchanged but whose clock moved is honest.

    The hash cannot distinguish "restored and integrated an exact steady
    state" from "did nothing"; the clock can, and it is the checkpoint's
    clock that it is measured against.
    """

    checkpoint_path = tmp_path / "gpuwmrst_000.npz"

    def restore_and_tick(outdir, prep, **kwargs):
        _restore_into(prep.initial_result.state, checkpoint_path)
        prep.initial_result.state.elapsed_seconds = 120.0
        return types.SimpleNamespace(completed_seconds=60.0, wrfout_paths=[])

    state = _stub_runtime(monkeypatch, restore_and_tick)
    _write_restart_checkpoint(checkpoint_path, state, elapsed_seconds=60.0)
    outcome = run_member(base_config=tmp_path / "base.toml",
                         member_dir=tmp_path / "m0", index=0, seed=1,
                         perturbation="none", restart=checkpoint_path)
    assert outcome.final_state_sha256 == checkpoint_state_sha256(
        checkpoint_path)


def test_a_no_op_integrator_without_a_restart_is_still_refused(monkeypatch,
                                                               tmp_path):
    """The non-restart half of the guard, unchanged by this fix."""

    def noop(outdir, prep, **kwargs):
        return types.SimpleNamespace(completed_seconds=60.0, wrfout_paths=[])

    _stub_runtime(monkeypatch, noop)
    with pytest.raises(RuntimeError, match="byte-identical"):
        run_member(base_config=tmp_path / "base.toml",
                   member_dir=tmp_path / "m0", index=0, seed=1,
                   perturbation="none")


# ---------------------------------------------------------------------------
# Seam 3 -- GridGeometry from a run artifact
# ---------------------------------------------------------------------------


def analytic_target_grid(nx=25, ny=21, dx=3000.0, nz=8, top_m=12000.0):
    """The radar lane's own grid object, built the way its tests build it."""

    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, top_m, nz + 1), name="analytic")


def write_georeferenced_wrfout(path, grid, *, corrupt_lat=False):
    """A wrfout carrying exactly what ``TargetGrid.from_wrfout`` reads."""

    netCDF4 = pytest.importorskip("netCDF4")
    lat = np.array(grid.lat, dtype=np.float32)
    if corrupt_lat:
        lat[grid.ny // 2, grid.nx // 2] += np.float32(0.01)
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("south_north", grid.ny)
        dataset.createDimension("west_east", grid.nx)
        dataset.createDimension("bottom_top_stag", grid.nz + 1)
        for key, value in (("MAP_PROJ", np.int32(1)),
                           ("TRUELAT1", grid.truelat1),
                           ("TRUELAT2", grid.truelat2),
                           ("STAND_LON", grid.stand_lon),
                           ("CEN_LAT", grid.ref_lat),
                           ("CEN_LON", grid.ref_lon),
                           ("DX", grid.dx_m), ("DY", grid.dy_m)):
            dataset.setncattr(key, value)

        def put(name, dims, values):
            variable = dataset.createVariable(name, "f4", dims)
            variable[0] = np.asarray(values, dtype=np.float32)

        put("XLAT", ("Time", "south_north", "west_east"), lat)
        put("XLONG", ("Time", "south_north", "west_east"), grid.lon)
        put("HGT", ("Time", "south_north", "west_east"), grid.terrain_m)
        geopotential = np.asarray(grid.z_w, dtype=np.float64) * 9.81
        put("PHB", ("Time", "bottom_top_stag", "south_north", "west_east"),
            geopotential)
        put("PH", ("Time", "bottom_top_stag", "south_north", "west_east"),
            np.zeros_like(geopotential))
    return path


def test_grid_geometry_lands_on_the_radar_lanes_own_layer_midpoints():
    """The operator and the observations must mean one cell.

    ``TargetGrid`` places a gate in the layer whose interfaces bracket it,
    in that gate's own column.  H(x) is evaluated at mass points.  The
    mass point of that layer is the midpoint of the two interfaces that
    defined it, exactly -- anything else makes an innovation a difference
    between two different places.
    """

    from gpuwm.da import obsop

    grid = analytic_target_grid()
    geometry = obsop.GridGeometry.from_target_grid(grid)
    assert geometry.shape == (grid.nz, grid.ny, grid.nx)
    expected = 0.5 * (grid.z_w[:-1] + grid.z_w[1:])
    np.testing.assert_array_equal(np.asarray(geometry.height_m), expected)
    np.testing.assert_array_equal(np.asarray(geometry.latitude_deg),
                                  np.asarray(grid.lat, dtype=np.float64))


def test_grid_geometry_from_a_wrfout_is_the_same_geometry(tmp_path):
    """The cycle driver's supply route, against the artifact."""

    from gpuwm.da import obsop

    grid = analytic_target_grid()
    path = write_georeferenced_wrfout(tmp_path / "wrfout_d02_x.nc", grid)
    from_file = obsop.GridGeometry.from_wrfout(path)
    from_grid = obsop.GridGeometry.from_target_grid(grid)
    # float32 storage in the file is the only difference allowed.
    np.testing.assert_allclose(np.asarray(from_file.height_m),
                               np.asarray(from_grid.height_m), rtol=1e-6,
                               atol=1e-3)
    np.testing.assert_allclose(np.asarray(from_file.latitude_deg),
                               np.asarray(from_grid.latitude_deg), atol=1e-5)


def test_a_wrfout_that_contradicts_its_own_georeference_is_refused(tmp_path):
    """The check is why this route exists rather than a raw XLAT read.

    Reading XLAT/XLONG straight out of the file is shorter and skips the
    projection cross-check; a nest with an off-centre reference point then
    shifts every observation by a cell, silently.
    """

    from gpuwm.da import obsop
    from gpuwm.obs.target_grid import GridMismatchError

    grid = analytic_target_grid()
    path = write_georeferenced_wrfout(tmp_path / "bad.nc", grid,
                                      corrupt_lat=True)
    with pytest.raises(GridMismatchError, match="XLAT"):
        obsop.GridGeometry.from_wrfout(path)


def test_radial_velocity_accepts_the_supplied_geometry(tmp_path):
    """The seam's whole point: Vr runs without the state knowing where it is.

    ``GridGeometry.from_state`` raises when the state has no XLAT/XLONG,
    by design -- it will not guess where a domain is.  The supply route
    has to close that, and this is the call the cycle driver makes.
    """

    from gpuwm.da import obsop

    grid = analytic_target_grid()
    geometry = obsop.GridGeometry.from_wrfout(
        write_georeferenced_wrfout(tmp_path / "wrfout_d02_y.nc", grid))
    state = numpy_pseudo_state(nz=grid.nz, ny=grid.ny, nx=grid.nx)
    with pytest.raises(ValueError, match="XLAT"):
        obsop.GridGeometry.from_state(state)

    state.u[...] = np.float32(12.0)
    state.v[...] = np.float32(-4.0)
    state.sina = np.zeros((grid.ny, grid.nx), np.float32)
    state.cosa = np.ones((grid.ny, grid.nx), np.float32)
    site = obsop.RadarSite(latitude_deg=float(grid.lat[grid.ny // 2,
                                                       grid.nx // 2]),
                           longitude_deg=float(grid.lon[grid.ny // 2,
                                                        grid.nx // 2]),
                           altitude_m=370.0, name="site")
    # Air motion only: the pseudo-state carries no hydrometeors, and the
    # operator refuses to invent a fall speed for species it cannot see.
    vr = obsop.radial_velocity(state, site, geometry, fall_speed="none")
    vr = np.asarray(vr)
    assert vr.shape == (grid.nz, grid.ny, grid.nx)
    # A uniform wind of 12.6 m/s cannot project to more than its own speed.
    finite = vr[np.isfinite(vr)]
    assert finite.size > 0
    assert np.abs(finite).max() <= np.hypot(12.0, 4.0) + 1e-5
    # East of the radar the flow is outbound, west of it inbound: the
    # sign convention is positive away, and it is load-bearing.
    mid_j, mid_i = grid.ny // 2, grid.nx // 2
    assert vr[0, mid_j, mid_i + 6] > 0.0
    assert vr[0, mid_j, mid_i - 6] < 0.0


# ---------------------------------------------------------------------------
# Seam 4 -- gpuwm-obs.radar-grid.v1 into the filter's observation structures
# ---------------------------------------------------------------------------


def two_radar_grid_file(path, grid, *, overwrite=False):
    """A real ``gpuwm-obs.radar-grid.v1`` file, through the real writer.

    Two radars, deliberately looking from opposite sides, each seeing the
    *same* cells: that is the configuration in which averaging the two
    projections would look plausible and be wrong, so it is the one the
    adapter has to be tested on.
    """

    pytest.importorskip("netCDF4")
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import GriddedObservations, SuperobParams

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    shape = (nz, ny, nx)
    rng = np.random.default_rng(7)

    z_mask = np.zeros(shape, np.int8)
    z_mask[1:4, ny // 3: 2 * ny // 3, nx // 3: 2 * nx // 3] = 1
    z_obs = np.where(z_mask.astype(bool),
                     35.0 + rng.normal(0.0, 2.0, shape), -35.0)
    z_err = np.where(z_mask.astype(bool), 3.0, 0.0)

    vr_mask = np.zeros((2,) + shape, np.int8)
    vr_mask[:, 1:4, ny // 3: 2 * ny // 3, nx // 3: 2 * nx // 3] = 1
    # Radar 0 due west of the cells (beam points east), radar 1 due east
    # (beam points west).  One wind, two opposite projections.
    east = np.zeros((2,) + shape)
    east[0] = 1.0
    east[1] = -1.0
    north = np.zeros((2,) + shape)
    up = np.zeros((2,) + shape)
    truth_u = 14.0
    vr_obs = east * truth_u
    vr_err = np.where(vr_mask.astype(bool), 1.5, 0.0)

    observations = GriddedObservations(
        z_obs=z_obs.astype(np.float32), z_mask=z_mask,
        z_err=z_err.astype(np.float32),
        z_max=(z_obs + 1.0).astype(np.float32),
        z_mean=(z_obs - 1.0).astype(np.float32),
        z_count=z_mask.astype(np.int32),
        vr_obs=vr_obs.astype(np.float32), vr_mask=vr_mask,
        vr_err=vr_err.astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((2,) + shape, np.int32),
        vr_beam_east=east.astype(np.float32),
        vr_beam_north=north.astype(np.float32),
        vr_beam_up=up.astype(np.float32),
        radars=[{"id": "AAAA", "lat_deg": float(grid.lat[ny // 2, 0]),
                 "lon_deg": float(grid.lon[ny // 2, 0]), "alt_m": 300.0,
                 "valid_time": "1970-01-01T18:00:00Z"},
                {"id": "BBBB", "lat_deg": float(grid.lat[ny // 2, nx - 1]),
                 "lon_deg": float(grid.lon[ny // 2, nx - 1]), "alt_m": 320.0,
                 "valid_time": "1970-01-01T18:00:00Z"}],
        counts=[], provenance=[])
    write_radar_grid(path, observations, grid,
                     valid_time="1970-01-01T18:00:00Z",
                     params=SuperobParams(), overwrite=overwrite)
    return path, truth_u


def test_the_adapter_keeps_one_batch_per_radar_and_never_averages(tmp_path):
    """The schema deviation the radar lane asked the DA lanes to honour.

    Two radars on opposite sides of one storm measure +14 and -14 m/s of
    the same westerly.  Their mean is 0 -- a number with no observation
    operator behind it.  One GriddedObs per radar is the only reading that
    keeps both.
    """

    from gpuwm.da import obs_radar

    grid = analytic_target_grid()
    path, truth_u = two_radar_grid_file(tmp_path / "obs.nc", grid)
    shape = (grid.nz, grid.ny, grid.nx)

    def simulate(radar_index, radar):
        beam = obs_radar.beam_unit_vectors(
            obs_radar.read_document(path, expected_grid=grid), radar_index)
        members = [obs_radar.simulated_radial_velocity(
            np.full(shape, truth_u + offset), np.zeros(shape),
            np.zeros(shape), beam) for offset in (-1.0, 0.0, 1.0)]
        return np.stack(members)

    batches, provenance = obs_radar.radar_grid_to_gridded_obs(
        path, velocity_simulated=simulate, expected_grid=grid,
        expected_grid_identity=grid.identity_sha256())

    assert [batch.name for batch in batches] == ["vr:AAAA", "vr:BBBB"]
    masked = batches[0].mask
    # Opposite signs survive: nothing collapsed them.
    assert batches[0].values[masked].mean() == pytest.approx(truth_u, abs=1e-4)
    assert batches[1].values[masked].mean() == pytest.approx(-truth_u,
                                                             abs=1e-4)
    # And H(x) tracks the beam, so the innovation is near zero on BOTH --
    # which is only true because the operator direction came from the file.
    for batch in batches:
        innovation = (batch.values - batch.simulated[1])[batch.mask]
        assert np.abs(innovation).max() < 1e-4
    assert provenance["schema"] == "gpuwm-da.radar-obs-adapter.v1"
    assert {entry["radar"] for entry in provenance["batches"]} == {"AAAA",
                                                                   "BBBB"}


def test_a_member_wind_of_the_wrong_sign_shows_up_on_both_radars(tmp_path):
    """The test above would pass on averaged projections if H were averaged.

    Give the members an easterly instead.  A correct per-radar operator
    reports +28 innovation on one radar and -28 on the other; anything
    that averaged would report zero on both and look perfect.
    """

    from gpuwm.da import obs_radar

    grid = analytic_target_grid()
    path, truth_u = two_radar_grid_file(tmp_path / "obs.nc", grid)
    shape = (grid.nz, grid.ny, grid.nx)
    document = obs_radar.read_document(path, expected_grid=grid)

    def simulate(radar_index, radar):
        beam = obs_radar.beam_unit_vectors(document, radar_index)
        wrong = obs_radar.simulated_radial_velocity(
            np.full(shape, -truth_u), np.zeros(shape), np.zeros(shape), beam)
        return np.stack([wrong, wrong, wrong])

    batches, _ = obs_radar.radar_grid_to_gridded_obs(
        document, velocity_simulated=simulate, expected_grid=grid)
    first, second = (
        (batch.values - batch.simulated[0])[batch.mask].mean()
        for batch in batches)
    assert first == pytest.approx(2 * truth_u, abs=1e-4)
    assert second == pytest.approx(-2 * truth_u, abs=1e-4)


def test_the_adapter_reads_z_from_the_named_reduction(tmp_path):
    """z_obs vs z_max per the schema doc; z_mean refused with a reason."""

    from gpuwm.da import obs_radar

    grid = analytic_target_grid()
    path, _ = two_radar_grid_file(tmp_path / "obs.nc", grid)
    shape = (grid.nz, grid.ny, grid.nx)
    simulated = np.zeros((3,) + shape)

    default, _ = obs_radar.radar_grid_to_gridded_obs(
        path, reflectivity_simulated=simulated, expected_grid=grid)
    maxima, _ = obs_radar.radar_grid_to_gridded_obs(
        path, reflectivity_simulated=simulated, expected_grid=grid,
        z_source="z_max")
    mask = default[0].mask
    np.testing.assert_allclose(maxima[0].values[mask],
                               default[0].values[mask] + 1.0, atol=1e-4)
    with pytest.raises(obs_radar.RadarObsAdapterError, match="z_mean"):
        obs_radar.radar_grid_to_gridded_obs(
            path, reflectivity_simulated=simulated, expected_grid=grid,
            z_source="z_mean")


def test_the_adapter_refuses_a_file_bound_to_another_grid(tmp_path):
    from gpuwm.da import obs_radar
    from gpuwm.obs.target_grid import GridMismatchError

    grid = analytic_target_grid()
    path, _ = two_radar_grid_file(tmp_path / "obs.nc", grid)
    other = analytic_target_grid(nx=27)
    with pytest.raises(GridMismatchError):
        obs_radar.radar_grid_to_gridded_obs(
            path, reflectivity_simulated=np.zeros(
                (3, grid.nz, grid.ny, grid.nx)),
            expected_grid=other)


def test_the_adapter_refuses_a_zero_sigma_under_a_true_mask(tmp_path):
    """Errors are standard deviations; a zero sigma is not a missing one."""

    from gpuwm.da import obs_radar

    grid = analytic_target_grid()
    path, _ = two_radar_grid_file(tmp_path / "obs.nc", grid)
    document = dict(obs_radar.read_document(path, expected_grid=grid))
    document["variables"] = dict(document["variables"])
    broken = np.array(document["variables"]["z_err"])
    broken[np.asarray(document["variables"]["z_mask"]).astype(bool)] = 0.0
    document["variables"]["z_err"] = broken
    with pytest.raises(obs_radar.RadarObsAdapterError,
                       match="standard deviation"):
        obs_radar.radar_grid_to_gridded_obs(
            document, reflectivity_simulated=np.zeros(
                (3, grid.nz, grid.ny, grid.nx)), expected_grid=grid)


def test_every_observed_variable_in_the_file_is_three_dimensional(tmp_path):
    """The U10/V10-class trap, checked against the artifact.

    A 10 m diagnostic packed beside 3-D fields is the classic way a DA
    adapter ends up assimilating a surface value into a model level.  The
    adapter assumes every observed quantity is on ``(level, south_north,
    west_east)``, optionally behind a leading ``radar`` axis, and takes no
    surface special case anywhere -- so pin the assumption on the file's
    own dimension tuples rather than on a reading of the docs.
    """

    netCDF4 = pytest.importorskip("netCDF4")

    grid = analytic_target_grid()
    path, _ = two_radar_grid_file(tmp_path / "obs.nc", grid)
    plane = ("level", "south_north", "west_east")
    volume = ("radar",) + plane
    georeference = {"XLAT", "XLONG", "HGT", "radar_id", "radar_lat",
                    "radar_lon", "radar_alt", "radar_valid_time"}
    with netCDF4.Dataset(path, "r") as dataset:
        observed = {name: variable.dimensions
                    for name, variable in dataset.variables.items()
                    if name not in georeference}
    assert observed, "the file declared no observed variables at all"
    for name, dims in sorted(observed.items()):
        assert dims in (plane, volume), (
            f"{name} has dimensions {dims}; the adapter treats every "
            "observed variable as 3-D on the model grid, and a surface "
            "diagnostic here would be assimilated into a model level")
    # And the leading axis is on exactly the velocity family, which is the
    # whole reason the adapter emits one batch per radar.
    assert {name for name, dims in observed.items() if dims == volume} == {
        "vr_obs", "vr_mask", "vr_err", "vr_count", "vr_rejected",
        "vr_beam_east", "vr_beam_north", "vr_beam_up"}


def test_the_letkf_geometry_comes_off_the_same_target_grid():
    """The whole 3-D georeference, not a summary of it.

    ``TargetGrid`` carries interface heights per column and the mass-point
    latitude/longitude the observations were placed with.  Reducing that to
    one domain-mean height column and a nominal grid spacing is where the
    filter's "metres" stopped being metres: two columns whose terrain
    differs by 2 km come out at the same height, and a projected grid's
    physical spacing is dx/mapfac, not dx.  Both are on the file; hand both
    over.
    """
    from gpuwm.da import obs_radar

    grid = analytic_target_grid()
    geometry = obs_radar.letkf_grid_geometry(grid)
    assert geometry.nz == grid.nz
    assert geometry.dx_m == grid.dx_m
    midpoints = 0.5 * (grid.z_w[:-1] + grid.z_w[1:])
    assert geometry.heights_m.shape == (grid.nz, grid.ny, grid.nx)
    np.testing.assert_allclose(geometry.heights_m, midpoints, atol=1e-9)
    assert geometry.geodesic
    np.testing.assert_array_equal(geometry.lat_deg, grid.lat)
    np.testing.assert_array_equal(geometry.lon_deg, grid.lon)


def test_the_letkf_geometry_keeps_terrain_out_of_the_vertical_weight():
    """The reduction this replaces, measured on a grid that has terrain.

    A domain-mean column reports one height per level, so every same-level
    pair is zero metres apart vertically whatever the terrain does.  The
    per-column field keeps the spread the grid actually has, and it is not
    small: this is the difference between a vertical localisation weight of
    1 and one well down the Gaspari-Cohn curve.
    """
    from gpuwm.da import obs_radar
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    # The shared analytic fixture is flat, so it cannot show this at all --
    # which is itself worth saying, since a flat fixture is exactly how a
    # domain-mean column passes for a height field.  Build a ridge.
    nx, ny, nz, dx = 25, 21, 8, 3000.0
    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    eta = np.linspace(0.0, 1.0, nz + 1)
    terrain = 1500.0 * np.exp(
        -((np.arange(nx)[None, :] - nx / 2.0) ** 2) / 20.0) * np.ones((ny, 1))
    z_w = (terrain[None] * (1.0 - eta)[:, None, None]
           + 12000.0 * eta[:, None, None])
    grid = TargetGrid.from_projection(projection, z_w=z_w, name="ridge")

    geometry = obs_radar.letkf_grid_geometry(grid)
    zmin, zmax = geometry.level_bounds()
    # Near the surface the same model level spans the terrain's full relief.
    assert float((zmax - zmin).max()) > 1000.0, float((zmax - zmin).max())
    # A domain-mean column would have reported exactly zero spread here.
    midpoints = 0.5 * (np.asarray(grid.z_w)[:-1] + np.asarray(grid.z_w)[1:])
    mean_column = midpoints.reshape(grid.nz, -1).mean(axis=1)
    assert float(np.abs(midpoints[0] - mean_column[0]).max()) > 100.0


# ---------------------------------------------------------------------------
# Seam 5 -- the hydrometeor positivity policy
# ---------------------------------------------------------------------------


def test_clipping_bounds_the_analysis_and_says_it_added_mass():
    """The receipt has to name the direction, because it is counter-intuitive.

    Clipping a negative analysis to zero RAISES it.  Calling that "clipped
    mass" reads as mass removed; it is mass added, always, and always
    wetward for a mixing ratio.
    """

    from gpuwm.da import positivity

    prior = {"qr": np.array([[[1.0e-3, 2.0e-4, 0.0]]]),
             "thp": np.array([[[0.0, 0.0, 0.0]]])}
    increments = {"qr": np.array([[[-2.0e-3, 1.0e-4, -5.0e-4]]]),
                  "thp": np.array([[[-3.0, 1.0, -1.0]]])}
    adjusted, receipt = positivity.apply_positivity(prior, increments)

    analysis = prior["qr"] + adjusted["qr"]
    assert analysis.min() >= 0.0
    np.testing.assert_allclose(analysis, [[[0.0, 3.0e-4, 0.0]]], atol=1e-12)
    # theta is not bounded below and must come through untouched.
    np.testing.assert_array_equal(adjusted["thp"], increments["thp"])
    assert receipt["constrained_fields"] == ["qr"]
    assert receipt["unconstrained_fields"] == ["thp"]
    assert receipt["negative_points"] == 2
    # (2e-3 - 1e-3) + (5e-4 - 0) = 1.5e-3 added.
    assert receipt["mass_added_by_clip"] == pytest.approx(1.5e-3)
    assert receipt["per_field"][0]["worst_negative"] == pytest.approx(-1.0e-3)
    json.dumps(receipt)


def test_reject_reverts_the_whole_column_and_conserves_the_background():
    from gpuwm.da import positivity

    prior = {"qr": np.array([1.0e-3, 2.0e-4]),
             "qs": np.array([1.0e-3, 1.0e-3])}
    increments = {"qr": np.array([-2.0e-3, 1.0e-4]),
                  "qs": np.array([5.0e-4, 5.0e-4])}
    adjusted, receipt = positivity.apply_positivity(prior, increments,
                                                    policy="reject")
    # Point 0 goes negative in qr, so BOTH species revert there -- a column
    # whose species came from different analyses is not a state.
    np.testing.assert_array_equal(adjusted["qr"], [0.0, 1.0e-4])
    np.testing.assert_array_equal(adjusted["qs"], [0.0, 5.0e-4])
    assert receipt["mass_added_by_clip"] == 0.0
    assert receipt["policy"] == "reject"


def test_a_policy_of_none_counts_without_repairing():
    from gpuwm.da import positivity

    prior = {"qr": np.array([1.0e-3])}
    increments = {"qr": np.array([-2.0e-3])}
    adjusted, receipt = positivity.apply_positivity(prior, increments,
                                                    policy="none")
    np.testing.assert_array_equal(adjusted["qr"], increments["qr"])
    assert receipt["negative_points"] == 1
    assert receipt["mass_left_negative"] == pytest.approx(1.0e-3)
    with pytest.raises(positivity.PositivityError, match="still negative"):
        positivity.verify_non_negative(prior, adjusted)


def test_positivity_refuses_an_increment_with_no_background():
    from gpuwm.da import positivity

    with pytest.raises(positivity.PositivityError, match="without its"):
        positivity.apply_positivity({}, {"qr": np.array([1.0])})


def test_the_cycle_driver_applies_the_policy_and_receipts_it(tmp_path):
    """The seam: the driver is the caller, so the driver chooses.

    Drives ``run_cycles`` with an assimilation step that deliberately
    proposes a qr increment large enough to go negative, and asserts the
    written analysis is bounded and the manifest says by how much.
    """

    from gpuwm.ensemble.cycle import ANALYSIS_NAME, run_cycles
    from gpuwm.ensemble.manifest import CYCLE_MANIFEST_NAME

    overlay = _write_configs(tmp_path, n_members=2)
    root = tmp_path / "ens"
    shape = (4, 6, 6)
    background_qr = np.full(shape, 1.0e-3, np.float32)

    def runner(*, base_config, member_dir, index, seed, perturbation,
               perturbation_options, run_seconds=None, **_):
        member_dir.mkdir(parents=True, exist_ok=True)
        np.savez(member_dir / "gpuwmrst_000060.npz",
                 **{"state/qr": background_qr,
                    "state/thp": np.zeros(shape, np.float32)})
        sha = f"{index:064d}"
        return MemberOutcome(
            index=index, seed=seed, member_dir=member_dir,
            initial_state_sha256=sha, final_state_sha256=sha,
            wall_seconds=1.0, sim_seconds=60.0, wrfout_count=0,
            last_checkpoint=None, perturbation={})

    def assimilate(cycle_index, member_states):
        # -2e-3 against a 1e-3 background: every point goes negative.
        return {index: {"qr": np.full(shape, -2.0e-3, np.float32)}
                for index in member_states}

    run_cycles(load_ensemble_config(overlay), root, n_cycles=1,
               cycle_seconds=60.0, runner=runner, assimilate=assimilate)

    manifest = json.loads((root / CYCLE_MANIFEST_NAME).read_text("utf-8"))
    assimilation = manifest["cycles"][0]["assimilation"]
    assert assimilation["positivity_policy"] == "clip"
    assert assimilation["negative_points_total"] == 2 * int(np.prod(shape))
    # Two members, every point raised from -1e-3 to 0.
    assert assimilation["mass_added_by_clip_total"] == pytest.approx(
        2 * np.prod(shape) * 1.0e-3, rel=1e-6)

    for record in assimilation["receipts"]:
        assert record["positivity"]["policy"] == "clip"
    for member in ("member_000", "member_001"):
        with np.load(root / "cycle_000" / member / ANALYSIS_NAME) as data:
            assert float(data["state/qr"].min()) >= 0.0
            assert float(data["state/qr"].max()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Wave B -- the gate: the whole chain, once, and what must be true of it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate_report(tmp_path_factory):
    """Run the composition gate once and let every assertion read it.

    Module-scoped because the run is the expensive part and because every
    assertion below is about the same run: a gate whose criteria were each
    checked against a different execution would not be a gate.
    """

    pytest.importorskip("netCDF4")
    from gpuwm.da.synthetic_cycle import GateConfig, run_gate

    out = tmp_path_factory.mktemp("wave-b")
    cfg = GateConfig(members=8, nx=20, ny=20, nz=6,
                     observed_margin_cells=6, products=False)
    return run_gate(out, cfg)


def test_the_analysis_mean_is_closer_to_the_truth_than_the_prior(
        gate_report):
    """The gate's headline."""

    rmse = gate_report["rmse"]
    assert rmse["analysis"]["total"] < rmse["prior"]["total"]
    assert rmse["improvement_pct"]["total"] > 0.0
    # Radial velocity is the observed quantity, so the wind components are
    # the ones that must improve.  qr is corrected only through
    # cross-covariance at R=8 and is deliberately not gated -- see the
    # handoff for the number it actually produces.
    for name in ("u", "v"):
        assert rmse["analysis"][name] < rmse["prior"][name], name


def test_the_second_leg_starts_from_the_analysis_and_stays_ahead(
        gate_report):
    """Restart-from-analysis is wired, and it is worth something.

    Compared against the same forecast operator applied to the *prior*
    members -- the free run -- so the comparison isolates the analysis
    rather than the leg.
    """

    assert all(gate_report["restart_from_analysis"]), (
        "some member re-prepared from the base config; the cycling run "
        "computed an analysis and then threw it away")
    second = gate_report["second_leg_rmse"]
    assert second["from_analysis"]["total"] < second["free_run"]["total"]


def test_the_second_production_cycle_ran_on_the_cumulative_horizon(
        gate_report):
    """F-01, end to end: the headline defect, proven fixed.

    The broken engine handed leg 1 ``run_seconds = cycle_seconds`` and the
    integrator -- which measures the total forecast length from the
    experiment start -- refused a restart already standing on it. This
    gate's stand-in runner implements that exact contract, so a two-cycle
    run completing at all is the proof, and the manifest numbers say leg 1
    was given the CUMULATIVE horizon while advancing exactly one leg.
    """

    cycles = {int(entry["cycle"]): entry for entry in gate_report["cycles"]}
    assert set(cycles) == {0, 1}, "both production cycles must be recorded"
    assert cycles[0]["run_seconds_total"] == cycles[0]["forecast_seconds"]
    assert cycles[1]["run_seconds_total"] == \
        2.0 * cycles[1]["forecast_seconds"], (
        "leg 1 must be given the cumulative horizon, not the leg length "
        "the integrator refuses")
    assert cycles[0]["restarted"] is False
    assert cycles[1]["restarted"] is True
    assert cycles[1]["start_seconds"] == cycles[0]["run_seconds_total"]
    assert all(entry["status"] == "DONE" for entry in cycles.values())
    assert all(entry["attempt"] == 1 for entry in cycles.values())


def test_increments_are_exactly_zero_outside_localization(gate_report):
    """Exactly, not nearly -- and the check is not vacuous either way."""

    localization = gate_report["localization"]
    assert localization["gridpoints_outside_every_lens"] > 0
    assert localization["largest_increment_outside"] == 0.0
    assert localization["largest_increment_inside"] > 0.0


def test_the_cycle_manifest_carries_the_assimilation_receipts(gate_report):
    """The DA lane's slot in gpuwm-da-cycle-manifest.v1 is filled."""

    assimilation = gate_report["assimilation"]
    assert assimilation["status"] == "APPLIED"
    assert assimilation["increment_contract"] == ["gpuwm-da-increments.v1"]
    assert assimilation["members_receipted"] == gate_report["members"]


def test_the_positivity_policy_ran_and_reported_the_mass_it_added(
        gate_report):
    """Clipping is not free, and the receipt says how much it cost."""

    assimilation = gate_report["assimilation"]
    assert assimilation["positivity_policy"] == "clip"
    assert assimilation["negative_points_total"] > 0, (
        "no analysis went negative, so the positivity seam was not "
        "exercised and this gate does not test it")
    assert assimilation["mass_added_by_clip_total"] > 0.0


def test_the_observations_are_a_real_radar_file_from_two_radars(gate_report):
    """Not a dict pretending to be one: the radar lane's own writer."""

    from pathlib import Path as _Path

    observations = gate_report["observations"]
    assert observations["schema"] == "gpuwm-obs.radar-grid.v1"
    assert len(observations["sha256"]) == 64
    assert _Path(observations["path"]).is_file()
    assert gate_report["radars"] == ["AAAA", "BBBB"]
    batches = gate_report["method"]["observations"]["batches"]
    assert [entry["radar"] for entry in batches] == ["AAAA", "BBBB"]
    assert all(entry["operator_direction"].startswith("vr_beam")
               for entry in batches)


def test_the_gate_binds_every_stage_to_one_grid_identity(gate_report):
    identity = gate_report["grid"]["identity_sha256"]
    assert len(identity) == 64
    assert gate_report["method"]["observations"][
        "grid_identity_sha256"] == identity


def test_the_gate_report_says_what_it_does_not_support(gate_report):
    """The caveats are load-bearing and must survive a refactor."""

    caveats = " ".join(gate_report["caveats"])
    assert "perfect model" in caveats
    assert "dycore is stood in for" in caveats
    json.dumps(gate_report)


def test_the_products_stage_renders_with_ensemble_tokens(tmp_path):
    """enprod over the analysed ensemble, the end of the chain."""

    pytest.importorskip("netCDF4")
    pytest.importorskip("wrf")
    pytest.importorskip("matplotlib")
    from gpuwm.da.synthetic_cycle import GateConfig, run_gate

    report = run_gate(tmp_path / "gate",
                      GateConfig(members=6, nx=18, ny=18, nz=6,
                                 observed_margin_cells=6, products=True))
    products = report["products"]
    assert products["rendered"] is True and products["exit_code"] == 0
    assert products["files"], "the products stage wrote nothing"
    for name in products["files"]:
        assert "-ens-" in name, (
            f"{name} carries no ensemble token; a deterministic filename "
            "on an ensemble product is how one gets mistaken for the other")


# ---------------------------------------------------------------------------
# The engine gaps Wave B needed: --assimilate, and restart-from-analysis
# ---------------------------------------------------------------------------


def test_the_assimilate_resolver_accepts_both_spellings():
    from tools.ensemble_forecast import resolve_assimilate

    assert resolve_assimilate("json:loads") is json.loads
    assert resolve_assimilate("json.loads") is json.loads


@pytest.mark.parametrize("reference,expected", [
    ("", "empty string"),
    ("loads", "not a dotted path"),
    ("gpuwm.da.no_such_module:go", "cannot import"),
    ("json:no_such_attribute", "has no"),
    ("json:__doc__", "not a callable"),
])
def test_the_assimilate_resolver_fails_closed(reference, expected):
    """A cycling run that quietly fell back to forecast-only would write a
    manifest full of true statements describing a run nobody asked for."""

    from tools.ensemble_forecast import resolve_assimilate

    with pytest.raises(ValueError, match=expected):
        resolve_assimilate(reference)


def test_a_leg_told_to_restart_from_a_missing_analysis_refuses(tmp_path):
    from gpuwm.ensemble.engine import run_ensemble

    overlay = _write_configs(tmp_path, n_members=1)
    with pytest.raises(ValueError, match="does not exist"):
        run_ensemble(load_ensemble_config(overlay), tmp_path / "ens",
                     runner=lambda **kw: None,
                     restarts={0: tmp_path / "nowhere.npz"})


def test_a_half_analysed_predecessor_is_refused_not_mixed(tmp_path):
    """Some members restarting and others re-preparing is not one ensemble."""

    from gpuwm.ensemble.cycle import _analysis_restarts

    root = tmp_path / "ens"
    (root / "cycle_000" / "member_000").mkdir(parents=True)
    (root / "cycle_000" / "member_001").mkdir(parents=True)
    (root / "cycle_000" / "member_000" / "analysis.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="not one ensemble"):
        _analysis_restarts(root, 1, n_members=2, required=True)
    # No analysis at all is a forecast-only predecessor, not an error.
    (root / "cycle_000" / "member_000" / "analysis.npz").unlink()
    assert _analysis_restarts(root, 1, n_members=2, required=True) is None

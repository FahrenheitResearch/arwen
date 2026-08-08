"""The statics corridor on the HRRR chain: emitted, bound, cropped.

1.8.4 sealed the corridor on the GFS chain and taught the tree runner to
crop it.  The runner never cared which preparation wrote the bundle --
it reads ``statics_corridor`` out of whichever hierarchy document
matched the pinned digest -- so what kept a moving nest off a nested
HRRR tree was purely that nothing on the HRRR path EMITTED one.

The emission lives in ``gpuwm.hrrr_hierarchy_direct``, which is the
stage that builds ``d02..dNN`` and holds ``--geog-root``; the root
preparer knows only d01 and could not build child-resolution statics if
it wanted to.  This file holds that emission to the same bar the GFS one
was held to:

* it goes through the SHARED emitter
  (:func:`gpuwm.static.corridor.emit_statics_corridor_set`), so there is
  no second corridor format wearing the same schema;
* it is byte-deterministic;
* the digest reaches ``receipt.json`` and the round trip is against the
  FILE's bytes, never a printed line;
* and a footprint cropped from a corridor built through this path is
  bitwise the statics built directly for that footprint -- proven
  against the build, with the instrument validated by a planted
  single-ULP perturbation, exactly as the original lane proved it.

CPU-only.  The corridor build, the seal, the digest verification and the
crop comparison are all real; the sealed-root machinery around the
emission (decode, cache restore, artifact join) is stubbed at its own
seams, each of which has its own suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import hrrr_hierarchy_direct
from gpuwm.experiment import load_experiment
from gpuwm.hrrr_native_static import verified_static_catalog
from gpuwm.static.build import GeogSelection, build_static
from gpuwm.static.corridor import (
    STATICS_CORRIDOR_DIRNAME,
    STATICS_CORRIDOR_RECEIPT,
    ChildStaticsCorridor,
    CorridorRefusal,
    emit_statics_corridor_set,
    load_child_statics_corridor,
)
from gpuwm.static.lambert import grids_from_projection_config
# The synthetic WPS_GEOG tree the corridor's own suite builds: nine
# tiny datasets exercising the build's distinct paths (fine gcell
# accumulation, the categorical pixel-count path, coarse-global
# interpolation, the halo'd terrain smoother).  Shared rather than
# copied so both chains' bit-identity proofs stand on the same source.
from test_statics_corridor import _synthetic_wps_geog


# ---------------------------------------------------------------------------
# A small, real two-domain tree over the synthetic WPS_GEOG
# ---------------------------------------------------------------------------

#: 24x24 at 6 km with a 9x9 nest at ratio 3, placed to clear the
#: parent's Davies and terrain-blend zones (the experiment loader
#: enforces that, and a nest one row closer is refused before any of
#: this runs).  The corridor is therefore 72x72 child cells -- parent
#: extent at child resolution, which is the whole point of the artifact
#: and the reason it is worth pricing.
_TREE_TOML = """
[experiment]
name = "hrrr-corridor-tree"
start_time = 2026-07-23T00:00:00
run_seconds = 3600.0
restart_interval_s = 0.0

[projection]
map_proj = "lambert"
ref_lat = 35.0
ref_lon = -97.0
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -97.0

[shared]
nz = 8
ztop = 12000.0
p_top = 10000.0
eta_levels = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
hybrid_opt = 2
etac = 0.2
map_proj = 1
moist = true
mp_physics = 6

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 24
ny = 24
dx = 6000.0
time_step = 36
history_interval_s = 3600.0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 11
j_parent_start = 11
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 9
ny = 9
history_interval_s = 3600.0
"""

_REF_I = _REF_J = 11
_RATIO = 3


@dataclass(frozen=True)
class _Chain:
    """Everything the HRRR hierarchy stage holds when it emits."""

    exp: object
    grids: tuple
    static_catalog: object
    geog_root: Path
    wps_namelist: Path
    config: Path


@pytest.fixture(scope="module")
def chain(tmp_path_factory) -> _Chain:
    root = tmp_path_factory.mktemp("hrrr-corridor")
    geog = _synthetic_wps_geog(root / "WPS_GEOG")
    wps = root / "namelist.wps"
    wps.write_text("&share\n max_dom = 2,\n/\n&geogrid\n/\n",
                   encoding="utf-8")
    config = root / "tree.toml"
    config.write_text(textwrap.dedent(_TREE_TOML), encoding="utf-8")
    exp = load_experiment(config)
    # The stage's OWN two inputs to the emission, built by the stage's
    # own helpers: the GEOG catalog it verifies and the projection grids
    # it hands the artifact writer.  Nothing here is a test-local
    # reconstruction of either.
    catalog, _receipt = verified_static_catalog(
        wps, geog, [domain.grid_id for domain in exp.domains])
    return _Chain(exp=exp, grids=tuple(grids_from_projection_config(exp)),
                  static_catalog=catalog, geog_root=geog,
                  wps_namelist=wps, config=config)


def _emit(chain: _Chain, directory: Path, statics_corridor="all"):
    return emit_statics_corridor_set(
        exp=chain.exp, grids=chain.grids,
        static_catalog=chain.static_catalog, directory=directory,
        statics_corridor=statics_corridor)


@pytest.fixture(scope="module")
def sealed(chain, tmp_path_factory):
    """One sealed corridor set, as the HRRR stage writes it."""

    directory = (tmp_path_factory.mktemp("sealed")
                 / "hierarchy-artifacts" / STATICS_CORRIDOR_DIRNAME)
    return directory, _emit(chain, directory)


# ---------------------------------------------------------------------------
# 1. Emission: shared, complete, deterministic
# ---------------------------------------------------------------------------

def test_the_hrrr_chain_seals_a_parent_extent_corridor(chain, sealed):
    directory, receipt = sealed

    assert receipt["schema"] == "gpuwm-statics-corridor-set-v1"
    assert receipt["status"] == "READY"
    entry = receipt["domains"]["d02"]
    # Parent extent at CHILD resolution: the 24x24 root times the nest's
    # refinement ratio, not the 9x9 nest.
    assert entry["corridor_nx"] == 24 * _RATIO
    assert entry["corridor_ny"] == 24 * _RATIO
    assert entry["cells"] == 72 * 72
    assert entry["reference_i_parent_start"] == _REF_I
    assert (directory / "d02.npz").is_file()
    assert (directory / STATICS_CORRIDOR_RECEIPT).is_file()


def test_the_emission_is_deterministic_on_the_hrrr_chain(chain, tmp_path):
    """Same inputs in, same bytes out -- twice, from scratch.

    The digest of this artifact is bound into the preparation document,
    so a wall-clock stamp anywhere inside it would make two identical
    preparations disagree about their own bundle.
    """
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first = _emit(chain, first_dir)
    second = _emit(chain, second_dir)

    assert first == second
    assert ((first_dir / "d02.npz").read_bytes()
            == (second_dir / "d02.npz").read_bytes())
    assert ((first_dir / STATICS_CORRIDOR_RECEIPT).read_bytes()
            == (second_dir / STATICS_CORRIDOR_RECEIPT).read_bytes())


def test_both_chains_emit_through_one_function():
    """No forked builder: the GFS join and the HRRR stage call the same
    emitter, so one runner cannot meet two corridor formats."""

    import inspect

    from gpuwm import source_hierarchy

    for module in (source_hierarchy, hrrr_hierarchy_direct):
        source = inspect.getsource(module)
        assert "emit_statics_corridor_set(" in source, module.__name__
        # And neither walks the children itself any more.
        assert "build_child_statics_corridor(" not in source, module.__name__


def test_a_single_domain_namelist_is_refused_before_the_expensive_work():
    """The flag is a typing mistake on a rootless tree, not a failure of
    the preparation -- so it is refused from the config alone."""

    from gpuwm.static.corridor import validated_corridor_selection

    single = SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, parent_id=0),))
    with pytest.raises(ValueError, match="no child domain"):
        validated_corridor_selection(single, "all")


# ---------------------------------------------------------------------------
# 2. THE LOAD-BEARING TEST: crop == direct build, through the HRRR path
# ---------------------------------------------------------------------------

def test_crop_equals_direct_footprint_build_bitwise(chain, sealed):
    """A corridor built through the HRRR chain's own grids and catalog
    crops to exactly the statics a direct footprint build produces.

    Re-proven here rather than inherited: the GFS lane proved the
    corridor module's arithmetic, and this proves that the grids THIS
    chain derives (``grids_from_projection_config`` off the namelist-
    imported experiment) sit on the same lattice that argument rests on.
    """
    directory, receipt = sealed
    child = chain.exp.domains[1]
    parent_run = chain.exp.domains[0].run
    corridor = load_child_statics_corridor(
        directory, expected_set_receipt=receipt, grid_id=2,
        child_dc=child, parent_run=parent_run,
        reference_grid=chain.grids[1])

    reference = chain.grids[1]
    selection = GeogSelection.fallback(chain.geog_root)
    checked = 0
    for ip, jp in ((_REF_I, _REF_J), (1, 1), (11, 4), (5, 14), (22, 22)):
        direct = build_static(
            reference.translated((ip - _REF_I) * _RATIO,
                                 (jp - _REF_J) * _RATIO),
            chain.geog_root, selection=selection)
        crop = corridor.crop(ip, jp)
        assert sorted(crop) == sorted(direct)
        for name in sorted(direct):
            left = np.asarray(crop[name])
            right = np.asarray(direct[name])
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes(), (
                f"{name} differs at placement ({ip}, {jp})")
            checked += 1
    assert checked == 5 * 14

    # VALIDATE THE INSTRUMENT: one flipped ULP planted in the corridor
    # must be caught by exactly the comparison above, and by nothing
    # else -- a comparison that passes everything proves nothing.
    fields = dict(corridor.fields)
    tampered = np.array(fields["HGT_M"], copy=True)
    tampered[35, 34] = np.nextafter(tampered[35, 34], np.inf)
    fields["HGT_M"] = tampered
    perturbed = ChildStaticsCorridor(
        geometry=corridor.geometry, fields=fields,
        cache_sha256=corridor.cache_sha256)
    bad = perturbed.crop(_REF_I, _REF_J)     # covers corridor cell (35, 34)
    good = build_static(reference, chain.geog_root, selection=selection)
    assert bad["HGT_M"].tobytes() != good["HGT_M"].tobytes()
    mismatches = np.flatnonzero(
        bad["HGT_M"].view(np.uint64) != good["HGT_M"].view(np.uint64))
    assert mismatches.size == 1


# ---------------------------------------------------------------------------
# 3. The digest binding: receipt.json's BYTES, round trip
# ---------------------------------------------------------------------------

def _hrrr_receipt(prepared_root: Path, corridor_receipt) -> Path:
    """The HRRR preparation document, carrying the corridor set."""

    prepared_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": hrrr_hierarchy_direct.SCHEMA,
        "status": "PASS",
        "valid_time": "2026-07-23T00:00:00",
        "domain_count": 2,
        "statics_corridor": dict(corridor_receipt),
    }
    path = prepared_root / "receipt.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n", encoding="utf-8")
    return path


def test_the_receipt_binds_the_corridor_and_a_reader_gets_it_back(
        chain, tmp_path):
    """Written into receipt.json, read back out of receipt.json's bytes.

    The relay discipline the whole chain runs on: the next stage is
    handed the sha256 of these bytes, reads the document those bytes
    parse to, and verifies the corridor against the copy sealed there --
    never against anything a stage printed on stdout.
    """
    import hashlib

    prepared = tmp_path / "hrrr-hierarchy"
    directory = prepared / "hierarchy-artifacts" / STATICS_CORRIDOR_DIRNAME
    corridor_receipt = _emit(chain, directory)
    receipt_path = _hrrr_receipt(prepared, corridor_receipt)

    # What the forecast stage is given, computed the way it computes it.
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert hrrr_hierarchy_direct.sha256_file(receipt_path) == digest

    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    corridor = load_child_statics_corridor(
        directory, expected_set_receipt=document["statics_corridor"],
        grid_id=2, child_dc=chain.exp.domains[1],
        parent_run=chain.exp.domains[0].run, reference_grid=chain.grids[1])
    assert corridor.geometry["corridor_nx"] == 72
    assert (corridor.cache_sha256
            == corridor_receipt["domains"]["d02"]["cache"]["sha256"])


def test_a_tampered_cache_refuses_against_the_hrrr_receipt(chain, tmp_path):
    """The bound digest is load-bearing, not decorative."""

    prepared = tmp_path / "hrrr-hierarchy"
    directory = prepared / "hierarchy-artifacts" / STATICS_CORRIDOR_DIRNAME
    receipt_path = _hrrr_receipt(prepared, _emit(chain, directory))
    cache = directory / "d02.npz"
    blob = bytearray(cache.read_bytes())
    blob[-1] ^= 0x01
    cache.write_bytes(bytes(blob))

    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    with pytest.raises(CorridorRefusal, match="digest mismatch"):
        load_child_statics_corridor(
            directory, expected_set_receipt=document["statics_corridor"],
            grid_id=2, child_dc=chain.exp.domains[1],
            parent_run=chain.exp.domains[0].run,
            reference_grid=chain.grids[1])


# ---------------------------------------------------------------------------
# 4. The real preparation function: it emits, and it binds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Report:
    note: str = "stub translation report"


def _stub_sealed_root(monkeypatch, chain, root: Path):
    """Stand in for everything between the sealed root and the join.

    Each stub replaces a collaborator with its own suite -- the namelist
    import, the physics slice, the cache restore, the decode, the
    artifact join -- so what is left running for real in
    ``prepare_hrrr_hierarchy`` is the part under test: the corridor
    emission and the receipt binding, with the REAL GEOG catalog and the
    REAL projection grids the stage derives.
    """
    module = hrrr_hierarchy_direct

    paths = module._root_paths(root)
    (root / "native" / "native-bridge").mkdir(parents=True)
    (root / "native" / "preparation-report").mkdir(parents=True)
    paths["prepared_cache"].mkdir(parents=True)
    paths["static_cache"].write_bytes(b"static-cache")
    paths["static_receipt"].write_text("{}", encoding="utf-8")
    paths["bridge_manifest"].write_text("bridge", encoding="utf-8")

    bridge_sha = module.sha256_file(paths["bridge_manifest"])
    static_sha = module.sha256_file(paths["static_cache"])
    manifest = root / "SHA256SUMS"
    manifest.write_text("source", encoding="utf-8")
    manifest_sha = module.sha256_file(manifest)
    identity = {
        "bridge_manifest_sha256": bridge_sha,
        "source_manifest_sha256": manifest_sha,
        "static_cache_sha256": static_sha,
        "namelist_sha256": "0" * 64,
        "forcing_hours": [0, 1],
        "source_identity": {"adapter": "fixture"},
        "domain_config": {"run": {}},
    }
    paths["prepared_cache"].joinpath("header.json").write_text(json.dumps({
        "schema": "gpuwm-prepared-real-cache-v1",
        "status": "READY",
        "identity": identity,
        "content_sha256": "d" * 64,
        "metadata": {"user": {
            "initial_valid_time": "2026-07-23T00:00:00",
            "source_cycle": "2026-07-23T00:00:00"}},
    }), encoding="utf-8")
    paths["preparation_report"].write_text(json.dumps({
        "status": "PASS",
        "source_identity": identity["source_identity"],
        "prepared_cache": {"content_sha256": "d" * 64},
    }), encoding="utf-8")

    (root / "d01-target.json").write_text("{}", encoding="utf-8")
    stock = root / "namelist.input.stock"
    stock.write_text("&physics\n/\n", encoding="utf-8")
    namelist_input = root / "namelist.input"
    namelist_input.write_text("&physics\n/\n", encoding="utf-8")
    cpu_bridge = root / "cpu-bridge.bin"
    cpu_bridge.write_bytes(b"bridge")

    restored = SimpleNamespace(
        met=SimpleNamespace(fields={}),
        surface=SimpleNamespace(fields={
            name: np.zeros((1,)) for name in (
                "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE",
                "XLAND", "LANDMASK", "SNOW", "SNOWH")}),
        initial_result=SimpleNamespace(state=SimpleNamespace()),
        boundaries=object(),
        receipt={"content_sha256": "d" * 64},
    )
    result = SimpleNamespace(
        timings_seconds={"join": 0.0},
        artifacts=SimpleNamespace(receipt={"schema": "stub-artifacts"}),
        wrf_manifest={"status": "NOT_REQUESTED"},
    )

    monkeypatch.setattr(module, "resolve_cpu_bridge", lambda _p: cpu_bridge)
    monkeypatch.setattr(module, "_require_raw_stock_delta",
                        lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(module, "_require_raw_wps_contract",
                        lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(
        module, "_native_experiment",
        lambda *a, **k: (chain.exp, "resolved = true\n", _Report()))
    monkeypatch.setattr(module, "load_hrrr_target_domain",
                        lambda _spec: SimpleNamespace(
                            surface_fallback_radius_cells=3))
    monkeypatch.setattr(module, "_supported_hierarchy_slice",
                        lambda *a, **k: None)
    monkeypatch.setattr(module, "verify_hrrr_native_static",
                        lambda *a, **k: ({}, {"geog_selection": None}))
    monkeypatch.setattr(module, "verified_root_forcing_inventory",
                        lambda hours, **k: tuple(hours))
    monkeypatch.setattr(module, "_expected_root_cache_identity",
                        lambda *a, **k: {"namelist_sha256": "0" * 64})
    monkeypatch.setattr(module, "PreparedCacheReader",
                        lambda *a, **k: SimpleNamespace(
                            verify_all=lambda: None))
    monkeypatch.setattr(module, "restore_prepared_cache",
                        lambda *a, **k: restored)
    monkeypatch.setattr(module, "_surface_state",
                        lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(module, "load_hrrr_native_series",
                        lambda *a, **k: (object(),))
    monkeypatch.setattr(module, "_source_identity",
                        lambda _bridge: {"identity_source": "fixture"})
    monkeypatch.setattr(module, "ParentInitView",
                        lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "NestedInputCatalog",
                        lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "initialize_and_export_native_hierarchy",
                        lambda **kwargs: result)
    # The one selection check the stubbed slice would have skipped:
    # keep the REAL comparison satisfied rather than removed.
    monkeypatch.setattr(module, "verified_static_catalog",
                        lambda wps, geog, ids: (
                            chain.static_catalog,
                            {"selections": {"d01": None}}))
    from gpuwm.static import highres_production
    monkeypatch.setattr(highres_production, "refuse_inert_highres",
                        lambda *a, **k: None)

    return SimpleNamespace(
        root_preparation=root, root_domain_spec=root / "d01-target.json",
        wps_namelist=chain.wps_namelist, namelist_input=namelist_input,
        stock_wrf_namelist_input=stock, geog_root=chain.geog_root,
        source_manifest=manifest, source_manifest_sha256=manifest_sha)


def _prepare(inputs, output_root: Path, **extra):
    return hrrr_hierarchy_direct.prepare_hrrr_hierarchy(
        root_preparation=inputs.root_preparation,
        root_domain_spec=inputs.root_domain_spec,
        wps_namelist=inputs.wps_namelist,
        namelist_input=inputs.namelist_input,
        stock_wrf_namelist_input=inputs.stock_wrf_namelist_input,
        geog_root=inputs.geog_root,
        source_manifest=inputs.source_manifest,
        source_manifest_sha256=inputs.source_manifest_sha256,
        valid_time=datetime(2026, 7, 23, 0, 0, 0),
        output_root=output_root, **extra)


def test_prepare_hrrr_hierarchy_seals_and_binds_the_corridor(
        chain, tmp_path, monkeypatch):
    """The real function, with the flag: artifact on disk, digest in the
    receipt, and the two agreeing."""

    inputs = _stub_sealed_root(monkeypatch, chain, tmp_path / "root")
    output = tmp_path / "hierarchy"
    payload = _prepare(inputs, output, statics_corridor="all")

    corridor = payload["statics_corridor"]
    assert corridor["schema"] == "gpuwm-statics-corridor-set-v1"
    entry = corridor["domains"]["d02"]
    assert entry["corridor_nx"] == 72 and entry["corridor_ny"] == 72

    # The artifact landed inside the atomically published tree, where
    # the runner looks for it -- the same relative path the GFS bundle
    # carries it at.
    cache = (output / "hierarchy-artifacts" / STATICS_CORRIDOR_DIRNAME
             / entry["cache"]["path"])
    assert cache.is_file()
    assert hrrr_hierarchy_direct.sha256_file(cache) == entry["cache"]["sha256"]

    # And the published receipt.json says the same thing the returned
    # payload does, which is what makes hashing the file equivalent to
    # trusting the return value.
    document = json.loads(
        (output / "receipt.json").read_text(encoding="utf-8"))
    assert document["statics_corridor"] == corridor

    # The whole point: a reader with only the file can verify the
    # corridor and crop it.
    loaded = load_child_statics_corridor(
        output / "hierarchy-artifacts" / STATICS_CORRIDOR_DIRNAME,
        expected_set_receipt=document["statics_corridor"], grid_id=2,
        child_dc=chain.exp.domains[1],
        parent_run=chain.exp.domains[0].run,
        reference_grid=chain.grids[1])
    assert loaded.crop(_REF_I, _REF_J)["HGT_M"].shape == (9, 9)


def test_without_the_flag_the_receipt_carries_no_corridor_key(
        chain, tmp_path, monkeypatch):
    """Absent, not null.  A bundle nobody asked a corridor of must be
    what it always was, including the shape of its own document."""

    inputs = _stub_sealed_root(monkeypatch, chain, tmp_path / "root")
    output = tmp_path / "hierarchy"
    payload = _prepare(inputs, output)

    assert "statics_corridor" not in payload
    document = json.loads(
        (output / "receipt.json").read_text(encoding="utf-8"))
    assert "statics_corridor" not in document
    assert not (output / "hierarchy-artifacts"
                / STATICS_CORRIDOR_DIRNAME).exists()


def test_a_named_child_seals_only_that_child(chain, tmp_path, monkeypatch):
    inputs = _stub_sealed_root(monkeypatch, chain, tmp_path / "root")
    payload = _prepare(inputs, tmp_path / "hierarchy",
                       statics_corridor=(2,))
    assert sorted(payload["statics_corridor"]["domains"]) == ["d02"]


def test_an_unknown_child_refuses_before_the_root_is_restored(
        chain, tmp_path, monkeypatch):
    """A mistyped grid id must not cost a preparation.

    The selection is resolved as soon as a domain tree exists, which is
    long before the restore/decode/join -- so the refusal arrives in
    seconds and nothing is published.
    """
    inputs = _stub_sealed_root(monkeypatch, chain, tmp_path / "root")
    output = tmp_path / "hierarchy"
    with pytest.raises(ValueError, match="not child domains"):
        _prepare(inputs, output, statics_corridor=(9,))
    assert not output.exists()


# ---------------------------------------------------------------------------
# 5. The command line
# ---------------------------------------------------------------------------

def _parsed(argv):
    return hrrr_hierarchy_direct._parser().parse_args(argv)


_BASE_ARGV = [
    "--root-preparation", "root", "--root-domain-spec", "spec.json",
    "--wps-namelist", "namelist.wps", "--namelist-input", "namelist.input",
    "--stock-wrf-namelist-input", "stock.input", "--geog-root", "GEOG",
    "--source-manifest", "SHA256SUMS", "--source-manifest-sha256", "0" * 64,
    "--output-root", "out",
]


def test_the_flag_is_optional_and_bare_means_every_child():
    assert _parsed(_BASE_ARGV).statics_corridor is None
    assert _parsed(
        _BASE_ARGV + ["--statics-corridor"]).statics_corridor == "all"
    assert _parsed(
        _BASE_ARGV + ["--statics-corridor", "2,3"]).statics_corridor == "2,3"


def test_the_flag_is_spelled_the_way_the_refusals_name_it():
    """One runner reads both chains' bundles, and its corridor-less
    refusal names ONE flag.  A chain that spelled it differently would
    hand a reader a remedy that does not exist on the tool in their
    hand, which is why the spelling is a module constant rather than a
    string typed per door.
    """
    from gpuwm.static.corridor import STATICS_CORRIDOR_FLAG

    options = {option
               for action in hrrr_hierarchy_direct._parser()._actions
               for option in action.option_strings}
    assert STATICS_CORRIDOR_FLAG in options

    # And that constant is what the tree runner tells a reader to go and
    # add, so following the refusal lands on the flag above.
    from gpuwm import prepared_domain_tree_forecast as runner

    assert STATICS_CORRIDOR_FLAG in runner.runner_capabilities()[
        "relocation"]["follow_sources"]


def test_a_malformed_grid_id_list_is_refused_by_name(monkeypatch, tmp_path):
    """`--statics-corridor two` is a typo, and it says so."""

    monkeypatch.setattr(
        hrrr_hierarchy_direct, "prepare_hrrr_hierarchy",
        lambda **kwargs: pytest.fail("preparation must not start"))
    with pytest.raises(ValueError, match="comma-separated child grid ids"):
        hrrr_hierarchy_direct.main(
            _BASE_ARGV + ["--cycle", "2026-07-23_00:00:00",
                          "--statics-corridor", "two"])


def test_main_forwards_the_parsed_selection(monkeypatch, tmp_path):
    seen = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True)
        (output / "receipt.json").write_text("{}", encoding="utf-8")
        return {"status": "PASS", "workers": 8, "timing_seconds": {},
                "wrf_manifest": {"status": "NOT_REQUESTED"}}

    monkeypatch.setattr(hrrr_hierarchy_direct, "prepare_hrrr_hierarchy",
                        _fake)
    argv = [token if token != "out" else str(tmp_path / "out")
            for token in _BASE_ARGV]
    assert hrrr_hierarchy_direct.main(
        argv + ["--cycle", "2026-07-23_00:00:00",
                "--statics-corridor", "2,3"]) == 0
    assert seen["statics_corridor"] == (2, 3)

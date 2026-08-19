"""HRRR as a prepared background for the cycling radar-DA driver.

The bundle tests here are deliberately WRITER-TO-READER: the fixture
builds a prepared cache with the shipped cache writer, publishes the
portable authorities with the shipped bundle writer
(:mod:`gpuwm.hrrr_prepared_bundle`), and then admits the result through
the front door the driver actually calls
(``preflight_prepared_forecast``).  Nothing here re-spells a schema, so
the two sides cannot drift into agreeing with each other's fixtures
while disagreeing with each other.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import gpuwm.prepared_single_domain_forecast as runner  # noqa: E402
from gpuwm.experiment import VerticalConfig, load_experiment  # noqa: E402
from gpuwm.hrrr_route_inputs import (  # noqa: E402
    ROUTE_DEFAULT_PHYSICS_PROFILE,
)
from gpuwm.experiment_document import (  # noqa: E402
    ExperimentDocumentError, publish_experiment_document,
    render_experiment_document)
from gpuwm.hrrr_prepared_bundle import (  # noqa: E402
    HrrrBundleError, publish_hrrr_prepared_bundle)
from gpuwm.ingest.hrrr_target import HrrrTargetDomain  # noqa: E402
from gpuwm.ingest.prepared_cache import (  # noqa: E402
    PREPARED_CACHE_SCHEMA, _array_sha256, prepared_cache_identity,
    prepared_domain_config_identity)

from hrrr_single_domain_benchmark import (  # noqa: E402
    _experiment, _experiment_tables)


#: The suite these bundles are built with.  It is the ROUTE's own
#: default rather than a literal, because the fixtures below build
#: their experiment through ``_experiment_tables`` -- which binds
#: that default -- and every downstream stage is handed the same
#: name.  Pinning a literal here is how the two halves drifted when
#: 1.8 flipped the default.
PROFILE = ROUTE_DEFAULT_PHYSICS_PROFILE

CYCLE = datetime(2026, 8, 5, 4)
SOURCE_HOURS = (0, 1, 2)
NZ = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _hrrr_experiment(*, run_seconds: float = 7200.0,
                     history_interval_seconds: float = 900.0):
    vertical = VerticalConfig(
        eta_levels=tuple(1.0 - index / NZ for index in range(NZ + 1)),
        p_top=5000.0, hybrid_opt=2, etac=0.2)
    target = dataclasses.replace(
        HrrrTargetDomain.legacy_500x500(), nz=NZ, nx=48, ny=48)
    kwargs = dict(run_seconds=run_seconds, target=target,
                  start_time=CYCLE + timedelta(hours=SOURCE_HOURS[0]),
                  history_interval_seconds=history_interval_seconds)
    tables, _resolved = _experiment_tables(vertical, **kwargs)
    return tables, _experiment(vertical, **kwargs)


def _write_prepared_cache(cache: Path, identity, forcing_hours, exp) -> dict:
    """A minimal cache with the header shape the reader verifies.

    Written by hand rather than by ``write_prepared_cache`` because the
    real writer needs decoded HRRR state; the HEADER contract -- schema,
    identity, metadata, arrays, content digest -- is what preflight
    reads, and it is reproduced here exactly.
    """

    cache.mkdir(parents=True)
    array = np.asarray([1.0], dtype=np.float32)
    with (cache / "a00000.npy").open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    arrays = {"test": {"file": "a00000.npy", "shape": [1],
                       "dtype": "float32", "nbytes": 4,
                       "sha256": _array_sha256(array)}}
    metadata = {
        "user": {
            "initial_valid_time": exp.start_time.isoformat(),
            "last_valid_time": (
                exp.start_time
                + timedelta(hours=forcing_hours[-1])).isoformat(),
            "source_cycle": CYCLE.isoformat(),
            "source_forecast_hours": list(SOURCE_HOURS),
            "model_forcing_hours": list(forcing_hours),
            "forcing_hours": list(forcing_hours),
            "mapping_reports": {"hrrr": "synthetic-decode-report"},
        },
        "state_names": [], "coord_arrays": [], "coord_scalars": {},
        "base_arrays": [], "base_scalars": {},
        "met_fields": sorted(runner._REQUIRED_MET_FIELDS),
        "surface_fields": sorted(runner._CANONICAL_SURFACE_FIELDS),
        "lbc": {
            "spec_bdy_width": exp.root.run.spec_bdy_width,
            "spec_zone": exp.root.run.spec_zone,
            "relax_zone": exp.root.run.relax_zone,
            "intervals": [
                {"start_seconds": float(first * 3600),
                 "end_seconds": float(second * 3600),
                 "fields": list(runner._LBC_FIELDS)}
                for first, second in zip(forcing_hours, forcing_hours[1:])],
        },
        "setup_fingerprint": "test",
    }
    basis = {"schema": PREPARED_CACHE_SCHEMA, "identity": identity,
             "metadata": metadata, "arrays": arrays, "payload_bytes": 4}
    header = {**basis, "status": "READY",
              "created_utc": "2026-08-05T00:00:00+00:00",
              "content_sha256": hashlib.sha256(
                  _canonical(basis).encode()).hexdigest()}
    (cache / "header.json").write_text(
        json.dumps(header, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return header


@dataclasses.dataclass
class _Bundle:
    root: Path
    handoff: dict
    experiment: object
    run_seconds: float
    history_interval_seconds: float


def _hrrr_bundle(tmp_path: Path, *, run_seconds: float = 7200.0,
                 history_interval_seconds: float = 900.0,
                 physics_profile: str | None = PROFILE,
                 forcing_hours=(0, 1, 2),
                 rendered_wps: bool = False,
                 publish_cycle: datetime | None = None) -> _Bundle:
    tables, exp = _hrrr_experiment(
        run_seconds=run_seconds,
        history_interval_seconds=history_interval_seconds)
    root = tmp_path / "prepared"
    native = root / "native"
    bridge_dir = native / "native-bridge"
    bridge_dir.mkdir(parents=True)
    bridge = bridge_dir / "SHA256SUMS"
    bridge.write_bytes(b"decoded-bridge-manifest\n")

    static = root / "native-static.npz"
    static.write_bytes(b"hash-bound-hrrr-static")
    static_receipt = root / "native-static-receipt.json"
    static_receipt.write_text('{"schema": "test"}\n', encoding="utf-8")
    geometry = root / "native-geometry-receipt.json"
    geometry.write_text(json.dumps({
        "schema": "gpuwm-native-static-direct-v1", "status": "PASS",
        "geometry": {"test": "hrrr"},
        "cache": {"path": static.name, "bytes": static.stat().st_size,
                  "sha256": _sha256(static)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_manifest = tmp_path / "SHA256SUMS"
    source_manifest.write_bytes(b"raw-hrrr-grib2-manifest\n")
    namelist_input = tmp_path / "namelist.input"
    namelist_input.write_text("&domains\n/\n", encoding="utf-8")
    wps_namelist = tmp_path / "namelist.wps"
    wps_namelist.write_text("&share\n/\n", encoding="utf-8")

    source_identity = {
        "installed_wheel": None,
        "source_sha256": {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in runner._HRRR_DECODE_SOURCES},
        "source_cycle": CYCLE.isoformat(),
        "model_start_time": exp.start_time.isoformat(),
        "source_forecast_hours": list(SOURCE_HOURS),
        "model_forcing_hours": list(forcing_hours),
    }
    identity = prepared_cache_identity(
        bridge_manifest_sha256=_sha256(bridge),
        source_manifest_sha256=_sha256(source_manifest),
        static_cache_sha256=_sha256(static),
        namelist_sha256=_sha256(namelist_input),
        domain_config=exp.root, forcing_hours=forcing_hours,
        source_identity=source_identity)
    _write_prepared_cache(
        native / "prepared-cache", identity, forcing_hours, exp)

    # Published by the benchmark's own renderer, exactly as the
    # preparation does it, then bound by the bundle writer.
    config = publish_experiment_document(
        root / "experiment.toml", tables, exp)

    handoff = publish_hrrr_prepared_bundle(
        output_root=root,
        prepared_cache=native / "prepared-cache",
        static_cache=static, static_receipt=static_receipt,
        geometry_receipt=geometry, bridge_manifest=bridge,
        namelist_input=namelist_input,
        # ``None`` is the ROUTE's own case: the native preparation is
        # driven by a target domain and has no namelist.wps to copy, so
        # the writer renders one from this experiment.
        wps_namelist=(None if rendered_wps else wps_namelist),
        source_manifest=source_manifest,
        experiment_config=config,
        source_cycle=publish_cycle or CYCLE,
        source_forecast_hours=SOURCE_HOURS,
        model_forcing_hours=forcing_hours,
        preprocessing={"schema": "gpuwm-preprocess-implementation-v2",
                       "backend": "cpu",
                       "implementation": "test-native-preprocess",
                       "workers": 2},
        source_identity=source_identity,
        physics_profile=physics_profile)
    return _Bundle(root=root, handoff=handoff, experiment=exp,
                   run_seconds=run_seconds,
                   history_interval_seconds=history_interval_seconds)


def _bind_synthetic_geometry(monkeypatch):
    """Stand in for the Lambert/static verification the fixture cannot do."""

    monkeypatch.setattr(
        runner, "validate_native_lambert_contract",
        lambda exp, wps, source_name=None: object())
    monkeypatch.setattr(
        runner, "verify_native_static_receipt",
        lambda receipt, static, grid, cfg: None)
    monkeypatch.setattr(
        runner, "load_native_static_cache",
        lambda path, grid, ny, nx: {"LANDMASK": np.zeros((ny, nx))})


def _preflight(bundle: _Bundle, **overrides):
    kwargs = dict(
        source="hrrr", prepared_root=bundle.root,
        proof_sha256=bundle.handoff["proof_sha256"],
        source_manifest_sha256=bundle.handoff["source_manifest_sha256"],
        prepared_content_sha256=bundle.handoff["prepared_content_sha256"],
        experiment_config=Path(bundle.handoff["experiment_config"]),
        wps_namelist=Path(bundle.handoff["wps_namelist"]),
        physics_profile=bundle.handoff["physics_profile"],
        run_seconds=bundle.run_seconds,
        history_interval_seconds=bundle.history_interval_seconds)
    kwargs.update(overrides)
    return runner.preflight_prepared_forecast(**kwargs)


# ---------------------------------------------------------------------
# the experiment document the whole handoff rests on
# ---------------------------------------------------------------------

def test_published_experiment_document_reloads_to_the_prepared_identity(
        tmp_path):
    """The bridge between a programmatic preparation and a config-driven run.

    The native HRRR route builds its experiment in code.  The prepared
    cache stores ``domain_config`` and every restore compares it by
    strict equality, so the document this publishes has to reload to
    exactly that -- not approximately, and not to a hand-written config
    that happens to agree today.
    """

    tables, exp = _hrrr_experiment()
    path = publish_experiment_document(
        tmp_path / "experiment.toml", tables, exp)
    reloaded = load_experiment(path)
    assert (prepared_domain_config_identity(reloaded.root)
            == prepared_domain_config_identity(exp.root))
    assert reloaded.start_time == exp.start_time


def test_experiment_document_refuses_an_unrenderable_value():
    tables, _exp = _hrrr_experiment()
    tables["experiment"]["name"] = object()
    with pytest.raises(ExperimentDocumentError, match="cannot render"):
        render_experiment_document(tables)


def test_experiment_document_leaves_no_file_when_the_reload_disagrees(
        tmp_path, monkeypatch):
    """A document that does not reload must not be left for the next stage."""

    tables, exp = _hrrr_experiment()
    tables["shared"]["p_top"] = 4999.0        # not what `exp` was built with
    target = tmp_path / "experiment.toml"
    with pytest.raises(ExperimentDocumentError, match="does not reload"):
        publish_experiment_document(target, tables, exp)
    assert not target.exists()


# ---------------------------------------------------------------------
# the bundle: published by the shipped writer, admitted by the front door
# ---------------------------------------------------------------------

def test_the_route_renders_the_namelist_it_has_no_file_for(tmp_path):
    """The native route's missing authority, produced from what it holds.

    A native HRRR preparation is driven by a strict target domain, not by
    a ``namelist.wps`` -- so there was no such file to publish, and the
    portable authorities were therefore opt-in behind a flag that made
    the caller supply one.  A finished tree could then exist that no
    forecast front door would accept, which is exactly what the Linux
    shakeout produced on 2026-08-18.

    The numbers were never missing, only the file.  This asserts the
    rendered namelist reads back as the experiment's own geometry
    through the FORECAST STAGE'S parser, not through a fixture of its
    own: same grids, same order, same values.
    """

    from gpuwm.static.projection import (
        grids_from_projection_config, grids_from_wps_namelist)

    bundle = _hrrr_bundle(tmp_path, rendered_wps=True)
    rendered = bundle.root / "namelist.wps"
    assert rendered.is_file()
    assert Path(bundle.handoff["wps_namelist"]) == rendered

    observed = grids_from_wps_namelist(rendered)
    expected = grids_from_projection_config(bundle.experiment)
    assert len(observed) == len(expected) == 1
    for name in ("ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
                 "dx", "dy", "e_we", "e_sn", "known_x", "known_y"):
        assert getattr(observed[0], name) == getattr(expected[0], name), name


def test_preflight_admits_a_bundle_whose_namelist_was_rendered(
        tmp_path, monkeypatch):
    """Rendered or supplied, the front door cannot tell and must not.

    The bundle writer binds ``namelist.wps`` by digest either way, so
    this is the assertion that a default `gpuwm prep --source hrrr` tree
    -- one where nobody hand-wrote a namelist -- is admitted by the same
    preflight a hand-supplied one is.
    """

    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path, rendered_wps=True)
    inputs = _preflight(bundle)
    assert Path(inputs.wps_namelist) == bundle.root / "namelist.wps"


def test_sim_accepts_a_published_hrrr_tree_and_composes_its_runner_line(
        tmp_path):
    """The other direction of the shakeout's refusal: this tree RUNS.

    ``gpuwm sim`` reads the bundle's own document for source and layout
    and relays the three digests the single-domain runner binds.  Every
    path it needs is inside the tree the preparation wrote, so the two
    commands a reader types name no file they have to produce
    themselves.
    """

    from gpuwm import stage_cli

    bundle = _hrrr_bundle(tmp_path, rendered_wps=True)
    resolved = stage_cli.resolve_bundle(bundle.root)
    assert resolved["source"] == "hrrr"
    assert resolved["layout"] == "single"

    command = stage_cli.sim_command(
        resolved,
        experiment_config=bundle.root / "experiment.toml",
        wps_namelist=bundle.root / "namelist.wps",
        outdir=tmp_path / "out")
    # The digests are the ones the publisher handed back, relayed rather
    # than recomputed by this seam.
    assert (command[command.index("--proof-sha256") + 1]
            == bundle.handoff["proof_sha256"])
    assert (command[command.index("--source-manifest-sha256") + 1]
            == bundle.handoff["source_manifest_sha256"])
    assert (command[command.index("--prepared-content-sha256") + 1]
            == bundle.handoff["prepared_content_sha256"])
    assert command[command.index("--source") + 1] == "hrrr"


def test_preflight_admits_a_published_hrrr_bundle(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    inputs = _preflight(bundle)
    assert inputs.source == "hrrr"
    assert inputs.layout == runner.HRRR_DIRECT_LAYOUT
    assert inputs.forcing_hours == (0, 1, 2)
    assert inputs.experiment.root.run.nx == 48


def test_preflight_binds_the_three_hrrr_digests_separately(
        tmp_path, monkeypatch):
    """Bridge, source manifest and namelist are three values, not one.

    On the portable route the bridge and source manifest digests happen
    to be the same file, and the namelist digest is the experiment
    TOML's.  On this route all three differ, and reading any of them
    from the wrong place would produce an identity that cannot match a
    real HRRR cache.
    """

    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    manifest = json.loads(
        (bundle.root / "source-input-manifest.json").read_text())
    files = manifest["files"]
    digests = {files["bridge"]["sha256"], files["source_manifest"]["sha256"],
               files["namelist_input"]["sha256"],
               files["experiment_config"]["sha256"]}
    assert len(digests) == 4
    header = json.loads(
        (bundle.root / "native" / "prepared-cache"
         / "header.json").read_text())
    identity = header["identity"]
    assert identity["bridge_manifest_sha256"] == files["bridge"]["sha256"]
    assert identity["source_manifest_sha256"] \
        == files["source_manifest"]["sha256"]
    assert identity["namelist_sha256"] == files["namelist_input"]["sha256"]
    _preflight(bundle)                       # and the front door agrees


def test_preflight_refuses_a_cache_decoded_by_another_ingest(
        tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    proof_path = bundle.root / "proof.json"
    proof = json.loads(proof_path.read_text())
    name = runner._HRRR_DECODE_SOURCES[0]
    proof["source_sha256"][name] = "0" * 64
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="decode identity differs"):
        _preflight(bundle, proof_sha256=_sha256(proof_path))


def test_preflight_refuses_a_swapped_bridge_manifest(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    bridge = bundle.root / "native" / "native-bridge" / "SHA256SUMS"
    bridge.write_bytes(b"a-different-decode\n")
    with pytest.raises(ValueError, match="native bridge manifest differs"):
        _preflight(bundle)


def test_preflight_refuses_a_swapped_namelist(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    (bundle.root / "namelist.input").write_text(
        "&domains\ne_vert = 9\n/\n", encoding="utf-8")
    with pytest.raises(ValueError, match="namelist.input differs"):
        _preflight(bundle)


def test_preflight_refuses_a_non_hourly_hrrr_cadence(tmp_path, monkeypatch):
    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    proof_path = bundle.root / "proof.json"
    proof = json.loads(proof_path.read_text())
    proof["forcing_hours"] = [0, 2, 4]
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="not hourly"):
        _preflight(bundle, proof_sha256=_sha256(proof_path))


def test_preflight_refuses_an_hrrr_hierarchy_proof(tmp_path, monkeypatch):
    """The tree route is a division of labour, and the refusal says so."""

    _bind_synthetic_geometry(monkeypatch)
    bundle = _hrrr_bundle(tmp_path)
    proof_path = bundle.root / "proof.json"
    proof = json.loads(proof_path.read_text())
    proof["schema"] = "gpuwm-hrrr-native-hierarchy-proof-v1"
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="prepared_domain_tree_forecast"):
        _preflight(bundle, proof_sha256=_sha256(proof_path))


# ---------------------------------------------------------------------
# path separators: the identity is a set of repository paths, not a set
# of strings some machine happened to spell
# ---------------------------------------------------------------------
#
# Live defect, released 1.8.2, Windows: the prepared HRRR chain ALWAYS
# died at the forecast handoff with
#
#     ValueError: HRRR prepared cache decode identity omits
#     ['gpuwm/hrrr_forecast.py', ...all ten...]
#
# and the artifact said why -- proof.json was keyed
# ``gpuwm\hrrr_forecast.py``.  Both producers built their digest dict
# with ``str(path.relative_to(REPO))``, which is backslashed on Windows,
# while the reader looks names up by the forward-slash constants in
# ``_HRRR_DECODE_SOURCES``.  On Linux the two spellings are the same
# string, which is why every end-to-end HRRR proof -- all of them run on
# rentals or WSL -- passed while the Windows route had never worked.


def _identity_pair(*, separator: str = "/", digest: str = "a"):
    """An identity and the proof beside it, keyed with one separator."""

    digests = {
        name.replace("/", separator): (digest * 64)[:64]
        for name in runner._HRRR_DECODE_SOURCES
    }
    body = {
        "source_cycle": CYCLE.isoformat(),
        "model_start_time": CYCLE.isoformat(),
        "source_forecast_hours": list(SOURCE_HOURS),
        "model_forcing_hours": list(SOURCE_HOURS),
    }
    return ({"source_sha256": dict(digests), **body},
            {"source_sha256": dict(digests), **body})


def test_a_windows_sealed_cache_identity_validates():
    """The shape 1.8.2 sealed on Windows must still restore.

    Those caches are keyed with backslashes and cannot be rewritten,
    but their digests bind the same file BYTES as a POSIX-keyed cache
    of the same tree.  Refusing them would orphan real work over a
    property of the machine that wrote the JSON.
    """

    identity, proof = _identity_pair(separator="\\")
    assert any("\\" in key for key in identity["source_sha256"])
    assert runner._validate_hrrr_source_identity(identity, proof) is identity


def test_a_posix_keyed_identity_validates():
    identity, proof = _identity_pair(separator="/")
    assert runner._validate_hrrr_source_identity(identity, proof) is identity


def test_a_windows_cache_validates_against_a_posix_proof():
    """The mixed estate: a cache sealed on one machine, read on another."""

    identity, _ = _identity_pair(separator="\\")
    _, proof = _identity_pair(separator="/")
    assert runner._validate_hrrr_source_identity(identity, proof) is identity


def test_an_absent_decode_source_is_still_refused():
    """Normalizing keys must not turn a missing file into a present one."""

    identity, proof = _identity_pair(separator="\\")
    dropped = runner._HRRR_DECODE_SOURCES[0].replace("/", "\\")
    del identity["source_sha256"][dropped]
    del proof["source_sha256"][dropped]
    with pytest.raises(ValueError, match="decode identity omits"):
        runner._validate_hrrr_source_identity(identity, proof)


def test_a_different_ingest_is_still_refused_across_separators():
    """The bar the leniency must not cross.

    Reading both sides POSIX-normalized is only safe if a genuine
    digest difference still lands.  Same file, two spellings, two
    digests: refused.
    """

    identity, _ = _identity_pair(separator="\\", digest="a")
    _, proof = _identity_pair(separator="/", digest="b")
    with pytest.raises(ValueError, match="decode identity differs"):
        runner._validate_hrrr_source_identity(identity, proof)


def test_one_file_under_two_spellings_is_refused():
    """A merge would silently drop one of two digests for one file."""

    identity, proof = _identity_pair(separator="/")
    name = runner._HRRR_DECODE_SOURCES[0]
    identity["source_sha256"][name.replace("/", "\\")] = "c" * 64
    with pytest.raises(ValueError, match="two path spellings"):
        runner._validate_hrrr_source_identity(identity, proof)


def test_the_identity_key_expression_is_posix_on_a_windows_path():
    """Pins the expression itself, so a Linux runner catches a regression.

    ``str()`` of a relative ``PureWindowsPath`` is the shipped bug,
    evaluated identically on every platform.  This is the assertion
    that would have failed in CI before a Windows user ever ran it.
    """

    from pathlib import PureWindowsPath

    repo = PureWindowsPath(r"C:\gpuwm")
    path = repo / "gpuwm" / "hrrr_forecast.py"
    assert path.relative_to(repo).as_posix() == "gpuwm/hrrr_forecast.py"
    assert str(path.relative_to(repo)) == "gpuwm\\hrrr_forecast.py"


@pytest.mark.parametrize("module_name", [
    "hrrr_single_domain_benchmark", "hrrr_state_proof"])
def test_both_producers_emit_posix_identity_keys(module_name):
    """The real producers, on whatever platform this runs.

    Not a re-spelling of the expression above: this calls the shipped
    ``_source_identity`` and looks at the dict a proof gets written
    from, which is where the released defect actually lived.
    """

    import importlib

    module = importlib.import_module(module_name)
    digests = module._source_identity()["source_sha256"]
    offenders = [key for key in digests if "\\" in key]
    assert not offenders, (
        f"{module_name} keyed its serialized identity with backslashes, "
        f"which the forecast reader cannot look up: {offenders}")


def test_the_benchmark_producer_covers_every_name_the_reader_demands():
    """Producer and reader, checked against each other rather than a list.

    ``_HRRR_DECODE_SOURCES`` is the reader's demand; the benchmark's
    ``_source_identity`` is what a real cache is keyed with.  The
    released defect was these two agreeing on Linux and not on Windows,
    so the agreement itself is worth a test rather than only the
    spelling.
    """

    from hrrr_single_domain_benchmark import _source_identity

    keys = set(_source_identity()["source_sha256"])
    assert set(runner._HRRR_DECODE_SOURCES) <= keys


def test_bundle_writer_refuses_a_start_time_that_is_not_the_lead(tmp_path):
    """A cycle/lead pair that does not date the experiment is refused.

    The two vocabularies are the thing this route gets wrong most
    easily: an absolute NOAA lead of one cycle, and the model's own
    start.  A bundle whose cycle plus first lead is not the experiment
    start would produce forcing frames dated at times the run never
    reaches, so it is refused at publication rather than at restore.
    """

    with pytest.raises(HrrrBundleError, match="is not cycle"):
        _hrrr_bundle(tmp_path, publish_cycle=CYCLE + timedelta(hours=1))


# ---------------------------------------------------------------------
# the prepare front door: opting into the portable bundle
# ---------------------------------------------------------------------

def test_the_prepare_front_door_forwards_the_portable_opt_in(tmp_path):
    """``--wps-namelist`` on the single-domain HRRR route means one thing.

    It used to be refused outright ("not used by --source hrrr").  It
    now selects the PORTABLE publication, and omitting it has to leave
    the certified native command exactly as it was -- that second half
    is the reproducibility half and is asserted here beside the first.
    """

    import argparse

    from gpuwm.source_cli import _hrrr_command, _required_hrrr_args

    def _args(**extra):
        base = dict(
            root_preparation=None, source_root=tmp_path,
            source_sha256s=tmp_path / "SHA256SUMS",
            source_sha256s_sha256="a" * 64,
            namelist_input=tmp_path / "namelist.input",
            valid_time="2026-08-05_04:00:00", output_root=tmp_path / "out",
            physics_profile=None, run_seconds=7200,
            forecast_start_hour=0, forecast_end_hour=2,
            history_interval_seconds=900.0, geog_root=None,
            static_cache=tmp_path / "s.npz",
            static_receipt=tmp_path / "s.json",
            domain_spec=tmp_path / "domain.json", prepare_workers=None,
            preprocess_backend=None, preprocess_workers=None,
            cpu_preprocess_bridge=None, sealed_prepared_cache=False,
            extend_root_preparation=None, ack=(), pipeline_workers=None,
            wps_namelist=None, child_workers=None,
        )
        base.update(extra)
        return argparse.Namespace(**base)

    without = _hrrr_command(_args())
    assert "--wps-namelist" not in without

    with_wps = _hrrr_command(
        _args(wps_namelist=tmp_path / "namelist.wps"))
    assert "--wps-namelist" in with_wps
    # and the only difference is that pair
    index = with_wps.index("--wps-namelist")
    assert with_wps[:index] + with_wps[index + 2:] == without


def _run_wrapper(tmp_path, monkeypatch, *, publish: bool):
    """Drive ``tools/prepare_hrrr_wrf.py`` with a stand-in benchmark.

    The stand-in produces what the real benchmark produces -- the
    preparation report, the sealed bridge manifest, the prepared cache,
    and, when asked, the experiment authority rendered by the SHIPPED
    renderer.  Everything downstream of it is production code: the
    wrapper's own publication step and the bundle writer.
    """

    import tools.prepare_hrrr_wrf as prepare
    from test_prepare_hrrr_wrf import _cold_start_receipt

    run_seconds, history_seconds = 7200.0, 900.0
    tables, exp = _hrrr_experiment(
        run_seconds=run_seconds,
        history_interval_seconds=history_seconds)
    forcing_hours = (0, 1, 2)

    source = tmp_path / "source"
    source.mkdir()
    for hour in SOURCE_HOURS:
        (source / f"hrrr.t04z.wrfnatf{hour:02d}.grib2").write_bytes(
            f"atmos-{hour}".encode())
        (source / f"hrrr.t04z.soilf{hour:02d}.grib2").write_bytes(
            f"soil-{hour}".encode())
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    wps = tmp_path / "namelist.wps"
    for path in (static_cache, static_receipt, namelist, wps):
        path.write_bytes(b"fixture")
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)
    output = tmp_path / "output"

    def fake_run(command, _env, cwd=None):
        if "gpuwm.wrf_direct" in command:
            # What the stock-WRF export leaves behind.  It is simulated
            # because the step is fail-closed by ruling (2026-08-04):
            # a REQUESTED export must land a manifest and every file the
            # manifest declares, or prepare_hrrr_wrf refuses rather than
            # reporting PASS over outputs that do not exist.  A fixture
            # that runs the step and writes nothing is asking for that
            # refusal.
            export_out = Path(command[command.index("--output") + 1])
            export_out.mkdir(parents=True, exist_ok=True)
            names = ("wrfinput_d01", "wrfbdy_d01")
            for name in names:
                (export_out / name).write_bytes(name.encode() + b"\n")
            (export_out / "manifest.json").write_text(json.dumps({
                "schema": "gpuwm-wrf-direct-export-v1",
                "files": {
                    name: {"bytes": (export_out / name).stat().st_size,
                           "sha256": _sha256(export_out / name)}
                    for name in names
                },
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        if any(str(value).endswith("write_hrrr_native_geometry_receipt.py")
               for value in command):
            receipt = Path(command[command.index("--output") + 1])
            static = Path(command[command.index("--static-cache") + 1])
            receipt.write_text(json.dumps({
                "schema": "gpuwm-native-static-direct-v1", "status": "PASS",
                "geometry": {"test": "hrrr"},
                "cache": {"path": static.name,
                          "bytes": static.stat().st_size,
                          "sha256": _sha256(static)},
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        if not any(str(value).endswith("hrrr_single_domain_benchmark.py")
                   for value in command):
            return
        # what the real benchmark leaves behind
        if "--publish-experiment-config" in command:
            config = Path(
                command[command.index("--publish-experiment-config") + 1])
            publish_experiment_document(config, tables, exp)
        bridge = Path(command[command.index("--bridge") + 1]) / "SHA256SUMS"
        bridge.parent.mkdir(parents=True, exist_ok=True)
        bridge.write_bytes(b"sealed-bridge\n")
        cache = Path(command[command.index("--prepared-cache") + 1])
        identity = prepared_cache_identity(
            bridge_manifest_sha256=_sha256(bridge),
            source_manifest_sha256=_sha256(source_manifest),
            static_cache_sha256=_sha256(static_cache),
            namelist_sha256=_sha256(namelist),
            domain_config=exp.root, forcing_hours=forcing_hours,
            source_identity={
                "source_sha256": {
                    name: hashlib.sha256(name.encode()).hexdigest()
                    for name in runner._HRRR_DECODE_SOURCES},
                "source_cycle": CYCLE.isoformat(),
                "model_start_time": exp.start_time.isoformat(),
                "source_forecast_hours": list(SOURCE_HOURS),
                "model_forcing_hours": list(forcing_hours),
            })
        _write_prepared_cache(cache, identity, forcing_hours, exp)
        outdir = Path(command[command.index("--outdir") + 1])
        outdir.mkdir(parents=True)
        (outdir / "report.json").write_text(json.dumps({
            "status": "PASS",
            "history_interval_seconds": history_seconds,
            "physics": {
                "schema": "gpuwm-prepared-physics-profile-v1",
                "profile": PROFILE,
                "hrrr_initialization": _cold_start_receipt(
                    PROFILE),
            },
            "preparation": {
                "preprocess_backend": {"backend": "cuda"},
                "preprocess_worker_budget": {
                    "schema": "gpuwm-preprocess-worker-budget-v1",
                    "backend": "cuda", "applicable": False,
                    "pipeline_decoder_workers_included": False,
                    "peak_active_native_workers": 0,
                },
            },
            "pipeline": {"workers": {"requested": "8", "selected": 8}},
        }), encoding="utf-8")

    monkeypatch.setattr(prepare, "_run", fake_run)
    argv = [
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--physics-profile", PROFILE,
        "--valid-time", "2026-08-05_04:00:00",
        "--forecast-start-hour", str(SOURCE_HOURS[0]),
        "--forecast-end-hour", str(SOURCE_HOURS[-1]),
        "--run-seconds", str(int(run_seconds)),
        "--history-interval-seconds", str(history_seconds),
        "--output-root", str(output),
    ]
    if publish:
        argv[argv.index("--namelist-input") + 2:
             argv.index("--namelist-input") + 2] = [
                 "--wps-namelist", str(wps)]
    assert prepare.main(argv) == 0
    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    return output, receipt, run_seconds, history_seconds, forcing_hours


def test_the_wrapper_publishes_a_bundle_the_front_door_admits(
        tmp_path, monkeypatch):
    """The whole chain, from the wrapper's argv to the driver's binding.

    What this pins is that the digests the wrapper prints are exactly the
    digests ``preflight_prepared_forecast`` wants -- a caller runs the
    case it just prepared without hashing a file by hand.
    """

    output, receipt, run_seconds, history_seconds, forcing_hours = (
        _run_wrapper(tmp_path, monkeypatch, publish=True))
    handoff = receipt["portable_bundle"]
    assert handoff is not None

    _bind_synthetic_geometry(monkeypatch)
    inputs = runner.preflight_prepared_forecast(
        source="hrrr", prepared_root=output,
        proof_sha256=handoff["proof_sha256"],
        source_manifest_sha256=handoff["source_manifest_sha256"],
        prepared_content_sha256=handoff["prepared_content_sha256"],
        experiment_config=Path(handoff["experiment_config"]),
        wps_namelist=Path(handoff["wps_namelist"]),
        physics_profile=PROFILE,
        run_seconds=run_seconds,
        history_interval_seconds=history_seconds)
    assert inputs.layout == runner.HRRR_DIRECT_LAYOUT
    assert inputs.forcing_hours == forcing_hours
    assert inputs.proof["source_forecast_hours"] == list(SOURCE_HOURS)


def test_a_bare_preparation_publishes_the_bundle_and_sim_accepts_it(
        tmp_path, monkeypatch):
    """The default run, with no --wps-namelist, is a runnable tree.

    This is the shakeout's exact argv shape.  Before, it produced a
    finished native bundle carrying none of the portable authorities,
    and ``gpuwm sim`` refused it -- calling it partial, which it was
    not.  A prepared tree with no front door is not a prepared tree, so
    publication is the default now and the flag only chooses WHICH
    namelist gets bound.

    Every artifact is asserted where a reader would look for it, and the
    tree is then handed to ``gpuwm sim``'s own resolver rather than to a
    restatement of what that resolver wants.
    """

    from gpuwm import stage_cli
    from gpuwm.hrrr_prepared_bundle import (
        EXPERIMENT_CONFIG_NAME, PROOF_NAME, SOURCE_MANIFEST_NAME,
        WPS_NAMELIST_NAME, WRF_NAMELIST_NAME)

    output, receipt, *_rest = _run_wrapper(
        tmp_path, monkeypatch, publish=False)
    handoff = receipt["portable_bundle"]
    assert handoff is not None
    assert receipt["portable_bundle_refusal"] is None
    for name in (PROOF_NAME, SOURCE_MANIFEST_NAME, EXPERIMENT_CONFIG_NAME,
                 WPS_NAMELIST_NAME, WRF_NAMELIST_NAME):
        assert (output / name).is_file(), name
    # the artifacts the native route has always written are untouched
    assert (output / "native" / "prepared-cache" / "header.json").is_file()
    assert (output / "public-wrapper-result.json").is_file()

    resolved = stage_cli.resolve_bundle(output)
    assert resolved["source"] == "hrrr"
    assert resolved["layout"] == "single"
    command = stage_cli.sim_command(
        resolved, experiment_config=output / EXPERIMENT_CONFIG_NAME,
        wps_namelist=output / WPS_NAMELIST_NAME, outdir=tmp_path / "out")
    assert (command[command.index("--proof-sha256") + 1]
            == handoff["proof_sha256"])


def test_a_publication_that_cannot_finish_leaves_the_native_tree_and_says_why(
        tmp_path, monkeypatch):
    """The native preparation is not held hostage by the portable half.

    A tree whose portable authorities could not be published is still a
    complete native tree that the native benchmark runs, so the
    preparation succeeds -- and the reason is recorded in the receipt
    where ``gpuwm sim`` reads it back, rather than being lost to a
    stderr line nobody kept.
    """

    import tools.prepare_hrrr_wrf as prepare

    from gpuwm import stage_cli
    from gpuwm.hrrr_prepared_bundle import HrrrBundleError, PROOF_NAME

    def refuse(**_kwargs):
        raise HrrrBundleError("a prepared case needs at least two frames")

    monkeypatch.setattr(
        prepare, "publish_hrrr_prepared_bundle", refuse)
    output, receipt, *_rest = _run_wrapper(
        tmp_path, monkeypatch, publish=False)

    assert receipt["status"] == "PASS"
    assert receipt["portable_bundle"] is None
    assert ("a prepared case needs at least two frames"
            in receipt["portable_bundle_refusal"])
    assert not (output / PROOF_NAME).exists()
    assert (output / "native" / "prepared-cache" / "header.json").is_file()

    with pytest.raises(stage_cli.StageRefusal) as refusal:
        stage_cli.resolve_bundle(output)
    sentence = str(refusal.value)
    assert "a prepared case needs at least two frames" in sentence
    assert "complete, not partial" in sentence

"""Public HRRR wrapper source-window orchestration regressions.

The cold-start receipt this wrapper validates is **not** hand-built here.
It used to be, and that is exactly how a one-line schema desync shipped
through 800+ green tests: ``tools/hrrr_single_domain_benchmark.py`` bumped
the receipt to ``...-initialization-v2`` while this consumer still demanded
``v1``, so the documented HRRR preparation front door raised ``RuntimeError:
HRRR preparation omitted deterministic cold-start evidence`` *after* a fully
successful preparation, for every one of the seven profiles -- and the suite
could not see it, because the fixture below constructed the receipt it
wanted rather than the one the producer emits.

So the fixture now RUNS the producer: real decoded native fields through
``gpuwm.ingest.real.initialize_real`` and then through
``_initial_hrrr_microphysics_receipt``, the same function the benchmark
binds into ``report.json``.  A future schema bump, a renamed field, or a
dropped correspondence breaks this consumer's tests the moment the producer
changes, which is the only property that would have caught the regression.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.physics_compat import (
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from gpuwm.ingest.prepared_cache import (
    _reader_boundaries, PreparedCacheReader,
    prepared_domain_config_identity, write_prepared_cache,
)
from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
from gpuwm.core.grid import BaseState, make_vertical_coord
from gpuwm.hrrr_native_static import _array_sha256, sha256_file
from gpuwm.hrrr_hierarchy_direct import _expected_root_cache_identity
from gpuwm.native_wrf_contract import native_geometry_contract
from gpuwm.native_domain_artifacts import write_native_hierarchy_artifacts
from gpuwm.static.lambert import grids_from_projection_config
from gpuwm.stream import _materialize_input_namelist
from gpuwm.vertical_contract import explicit_vertical_from_wrf_namelist
from gpuwm.experiment import load_experiment
from gpuwm.io import restart as restart_io
from tools import prepare_hrrr_wrf as prepare
from tools import prepared_domain_tree_forecast as tree_runner
from tools.hrrr_single_domain_benchmark import (
    _experiment, _initial_hrrr_microphysics_receipt)

# The real decoded-source -> initialized-state driver.  It lives beside the
# producer's own tests; importing it is what makes this a cross-lane seam
# test instead of a second hand-built fixture.
from test_hrrr_single_domain_benchmark import (
    _decoded_native_hrrr_initialization, _write_native_physics_namelist)
from test_hrrr_native_static import _fixture as _native_static_fixture
from test_prepared_cache import (
    _extension_identity, _fixture as _prepared_fixture,
    _suffix_fixture as _prepared_suffix_fixture,
)
from test_prepared_domain_tree_forecast import _write_two_domain_config
from test_native_domain_artifacts import _inputs as _domain_artifact_inputs
from test_restart import (
    _cfg as _restart_cfg, _fill_setup as _restart_fill_setup,
    _sealed_tree_fixture,
    _shim_state as _restart_shim_state,
)


#: The microphysics option each fixed profile pins, so the producer runs the
#: scheme whose cold start the consumer is about to validate.
_PROFILE_MICROPHYSICS = {
    KESSLER_PROFILE_ID: 1,
    WSM6_PROFILE_ID: 6,
    THOMPSON_PROFILE_ID: 8,
    MORRISON_PROFILE_ID: 10,
    NSSL2_PROFILE_ID: 18,
    NSSL2_LEGACY_RRTMG_PROFILE_ID: 18,
}


@functools.lru_cache(maxsize=None)
def _producer_receipt(profile: str) -> dict[str, object]:
    """Whatever the real producer emits for ``profile``, verbatim."""

    result = _decoded_native_hrrr_initialization(
        _PROFILE_MICROPHYSICS[profile])
    return _initial_hrrr_microphysics_receipt(
        result.state, profile, result.hydrometeor_initialization)


def _cold_start_receipt(profile: str) -> dict[str, object]:
    return copy.deepcopy(_producer_receipt(profile))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_decoder(tmp_path: Path) -> Path:
    """Executable decoder fixture; all orchestration around it is production.

    The fixture replaces only GRIB interpretation with small deterministic
    meteorological arrays.  The public wrapper still launches a real process,
    consumes the production READY protocol, maps/initializes through the native
    CPU backend, seals the bridge, writes the cache, and later extends both.
    """

    script = tmp_path / "minimal_hrrr_decoder.py"
    script.write_text(textwrap.dedent(r'''
        from pathlib import Path
        import os
        import shutil
        import sys
        import time

        import numpy as np

        def put(path, value):
            np.ascontiguousarray(value, dtype="<f4").tofile(path)

        def ready(path, rows):
            path.write_text("".join(f"{key}\t{value}\n" for key, value in rows),
                            encoding="ascii")

        def main(argv):
            if len(argv) != 11 or argv[1] != "--series-workers-ready":
                raise SystemExit(f"unexpected decoder arguments: {argv!r}")
            workers, series, output, signals, cycle = (
                argv[2], Path(argv[3]), Path(argv[4]), Path(argv[5]), argv[6])
            i0, i1, j0, j1 = map(int, argv[7:11])
            rows = [line.split("\t") for line in series.read_text().splitlines()
                    if line.strip()]
            hours = [int(row[0]) for row in rows]
            ny, nx = j1 - j0 + 1, i1 - i0 + 1
            started = time.perf_counter()
            # On Windows the executable path is a .cmd shim; the producer owns
            # that waiting cmd.exe PID, while this Python worker is its child.
            if os.name == "nt":
                import psutil
                marker = Path(__file__).with_suffix(".cmd").name.lower()
                parents = psutil.Process().parents()
                launchers = []
                for candidate in parents:
                    try:
                        command = " ".join(candidate.cmdline()).lower()
                        name = candidate.name().lower()
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        continue
                    if name in {"cmd", "cmd.exe"} and marker in command:
                        launchers.append(candidate)
                # The nearest marker-bearing cmd.exe is the process returned
                # by Popen.  No fixed ancestry depth: redirectors and shell
                # policy can add ancestors, but cannot change this marker.
                owner_pid = (launchers[0] if launchers
                             else parents[0]).pid
            else:
                owner_pid = os.getpid()
            staging = output.parent / f".{output.name}.partial-{owner_pid}"
            publish = output.parent / f".{output.name}.publish-{owner_pid}"
            staging.mkdir(parents=True)
            signals.mkdir(parents=True)
            gate = (
                "status\tPASS\n"
                f"cycle\t{cycle}\n"
                "atmosphere_selected_per_time\t561\n"
                "hybrid_levels\t50\n"
                "soil_selected_per_time\t18\n"
                f"window_zero_based_inclusive\ti={i0}..{i1} j={j0}..{j1}\n"
                f"window_shape\t{ny}x{nx}\n"
                "qice_mapping\tPASS discipline=0 category=1 parameter=82 "
                "level_type=105; deterministic minimal fixture\n"
                "cross_time_inventory\tPASS deterministic minimal fixture\n"
                f"forecast_hours\t{','.join(map(str, hours))}\n"
                f"series_count\t{len(hours)}\n"
            )
            (staging / "gate.txt").write_text(gate, encoding="ascii")
            ready(signals / "preflight.ready", (
                ("status", "PASS"), ("series_count", len(hours)),
                ("staging_retention", "until_consumer_finish"),
                ("canonical_output", output.resolve()), ("workers", workers),
                ("producer_elapsed_seconds", time.perf_counter() - started),
                ("staging_root", staging.resolve()),
            ))
            pressure_levels = np.linspace(5000.0, 100000.0, 50,
                                          dtype=np.float32)
            pressure = np.broadcast_to(
                pressure_levels[:, None, None], (50, ny, nx)).copy()
            height = np.broadcast_to(
                -7900.0 * np.log(pressure_levels / 100000.0)[:, None, None],
                (50, ny, nx)).copy()
            level = np.arange(50, dtype=np.float32)[:, None, None]
            row = np.arange(ny, dtype=np.float32)[None, :, None]
            column = np.arange(nx, dtype=np.float32)[None, None, :]
            for hour in hours:
                atmosphere = staging / f"atmosphere-f{hour:02d}"
                soil = staging / f"soil-f{hour:02d}"
                atmosphere.mkdir()
                soil.mkdir()
                temperature = (215.0 + 75.0 * (pressure / 100000.0) ** 0.22
                               + np.float32(hour) * np.float32(0.05))
                fields3 = {
                    "PRES": pressure, "HGT": height, "TT": temperature,
                    "SPFH": np.full_like(pressure, 0.004),
                    "U_MASS": np.full_like(pressure, 8.0 + 0.1 * hour),
                    "V_MASS": np.full_like(pressure, -2.0 + 0.05 * hour),
                }
                for index, name in enumerate(("QC", "QI", "QR", "QS", "QG"), 1):
                    fields3[name] = np.asarray(
                        index * 1.0e-7 * (1.0 + level + 0.01 * row
                                          + 0.001 * column), dtype=np.float32)
                for name, value in fields3.items():
                    put(atmosphere / f"{name}.f32le", value)
                surface = np.ones((ny, nx), dtype=np.float32)
                fields2 = {
                    "PSFC": surface * 100000.0,
                    "SOILHGT": surface * 100.0,
                    "SKINTEMP": surface * (289.0 + 0.05 * hour),
                    "SNOW": surface * 0.0, "SNOWH": surface * 0.0,
                    "T2": surface * (289.0 + 0.05 * hour),
                    "Q2": surface * 0.004,
                    "U10_MASS": surface * (7.0 + 0.1 * hour),
                    "V10_MASS": surface * (-1.0 + 0.05 * hour),
                    "LANDSEA": surface, "XICE": surface * 0.0,
                }
                for name, value in fields2.items():
                    put(atmosphere / f"{name}.f32le", value)
                put(soil / "SOILT.f32le", np.full((9, ny, nx),
                                                   283.0 + 0.05 * hour))
                put(soil / "SOILW.f32le", np.full((9, ny, nx), 0.25))
                ready(signals / f"f{hour:02d}.ready", (
                    ("status", "PASS"), ("forecast_hour", hour),
                    ("payload_files", 24),
                    ("producer_elapsed_seconds", time.perf_counter() - started),
                ))
            shutil.copytree(staging, publish)
            os.replace(publish, output)
            ready(signals / "complete.ready", (
                ("status", "PASS"), ("series_count", len(hours)),
                ("staging_retained", "true"),
                ("canonical_output", output.resolve()),
                ("producer_elapsed_seconds", time.perf_counter() - started),
            ))

        if __name__ == "__main__":
            main(sys.argv)
    ''').lstrip(), encoding="utf-8")
    if os.name == "nt":
        launcher = tmp_path / "minimal_hrrr_decoder.cmd"
        launcher.write_text(
            f'@echo off\n"{sys.executable}" "{script}" %*\n',
            encoding="ascii")
    else:
        launcher = tmp_path / "minimal_hrrr_decoder"
        launcher.write_text(
            f"#!{sys.executable}\n"
            f"exec(compile(open({str(script)!r}, 'rb').read(), "
            f"{str(script)!r}, 'exec'))\n",
            encoding="utf-8")
        launcher.chmod(0o755)
    return launcher.resolve()


def _real_wrapper_inputs(tmp_path: Path):
    """Small input data for the unmodified public preparation tool path."""

    static_root = tmp_path / "static-fixture"
    static_root.mkdir()
    target, static_cache, static_receipt = _native_static_fixture(static_root)
    with np.load(static_cache, allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name]).copy()
                  for name in archive.files}
    fields["MAPFAC_M"].fill(1.0)
    fields["LANDMASK"].fill(1.0)
    fields["HGT_M"].fill(100.0)
    fields["F"].fill(8.0e-5)
    fields["SOILTEMP"].fill(285.0)
    fields["TMN"].fill(285.0)
    fields["GREENFRAC"].fill(0.5)
    fields["LAI12M"].fill(2.0)
    fields["ALBEDO12M"].fill(0.2)
    for name in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        fields[name].fill(0.0)
        fields[name][0].fill(1.0)
    np.savez(static_cache, **fields)
    receipt = json.loads(static_receipt.read_text(encoding="utf-8"))
    receipt["cache"].update({
        "bytes": static_cache.stat().st_size,
        "sha256": sha256_file(static_cache),
    })
    receipt["array_sha256"] = {
        name: _array_sha256(value) for name, value in sorted(fields.items())}
    receipt["geometry"] = native_geometry_contract(
        target.grid(), target.contract_cfg())
    static_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    domain = tmp_path / "domain.json"
    domain.write_text(
        json.dumps(target.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    namelist = tmp_path / "namelist.input"
    _write_native_physics_namelist(namelist)
    eta = ", ".join(map(str, np.linspace(1.0, 0.0, target.nz + 1)))
    text = namelist.read_text(encoding="ascii").replace(
        "&dynamics\n", "&dynamics\n hybrid_opt = 2,\n etac = 0.2,\n")
    namelist.write_text(
        "&time_control\n/\n" + text + ("&domains\n"
                " max_dom = 2,\n"
                f" e_vert = {target.nz + 1},\n"
                " p_top_requested = 10000.0,\n"
                f" eta_levels = {eta},\n"
                "/\n"),
        encoding="ascii")
    return target, static_cache, static_receipt, domain, namelist


def _fake_sealed_bridge(root: Path, hours, *, cycle="2026-07-18 05:00:00"):
    root.mkdir()
    for hour in hours:
        for role, count in (("atmosphere", 22), ("soil", 2)):
            directory = root / f"{role}-f{hour:02d}"
            directory.mkdir()
            for index in range(count):
                (directory / f"field-{index:02d}.f32le").write_bytes(
                    f"{role}:{hour}:{index}".encode("ascii"))
    gate = root / "gate.txt"
    gate.write_text(
        "status\tPASS\n"
        f"forecast_hours\t{','.join(map(str, hours))}\n"
        f"series_count\t{len(hours)}\n"
        f"cycle\t{cycle}\n"
        "window_shape\t207x207\n",
        encoding="utf-8")
    payloads = sorted(
        path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{_file_sha256(path)}  ./{path.relative_to(root).as_posix()}\n"
            for path in payloads),
        encoding="utf-8")
    return root


@pytest.mark.parametrize("profile", sorted(_PROFILE_MICROPHYSICS))
def test_the_consumer_accepts_exactly_what_the_producer_emits(profile: str):
    """The binding the schema desync escaped through.

    Nothing here is written down twice: the receipt comes from the producer
    and the constant comes from the consumer, so they cannot drift apart
    silently again.
    """

    receipt = _producer_receipt(profile)
    assert receipt["schema"] == prepare.HRRR_INITIALIZATION_SCHEMA
    assert prepare._validated_physics_receipt(
        {"physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": profile,
            "hrrr_initialization": receipt,
        }},
        requested_profile=profile)["profile"] == profile


def test_the_consumer_refuses_the_previous_producer_schema():
    """v1/v2 acceptance is deliberately not retained.

    ``main`` runs the producer itself and reads back the ``report.json``
    that invocation just wrote, so an older-schema receipt cannot reach this
    consumer; accepting one would widen the gate for a case that does not
    exist.  Dropping the version is therefore a refusal, not a widening.
    """

    for stale in ("gpuwm-hrrr-microphysics-initialization-v1",
                  "gpuwm-hrrr-microphysics-initialization-v2",
                  "gpuwm-hrrr-microphysics-initialization-v3"):
        receipt = _cold_start_receipt(NSSL2_PROFILE_ID)
        receipt["schema"] = stale
        with pytest.raises(RuntimeError, match="cold-start evidence"):
            prepare._validated_physics_receipt(
                {"physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": NSSL2_PROFILE_ID,
                    "hrrr_initialization": receipt,
                }},
                requested_profile=NSSL2_PROFILE_ID)


def test_the_consumer_validates_the_correspondence_fields_not_the_string():
    """Every v2/v3/v4 field the producer added is checked, one at a time."""

    def _refuse(mutate, match="correspondence evidence"):
        receipt = _cold_start_receipt(NSSL2_PROFILE_ID)
        mutate(receipt)
        with pytest.raises(RuntimeError, match=match):
            prepare._validated_physics_receipt(
                {"physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": NSSL2_PROFILE_ID,
                    "hrrr_initialization": receipt,
                }},
                requested_profile=NSSL2_PROFILE_ID)

    _refuse(lambda r: r.pop("source_to_state_correspondence"))
    _refuse(lambda r: r.pop("retention_evidence"))
    _refuse(lambda r: r.pop("state_mass_fields"))
    _refuse(lambda r: r.pop("vertical_disposition"))
    _refuse(lambda r: r["source_to_state_correspondence"].pop("QG"))
    _refuse(lambda r: r["state_mass_fields"]["qg"].pop("nonzero_mask_sha256"))
    _refuse(lambda r: r["state_mass_fields"]["qg"].pop("decoded_source"))
    # A species claimed PROVEN on a source that carried no mass.
    _refuse(
        lambda r: r["retention_evidence"]["QG"].update(
            {"source_nonzero_count": 0}),
        match="disagrees with its own fingerprints")
    # Evidence that disagrees with the fingerprints beside it.
    _refuse(
        lambda r: r["retention_evidence"]["QG"].update(
            {"state_nonzero_count": 1}),
        match="disagrees with its own fingerprints")
    # The headline B-H1 failure: every nonzero analyzed cell lost.
    def _lose_all_graupel(receipt):
        receipt["state_mass_fields"]["qg"]["nonzero_count"] = 0
        receipt["retention_evidence"]["QG"]["state_nonzero_count"] = 0
    _refuse(_lose_all_graupel, match="lost every nonzero analyzed QG")


def test_the_consumer_reports_a_vacuous_retention_claim_without_refusing_it():
    """A cloud-free analysis is a legitimate preparation, and says so.

    The wrapper must not pass a vacuous receipt off as proof, and must not
    refuse it either -- a genuinely cloud-free domain prepares correctly.
    """

    empty = _decoded_native_hrrr_initialization(18, analyzed_species=())
    receipt = _initial_hrrr_microphysics_receipt(
        empty.state, NSSL2_PROFILE_ID, empty.hydrometeor_initialization)
    summary = prepare._validated_retention_evidence(receipt)
    assert summary["strength"] == "VACUOUS"
    assert summary["vacuous_species"] == ["QC", "QG", "QI", "QR", "QS"]

    proven = prepare._validated_retention_evidence(
        _producer_receipt(NSSL2_PROFILE_ID))
    assert proven["strength"] == "PARTIALLY_WRF_EXCLUDED"
    assert proven["vacuous_species"] == []
    assert proven["partially_excluded_species"] == [
        "QC", "QG", "QI", "QR", "QS"]


def test_the_consumer_accepts_an_exhaustive_wrf_exclusion_partition():
    excluded = _decoded_native_hrrr_initialization(
        18, analyzed_source_levels={"QC": 5})
    receipt = _initial_hrrr_microphysics_receipt(
        excluded.state, NSSL2_PROFILE_ID,
        excluded.hydrometeor_initialization)

    summary = prepare._validated_retention_evidence(receipt)

    assert summary["strength"] == "WRF_EXCLUDED"
    assert summary["excluded_species"] == ["QC"]
    assert summary["vacuous_species"] == ["QG", "QI", "QR", "QS"]


def test_decoder_workers_are_independent_of_the_native_preprocess_budget():
    """The wrapper's own point: two worker budgets, separately accounted.

    Written at the decode pipeline's real ceiling.  It used to be
    written at 64 -- a number no run could reach, because the sealed
    decoder refuses anything over 13 and the pipeline refused it in
    turn, after the static build had already been paid for.
    """
    workers = prepare.MAX_PIPELINE_WORKERS
    report = {
        "status": "PASS",
        "preparation": {
            "preprocess_backend": {"backend": "cuda"},
            "preprocess_worker_budget": {
                "schema": "gpuwm-preprocess-worker-budget-v1",
                "backend": "cuda",
                "applicable": False,
                "pipeline_decoder_workers_included": False,
                "peak_active_native_workers": 0,
            },
        },
        "pipeline": {"workers": {"requested": str(workers),
                                 "selected": workers}},
    }

    _preprocess, native_budget, decoder = prepare._validated_worker_receipts(
        report, selected_backend="cuda", requested_preprocess_workers=None,
        requested_pipeline_workers=str(workers), final_hour=12)

    assert decoder["selected"] == workers
    assert native_budget["pipeline_decoder_workers_included"] is False


def test_pipeline_workers_refuses_over_the_decoders_ceiling_at_parse_time():
    """The late refusal this replaces cost a whole static build first.

    The wrapper advertised 1..64 while the decoder it spawns accepts
    1..13, so `--pipeline-workers 32` was accepted, the geometry receipt
    and native static cache were built, and only then did the pipeline
    refuse.  argparse now answers with a usage message instead.
    """
    from tools.hrrr_pipeline import MAX_PIPELINE_WORKERS

    assert prepare.MAX_PIPELINE_WORKERS == MAX_PIPELINE_WORKERS

    parser = prepare._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--source-root", "src", "--source-manifest", "m",
            "--source-manifest-sha256", "0" * 64, "--namelist-input", "n",
            "--valid-time", "2026-08-01_00:00:00", "--output-root", "out",
            "--static-cache", "c", "--static-receipt", "r",
            "--pipeline-workers", str(MAX_PIPELINE_WORKERS + 1)])

    # Watched firing: the ceiling itself, and 'auto', both parse.
    for accepted in (str(MAX_PIPELINE_WORKERS), "auto"):
        args = parser.parse_args([
            "--source-root", "src", "--source-manifest", "m",
            "--source-manifest-sha256", "0" * 64, "--namelist-input", "n",
            "--valid-time", "2026-08-01_00:00:00", "--output-root", "out",
            "--static-cache", "c", "--static-receipt", "r",
            "--pipeline-workers", accepted])
        assert args.pipeline_workers == accepted
def test_bridge_extension_hardlinks_prefix_and_one_absolute_new_hour(tmp_path):
    prior = _fake_sealed_bridge(tmp_path / "prior", [0, 1])
    suffix = _fake_sealed_bridge(tmp_path / "suffix", [1, 2])
    output = tmp_path / "merged"

    receipt = prepare._bridge_manifest_extension(
        predecessor=prior, suffix=suffix, output=output,
        old_hours=[0, 1], new_hours=[0, 1, 2])

    assert receipt["old_source_forecast_hours"] == [0, 1]
    assert receipt["suffix_source_forecast_hours"] == [1, 2]
    assert receipt["new_source_forecast_hours"] == [0, 1, 2]
    assert receipt["extended_sha256"] == _file_sha256(
        output / "SHA256SUMS")
    assert (output / "atmosphere-f00" / "field-00.f32le").samefile(
        prior / "atmosphere-f00" / "field-00.f32le")
    assert (output / "atmosphere-f01" / "field-00.f32le").samefile(
        prior / "atmosphere-f01" / "field-00.f32le")
    assert (output / "atmosphere-f02" / "field-00.f32le").samefile(
        suffix / "atmosphere-f02" / "field-00.f32le")
    gate = dict(
        row.split("\t", 1)
        for row in (output / "gate.txt").read_text().splitlines())
    assert gate["forecast_hours"] == "0,1,2"
    assert gate["series_count"] == "3"
    entries = prepare._manifest_entries(output / "SHA256SUMS")
    prepare._verify_manifest_payloads(output, entries)


def test_bridge_extension_refuses_changed_suffix_authority(tmp_path):
    prior = _fake_sealed_bridge(tmp_path / "prior", [0, 1])
    suffix = _fake_sealed_bridge(
        tmp_path / "suffix", [1, 2], cycle="2026-07-18 06:00:00")

    with pytest.raises(ValueError, match="immutable gate field cycle"):
        prepare._bridge_manifest_extension(
            predecessor=prior, suffix=suffix, output=tmp_path / "merged",
            old_hours=[0, 1], new_hours=[0, 1, 2])


def test_source_manifest_extension_requires_exact_unchanged_prefix(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    cycle = datetime(2026, 7, 18, 5)
    old_names = []
    for hour in (0, 1, 2):
        for product in ("wrfnat", "soil"):
            name = f"hrrr.t05z.{product}f{hour:02d}.grib2"
            (root / name).write_bytes(f"{product}:{hour}".encode("ascii"))
            if hour < 2:
                old_names.append(name)
    old = tmp_path / "prior-SHA256SUMS"
    old.write_text("".join(
        f"{_file_sha256(root / name)}  {name}\n"
        for name in sorted(old_names)), encoding="utf-8")
    current = root / "SHA256SUMS"
    current.write_text("".join(
        f"{_file_sha256(path)}  {path.name}\n"
        for path in sorted(root.glob("*.grib2"))), encoding="utf-8")

    receipt = prepare._source_manifest_extension(
        predecessor=old, extended=current, source_root=root,
        old_hours=[0, 1], new_hours=[0, 1, 2], cycle=cycle)

    assert receipt["retained_entries"] == 4
    assert [row["path"] for row in receipt["added_entries"]] == [
        "hrrr.t05z.soilf02.grib2", "hrrr.t05z.wrfnatf02.grib2"]
    target = root / old_names[0]
    old_digest = _file_sha256(target)
    target.write_bytes(b"mutated-prefix")
    changed = current.read_text().replace(
        old_digest, _file_sha256(target), 1)
    current.write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="changes its sealed prefix"):
        prepare._source_manifest_extension(
            predecessor=old, extended=current, source_root=root,
            old_hours=[0, 1], new_hours=[0, 1, 2], cycle=cycle)


def test_extension_refuses_legacy_hydrometeor_cache_before_suffix_work(
        tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    native = prior / "native"
    cache = native / "prepared-cache"
    report = native / "preparation-report"
    bridge = native / "native-bridge"
    for directory in (cache, report, bridge):
        directory.mkdir(parents=True, exist_ok=True)
    snapshot = native / "source-manifest.snapshot"
    snapshot.write_text("sealed\n", encoding="utf-8")
    for path in (
            report / "report.json",
            prior / "native-static.npz",
            prior / "native-static-receipt.json",
            prior / "native-geometry-receipt.json"):
        path.write_bytes(b"fixture")
    (prior / "public-wrapper-result.json").write_text(json.dumps({
        "status": "PASS",
        "source_cycle": "2026-07-18T05:00:00",
        "source_forecast_hours": [0, 1],
        "history_interval_seconds": 3600.0,
        "physics": {"profile": WSM6_PROFILE_ID},
        "prepared_cache_contract": {"mode": "sealed-prefix-v1"},
    }), encoding="utf-8")
    (cache / "header.json").write_text(json.dumps({
        "identity": {},
        "metadata": {
            "forcing_extension_mode": "sealed-prefix-v1",
            "hydrometeor_initialization": {
                "schema": "gpuwm-real-hydrometeor-correspondence-v1",
            },
        },
    }), encoding="utf-8")
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = source_root / "SHA256SUMS"
    source_manifest.write_text("extended\n", encoding="utf-8")
    suffix_runs = []
    monkeypatch.setattr(
        prepare, "_source_manifest_extension", lambda **_kwargs: {})
    monkeypatch.setattr(
        prepare, "_run", lambda *args, **kwargs:
        suffix_runs.append((args, kwargs)))
    args = SimpleNamespace(
        extend_root_preparation=prior,
        forecast_start_hour=0,
        run_seconds=7200.0,
        physics_profile=WSM6_PROFILE_ID,
        history_interval_seconds=3600.0,
        source_manifest=source_manifest,
        source_manifest_sha256=_file_sha256(source_manifest),
        source_root=source_root,
    )
    output = tmp_path / "extended"

    with pytest.raises(ValueError, match="rebuild it with the current GPUWM"):
        prepare._sealed_extension(
            args, valid_time=datetime(2026, 7, 18, 5),
            source_forecast_hours=(0, 1, 2), output=output,
            env={}, decoder=tmp_path / "decoder", started=0.0,
            namelist_invariant={
                "schema": "gpuwm-namelist-extension-invariant-v1",
                "sha256": "a" * 64,
            })

    assert suffix_runs == []
    assert not output.exists()


def _write_minimal_stream_tree_config(
        path: Path, *, cycle: datetime, run_seconds: float) -> Path:
    path.write_text(textwrap.dedent(f"""
        [experiment]
        name = "stream-extension-tree"
        start_time = {cycle.isoformat()}
        run_seconds = {run_seconds}
        restart_interval_s = 3600.0

        [projection]
        map_proj = "lambert"
        ref_lat = 35.0
        ref_lon = -97.0
        truelat1 = 30.0
        truelat2 = 60.0
        stand_lon = -97.0

        [shared]
        nz = 4
        ztop = 16000.0
        p_top = 10000.0
        eta_levels = [1.0, 0.75, 0.5, 0.25, 0.0]
        hybrid_opt = 2
        etac = 0.2
        map_proj = 1
        moist = true
        mp_physics = 6
        bl_pbl_physics = 1
        sf_sfclay_physics = 91
        sf_surface_physics = 2

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 30
        ny = 30
        dx = 12000.0
        time_step = 60
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 11
        j_parent_start = 11
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        nx = 6
        ny = 6
        history_interval_s = 1800.0
    """).lstrip(), encoding="utf-8")
    return path


def _stream_tree_static(ny: int, nx: int) -> dict[str, np.ndarray]:
    plane = np.ones((ny, nx), dtype=np.float32)
    landuse = np.zeros((21, ny, nx), dtype=np.float32)
    soil_category = np.zeros((16, ny, nx), dtype=np.float32)
    landuse[0] = 1.0
    soil_category[0] = 1.0
    return {
        "HGT_M": 100.0 * plane,
        "LANDMASK": plane,
        "LU_INDEX": plane,
        "SCT_DOM": plane,
        "SCB_DOM": plane,
        "LANDUSEF": landuse,
        "SOILCTOP": soil_category,
        "SOILCBOT": soil_category.copy(),
        "GREENFRAC": np.ones((12, ny, nx), dtype=np.float32),
        "LAI12M": np.ones((12, ny, nx), dtype=np.float32),
        "ALBEDO12M": np.ones((12, ny, nx), dtype=np.float32),
        "SNOALB": plane,
        "SOILTEMP": 278.0 * plane,
        "TMN": 278.0 * plane,
    }


def _build_and_preflight_stream_tree(
        root: Path, *, config: Path, wrapper: Path,
        cycle: datetime, forcing_hours: tuple[int, ...], root_cache_domain):
    experiment = load_experiment(config)
    grids = tuple(grids_from_projection_config(experiment))
    root_initial, root_met, root_soil, _static, _boundaries, _grid = \
        _domain_artifact_inputs()
    child_initial, child_met, child_soil, _static, _boundaries, _grid = \
        _domain_artifact_inputs()
    for initial in (root_initial, child_initial):
        initial.coord = make_vertical_coord(
            4, hybrid_opt=2, etac=0.2,
            eta_levels=(1.0, 0.75, 0.5, 0.25, 0.0))
        initial.base = BaseState(
            mub=np.full((2, 2), 90_000.0), p_top=10_000.0,
            pb=np.full((4, 2, 2), 50_000.0),
            alb=np.full((4, 2, 2), 0.8),
            thb=np.full((4, 2, 2), 290.0),
            phb=np.zeros((5, 2, 2)), terrain_z=np.zeros((2, 2)))
    wrapper_header = json.loads((
        wrapper / "native" / "prepared-cache" / "header.json"
    ).read_text(encoding="utf-8"))
    identity = wrapper_header["identity"]
    expected_root_identity = _expected_root_cache_identity(
        identity, root_domain=root_cache_domain,
        bridge_manifest_sha256=identity["bridge_manifest_sha256"],
        source_manifest_sha256=identity["source_manifest_sha256"],
        static_cache_sha256=identity["static_cache_sha256"],
        forcing_hours=forcing_hours)
    wrapper_reader = PreparedCacheReader(
        wrapper / "native" / "prepared-cache",
        expected_identity=expected_root_identity)
    wrapper_reader.verify_all()
    boundaries = _reader_boundaries(wrapper_reader)
    root_initial.state.lateral_boundaries = boundaries
    child_initial.state.lateral_boundaries = None
    child = SimpleNamespace(
        domain=experiment.domains[1], real=child_initial, grid=grids[1],
        horizontal=child_met, soil=child_soil,
        static_fields=_stream_tree_static(
            experiment.domains[1].run.ny, experiment.domains[1].run.nx),
        preprocess_receipt={"backend": "cpu", "workers": 1},
        input_preparation_seconds=0.0)
    root.mkdir()
    artifacts = write_native_hierarchy_artifacts(
        root / "hierarchy-artifacts", exp=experiment,
        root_grid=grids[0], root_initial_result=root_initial,
        root_met=root_met, root_soil=root_soil,
        root_static_fields=_stream_tree_static(
            experiment.root.run.ny, experiment.root.run.nx),
        root_boundaries=boundaries, child_results=(child,),
        bridge_manifest_sha256=identity["bridge_manifest_sha256"],
        source_manifest_sha256=identity["source_manifest_sha256"],
        namelist_sha256=identity["namelist_sha256"],
        forcing_hours=forcing_hours,
        source_identity=identity["source_identity"], valid_time=cycle,
        root_metadata={
            "source_prepared_content_sha256":
                wrapper_header["content_sha256"],
        })
    receipt = {
        "schema": tree_runner.HIERARCHY_SCHEMA,
        "status": "PASS",
        "valid_time": cycle.isoformat(),
        "domain_count": len(experiment.domains),
        "forcing_hours": list(forcing_hours),
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "provenance": {
            "bridge_manifest_sha256": identity["bridge_manifest_sha256"],
            "source_manifest_sha256": identity["source_manifest_sha256"],
            "native_namelist_input_sha256": identity["namelist_sha256"],
        },
        "artifact_receipt": dict(artifacts.receipt),
    }
    receipt_path = root / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    inputs = tree_runner.preflight_prepared_tree(
        prepared_root=root,
        preparation_receipt_sha256=_file_sha256(receipt_path),
        experiment_config=config,
        experiment_config_sha256=_file_sha256(config))
    return inputs, wrapper_reader, boundaries


def test_public_wrapper_extension_passes_production_tree_contracts(
        tmp_path, monkeypatch):
    """Run f001 -> f002 through the public root-preparation tool.

    Only the source bytes and decoded meteorological arrays are fixture data.
    ``prepare.main``, its subprocess benchmark, producer protocol, bridge
    sealer/merge, CPU preprocessing, prepared-cache writer/extension, and all
    integrity readers are the production components shipped to operators.
    The downstream assertions exercise production artifact/preflight/restart
    contracts; actual forecast advancement remains the live controller proof.
    """

    cycle = datetime(2026, 7, 18, 5)
    source = tmp_path / "source"
    source.mkdir()
    names_by_hour = {}
    for hour in (0, 1, 2):
        names_by_hour[hour] = []
        for product in ("wrfnat", "soil"):
            name = f"hrrr.t05z.{product}f{hour:02d}.grib2"
            (source / name).write_bytes(
                f"minimal-fixture:{product}:{hour}".encode("ascii"))
            names_by_hour[hour].append(name)

    source_manifest = source / "SHA256SUMS"

    def publish_source_manifest(hours):
        source_manifest.write_text("".join(
            f"{_file_sha256(source / name)}  {name}\n"
            for hour in hours for name in sorted(names_by_hour[hour])
        ), encoding="utf-8")
        return _file_sha256(source_manifest)

    (target, static_cache, static_receipt,
     domain, namelist_template) = _real_wrapper_inputs(tmp_path)
    f001_namelist = tmp_path / "namelist-f001.input"
    f002_namelist = tmp_path / "namelist-f002.input"
    _materialize_input_namelist(
        namelist_template, f001_namelist, cycle=cycle, lead=1,
        domain_starts=[cycle, cycle])
    _materialize_input_namelist(
        namelist_template, f002_namelist, cycle=cycle, lead=2,
        domain_starts=[cycle, cycle])
    assert _file_sha256(f001_namelist) != _file_sha256(f002_namelist)
    vertical = explicit_vertical_from_wrf_namelist(
        f001_namelist, expected_nz=target.nz,
        context="wrapper/hierarchy composition test")
    prior_root_domain = _experiment(
        vertical, run_seconds=3600.0, start_time=cycle, target=target,
        physics_profile=WSM6_PROFILE_ID,
        history_interval_seconds=3600.0).root
    extended_root_domain = _experiment(
        vertical, run_seconds=7200.0, start_time=cycle, target=target,
        physics_profile=WSM6_PROFILE_ID,
        history_interval_seconds=3600.0).root
    decoder = _fixture_decoder(tmp_path)
    monkeypatch.setenv("GPUWM_HRRR_DECODER", str(decoder))
    try:
        cpu_bridge = resolve_cpu_bridge()
    except FileNotFoundError:
        pytest.skip("native CPU preprocessing bridge is not installed")

    common = [
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--domain-spec", str(domain),
        "--physics-profile", WSM6_PROFILE_ID,
        "--valid-time", cycle.strftime("%Y-%m-%d_%H:%M:%S"),
        "--forecast-start-hour", "0",
        "--history-interval-seconds", "3600",
        "--pipeline-workers", "1",
        "--prepare-workers", "1",
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "1",
        "--cpu-preprocess-bridge", str(cpu_bridge),
        "--sealed-prepared-cache",
        "--skip-stock-wrf-export",
    ]
    prior = tmp_path / "prepared-f001"
    initial_manifest_sha = publish_source_manifest((0, 1))
    assert prepare.main(common + [
        "--source-manifest-sha256", initial_manifest_sha,
        "--namelist-input", str(f001_namelist),
        "--forecast-end-hour", "1", "--run-seconds", "3600",
        "--output-root", str(prior),
    ]) == 0

    output = tmp_path / "prepared-f002"
    extended_manifest_sha = publish_source_manifest((0, 1, 2))
    assert prepare.main(common + [
        "--source-manifest-sha256", extended_manifest_sha,
        "--namelist-input", str(f002_namelist),
        "--forecast-end-hour", "2", "--run-seconds", "7200",
        "--output-root", str(output),
        "--extend-root-preparation", str(prior),
    ]) == 0

    result = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert result["source_forecast_hours"] == [0, 1, 2]
    assert result["prepared_cache_contract"]["operation"] == "extend-one-hour"
    prior_bridge = prior / "native" / "native-bridge"
    extended_bridge = output / "native" / "native-bridge"
    assert (extended_bridge / "atmosphere-f00" / "TT.f32le").samefile(
        prior_bridge / "atmosphere-f00" / "TT.f32le")
    assert (extended_bridge / "atmosphere-f01" / "TT.f32le").samefile(
        prior_bridge / "atmosphere-f01" / "TT.f32le")
    assert (output / "native-static.npz").samefile(
        prior / "native-static.npz")
    prior_header = json.loads((
        prior / "native" / "prepared-cache" / "header.json"
    ).read_text(encoding="utf-8"))
    header = json.loads((
        output / "native" / "prepared-cache" / "header.json"
    ).read_text(encoding="utf-8"))
    assert prior_header["identity"]["namelist_sha256"] == _file_sha256(
        f001_namelist)
    assert header["identity"]["namelist_sha256"] == _file_sha256(
        f002_namelist)
    assert prior_header["identity"]["namelist_extension_invariant"] == \
        header["identity"]["namelist_extension_invariant"]
    assert header["identity"]["forcing_hours"] == [0, 1, 2]
    assert len(header["metadata"]["lbc"]["intervals"]) == 2
    PreparedCacheReader(
        output / "native" / "prepared-cache",
        expected_identity=header["identity"]).verify_all()
    old_name, old_entry = next(iter(prior_header["arrays"].items()))
    new_entry = header["arrays"][old_name]
    assert (output / "native" / "prepared-cache" /
            new_entry["file"]).samefile(
        prior / "native" / "prepared-cache" / old_entry["file"])
    assert header["metadata"]["source_manifest_extension"][
        "suffix_source_forecast_hours"] \
        == [1, 2]
    assert header["metadata"]["bridge_manifest_extension"][
        "suffix_source_forecast_hours"] \
        == [1, 2]
    assert not (output / "native" / "extension-work").exists()

    prior_config = _write_minimal_stream_tree_config(
        tmp_path / "tree-f001.toml", cycle=cycle, run_seconds=3600.0)
    extended_config = _write_minimal_stream_tree_config(
        tmp_path / "tree-f002.toml", cycle=cycle, run_seconds=7200.0)
    prior_tree, prior_tree_reader, prior_boundaries = \
        _build_and_preflight_stream_tree(
            tmp_path / "tree-f001", config=prior_config, wrapper=prior,
            cycle=cycle, forcing_hours=(0, 1),
            root_cache_domain=prior_root_domain)
    extended_tree, extended_tree_reader, extended_boundaries = \
        _build_and_preflight_stream_tree(
            tmp_path / "tree-f002", config=extended_config,
            wrapper=output, cycle=cycle, forcing_hours=(0, 1, 2),
            root_cache_domain=extended_root_domain)
    assert prior_tree.forcing_hours == (0, 1)
    assert extended_tree.forcing_hours == (0, 1, 2)
    assert [bundle.grid_id for bundle in prior_tree.domains] == [1, 2]
    assert [bundle.grid_id for bundle in extended_tree.domains] == [1, 2]
    assert prior_tree.domains[0].cache_identity[
        "source_manifest_sha256"] == prior_header["identity"][
            "source_manifest_sha256"]
    assert extended_tree.domains[0].cache_identity[
        "source_manifest_sha256"] == header["identity"][
            "source_manifest_sha256"]
    hierarchy_lbc_keys = sorted(
        key for key in extended_tree.domains[0].cache_reader.arrays
        if key.startswith("lbc/"))
    wrapper_lbc_keys = sorted(
        key for key in extended_tree_reader.arrays
        if key.startswith("lbc/"))
    assert hierarchy_lbc_keys == wrapper_lbc_keys
    for key in hierarchy_lbc_keys:
        np.testing.assert_array_equal(
            extended_tree.domains[0].cache_reader.read_array(key),
            extended_tree_reader.read_array(key))

    source_model, _model_start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0,
        payload_seed=131)
    source_model.schedule.clock.start_time = cycle
    source_model.root.cfg.run = replace(
        source_model.root.cfg.run,
        spec_bdy_width=prior_boundaries.spec_bdy_width,
        spec_zone=prior_boundaries.spec_zone,
        relax_zone=prior_boundaries.relax_zone)
    source_model.root.state.lateral_boundaries = prior_boundaries
    resumed_model, _model_start = _sealed_tree_fixture(
        monkeypatch, forcing_count=3, run_seconds=7200.0,
        payload_seed=231)
    resumed_model.schedule.clock.start_time = cycle
    resumed_model.root.cfg.run = replace(
        resumed_model.root.cfg.run,
        spec_bdy_width=extended_boundaries.spec_bdy_width,
        spec_zone=extended_boundaries.spec_zone,
        relax_zone=extended_boundaries.relax_zone)
    resumed_model.root.state.lateral_boundaries = extended_boundaries
    checkpoint_root = tmp_path / "checkpoint-generations"
    f001_root = restart_io.write_tree_restart(
        checkpoint_root, source_model, cycle + timedelta(hours=1),
        sealed_forcing_extension=True)
    f001 = restart_io.restore_tree_restart(
        f001_root, resumed_model, sealed_forcing_extension=True)
    assert f001.elapsed_ticks / f001.tick_den == 3600.0
    assert sorted(f001.headers_by_grid_id) == [1, 2]
    f001_ids = {
        row["checkpoint_set_id"]
        for row in f001.headers_by_grid_id.values()}
    assert len(f001_ids) == 1
    assert all(row["elapsed_seconds"] == 3600.0
               for row in f001.headers_by_grid_id.values())

    for node in resumed_model.walk_parent_first():
        node.clock.ticks = 7200
        node.clock.step_count = 7200
        node.state.elapsed_seconds = 7200.0
    f002_root = restart_io.write_tree_restart(
        checkpoint_root, resumed_model, cycle + timedelta(hours=2),
        sealed_forcing_extension=True)
    f002_paths = restart_io._tree_restart_paths(f002_root, {1, 2})
    f002_headers = {
        grid_id: restart_io.read_restart_header(path)
        for grid_id, path in f002_paths.items()}
    f002_ids = {
        row["checkpoint_set_id"] for row in f002_headers.values()}
    assert len(f002_ids) == 1
    assert f002_ids != f001_ids
    assert sorted(f002_headers) == [1, 2]
    assert all(row["domain_ids"] == [1, 2]
               for row in f002_headers.values())
    assert all(row["elapsed_seconds"] == 7200.0
               for row in f002_headers.values())


def _legacy_public_wrapper_extends_one_hour_without_rebuilding_prefix(
        tmp_path, monkeypatch):
    cycle = datetime(2026, 7, 18, 5)
    source = tmp_path / "source"
    source.mkdir()
    names_by_hour = {}
    for hour in (0, 1, 2):
        names = []
        for product in ("wrfnat", "soil"):
            name = f"hrrr.t05z.{product}f{hour:02d}.grib2"
            (source / name).write_bytes(f"{product}:{hour}".encode("ascii"))
            names.append(name)
        names_by_hour[hour] = names
    prior_sums_bytes = "".join(
        f"{_file_sha256(source / name)}  {name}\n"
        for hour in (0, 1) for name in sorted(names_by_hour[hour]))
    current_sums = source / "SHA256SUMS"
    current_sums.write_text(
        "".join(
            f"{_file_sha256(source / name)}  {name}\n"
            for hour in (0, 1, 2) for name in sorted(names_by_hour[hour])),
        encoding="utf-8")
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&time_control\n/\n", encoding="utf-8")
    domain = tmp_path / "domain.json"
    domain.write_text("{}\n", encoding="utf-8")
    decoder = tmp_path / "decoder"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)
    tree_config = _write_two_domain_config(tmp_path)
    tree_config.write_text(
        tree_config.read_text(encoding="utf-8").replace(
            "2026-07-23T00:00:00", cycle.isoformat()),
        encoding="utf-8")
    initial_tree = load_experiment(tree_config)
    final_tree_config = tmp_path / "tree-f002.toml"
    final_tree_config.write_text(
        tree_config.read_text(encoding="utf-8").replace(
            "run_seconds = 3600.0", "run_seconds = 7200.0"),
        encoding="utf-8")

    prior = tmp_path / "prior"
    native = prior / "native"
    native.mkdir(parents=True)
    (native / "source-manifest.snapshot").write_text(
        prior_sums_bytes, encoding="utf-8")
    static = prior / "native-static.npz"
    with static.open("wb") as stream:
        np.savez(stream, DUMMY=np.ones((1,), dtype=np.float32))
    for name in ("native-static-receipt.json",
                 "native-geometry-receipt.json"):
        (prior / name).write_text("{}\n", encoding="utf-8")
    prior_bridge = _fake_sealed_bridge(native / "native-bridge", [0, 1])
    prior_identity = _extension_identity(
        source_hours=[0, 1], model_start=cycle, domain_start=cycle,
        bridge=_file_sha256(prior_bridge / "SHA256SUMS"),
        source_manifest=_file_sha256(
            native / "source-manifest.snapshot"))
    prior_identity.update({
        "static_cache_sha256": _file_sha256(static),
        "namelist_sha256": _file_sha256(namelist),
        "domain_config": prepared_domain_config_identity(
            initial_tree.root),
    })
    prior_identity["source_identity"]["source_cycle"] = cycle.isoformat()
    prior_identity["source_identity"]["grid_id"] = 1
    initial, met, boundaries = _prepared_fixture()
    prior_cache = native / "prepared-cache"
    prior_cache_receipt = write_prepared_cache(
        prior_cache, identity=prior_identity, initial_result=initial,
        met=met, boundaries=boundaries,
        metadata={
            "initial_valid_time": cycle.isoformat(),
            "last_valid_time": (cycle + timedelta(hours=1)).isoformat(),
            "source_cycle": cycle.isoformat(),
            "source_forecast_hours": [0, 1],
            "model_forcing_hours": [0, 1],
            "forcing_hours": [0, 1],
        }, sealed_forcing_extension=True)
    prior_report = native / "preparation-report" / "report.json"
    prior_report.parent.mkdir()
    prior_report.write_text(json.dumps({
        "status": "PASS",
        "source_cycle": cycle.isoformat(),
        "model_start_time": cycle.isoformat(),
        "source_forecast_hours": [0, 1],
        "model_forcing_hours": [0, 1],
        "source_identity": prior_identity["source_identity"],
        "prepared_cache": prior_cache_receipt,
        "target_domain_sha256": "target",
    }), encoding="utf-8")
    (prior / "public-wrapper-result.json").write_text(json.dumps({
        "status": "PASS",
        "source_cycle": cycle.isoformat(),
        "source_forecast_hours": [0, 1],
        "history_interval_seconds": 3600.0,
        "physics": {"profile": WSM6_PROFILE_ID},
        "prepared_cache_contract": {
            "mode": "sealed-prefix-v1", "operation": "initial"},
    }), encoding="utf-8")

    def fake_run(command, _env, cwd=None):
        del cwd
        if any(str(value).endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            suffix_bridge = Path(command[command.index("--bridge") + 1])
            _fake_sealed_bridge(suffix_bridge, [1, 2])
            suffix_identity = _extension_identity(
                source_hours=[1, 2],
                model_start=cycle + timedelta(hours=1),
                domain_start=cycle + timedelta(hours=1),
                bridge=_file_sha256(suffix_bridge / "SHA256SUMS"),
                source_manifest=_file_sha256(current_sums))
            suffix_identity.update({
                "static_cache_sha256": _file_sha256(static),
                "namelist_sha256": _file_sha256(namelist),
            })
            suffix_domain = copy.deepcopy(prior_identity["domain_config"])
            suffix_domain["start_time"] = (
                cycle + timedelta(hours=1)).isoformat()
            suffix_domain["run"]["run_seconds"] = 3600.0
            suffix_identity["domain_config"] = suffix_domain
            suffix_identity["source_identity"][
                "source_cycle"] = cycle.isoformat()
            suffix_identity["source_identity"]["grid_id"] = 1
            suffix_initial, suffix_met, suffix_boundaries = \
                _prepared_suffix_fixture()
            write_prepared_cache(
                Path(command[command.index("--prepared-cache") + 1]),
                identity=suffix_identity, initial_result=suffix_initial,
                met=suffix_met, boundaries=suffix_boundaries)
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True)
            (outdir / "report.json").write_text(json.dumps({
                "status": "PASS",
                "source_forecast_hours": [1, 2],
                "model_forcing_hours": [0, 1],
                "source_identity": suffix_identity["source_identity"],
                "physics": {"profile": WSM6_PROFILE_ID},
                "pipeline": {}, "preparation": {},
            }), encoding="utf-8")
        elif "gpuwm.wrf_direct" in command:
            Path(command[command.index("--output") + 1]).mkdir()

    monkeypatch.setattr(prepare, "_run", fake_run)
    monkeypatch.setattr(
        prepare, "_validated_worker_receipts",
        lambda *args, **kwargs: ({"backend": "fixture"}, {}, {}))
    monkeypatch.setattr(
        prepare, "_validated_physics_receipt",
        lambda *args, **kwargs: {"profile": WSM6_PROFILE_ID})
    output = tmp_path / "extended"
    argv = [
        "--source-root", str(source),
        "--source-manifest", str(current_sums),
        "--source-manifest-sha256", _file_sha256(current_sums),
        "--static-cache", str(static),
        "--static-receipt", str(prior / "native-static-receipt.json"),
        "--domain-spec", str(domain),
        "--namelist-input", str(namelist),
        "--physics-profile", WSM6_PROFILE_ID,
        "--valid-time", "2026-07-18_05:00:00",
        "--forecast-start-hour", "0", "--forecast-end-hour", "2",
        "--run-seconds", "7200",
        "--history-interval-seconds", "3600",
        "--output-root", str(output),
        "--sealed-prepared-cache",
        "--extend-root-preparation", str(prior),
    ]

    assert prepare.main(argv) == 0

    result = json.loads(
        (output / "public-wrapper-result.json").read_text())
    assert result["source_forecast_hours"] == [0, 1, 2]
    assert result["prepared_cache_contract"]["operation"] == \
        "extend-one-hour"
    header = json.loads(
        (output / "native" / "prepared-cache" / "header.json").read_text())
    assert header["identity"]["forcing_hours"] == [0, 1, 2]
    assert len(header["metadata"]["lbc"]["intervals"]) == 2
    PreparedCacheReader(
        output / "native" / "prepared-cache",
        expected_identity=header["identity"]).verify_all()
    report = json.loads((
        output / "native" / "preparation-report" / "report.json"
    ).read_text())
    report_input = report["input"]
    assert report_input["bridge"] == str(
        (output / "native" / "native-bridge").resolve())
    assert report_input["bridge_manifest_sha256"] == \
        _file_sha256(output / "native" / "native-bridge" / "SHA256SUMS")
    assert report_input["source_manifest_sha256"] == \
        _file_sha256(current_sums)
    assert report_input["source_forecast_hours"] == [0, 1, 2]
    assert report_input["model_forcing_hours"] == [0, 1, 2]
    assert report_input["forcing_hours"] == [0, 1, 2]
    assert report_input["native_physics_profile"] == WSM6_PROFILE_ID
    assert isinstance(report_input["native_physics_profile"], str)
    assert (output / "native" / "native-bridge" /
            "atmosphere-f00" / "field-00.f32le").samefile(
                prior_bridge / "atmosphere-f00" / "field-00.f32le")
    assert not (output / "native" / "extension-work").exists()

    # Consume the wrapper's actual extended cache through the hierarchy
    # runner's real CPU preflight.  Only grid/static loaders are replaced:
    # both PreparedCacheReaders, their payload hashes, domain identities,
    # forcing ownership, and every hierarchy receipt remain production code.
    final_tree = load_experiment(final_tree_config)
    prepared_tree = tmp_path / "prepared-tree-f002"
    artifacts = prepared_tree / "hierarchy-artifacts"
    domains = artifacts / "domains"
    domains.mkdir(parents=True)
    domain_receipts = []
    common_static = output / "native-static.npz"
    for domain_config in final_tree.domains:
        grid_id = int(domain_config.grid_id)
        bundle = domains / f"d{grid_id:02d}"
        bundle.mkdir()
        cache = bundle / "prepared-cache"
        if grid_id == 1:
            shutil.copytree(output / "native" / "prepared-cache", cache)
        else:
            child_initial, child_met, _child_boundaries = \
                _prepared_fixture()
            child_identity = copy.deepcopy(header["identity"])
            child_identity["domain_config"] = \
                prepared_domain_config_identity(domain_config)
            child_identity["source_identity"]["grid_id"] = grid_id
            write_prepared_cache(
                cache, identity=child_identity,
                initial_result=child_initial, met=child_met,
                boundaries=None)
        static_path = bundle / "native-static.npz"
        shutil.copy2(common_static, static_path)
        geometry = {"geometry": {"grid_id": grid_id}}
        geometry_path = bundle / "geometry-receipt.json"
        geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
        cache_header = json.loads((cache / "header.json").read_text())
        cache_reader = PreparedCacheReader(
            cache, expected_identity=cache_header["identity"])
        verification = cache_reader.verify_all()
        with np.load(static_path, allow_pickle=False) as archive:
            static_fields = sorted(archive.files)
        domain_receipt = {
            "schema": "gpuwm-native-domain-artifact-build-v1",
            "status": "READY",
            "grid_id": grid_id,
            "parent_id": int(domain_config.parent_id),
            "boundary_mode": (
                "external-specified" if grid_id == 1
                else "nested-parent-forced"),
            "artifacts": {
                "prepared_cache": {
                    "path": "prepared-cache",
                    "content_sha256": cache_reader.content_sha256,
                    "payload_bytes": cache_reader.payload_bytes,
                    "array_count": len(cache_reader.arrays),
                },
                "static_cache": {
                    "path": "native-static.npz",
                    "bytes": static_path.stat().st_size,
                    "sha256": _file_sha256(static_path),
                    "fields": static_fields,
                },
                "geometry_receipt": {
                    "path": "geometry-receipt.json",
                    "sha256": _file_sha256(geometry_path),
                    "geometry": geometry["geometry"],
                },
            },
            "verification": verification,
        }
        (bundle / "receipt.json").write_text(
            json.dumps(domain_receipt), encoding="utf-8")
        domain_receipts.append(domain_receipt)

    grid_ids = [int(domain.grid_id) for domain in final_tree.domains]
    manifest = {
        "schema": tree_runner.ARTIFACT_MANIFEST_SCHEMA,
        "domains": [
            {
                "grid_id": grid_id,
                "prepared_cache":
                    f"domains/d{grid_id:02d}/prepared-cache",
                "static_cache":
                    f"domains/d{grid_id:02d}/native-static.npz",
                "geometry_receipt":
                    f"domains/d{grid_id:02d}/geometry-receipt.json",
            }
            for grid_id in grid_ids
        ],
    }
    manifest_path = artifacts / "domain-artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_receipt = {
        "schema": tree_runner.ARTIFACT_RECEIPT_SCHEMA,
        "status": "READY",
        "domain_count": len(grid_ids),
        "grid_ids": grid_ids,
        "manifest": {
            "path": manifest_path.name,
            "sha256": _file_sha256(manifest_path),
        },
        "boundary_inventory": {
            "external": [grid_ids[0]],
            "nested_parent_forced": grid_ids[1:],
        },
        "domains": domain_receipts,
    }
    (artifacts / "receipt.json").write_text(
        json.dumps(artifact_receipt), encoding="utf-8")
    preparation_receipt = {
        "schema": tree_runner.HIERARCHY_SCHEMA,
        "status": "PASS",
        "valid_time": cycle.isoformat(),
        "domain_count": len(grid_ids),
        "forcing_hours": [0, 1, 2],
        "provenance": {
            "bridge_manifest_sha256": header["identity"][
                "bridge_manifest_sha256"],
            "source_manifest_sha256": header["identity"][
                "source_manifest_sha256"],
            "native_namelist_input_sha256": header["identity"][
                "namelist_sha256"],
        },
        "artifact_receipt": artifact_receipt,
    }
    receipt_path = prepared_tree / "receipt.json"
    receipt_path.write_text(
        json.dumps(preparation_receipt), encoding="utf-8")
    monkeypatch.setattr(
        tree_runner, "grids_from_projection_config",
        lambda _exp: tuple(object() for _ in grid_ids))
    monkeypatch.setattr(
        tree_runner, "verify_native_static_receipt",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tree_runner, "load_native_static_cache",
        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        tree_runner, "_validate_vertical",
        lambda *_args, **_kwargs: None)
    tree_inputs = tree_runner.preflight_prepared_tree(
        prepared_root=prepared_tree,
        preparation_receipt_sha256=_file_sha256(receipt_path),
        experiment_config=final_tree_config,
        experiment_config_sha256=_file_sha256(final_tree_config))
    assert tree_inputs.forcing_hours == (0, 1, 2)
    assert tree_inputs.domains[0].cache_reader.content_sha256 == \
        header["content_sha256"]

    # The actual restart-extension validator accepts the f001 checkpoint
    # against the wrapper's f002 forcing cache.  This is the CPU seam the
    # hierarchy forecast runner exercises before applying checkpoint bytes.
    prior_cache_header = json.loads((
        prior / "native" / "prepared-cache" / "header.json").read_text())
    prior_reader = PreparedCacheReader(
        prior / "native" / "prepared-cache",
        expected_identity=prior_cache_header["identity"])
    extended_reader = PreparedCacheReader(
        output / "native" / "prepared-cache",
        expected_identity=header["identity"])
    prior_boundaries = _reader_boundaries(prior_reader)
    extended_boundaries = _reader_boundaries(extended_reader)
    restart_cfg = _restart_cfg(
        moist=True, mp_physics=1, specified=True,
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    checkpoint_state = _restart_shim_state(restart_cfg, monkeypatch)
    _restart_fill_setup(checkpoint_state)
    checkpoint_state.lateral_boundaries = prior_boundaries
    checkpoint_state.elapsed_seconds = 3600.0
    checkpoint = restart_io.write_restart(
        tmp_path / "f001.npz", checkpoint_state, restart_cfg,
        sealed_forcing_extension=True)
    resumed_state = _restart_shim_state(restart_cfg, monkeypatch)
    _restart_fill_setup(resumed_state)
    resumed_state.lateral_boundaries = extended_boundaries
    resumed_state._lateral_boundary_device = SimpleNamespace(
        rolling=False, clock=None)
    validated = restart_io._validate_restart(
        checkpoint, resumed_state, restart_cfg,
        sealed_forcing_extension=True)
    assert validated.elapsed == 3600.0
    assert len(extended_boundaries.intervals) == 2


@pytest.mark.parametrize(
    "physics_profile",
    (
        KESSLER_PROFILE_ID,
        WSM6_PROFILE_ID,
        THOMPSON_PROFILE_ID,
        MORRISON_PROFILE_ID,
        NSSL2_PROFILE_ID,
        NSSL2_LEGACY_RRTMG_PROFILE_ID,
    ),
)
def test_public_wrapper_preserves_absolute_source_leads_and_rebases_model_time(
        tmp_path: Path, monkeypatch, physics_profile: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for hour in (12, 13):
        (source / f"hrrr.t05z.wrfnatf{hour:02d}.grib2").write_bytes(
            f"atmos-{hour}".encode("ascii"))
        (source / f"hrrr.t05z.soilf{hour:02d}.grib2").write_bytes(
            f"soil-{hour}".encode("ascii"))
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    for path in (static_cache, static_receipt, namelist):
        path.write_bytes(b"fixture")
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)

    commands: list[list[str]] = []

    def fake_run(command: list[str], _env: dict[str, str]) -> None:
        commands.append(command)
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True)
            (outdir / "report.json").write_text(json.dumps({
                "status": "PASS",
                "history_interval_seconds": 3600.0,
                "physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": physics_profile,
                    "hrrr_initialization": _cold_start_receipt(
                        physics_profile),
                },
                "preparation": {
                    "preprocess_backend": {"backend": "cuda"},
                    "preprocess_worker_budget": {
                        "schema": "gpuwm-preprocess-worker-budget-v1",
                        "backend": "cuda",
                        "applicable": False,
                        "pipeline_decoder_workers_included": False,
                        "peak_active_native_workers": 0,
                    },
                },
                "pipeline": {"workers": {"requested": "8", "selected": 8}},
            }), encoding="utf-8")
        elif "gpuwm.wrf_direct" in command:
            # Behave like the real atomic exporter on success: the
            # fail-closed step verifies the declared outputs at PASS.
            export_output = Path(command[command.index("--output") + 1])
            export_output.mkdir(parents=True)
            for name in ("wrfinput_d01", "wrfbdy_d01"):
                (export_output / name).write_bytes(b"exported")
            (export_output / "manifest.json").write_text(json.dumps({
                "files": {"wrfinput_d01": {"sha256": "0" * 64},
                          "wrfbdy_d01": {"sha256": "0" * 64}},
            }), encoding="utf-8")

    monkeypatch.setattr(prepare, "_run", fake_run)
    output = tmp_path / "output"
    assert prepare.main([
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--physics-profile", physics_profile,
        "--valid-time", "2026-07-18_05:00:00",
        "--forecast-start-hour", "12",
        "--forecast-end-hour", "13",
        "--run-seconds", "3600",
        "--history-interval-seconds", "3600",
        "--output-root", str(output),
    ]) == 0

    series = output / "native" / "hrrr-f12-f13-series.tsv"
    assert [line.split("\t", 1)[0]
            for line in series.read_text(encoding="utf-8").splitlines()] \
        == ["12", "13"]
    benchmark = next(
        command for command in commands
        if any(value.endswith("hrrr_single_domain_benchmark.py")
               for value in command))
    assert benchmark[benchmark.index("--forecast-start-hour") + 1] == "12"
    assert benchmark[benchmark.index("--forecast-end-hour") + 1] == "13"
    assert benchmark[benchmark.index("--physics-profile") + 1] \
        == physics_profile
    assert benchmark[benchmark.index("--history-interval-seconds") + 1] \
        == "3600.0"
    assert "--io-mode" not in benchmark
    export = next(command for command in commands if "gpuwm.wrf_direct" in command)
    assert export[export.index("--valid-time") + 1] == "2026-07-18_17:00:00"

    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["source_cycle"] == "2026-07-18T05:00:00"
    assert receipt["model_start_time"] == "2026-07-18T17:00:00"
    assert receipt["source_forecast_hours"] == [12, 13]
    assert receipt["model_forcing_hours"] == [0, 1]
    assert receipt["forcing_hours"] == [0, 1]
    assert receipt["history_interval_seconds"] == 3600.0
    assert receipt["physics"]["profile"] == physics_profile


def test_public_wrapper_rejects_mismatched_or_incomplete_physics_receipt():
    complete = {
        "physics": {
            "schema": "gpuwm-prepared-physics-profile-v1",
            "profile": MORRISON_PROFILE_ID,
            "hrrr_initialization": _cold_start_receipt(
                MORRISON_PROFILE_ID),
        },
    }
    assert prepare._validated_physics_receipt(
        complete, requested_profile=MORRISON_PROFILE_ID) \
        == complete["physics"]

    with pytest.raises(RuntimeError, match="differs from the request"):
        prepare._validated_physics_receipt(
            complete, requested_profile=NSSL2_PROFILE_ID)
    incomplete = json.loads(json.dumps(complete))
    del incomplete["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            incomplete, requested_profile=MORRISON_PROFILE_ID)

    wrong_fields = json.loads(json.dumps(complete))
    wrong_fields["physics"]["hrrr_initialization"][
        "source_absent_wrf_fields"] = []
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_fields, requested_profile=MORRISON_PROFILE_ID)

    false_exact = json.loads(json.dumps(complete))
    false_exact["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["all_exact_expected"] = False
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            false_exact, requested_profile=MORRISON_PROFILE_ID)

    wrong_bits = json.loads(json.dumps(complete))
    wrong_bits["physics"]["hrrr_initialization"][
        "state_source_absent_fields"]["ng"]["expected_uint32_bits"] = 1
    with pytest.raises(RuntimeError, match="cold-start evidence"):
        prepare._validated_physics_receipt(
            wrong_bits, requested_profile=MORRISON_PROFILE_ID)


# ---------------------------------------------------------------------------
# The decoder this wrapper launches, and the disposition of the optional
# stock-WRF export.  Both are field failures: the first refused on a healthy
# wheel install, and the second threw away a completed preparation.
# ---------------------------------------------------------------------------

def _wrapper_case(tmp_path: Path, monkeypatch, *, export_returncode: int = 0):
    """One runnable wrapper invocation, with every subprocess faked.

    Returns ``(argv, commands, output)``: the argument vector, the list
    every faked ``_run`` appends to, and the output root -- so a test can
    assert what was launched and what was written.
    """

    import subprocess as subprocess_module

    source = tmp_path / "source"
    source.mkdir()
    for hour in (0, 1):
        (source / f"hrrr.t05z.wrfnatf{hour:02d}.grib2").write_bytes(b"atmos")
        (source / f"hrrr.t05z.soilf{hour:02d}.grib2").write_bytes(b"soil")
    source_manifest = source / "SHA256SUMS"
    source_manifest.write_text("fixture\n", encoding="ascii")
    static_cache = tmp_path / "static.npz"
    static_receipt = tmp_path / "static.json"
    namelist = tmp_path / "namelist.input"
    for path in (static_cache, static_receipt, namelist):
        path.write_bytes(b"fixture")
    decoder = tmp_path / "hrrr_grib2_bridge"
    decoder.write_bytes(b"decoder")
    monkeypatch.setattr(prepare, "_decoder", lambda _env: decoder)

    commands: list[list[str]] = []

    def fake_run(command, _env, cwd=None):
        commands.append(command)
        if any(str(value).endswith("hrrr_single_domain_benchmark.py")
               for value in command):
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True)
            (outdir / "report.json").write_text(json.dumps({
                "status": "PASS",
                "history_interval_seconds": 3600.0,
                "physics": {
                    "schema": "gpuwm-prepared-physics-profile-v1",
                    "profile": WSM6_PROFILE_ID,
                    "hrrr_initialization": _cold_start_receipt(
                        WSM6_PROFILE_ID),
                },
                "preparation": {
                    "preprocess_backend": {"backend": "cuda"},
                    "preprocess_worker_budget": {
                        "schema": "gpuwm-preprocess-worker-budget-v1",
                        "backend": "cuda",
                        "applicable": False,
                        "pipeline_decoder_workers_included": False,
                        "peak_active_native_workers": 0,
                    },
                },
                "pipeline": {"workers": {"requested": "8", "selected": 8}},
            }), encoding="utf-8")
        elif "gpuwm.wrf_direct" in command and export_returncode:
            raise subprocess_module.CalledProcessError(
                export_returncode, command)
        elif "gpuwm.wrf_direct" in command:
            # The fake converter behaves like the real atomic exporter:
            # exit 0 means the declared outputs and their manifest exist
            # (the fail-closed step verifies exactly that at PASS).
            export_output = Path(command[command.index("--output") + 1])
            export_output.mkdir(parents=True)
            for name in ("wrfinput_d01", "wrfbdy_d01"):
                (export_output / name).write_bytes(b"exported")
            (export_output / "manifest.json").write_text(json.dumps({
                "files": {"wrfinput_d01": {"sha256": "0" * 64},
                          "wrfbdy_d01": {"sha256": "0" * 64}},
            }), encoding="utf-8")

    monkeypatch.setattr(prepare, "_run", fake_run)
    output = tmp_path / "output"
    argv = [
        "--source-root", str(source),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", "0" * 64,
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist),
        "--physics-profile", WSM6_PROFILE_ID,
        "--valid-time", "2026-07-18_05:00:00",
        "--run-seconds", "3600",
        "--history-interval-seconds", "3600",
        "--output-root", str(output),
    ]
    return argv, commands, output


def test_a_refused_stock_wrf_export_fails_the_command_keeping_the_work(
        tmp_path: Path, monkeypatch) -> None:
    """The 2026-08-04 ruling: a requested export that fails is fatal.

    History in two swings: the original ``check=True`` destroyed a
    completed preparation on any converter refusal; the warn-and-continue
    fix for that then PASS-logged the battery case run whose converter
    died on EMFILE -- no WRF-arm inputs, no manifest, one warning line
    (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).  Now the command
    fails LOUDLY, and the preparation's own artifacts stay on disk --
    the refusal costs a re-run of the export, never the preparation.
    """

    argv, commands, output = _wrapper_case(
        tmp_path, monkeypatch, export_returncode=1)
    with pytest.raises(RuntimeError, match="requested and failed"):
        prepare.main(argv)

    assert any("gpuwm.wrf_direct" in command for command in commands)
    # The preparation completed before the export and stands on disk.
    report = (output / "native" / "preparation-report" / "report.json")
    assert report.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "PASS"
    # No wrapper PASS receipt is written over the failure.
    assert not (output / "public-wrapper-result.json").exists()


def test_a_required_stock_wrf_export_still_fails_the_command(
        tmp_path: Path, monkeypatch) -> None:
    """The legacy flag is honored: failure is fatal with it too."""

    argv, _commands, _output = _wrapper_case(
        tmp_path, monkeypatch, export_returncode=1)
    with pytest.raises(RuntimeError, match="requested and failed"):
        prepare.main(argv + ["--require-stock-wrf-export"])


def test_skipping_the_stock_wrf_export_never_launches_it(
        tmp_path: Path, monkeypatch) -> None:
    argv, commands, output = _wrapper_case(tmp_path, monkeypatch)
    assert prepare.main(argv + ["--skip-stock-wrf-export"]) == 0
    assert not any("gpuwm.wrf_direct" in command for command in commands)
    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["stock_wrf_export"]["status"] == "SKIPPED"
    assert receipt["wrf_input"] is None


def test_a_successful_stock_wrf_export_still_names_its_output(
        tmp_path: Path, monkeypatch) -> None:
    argv, commands, output = _wrapper_case(tmp_path, monkeypatch)
    assert prepare.main(argv) == 0
    assert any("gpuwm.wrf_direct" in command for command in commands)
    receipt = json.loads(
        (output / "public-wrapper-result.json").read_text(encoding="utf-8"))
    assert receipt["stock_wrf_export"]["status"] == "PASS"
    assert receipt["wrf_input"] == str(output / "wrf-native-input")
    # PASS is earned: the receipt binds the verified outputs by name and
    # by the manifest's own digest.
    assert receipt["stock_wrf_export"]["files"] == [
        "wrfbdy_d01", "wrfinput_d01"]
    assert len(receipt["stock_wrf_export"]["manifest_sha256"]) == 64


def test_the_two_export_flags_are_mutually_exclusive(tmp_path: Path,
                                                     monkeypatch) -> None:
    argv, _commands, _output = _wrapper_case(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        prepare.main(argv + ["--skip-stock-wrf-export",
                             "--require-stock-wrf-export"])


def test_the_wrapper_resolves_the_decoder_the_shared_ladder_resolves(
        monkeypatch, tmp_path: Path) -> None:
    """Issue two, at the wrapper: a wheel has no cargo workspace.

    ``REPO`` is site-packages on a pip install, so the old
    ``REPO/tools/grib1_bridge/target/release/hrrr_grib2_bridge`` named a
    path that cannot exist there, and the cargo fallback ran in a
    directory that does not exist either.  Now it is the same ladder
    ``gpuwm doctor`` reports, and the staged bridge wins.
    """

    from gpuwm import bridges

    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    staged = tmp_path / "userdir"
    staged.mkdir()
    for variable in bridges.BRIDGE_ENV.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(bridges, "crate_dir", lambda: tmp_path / "no-crate")
    monkeypatch.setattr(bridges, "_package_parent", lambda: site_packages)
    monkeypatch.setattr(bridges, "default_bridge_dir", lambda: staged)
    monkeypatch.setattr(prepare, "BRIDGE_CRATE", tmp_path / "no-crate")

    # Negative control, watched firing: nothing staged, named refusal --
    # and it names ~/.gpuwm/bridges rather than a cargo target directory.
    with pytest.raises(FileNotFoundError) as error:
        prepare._decoder({})
    assert str(staged) in str(error.value)

    decoder = staged / bridges.executable_name("hrrr_grib2_bridge")
    decoder.write_bytes(b"decoder")
    assert prepare._decoder({}) == decoder.resolve()


def test_an_explicit_decoder_override_is_honored_and_fails_loud(
        monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "my_bridge"
    override.write_bytes(b"decoder")
    monkeypatch.setenv("GPUWM_HRRR_DECODER", str(override))
    assert prepare._decoder({}) == override.resolve()

    monkeypatch.setenv("GPUWM_HRRR_DECODER", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="GPUWM_HRRR_DECODER"):
        prepare._decoder({})


def test_requested_stock_wrf_export_failure_is_fatal(tmp_path):
    """The fail-closed contract, driven by an injected converter death.

    The 2026-08-04 battery case run's converter subprocess died on EMFILE
    and the old warn-and-continue posture PASS-logged a preparation with
    no WRF-arm inputs and no manifest.  A requested export that fails now
    refuses loudly, whatever the converter's exit reason -- the subprocess
    boundary cannot tell an environmental death from a typed refusal.
    """

    import sys

    with pytest.raises(RuntimeError, match="requested and failed"):
        prepare._stock_wrf_export(
            [sys.executable, "-c", "import sys; sys.exit(24)"],
            dict(os.environ), skip=False, required=False,
            output=tmp_path / "wrf-native-input")

    # --skip-stock-wrf-export remains the deliberate opt-out, recorded.
    receipt = prepare._stock_wrf_export(
        [sys.executable, "-c", "import sys; sys.exit(24)"],
        dict(os.environ), skip=True, required=False,
        output=tmp_path / "wrf-native-input")
    assert receipt == {"status": "SKIPPED",
                       "reason": "--skip-stock-wrf-export",
                       "required": False}


def test_stock_wrf_export_pass_is_earned_by_manifested_outputs(tmp_path):
    """PASS requires the declared outputs to exist and hash-manifest."""

    import sys

    output = tmp_path / "wrf-native-input"
    quiet = [sys.executable, "-c", "pass"]

    # Exit 0 with no manifest at all: refused by name.
    with pytest.raises(RuntimeError, match="left no manifest"):
        prepare._stock_wrf_export(quiet, dict(os.environ), skip=False,
                                  required=False, output=output)

    # A manifest whose declared file is missing on disk: refused by name.
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({
        "files": {"wrfinput_d01": {"sha256": "0" * 64},
                  "wrfbdy_d01": {"sha256": "0" * 64}},
    }), encoding="utf-8")
    (output / "wrfinput_d01").write_bytes(b"present")
    with pytest.raises(RuntimeError, match="missing on disk.*wrfbdy_d01"):
        prepare._stock_wrf_export(quiet, dict(os.environ), skip=False,
                                  required=False, output=output)

    # Every declared file present: PASS, with the manifest's own digest.
    (output / "wrfbdy_d01").write_bytes(b"present")
    receipt = prepare._stock_wrf_export(quiet, dict(os.environ), skip=False,
                                        required=False, output=output)
    assert receipt["status"] == "PASS"
    assert receipt["files"] == ["wrfbdy_d01", "wrfinput_d01"]
    assert receipt["manifest_sha256"] == hashlib.sha256(
        (output / "manifest.json").read_bytes()).hexdigest()
    assert "REFUSED" not in prepare.STOCK_WRF_EXPORT_STATES

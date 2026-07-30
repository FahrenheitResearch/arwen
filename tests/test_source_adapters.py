from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from gpuwm.source_adapters import (
    AdapterStatus,
    get_source_adapter,
    source_adapters,
    source_capability_manifest,
)
from gpuwm.source_cli import EXIT_CONFIG, EXIT_USAGE, _parser, main
from gpuwm.source_frame import (
    FieldDescriptor,
    GridDescriptor,
    POLICY_CONTROLLED_FIELDS,
    REQUIRED_3D_FIELDS,
    REQUIRED_SURFACE_FIELDS,
    SourceFrameHeader,
    TimeDescriptor,
    VerticalDescriptor,
    validate_source_frame,
)


EXPECTED_SOURCE_IDS = (
    "hrrr",
    "hrrr-ak",
    "gfs",
    "gdas",
    "gefs",
    "aigfs",
    "aigefs",
    "hgefs",
    "ecmwf-open-data",
    "aifs",
    "rap",
    "nam",
    "hiresw",
    "href",
    "sref",
    "rtma",
    "urma",
    "nbm",
    "rrfs-a",
    "rrfs-public",
    "refs",
    "rrfs-firewx",
    "wrf",
    "era5",
    "20crv3",
    "20crv3-cf",
    "mapped",
)
ROOT = Path(__file__).parents[1]


def test_source_cli_import_is_cpu_only():
    script = """
from importlib.abc import MetaPathFinder
import sys

class RejectCupy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise AssertionError(f"unexpected GPU import: {fullname}")
        return None

sys.meta_path.insert(0, RejectCupy())
import gpuwm.source_cli
assert "cupy" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_forecast_start_hour_is_absent_globally_and_defaults_only_in_hrrr():
    parsed = _parser().parse_args([])
    assert parsed.forecast_start_hour is None
    assert parsed.history_interval_seconds is None


def test_registry_covers_bound_inventory_and_external_source_routes():
    adapters = source_adapters()
    assert tuple(adapter.source_id for adapter in adapters) == EXPECTED_SOURCE_IDS
    assert len({adapter.source_id for adapter in adapters}) == 27
    assert [adapter.source_id for adapter in adapters if adapter.runnable] == [
        "hrrr",
        "gfs",
        "era5",
        "20crv3",
        "mapped",
    ]
    hrrr = get_source_adapter("HRRR")
    assert hrrr.status is AdapterStatus.CERTIFIED
    assert hrrr.stock_wrf_gate.startswith("wrf-v4.6.1-pass")
    assert get_source_adapter("rrfs_a").source_id == "rrfs-a"
    assert get_source_adapter("ifs").source_id == "ecmwf-open-data"
    era5 = get_source_adapter("era5")
    assert era5.upstream_model_id is None
    assert era5.status is AdapterStatus.CERTIFIED
    assert era5.runner == "era5_combined_grib1_v1"
    assert era5.stock_wrf_gate.startswith("wrf-v4.6.1-pass")
    gfs = get_source_adapter("gfs-0.25")
    assert gfs.status is AdapterStatus.CERTIFIED
    assert gfs.runner == "gfs_pgrb2_0p25_v1"
    assert gfs.stock_wrf_gate.startswith("wrf-v4.6.1-pass")
    twentycr = get_source_adapter("20cr")
    assert twentycr.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert twentycr.runner == "twentycrv3_member_grib2_v1"
    assert "max_dom=4" in twentycr.notes
    twentycr_cf = get_source_adapter("20cr-netcdf")
    assert twentycr_cf.source_id == "20crv3-cf"
    assert twentycr_cf.status is AdapterStatus.MEMBER_SELECTION_REQUIRED
    assert twentycr_cf.runnable is False
    assert twentycr_cf.runner is None
    mapped = get_source_adapter("mapping-v1")
    assert mapped.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert mapped.runnable is True
    assert mapped.runner == "mapped_composition_v1"
    assert mapped.stock_wrf_gate == "exact-mapping-composition-hash-evidence-required"


def test_manifest_binds_inventory_and_does_not_confuse_decode_with_readiness():
    manifest = source_capability_manifest()
    assert manifest["schema"] == "gpuwm-native-source-adapters-v1"
    assert manifest["runtime_forbidden"] == ["WPS", "real.exe"]
    assert manifest["rusty_weather_inventory"]["model_id_count"] == 23
    assert len(manifest["rusty_weather_inventory"]["head"]) == 40
    assert manifest["source_count"] == 27
    assert manifest["runnable_source_count"] == 5
    assert set(manifest["packaged_source_authorities"][
        "20crv3_member_grib2"
    ]) == {"mapping", "composition", "provenance"}
    assert all("stock_wrf_gate" in value for value in manifest["sources"])
    assert len(manifest["mapped_stock_wrf_evidence"]) == 3
    assert all(
        len(value["mapping_sha256"]) == len(value["composition_sha256"]) == 64
        for value in manifest["mapped_stock_wrf_evidence"]
    )
    retained_pairs = (
        (
            "rw-wps-era5-1974-probe.mapping.json",
            "rw-wps-era5-1974-terrain.composition.json",
        ),
        (
            "rw-wps-era5-netcdf.mapping.json",
            "rw-wps-era5-netcdf-terrain.composition.json",
        ),
    )
    for evidence, (mapping_name, composition_name) in zip(
        manifest["mapped_stock_wrf_evidence"][:2],
        retained_pairs,
        strict=True,
    ):
        assert evidence["mapping_sha256"] == hashlib.sha256(
            (ROOT / "configs" / mapping_name).read_bytes()
        ).hexdigest()
        # Replacing the named soil packings changes the composition authority.
        # Exact historical stock-WRF evidence must not silently certify the new
        # declarative contract even though unit gates prove identical soil
        # arrays for ERA5/GFS.
        assert evidence["composition_sha256"] != hashlib.sha256(
            (ROOT / "configs" / composition_name).read_bytes()
        ).hexdigest()

    current_gfs = manifest["mapped_stock_wrf_evidence"][2]
    assert current_gfs["mapping_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json").read_bytes()
    ).hexdigest()
    assert current_gfs["composition_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-terrain.composition.json").read_bytes()
    ).hexdigest()
    assert "clean-d66e442" in current_gfs["gate"]
    assert current_gfs["wrf_commit"] == \
        "d66e442fccc04111067e29274c9f9eaccc3cef28"
    assert current_gfs["wrf_exe_sha256"] == \
        "cfac96554c8f9796c7522aaf023131ea7681ddf12110a327e51a548958874089"
    assert "d01 120x100x49 at 12 km" in current_gfs["domain_envelope"]
    assert "d02-d04" in current_gfs["domain_envelope"]
    assert "60x60x49 at 4 km" in current_gfs["domain_envelope"]
    assert current_gfs["stock_wrf_result"] == \
        "exit_0_success_complete_wrf_d02_d03_d04_six_steps_each"
    assert current_gfs["stock_wrf_input_receipt_sha256"] == \
        "25a643fe34ff1ddd39464129bdcfff074e938ced216608f8cab8d17196ea524c"

    invalidated = manifest["invalidated_mapped_stock_wrf_evidence"]
    assert len(invalidated) == 1
    gfs = invalidated[0]
    assert gfs["gate"] == "wrf-v4.6.1-pass-gfs-grib2-d01-d02"
    assert gfs["mapping_sha256"] == \
        "726677d8c2365e6f533cc6dd5d7c795e198164326660c3630d885c83f406a11e"
    assert gfs["composition_sha256"] == \
        "266c98099b24f03a3bc986f275b44bbd6bf20dce1006ad05f6da39bc4a373bfb"
    assert gfs["replacement_mapping_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json").read_bytes()
    ).hexdigest()
    assert gfs["replacement_composition_sha256"] == hashlib.sha256(
        (ROOT / "configs" / "rw-wps-gfs-terrain.composition.json").read_bytes()
    ).hexdigest()
    assert gfs["replacement_status"] == "stock_wrf_certified_d01_d04_z49"
    assert gfs["replacement_gate"] == current_gfs["gate"]
    assert "evidence does not transfer" in gfs["reason"]


def _time() -> TimeDescriptor:
    return TimeDescriptor(
        reference_time="2026-07-18T00:00:00+00:00",
        valid_time="2026-07-18T01:00:00+00:00",
        lead_seconds=3600,
    )


def _field(name: str) -> FieldDescriptor:
    is_3d = name in REQUIRED_3D_FIELDS or name == "air_pressure"
    if name in {"soil_temperature", "volumetric_soil_moisture"}:
        dimensions = ("soil_depth", "y", "x")
        shape = (4, 3, 4)
        vertical = "soil"
    elif is_3d:
        dimensions = ("level", "y", "x")
        shape = (2, 3, 4)
        vertical = "pressure"
    else:
        dimensions = ("y", "x")
        shape = (3, 4)
        vertical = None
    return FieldDescriptor(
        canonical_name=name,
        units="1" if "fraction" in name else "SI",
        dimensions=dimensions,
        grid_location="mass",
        vertical_coordinate=vertical,
        time=_time(),
        data_reference=f"sha256:{name}",
        dtype="<f4",
        shape=shape,
        source_field=name.upper(),
    )


def _valid_header() -> SourceFrameHeader:
    names = sorted(REQUIRED_3D_FIELDS | REQUIRED_SURFACE_FIELDS | {"air_pressure"})
    return SourceFrameHeader(
        source_id="test",
        source_cycle="2026-07-18T00:00:00+00:00",
        grid=GridDescriptor(
            projection="lambert_conformal",
            nx=4,
            ny=3,
            earth_shape="sphere:6371229m",
            scan_order="+x,+y",
            wind_basis="earth_relative",
            parameters={"lov": -97.5},
        ),
        vertical_coordinates={
            "pressure": VerticalDescriptor(
                coordinate="pressure",
                level_count=2,
                level_values=(100000.0, 90000.0),
            ),
            "soil": VerticalDescriptor(
                coordinate="soil_depth",
                level_count=4,
                level_values=(0.05, 0.25, 0.70, 1.50),
                units="m",
                positive="down",
            ),
        },
        fields=tuple(_field(name) for name in names),
        initialization_policies={
            name: "explicit_zero_with_adapter_validation"
            for name in POLICY_CONTROLLED_FIELDS
        },
    )


def test_canonical_source_frame_accepts_complete_explicit_state():
    validate_source_frame(_valid_header())


def test_canonical_source_frame_rejects_missing_science_state():
    header = _valid_header()
    fields = tuple(
        value for value in header.fields if value.canonical_name != "specific_humidity"
    )
    with pytest.raises(ValueError, match="specific_humidity"):
        validate_source_frame(
            SourceFrameHeader(**{**header.__dict__, "fields": fields})
        )


def test_canonical_source_frame_rejects_ambiguous_missing_policy():
    header = _valid_header()
    policies = dict(header.initialization_policies)
    policies.pop("sea_ice_fraction")
    with pytest.raises(ValueError, match="sea_ice_fraction"):
        validate_source_frame(
            SourceFrameHeader(
                **{**header.__dict__, "initialization_policies": policies}
            )
        )


def test_interval_time_semantics_are_explicit():
    bad_time = TimeDescriptor(
        reference_time="2026-07-18T00:00:00Z",
        valid_time="2026-07-18T01:00:00Z",
        lead_seconds=3600,
        statistic="accumulation",
        interval_start="2026-07-18T00:00:00Z",
        interval_end="2026-07-18T01:00:00Z",
    )
    header = _valid_header()
    first = header.fields[0]
    fields = (
        FieldDescriptor(**{**first.__dict__, "time": bad_time}),
        *header.fields[1:],
    )
    with pytest.raises(ValueError, match="accumulation_reset"):
        validate_source_frame(
            SourceFrameHeader(**{**header.__dict__, "fields": fields})
        )


def test_cli_lists_machine_readable_inventory(capsys):
    assert main(["--list-sources"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_count"] == 27
    assert (
        payload["canonical_source_frame"]["schema"] == "gpuwm-canonical-source-frame-v1"
    )


def test_cli_reports_installed_version_without_run_arguments(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == "RW-WPS 0.1.1"


def test_cli_exposes_fail_closed_public_support_matrix(capsys):
    assert main(["--show-support-matrix"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "gpuwm-native-wrf-support-matrix-v1"
    assert payload["public_release_ready"] is False
    assert payload["release_state"] \
        == (
            "dedicated_rw_wps_cross_checkout_reproducible_clean_linux_cpu_"
            "package_gate"
        )
    assert (
        payload["final_products"]["wrfinput_d02_through_d06"]
        == "stock_wrf_proof_passed_for_certified_hrrr_slice"
    )
    assert payload["final_products"]["wrfinput_d02"] == (
        "stock_wrf_proof_passed_for_certified_hrrr_slice_and_canonical_lf_"
        "current_contract_gfs"
    )
    assert "stock-WRF proof passed through declared max_dom 4" in (
        payload["meteorological_sources"]["gfs_pgrb2_0p25"][
            "nested_hierarchy"
        ]
    )
    assert (
        payload["final_products"]["wrfinput_d07_through_d21"]
        == "parser_and_atomic_export_covered_not_stock_wrf_gated"
    )
    assert (
        payload["final_products"]["wrfinput_d03_through_dNN"]
        == "deprecated_aggregate_key_see_split_depth_keys"
    )
    assert payload["final_products"]["wrfinput_d02_through_d04_gfs"] == (
        "stock_wrf_proof_passed_for_canonical_lf_current_contract"
    )
    assert (
        payload["execution_backends"]["parallel_per_domain_initialization"]
        == "implemented_for_hrrr_d01_through_d21_and_live_gated_through_d06"
    )
    twentycr = payload["meteorological_sources"]["20crv3_member_grib2"]
    assert "adapter_and_native_preparation_implemented" in twentycr["single_domain"]
    assert "max_dom_4" in twentycr["nested_hierarchy"]
    assert twentycr["public_cli"] == (
        "rw-wps_source_20crv3_with_create_only_member_manifest_authoring"
    )
    assert "packaged_in_wheel" in twentycr["distribution"]
    assert "without_cupy_import" in (
        payload["execution_backends"]["cpu_only_setup_export"]
    )
    assert (
        payload["release_artifacts"]["model_independent_python_wheel"]
        == "dedicated_rw_wps_distribution_clean_linux_cpu_gate_pass_"
           "forecast_executors_absent"
    )
    assert "license_owner_decision" in payload["known_release_blockers"]


def test_cli_refuses_unimplemented_source_before_touching_files(capsys):
    assert main(["--source", "gdas"]) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert "REFUSED source=gdas" in error
    assert "stock-wrf evidence is a separate certification gate" in error


def test_cli_reports_and_refuses_uncertified_20crv3_cf_route(capsys):
    assert main(["--show-source", "20crv3-netcdf"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_id"] == "20crv3-cf"
    assert payload["runnable"] is False
    assert payload["status"] == "member_selection_and_mapping_required"

    assert main(["--source", "20cr-netcdf"]) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert "REFUSED source=20crv3-cf" in error
    assert "NetCDF-CF route has no public WRF runner" in error


def _twentycr_args():
    return [
        "--source",
        "20crv3",
        "--source-manifest",
        "/case/member072.manifest.json",
        "--source-manifest-sha256",
        "abc123",
        "--grib2-inventory",
        "/bin/grib2_inventory",
        "--grib2-dump",
        "/bin/grib2_dump",
        "--wps-namelist",
        "/case/namelist.wps",
        "--geog-root",
        "/static/WPS_GEOG",
        "--experiment-config",
        "/case/experiment.toml",
        "--output-root",
        "/output",
    ]


def test_cli_routes_20crv3_through_immutable_packaged_authorities(capsys):
    result = main(
        _twentycr_args()
        + [
            "--preprocess-backend",
            "cpu",
            "--preprocess-workers",
            "8",
            "--cpu-preprocess-bridge",
            "/bin/libgpuwm_preprocess_cpu.so",
            "--hierarchy-workers",
            "8",
            "--dry-run",
        ]
    )

    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.twentycrv3_wrf" in command
    assert "gpuwm/authorities/rw-wps-20crv3-member-grib2.mapping.json" in command
    assert "gpuwm/authorities/rw-wps-20crv3-member-grib2.composition.json" in command
    assert "gpuwm/authorities/rw-wps-20crv3-member-grib2.provenance.json" in command
    assert "--manifest /case/member072.manifest.json" in command
    assert "--hierarchy-workers 8" in command


def test_cli_20crv3_rejects_authority_override(capsys):
    result = main(
        _twentycr_args()
        + ["--mapping", "/case/replacement.json", "--dry-run"]
    )

    assert result == EXIT_USAGE
    assert "--mapping is not used by --source 20crv3" in capsys.readouterr().err


@pytest.mark.parametrize("source", ("gfs", "era5", "20crv3"))
@pytest.mark.parametrize(
    "flag",
    (
        "--forecast-start-hour",
        "--forecast-end-hour",
        "--history-interval-seconds",
    ),
)
def test_non_hrrr_sources_reject_explicit_zero_hrrr_only_flag(
        source, flag, capsys):
    result = main([
        "--source", source,
        flag, "0",
        "--dry-run",
    ])

    assert result == EXIT_USAGE
    assert f"{flag} is not used by --source {source}" \
        in capsys.readouterr().err


def test_cli_20crv3_authors_create_only_member_manifest(
    tmp_path: Path, capsys
):
    source = tmp_path / "member072"
    source.mkdir()
    for stamp in ("1932032100", "1932032103"):
        for role in ("pl", "sfc"):
            (source / f"mem072_{stamp}_{role}.grb2").write_bytes(
                f"fixture:{stamp}:{role}".encode()
            )
    manifest = tmp_path / "member072.manifest.json"

    result = main([
        "--source",
        "20crv3",
        "--source-root",
        str(source),
        "--author-input-manifest",
        str(manifest),
        "--author-only",
    ])

    assert result == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["manifest"]["member"] == "072"
    assert receipt["manifest"]["file_count"] == 4
    assert receipt["manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def _mapped_args(source_format="grib2"):
    values = [
        "--source",
        "mapped",
        "--source-format",
        source_format,
        "--mapping",
        "/case/mapping.json",
        "--composition",
        "/case/composition.json",
        "--input",
        "/data/source-f000.grb2",
        "--input",
        "/data/source-f003.grb2",
        "--supplement",
        "terrain=/data/terrain.grb2",
        "--provenance",
        "terrain=/data/terrain-provenance.md",
        "--wps-namelist",
        "/case/namelist.wps",
        "--geog-root",
        "/static/WPS_GEOG",
        "--experiment-config",
        "/case/experiment.toml",
        "--source-sha256s",
        "/case/input-manifest.json",
        "--source-sha256s-sha256",
        "abc123",
        "--output-root",
        "/output",
    ]
    if source_format == "grib1":
        values.extend(("--bridge", "/bin/grib1_bridge"))
    elif source_format == "grib2":
        values.extend(
            (
                "--grib2-inventory",
                "/bin/grib2_inventory",
                "--grib2-dump",
                "/bin/grib2_dump",
            )
        )
    return values


def test_cli_routes_mapped_hierarchy_to_native_composition_engine(capsys):
    assert (
        main(
            _mapped_args()
            + [
                "--preprocess-backend",
                "cpu",
                "--preprocess-workers",
                "8",
                "--cpu-preprocess-bridge",
                "/bundle/libgpuwm_preprocess_cpu.so",
                "--hierarchy-workers",
                "8",
                "--dry-run",
            ]
        )
        == 0
    )
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.mapped_direct" in command
    assert command.count("--input ") == 2
    assert "--supplement terrain=/data/terrain.grb2" in command
    assert "--provenance terrain=/data/terrain-provenance.md" in command
    assert "--grib2-inventory /bin/grib2_inventory" in command
    assert "--grib2-dump /bin/grib2_dump" in command
    assert "--preprocess-backend cpu" in command
    assert "--hierarchy-workers 8" in command


def test_cli_mapped_rejects_hrrr_history_interval(capsys):
    result = main(
        _mapped_args()
        + ["--history-interval-seconds", "3600", "--dry-run"]
    )

    assert result == EXIT_USAGE
    assert "--history-interval-seconds is not used by --source mapped" in (
        capsys.readouterr().err
    )


def test_cli_mapped_fails_closed_on_incomplete_decoder_or_mixed_args(capsys):
    args = _mapped_args()
    dump_index = args.index("--grib2-dump")
    del args[dump_index : dump_index + 2]
    result = main(args + ["--gfs-series", "/wrong/route.tsv", "--dry-run"])
    assert result == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--grib2-dump is required for mapped grib2" in error
    assert "--gfs-series is not used by --source mapped" in error


def test_cli_mapped_dry_run_rejects_malformed_or_duplicate_role_bindings(capsys):
    args = _mapped_args("netcdf")
    supplement_index = args.index("--supplement")
    args[supplement_index + 1] = "bad role=/data/terrain.nc"
    args.extend(
        (
            "--provenance",
            "terrain=/data/duplicate-provenance.md",
        )
    )
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--supplement must use" in error
    assert "--provenance repeats singleton role 'terrain'" in error


def test_cli_mapped_author_only_authors_descriptor_and_exact_manifest(
    monkeypatch, capsys
):
    args = _mapped_args()
    for flag in (
        "--mapping",
        "--source-sha256s",
        "--source-sha256s-sha256",
    ):
        index = args.index(flag)
        del args[index : index + 2]
    args.extend(
        (
            "--descriptor",
            "/case/source.descriptor.json",
            "--vtable",
            "/case/Vtable.GENERIC",
            "--author-mapping",
            "/case/generated.mapping.json",
            "--author-input-manifest",
            "/case/generated.inputs.json",
        )
    )
    observed = {}

    def mapping(descriptor, output, *, vtable_path, expected_format):
        observed["mapping"] = (
            descriptor,
            output,
            vtable_path,
            expected_format,
        )
        return {
            "format": "grib2",
            "mapping": {"sha256": "1" * 64},
        }

    def manifest(output, **kwargs):
        observed["manifest"] = (output, kwargs)
        return {
            "source_format": "grib2",
            "manifest": {"sha256": "2" * 64},
        }

    monkeypatch.setattr("gpuwm.source_cli.author_mapping", mapping)
    monkeypatch.setattr("gpuwm.source_cli.author_input_manifest", manifest)

    assert main(args + ["--author-only"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["mapping"]["mapping"]["sha256"] == "1" * 64
    assert result["input_manifest"]["manifest"]["sha256"] == "2" * 64
    assert observed["mapping"] == (
        Path("/case/source.descriptor.json"),
        Path("/case/generated.mapping.json"),
        Path("/case/Vtable.GENERIC"),
        "grib2",
    )
    assert observed["manifest"][1]["primary_files"] == [
        Path("/data/source-f000.grb2"),
        Path("/data/source-f003.grb2"),
    ]
    assert "AUTHORED mapping=" in captured.err
    assert "AUTHORED input_manifest=" in captured.err


def test_cli_dry_run_is_side_effect_free_for_authoring(monkeypatch, capsys):
    called = False

    def author(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("gpuwm.source_cli.author_mapping", author)
    args = _mapped_args()
    args.extend(
        (
            "--descriptor",
            "/case/source.descriptor.json",
            "--author-mapping",
            "/case/generated.mapping.json",
            "--author-input-manifest",
            "/case/generated.inputs.json",
        )
    )
    assert main(args + ["--dry-run"]) == EXIT_USAGE
    assert called is False
    assert "--dry-run is side-effect free" in capsys.readouterr().err


def test_cli_mapped_authoring_modes_are_exclusive_and_fail_before_write(
    monkeypatch, capsys
):
    called = False

    def author(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("gpuwm.source_cli.author_mapping", author)
    result = main(
        _mapped_args()
        + [
            "--descriptor",
            "/case/also-descriptor.json",
            "--author-mapping",
            "/case/generated.mapping.json",
            "--vtable",
            "/case/Vtable",
        ]
    )
    assert result == EXIT_USAGE
    assert not called
    assert "choose exactly one of --mapping" in capsys.readouterr().err


def test_cli_mapped_author_only_does_not_require_run_geometry(monkeypatch, capsys):
    args = _mapped_args()
    for flag in (
        "--wps-namelist",
        "--geog-root",
        "--experiment-config",
        "--output-root",
        "--source-sha256s",
        "--source-sha256s-sha256",
    ):
        index = args.index(flag)
        del args[index : index + 2]
    args.extend(
        (
            "--author-input-manifest",
            "/case/generated.inputs.json",
            "--author-only",
        )
    )
    monkeypatch.setattr(
        "gpuwm.source_cli.author_input_manifest",
        lambda *_args, **_kwargs: {
            "source_format": "grib2",
            "manifest": {"sha256": "3" * 64},
            "status": "PASS",
        },
    )

    assert main(args) == 0
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert receipt["schema"] == "rw-wps.contract-authoring.v1"
    assert receipt["input_manifest"]["status"] == "PASS"
    assert "AUTHORED input_manifest=" in captured.err


def test_cli_combined_authoring_rolls_back_owned_mapping_on_manifest_failure(
    tmp_path, monkeypatch, capsys
):
    args = _mapped_args()
    for flag in ("--mapping", "--source-sha256s", "--source-sha256s-sha256"):
        index = args.index(flag)
        del args[index : index + 2]
    mapping_path = tmp_path / "generated.mapping.json"
    manifest_path = tmp_path / "generated.inputs.json"
    args.extend(
        (
            "--descriptor",
            str(tmp_path / "descriptor.json"),
            "--vtable",
            str(tmp_path / "Vtable"),
            "--author-mapping",
            str(mapping_path),
            "--author-input-manifest",
            str(manifest_path),
            "--author-only",
        )
    )

    def mapping(_descriptor, output, **_kwargs):
        output = Path(output)
        output.write_bytes(b"owned mapping\n")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        receipt = {"mapping": {"sha256": digest}}
        output.with_name(f"{output.stem}.authoring.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return receipt

    monkeypatch.setattr("gpuwm.source_cli.author_mapping", mapping)
    monkeypatch.setattr(
        "gpuwm.source_cli.author_input_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("manifest failed")),
    )

    assert main(args) == EXIT_CONFIG
    assert not mapping_path.exists()
    assert not mapping_path.with_name(f"{mapping_path.stem}.authoring.json").exists()
    assert not manifest_path.exists()
    assert "manifest failed" in capsys.readouterr().err


def test_cli_validates_preprocess_options_before_authoring(
    tmp_path, monkeypatch, capsys
):
    args = _mapped_args()
    for flag in ("--source-sha256s", "--source-sha256s-sha256"):
        index = args.index(flag)
        del args[index : index + 2]
    output = tmp_path / "inputs.json"
    args.extend(
        (
            "--author-input-manifest",
            str(output),
            "--preprocess-backend",
            "cuda",
            "--preprocess-workers",
            "8",
        )
    )
    called = False

    def author(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("gpuwm.source_cli.author_input_manifest", author)
    assert main(args) == EXIT_USAGE
    assert called is False
    assert not output.exists()
    assert "preprocess-workers requires" in capsys.readouterr().err


def test_cli_hrrr_dry_run_routes_to_certified_internal_adapter(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--static-cache",
            "/static/cache.npz",
            "--static-receipt",
            "/static/receipt.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_00:00:00",
            "--output-root",
            "/output",
            "--pipeline-workers",
            "8",
            "--history-interval-seconds",
            "3600",
            "--prepare-workers",
            "4",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out
    assert "prepare_hrrr_wrf.py" in command
    assert "/source/SHA256SUMS" in command.replace("\\", "/")
    assert "--run-seconds 43200" in command
    assert "--forecast-start-hour 0" in command
    assert "--physics-profile wsm6-ysu-mm5-noah-no-radiation-v1" in command
    assert "--pipeline-workers 8" in command
    assert "--history-interval-seconds 3600.0" in command
    assert command.rstrip().endswith("--prepare-workers 4")


@pytest.mark.parametrize("cadence", ("0", "-1", "nan", "inf"))
def test_cli_hrrr_rejects_invalid_history_interval(cadence, capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--static-cache",
            "/static/cache.npz",
            "--static-receipt",
            "/static/receipt.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_00:00:00",
            "--output-root",
            "/output",
            "--history-interval-seconds",
            cadence,
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "history-interval-seconds must be positive and finite" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("cycle", "start", "end"),
    (("2026-07-18_05:00:00", "12", "18"),
     ("2026-07-18_18:00:00", "40", "46")),
)
def test_cli_hrrr_routes_absolute_source_windows_and_thompson_profile(
        cycle, start, end, capsys):
    result = main([
        "--source", "hrrr",
        "--source-root", "/source",
        "--source-sha256s", "/source/SHA256SUMS",
        "--source-sha256s-sha256", "abc123",
        "--static-cache", "/static/cache.npz",
        "--static-receipt", "/static/receipt.json",
        "--namelist-input", "/case/namelist.input",
        "--valid-time", cycle,
        "--output-root", "/output",
        "--run-seconds", str(6 * 3600),
        "--forecast-start-hour", start,
        "--forecast-end-hour", end,
        "--physics-profile", "thompson-mp8-ysu-mm5-noah-validation-v1",
        "--dry-run",
    ])
    assert result == 0
    command = capsys.readouterr().out
    assert f"--forecast-start-hour {start}" in command
    assert f"--forecast-end-hour {end}" in command
    assert "--physics-profile thompson-mp8-ysu-mm5-noah-validation-v1" \
        in command


def test_cli_hrrr_rejects_source_window_beyond_standard_cycle(capsys):
    result = main([
        "--source", "hrrr",
        "--source-root", "/source",
        "--source-sha256s", "/source/SHA256SUMS",
        "--source-sha256s-sha256", "abc123",
        "--static-cache", "/static/cache.npz",
        "--static-receipt", "/static/receipt.json",
        "--namelist-input", "/case/namelist.input",
        "--valid-time", "2026-07-18_05:00:00",
        "--output-root", "/output",
        "--run-seconds", "3600",
        "--forecast-start-hour", "18",
        "--forecast-end-hour", "19",
        "--dry-run",
    ])
    assert result == EXIT_USAGE
    assert "horizon f18" in capsys.readouterr().err


def test_cli_hrrr_domain_dry_run_is_one_command_from_geog(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--geog-root",
            "/static/WPS_GEOG",
            "--domain-spec",
            "/case/domain.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_06:00:00",
            "--output-root",
            "/output",
            "--pipeline-workers",
            "8",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "prepare_hrrr_wrf.py" in command
    assert command.index("/static/WPS_GEOG") < command.index("/case/domain.json")
    assert "--run-seconds 43200" in command
    assert "--pipeline-workers 8" in command


def _hrrr_hierarchy_args():
    return [
        "--source",
        "hrrr",
        "--root-preparation",
        "/prepared/root",
        "--domain-spec",
        "/case/domain.json",
        "--wps-namelist",
        "/case/namelist.wps",
        "--namelist-input",
        "/case/namelist.native.input",
        "--stock-wrf-namelist-input",
        "/case/namelist.stock.input",
        "--geog-root",
        "/static/WPS_GEOG",
        "--source-sha256s",
        "/source/SHA256SUMS",
        "--source-sha256s-sha256",
        "abc123",
        "--valid-time",
        "2026-07-18_00:00:00",
        "--output-root",
        "/output",
    ]


def test_cli_hrrr_hierarchy_routes_to_public_parallel_orchestrator(capsys):
    result = main(
        _hrrr_hierarchy_args()
        + [
            "--child-workers",
            "8",
            "--cpu-preprocess-bridge",
            "/bundle/libgpuwm_preprocess_cpu.so",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.hrrr_hierarchy_direct" in command
    assert "--root-preparation /prepared/root" in command
    assert "--stock-wrf-namelist-input /case/namelist.stock.input" in command
    assert "--workers 8" in command
    assert "--cpu-preprocess-bridge /bundle/libgpuwm_preprocess_cpu.so" in command


def test_cli_hrrr_hierarchy_rejects_history_interval(capsys):
    result = main(
        _hrrr_hierarchy_args()
        + ["--history-interval-seconds", "3600", "--dry-run"]
    )

    assert result == EXIT_USAGE
    assert "--history-interval-seconds is not used by HRRR hierarchy export" in (
        capsys.readouterr().err
    )


def test_public_hierarchy_help_and_installed_entrypoint_cover_d01_through_d21():
    help_text = _parser().format_help()
    assert "d01..dNN hierarchy export" in help_text
    assert "max_dom 1..21" in help_text
    assert "d02..dNN" in help_text
    assert "initialization (1..32)" in help_text
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["scripts"]["gpuwm-wrf-init"] == "gpuwm.source_cli:main"
    assert project["project"]["scripts"]["rw-wps"] == "gpuwm.source_cli:main"


def test_public_support_matrix_separates_live_depth_from_max_dom_coverage(capsys):
    assert main(["--show-support-matrix"]) == 0
    matrix = json.loads(capsys.readouterr().out)
    assert (
        matrix["domain_geometry"]["grid_ids"]
        == "contiguous_parent_before_child_d01_through_d21"
    )
    assert (
        matrix["final_products"]["wrfinput_d02_through_d06"]
        == "stock_wrf_proof_passed_for_certified_hrrr_slice"
    )
    assert (
        matrix["final_products"]["wrfinput_d07_through_d21"]
        == "parser_and_atomic_export_covered_not_stock_wrf_gated"
    )


@pytest.mark.parametrize("workers", (0, 33))
def test_cli_hrrr_hierarchy_rejects_out_of_bounds_workers(workers, capsys):
    result = main(
        _hrrr_hierarchy_args() + ["--child-workers", str(workers), "--dry-run"]
    )
    assert result == EXIT_USAGE
    assert "child-workers must be between 1 and 32" in capsys.readouterr().err


def test_cli_hrrr_hierarchy_refuses_single_domain_source_arguments(capsys):
    result = main(
        _hrrr_hierarchy_args() + ["--source-root", "/source/raw", "--dry-run"]
    )
    assert result == EXIT_USAGE
    assert (
        "--source-root is not used by HRRR hierarchy export" in capsys.readouterr().err
    )


def test_cli_hrrr_refuses_mixed_static_modes(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--geog-root",
            "/static/WPS_GEOG",
            "--domain-spec",
            "/case/domain.json",
            "--static-cache",
            "/case/static.npz",
            "--static-receipt",
            "/case/static.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_00:00:00",
            "--output-root",
            "/output",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "cannot be mixed" in capsys.readouterr().err


def test_cli_hrrr_refuses_non_hourly_cycle(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--static-cache",
            "/case/static.npz",
            "--static-receipt",
            "/case/static.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_00:30:00",
            "--output-root",
            "/output",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "exact hourly HRRR cycle" in capsys.readouterr().err


def test_cli_hrrr_rejects_worker_count_above_decoder_limit(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--static-cache",
            "/case/static.npz",
            "--static-receipt",
            "/case/static.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_00:00:00",
            "--output-root",
            "/output",
            "--pipeline-workers",
            "65",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "between 1 and 64" in capsys.readouterr().err


def test_cli_hrrr_accepts_64_decoder_hour_workers(capsys):
    result = main(
        [
            "--source", "hrrr",
            "--source-root", "/source",
            "--source-sha256s", "/source/SHA256SUMS",
            "--source-sha256s-sha256", "abc123",
            "--static-cache", "/case/static.npz",
            "--static-receipt", "/case/static.json",
            "--namelist-input", "/case/namelist.input",
            "--valid-time", "2026-07-18_00:00:00",
            "--output-root", "/output",
            "--pipeline-workers", "64",
            "--dry-run",
        ]
    )

    assert result == 0
    assert "--pipeline-workers 64" in capsys.readouterr().out


def test_cli_hrrr_refuses_era5_only_arguments(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--grib",
            "/source/era5.grb",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "--grib is not used by --source hrrr" in capsys.readouterr().err


def test_cli_era5_dry_run_routes_to_certified_internal_adapter(capsys):
    result = main(
        [
            "--source",
            "era5",
            "--grib",
            "/source/era5.grb",
            "--vtable",
            "/source/Vtable.ERA5_CDO",
            "--bridge",
            "/bin/grib1_bridge",
            "--wps-namelist",
            "/case/namelist.wps",
            "--static-input",
            "/case/static.npz",
            "--static-receipt",
            "/case/static-receipt.json",
            "--source-orography",
            "/case/met_em.nc",
            "--experiment-config",
            "/case/experiment.toml",
            "--source-sha256s",
            "/source/input-manifest.json",
            "--source-sha256s-sha256",
            "abc123",
            "--output-root",
            "/output",
            "--preprocess-backend",
            "auto",
            "--preprocess-workers",
            "4",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.era5_direct" in command
    assert "--bridge" in command and "/bin/grib1_bridge" in command
    assert "--input-manifest" in command
    assert "/source/input-manifest.json" in command
    assert "--input-manifest-sha256 abc123" in command
    assert "--source-orography-variable SOILHGT" in command
    assert "--preprocess-backend auto" in command
    assert "--preprocess-workers 4" in command


def test_cli_era5_d06_hierarchy_routes_all_domain_contracts(capsys):
    arguments = [
        "--source", "era5",
        "--grib", "/source/era5.grb",
        "--vtable", "/source/Vtable.ERA5_CDO",
        "--bridge", "/bin/grib1_bridge",
        "--wps-namelist", "/case/namelist.wps",
        "--experiment-config", "/case/experiment.toml",
        "--source-sha256s", "/source/input-manifest.json",
        "--source-sha256s-sha256", "abc123",
        "--output-root", "/output",
        "--geog-root", "/wps-geog",
        "--hierarchy-workers", "6",
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "8",
        "--cpu-preprocess-bridge", "/bundle/libgpuwm_preprocess_cpu.so",
    ]
    assert main(arguments + ["--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--geog-root /wps-geog" in command
    assert "--hierarchy-workers 6" in command
    assert "--static-input" not in command
    assert "--source-orography" not in command


def test_cli_era5_explicit_orography_contract_is_atomic(capsys):
    result = main([
        "--source", "era5",
        "--geog-root", "/wps-geog",
        "--domain-source-orography", "d01=/case/d01.nc",
        "--dry-run",
    ])
    assert result == EXIT_USAGE
    assert "--source-orography is required with explicit" in capsys.readouterr().err


def test_cli_era5_refuses_hrrr_only_arguments(capsys):
    result = main(
        [
            "--source",
            "era5",
            "--source-root",
            "/hrrr-style-directory",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--source-root is not used by --source era5" in error
    assert "--grib" in error


def test_cli_gfs_dry_run_routes_to_certified_internal_adapter(capsys):
    result = main(
        [
            "--source",
            "gfs",
            "--gfs-series",
            "/source/gfs-series.tsv",
            "--cycle",
            "2026-07-20_00:00:00",
            "--bridge",
            "/bin/gfs_grib2_bridge",
            "--wps-namelist",
            "/case/namelist.wps",
            "--static-input",
            "/case/static.npz",
            "--static-receipt",
            "/case/static-receipt.json",
            "--experiment-config",
            "/case/experiment.toml",
            "--source-sha256s",
            "/source/input-manifest.json",
            "--source-sha256s-sha256",
            "abc123",
            "--output-root",
            "/output",
            "--preprocess-backend",
            "cpu",
            "--preprocess-workers",
            "8",
            "--cpu-preprocess-bridge",
            "/bundle/libgpuwm_preprocess_cpu.so",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.gfs_direct" in command
    assert "--series" in command
    assert "/source/gfs-series.tsv" in command
    assert "--cycle 2026-07-20_00:00:00" in command
    assert "--preprocess-backend cpu" in command
    assert "--preprocess-workers 8" in command
    assert "--cpu-preprocess-bridge /bundle/libgpuwm_preprocess_cpu.so" in command


def test_cli_gfs_d06_hierarchy_routes_source_neutral_controls(capsys):
    result = main([
        "--source", "gfs",
        "--gfs-series", "/source/gfs-series.tsv",
        "--cycle", "2026-07-20_00:00:00",
        "--bridge", "/bin/gfs_grib2_bridge",
        "--wps-namelist", "/case/namelist.wps",
        "--experiment-config", "/case/experiment.toml",
        "--source-sha256s", "/source/input-manifest.json",
        "--source-sha256s-sha256", "abc123",
        "--output-root", "/output",
        "--geog-root", "/wps-geog",
        "--hierarchy-workers", "6",
        "--preprocess-backend", "cpu",
        "--preprocess-workers", "8",
        "--cpu-preprocess-bridge", "/bundle/libgpuwm_preprocess_cpu.so",
        "--dry-run",
    ])
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "-m gpuwm.gfs_direct" in command
    assert "--geog-root /wps-geog" in command
    assert "--hierarchy-workers 6" in command
    assert "--static-input" not in command


@pytest.mark.parametrize("source", ("era5", "gfs"))
def test_cli_named_source_static_cache_pair_is_atomic(source, capsys):
    result = main([
        "--source", source,
        "--static-input", "/case/static.npz",
        "--dry-run",
    ])
    assert result == EXIT_USAGE
    assert "must be supplied together" in capsys.readouterr().err


@pytest.mark.parametrize("source", ("era5", "gfs"))
def test_cli_named_source_refuses_parallel_cuda_hierarchy(source, capsys):
    common = [
        "--source", source,
        "--bridge", "/bin/source_bridge",
        "--wps-namelist", "/case/namelist.wps",
        "--experiment-config", "/case/experiment.toml",
        "--source-sha256s", "/source/input-manifest.json",
        "--source-sha256s-sha256", "abc123",
        "--output-root", "/output",
        "--geog-root", "/wps-geog",
        "--hierarchy-workers", "2",
        "--dry-run",
    ]
    if source == "era5":
        arguments = [*common,
            "--grib", "/source/era5.grb",
            "--vtable", "/source/Vtable.ERA5_CDO",
        ]
    else:
        arguments = [*common,
            "--gfs-series", "/source/gfs-series.tsv",
            "--cycle", "2026-07-20_00:00:00",
        ]
    result = main(arguments)
    assert result == EXIT_USAGE
    assert "explicit --preprocess-backend cpu" in capsys.readouterr().err


def test_cli_hrrr_routes_explicit_cpu_backend_options(capsys):
    result = main(
        [
            "--source",
            "hrrr",
            "--source-root",
            "/source",
            "--source-sha256s",
            "/source/SHA256SUMS",
            "--source-sha256s-sha256",
            "abc123",
            "--static-cache",
            "/case/static.npz",
            "--static-receipt",
            "/case/static.json",
            "--namelist-input",
            "/case/namelist.input",
            "--valid-time",
            "2026-07-18_00:00:00",
            "--output-root",
            "/output",
            "--preprocess-backend",
            "cpu",
            "--preprocess-workers",
            "8",
            "--cpu-preprocess-bridge",
            "/bundle/libgpuwm_preprocess_cpu.so",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--preprocess-backend cpu" in command
    assert "--preprocess-workers 8" in command
    assert (
        "--cpu-preprocess-bridge /bundle/libgpuwm_preprocess_cpu.so"
        in command
    )


def test_cli_hrrr_routes_auto_backend_workers(capsys):
    result = main(
        [
            "--source", "hrrr",
            "--source-root", "/source",
            "--source-sha256s", "/source/SHA256SUMS",
            "--source-sha256s-sha256", "abc123",
            "--static-cache", "/case/static.npz",
            "--static-receipt", "/case/static.json",
            "--namelist-input", "/case/namelist.input",
            "--valid-time", "2026-07-18_00:00:00",
            "--output-root", "/output",
            "--preprocess-backend", "auto",
            "--preprocess-workers", "4",
            "--dry-run",
        ]
    )
    assert result == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert "--preprocess-backend auto" in command
    assert "--preprocess-workers 4" in command


def test_cli_hrrr_rejects_cpu_bridge_with_auto(capsys):
    result = main(
        [
            "--source", "hrrr",
            "--source-root", "/source",
            "--source-sha256s", "/source/SHA256SUMS",
            "--source-sha256s-sha256", "abc123",
            "--static-cache", "/case/static.npz",
            "--static-receipt", "/case/static.json",
            "--namelist-input", "/case/namelist.input",
            "--valid-time", "2026-07-18_00:00:00",
            "--output-root", "/output",
            "--preprocess-backend", "auto",
            "--cpu-preprocess-bridge", "/bundle/libgpuwm_preprocess_cpu.so",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert "requires --preprocess-backend cpu" in capsys.readouterr().err


def test_cli_gfs_uses_distribution_bridge_environment(monkeypatch, capsys):
    monkeypatch.setenv("GPUWM_GFS_GRIB2_BRIDGE", "/bundle/gfs_grib2_bridge")
    result = main(
        [
            "--source",
            "gfs",
            "--gfs-series",
            "/source/gfs-series.tsv",
            "--cycle",
            "2026-07-20_00:00:00",
            "--wps-namelist",
            "/case/namelist.wps",
            "--static-input",
            "/case/static.npz",
            "--static-receipt",
            "/case/static-receipt.json",
            "--experiment-config",
            "/case/experiment.toml",
            "--source-sha256s",
            "/source/input-manifest.json",
            "--source-sha256s-sha256",
            "abc123",
            "--output-root",
            "/output",
            "--dry-run",
        ]
    )
    assert result == 0
    assert "/bundle/gfs_grib2_bridge" in capsys.readouterr().out.replace("\\", "/")


def test_cli_installed_distribution_rejects_explicit_decoder_substitution(
    tmp_path, monkeypatch, capsys
):
    bridge_root = tmp_path / "libexec" / "bridges"
    bridge_root.mkdir(parents=True)
    inventory = bridge_root / "grib2_inventory"
    dump = bridge_root / "grib2_dump"
    for path, payload in ((inventory, b"inventory"), (dump, b"dump")):
        path.write_bytes(payload)
        path.chmod(0o755)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gpuwm-native-wrf-runtime-v1",
                "status": "READY",
                "payload": {
                    f"libexec/bridges/{path.name}": {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "executable": True,
                    }
                    for path in (inventory, dump)
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    monkeypatch.setenv("GPUWM_GRIB2_INVENTORY", str(inventory))
    monkeypatch.setenv("GPUWM_GRIB2_DUMP", str(dump))

    assert main(_mapped_args() + ["--dry-run"]) == EXIT_CONFIG
    error = capsys.readouterr().err
    assert "--grib2-inventory differs from the decoder bound" in error


def test_cli_installed_distribution_uses_manifest_bound_decoders(
    tmp_path, monkeypatch, capsys
):
    bridge_root = tmp_path / "libexec" / "bridges"
    bridge_root.mkdir(parents=True)
    inventory = bridge_root / "grib2_inventory"
    dump = bridge_root / "grib2_dump"
    for path, payload in ((inventory, b"inventory"), (dump, b"dump")):
        path.write_bytes(payload)
        path.chmod(0o755)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gpuwm-native-wrf-runtime-v1",
                "status": "READY",
                "payload": {
                    f"libexec/bridges/{path.name}": {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "executable": True,
                    }
                    for path in (inventory, dump)
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    monkeypatch.setenv("GPUWM_GRIB2_INVENTORY", str(inventory))
    monkeypatch.setenv("GPUWM_GRIB2_DUMP", str(dump))
    args = _mapped_args()
    for flag in ("--grib2-inventory", "--grib2-dump"):
        index = args.index(flag)
        del args[index : index + 2]

    assert main(args + ["--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert str(inventory).replace("\\", "/") in command
    assert str(dump).replace("\\", "/") in command


def test_cli_installed_windows_distribution_uses_exe_decoders(
    tmp_path, monkeypatch, capsys
):
    bridge_root = tmp_path / "libexec" / "bridges"
    bridge_root.mkdir(parents=True)
    inventory = bridge_root / "grib2_inventory.exe"
    dump = bridge_root / "grib2_dump.exe"
    for path, payload in ((inventory, b"inventory"), (dump, b"dump")):
        path.write_bytes(payload)
        path.chmod(0o755)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gpuwm-native-wrf-runtime-v1",
                "status": "READY",
                "payload": {
                    f"libexec/bridges/{path.name}": {
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "executable": True,
                    }
                    for path in (inventory, dump)
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST", str(manifest))
    monkeypatch.setenv("GPUWM_GRIB2_INVENTORY", str(inventory))
    monkeypatch.setenv("GPUWM_GRIB2_DUMP", str(dump))
    args = _mapped_args()
    for flag in ("--grib2-inventory", "--grib2-dump"):
        index = args.index(flag)
        del args[index : index + 2]

    assert main(args + ["--dry-run"]) == 0
    command = capsys.readouterr().out.replace("\\", "/")
    assert str(inventory).replace("\\", "/") in command
    assert str(dump).replace("\\", "/") in command


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--run-seconds", "3600"),
        ("--history-interval-seconds", "3600"),
        ("--pipeline-workers", "8"),
        ("--source-orography-variable", "SOILHGT"),
    ],
)
def test_cli_gfs_rejects_common_options_that_it_does_not_apply(flag, value, capsys):
    result = main(
        [
            "--source",
            "gfs",
            "--gfs-series",
            "/source/gfs-series.tsv",
            "--cycle",
            "2026-07-20_00:00:00",
            "--bridge",
            "/bin/gfs_grib2_bridge",
            "--wps-namelist",
            "/case/namelist.wps",
            "--static-input",
            "/case/static.npz",
            "--static-receipt",
            "/case/static-receipt.json",
            "--experiment-config",
            "/case/experiment.toml",
            "--source-sha256s",
            "/source/input-manifest.json",
            "--source-sha256s-sha256",
            "abc123",
            "--output-root",
            "/output",
            flag,
            value,
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    assert f"{flag} is not used by --source gfs" in capsys.readouterr().err


def test_cli_gfs_rejects_non_synoptic_cycle_and_era5_inputs(capsys):
    result = main(
        [
            "--source",
            "gfs",
            "--cycle",
            "2026-07-20_03:00:00",
            "--grib",
            "/source/era5.grb",
            "--dry-run",
        ]
    )
    assert result == EXIT_USAGE
    error = capsys.readouterr().err
    assert "exact 00/06/12/18 UTC GFS cycle" in error
    assert "--grib is not used by --source gfs" in error

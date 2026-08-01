"""Phase-5 Task 2: [case_data] loading, provenance, and the exp-TOML pin."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from gpuwm.case_data import (CaseDataConfig, PerDomainSourceOrography,
                             ResolvedInput, SourceOrography, case_data_root,
                             load_case_data, load_experiment_case)

REPO = Path(__file__).resolve().parents[1]
# Same root the configs resolve ${GPUWM_CASE_DATA_ROOT} against, so the
# fixture bundle and the config under test relocate together.
BUNDLE = case_data_root() / "WRF_1974_MP55_reference_bundle"
requires_bundle = pytest.mark.skipif(
    not (BUNDLE / "geo_em" / "geo_em.d01.nc").is_file(),
    reason="WRF_1974_MP55 reference bundle not present",
)

_EXPERIMENT_TOML = """
[experiment]
name = "case_data_fixture"
start_time = 1999-05-03T12:00:00
run_seconds = 3600.0
restart_interval_s = 0.0

[shared]
nz = 8
ztop = 16000.0
p_top = 10000.0
eta_levels = [1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.1, 0.0]

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 24
ny = 20
time_step = 60
dx = 12000.0
history_interval_s = 3600.0
"""

_CASE_DATA_TOML = """
[case_data]
forcing = ["forcing/era5_a.grb", "forcing/era5_b.grb"]
vtable = "Vtable.ERA5"
wps_namelist = "namelist.wps"
geog_root = "GEOG"
source_orography = "invariant.nc"
source_orography_variable = "SOILHGT"
sfcp_to_sfcp = true
co2_vmr = 330.0e-6
forcing_interval_s = 21600.0
output_domain = 1
output_title = "fixture title"
"""


def make_case_toml(tmp_path, *, case_data=_CASE_DATA_TOML,
                   experiment=_EXPERIMENT_TOML, files=True) -> Path:
    """Write a loadable experiment+case_data TOML with its dummy inputs."""
    if files:
        (tmp_path / "forcing").mkdir(exist_ok=True)
        for name in ("forcing/era5_a.grb", "forcing/era5_b.grb",
                     "Vtable.ERA5", "namelist.wps", "invariant.nc"):
            (tmp_path / name).write_bytes(b"stub")
        (tmp_path / "GEOG").mkdir(exist_ok=True)
    path = tmp_path / "case.toml"
    path.write_text(experiment + case_data, encoding="utf-8")
    return path


def test_load_case_data_resolves_relative_paths_and_types(tmp_path):
    path = make_case_toml(tmp_path)
    data = load_case_data(path)
    assert isinstance(data, CaseDataConfig)
    assert data.forcing == (tmp_path / "forcing" / "era5_a.grb",
                            tmp_path / "forcing" / "era5_b.grb")
    assert data.vtable == tmp_path / "Vtable.ERA5"
    assert data.wps_namelist == tmp_path / "namelist.wps"
    assert data.geog_root == tmp_path / "GEOG"
    assert data.source_orography == SourceOrography(
        path=tmp_path / "invariant.nc", variable="SOILHGT")
    assert data.sfcp_to_sfcp is True
    assert data.co2_vmr == 330.0e-6
    assert data.forcing_interval_s == 21600.0
    assert data.output_domain == 1
    assert data.output_title == "fixture title"


def test_declared_paths_follow_a_portable_root(tmp_path, monkeypatch):
    """``${VAR}`` in a declared path expands from the environment.

    The same TOML text must read the same case from two different
    directories -- that is the property that lets a committed config stop
    hard-coding one machine's absolute paths."""
    text = _CASE_DATA_TOML.replace(
        'forcing = ["forcing/era5_a.grb", "forcing/era5_b.grb"]',
        'forcing = ["${GPUWM_CASE_DATA_ROOT}/forcing/era5_a.grb"]')
    for key, value in (("vtable", "Vtable.ERA5"),
                       ("wps_namelist", "namelist.wps"),
                       ("geog_root", "GEOG"),
                       ("source_orography", "invariant.nc")):
        text = text.replace(f'{key} = "{value}"',
                            f'{key} = "${{GPUWM_CASE_DATA_ROOT}}/{value}"')
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    for where in ("alpha", "beta"):
        root = tmp_path / where
        root.mkdir()
        make_case_toml(root)
        monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(root))
        path = make_case_toml(config_dir, case_data=text, files=False)
        data = load_case_data(path)
        assert data.forcing == (root / "forcing" / "era5_a.grb",)
        assert data.vtable == root / "Vtable.ERA5"
        assert data.wps_namelist == root / "namelist.wps"
        assert data.geog_root == root / "GEOG"
        assert data.source_orography.path == root / "invariant.nc"


def test_case_data_root_defaults_and_unknown_variables_are_loud(monkeypatch):
    """One variable resolves unset; every other name must be exported.

    A silent default for an arbitrary name would put back the "works on
    one machine" failure this mechanism exists to remove."""
    from gpuwm.case_data import (CASE_DATA_ROOT_ENV, default_case_data_root,
                                 expand_path_variables)

    monkeypatch.delenv(CASE_DATA_ROOT_ENV, raising=False)
    expanded = expand_path_variables(
        "${" + CASE_DATA_ROOT_ENV + "}/bundle", "forcing", "fixture.toml")
    assert Path(expanded) == default_case_data_root() / "bundle"

    monkeypatch.setenv(CASE_DATA_ROOT_ENV, "   ")
    assert Path(expand_path_variables(
        "${" + CASE_DATA_ROOT_ENV + "}/bundle", "forcing", "fixture.toml")
    ) == default_case_data_root() / "bundle"

    monkeypatch.delenv("GPUWM_NO_SUCH_ROOT", raising=False)
    with pytest.raises(ValueError, match="GPUWM_NO_SUCH_ROOT"):
        expand_path_variables("${GPUWM_NO_SUCH_ROOT}/x", "vtable", "f.toml")


def test_committed_configs_declare_no_machine_path():
    """Release hygiene: no committed config ships a developer's absolute
    path.  Machine paths in a source distribution are both unportable and
    a privacy leak, and one config re-introducing one is exactly how the
    previous state came about."""
    offenders = []
    for path in sorted((REPO / "configs").rglob("*")):
        if path.suffix.lower() not in {".toml", ".json"} or not path.is_file():
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r'"[A-Za-z]:[\\/]|"/(home|Users)/', stripped):
                offenders.append(f"{path.relative_to(REPO)}:{number}: "
                                 f"{stripped}")
    assert not offenders, (
        "declare these through ${GPUWM_CASE_DATA_ROOT} (or another "
        "environment variable) instead of an absolute path:\n"
        + "\n".join(offenders))


def test_forcing_glob_expands_sorted_and_rejects_empty(tmp_path):
    text = _CASE_DATA_TOML.replace(
        '["forcing/era5_a.grb", "forcing/era5_b.grb"]',
        '["forcing/*.grb"]')
    path = make_case_toml(tmp_path, case_data=text)
    data = load_case_data(path)
    assert data.forcing == (tmp_path / "forcing" / "era5_a.grb",
                            tmp_path / "forcing" / "era5_b.grb")

    empty = text.replace("forcing/*.grb", "forcing/*.nope")
    path = make_case_toml(tmp_path, case_data=empty)
    with pytest.raises(ValueError, match="matched no files"):
        load_case_data(path)


def test_load_experiment_case_returns_validated_pair(tmp_path):
    path = make_case_toml(tmp_path)
    exp, data = load_experiment_case(path)
    assert exp.name == "case_data_fixture"
    assert exp.start_time == datetime(1999, 5, 3, 12)
    assert data.output_title == "fixture title"


def test_catalog_orography_and_dated_co2_policies_are_reachable_by_omission(
        tmp_path):
    text = "\n".join(
        line for line in _CASE_DATA_TOML.splitlines()
        if not line.startswith(("source_orography =",
                                "source_orography_variable =",
                                "co2_vmr ="))
    ) + "\n"
    data = load_case_data(make_case_toml(tmp_path, case_data=text))
    assert data.source_orography is None
    assert all(data.source_orography_for_domain(domain_id) is None
               for domain_id in (1, 2, 3, 4))
    assert data.co2_vmr is None
    assert "source_orography" not in {
        record.role for record in data.resolved_inputs()}


def test_source_orography_declaration_pair_is_atomic(tmp_path):
    text = "\n".join(
        line for line in _CASE_DATA_TOML.splitlines()
        if not line.startswith("source_orography_variable =")
    ) + "\n"
    with pytest.raises(ValueError, match="declared together or both omitted"):
        load_case_data(make_case_toml(tmp_path, case_data=text))


def test_per_domain_source_orography_resolves_every_declared_grid(tmp_path):
    paths = {
        domain_id: tmp_path / f"orography.d{domain_id:02d}.nc"
        for domain_id in (1, 2, 3, 4)
    }
    for path in paths.values():
        path.write_bytes(b"stub")
    declarations = "\n".join(
        f'source_orography.d{domain_id:02d} = "{path.name}"'
        for domain_id, path in paths.items())
    text = _CASE_DATA_TOML.replace(
        'source_orography = "invariant.nc"', declarations)

    data = load_case_data(make_case_toml(tmp_path, case_data=text))

    assert isinstance(data.source_orography, PerDomainSourceOrography)
    for domain_id, path in paths.items():
        assert data.source_orography_for_domain(domain_id) == SourceOrography(
            path=path, variable="SOILHGT")
    source_records = tuple(
        record for record in data.resolved_inputs()
        if record.role == "source_orography")
    assert tuple(record.path for record in source_records) == tuple(
        paths.values())
    assert tuple(record.detail for record in source_records) == tuple(
        f"variable=SOILHGT;domain=d{domain_id:02d}"
        for domain_id in paths)


def test_single_source_orography_is_root_only_and_names_missing_child(tmp_path):
    data = load_case_data(make_case_toml(tmp_path))
    assert data.source_orography_for_domain(1) is data.source_orography
    with pytest.raises(ValueError, match=r"d02"):
        data.source_orography_for_domain(2)


def test_missing_case_data_table_is_loud(tmp_path):
    path = make_case_toml(tmp_path, case_data="")
    with pytest.raises(ValueError, match=r"\[case_data\]"):
        load_experiment_case(path)
    with pytest.raises(ValueError, match=r"\[case_data\]"):
        load_case_data(path)


def test_invalid_experiment_error_wins_over_missing_case_data(tmp_path):
    """The experiment's own validation error surfaces first."""
    broken = _EXPERIMENT_TOML.replace("parent_id = 0", "parent_id = 7")
    path = make_case_toml(tmp_path, experiment=broken, case_data="")
    with pytest.raises(ValueError, match="root"):
        load_experiment_case(path)


@pytest.mark.parametrize("mutation, message", [
    ("sfcp_to_sfcp = true", None),  # baseline sanity, no error expected
    ("sfcp_to_sfcp = false", "false branch is not implemented"),
    ("sfcp_to_sfcp = 1", "must be a boolean"),
    ("co2_vmr = 330.0e-6", None),
    ("co2_vmr = -1.0e-6", "positive mole fraction"),
    ("forcing_interval_s = 21600.0", None),
    ("forcing_interval_s = -60.0", "finite positive interval"),
    ("output_domain = 1", None),
    ("output_domain = 0", "positive integer"),
    ('output_title = "fixture title"', None),
    ('source_orography_variable = "SOILHGT"', None),
    ('source_orography_variable = ""', "non-empty variable name"),
])
def test_key_validation_is_fail_loud(tmp_path, mutation, message):
    key = mutation.split(" =")[0]
    lines = [line for line in _CASE_DATA_TOML.splitlines()
             if not line.startswith(key)]
    lines.append(mutation)
    path = make_case_toml(tmp_path, case_data="\n".join(lines) + "\n")
    if message is None:
        load_case_data(path)
    else:
        with pytest.raises(ValueError, match=message):
            load_case_data(path)


@pytest.mark.parametrize("mutation, needle, attr, expected", [
    ("co2_vmr = 0.5", "sanity envelope", "co2_vmr", 0.5),
    ('output_title = ""', "output_title", "output_title", "gpuwm"),
])
def test_advisory_values_warn_and_continue(tmp_path, capsys, mutation,
                                           needle, attr, expected):
    """Warn-not-block: sanity envelopes report in one line and keep
    going; the loaded value is stated by the warning."""
    key = mutation.split(" =")[0]
    lines = [line for line in _CASE_DATA_TOML.splitlines()
             if not line.startswith(key)]
    lines.append(mutation)
    path = make_case_toml(tmp_path, case_data="\n".join(lines) + "\n")
    data = load_case_data(path)
    err = capsys.readouterr().err
    assert "warning:" in err and needle in err
    assert getattr(data, attr) == expected


def test_unknown_keys_warn_and_missing_keys_are_loud(tmp_path, capsys):
    # Warn-not-block: a typo'd extra key is named and ignored.
    path = make_case_toml(
        tmp_path, case_data=_CASE_DATA_TOML + "mystery_key = 1\n")
    load_case_data(path)
    err = capsys.readouterr().err
    assert "warning:" in err and "mystery_key" in err

    # KEEP-HARD negative: a missing REQUIRED key still refuses.
    lines = [line for line in _CASE_DATA_TOML.splitlines()
             if not line.startswith("vtable")]
    path = make_case_toml(tmp_path, case_data="\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="missing required key"):
        load_case_data(path)


def test_missing_declared_files_are_loud(tmp_path):
    path = make_case_toml(tmp_path)
    (tmp_path / "Vtable.ERA5").unlink()
    with pytest.raises(ValueError, match="does not exist"):
        load_case_data(path)


def test_resolved_inputs_enumerate_every_declared_artifact(tmp_path):
    data = load_case_data(make_case_toml(tmp_path))
    records = data.resolved_inputs()
    assert all(isinstance(record, ResolvedInput) for record in records)
    roles = [record.role for record in records]
    assert roles == ["forcing", "forcing", "vtable", "wps_namelist",
                     "geog_root", "source_orography"]
    orography = records[-1]
    assert orography.detail == "variable=SOILHGT"
    declared = {data.vtable, data.wps_namelist, data.geog_root,
                data.source_orography.path, *data.forcing}
    assert {record.path for record in records} == declared


# ---------------------------------------------------------------------------
# The real74 experiment-schema twin (the bitwise A/B config).
# ---------------------------------------------------------------------------

@requires_bundle
def test_real74_exp_toml_resolves_to_the_frozen_profile_values():
    """configs/real74_d01_exp.toml is the equivalence-gate twin: its
    resolved per-domain RunConfig equals the frozen profile's
    phase3_config() (modulo the legacy case registry key), and every
    formerly implicit path/time/policy resolves to the pinned value."""
    from gpuwm.verify.cases.real74_d01 import (ETA_LEVELS, START_TIME,
                                               phase3_config)

    exp, data = load_experiment_case(REPO / "configs" / "real74_d01_exp.toml")
    assert len(exp.domains) == 1
    run = exp.domains[0].run
    # The one intended difference: the experiment path never uses the
    # legacy case registry, so run.case stays "".
    assert run.case == ""
    assert replace(run, case="real74_d01") == phase3_config()

    assert exp.start_time == START_TIME
    assert exp.run_seconds == 43200.0
    assert exp.restart_interval_s == 0.0
    assert exp.domains[0].history_interval_s == 3600.0
    assert tuple(exp.vertical.eta_levels) == tuple(
        float(value) for value in ETA_LEVELS)
    assert exp.vertical.p_top == 10000.0
    assert exp.vertical.hybrid_opt == 2 and exp.vertical.etac == 0.2

    assert data.sfcp_to_sfcp is True
    assert data.co2_vmr == 330.0e-6
    assert data.forcing_interval_s == 21600.0
    assert data.output_domain == 1
    assert data.output_title == "gpuwm Phase 3 April 3 1974 d01"
    assert data.source_orography.variable == "SOILHGT"
    assert data.source_orography_for_domain(1) is data.source_orography
    # Repo-local reference-run WPS SOILHGT (the bundle's regenerated met_em
    # SOILHGT is byte-equal to HGT_M and would silently disable sfcp_to_sfcp).
    assert data.source_orography_for_domain(1).path == (
        REPO / "configs" / "real74" / "soilhgt.d01.nc")
    for record in data.resolved_inputs():
        assert record.path.exists(), record


@requires_bundle
def test_real74_four_domain_source_orography_resolves_repo_soilhgt_per_grid():
    """Each grid resolves its repo-local reference-run WPS SOILHGT copy.

    The Downloads bundle's regenerated met_em SOILHGT is byte-equal to HGT_M;
    declaring it would silently disable the sfcp_to_sfcp source-orography
    remap that the reference real.exe actually performed, so the config
    pins the retained run-lineage copies instead.
    """
    exp, data = load_experiment_case(REPO / "configs" / "real74_4dom.toml")
    assert isinstance(data.source_orography, PerDomainSourceOrography)
    for dc in exp.domains:
        expected = (REPO / "configs" / "real74" /
                    f"soilhgt.d{dc.grid_id:02d}.nc")
        artifact = data.source_orography_for_domain(dc.grid_id)
        assert artifact == SourceOrography(expected, "SOILHGT")
        assert artifact.path.is_file()


@requires_bundle
def test_real74_exp_grid_matches_the_wps_namelist_grid():
    """The declared [projection] cross-checks against the declared WPS
    namelist and the resolved grid equals the frozen profile's."""
    from gpuwm import runtime
    from gpuwm.static.lambert import grids_from_wps_namelist

    exp, data = load_experiment_case(REPO / "configs" / "real74_d01_exp.toml")
    grid = runtime.experiment_grid(exp, data)
    legacy = grids_from_wps_namelist(
        BUNDLE / "namelists" / "namelist.wps")[0]
    for name in ("ref_lat", "ref_lon", "truelat1", "truelat2",
                 "stand_lon", "dx", "dy", "e_we", "e_sn"):
        assert getattr(grid, name) == getattr(legacy, name), name

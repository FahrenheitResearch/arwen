"""Wizard/adapter whole round trips, including task #204 and forcing-name regressions.

The emitted [case_data] must be consumed by the source adapter, its native
geometry must validate, and ERA5's forcing declaration must name the object
its provider publishes. Each emission exercises these contracts together.
"""
from __future__ import annotations

import datetime
from pathlib import Path
import tomllib

import pytest

from gpuwm.cli import main as cli_main
from gpuwm.fetch import ERA5_COMBINED_NAMES


def _run_wizard(tmp_path, source, cycle, *extra):
    out = tmp_path / "area.toml"
    rc = cli_main([
        "domain", "--point=39.7,-96.6", "--card", "16gb",
        "--ladder", "12-3", "--source", source, "--cycle", cycle,
        "--out", str(out), *extra])
    assert rc == 0, f"wizard refused its own documented {source} emission"
    return out


@pytest.mark.parametrize("source,provider,cycle", [
    ("era5", provider, "1999-05-03T12") for provider in sorted(ERA5_COMBINED_NAMES)
] + [("gfs", None, "2026-07-28T06"), ("hrrr", None, "2026-07-28T05")])
def test_wizard_emission_passes_its_adapter_and_launch_contracts(tmp_path, source, provider, cycle):
    from gpuwm.experiment import load_experiment
    extra = ("--era5-provider", provider) if provider else ()
    out = _run_wizard(tmp_path, source, cycle, *extra)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    exp = load_experiment(out)
    assert len(exp.domains) == 2
    if source in ("gfs", "era5"):
        from gpuwm.era5_direct import load_era5_adapter_config
        adapter_exp, declared = load_era5_adapter_config(out)
        if source == "era5":
            assert "case_data" in raw
            assert declared is not None, "the [case_data] table must be consumed, not dropped"
            assert declared.geog_root is not None
            assert declared.forcing_interval_s == 21600.0
        else:
            assert declared is None
        assert len(adapter_exp.domains) == 2
        from gpuwm.native_wrf_contract import validate_native_lambert_contracts
        grids = validate_native_lambert_contracts(
            adapter_exp if source == "era5" else exp,
            out.parent / "area.namelist.wps", source_name=source.upper())
        assert len(grids) == len(exp.domains) == 2
    if source == "gfs":
        from gpuwm.experiment import refuse_unrouted_perturbation, refuse_unrouted_spawn
        refuse_unrouted_perturbation(exp, "GFS-direct prepared-cache")
        refuse_unrouted_spawn(exp, "GFS-direct prepared-cache")
    elif source == "hrrr":
        from gpuwm.hrrr_route_inputs import route_input_paths
        paths = route_input_paths(out)
        for role, path in paths.items():
            assert path.is_file(), f"missing wizard-written route input {role}"
        from gpuwm.source_cli import _hrrr_domain_validation
        receipt = _hrrr_domain_validation(paths["target_domain"])
        assert receipt["status"] == "PASS", receipt["error"]
        assert receipt["window"] is not None
    else:
        from gpuwm import runplan
        from gpuwm.case_data import build_case_data
        assert raw["fetch"]["era5_provider"] == provider
        assert [Path(item).name for item in raw["case_data"]["forcing"]] == [ERA5_COMBINED_NAMES[provider]]
        data = build_case_data(raw["case_data"], source=str(out), base_dir=out.parent,
                               require_inputs=False, require_met_inputs=False)
        arguments = runplan.declared_forcing_fetch(raw, data, base_dir=out.parent)
        assert arguments is not None
        assert "--era5-provider" in arguments
        assert arguments[arguments.index("--era5-provider") + 1] == provider
        # Detached case_data must still be validated, rather than silently dropped.
        broken = tmp_path / "broken.toml"
        broken.write_text(out.read_text().replace("sfcp_to_sfcp", "sfcp_to_sfpc"),
                          encoding="utf-8")
        with pytest.raises(ValueError, match="case_data"):
            load_experiment(broken)
        if provider == "arco":
            # The shipped failure: ARCO fetch with the CDS forcing basename.
            raw["case_data"]["forcing"] = [item.replace(
                ERA5_COMBINED_NAMES["arco"], ERA5_COMBINED_NAMES["cds"])
                for item in raw["case_data"]["forcing"]]
            data = build_case_data(raw["case_data"], source=str(out), base_dir=out.parent,
                                   require_inputs=False, require_met_inputs=False)
            with pytest.raises(runplan.PlanError) as refusal:
                runplan.declared_forcing_fetch(raw, data, base_dir=out.parent)
            message = str(refusal.value)
            assert ERA5_COMBINED_NAMES["arco"] in message
            assert ERA5_COMBINED_NAMES["cds"] in message


def _minimal_experiment_raw():
    return {
        "experiment": {
            "name": "roundtrip-gate",
            "start_time": datetime.datetime(1999, 5, 3, 12),
            "run_seconds": 3600.0,
            "restart_interval_s": 0.0,
        },
        "domain": [{"grid_id": 1}],
    }



def test_a_present_companion_table_is_reported_as_unconsumed_not_absent():
    from gpuwm.experiment import build_experiment

    raw = _minimal_experiment_raw()
    raw["case_data"] = {"forcing": ["x.grib"]}
    with pytest.raises(ValueError) as refusal:
        build_experiment(raw, source="<test>")
    message = str(refusal.value)
    assert "does not have a table" not in message
    assert "case_data" in message
    assert "PRESENT" in message



def test_a_genuinely_unknown_table_keeps_the_absent_refusal():
    from gpuwm.experiment import build_experiment

    raw = _minimal_experiment_raw()
    raw["case_dtaa"] = {"forcing": ["x.grib"]}
    with pytest.raises(ValueError, match="does not have a table"):
        build_experiment(raw, source="<test>")



def test_the_catalog_emitter_takes_the_same_name_from_the_same_table():
    """`gpuwm case create` writes its own [case_data]; it must not spell
    a name the fetch table decides."""

    import inspect

    from gpuwm import case_catalog

    body = inspect.getsource(case_catalog._write_geometry_case)
    assert "era5_combined_name(" in body
    for name in ERA5_COMBINED_NAMES.values():
        assert name not in body, (
            f"{name!r} is spelled in the catalog emitter; the provider "
            "table decides it")


def test_the_research_compiler_rebinds_the_directory_and_keeps_the_name():
    """The research compiler rewrites the wizard's forcing line.  It must
    move the directory only: re-spelling the basename would overwrite a
    correct provider name with the other provider's."""

    import inspect

    from gpuwm import research_workspaces

    body = inspect.getsource(research_workspaces._final_text)
    for name in ERA5_COMBINED_NAMES.values():
        assert name not in body, (
            f"{name!r} is spelled in the research compiler; it must keep "
            "the name the emitter wrote")


#: The module and entry point that actually WRITE each provider's
#: combined file.  The table in gpuwm/fetch.py is what every emitter of a
#: [case_data] table asks; these two are what publishes the object it
#: names, and one name kept in two places is what the door died of.
_ERA5_PUBLISHERS = {"cds": ("gpuwm.era5_acquisition", "retrieve_era5"),
                    "arco": ("gpuwm.era5_arco", "retrieve_era5_arco")}


class _TransportReached(Exception):
    """Stands in for every download this cell must never perform."""


@pytest.mark.parametrize("provider", sorted(ERA5_COMBINED_NAMES))
def test_each_provider_publishes_the_file_its_own_table_row_declares(
        tmp_path, provider, monkeypatch):
    """Observed on the publisher, with no transport and no credentials.

    Both publishers look at the file they are about to write before they
    ask for a byte, and refuse to overwrite one they cannot verify as
    this request's own.  So the name a publisher works from is readable
    from that refusal: a publisher spelling a name of its own stops
    seeing the file the table declares and walks into its transport
    instead of refusing.  Both transports are stopped here, because the
    failing state of this cell is a publisher that walks past the refusal
    and asks a public archive for tens of megabytes.
    """

    import importlib

    from gpuwm import era5_acquisition, zarr_bridge

    def stop(*arguments, **keywords):
        raise _TransportReached()

    monkeypatch.setattr(zarr_bridge, "extract_regular_zarr", stop)
    monkeypatch.setattr(era5_acquisition, "_client", stop)
    assert provider in _ERA5_PUBLISHERS, (
        f"the {provider!r} row declares {ERA5_COMBINED_NAMES[provider]!r} "
        "and no publisher here is known to write it")
    module_name, function_name = _ERA5_PUBLISHERS[provider]
    publish = getattr(importlib.import_module(module_name), function_name)
    out = tmp_path / provider
    out.mkdir()
    declared = out / ERA5_COMBINED_NAMES[provider]
    declared.write_bytes(b"bytes from some earlier request")
    with pytest.raises(FileExistsError) as refusal:
        publish(cycle="1999-05-03T12", hours=6, area="30,-99,31,-98",
                out=out, progress=lambda message: None)
    assert str(declared) in str(refusal.value)
    assert declared.read_bytes() == b"bytes from some earlier request"


def _names_spelled_outside_a_docstring(module_name):
    """Every string literal in a module except its documentation.

    A publisher may DESCRIBE the file it writes in prose; what it may not
    do is keep an executable second copy of the name.
    """

    import ast
    import importlib
    from pathlib import Path

    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    documentation = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            documentation.add(id(first.value))
    return {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in documentation}


@pytest.mark.parametrize("provider", sorted(ERA5_COMBINED_NAMES))
def test_no_publisher_keeps_its_own_copy_of_the_name_it_publishes(provider):
    """The same defect one layer down from the one that shipped.

    gpuwm/era5_arco.py declared ARCO_COMBINED_NAME = "era5-combined.nc"
    while gpuwm/fetch.py's table declared the same string, so a reader
    changing the table was told by era5_combined_name's own docstring
    that the publisher followed, and it did not.  The CDS publisher never
    had the problem: it reads fetch.ERA5_COMBINED_NAME.
    """

    module_name, _ = _ERA5_PUBLISHERS[provider]
    spelled = _names_spelled_outside_a_docstring(module_name)
    for name in ERA5_COMBINED_NAMES.values():
        assert name not in spelled, (
            f"{module_name} spells {name!r}; the provider table in "
            "gpuwm/fetch.py decides that name and the publisher asks for it")


def test_an_unregistered_provider_is_refused_by_name_not_by_key_error():
    from gpuwm.fetch import era5_combined_name

    with pytest.raises(ValueError) as refusal:
        era5_combined_name("gcs")
    message = str(refusal.value)
    assert "gcs" in message
    for known in ERA5_COMBINED_NAMES:
        assert known in message

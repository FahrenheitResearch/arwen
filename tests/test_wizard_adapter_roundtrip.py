"""The wizard-emits/adapter-accepts round-trip gate (task #204).

``gpuwm domain --source X`` writes a config, and source X's OWN adapter
must accept that exact file at least through config validation.  Nothing
gated that until the 2.2.0 verification campaign ran the documented ERA5
chain end-to-end and found the door dead on arrival: the wizard emits
``[case_data]`` for ERA5, the ERA5 adapter loaded through
``gpuwm.experiment.load_experiment`` (which split off ``[fetch]`` but not
``[case_data]``), and the refusal claimed the config "does not have a
table 'case_data'" -- about a table that sat in the file, present and
valid.  Every emission here goes through the real wizard and every
acceptance through the real adapter seam, CPU-hermetic: no fetch, no
bridge, no card.

The negative controls at the bottom pin the refusal-message CLASS the
incident exposed: a table that is present but unconsumed must be
reported as a caller routing defect, never as absent, while a genuinely
unknown table keeps its "does not have a table" refusal.
"""
from __future__ import annotations

import datetime
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


# ---------------------------------------------------------------------------
# ERA5: the one-file [case_data] config through the ERA5 adapter's door.
# ---------------------------------------------------------------------------

def test_era5_wizard_config_is_accepted_by_the_era5_adapter(
        tmp_path, capsys):
    out = _run_wizard(tmp_path, "era5", "1999-05-03T12")
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert "case_data" in raw  # the shape that was refused

    # The adapter's own config door (prepare_era5_wrf loads through it).
    from gpuwm.era5_direct import load_era5_adapter_config
    exp, declared = load_era5_adapter_config(out)
    assert declared is not None, (
        "the [case_data] table must be consumed, not dropped")
    assert declared.geog_root is not None
    assert declared.forcing_interval_s == 21600.0

    # And on through the geometry contract, against the wizard's own
    # companion namelist -- the same validation prepare_era5_wrf runs.
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts
    grids = validate_native_lambert_contracts(
        exp, out.parent / "area.namelist.wps", source_name="ERA5")
    assert len(grids) == len(exp.domains) == 2


def test_era5_wizard_config_loads_through_load_experiment(tmp_path, capsys):
    """Every front door that wants only the experiment portion (fetch,
    go, stream, the profile identifier) loads through
    ``gpuwm.experiment.load_experiment``; the wizard's ERA5 emission must
    load there too, with [case_data] validated and detached."""
    out = _run_wizard(tmp_path, "era5", "1999-05-03T12")
    from gpuwm.experiment import load_experiment
    exp = load_experiment(out)
    assert len(exp.domains) == 2


def test_era5_adapter_door_accepts_the_bare_experiment_shape(
        tmp_path, capsys):
    """A config with no [case_data] (the pre-wizard adapter shape) keeps
    loading, with no declarations returned."""
    out = _run_wizard(tmp_path, "gfs", "2026-07-28T06")
    from gpuwm.era5_direct import load_era5_adapter_config
    exp, declared = load_era5_adapter_config(out)
    assert declared is None
    assert len(exp.domains) == 2


# ---------------------------------------------------------------------------
# GFS: the adapter loads the emission through load_experiment (its own
# seam, gpuwm/gfs_direct.py) and validates the geometry contract.
# ---------------------------------------------------------------------------

def test_gfs_wizard_config_is_accepted_by_the_gfs_adapter(tmp_path, capsys):
    out = _run_wizard(tmp_path, "gfs", "2026-07-28T06")
    from gpuwm.experiment import (
        load_experiment, refuse_unrouted_perturbation, refuse_unrouted_spawn)
    exp = load_experiment(out)
    refuse_unrouted_perturbation(exp, "GFS-direct prepared-cache")
    refuse_unrouted_spawn(exp, "GFS-direct prepared-cache")
    from gpuwm.native_wrf_contract import validate_native_lambert_contracts
    grids = validate_native_lambert_contracts(
        exp, out.parent / "area.namelist.wps", source_name="GFS")
    assert len(grids) == len(exp.domains) == 2


# ---------------------------------------------------------------------------
# HRRR: the route reads the wizard's companion files, not the TOML, so
# acceptance means every route input exists and passes its own reader.
# ---------------------------------------------------------------------------

def test_hrrr_wizard_outputs_are_accepted_by_the_hrrr_route(
        tmp_path, capsys):
    out = _run_wizard(tmp_path, "hrrr", "2026-07-28T05")
    from gpuwm.hrrr_route_inputs import route_input_paths
    paths = route_input_paths(out)
    for role, path in paths.items():
        assert path.is_file(), f"missing wizard-written route input {role}"
    from gpuwm.experiment import load_experiment
    exp = load_experiment(out)
    assert len(exp.domains) == 2
    # The launch-preflight coverage receipt over the wizard's own
    # target-domain document must be a PASS, not a REFUSED.
    from gpuwm.source_cli import _hrrr_domain_validation
    receipt = _hrrr_domain_validation(paths["target_domain"])
    assert receipt["status"] == "PASS", receipt["error"]
    assert receipt["window"] is not None


# ---------------------------------------------------------------------------
# The refusal-message class: present-but-unconsumed is never "absent".
# ---------------------------------------------------------------------------

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


def test_load_experiment_validates_the_case_table_it_detaches(tmp_path):
    """Detached is not dropped: a [case_data] with an unknown key must
    still refuse through load_experiment, exactly as the case loader
    itself would refuse it."""
    out = _run_wizard(tmp_path, "era5", "1999-05-03T12")
    text = out.read_text(encoding="utf-8")
    text = text.replace("sfcp_to_sfcp", "sfcp_to_sfpc")
    broken = tmp_path / "broken.toml"
    broken.write_text(text, encoding="utf-8")
    from gpuwm.experiment import load_experiment
    with pytest.raises(ValueError, match="case_data"):
        load_experiment(broken)


# ---------------------------------------------------------------------------
# The same defect class, one command later: the config the wizard writes
# must declare the forcing file its OWN [fetch] table publishes.
#
# What shipped: `gpuwm domain --source era5 --era5-provider arco` wrote
# `forcing = [".../era5-combined.grib"]` while the ARCO fetch it printed as
# step 1 publishes `era5-combined.nc`.  Both commands succeeded -- the
# download wrote real bytes -- and `gpuwm go` then refused with "[fetch].out
# does not produce the file named by [case_data].forcing".  The door was
# dead by default and the only way through was to hand-edit the generated
# TOML.  Parametrized over the provider table itself, so a third provider
# added to ERA5_COMBINED_NAMES fails here until the emitters follow it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", sorted(ERA5_COMBINED_NAMES))
def test_wizard_declares_the_forcing_file_its_own_fetch_publishes(
        tmp_path, provider):
    from pathlib import Path

    out = _run_wizard(tmp_path, "era5", "1999-05-03T12",
                      "--era5-provider", provider)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["fetch"]["era5_provider"] == provider
    declared = [Path(item).name for item in raw["case_data"]["forcing"]]
    assert declared == [ERA5_COMBINED_NAMES[provider]], (
        f"the {provider} wizard emission declares a forcing file its own "
        "printed fetch does not publish")


@pytest.mark.parametrize("provider", sorted(ERA5_COMBINED_NAMES))
def test_the_emitted_config_passes_the_launch_time_forcing_agreement(
        tmp_path, provider):
    """The gate that actually refused, run on the bare emission.

    `gpuwm go` calls this exact function before it acquires anything.  It
    returns the fetch argv when the declaration and the recipe agree, and
    raises when they do not, which is what a user met after the download.
    """

    from gpuwm import runplan
    from gpuwm.case_data import build_case_data

    out = _run_wizard(tmp_path, "era5", "1999-05-03T12",
                      "--era5-provider", provider)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    data = build_case_data(raw["case_data"], source=str(out),
                           base_dir=out.parent, require_inputs=False,
                           require_met_inputs=False)
    arguments = runplan.declared_forcing_fetch(raw, data)
    assert arguments is not None
    assert "--era5-provider" in arguments
    assert arguments[arguments.index("--era5-provider") + 1] == provider


def test_the_launch_time_refusal_names_the_file_the_provider_publishes(
        tmp_path):
    """A refusal that says two declarations disagree must say which one
    is wrong.  The provider decides what the fetch writes, so the forcing
    declaration is the side that follows, and the refusal names it."""

    from gpuwm import runplan
    from gpuwm.case_data import build_case_data

    out = _run_wizard(tmp_path, "era5", "1999-05-03T12",
                      "--era5-provider", "arco")
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    # Exactly the config that shipped: the ARCO recipe, the CDS name.
    raw["case_data"]["forcing"] = [
        item.replace(ERA5_COMBINED_NAMES["arco"], ERA5_COMBINED_NAMES["cds"])
        for item in raw["case_data"]["forcing"]]
    data = build_case_data(raw["case_data"], source=str(out),
                           base_dir=out.parent, require_inputs=False,
                           require_met_inputs=False)
    with pytest.raises(runplan.PlanError) as refusal:
        runplan.declared_forcing_fetch(raw, data)
    message = str(refusal.value)
    assert ERA5_COMBINED_NAMES["arco"] in message
    assert ERA5_COMBINED_NAMES["cds"] in message


def test_an_unregistered_provider_is_refused_by_name_not_by_key_error():
    from gpuwm.fetch import era5_combined_name

    with pytest.raises(ValueError) as refusal:
        era5_combined_name("gcs")
    message = str(refusal.value)
    assert "gcs" in message
    for known in ERA5_COMBINED_NAMES:
        assert known in message


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

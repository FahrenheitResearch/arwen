"""The chem species table: schema, loader, selection and identity.

CPU only.  Every refusal here names the breakage it prevents; each test
builds a throwaway table under ``tmp_path`` so the packaged one is never
edited to provoke a failure.
"""

from __future__ import annotations

import json
import shutil

import pytest

from gpuwm import chem_table
from gpuwm.chem_table import ChemTableError


def _copy_table(tmp_path):
    root = tmp_path / "chem"
    shutil.copytree(chem_table.CHEM_DATA_ROOT, root,
                    ignore=shutil.ignore_patterns("oracle"))
    return root


def _species_file(root, name="extra.json", rows=None):
    path = root / "species" / name
    path.write_text(json.dumps({"schema": "gpuwm.chem.species.v1",
                                "owner": "test", "rows": rows or []}),
                    encoding="utf-8")
    return path


def _row(**over):
    row = {"name": "extra_1", "output_name": "EXTRA_1",
           "long_name": "extra", "units": "ug kg-1", "phase": "aerosol",
           "family": "tracer", "sets": ["tracer_test"],
           "default_inflow": 0.0, "processes": [], "provenance": "test"}
    row.update(over)
    return row


def test_packaged_table_loads_and_the_test_set_is_one_passive_row():
    cat = chem_table.catalog()
    assert "tracer_test" in cat.sets
    table = chem_table.load_sets(["tracer_test"])
    assert table.names == ("passive_1", "passive_2")
    row = table.row("passive_1")
    assert row.transported and row.processes == ()
    assert row.state_attr == "chem_passive_1"
    assert row.time_attr == "chem0_passive_1"
    assert table.processes == ()


def test_chem_off_reads_nothing():
    class Cfg:
        chem_sets = ()
    assert chem_table.load(Cfg()) is None


def test_every_packaged_file_validates_and_is_hashed():
    cat = chem_table.catalog()
    root = chem_table.CHEM_DATA_ROOT
    on_disk = {f"{sub}/{p.name}" for sub in
               ("species", "sets", "sources", "diagnostics")
               for p in (root / sub).glob("*.json")}
    assert on_disk | {"schema.v1.json"} == set(cat.file_hashes)


def test_identity_moves_with_any_table_byte(tmp_path):
    root = _copy_table(tmp_path)
    before = chem_table.load_sets(["tracer_test"], root=root).identity
    path = root / "species" / "tracer_test.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["rows"][0]["provenance"] += " (edited)"
    path.write_text(json.dumps(doc), encoding="utf-8")
    chem_table._catalog_at.cache_clear()
    after = chem_table.load_sets(["tracer_test"], root=root).identity
    assert before != after


def test_unknown_column_is_refused(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(colour="red")])
    with pytest.raises(ChemTableError, match="colour"):
        chem_table.catalog(root)


def test_registry_array_defaults_to_chem_without_family_inference(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row()])
    row = chem_table.catalog(root).species["extra_1"]
    assert row.family == "tracer"
    assert row.wrf_array == "chem"


def test_registry_array_is_explicit_and_preserves_species_order():
    table = chem_table.load_sets(["cams_aq", "smoke", "tracer_test"])
    assert table.names == ("o3", "no2", "co", "so2", "smoke",
                           "passive_1", "passive_2")
    assert [row.wrf_array for row in table.transported] == [
        "chem", "chem", "chem", "chem", "tracer", "tracer", "tracer"]


def test_unknown_registry_array_is_refused(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(wrf_array="moist")])
    with pytest.raises(ChemTableError, match="wrf_array"):
        chem_table.catalog(root)


def test_a_species_is_one_row(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(name="passive_1", output_name="X")])
    with pytest.raises(ChemTableError, match="already defined"):
        chem_table.catalog(root)


def test_unknown_process_key_is_refused(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(processes=["emission.volcano"])])
    with pytest.raises(ChemTableError, match="not a process key"):
        chem_table.catalog(root)


def test_row_naming_a_set_without_a_file_is_refused(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(sets=["nowhere"])])
    with pytest.raises(ChemTableError, match="has no file"):
        chem_table.catalog(root)


def test_gas_row_needs_a_molar_mass_for_the_ledger(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(phase="gas", units="ppmv")])
    with pytest.raises(ChemTableError, match="molar_mass_g_mol"):
        chem_table.catalog(root)


def test_two_rows_cannot_write_one_output(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(output_name="PASSIVE_1")])
    with pytest.raises(ChemTableError, match="already written"):
        chem_table.catalog(root)


def test_unknown_set_and_duplicate_selection_are_refused():
    with pytest.raises(ChemTableError, match="unknown set"):
        chem_table.load_sets(["not_a_set"])
    with pytest.raises(ChemTableError, match="twice"):
        chem_table.load_sets(["tracer_test", "tracer_test"])


def test_rows_for_selects_by_the_processes_column(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(processes=["mixing.vertmx"])])
    table = chem_table.load_sets(["tracer_test"], root=root)
    assert [r.name for r in table.rows_for("mixing.vertmx")] == ["extra_1"]
    assert table.processes == ("mixing.vertmx",)
    with pytest.raises(KeyError):
        table.rows_for("not.a.key")


def test_process_sets_gate_a_process_to_the_named_sets(tmp_path):
    """One species shared by two sets with different physics stays one row:
    its gated process acts only when one of the gate's sets is selected."""
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(
        sets=["tracer_test", "tracer_mixing"],
        processes=["mixing.vertmx", "wetdep.ls"],
        process_sets={"wetdep.ls": ["tracer_mixing"]})])
    alone = chem_table.load_sets(["tracer_test"], root=root)
    assert [r.name for r in alone.rows_for("mixing.vertmx")] == ["extra_1"]
    assert alone.rows_for("wetdep.ls") == ()
    assert "wetdep.ls" not in alone.processes
    both = chem_table.load_sets(["tracer_test", "tracer_mixing"], root=root)
    assert "extra_1" in [r.name for r in both.rows_for("wetdep.ls")]
    assert "wetdep.ls" in both.processes


def test_the_packaged_so2_row_is_oxidized_only_by_the_gocart_sulfur_sets():
    """cams_aq carries SO2 without chemistry; GOCART's sulfur sets oxidize it;
    every set deposits it through Wesely (one row, gated by process_sets)."""
    cams = chem_table.load_sets(["cams_aq"])
    assert "chem.sulfur" not in cams.processes
    assert "so2" in [r.name for r in cams.rows_for("drydep.wesely")]
    lite = chem_table.load_sets(["gocart_lite"])
    assert [r.name for r in lite.rows_for("chem.sulfur")] == ["so2", "sulf"]
    assert [r.name for r in lite.rows_for("drydep.wesely")] == ["so2"]
    assert chem_table.catalog().species["so2"].sets == (
        "gocart_simple", "gocart_lite", "cams_aq")


@pytest.mark.parametrize("gate,message", [
    ({"settling.gocart": ["tracer_test"]}, "do not name"),
    ({"mixing.vertmx": ["tracer_mixing"]}, "are not the row's sets"),
])
def test_a_process_gate_that_could_never_act_is_refused(tmp_path, gate, message):
    root = _copy_table(tmp_path)
    _species_file(root, rows=[_row(processes=["mixing.vertmx"],
                                   process_sets=gate)])
    chem_table._catalog_at.cache_clear()
    with pytest.raises(ChemTableError, match=message):
        chem_table.catalog(root)


@pytest.mark.parametrize("sets", [
    ["smoke"], ["dust"], ["gocart_primary"], ["gocart_lite"], ["gocart_simple"],
    ["cams_aq"], ["smoke", "gocart_primary"], ["cams_aq", "gocart_lite"],
    ["smoke", "gocart_lite", "cams_aq"],
])
def test_every_packaged_set_and_the_program_combinations_load_as_one_table(sets):
    """The four lanes' rows are one catalog: each packaged set, and the
    combinations a user runs together, select without a duplicate species,
    an unresolved aging target or an unknown process key."""
    table = chem_table.load_sets(sets)
    assert len(set(table.names)) == len(table.names)
    assert set(table.processes) <= set(chem_table.CHEM_PROCESS_MODULES)
    assert all(chem_table.process_in_build(key) for key in table.processes)


def test_arena_order_is_first_appearance_over_sets_then_files(tmp_path):
    root = _copy_table(tmp_path)
    _species_file(root, "aaa.json", rows=[_row(name="early", output_name="E")])
    table = chem_table.load_sets(["tracer_test"], root=root)
    assert table.names == ("early", "passive_1", "passive_2")

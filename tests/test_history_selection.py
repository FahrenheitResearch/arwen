"""The ``[output]`` history-variable surface: grammar, presets, refusals.

Storage is the breakage this feature prevents.  A four-domain run writing
the full history inventory every period fills a disk with fields nobody
reads; WRF's own answer is ``iofields_filename``, and this is gpuwm's.

Every test here is about the SURFACE -- what a config may say, what it
refuses, and what a refusal tells the reader.  The writer-side proof (a
trimmed frame really is smaller) lives in ``tests/test_wrfout.py``, and
the artifact proof is a real sim.
"""
from __future__ import annotations

import pytest

from gpuwm.io import history_selection as hs


# ---------------------------------------------------------------- vocabulary

def test_vocabulary_is_the_union_of_the_schema_tables():
    """The valid names are a TABLE UNION, never a hand-kept list.

    A new physics scheme adds its rows to the wrfout schema and joins the
    vocabulary for free -- the arbitrary-model test applied to variables.
    """
    from gpuwm.io.wrf_output_schema import (HISTORY_FIELDS_BY_NETCDF_NAME,
                                            REGISTRY_VAR_META)

    assert set(HISTORY_FIELDS_BY_NETCDF_NAME) <= hs.HISTORY_VOCABULARY
    assert set(REGISTRY_VAR_META) <= hs.HISTORY_VOCABULARY
    # The time coordinate WRF tooling reads is part of the vocabulary too,
    # and it is structural, so it can be named but never dropped.
    assert {"XTIME", "ITIMESTEP", "Times"} <= hs.HISTORY_VOCABULARY


def test_the_writer_binds_the_same_table_the_vocabulary_reads():
    """One table, two names -- not two tables that agree today.

    ``gpuwm.io.wrfout._VAR_META`` is the name the writer has always used
    and the one every reader of the writer knows.  It must stay the SAME
    object as the schema module's, or the vocabulary would accept names
    the writer cannot describe (and refuse names it can) the first time
    somebody edited one copy.
    """
    from gpuwm.io.wrf_output_schema import REGISTRY_VAR_META
    from gpuwm.io import wrfout

    assert wrfout._VAR_META is REGISTRY_VAR_META


def test_reading_a_config_does_not_import_the_forecast_output_writer():
    """The boundary that keeps the preprocessing wheel importable.

    ``HISTORY_VOCABULARY`` is built AT MODULE SCOPE, and this module is
    what the config loader reads, so anything this module imports is
    imported by every route that merely reads a ``.toml``.
    ``gpuwm.io.wrfout`` is the forecast output side: it imports
    ``gpuwm.supervisor``, ``netCDF4`` and the runtime behind them, and
    ``tools/build_rw_wps_release.py`` names it in
    ``_FORBIDDEN_STAGED_FILES`` -- the RW-WPS preprocessing wheel does
    not ship it.

    MEASURED, and this is the breakage: while the Registry metadata
    table lived in the writer, this module reached it through a
    function-local import that module scope then CALLED.  Function-local
    made the staging AST scan blind to it, and the call made the import
    happen anyway, so ``import gpuwm.era5_direct`` -- a preprocessing
    entry point -- pulled the forecast output side in through
    case_data -> experiment -> history_selection -> wrfout ->
    gpuwm.supervisor, and the wheel's no-CuPy import proof went red.

    Asserted on the ARTIFACT, in a fresh interpreter, rather than on the
    AST: the defect was an import that no static reading of this file
    could see.
    """

    import subprocess
    import sys

    probe = (
        "import sys\n"
        "import gpuwm.io.history_selection as hs\n"
        "assert hs.HISTORY_VOCABULARY\n"
        "leaked = sorted(name for name in sys.modules\n"
        "                if name in ('gpuwm.io.wrfout', 'gpuwm.supervisor',\n"
        "                            'gpuwm.runtime', 'netCDF4'))\n"
        "assert not leaked, leaked\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_structural_fields_are_inside_the_vocabulary():
    assert hs.STRUCTURAL_FIELDS <= hs.HISTORY_VOCABULARY


# -------------------------------------------------------------------- grammar

def test_absent_output_table_is_the_full_default():
    """A config that says nothing writes everything it produces."""
    selection = hs.HistorySelection.from_mapping(None)
    assert selection is hs.FULL
    assert selection.preset == "full"
    assert selection.writes_everything


def test_full_preset_keeps_every_produced_name():
    produced = ("T", "U", "V", "QCLOUD", "T2", "XLAT", "XLONG")
    assert hs.FULL.select(produced) == tuple(produced)
    assert hs.FULL.dropped(produced) == ()


def test_history_drop_removes_exactly_the_named_fields():
    selection = hs.HistorySelection.from_mapping(
        {"history_drop": ["QCLOUD", "QRAIN"]})
    produced = ("T", "U", "QCLOUD", "QRAIN", "T2", "XLAT", "XLONG")
    assert selection.select(produced) == ("T", "U", "T2", "XLAT", "XLONG")
    assert selection.dropped(produced) == ("QCLOUD", "QRAIN")


def test_history_vars_keeps_the_named_fields_plus_the_structural_ones():
    """An include list never has to remember the georeference.

    Naming ``T2`` and getting a file with no XLAT/XLONG is a file nobody
    can plot, so the structural set rides along and the list stays about
    the science the user wants.
    """
    selection = hs.HistorySelection.from_mapping({"history_vars": ["T2"]})
    produced = ("T", "U", "T2", "XLAT", "XLONG", "ZNU", "ZNW", "P_TOP",
                "MUB", "PHB", "HGT", "XTIME", "ITIMESTEP")
    kept = selection.select(produced)
    assert "T2" in kept
    assert "U" not in kept
    assert {"XLAT", "XLONG", "ZNU", "ZNW", "P_TOP", "MUB", "PHB", "HGT",
            "XTIME", "ITIMESTEP", "T"} <= set(kept)


def test_select_preserves_the_produced_order():
    """Variable-creation order is the file's name-heap order.

    Two files with the same numbers in a different order hash
    differently while every variable in them compares equal, so the
    filter may only REMOVE names, never reorder them.
    """
    selection = hs.HistorySelection.from_mapping({"history_drop": ["U"]})
    produced = ("W", "T", "U", "T2", "XLAT")
    assert selection.select(produced) == ("W", "T", "T2", "XLAT")


# -------------------------------------------------------------------- presets

def test_named_presets_are_reachable_and_full_is_the_default():
    assert set(hs.HISTORY_PRESETS) == {"full", "minimal", "severe"}
    assert hs.HistorySelection.from_mapping({}).preset == "full"


def test_minimal_keeps_the_surface_products_and_drops_the_3d_volumes():
    selection = hs.HistorySelection.from_mapping({"preset": "minimal"})
    produced = ("T", "U", "V", "W", "QVAPOR", "QCLOUD", "REFL_10CM",
                "T2", "U10", "V10", "RAINNC", "PSFC", "XLAT", "XLONG")
    kept = set(selection.select(produced))
    assert {"T2", "U10", "V10", "RAINNC", "PSFC", "XLAT", "XLONG"} <= kept
    assert not ({"U", "V", "W", "QVAPOR", "QCLOUD", "REFL_10CM"} & kept)
    # T is structural, MEASURED: without it rw_wrfbatch refuses the whole
    # file ("not a supported post-processed WRF archive"), so `minimal`
    # would be selling a tape nothing in the estate can open.
    assert "T" in kept


def test_theta_is_structural_because_the_file_will_not_open_without_it():
    assert "T" in hs.STRUCTURAL_FIELDS
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping({"history_drop": ["T"]})
    assert "'T'" in str(excinfo.value)


def test_every_preset_keeps_the_structural_set():
    for name in hs.HISTORY_PRESETS:
        selection = hs.HistorySelection.from_mapping({"preset": name})
        produced = tuple(sorted(hs.HISTORY_VOCABULARY))
        assert hs.STRUCTURAL_FIELDS <= set(selection.select(produced)), name


def test_severe_keeps_the_storm_volumes_minimal_drops():
    selection = hs.HistorySelection.from_mapping({"preset": "severe"})
    produced = ("T", "U", "V", "W", "QVAPOR", "QCLOUD", "QRAIN",
                "REFL_10CM", "T2", "U10", "V10", "XLAT", "XLONG",
                "EXCH_H", "TSQ")
    kept = set(selection.select(produced))
    assert {"T", "U", "V", "W", "QVAPOR", "REFL_10CM", "T2"} <= kept
    # The restart-only PBL carriers no product reads are what it sheds.
    assert not ({"EXCH_H", "TSQ"} & kept)


def test_a_preset_may_be_trimmed_further_by_history_drop():
    selection = hs.HistorySelection.from_mapping(
        {"preset": "severe", "history_drop": ["QRAIN"]})
    produced = ("QVAPOR", "QRAIN", "T2", "XLAT")
    assert "QRAIN" not in selection.select(produced)
    assert "QVAPOR" in selection.select(produced)


# ------------------------------------------------------------------- refusals

def test_history_vars_and_history_drop_together_refuse():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping(
            {"history_vars": ["T2"], "history_drop": ["QCLOUD"]})
    message = str(excinfo.value)
    assert "history_vars" in message and "history_drop" in message
    # The refusal must name the remedy, not merely the fault.
    assert "one" in message.lower()


def test_preset_and_history_vars_together_refuse():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping(
            {"preset": "minimal", "history_vars": ["T2"]})
    assert "history_vars" in str(excinfo.value)
    assert "preset" in str(excinfo.value)


def test_unknown_variable_name_refuses_and_lists_the_valid_set():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping({"history_drop": ["QCLOUDD"]})
    message = str(excinfo.value)
    assert "QCLOUDD" in message
    # A real neighbour is suggested, and the whole vocabulary is reachable.
    assert "QCLOUD" in message
    assert "gpuwm render --list-products" in message or "QVAPOR" in message


def test_unknown_preset_refuses_and_lists_the_presets():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping({"preset": "tiny"})
    message = str(excinfo.value)
    assert "tiny" in message
    assert "minimal" in message and "severe" in message and "full" in message


def test_dropping_a_structural_field_refuses_by_name():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping({"history_drop": ["XLAT"]})
    message = str(excinfo.value)
    assert "XLAT" in message
    # It must say WHAT breaks, not merely that it is not allowed.
    assert "structural" in message.lower() or "readable" in message.lower()


def test_unknown_key_refuses():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping({"history_variables": ["T2"]})
    assert "history_variables" in str(excinfo.value)
    assert "history_vars" in str(excinfo.value)


def test_output_table_must_be_a_table():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping(["T2"])
    assert "[output]" in str(excinfo.value)


def test_empty_history_vars_refuses_rather_than_writing_a_bare_skeleton():
    with pytest.raises(ValueError) as excinfo:
        hs.HistorySelection.from_mapping({"history_vars": []})
    assert "history_vars" in str(excinfo.value)


# --------------------------------------------------- product dependency check

def test_product_inputs_table_covers_the_shared_product_names():
    from gpuwm.render import PRODUCTS

    assert set(PRODUCTS) <= set(hs.PRODUCT_HISTORY_INPUTS)


def test_lost_products_names_which_product_dies_of_which_variable():
    lost = hs.lost_products(("REFL_10CM",))
    assert "refl" in lost
    assert "REFL_10CM" in lost["refl"]
    # A product whose inputs survive is not reported.
    assert "t2" not in lost


def test_an_either_or_requirement_survives_one_of_its_options():
    """``RAINC|RAINNC`` means either satisfies it."""
    assert "precip" not in hs.lost_products(("RAINC",))
    assert "precip" in hs.lost_products(("RAINC", "RAINNC"))


def test_dependency_warning_prints_the_product_to_variable_mapping(capsys):
    selection = hs.HistorySelection.from_mapping(
        {"history_drop": ["REFL_10CM"]})
    selection.warn_lost_products(("REFL_10CM", "T2", "XLAT"), where="d01")
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "refl" in err and "REFL_10CM" in err
    assert "d01" in err


def test_a_preset_warning_counts_the_drops_instead_of_listing_hundreds(capsys):
    """A wall of 150 names is a warning nobody reads.

    MEASURED: ``preset = "minimal"`` on a real two-domain run printed
    every dropped carrier -- well over a thousand characters of scheme
    state -- and buried the one actionable half, which is WHICH PRODUCT
    dies of WHICH variable.
    """
    selection = hs.HistorySelection.from_mapping({"preset": "minimal"})
    selection.warn_lost_products(hs.HISTORY_VOCABULARY, where="d02")
    err = capsys.readouterr().err
    assert "refl" in err and "REFL_10CM" in err
    assert "d02" in err
    assert len(err) < 500, err
    # It still says HOW MANY, so the reader knows the scale of the trim.
    assert "variables" in err


def test_no_warning_when_nothing_a_product_needs_was_dropped(capsys):
    selection = hs.HistorySelection.from_mapping({"history_drop": ["QKE"]})
    selection.warn_lost_products(("QKE", "T2", "XLAT"), where="d01")
    assert capsys.readouterr().err == ""


# ------------------------------------------------------------- receipt/attrs

def test_selection_stamps_what_it_did_into_the_wrfout_attributes():
    """A trimmed file must SAY it was trimmed, by name.

    Without this the render front door can only report a variable as
    absent; with it, it can report that this run dropped it and how to
    get it back.
    """
    selection = hs.HistorySelection.from_mapping(
        {"preset": "severe", "history_drop": ["QRAIN"]})
    attrs = selection.wrfout_attrs(("QVAPOR", "QRAIN", "EXCH_H", "T2"))
    assert attrs["GPUWM_HISTORY_PRESET"] == "severe"
    assert "QRAIN" in attrs["GPUWM_HISTORY_DROPPED"]
    assert "EXCH_H" in attrs["GPUWM_HISTORY_DROPPED"]


def test_full_selection_stamps_nothing_so_default_files_keep_their_header():
    assert hs.FULL.wrfout_attrs(("T2", "XLAT")) == {}


# ------------------------------------------------------- assembled frames

def test_apply_trims_an_assembled_frame_and_returns_its_attributes():
    """The synchronous writer's seam (``gpuwm downscale``'s child)."""
    selection = hs.HistorySelection.from_mapping(
        {"history_drop": ["QCLOUD"]})
    frame = {"T": 1, "QCLOUD": 2, "T2": 3, "XLAT": 4}
    kept, attrs = selection.apply(frame)
    assert set(kept) == {"T", "T2", "XLAT"}
    assert attrs["GPUWM_HISTORY_DROPPED"] == "QCLOUD"
    # The caller's frame is not mutated.
    assert "QCLOUD" in frame


def test_apply_of_a_full_default_returns_the_frame_object_untouched():
    frame = {"T": 1, "T2": 2}
    kept, attrs = hs.FULL.apply(frame)
    assert kept is frame
    assert attrs == {}

"""``gpuwm check`` prices the forcing decode's HOST RAM, and says INCOMPLETE.

THE DEFECT, reproduced on the published 2.6.0 wheel and on 2.6.5.  A user
ran a three-domain ERA5 forecast, `gpuwm run` printed the single word
``Terminated`` and nothing else, and the `gpuwm check` they ran afterwards
had said this:

    gpuwm input preflight: PASS
      BINDING PHASE: the forecast is the memory-binding phase at 23.06 GiB
                     peak envelope; it fits the 31.50 GiB budget with
                     8.44 GiB to spare
      alloc_fits_vram_budget:        not measured
      alloc_measured_le_estimate:    not measured
      alloc_estimate_le_vram_budget: PASS
    $ echo $?
    0

Two things are wrong with that page and both are here.

FIRST, EVERY FIGURE ON IT IS DEVICE MEMORY.  ``IngestMemoryEstimate``
opens by saying so -- "Itemized device memory for the preprocessing phase
of one TREE" -- and the phase it prices decodes the forcing to float64 on
the SOURCE grid, 204 two-dimensional fields per valid time for ERA5, and
holds two copies of every one of them at once while the case is prepared
(the decoder's arrays under ``_decode_era5_forcing_partials_resolved``
and the frozen snapshot's copies under ``_decode_era5_gribs_resolved``,
both ``@lru_cache``).  On a global 0.25 degree field over eight valid times
that is 25.25 GiB of RAM that no door in this product had ever weighed.
A host allocation that does not fit is refused by the kernel or reaped by
a watchdog, which kills the process from OUTSIDE: no traceback, nothing
for gpuwm to print, one word from the shell.

SECOND, ONE OF THREE GATES IS NOT A VERDICT.  ``evaluate_alloc_gates`` is
scrupulous -- ``None`` for every leg it could not evaluate, and its own
docstring says "a missing measurement can never report a pass".  The
report then dropped the ``None`` legs and took ``all()`` over what was
left, and ``all([True])`` is ``all([True, True, True])``.  The memory
section printed no verdict word of its own at all, so the only one on the
page came from ``gpuwm/ingest/preflight.py`` about nineteen CPU file, time
and table checks, none of which is about memory.

THE CONTROL IS THE OTHER HALF of both: a chain whose legs were all
measured still reads PASS, and a decode that fits still exits 0.

The data is stubbed and the logic is not: these tests drive the real
``gpuwm check`` front door, the real handoff between its two halves and
the real arithmetic, with a stand-in for the decoded catalog that only a
multi-gigabyte GRIB could otherwise supply.
"""
from __future__ import annotations

import json
import sys
import textwrap
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

from gpuwm import cli
from gpuwm.core import preflight as memory_preflight
from gpuwm.ingest import preflight as input_preflight


GIB = 1024 ** 3

#: A global ERA5 0.25 degree analysis grid, which is what the CDS hands
#: back when no ``--area`` narrows the retrieval.
_GLOBAL_ERA5_SHAPE = (721, 1440)

#: Eight valid times in the forcing files. The forecast horizon needs
#: two; preparation retains every contiguous time it finds and never sees
#: ``run_seconds``, which is the whole reason these two numbers differ.
_DECODED_VALID_TIMES = 8


def _era5_case_config(tmp_path, name="hostmem"):
    """One small domain with a ``[case_data]`` table, as a user writes it.

    The domain is deliberately tiny: every device figure fits any card, so
    nothing here can pass or fail for a VRAM reason and the host term is
    isolated.
    """
    path = tmp_path / f"{name}.toml"
    path.write_text(textwrap.dedent("""\
        [experiment]
        name = "hostmem"
        start_time = 2024-05-03T12:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [shared]
        nz = 49
        ztop = 20000.0
        moist = true
        moist_cq = true
        mp_physics = 10
        ra_lw_physics = 4
        ra_sw_physics = 4
        sf_sfclay_physics = 91
        sf_surface_physics = 2
        bl_pbl_physics = 1
        cu_physics = 1
        nwp_diagnostics = 1

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 200
        ny = 200
        time_step = 60
        dx = 12000.0
        history_interval_s = 3600.0

        [case_data]
        forcing = ["era5-combined.grib"]
        vtable = "Vtable.ERA5_CDO"
        forcing_interval_s = 21600.0
        sfcp_to_sfcp = true
        wps_namelist = "hostmem.namelist.wps"
        geog_root = "WPS_GEOG"
        output_domain = 1
        output_title = "host memory fixture"
        """), encoding="utf-8")
    return path


def _gfs_sourced_case_config(tmp_path, name="gfscase"):
    """A ``[case_data]`` config whose ``[fetch]`` names a different road."""
    path = tmp_path / f"{name}.toml"
    body = _era5_case_config(tmp_path, name=f"{name}-src").read_text(
        encoding="utf-8")
    # BEFORE [case_data], because a TOML table swallows every key under it.
    fetch = textwrap.dedent("""\
        [fetch]
        source = "gfs"
        cycle = "2024-05-03T12"
        hours = 1

        """)
    path.write_text(body.replace("[case_data]", fetch + "[case_data]", 1),
                    encoding="utf-8")
    return path


def _fetch_only_config(tmp_path, name="fetchonly"):
    """The same domain with a ``[fetch]`` hint and no ``[case_data]``."""
    path = tmp_path / f"{name}.toml"
    source = _era5_case_config(tmp_path, name=f"{name}-src").read_text(
        encoding="utf-8")
    body = source.split("[case_data]")[0]
    path.write_text(body + textwrap.dedent("""\
        [fetch]
        source = "era5"
        cycle = "2024-05-03T12"
        hours = 1
        """), encoding="utf-8")
    return path


class _StubCoverage:
    def __init__(self, shape):
        self.shape = shape


class _StubField:
    """A decoded array as the geometry read sees it: a shape, no bytes.

    A real ERA5 valid time on the grid below is 1.58 GiB; two hundred of
    them would make this a memory test of its own.  Only ``shape`` is
    read, and reading only the shape is the point -- the count is
    horizontal slices, not variables.
    """

    def __init__(self, shape):
        self.shape = shape


class _StubSnapshot:
    def __init__(self, fields):
        self.fields = fields


class _StubCatalog:
    """The decoded inventory and actual timestamps needed by memory sizing.

    ``snapshots`` carries the DECODED inventory, which is where the field
    count comes from: the nominal table in ``gpuwm/core/preflight.py``
    describes the standard 37-level ERA5 retrieval and a legal config may
    decode far less.
    """

    def __init__(self, shape, valid_times, levels=37, surface=19,
                 level_fields=5):
        self.spatial_coverage = _StubCoverage(shape)
        self.valid_times = tuple(
            datetime(2024, 5, 3, 12) + timedelta(hours=6 * index)
            for index in range(valid_times))
        fields = {f"L{index}": _StubField((levels, *shape))
                  for index in range(level_fields)}
        fields.update({f"S{index}": _StubField(shape)
                       for index in range(surface)})
        self.snapshots = (_StubSnapshot(fields),)


class _StubReport:
    def __init__(self, catalog):
        self.catalog = catalog
        self.ok = True

    def format(self):
        return "gpuwm input preflight: PASS"


def _stub_the_decoded_inputs(monkeypatch, *, shape=_GLOBAL_ERA5_SHAPE,
                             valid_times=_DECODED_VALID_TIMES, **inventory):
    """Stand in for the multi-gigabyte GRIB, and for nothing else.

    ``preflight_report`` is what decodes the forcing, so a test that could
    not replace it would have to ship the forcing.  Everything downstream
    of it -- the handoff onto ``args``, the geometry read, the arithmetic,
    the comparison and the exit code -- is the real code.

    ``load_experiment_case`` is replaced with a delegation rather than a
    stub value because ``check_main`` loads the experiment through it too,
    and a ``[case_data]`` table naming files that do not exist is exactly
    the part this fixture is standing in for.
    """
    def _load(path, **_kwargs):
        from gpuwm.experiment import build_experiment

        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        # The one-file case schema the real loader splits: the experiment
        # builder takes neither table.
        raw.pop("case_data", None)
        raw.pop("fetch", None)
        return build_experiment(raw, source=str(path)), object()

    catalog = _StubCatalog(shape, valid_times, **inventory)
    monkeypatch.setattr("gpuwm.case_data.load_experiment_case", _load)
    monkeypatch.setattr(input_preflight, "preflight_report",
                        lambda exp, case_data: _StubReport(catalog))
    return catalog


def _check(capsys, config, *flags):
    # Host admission is isolated from compilation; its failure controls
    # live in test_check_gpu_readiness.py.
    from unittest.mock import patch
    from gpuwm.doctor import Check
    with patch("gpuwm.doctor._cuda_headers_check", return_value=Check(
            "CUDA kernel headers", "verified", "fixture kernels ready")):
        code = cli.main(["check", str(config), "--explain", *flags])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# the verdict the memory section never had
# ---------------------------------------------------------------------------

def test_the_memory_section_prints_its_own_verdict_and_it_is_incomplete(
        tmp_path, capsys, monkeypatch):
    """One of three legs evaluated is INCOMPLETE, said in this section's
    own words rather than left to the PASS a different module printed
    about a different subject."""
    _stub_the_decoded_inputs(monkeypatch)
    config = _era5_case_config(tmp_path)
    _code, out, _err = _check(capsys, config, "--budget-gib", "13",
                              "--vram-gib", "16")
    headline = [line for line in out.splitlines()
                if line.startswith("gpuwm memory preflight:")]
    assert headline, "the memory section printed no verdict of its own"
    assert "INCOMPLETE" in headline[0]
    assert "1 of 3 allocation gates evaluated" in headline[0]


def test_an_absent_leg_is_named_beside_the_flag_that_would_measure_it(
        tmp_path, capsys, monkeypatch):
    """``not measured`` was the entire message: no leg named, no remedy,
    and no sentence saying that nothing allocated anything."""
    _stub_the_decoded_inputs(monkeypatch)
    config = _era5_case_config(tmp_path)
    _code, out, _err = _check(capsys, config, "--budget-gib", "13",
                              "--vram-gib", "16")
    assert "INCOMPLETE: 2 of 3 legs above were not measured" in out
    assert "no allocation was attempted in this command" in out
    assert memory_preflight.gate_display_name(
        "alloc_fits_wddm_budget", vram_gib=16) in out
    assert "alloc_measured_le_estimate" in out
    assert f"gpuwm check {config} --alloc" in out
    # ...and that the surviving leg weighed a declaration, not this card.
    assert "compares an ESTIMATE against the budget you declared" in out


def test_a_chain_whose_legs_were_all_measured_still_reads_pass():
    """THE CONTROL.  The reduction gained a third value; it did not gain a
    new way to withhold a pass from a chain that earned one."""
    measured = dict.fromkeys(memory_preflight.N0_GATE_METRICS, True)
    assert memory_preflight.memory_gate_verdict(measured) == "pass"
    assert memory_preflight.absent_gate_metrics(measured) == ()


def test_a_measured_failure_outranks_an_absent_measurement():
    """A leg that FAILED is a measurement, and the harder verdict wins."""
    gates = dict.fromkeys(memory_preflight.N0_GATE_METRICS, None)
    gates["alloc_estimate_le_wddm_budget"] = False
    assert memory_preflight.memory_gate_verdict(gates) == "fail"


# ---------------------------------------------------------------------------
# the host memory nothing priced
# ---------------------------------------------------------------------------

def test_the_forcing_decode_is_priced_in_host_bytes(tmp_path):
    """The arithmetic, on its own: float64 x 204 source fields x the source
    grid x every decoded valid time x two retained copies."""
    from gpuwm.core.preflight import estimate_ingest
    from gpuwm.experiment import build_experiment

    config = _era5_case_config(tmp_path)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    raw.pop("case_data")
    exp = build_experiment(raw, source=str(config))

    points = _GLOBAL_ERA5_SHAPE[0] * _GLOBAL_ERA5_SHAPE[1]
    ingest = estimate_ingest(exp, source="era5",
                             source_grid_points=points,
                             decoded_valid_times=_DECODED_VALID_TIMES)
    assert ingest.host_fields_per_time == 204     # 37 x (3 + U + V) + 19
    assert ingest.host_forcing_bytes == 8 * 204 * points * 8 * 2
    # The published worked example, to the hundredth of a GiB.
    assert round(ingest.host_forcing_bytes / GIB, 2) == 25.25
    # ...and it is HOST memory: no device total moved.
    assert estimate_ingest(exp, source="era5").peak_envelope_bytes \
        == ingest.peak_envelope_bytes
    # 204 is the NOMINAL inventory and stands in only when no decoded
    # count is in hand.  A caller holding a catalog passes its own, and
    # that one wins.
    decoded = estimate_ingest(exp, source="era5",
                              source_grid_points=points,
                              decoded_valid_times=_DECODED_VALID_TIMES,
                              source_fields_per_time=82)
    assert decoded.host_fields_per_time == 82
    assert decoded.host_forcing_bytes == 8 * 82 * points * 8 * 2


def test_the_field_count_is_the_decoded_one_not_the_nominal_table(
        tmp_path, capsys, monkeypatch):
    """A legal config that decodes less is charged less.

    ``SOURCE_ANALYSIS_LEVELS["era5"]`` is 37 and
    ``SOURCE_ANALYSIS_SURFACE_FIELDS["era5"]`` is 19, but nothing obliges
    a config to match them: the input preflight's ``_check_levels`` asks
    only that the levels be finite, strictly increasing, reach ``p_top``
    and go down to 1000 hPa, and two of the required surface names are
    optional.  A 13-level retrieval decodes 82 fields where the table
    charges 204 -- a 2.5x over-statement on a figure that gates a
    REFUSAL, which would refuse a run that would have completed.  The
    exact count is free, because the catalog handed over by the half of
    this command that runs first IS the decoded arrays.
    """
    _stub_the_decoded_inputs(monkeypatch, levels=13, surface=17)
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: 12 * GIB, raising=False)
    config = _era5_case_config(tmp_path)
    code, out, err = _check(capsys, config, "--budget-gib", "13",
                            "--vram-gib", "16")
    points = _GLOBAL_ERA5_SHAPE[0] * _GLOBAL_ERA5_SHAPE[1]
    charged = 8 * 82 * points * _DECODED_VALID_TIMES * 2
    assert "82 source fields" in out
    assert f"{charged / GIB:.2f} GiB, held at once while the case is prepared" in out
    # 10.15 GiB fits the 12 GiB rail; the nominal 25.25 GiB would not,
    # so this is the refusal that must NOT fire.
    assert code == 0, err
    assert "REFUSED" not in err


def test_a_forcing_this_command_cannot_see_is_reported_not_priced(
        tmp_path, capsys):
    """NOT PRICED rather than a plausible number, and never by decoding
    the forcing to find out -- that spends the memory being weighed.

    ``gpuwm domain --source era5`` can emit a config with no
    ``[case_data]`` table at all; the input preflight says "not
    applicable" and hands the memory section an ERA5 ingest lane it can
    price on the device and a forcing inventory it cannot see.
    """
    config = _fetch_only_config(tmp_path)
    _code, out, _err = _check(capsys, config, "--budget-gib", "13",
                              "--vram-gib", "16")
    assert "INGEST (preprocessing, --source era5)" in out
    assert "HOST FORCING NOT PRICED" in out
    assert "GiB, held at once while the case is prepared" not in out
    assert memory_preflight.ingest_host_geometry(object()) is None


def test_a_source_whose_decoder_was_never_measured_is_not_priced(
        tmp_path, capsys, monkeypatch):
    """The retention factor is a property of ONE decoder, not of forcing in
    general.  ERA5 is the native-GRIB1 road through ``gpuwm.ingest.grib``,
    which is the road that keeps two copies; GFS and HRRR go through the
    native front door and nothing here has measured what they hold.  A
    catalog is not a licence to charge them this road's number."""
    _stub_the_decoded_inputs(monkeypatch)
    config = _gfs_sourced_case_config(tmp_path)
    _code, out, _err = _check(capsys, config, "--budget-gib", "13",
                              "--vram-gib", "16")
    assert "INGEST (preprocessing, --source gfs)" in out
    assert "HOST FORCING NOT PRICED" in out
    assert "native front door" in out
    assert "retained copies" not in out


def test_a_decode_larger_than_available_ram_is_refused(
        tmp_path, capsys, monkeypatch):
    """A REFUSAL, not a note.  Nothing else in this report goes red for
    it: every device figure on the page fits with room to spare."""
    _stub_the_decoded_inputs(monkeypatch)
    # ``raising=False`` so this test fails on the OUTCOME rather than on
    # the absence of the probe: on a build that prices no host bytes it
    # runs to completion and reports exit 0 with nothing on stderr, which
    # is the defect stated as a result.
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: 16 * GIB, raising=False)
    config = _era5_case_config(tmp_path)
    code, out, err = _check(capsys, config, "--budget-gib", "13",
                            "--vram-gib", "16")
    assert "REFUSED" in err
    assert code == memory_preflight._EXIT_HOST_MEMORY_OVER_BUDGET
    assert "25.25 GiB of forcing into HOST RAM" in err
    assert "16.00 GiB this box had available before this command ran" in err
    # The lever, and the levers that are not it.
    assert "narrow the area" in err
    assert "do not move this number" in err
    # The device side of the same page still fits -- this is the point.
    assert "it fits the" in out


def test_the_host_refusal_has_a_way_through(tmp_path, capsys, monkeypatch):
    """A refusal with no override is one that gets worked around by not
    running the check.

    ``MemAvailable`` is a reading of this second and the figure beside it
    is a floor over a decoder, so a busy workstation, a shared login node
    or a box beside another job can be refused a configuration that fits.
    ``gpuwm go`` has carried ``--no-memory-gate`` for its own memory
    refusal since it had one; this is the counterpart for the other
    budget.  The figure is still printed -- skipping the gate is not
    hiding the number.
    """
    _stub_the_decoded_inputs(monkeypatch)
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: 16 * GIB, raising=False)
    config = _era5_case_config(tmp_path)
    code, out, err = _check(capsys, config, "--budget-gib", "13",
                            "--vram-gib", "16", "--no-host-memory-gate")
    assert code == 0
    assert "REFUSED" not in err
    assert "host memory gate SKIPPED by --no-host-memory-gate" in err
    assert "25.25 GiB, held at once while the case is prepared" in out


def test_the_refusal_never_names_an_exit_code_it_will_not_return(
        tmp_path, capsys, monkeypatch):
    """1, 2 and 3 outrank this refusal and return before it.

    The paragraph says "REFUSED (exit 5)" in its first sentence, so
    printing it ahead of the harder verdicts told a reader a number the
    command was not going to hand them.  Here nothing could be measured
    and no budget was declared, so the fail-closed exit 2 wins; the host
    figure is still itemized in the report above.
    """
    _stub_the_decoded_inputs(monkeypatch)
    # Exercise unavailable device measurement on GPU-equipped hosts too.
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: 16 * GIB, raising=False)
    config = _era5_case_config(tmp_path)
    code, out, err = _check(capsys, config)
    assert code == 2, err
    assert "REFUSED (exit 2, fail-closed)" in err
    assert f"exit {memory_preflight._EXIT_HOST_MEMORY_OVER_BUDGET}" not in err
    assert "25.25 GiB, held at once while the case is prepared" in out


def test_a_decode_that_fits_available_ram_does_not_refuse(
        tmp_path, capsys, monkeypatch):
    """THE CONTROL.  Same config, same code path, more RAM."""
    _stub_the_decoded_inputs(monkeypatch)
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: 64 * GIB, raising=False)
    config = _era5_case_config(tmp_path)
    code, out, err = _check(capsys, config, "--budget-gib", "13",
                            "--vram-gib", "16")
    assert code == 0
    assert "REFUSED" not in err
    assert "25.25 GiB, held at once while the case is prepared" in out


def test_the_rail_is_read_before_this_command_decodes_the_forcing(
        tmp_path, capsys, monkeypatch):
    """The half of ``gpuwm check`` that runs first decodes the forcing into
    caches nothing clears, so this process is already holding the bytes
    being priced.  Reading the headroom afterwards would subtract them
    from the rail they are compared against and refuse a run that fits.

    The rail below is moved BY THE DECODE rather than by a call counter:
    the stand-in for ``preflight_report`` -- which is the thing that
    decodes -- drops the box from 64 GiB to 1 GiB when it runs.  That is
    the hazard itself, and it is what makes this fail on the ordering: a
    reading taken at entry sees 64 GiB and does not refuse, and a reading
    taken in the memory section sees 1 GiB and refuses a run that fits.
    """
    _stub_the_decoded_inputs(monkeypatch)
    available = [64 * GIB]
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: available[0], raising=False)
    decode = input_preflight.preflight_report

    def spend_the_ram_being_priced(exp, case_data):
        report = decode(exp, case_data)
        available[0] = 1 * GIB
        return report

    monkeypatch.setattr(input_preflight, "preflight_report",
                        spend_the_ram_being_priced)
    config = _era5_case_config(tmp_path)
    code, out, err = _check(capsys, config, "--budget-gib", "13",
                            "--vram-gib", "16")
    assert code == 0, err
    assert "64.00 GiB of RAM this box had available" in out
    assert "REFUSED" not in err


def test_the_host_figure_counts_the_files_times_not_the_forecasts(
        tmp_path, capsys, monkeypatch):
    """The decode is charged on what is in the FILES.  A user who fetched
    a wider window than they integrate pays for the whole window, and no
    edit to ``run_seconds`` reduces it."""
    _stub_the_decoded_inputs(monkeypatch)
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: 64 * GIB, raising=False)
    config = _era5_case_config(tmp_path)
    _code, out, _err = _check(capsys, config, "--budget-gib", "13",
                              "--vram-gib", "16")
    assert f"{_DECODED_VALID_TIMES} decoded valid times" in out
    assert "shortening the forecast does not shorten the decode" in out


def test_the_json_report_publishes_the_verdict_and_the_host_terms(
        tmp_path, capsys, monkeypatch):
    """A machine reader gets the same two facts as fields, not as prose."""
    _stub_the_decoded_inputs(monkeypatch)
    monkeypatch.setattr(memory_preflight, "host_available_bytes",
                        lambda: 64 * GIB, raising=False)
    config = _era5_case_config(tmp_path)
    _code, out, _err = _check(capsys, config, "--budget-gib", "13",
                              "--vram-gib", "16", "--json")
    document = json.loads(out[out.index("{"):])
    assert document["memory_verdict"] == "incomplete"
    assert document["gates_evaluated"] == 1
    assert document["gates_total"] == 3
    assert document["gates_absent"] == ["alloc_fits_wddm_budget",
                                        "alloc_measured_le_estimate"]
    points = _GLOBAL_ERA5_SHAPE[0] * _GLOBAL_ERA5_SHAPE[1]
    assert document["ingest_host_forcing_bytes"] == 8 * 204 * points * 8 * 2
    assert document["ingest_host_forcing_exceeds_available"] is False
    assert document["ingest"]["host_source_grid_points"] == points
    assert document["ingest"]["host_decoded_valid_times"] == 8
    assert document["ingest"]["host_retained_copies"] == 2


# ---------------------------------------------------------------------------
# the handoff the two halves of `gpuwm check` needed
# ---------------------------------------------------------------------------

def test_the_input_preflight_hands_its_catalog_to_the_memory_section(
        tmp_path, monkeypatch, capsys):
    """The source grid and the decoded time count live in the forcing
    files and nowhere in the TOML, and the half of ``gpuwm check`` that
    decoded them is the half that runs first."""
    catalog = _stub_the_decoded_inputs(monkeypatch)
    config = _era5_case_config(tmp_path)

    class _Args:
        pass

    args = _Args()
    args.config = config
    assert input_preflight._check_command(args) == 0
    capsys.readouterr()
    assert args.input_catalog is catalog
    points = _GLOBAL_ERA5_SHAPE[0] * _GLOBAL_ERA5_SHAPE[1]
    assert memory_preflight.ingest_host_geometry(args) == (
        points, _DECODED_VALID_TIMES, 204)

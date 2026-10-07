"""A masked analyzed aerosol pair is withheld by the bridge and named by the run.

NCEP sometimes publishes PMTF (QNWFA, GRIB 0/13/193) on HRRR hybrid level
1 with a GRIB2 bitmap (2026-10-04 12Z, 2026-10-07 07Z among the cycles
sampled).  Since 608b9ac47 the native bridge selected the pair whenever a
file published it, and every selected field refuses a bitmap, so a plain
`gpuwm go` of an HRRR domain stopped at prepare on those cycles ("selected
initialization field unexpectedly carries a bitmap") although it never
asked for the pair.  The bridge now withholds a masked pair (2.8.6's
selection for that cycle) and says so in its gate; a run that DOES ask for
the analyzed pair gets the existing named refusal with the reason.  The
Rust half is held by the bridge's own cargo tests.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest.native_supplements import (
    gate_optional_hybrid_fields, gate_withheld_optional_hybrid_fields)

REASON = ("f00 QNWFA hybrid level 1 (message 600) carries a GRIB2 bitmap "
          "(2 of 1905141 points masked); no fill policy for masked points "
          "exists, so the pair is not read")
WITHHELD = {"optional_hybrid_withheld": "QNWFA,QNIFA",
            "optional_hybrid_withheld_reason": REASON}


def test_the_gate_names_a_withheld_pair_whole_with_its_reason():
    assert gate_withheld_optional_hybrid_fields({}) == {}
    assert gate_withheld_optional_hybrid_fields(WITHHELD) == {
        "QNWFA": REASON, "QNIFA": REASON}
    # Withheld is not declared: the payload count stays the 561 records.
    assert gate_optional_hybrid_fields(WITHHELD) == ()
    for gate in ({"optional_hybrid_withheld": "QNWFA",
                  "optional_hybrid_withheld_reason": REASON},
                 {"optional_hybrid_withheld": "QNWFA,QNIFA"},
                 {"optional_hybrid_withheld_reason": REASON},
                 dict(WITHHELD, optional_hybrid_fields="QNWFA,QNIFA",
                      optional_hybrid_units="QNWFA=kg-1,QNIFA=kg-1")):
        with pytest.raises(ValueError, match="withh"):
            gate_withheld_optional_hybrid_fields(gate)


def _publication(root: Path, gate_extra: dict[str, str]) -> str:
    """A sealed two-lead bridge publication on a 2 x 3 window."""
    from gpuwm.ingest.hrrr import _ATMOSPHERE_2D, _ATMOSPHERE_3D, _SOIL_3D

    ny, nx = 2, 3
    gate = {
        "status": "PASS",
        "cycle": "2026-10-07 07:00:00",
        "forecast_hours": "0,1",
        "series_count": "2",
        "atmosphere_selected_per_time": "561",
        "hybrid_levels": "50",
        **gate_extra,
        "soil_selected_per_time": "18",
        "window_zero_based_inclusive": "i=0..2 j=0..1",
        "window_shape": f"{ny}x{nx}",
        "qice_mapping": ("PASS discipline=0 category=1 parameter=82 "
                         "level_type=105; finite/nonnegative/nonzero"),
        "cross_time_inventory": "PASS exact selected keys/levels/grid",
    }
    root.mkdir()
    (root / "gate.txt").write_text(
        "".join(f"{key}\t{value}\n" for key, value in gate.items()))
    payloads = []
    for hour in (0, 1):
        for role, names, levels in (("atmosphere", _ATMOSPHERE_3D, 50),
                                    ("atmosphere", _ATMOSPHERE_2D, None),
                                    ("soil", _SOIL_3D, 9)):
            directory = root / f"{role}-f{hour:02d}"
            directory.mkdir(exist_ok=True)
            shape = (ny, nx) if levels is None else (levels, ny, nx)
            for name in names:
                path = directory / f"{name}.f32le"
                np.full(shape, 0.5, dtype="<f4").tofile(path)
                payloads.append(path)
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}\n" for path in payloads))
    return hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()


def test_a_withheld_pair_reaches_the_snapshot_and_is_not_mapped(tmp_path):
    from gpuwm.ingest.hrrr import load_hrrr_native_window

    sealed = _publication(tmp_path / "bridge", WITHHELD)
    snapshot = load_hrrr_native_window(
        tmp_path / "bridge", 0, expected_manifest_sha256=sealed)
    assert dict(snapshot.withheld_fields) == {"QNWFA": REASON,
                                               "QNIFA": REASON}
    assert "QNWFA" not in snapshot.fields and "QNIFA" not in snapshot.fields
    # A cycle that never published the pair withholds nothing.
    plain = _publication(tmp_path / "plain", {})
    snapshot = load_hrrr_native_window(
        tmp_path / "plain", 1, expected_manifest_sha256=plain)
    assert dict(snapshot.withheld_fields) == {}


def test_the_mapped_snapshots_carry_the_withheld_reason():
    """The start state, and both boundary-strip copies the HRRR route
    hands its workers, keep the reason the initializer's refusal reads."""
    import inspect
    from types import SimpleNamespace

    import gpuwm.ingest.hrrr as hrrr
    from tools import hrrr_single_domain_benchmark as route

    source = inspect.getsource(hrrr.interpolate_hrrr_to_lambert)
    assert ('withheld_fields=getattr(snapshot, "withheld_fields", None) '
            'or None') in source
    met = SimpleNamespace(valid_time=None, levels_hpa=np.ones(2),
                          fields={"Q2": np.zeros((4, 5), np.float32)},
                          withheld_fields={"QNWFA": REASON, "QNIFA": REASON})
    assert route._detach_mapped_snapshot(met).withheld_fields == {
        "QNWFA": REASON, "QNIFA": REASON}
    strip = route._crop_horizontal_snapshot(
        met, y0=0, y1=4, x0=0, x1=2, full_shape=(4, 5), detach=True)
    assert strip.withheld_fields == {"QNWFA": REASON, "QNIFA": REASON}


def test_analysis_on_a_masked_cycle_gets_the_named_refusal_with_the_reason(
        tmp_path):
    """mp28_aerosol_source = "analysis" (configs/recipes/
    hrrr_configuration_clock.toml) on a masked cycle: the existing named
    refusal, saying the pair was published with a bitmap, not a silent
    fallback to climatology."""
    from gpuwm.ingest.hrrr import load_hrrr_native_window
    from gpuwm.ingest.real import initialize_real
    from test_metem_differential import _synthetic

    sealed = _publication(tmp_path / "bridge", WITHHELD)
    withheld = load_hrrr_native_window(
        tmp_path / "bridge", 0, expected_manifest_sha256=sealed).withheld_fields
    snapshot, cfg, coord, terrain, orography = _synthetic(2500., 2500.)
    cfg = replace(cfg, mp_physics=28, mp28_aerosol_source="analysis")
    with pytest.raises(ValueError) as refused:
        initialize_real(replace(snapshot, withheld_fields=withheld), cfg,
                        coord, terrain, source_orography=orography,
                        preprocess_backend="cpu", state_backend="cpu",
                        analyzed_species=())
    message = str(refused.value)
    assert message.startswith(
        "analyzed aerosol IC/BC requires QNWFA and QNIFA on every initial "
        "and boundary frame; missing QNIFA, QNWFA; substituting climatology "
        "would change the requested forcing")
    assert ("the source published QNIFA, QNWFA with a GRIB2 bitmap (masked "
            "points) on this cycle and the decoder withheld them: "
            + REASON) in message
    # A source that never published the pair keeps the plain sentence.
    with pytest.raises(ValueError) as plain:
        initialize_real(snapshot, cfg, coord, terrain,
                        source_orography=orography, preprocess_backend="cpu",
                        state_backend="cpu", analyzed_species=())
    assert "GRIB2 bitmap" not in str(plain.value)
    assert str(plain.value).endswith("would change the requested forcing")


def test_the_decoder_reads_the_pair_only_for_a_configuration_that_uses_it(
        tmp_path):
    """2026-10-04 06Z publishes PMTF hybrid level 1 clean at f00-f05 and
    with a bitmap at f06 and f07.  `gpuwm go` decodes HRRR as posted: f00
    is the reference, so a pair read unasked there was declared and the
    first masked later lead refused the whole run, a default-profile
    (aer_init_opt = wif_input_opt = 1) run that never uses the pair.  The
    decoder is asked for the pair (--analyzed-aerosol) exactly when the
    configuration uses it, the rule the fetch's record subset follows;
    otherwise its command is 2.8.6's."""
    import inspect

    from gpuwm.experiment import load_experiment
    from gpuwm.preparation_assets import analyzed_aerosol_domains
    from tools import hrrr_single_domain_benchmark as route
    from tools.hrrr_pipeline import HrrrPipelineProducer

    root = Path(__file__).resolve().parents[1]
    clock = load_experiment(root / "configs/recipes/hrrr_configuration_clock.toml")
    assert analyzed_aerosol_domains(clock) == (1,)
    default = replace(clock, domains=tuple(
        replace(domain, run=replace(domain.run, use_rap_aero_icbc=False,
                                    mp28_aerosol_source="auto",
                                    aer_init_opt=1, wif_input_opt=1))
        for domain in clock.domains))
    assert analyzed_aerosol_domains(default) == ()

    series = tmp_path / "series.tsv"
    series.write_text("".join(
        f"{hour}\twrfnatf{hour:02d}.grib2\twrfprsf{hour:02d}.grib2\n"
        for hour in range(9)))

    def producer(**extra):
        return HrrrPipelineProducer(
            decoder=tmp_path / "hrrr_grib2_bridge", series=series,
            output=tmp_path / "bridge", signals=tmp_path / "signals",
            cycle="2026-10-04 06:00:00", window=(1, 2, 3, 4), workers="2",
            log=tmp_path / "decoder.log", **extra)

    tail = (str(tmp_path / "series.tsv"), str(tmp_path / "bridge"),
            str(tmp_path / "signals"))
    window = ("1", "2", "3", "4")
    decoder = str(tmp_path / "hrrr_grib2_bridge")
    admit = tmp_path / "admitted"
    assert producer().decoder_argv() == (
        decoder, "--series-workers-ready", "2", *tail,
        "2026-10-04 06:00:00", *window)
    assert producer(admissions=admit).decoder_argv() == (
        decoder, "--series-workers-posted", "2", *tail, str(admit),
        "2026-10-04 06:00:00", *window)
    assert producer(analyzed_aerosol=True).decoder_argv() == (
        decoder, "--analyzed-aerosol", "--series-workers-ready", "2", *tail,
        "2026-10-04 06:00:00", *window)
    assert producer(admissions=admit, analyzed_aerosol=True).decoder_argv() == (
        decoder, "--analyzed-aerosol", "--series-workers-posted", "2", *tail,
        str(admit), "2026-10-04 06:00:00", *window)

    # The HRRR route asks for the pair from the configuration, as posted
    # and with the window whole.
    source = inspect.getsource(route.run)
    assert ("decoder_reads_analyzed_aerosol = bool(analyzed_aerosol_domains(exp))"
            in source)
    assert source.count("analyzed_aerosol=decoder_reads_analyzed_aerosol") == 2

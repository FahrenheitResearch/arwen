"""The Earth-2 donor is the file the fetch delivered, whichever transport delivered it.

Named breakage (WOOF run_6b3482bc3f8d1785, 2026-10-26): under GPUWM_FETCH_POLICY=aws the GDAS analysis arrives whole
from S3 as gdas.tHHz.pgrb2.0p25.f000, not as the NOMADS subset (gdas.*.subset.grib2) the model table names, so every
Earth-2 run from a cycle inside the forecast-donor window stopped at "donor recipe expected one gdas.*.subset.grib2,
found 0". The file names below are the ones each transport wrote on a real fetch of 2026-10-26 12Z.
"""
from datetime import datetime, timedelta, timezone

import pytest

from gpuwm import earth2_run, fetch_routes


def _fetch_writes(name):
    def run(tokens, check):
        out = earth2_run.Path(tokens[tokens.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        payload = out / name
        payload.write_bytes(b"GRIB")
        (out / "gdas-series.tsv").write_text(f"0\t{name}\t81\n")
        (out / fetch_routes.INPUT_LIST_NAME).write_text(f"{payload.resolve()}\n")
    return run


@pytest.mark.parametrize("name", ["gdas.t12z.pgrb2.0p25.f000",               # aws: the whole S3 object
                                  "gdas.t12z.pgrb2.0p25.f000.subset.grib2"])  # nomads: the filtered subset
def test_gdas_donor_is_the_fetched_payload(tmp_path, monkeypatch, name):
    table = earth2_run.model_table()
    cycle = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    selected = earth2_run.resolve_donor_policy(table["models"]["fcn3"], cycle)
    assert table["runtime"]["donor_acquisition"][selected["provider"]]["source"] == "gdas"
    monkeypatch.setattr(earth2_run.subprocess, "run", _fetch_writes(name))
    donor = earth2_run._donor(table=table, row=table["models"]["fcn3"], cycle=cycle, area=None,
                              out=tmp_path, force=False)
    assert earth2_run.Path(donor["analysis"]).name == name


def test_two_listed_payloads_still_refuse(tmp_path, monkeypatch):
    table = earth2_run.model_table()
    cycle = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)

    def run(tokens, check):
        out = earth2_run.Path(tokens[tokens.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        names = ["gdas.t12z.pgrb2.0p25.f000", "gdas.t12z.pgrb2.0p25.f003"]
        for name in names:
            (out / name).write_bytes(b"GRIB")
        (out / fetch_routes.INPUT_LIST_NAME).write_text("".join(f"{(out / n).resolve()}\n" for n in names))
    monkeypatch.setattr(earth2_run.subprocess, "run", run)
    with pytest.raises(ValueError, match="donor recipe expected one"):
        earth2_run._donor(table=table, row=table["models"]["fcn3"], cycle=cycle, area=None, out=tmp_path, force=False)

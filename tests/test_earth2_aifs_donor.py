"""AIFS hindcasts take the ERA5 donor; forecasts keep the IFS analysis.

WOOF owner canary run_a0a151f3100948b1 (AIFS ENS, 2024-09-24 12Z, 2026-10-26): the row forced the IFS open-data
donor for every cycle and that cycle's file has no orography: 'donor lacks requested terrain_height'."""
from datetime import datetime, timezone


def test_aifs_rows_follow_the_age_rule():
    from gpuwm.earth2_run import model_table, resolve_donor_policy
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    for name in ("aifs", "aifs2", "aifs-ens", "aifs2-ens"):
        row = model_table()["models"][name]
        assert resolve_donor_policy(row, datetime(2024, 9, 24, 12, tzinfo=timezone.utc), now=now)["id"] == "era5-arco", name
        assert resolve_donor_policy(row, datetime(2026, 10, 26, 12, tzinfo=timezone.utc), now=now)["id"] == "ifs", name

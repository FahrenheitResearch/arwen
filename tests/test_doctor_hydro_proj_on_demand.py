"""An unstaged on-demand hydro bridge is an on-demand doctor row, not a missing engine.

THE BREAKAGE (2.9.7 pass 5 R12, Windows wheel smoke): a fresh install showed "MISSING rust engines (3)"
for rw_hydro_world, rw_hydrofabric and rw_hydro_init, which hydro (opt-in since 2.9.4.1) had never needed
to stage; their first use stages them from the release bundle, as every other on-demand bridge's does.
"""
from gpuwm import bridges, doctor


def test_unstaged_on_demand_hydro_bridges_read_on_demand(monkeypatch):
    monkeypatch.setattr(bridges, "artifact_candidates", lambda *a, **k: [])
    monkeypatch.setattr(bridges, "on_demand_fetch_offered", lambda artifact, crate_relative=None: True)
    rows = {row.name: row for row in doctor._hydro_proj_checks()}
    for name in ("rw_hydro_world", "rw_hydrofabric", "rw_hydro_init"):
        row = rows[f"hydro PROJ {name}"]
        assert row.brief == "on demand; staged by its first fetch", row
        assert not row.blocking
        assert row.action == f"gpuwm fetch-bridges --partial --artifact {name}"

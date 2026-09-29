"""Grell-Freitas ``clos_choice`` at the config doors (no card needed).

WRF's ``clos_choice`` (``ichoice`` in module_cu_gf_deep.F) is 0 for the
mean of the sixteen closure members and 1..16 for one member alone; the
kernel carries every value (gpuwm/core/gf.py hands it to
gf_gfdrv_stage).  The device proof that each member matches the float32
reference is tests/test_gf_closure_choice_cuda.py; this file holds the
admission, the warning and the refusal, and the registry row that
publishes them.
"""

from __future__ import annotations

import pytest

_GF_BASE = dict(nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
                dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
                bl_pbl_physics=1, sf_sfclay_physics=91, cu_physics=3,
                cudt_minutes=0.0)


@pytest.mark.parametrize("value", [1, 10, 16])
def test_a_single_member_closure_is_admitted_and_says_so(value, capsys):
    """clos_choice 1..16 runs one member of cup_forcing_ens_3d alone.

    WRF defines every one of them and the kernel carries them (gf.py
    hands clos_choice to gf_gfdrv_stage, gf.cu copies member ichoice
    into every slot).  They were refused only because no WRF run covers
    them, which names no breakage, so they run and the load says what
    is unverified.
    """
    from gpuwm.config import RunConfig, validate_run_config

    cfg = validate_run_config(RunConfig(**_GF_BASE, clos_choice=value))
    assert cfg.clos_choice == value
    said = capsys.readouterr().err
    rows = [row for row in said.splitlines()
            if f"clos_choice={value} runs Grell-Freitas" in row]
    assert len(rows) == 1, said
    assert f"closure member {value} alone" in rows[0]
    assert "not verified against WRF" in rows[0]


def test_the_ensemble_closure_says_nothing(capsys):
    """The control: 0, the WRF-compared ensemble mean, is silent."""
    from gpuwm.config import RunConfig, validate_run_config

    validate_run_config(RunConfig(**_GF_BASE, clos_choice=0))
    assert "clos_choice" not in capsys.readouterr().err


def test_a_closure_outside_the_member_array_is_refused_by_its_breakage():
    """17 would read past xf_ens(1:16); a negative value takes neither of
    WRF's two branches.  Both refusals say which."""
    from gpuwm.config import RunConfig, validate_run_config

    with pytest.raises(ValueError, match=r"reads past the end of that "
                                         r"array") as caught:
        validate_run_config(RunConfig(**_GF_BASE, clos_choice=17))
    assert "xf_ens(1:16)" in str(caught.value)
    with pytest.raises(ValueError, match="neither of cup_forcing_ens_3d's"):
        validate_run_config(RunConfig(**_GF_BASE, clos_choice=-1))


def test_the_registry_row_offers_every_member_and_says_what_is_unverified():
    """The published parameter row agrees with the run door."""
    from gpuwm.config import GF_CLOSURE_MEMBERS, gf_clos_choice_refusal
    from gpuwm.physics_registry import physics_registry

    row = physics_registry()["parameters"]["clos_choice"]
    assert row["enum"] == list(range(GF_CLOSURE_MEMBERS + 1))
    assert row["default"] == 0
    assert all(gf_clos_choice_refusal(v) is None for v in row["enum"])
    assert gf_clos_choice_refusal(GF_CLOSURE_MEMBERS + 1) is not None
    assert "not verified against a WRF run" in row["warnings"][0]


def test_every_member_belongs_to_exactly_one_family():
    from gpuwm.config import GF_CLOSURE_FAMILIES, GF_CLOSURE_MEMBERS

    members = sorted(m for group, _ in GF_CLOSURE_FAMILIES for m in group)
    assert members == list(range(1, GF_CLOSURE_MEMBERS + 1))


# ---------------------------------------------------------------------------
# A tree that mixes Grell-Freitas with another cumulus choice
# ---------------------------------------------------------------------------


def _grell_tree_toml(*, shared: str = "", root: str = "",
                     child: str = "") -> str:
    """A 9 km root over a 3 km child; each argument is extra table lines."""
    return f"""[experiment]
name = "gf_tree"
start_time = 2026-08-01T00:00:00
run_seconds = 72.0
restart_interval_s = 0.0
[projection]
map_proj = "lambert"
ref_lat = 40.0
ref_lon = -100.0
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -100.0
[shared]
nz = 30
ztop = 16000.0
p_top = 10000.0
moist = true
mp_physics = 6
sf_sfclay_physics = 1
bl_pbl_physics = 1
km_opt = 4
cu_physics = 0
cudt_minutes = 0.0
{shared}[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 60
ny = 60
dx = 9000.0
time_step = 36
specified = true
nested = false
history_interval_s = 3600.0
{root}[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 15
j_parent_start = 15
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 30
ny = 30
specified = false
nested = true
history_interval_s = 3600.0
{child}"""


def _load(tmp_path, text):
    from gpuwm.experiment import load_experiment

    path = tmp_path / "tree.toml"
    path.write_text(text, encoding="utf-8")
    return load_experiment(path)


@pytest.mark.parametrize("grell", ["root", "child"])
def test_a_shared_closure_reaches_only_the_grell_freitas_domains(
        tmp_path, grell):
    """clos_choice and ishallow are read only where cu_physics = 3.

    The wizard writes them in [shared], so on a Grell-Freitas root over a
    cumulus-off child (or the other way round) that is where a user sets
    them, and [shared] was refused on the cumulus-off domain.  A [shared]
    value now reaches the Grell-Freitas domains and leaves the other on
    the Registry defaults.
    """
    grell_line = "cu_physics = 3\n"
    exp = _load(tmp_path, _grell_tree_toml(
        shared="clos_choice = 1\nishallow = 1\n",
        root=grell_line if grell == "root" else "",
        child=grell_line if grell == "child" else ""))
    by_cu = {domain.run.cu_physics: domain.run for domain in exp.domains}
    assert (by_cu[3].clos_choice, by_cu[3].ishallow) == (1, 1)
    assert (by_cu[0].clos_choice, by_cu[0].ishallow) == (0, 0)


def test_a_grell_key_with_no_grell_domain_to_read_it_is_still_refused(
        tmp_path):
    """A tree with no Grell-Freitas domain keeps the refusal: the key
    would change nothing anywhere in it."""
    with pytest.raises(ValueError, match="Grell-family keys read only"):
        _load(tmp_path, _grell_tree_toml(shared="clos_choice = 1\n"))


def test_a_grell_key_written_on_a_cumulus_off_domain_is_checked_as_written(
        tmp_path):
    """The rule speaks for [shared] only: a value written into a
    cumulus-off domain's own table names that domain and is refused."""
    with pytest.raises(ValueError, match="Grell-family keys read only"):
        _load(tmp_path, _grell_tree_toml(root="cu_physics = 3\n",
                                         child="clos_choice = 1\n"))


def test_a_prepared_domain_does_not_bind_the_grell_selectors(tmp_path):
    """Preparation reads neither clos_choice nor ishallow.

    An HRRR hierarchy prepares every domain from WRF namelists and the
    forecast reads the experiment config, so a Grell-Freitas tree set to
    clos_choice = 1 in its TOML alone was refused after preparation.  A
    prepared domain matches the configured one at any value of the two; a
    field preparation does read still refuses.
    """
    import dataclasses

    from gpuwm.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config,
        prepared_domain_config_identity, undelayed_identity_defaults)

    exp = _load(tmp_path, _grell_tree_toml(
        shared="clos_choice = 1\nishallow = 1\n", root="cu_physics = 3\n"))
    live = exp.domains[0]
    assert (live.run.clos_choice, live.run.ishallow) == (1, 1)
    prepared = dataclasses.replace(live, run=dataclasses.replace(
        live.run, clos_choice=0, ishallow=0))

    def compare(cached):
        return compare_prepared_domain_config(
            effective_prepared_domain_config(
                prepared_domain_config_identity(cached)),
            effective_prepared_domain_config(
                prepared_domain_config_identity(live)),
            not_in_use=undelayed_identity_defaults(exp))

    _tolerated, differing = compare(prepared)
    assert differing == []
    drifted = dataclasses.replace(prepared, run=dataclasses.replace(
        prepared.run, mp_physics=8))
    _tolerated, differing = compare(drifted)
    assert "run.mp_physics" in differing

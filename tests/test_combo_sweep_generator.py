"""The combo-sweep generator's WRF rules and its determinism.

What breaks without these: stock WRF 4.6.1 refused 28 stratum A combos
(mp 28 without wif_input_opt = 1; MYNN's EDMF beside a Grell-Freitas shallow
plume), so the pairs only those combos carried were never exercised, while
the list claimed full coverage.  And the covering array read Python's
string-hash order, so "rebuild with --seed 20261007" gave a different list in
every process: the list the recordings were made from could not be rebuilt.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import OrderedDict
import hashlib

HERE = Path(__file__).resolve().parents[1] / "tools" / "combo_sweep"


def _gen():
    spec = importlib.util.spec_from_file_location("gen_combos_under_test", HERE / "gen_combos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mynn_with_edmf_never_meets_a_shallow_plume():
    g = _gen()
    space = g.Space(OrderedDict(list(g.CORE.items()) + list(g.DYN.items()) + list(g.SUB.items())))
    assert g.MYNN_IDENTITY["bl_mynn_edmf"] == 1
    assert space.solve({"pbl": 5, "ishallow": 1}) is None
    assert space.solve({"pbl": 5, "ishallow": 0}) is not None
    assert space.solve({"pbl": 1, "ishallow": 1}) is not None


def test_thompson_aerosol_is_rendered_with_the_whole_climatology_triple():
    g = _gen()
    space = g.Space(OrderedDict(list(g.CORE.items()) + list(g.DYN.items()) + list(g.SUB.items())))
    nl, _ = g.render(space.solve({"mp": 28}))
    assert nl["domains"]["wif_input_opt"] == 1
    assert nl["domains"]["num_wif_levels"] == 30
    assert nl["physics"]["use_aero_icbc"] is True
    nl, _ = g.render(space.solve({"mp": 8}))
    assert "wif_input_opt" not in nl["domains"]


def test_the_committed_list_obeys_both_rules_and_keeps_the_recorded_ids():
    new = json.loads((HERE / "combos.json").read_text(encoding="utf-8"))
    old_bytes = (HERE / "combos-v1.json").read_bytes()
    old = json.loads(old_bytes)
    for rec in new["combos"]:
        ph, dom = rec["namelist"]["physics"], rec["namelist"]["domains"]
        assert not (ph.get("bl_pbl_physics") == 5 and ph.get("ishallow") == 1), rec["id"]
        if ph["mp_physics"] == 28:
            assert (dom.get("wif_input_opt"), dom.get("num_wif_levels"),
                    ph.get("use_aero_icbc")) == (1, 30, True), rec["id"]
    for name, cover in new["coverage"].items():
        if "covered" in cover:
            assert cover["covered"] == cover["required"], name
    regen = new["regeneration"]
    assert regen["keep_rows_from"]["sha256"] == hashlib.sha256(old_bytes).hexdigest()
    old_ids = {r["id"]: r for r in old["combos"]}
    new_ids = {r["id"]: r for r in new["combos"]}
    dropped = {d["id"] for d in regen["dropped"]}
    assert dropped == set(old_ids) - set(new_ids)
    assert set(regen["added"]) == set(new_ids) - set(old_ids)
    for cid in set(old_ids) & set(new_ids):  # a kept id names the same assignment
        assert old_ids[cid]["factors"] == new_ids[cid]["factors"], cid


_PROBE = r"""
import importlib.util, json, random, sys
from collections import OrderedDict
spec = importlib.util.spec_from_file_location("g", sys.argv[1])
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
f = OrderedDict(list(g.CORE.items()) + [(n, g.SUB[n]) for n in
                                          ("bl_mynn_mixscalars", "scalar_pblmix", "ishallow")])
space = g.Space(f)
rows = g.build_cover(space, g.required_pairs(space), random.Random(20261007), candidates=8)
print(json.dumps(rows))
"""


def test_the_covering_array_does_not_depend_on_string_hashing():
    outputs = set()
    for seed in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        done = subprocess.run([sys.executable, "-c", _PROBE, str(HERE / "gen_combos.py")],
                              capture_output=True, text=True, env=env, check=True, timeout=600)
        outputs.add(done.stdout)
    assert len(outputs) == 1

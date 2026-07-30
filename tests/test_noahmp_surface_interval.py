"""A non-zero surface-call interval is opt-in, and it stays opt-in.

Noah-MP's host column cost makes ``bldt = 0`` -- a land-surface call on every
dynamics step -- unaffordable on the finest nest.  A longer ``bldt`` is the
single largest lever available, worth 36x at ``dt = 1.667 s``, and it was
authorised for Noah-MP specifically.  It is a **cost mitigation with a measured
forecast impact, not a physics recommendation**, and the authorisation came
with a constraint: it must not become permanent, and no future configuration
-- or future reader, human or model -- may come to treat it as normal.

The failure mode being guarded is inheritance, not authorship.  ``bldt`` is in
``allowed_parameter_keys`` for the domain-tree runner route, and every registry
template carries a ``parameters`` dict that routes merge in.  One
``"bldt": 1.0`` in a template would therefore propagate to every configuration
built from it, silently, with no author ever having typed it.  That is the
thing this gate exists to prevent.

So the rule is narrow and positional:

* the ``RunConfig`` dataclass default is ``0.0`` and stays ``0.0``;
* the registry's generic ``parameters["bldt"]["default"]`` is ``0.0``;
* **no registry template may mention ``bldt`` at all** -- not zero, not
  non-zero.  A template is the inheritance vector, so the gate refuses the key
  outright rather than refusing a value;
* no runner route may supply a ``bldt`` default (routes may *allow* it as an
  explicit per-run override; that is what ``allowed_parameter_keys`` is);
* a config file may carry a non-zero ``bldt`` only if it is named in
  :data:`OPT_IN_CONFIGS` **and** carries :data:`JUSTIFICATION` verbatim in the
  comments attached to the assignment.

The last clause is why this is not merely a lint.  The justification travels
with the value, in the file, where the next reader finds it.

A gate that only ever passes is not evidence of anything, so the real-tree
assertions are paired with scratch-tree negative controls that inject each
violation into a copy and prove the checker reports it.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from gpuwm.config import RunConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
REGISTRY_PATH = REPO_ROOT / "gpuwm" / "physics_registry_v2.json"

#: The exact sentence that must accompany any non-zero ``bldt``, verbatim.
#: Deliberately blunt: a reader skimming for the normal setting must trip over
#: the word "not".
JUSTIFICATION = "COST MITIGATION, NOT A PHYSICS RECOMMENDATION"

#: The only config files permitted to carry a non-zero ``bldt``, each with the
#: reason it was granted.  Adding a file here is a deliberate act that shows up
#: in review; nothing merges into this table by inheritance.
OPT_IN_CONFIGS: dict[str, str] = {
    "real74_4dom_noahmp_surface_interval.toml":
        "Noah-MP throughput mitigation on the finest nest only, authorised "
        "for this scheme; the measured forecast impact is recorded in "
        "docs/noahmp_device_column_report.md.",
}

_BLDT_ASSIGNMENT = re.compile(r"^[ \t]*bldt[ \t]*=[ \t]*([0-9eE+.\-]+)", re.M)


def _config_files() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.toml"))


def _declared_bldts(text: str) -> list[tuple[float, str]]:
    """Every ``bldt`` literal in the file, with the comment block above it.

    A file carries more than one: an experiment TOML has the ``[shared]``
    value that all domains inherit and may have a per-``[[domain]]`` override.
    Checking only the first would let a non-zero nest override hide behind a
    zero in ``[shared]``, which is exactly the shape of the one config that is
    allowed to have one.

    The justification has to be *attached* to its own assignment.  A copy of
    the sentence in an unrelated corner of the file, or above a different
    ``bldt``, does not license the value.
    """
    found: list[tuple[float, str]] = []
    for match in _BLDT_ASSIGNMENT.finditer(text):
        lines = text[:match.start()].splitlines()
        block: list[str] = []
        for line in reversed(lines):
            if line.lstrip().startswith("#"):
                block.append(line)
            elif line.strip() == "":
                continue
            else:
                break
        found.append((float(match.group(1)), "\n".join(reversed(block))))
    return found


def check_configs(config_dir: Path,
                  opt_in: dict[str, str] | None = None) -> list[str]:
    """Every way a non-zero ``bldt`` can reach a config that did not ask."""
    allowed = OPT_IN_CONFIGS if opt_in is None else opt_in
    problems: list[str] = []
    for path in sorted(config_dir.glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        for declared, comment in _declared_bldts(text):
            if declared == 0.0:
                continue
            if path.name not in allowed:
                problems.append(
                    f"{path.name}: bldt = {declared} but the file is not in "
                    f"OPT_IN_CONFIGS.  A non-zero surface-call interval is a "
                    f"cost mitigation authorised for Noah-MP only and must be "
                    f"opted into explicitly, never inherited or copied.")
                continue
            if JUSTIFICATION not in comment:
                problems.append(
                    f"{path.name}: bldt = {declared} is opted in but the "
                    f"assignment does not carry {JUSTIFICATION!r} in the "
                    f"comment block directly above it.  The justification "
                    f"must travel with the value.")
    return problems


def _keys_named(node, wanted: str, path: str = ""):
    """Every ``(json-path, value)`` where a mapping key equals ``wanted``.

    Whole-document rather than one blessed sub-key: a template that grew a
    ``settings`` or ``overrides`` block tomorrow would carry the value just as
    effectively as ``parameters`` does today, and the gate should not have to
    be taught each new spelling.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key == wanted:
                yield here, value
            else:
                yield from _keys_named(value, wanted, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _keys_named(value, wanted, f"{path}[{index}]")


def check_registry(document: dict) -> list[str]:
    """No template and no route may supply ``bldt``."""
    problems: list[str] = []

    generic = document.get("parameters", {}).get("bldt", {})
    if generic.get("default") != 0.0:
        problems.append(
            f"parameters['bldt']['default'] is {generic.get('default')!r}, "
            f"must be 0.0: this is the generic default every configuration "
            f"falls back to.")

    for name, template in document.get("templates", {}).items():
        for path, value in _keys_named(template, "bldt"):
            problems.append(
                f"template {name!r} sets bldt = {value!r} at {path}.  "
                f"Templates are merged into every configuration built from "
                f"them, so a bldt here is inherited by runs that never asked "
                f"for it.  Templates may not mention bldt at any value, "
                f"anywhere in the document.")

    routes = document.get("runner_routes", {})
    for name, route in (routes.items() if isinstance(routes, dict)
                        else enumerate(routes)):
        for key in ("parameters", "parameter_defaults", "defaults"):
            block = route.get(key) if isinstance(route, dict) else None
            if isinstance(block, dict) and "bldt" in block:
                problems.append(
                    f"runner route {name!r} supplies a default bldt = "
                    f"{block['bldt']!r} under {key!r}.  A route may ALLOW "
                    f"bldt as an explicit override; it may not default it.")
    return problems


# --------------------------------------------------------------------------
# The real tree
# --------------------------------------------------------------------------

def test_the_runconfig_default_is_zero_and_means_every_step():
    """The generic default is untouched, and it is the WRF meaning of zero.

    Read off the dataclass field rather than an instance: this is an assertion
    about the *declared default*, which is what an unspecified config inherits.
    """
    declared = {f.name: f.default for f in fields(RunConfig)}
    assert declared["bldt"] == 0.0


def test_no_template_and_no_route_carries_a_surface_interval():
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert check_registry(document) == []


def test_no_config_carries_an_unjustified_surface_interval():
    assert check_configs(CONFIG_DIR) == []


@pytest.fixture
def expert_budget(monkeypatch):
    """Acknowledge Noah-MP's column budget for the duration of one load.

    Loading an expert-only config to read its *cadence* is not running it.
    The budget gate is a separate, deliberately loud control on throughput and
    it stays exactly as it is; this fixture satisfies it for a parse and
    nothing more.  It is scoped to the tests that need it so the gate keeps
    firing everywhere else.
    """
    monkeypatch.setenv("GPUWM_NOAHMP_EXPERT_COLUMN_BUDGET", "360000")


def _resolved_bldt_by_domain(path: Path) -> dict[int, tuple[float, int]]:
    """``{grid_id: (bldt, sf_surface_physics)}`` as the loaders resolve it.

    The ``[case_data]`` table is split off exactly as
    :func:`gpuwm.case_data.load_experiment_case` does; this gate is about the
    physics cadence and must not depend on a case's inputs being present.
    """
    import tomllib

    from gpuwm.experiment import build_experiment, is_experiment_toml

    if is_experiment_toml(path):
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
        raw.pop("case_data", None)
        experiment = build_experiment(raw, source=str(path))
        return {dc.grid_id: (dc.run.bldt, int(dc.run.sf_surface_physics))
                for dc in experiment.domains}
    cfg = load_config(path)
    return {cfg.grid_id: (cfg.bldt, int(cfg.sf_surface_physics))}


def test_every_opted_in_config_actually_opted_in(expert_budget):
    """No stale allowlist entries, and each one really is a Noah-MP config.

    A name left in :data:`OPT_IN_CONFIGS` after its file lost the setting is a
    licence sitting unused, ready to excuse the next value that lands there.
    """
    for name in OPT_IN_CONFIGS:
        path = CONFIG_DIR / name
        assert path.exists(), f"{name} is in OPT_IN_CONFIGS but does not exist"
        text = path.read_text(encoding="utf-8")
        positive = [(value, comment)
                    for value, comment in _declared_bldts(text) if value > 0.0]
        assert positive, (
            f"{name} is in OPT_IN_CONFIGS but declares no positive bldt; "
            f"remove the entry rather than leaving the licence unused")
        for _, comment in positive:
            assert JUSTIFICATION in comment
        resolved = _resolved_bldt_by_domain(path)
        opted = {gid: value for gid, (value, _) in resolved.items()
                 if value > 0.0}
        assert opted, f"{name}: no domain actually resolves to a positive bldt"
        for gid, (value, scheme) in resolved.items():
            if value > 0.0:
                assert scheme == 4, (
                    f"{name} domain {gid} opts into a Noah-MP cost mitigation "
                    f"but selects sf_surface_physics = {scheme}")


def test_the_interval_is_scoped_to_the_nest_that_needs_it(expert_budget):
    """The coarse domains keep calling the surface every step.

    A file-wide interval would be the cheap version of this change.  The cost
    is on the finest nest; so is the override.
    """
    path = CONFIG_DIR / "real74_4dom_noahmp_surface_interval.toml"
    resolved = _resolved_bldt_by_domain(path)
    finest = max(resolved)
    for gid, (value, _) in sorted(resolved.items()):
        if gid == finest:
            assert value > 0.0, "the finest nest is the one that needed it"
        else:
            assert value == 0.0, (
                f"domain {gid} inherited bldt = {value}; only the finest nest "
                f"was granted an interval")


def test_every_other_config_loads_at_zero():
    """The resolved value, not the literal -- this catches inheritance.

    Per-domain, because an experiment TOML resolves one RunConfig per nest and
    a value arriving through ``[shared]`` would otherwise be invisible here.
    """
    for path in _config_files():
        if path.name in OPT_IN_CONFIGS:
            continue
        try:
            resolved = _resolved_bldt_by_domain(path)
        except Exception:
            continue  # cases whose data root is absent are not this gate's job
        for gid, (value, _) in resolved.items():
            assert value == 0.0, (
                f"{path.name} domain {gid} resolves to bldt = {value} without "
                f"appearing in OPT_IN_CONFIGS; something supplies it by "
                f"fallback")


def test_the_opt_in_config_reaches_the_physics_clock(expert_budget):
    """The setting is inert unless it survives into STEPBL.

    This is the assertion that makes the opt-in real rather than decorative.
    ``gpuwm/runtime.py`` assigned ``bldt_seconds = dt`` and ``stepbl = 1``
    unconditionally on the single-domain path, silently discarding whatever
    the config asked for; at ``bldt = 0`` that assignment was a no-op, so
    nothing in the suite ever caught it.  Pin the resolved clock, per domain,
    so a regression to a hardcoded cadence fails here.
    """
    import tomllib

    from gpuwm.core.clock import resolve_clock
    from gpuwm.experiment import build_experiment

    path = CONFIG_DIR / "real74_4dom_noahmp_surface_interval.toml"
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    raw.pop("case_data", None)
    clock = resolve_clock(build_experiment(raw, source=str(path)))
    stepbl = {spec.grid_id: spec.stepbl for spec in clock.domains}

    assert stepbl[4] == 36, (
        f"d04 resolved STEPBL = {stepbl[4]}, expected 36: a 60 s interval at "
        f"dt = 5/3 s is exactly 36 steps, and that ratio is the entire value "
        f"of the change")
    for grid_id in (1, 2, 3):
        assert stepbl[grid_id] == 1, (
            f"d0{grid_id} resolved STEPBL = {stepbl[grid_id]}, expected 1: "
            f"only the finest nest was granted an interval")


# --------------------------------------------------------------------------
# Negative controls: the gate has to bite
# --------------------------------------------------------------------------

def test_a_template_that_sets_bldt_is_reported():
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    name = next(iter(mutated["templates"]))
    mutated["templates"][name].setdefault("parameters", {})["bldt"] = 1.0
    problems = check_registry(mutated)
    assert any("template" in p and "bldt" in p for p in problems), problems


def test_a_template_that_sets_bldt_to_zero_is_also_reported():
    """Refusing only non-zero values would leave the vector open.

    A template carrying ``"bldt": 0.0`` is harmless today and is a one-character
    edit away from not being.  The key itself is what the gate refuses.
    """
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    name = next(iter(mutated["templates"]))
    mutated["templates"][name].setdefault("parameters", {})["bldt"] = 0.0
    assert check_registry(mutated) != []


def test_a_template_hiding_bldt_under_a_new_block_is_reported():
    """The scan is whole-document, not one blessed sub-key.

    A template that grows a ``settings`` block tomorrow carries the value just
    as effectively as ``parameters`` does today.
    """
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    name = next(iter(mutated["templates"]))
    mutated["templates"][name]["settings"] = {"physics": {"bldt": 1.0}}
    problems = check_registry(mutated)
    assert any("settings.physics.bldt" in p for p in problems), problems


def test_a_moved_generic_default_is_reported():
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    mutated["parameters"]["bldt"]["default"] = 1.0
    assert check_registry(mutated) != []


def test_a_route_default_is_reported():
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(document)
    routes = mutated["runner_routes"]
    name = next(iter(routes))
    routes[name]["parameters"] = {"bldt": 1.0}
    assert check_registry(mutated) != []


def test_an_unlisted_config_with_a_surface_interval_is_reported(tmp_path):
    scratch = tmp_path / "configs"
    scratch.mkdir()
    (scratch / "inherited.toml").write_text(
        "[physics]\n"
        f"# {JUSTIFICATION}\n"
        "bldt = 1.0\n", encoding="utf-8")
    problems = check_configs(scratch, opt_in={})
    assert any("not in OPT_IN_CONFIGS" in p for p in problems), problems


def test_an_opted_in_config_without_the_justification_is_reported(tmp_path):
    scratch = tmp_path / "configs"
    scratch.mkdir()
    (scratch / "listed.toml").write_text(
        "[physics]\n"
        "bldt = 1.0\n", encoding="utf-8")
    problems = check_configs(scratch, opt_in={"listed.toml": "granted"})
    assert any("does not carry" in p for p in problems), problems


def test_a_justification_elsewhere_in_the_file_does_not_license_the_value(
        tmp_path):
    """The sentence has to be attached to the assignment, not merely present."""
    scratch = tmp_path / "configs"
    scratch.mkdir()
    (scratch / "listed.toml").write_text(
        f"# {JUSTIFICATION}\n"
        "[physics]\n"
        "mp_physics = 6\n"
        "\n"
        "bldt = 1.0\n", encoding="utf-8")
    problems = check_configs(scratch, opt_in={"listed.toml": "granted"})
    assert any("does not carry" in p for p in problems), problems


def test_a_zero_bldt_config_is_never_reported(tmp_path):
    """The gate must not fire on the normal case, which is every other config."""
    scratch = tmp_path / "configs"
    scratch.mkdir()
    (scratch / "normal.toml").write_text(
        "[physics]\nbldt = 0.0\n", encoding="utf-8")
    assert check_configs(scratch, opt_in={}) == []

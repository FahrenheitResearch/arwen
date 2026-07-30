"""Whether a user can select RUC, and what had to be true first.

``tests/test_ruc_runtime.py`` proves RUC forecasts, restarts and writes output.
None of that made it selectable, and this file is the difference.  It runs on
the HOST: it imports no cupy, so ``tests/conftest.py`` does not auto-mark it
``gpu`` and it answers the admission question on a machine with no card --
which is the point, because the answer is about wiring and arithmetic rather
than about a device.

Two independent blockers held the door shut, each bound to code rather than to
prose, and **both are now closed**:

1. **The ingest was not wired.**  ``gpuwm/ingest/ruc_soil.py`` is
   ``init_soil_depth_3`` + ``init_soil_3_real`` at max_ulp 0 against
   ``gpuwm/data/ruc/oracle/soil_ingest.csv``, and it had ZERO importers inside
   the package: every initializer called ``preprocess_noah_soil`` and received
   a ``NoahSoilState`` whose four-layer shape, Noah SH2O partition and sea-ice
   column are Noah's.  ``preprocess_land_surface_soil`` is now that seam, all
   seven initializers call it, and it is fail-closed on the
   ``sf_surface_physics`` selector rather than on the array shape.

2. **The per-column cost was flat.**  ``ruc_land_surface_step`` ran a Python
   ``for i in range(ncolumn)`` loop over float32 scalars, so a land-surface
   call cost the same per column at 48 columns and at 24,576 -- which made the
   whole cost of a nest its width, and d04 of the production four-domain case
   is 360,000 columns, i.e. ten minutes for ONE call.  The column now runs on
   the card: 16.8 wall seconds per simulated minute at that width snow-free
   and 65.7 fully snow-covered, measured at 360,000 columns rather than
   extrapolated from a narrow grid.

The host driver is still here and is still measured flat, because it is the
REFERENCE the oracle fixtures compare against WRF and the device path is
gated against it at max_ulp 0.  What changed is which one a forecast runs.

RUC still has no column-count rail in ``gpuwm/physics_compat.py`` and that is
now a decision rather than a gap: Noah-MP has one because its call cannot
finish, and :func:`test_the_width_rail_covers_noahmp_and_not_ruc` records both
halves.

Every gate here carries its own control, because a gate nobody has seen fail
is not evidence: the import scan is shown finding the importers of a module
that IS wired, the flatness statistic is shown rejecting work that is not
per-column, and the width rail is shown firing for the scheme that has one.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.ruc import ruc_land_surface_step
from gpuwm.core.ruc_runtime import (C1SN, C2SN, DEFINED_ILNB, ISNCOVR_OPT,
                                    RucRuntimeParameters)
from gpuwm.physics_registry import physics_registry

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "gpuwm"

#: MODI-RUC grassland over STAS-RUC loam, the configuration the runtime gate
#: forecasts and the registry warning quotes.
_GRASSLAND = 10
_LOAM = 6
_NSOIL = 9

#: Production width, ``configs/real74_4dom.toml``: d01 250x200, d02 500x400,
#: d03 501x501, d04 600x600.  d04 alone is the number that decides this.
_PRODUCTION_D04_COLUMNS = 600 * 600

#: Every initializer that builds a land-surface initial state today.  Named
#: rather than globbed: the claim is about these seven call sites, and a new
#: initializer that quietly takes an eighth path should fail this list, not
#: slip past a pattern.
_INITIALIZERS = (
    "gpuwm/era5_direct.py",
    "gpuwm/gfs_direct.py",
    "gpuwm/hrrr_hierarchy_direct.py",
    "gpuwm/mapped_direct.py",
    "gpuwm/runtime.py",
    "gpuwm/ingest/nest_init.py",
    "gpuwm/ingest/hrrr_physics.py",
)


# ---------------------------------------------------------------------------
# 1.  the ingest exists and nothing calls it
# ---------------------------------------------------------------------------

def _imported_modules(source: str) -> set[str]:
    """Every module name this source imports, absolute and relative alike.

    Relative imports are returned with a leading dot count preserved
    (``.ruc_soil``, ``..ingest.ruc_soil``) so a sibling import inside
    ``gpuwm/ingest`` is caught as well as a fully qualified one.  An AST walk
    rather than a text search: a text search for ``ruc_soil`` matches the three
    places ``gpuwm/core/ruc_runtime.py`` names the module inside an error
    message, which are documentation and not wiring.
    """
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            base = f"{prefix}{node.module or ''}"
            names.add(base)
            names.update(f"{base}.{alias.name}".rstrip(".")
                         if base.endswith(".") or not node.module
                         else f"{base}.{alias.name}"
                         for alias in node.names)
    return names


def _package_importers(module: str) -> list[str]:
    """Files under ``gpuwm/`` that import ``gpuwm.<module path>``.

    ``module`` is spelled as the dotted suffix, e.g. ``ingest.ruc_soil``.  The
    module itself is never counted as its own importer.
    """
    absolute = f"gpuwm.{module}"
    tail = module.rsplit(".", 1)[-1]
    own = _PACKAGE / Path(*module.split("."))
    hits: list[str] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path.with_suffix("") == own:
            continue
        names = _imported_modules(path.read_text(encoding="utf-8"))
        relative = {name for name in names
                    if name.lstrip(".").split(".")[-1] == tail
                    and name.startswith(".")}
        if any(name == absolute or name.startswith(absolute + ".")
               for name in names) or relative:
            hits.append(path.relative_to(_ROOT).as_posix())
    return hits


def test_the_ruc_soil_ingest_is_now_reachable_from_every_initializer() -> None:
    """The blocker commit ``46ce211`` stated -- **closed**, and re-checked.

    This gate used to assert the opposite: that ``gpuwm/ingest/ruc_soil.py``
    had zero importers anywhere under ``gpuwm/``, which is what made the
    finished, oracle-matched remap dead code and a RUC template pointless.
    It was written to fail the day a lane wired it, and it did.  The
    substantive gates on what the wiring produces -- nine LEVEL depths, the
    Noah path bitwise unchanged, and the fail-closed selector -- are in
    ``tests/test_ruc_soil_wiring.py``.  What stays here is the import-graph
    claim in its new direction, because that is the one this file's
    conclusion about selectability rests on.
    """
    assert (_PACKAGE / "ingest" / "ruc_soil.py").is_file()

    importers = _package_importers("ingest.ruc_soil")
    for path in _INITIALIZERS:
        assert path in importers, (
            f"{path} no longer reaches gpuwm.ingest.ruc_soil, so a RUC "
            "config initialised through it is back on Noah's four layers: "
            f"{importers}")

    # The control, in two halves.  The scan must still find the Noah module
    # -- otherwise the result above is a property of the scan rather than of
    # the tree -- and the importer it must find is the RUC seam itself,
    # because that seam CALLS preprocess_noah_soil for the surface half of a
    # RUC initialization rather than restating it.  The initializers no
    # longer appear in this list, and that is the change: they reach Noah
    # through the seam now.
    wired = _package_importers("ingest.soil")
    assert "gpuwm/ingest/ruc_soil.py" in wired, (
        "the RUC ingest no longer imports gpuwm.ingest.soil, so the claim "
        "that Noah's path is CALLED rather than copied is no longer bound "
        f"to anything: {wired}")
    assert len(wired) >= 3, (
        f"the import scan found almost nothing at all; it is broken: {wired}")
    # ...and it must return nothing for a module that does not exist, so a
    # scan that silently matched everything would fail here.
    assert _package_importers("ingest.no_such_module") == []


def test_no_initializer_calls_the_noah_soil_ingest_directly() -> None:
    """One seam, not seven decisions.

    ``preprocess_noah_soil`` takes no land-surface selector, so any direct
    call to it from an initializer is a path on which a ``sf_surface_physics
    = 3`` request receives Noah's four layer MIDPOINTS and runs a complete,
    plausible forecast on the wrong soil discretization.  That is the exact
    failure mode this row exists to prevent, and it is checked per file.
    """
    for path in _INITIALIZERS:
        tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}
        assert "preprocess_noah_soil" not in called, path
        assert "preprocess_land_surface_soil" in called, path

    # ...and the driver's cold start still has exactly one caller, which reads
    # the nine-level TSLB/SMOIS out of ``fields`` -- now put there by the
    # ingest above rather than by whatever the caller happened to supply.
    physics = (_PACKAGE / "core" / "physics.py").read_text(encoding="utf-8")
    assert physics.count("ruc_cold_start(") == 1


def test_exactly_one_template_selects_ruc_and_no_route_overrides_it():
    """Selectable is a property of the templates and routes, so read those.

    This test used to assert that NO template selected RUC, and that was the
    correct reading while the soil ingest had no importer and the host column
    cost a flat 1.7 ms.  Both are closed, so the assertion is inverted -- but
    only for the template.  A land_surface COMPONENT OVERRIDE is a different
    claim: it would let a user put RUC on one domain of a nest whose other
    domains are Noah, which nothing has measured, so it stays refused and the
    second half of this test is unchanged.
    """
    registry = physics_registry()
    assert registry["components"]["land_surface"]["options"]["ruc-lsm"][
        "implemented"] is True, (
        "this file exists to separate implemented from reachable; if RUC is "
        "no longer implemented the separation is not the interesting part")

    selecting = sorted(
        template_id for template_id, template in registry["templates"].items()
        if template["components"].get("land_surface") == "ruc-lsm")
    assert selecting == [RUC_TEMPLATE_ID], (
        f"the RUC templates are {selecting}; exactly one is expected, and a "
        "second would need its own evidence rather than inheriting this one's")

    overriding = sorted(
        route_id for route_id, route in registry["runner_routes"].items()
        if "land_surface" in (route.get("allowed_component_overrides") or []))
    assert overriding == [], (
        f"a runner route now allows a land_surface override: {overriding}")

    # The control: the same two reads find the scheme that IS selectable, so
    # "no template" above is a fact about RUC and not about this query.
    noah = sorted(
        template_id for template_id, template in registry["templates"].items()
        if template["components"].get("land_surface") == "noah")
    assert len(noah) >= 4, noah


# ---------------------------------------------------------------------------
# Can a user select RUC?
# ---------------------------------------------------------------------------
#
# Until 2026-07-26 the answer was no, and the registry said so: the RUC option
# carried ``reachability.state = "unreachable"`` with a two-part blocker --
# the nine-level soil ingest had no importer inside the package, and the host
# column solver cost a MEASURED FLAT 1.7 ms per land column, which is ten
# minutes for one d04 land-surface call.  Both are closed.  These are the
# gates that say the door is actually open, rather than that a template
# object exists somewhere in the JSON.

RUC_TEMPLATE_ID = "wsm6-ysu-mm5-ruc-no-radiation-implemented-unverified-v1"
#: The sources whose initializers reach RUC's own soil ingest.  ``20crv3`` is
#: the MAPPED path and is deliberately absent; see the test below.
#: Sources whose declared template list still reaches RUC.  v1.1.1
#: withdrew "gfs": a GFS-initialised RUC forecast prepares in full and
#: then cannot complete its first step (see the V-1 tests at the end of
#: this file).  ERA5 keeps it because nobody has shown it to be broken.
RUC_TEMPLATE_SOURCES = ("era5",)


def _single_domain_ruc_plan(source_id: str) -> dict:
    from gpuwm.physics_registry import PLAN_SCHEMA, registry_sha256

    return {
        "schema": PLAN_SCHEMA,
        "plan_id": f"ruc-{source_id}-single-domain-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": source_id,
            "runner_id": "tools.prepared_single_domain_forecast",
            "topology_id": "single-domain-v1",
        },
        "domains": [{"domain_id": "d01", "template_id": RUC_TEMPLATE_ID}],
    }


@pytest.mark.parametrize("source_id", RUC_TEMPLATE_SOURCES)
def test_a_user_can_select_ruc(source_id):
    """The whole point of the template, asserted through the public validator.

    Not "a template object exists": a plan that names it, on a route that
    registers it, for a source whose initializer produces the nine-level
    soil, must come back launchable with ``sf_surface_physics = 3`` and
    ``num_soil_layers = 9`` resolved.
    """
    from gpuwm.physics_registry import validate_physics_plan

    report = validate_physics_plan(_single_domain_ruc_plan(source_id))
    assert report["errors"] == [], report["errors"]
    assert report["launchable"] is True
    settings = report["resolved_domains"][0]["settings"]
    assert settings["sf_surface_physics"] == 3
    assert settings["num_soil_layers"] == 9
    # RUC is admitted with the MM5 surface layer only, and the registry has
    # to enforce that rather than the template merely happening to pick one.
    assert settings["sf_sfclay_physics"] in (1, 91)


def test_selecting_ruc_still_warns_that_it_is_unverified():
    """Reachable is not the same as validated, and the user must be told.

    ``implemented-unverified`` is a warning maturity, so a launchable RUC
    plan must carry both the maturity warning and the component's own
    warnings.  A template that quietly upgraded itself to hide them would
    pass the test above and fail this one.
    """
    from gpuwm.physics_registry import physics_registry, validate_physics_plan

    report = validate_physics_plan(_single_domain_ruc_plan("era5"))
    codes = {warning["code"] for warning in report["warnings"]}
    assert "maturity" in codes
    assert "component-warning" in codes

    registry = physics_registry()
    option = registry["components"]["land_surface"]["options"]["ruc-lsm"]
    template = registry["templates"][RUC_TEMPLATE_ID]
    assert option["maturity"] == "implemented-unverified"
    assert template["maturity"] == "implemented-unverified"
    # The measured wall clock is the thing a user most needs before choosing
    # this scheme, so it must be IN the published text, not only in a doc.
    published = " ".join(option["warnings"]) + " ".join(template["warnings"])
    assert "per simulated minute" in published
    assert "360,000" in published


def test_the_ruc_option_declares_the_reachability_it_has():
    """The declaration and the route lists must not be able to disagree.

    ``reachability.state`` is what a GUI reads to decide whether to offer the
    scheme.  It is recomputed here from the templates and the routes rather
    than trusted: a template that stopped being listed, or a listing without
    a template, both show up as a mismatch.
    """
    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    option = registry["components"]["land_surface"]["options"]["ruc-lsm"]

    templates_with_ruc = {
        template_id
        for template_id, template in registry["templates"].items()
        if template["components"]["land_surface"] == "ruc-lsm"
    }
    assert RUC_TEMPLATE_ID in templates_with_ruc

    listed = set()
    for route in registry["runner_routes"].values():
        for declared in route.get("source_template_ids", {}).values():
            listed.update(declared)
    assert templates_with_ruc & listed, (
        "a RUC template exists but no runner route lists it, so no user can "
        "reach it and the option is unreachable whatever it declares")

    assert option["reachability"]["state"] == "template"
    assert "blocker" not in option["reachability"], (
        "a reachable option must not carry a blocker; the blocker field is "
        "the thing an unreachable one owes the reader")


def test_the_mapped_source_is_deliberately_not_offered_ruc():
    """The one gap the template must not paper over.

    ``gpuwm/mapped_direct.py`` hands the soil seam the composition's
    DECLARATIVE layer contract, and
    ``gpuwm/ingest/soil_contract.py:validate_soil_layer_contract`` still
    declares exactly one target -- Noah's four layers.  So the mapped source
    cannot produce a nine-level RUC soil column, and listing the template for
    it would be a reachability claim the ingest cannot honour.

    Both halves are checked: the route list omits it, AND the refusal is real
    rather than assumed.
    """
    import numpy as _np

    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    routes = registry["runner_routes"]
    for route in routes.values():
        declared = route.get("source_template_ids", {})
        assert RUC_TEMPLATE_ID not in declared.get("20crv3", []), (
            "the mapped source lists the RUC template, but its initializer "
            "passes a declarative soil contract that RUC's ingest refuses")

    contract = {
        "depth_units": "m",
        "source_layers": [{"top": 0.0, "bottom": 0.1}],
        "target_layers": [{"top": 0.0, "bottom": 0.1}],
    }
    with pytest.raises(ValueError, match="soil_layer_contract"):
        preprocess_land_surface_soil(
            {"soil_temperature": _np.zeros((1, 2, 2), dtype=_np.float32)},
            sf_surface_physics=3,
            soil_type=_np.ones((2, 2), dtype=_np.int32),
            soil_layer_contract=contract)


def test_the_ruc_warning_prose_names_exactly_the_sources_the_matrix_offers():
    """Guidance vs gates, in one template's own words.

    The route matrix withdrew ``gfs`` in v1.1.1 -- a GFS-initialised RUC
    forecast prepares and then cannot take its first step -- and the
    admission tests above check the matrix. Nothing checked the WARNING
    prose against it, so the template kept telling users RUC is "OFFERED
    FOR ... gfs" while the gate it describes had already stopped offering
    it. A user reads the prose, not the JSON route list.

    So the sources the prose claims to offer must equal the sources the
    matrix actually reaches -- no more (an offer the runner refuses) and
    no fewer (a working route the user is never told about).
    """
    import re

    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()

    matrix_sources = set()
    for route in registry["runner_routes"].values():
        for key in ("source_template_ids", "expert_template_ids"):
            for source, template_ids in (route.get(key) or {}).items():
                if RUC_TEMPLATE_ID in template_ids:
                    matrix_sources.add(source)
    # The matrix is the ground truth the prose must match.
    assert matrix_sources == {"era5", "hrrr"}, matrix_sources

    prose = " ".join(registry["templates"][RUC_TEMPLATE_ID]["warnings"])
    offered_clause = re.search(
        r"OFFERED FOR THE DIRECT (.+?) SOURCES", prose)
    assert offered_clause, prose
    # Every known direct source is named as offered iff the matrix reaches
    # it -- the exact prose/gate agreement the withdrawal broke.
    named = set(re.findall(r"\b(gfs|hrrr|era5)\b", offered_clause.group(1)))
    assert named == matrix_sources, (named, matrix_sources)
    assert "gfs" not in named, (
        "the RUC warning still offers gfs, which the route matrix "
        "withdrew in v1.1.1")


# ---------------------------------------------------------------------------
# 2.  the cost is flat, which is what makes width fatal
# ---------------------------------------------------------------------------

_PARAMS = RucRuntimeParameters()

#: A 16x range.  Wide enough that a per-call fixed cost would show, short
#: enough that the gate is seconds rather than the ten minutes the published
#: sweep to 24,576 columns takes.
_COST_COUNTS = (48, 96, 192, 384, 768)


def _land_values(n: int) -> dict:
    """``n`` identical warm grassland columns, LSMRUC's own argument names."""
    values = {
        "soilmois": np.full((_NSOIL, n), 0.28, np.float32),
        "sh2o": np.full((_NSOIL, n), 0.28, np.float32),
        "tso": np.stack([np.full(n, 297.0 - level, np.float32)
                         for level in np.linspace(0.0, 8.0, _NSOIL)]),
        "smfr3d": np.zeros((_NSOIL, n), np.float32),
        "keepfr3dflag": np.zeros((_NSOIL, n), np.float32),
    }
    scalars = {
        "snow": 0.0, "snowh": 0.0, "snowc": 0.0, "canwat": 0.0,
        "snoalb": 0.7, "alb": 0.2, "emiss": 0.98, "lai": 2.0,
        "mavail": 0.5, "sfcexc": 0.02, "z0": 0.05, "znt": 0.05,
        "soilt": 303.0, "hfx": 0.0, "qfx": 0.0, "lh": 0.0,
        "sfcevp": 0.0, "sfcrunoff": 0.0, "udrunoff": 0.0, "acrunoff": 0.0,
        "grdflx": 0.0, "acsnow": 0.0, "snom": 0.0, "qvg": 0.011, "qcg": 0.0,
        "dew": 0.0, "qsfc": 0.011, "qsg": 0.011, "chklowq": 1.0,
        "soilt1": 303.0, "tsnav": -6.0, "smavail": 0.0, "smmax": 0.0,
        "rhosnf": 200.0, "precipfr": 0.0, "snowfallac": 0.0,
        "z3d": 40.0, "p8w": 95000.0, "t3d": 300.0, "qv3d": 0.011,
        "qc3d": 0.0, "rho3d": 1.2, "rainbl": 0.0, "frzfrac": 1.0,
        "glw": 340.0, "gsw": 560.0, "chs": 0.02, "flqc": 0.02, "flhc": 20.0,
        "albbck": 0.2, "xland": 1.0, "xice": 0.0, "tbot": 288.0,
        "shdmin": 10.0, "shdmax": 80.0, "vegfra": 60.0,
    }
    values.update({name: np.full(n, value, np.float32)
                   for name, value in scalars.items()})
    return values


def _land_surface_call(n: int) -> None:
    ruc_land_surface_step(
        _land_values(n), dt=12.0, ktau=2, zs=_PARAMS.zs,
        ivgtyp=np.full(n, _GRASSLAND, np.int32),
        isltyp=np.full(n, _LOAM, np.int32),
        ilnb=DEFINED_ILNB, ilnb_chain=False,
        c1sn=C1SN, c2sn=C2SN, isncovr_opt=ISNCOVR_OPT,
        mminlu=_PARAMS.dataset_identifier, parameters=_PARAMS.bundle)


def _ms_per_column(work, counts=_COST_COUNTS, repeats: int = 2) -> list[float]:
    """Best-of-``repeats`` wall clock of ``work(n)``, divided by ``n``.

    Best-of rather than mean: this runs on whatever box has the suite, and the
    quantity of interest is the cost when nothing else is competing.  A slower
    sample is always contention, never a faster loop.
    """
    work(counts[0])                                   # warm the import paths
    out = []
    for n in counts:
        best = min(_elapsed(work, n) for _ in range(repeats))
        out.append(best / n * 1e3)
    return out


def _elapsed(work, n: int) -> float:
    start = time.perf_counter()
    work(n)
    return time.perf_counter() - start


def test_the_host_column_cost_is_flat_in_the_column_count() -> None:
    """What the REFERENCE path costs, measured rather than asserted.

    ``ruc_land_surface_step`` reshapes to ``ncolumn`` and then runs
    ``for i in range(ncolumn)`` over float32 scalars, so the prediction is
    that per-column cost does not move with the count.  Measured on the
    reference box over a 512x range (published in
    ``docs/wrf_ruc_runtime_admission.md``): 1.687 ms/column at 48 columns and
    1.724 at 24,576, a 1.07x spread; a snow-covered column costs 3.2 ms and is
    just as flat.  This gate re-runs the cheap end of that sweep.

    The bound is loose on purpose -- the absolute number belongs to the host
    -- because the claim is FLATNESS, not speed.  What makes flatness a
    measurement instead of an assertion is the control below.

    This is no longer the scaling blocker, and the change matters.  A
    forecast runs the CUDA leaves, the CUDA snow-preparation stage and the
    CuPy array namespace (``gpuwm/core/ruc_runtime.py``
    ``ruc_device_sfctmp_sets``), which at d04 width costs 0.47 s per call
    warm and 1.83 s over snow instead of the 613 s this loop projects to.
    What is measured here is the reference implementation the oracle
    fixtures compare against WRF and the device path is gated against at
    max_ulp 0 -- so its cost is a property worth knowing and no longer a
    property that decides whether RUC is usable.
    """
    # Best-of-three rather than best-of-two: this suite shares a box with
    # other lanes, and a single contended sample is the only thing that has
    # ever moved this statistic.  More repeats tightens the measurement; the
    # bound below is unchanged.
    costs = _ms_per_column(_land_surface_call, repeats=3)
    spread = max(costs) / min(costs)
    report = ", ".join(f"{n}: {cost:.3f} ms/col"
                       for n, cost in zip(_COST_COUNTS, costs))
    assert spread < 1.5, (
        f"RUC's per-column cost is not flat over {_COST_COUNTS} ({report}); "
        "this is the host reference path and it is a Python loop, so a "
        "trend here means the loop is gone and the claim needs re-reading")

    # The control.  Hold the work FIXED at the smallest count and divide by n
    # anyway: that is what a call whose cost is NOT per-column looks like
    # through this same statistic, and it must be rejected.  Without it,
    # "spread < 1.5" would also pass for a timer that measures nothing.
    control = _ms_per_column(lambda _n: _land_surface_call(_COST_COUNTS[0]))
    control_spread = max(control) / min(control)
    assert control_spread > 8.0, (
        "the flatness statistic did not reject constant work "
        f"(spread {control_spread:.1f}x over a "
        f"{_COST_COUNTS[-1] // _COST_COUNTS[0]}x count range), so it cannot "
        "tell per-column work from amortized work and the assertion above "
        "means nothing")

    # And the consequence, in the units a caller cares about.  d04 of the
    # production four-domain case is 360,000 columns and its dt is 1.667 s
    # with bldt=0, so this is the cost of ONE of 25,920 land-surface calls.
    projected = float(np.mean(costs)) * _PRODUCTION_D04_COLUMNS / 1e3
    assert projected > 60.0, (
        f"one 360,000-column land-surface call projects to {projected:.0f} s "
        "on the HOST path, which is fast enough that the reason the device "
        "path exists should be re-measured rather than trusted")


def test_the_width_rail_covers_noahmp_and_not_ruc() -> None:
    """A decision now, not a gap, and it is recorded either way.

    ``validate_run_config`` hands the readiness authority ``columns=nx*ny``
    and the authority refuses Noah-MP above its measured ceiling unless the
    caller states a budget.  RUC has no such rail and, as of the template,
    that is deliberate.

    The rail's PREMISE changed on 2026-07-27 and this docstring changed
    with it.  When Noah-MP's column ran on the host (7.18 ms per land
    column, flat, 43 minutes for one d04 call) the rail refused a call
    that could not finish; since the slab orchestration the same d04 call
    is MEASURED at 0.202-0.227 s (7.3-8.2 wall seconds per simulated
    minute), so the ceiling moved to that measured width -- 360,000
    columns -- and what the rail refuses now is a width NOTHING HAS
    MEASURED, exactly the "largest measured configuration and nothing
    wider" rule it always encoded.  RUC's d04 call is measured at 0.47 s
    warm and 1.83 s over a full snow pack -- 16.8 and 65.7 wall seconds
    per simulated minute -- disclosed in the registry template's own
    warnings; neither scheme's rail refuses production width today, and
    the control below proves the authority still CAN refuse on width by
    probing one column past the measured ceiling.

    If RUC's cost regresses -- if the device path stops being reached, say --
    this test still passes and the one above it does not, which is the right
    way round: the flatness gate measures the host loop, and
    ``tests/test_ruc_runtime.py`` booby-traps the host leaves so a forecast
    that quietly fell back to them fails outright.
    """
    from gpuwm.physics_compat import (
        NOAHMP_MEASURED_COLUMN_CEILING, pending_wrf_physics_components)

    selection = dict(mp_physics=6, sf_sfclay_physics=91, bl_pbl_physics=1)

    ruc = pending_wrf_physics_components(
        sf_surface_physics=3, num_soil_layers=9,
        columns=_PRODUCTION_D04_COLUMNS, **selection)
    assert ruc == (), (
        "RUC now has a width blocker; docs/wrf_ruc_runtime_admission.md lists "
        f"its absence as an open item and must be corrected: {ruc}")

    # Noah-MP's measured ceiling IS production width now, so d04 passes...
    noahmp_at_width = pending_wrf_physics_components(
        sf_surface_physics=4, num_soil_layers=4,
        columns=_PRODUCTION_D04_COLUMNS, **selection)
    assert not any(blocker.component == "Noah-MP column budget"
                   for blocker in noahmp_at_width), (
        "the rail refuses the width the slab path was measured at, so the "
        f"ceiling and the measurement have drifted apart: {noahmp_at_width}")

    # ... and the control: one column past the measured ceiling, the
    # authority still refuses on width, with the rail this test exists to
    # record.
    noahmp = pending_wrf_physics_components(
        sf_surface_physics=4, num_soil_layers=4,
        columns=NOAHMP_MEASURED_COLUMN_CEILING + 1, **selection)
    assert any(blocker.component == "Noah-MP column budget"
               for blocker in noahmp), (
        "the readiness authority did not refuse Noah-MP past its measured "
        f"ceiling, so the RUC result above is not evidence of anything: "
        f"{noahmp}")


# ---------------------------------------------------------------------------
# V-1: RUC is selectable on the GFS route and cannot complete a forecast
# ---------------------------------------------------------------------------
# A node-7 validation run selected the RUC template through the GFS front
# door.  It PREPARED cleanly -- proof PASS, `land_surface: ruc-lsm`, nine
# soil layers, 339 MB of prepared state -- and then died 2.8 s into
# integration with `mavail must be finite`, at
# gpuwm/core/ruc.py:_horizontal_float_field on the first surface-temperature
# call, with `model_elapsed_seconds: 0.0`.  It never advanced a step.
#
# The fail-closed guard did its job and the partial output was labelled
# PARTIAL_NOT_RUN_PASS, so nothing garbage was produced.  But selectable
# and unusable is the pattern this project refuses to ship: the refusal
# belongs at preparation, before anyone spends the prepare run to find out.


def test_the_gfs_route_is_deliberately_not_offered_ruc():
    """Both halves, as the mapped-source test above does it.

    The route list must omit it, AND the front door must actually refuse
    -- a declaration nothing enforces is how RUC came to be preparable
    on this route in the first place.
    """
    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    single = registry["runner_routes"]["tools.prepared_single_domain_forecast"]
    assert RUC_TEMPLATE_ID not in single["source_template_ids"]["gfs"], (
        "the GFS route lists the RUC template, but a GFS-initialised RUC "
        "forecast cannot complete its first step")


def test_the_gfs_front_door_refuses_ruc_at_preparation(tmp_path):
    """Before the prepare run, not 2.8 seconds into the forecast."""
    import dataclasses

    import pytest as _pytest

    from gpuwm.experiment import load_experiment
    from gpuwm.gfs_direct import front_door_physics_selection

    config = (Path(__file__).parents[1] / "configs"
              / "gfs_wrf_hierarchy_proof.toml")
    baseline = load_experiment(config)

    from gpuwm.physics_compat import single_domain_runtime_switches

    # The RUC template's OWN switches, so the only thing under test is
    # the route gate.  Hand-picking `sf_surface_physics = 3` on top of a
    # descriptor written for Noah trips an unrelated capability blocker
    # first and proves nothing about this gate.
    switches = single_domain_runtime_switches(RUC_TEMPLATE_ID)

    def _with_ruc(exp):
        domains = tuple(
            dataclasses.replace(domain, run=dataclasses.replace(
                domain.run, **{
                    name: value for name, value in switches.items()
                    if hasattr(domain.run, name)}))
            for domain in exp.domains)
        return dataclasses.replace(exp, domains=domains)

    # The tree route, which resolves selectors rather than a profile.
    with _pytest.raises(ValueError, match="RUC"):
        front_door_physics_selection(_with_ruc(baseline))

    # And the single-domain route, which names the template.
    single = dataclasses.replace(baseline, domains=baseline.domains[:1])
    with _pytest.raises(ValueError, match="RUC") as caught:
        front_door_physics_selection(
            _with_ruc(single), physics_profile=RUC_TEMPLATE_ID)
    message = str(caught.value)
    # It cites the declaration it enforces, names what was observed, and
    # says which sources are NOT being spoken for.
    assert "physics_registry_v2.json#/runner_routes/" in message
    assert "mavail" in message
    assert "ERA5" in message and "HRRR" in message


def test_era5_and_hrrr_ruc_are_left_alone_because_they_are_untested():
    """Untested is not the same as broken, and must not be gated as if.

    Node 7 exercised RUC through the GFS door only.  Gating ERA5 or HRRR
    on the inference that they share the defect would refuse paths no
    one has shown to be broken -- the same mistake, in the other
    direction, as the F21 regression this release exists to fix.
    """
    from gpuwm.physics_registry import physics_registry

    registry = physics_registry()
    routes = registry["runner_routes"]
    era5 = routes["tools.prepared_single_domain_forecast"]
    assert RUC_TEMPLATE_ID in era5["source_template_ids"]["era5"]
    hrrr = routes["tools.hrrr_single_domain_benchmark"]
    assert RUC_TEMPLATE_ID in hrrr["source_template_ids"]["hrrr"]

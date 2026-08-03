"""Composition gates for the ``mp_physics=28`` forecast adapter (WP-09).

Every kernel this adapter calls was gated by its own package against the
Fortran oracle before the adapter existed.  Those gates are necessary and not
sufficient: a package can reproduce its own block exactly and still be wired
into the call graph wrongly.  This module owns the five properties that only
the composition can have, plus the end-to-end oracle gate (G3).

WHAT IS PINNED HERE, AND WHY EACH IS INVISIBLE TO A UNIT TEST
------------------------------------------------------------
1. ACCUMULATOR ZEROING.  ``ncten``/``nwfaten``/``nifaten`` live in persistent
   ``state.scratch`` slots that survive across steps by design
   (``gpuwm/core/state.py:710-735``).  A unit test allocates a fresh zeroed
   array for every call and therefore *cannot* observe a missing ``fill(0)``.
   ``test_accumulators_are_zeroed_at_entry_not_carried_between_calls`` poisons
   the three slots between two otherwise identical calls on real device
   kernels and requires bit-identical output.

2. WARM/COLD ENTRY-MASK DISJOINTNESS.  Both kernel headers assert their gates
   are exact complements
   (``thompson_aerosol_warm.cu:68-82``, ``thompson_aerosol_cold.cu:311``).
   That is only true if the warm mask is captured on the ENTRY temperature.
   The cold network writes latent heating in place, so capturing the mask
   after it lets a cell it heated across 0 C run BOTH halves, double-counting
   ``pnc_wau``/``pnc_rcw``/``pna_rca``/``pnd_rcd``.  Neither kernel can see
   this; only the adapter's statement order decides it.

3. LAUNCH ORDER.  The ``ncten`` balance limiter (:2996-3019) is a standalone
   launch precisely so it can run ONCE, between the warm network and the
   saturation adjustment.  Surface emission must follow the terminal apply
   (mp_gt_driver:1310-1327) and is deliberately unclamped.  Both are pure
   sequencing facts.

4. NO COOPER.  mp=28 SUBSTITUTES ``iceDeMott`` for Cooper nucleation
   (:2537-2551 selects one or the other on ``is_aerosol_aware``).  Launching a
   Cooper-bearing classic kernel would ADD a second ice source; the result
   stays finite and plausible.  This is pinned at the launcher level AND at
   the CUDA-symbol level, by recording every ``get_kernel`` request a real
   device call makes -- on a not-due call AND on a reflectivity-due one,
   which is the path the G3 gate itself runs.

5. WHICH DENSITY FEEDS WHICH KERNEL.  WRF carries TWO air densities across
   the condensation seam and uses both: :3242-3243 builds the working rain
   moments from the :3193 (pre-condensation) density while :3505-3520 reads
   the :3490 (post-condensation) one.  Every kernel involved exposes exactly
   what it needs, so only the adapter's wiring can get this wrong, and
   getting it wrong scales rain evaporation by ``rho_post/rho_pre``.  No
   unit test can see it; it was worth 1.9e-03 on aero-reduces-to-classic
   until WP-12a, and it was the port's ONLY carved-out end-to-end tolerance.

WHAT THE END-TO-END GATE COMPARES (WP-12a widened it)
------------------------------------------------------
Twenty-three quantities per fixture: the 15 prognostic column fields, the
SEVEN surface diagnostics WRF's driver writes at mp_gt_driver:1298-1308, and
the ``refl_dbz`` column unmodified WRF ``calc_refl10cm`` produced.  Before
WP-12a it compared the 15 column fields and RAINNC and nothing else, so
``snownc``/``snowncv``/``graupelnc``/``graupelncv``/``sr`` and every dBZ in
the deck -- up to 51.98 -- were unmeasured by this gate.  Widening it found a
live defect in the SR expression immediately, and
:func:`test_the_g3_gate_leaves_no_fixture_column_uncompared` now derives the
coverage claim from the fixtures' own CSV headers so it cannot silently
shrink again.

The host gates (1-4 at the call-graph level, plus the guards and the scratch
registry agreement) run with NO GPU: they drive the real adapter over a
NumPy-backed state with recording launchers, exactly as
``tests/test_thompson_adapter_composition.py`` does for mp=8.  The device
gates import cupy inside the test function so ``conftest``'s AST marker keeps
the host gates unmarked.

THE ORACLE HARNESS AND ITS ENTRY-STATE RECONSTRUCTION
-----------------------------------------------------
``_apply_thompson_aerosol`` derives temperature and layer depth from the
prognostic state exactly as mp=8 does::

    th   = thb + thp
    pii  = (p/P0)**RCP
    T    = th * pii
    z8w  = (phb + php)/G ;  dz = z8w[1:] - z8w[:-1]

so driving it from a column fixture means solving for ``thp`` and ``php``
whose float32 round trip reproduces the fixture's ``temp_k`` and ``dz_m``
BIT-EXACTLY.  That is not pedantry: the sed package measured that a
one-float32-ulp shift in the saturation fit flips ``module_mp_thompson.F``'s
``ssatw > 1.E-15`` condensation gate (:3401, eps at :185) at every cloud-free
level of a saturated column.  A one-ulp temperature error is a ~20-ulp
``qvs`` error and would do the same, so an approximate reconstruction would
make the whole G3 measurement meaningless.

:func:`_reconstruct_entry_state` therefore searches:

* a base height offset ``z0`` such that every ``z8w`` level is in the image
  of ``phi -> float32(phi/G)`` and every float32 difference equals the
  fixture's ``dz`` exactly -- measured: all 22 fixtures solve, ``dz`` exact;
* per level, a ``thp`` with ``float32(thp*pii) == temp_k`` exactly, allowing
  the pressure to move by a few float32 ulps only where no ``thp`` exists at
  the fixture pressure.

The pressure perturbation is REPORTED, never hidden, and it is now ZERO on
every fixture at every level -- MEASURED, and asserted for equality below.
The previous wave recorded it as up to 15 float32 ulps at 10 of 24 levels on
three fixtures; that number is quoted from that wave's own record and cannot
be re-measured from this tree, because the cause was fixed AT THE SOURCE.
``tools/thompson_wrf461_oracle/run_column_aero.F90`` formed its Exner
function with ``rcp = 287.0/1004.0`` (its own comment at :33-46 states this)
while gpuwm uses
``RD/CP = 287.0/1004.5 == 2./7.`` exactly (gpuwm/core/constants.py:32,
share/module_model_constants.F:19-20 and :31).  The harness was corrected at
the source and the fixtures regenerated (run_column_aero.F90:33-46 carries the
finding); every one of the fixtures now reconstructs with the fixture pressure
BIT-UNTOUCHED at every level.  :data:`_ENTRY_STATE_PERTURBATION` records that
per fixture and
:func:`test_entry_state_reconstruction_is_exact_in_temperature_and_dz` asserts
it EXACTLY, so a harness change that starts perturbing the entry state again
fails loudly instead of quietly widening every other number in this file.

WHICH FIXTURES G3 DRIVES.  ``_FIXTURES`` is a glob over
``gpuwm/data/thompson/oracle-aero/*-column.csv`` and is deliberately NOT a
hand-written list: it currently resolves to TWENTY-TWO columns, the nineteen
scenarios MP28_PORT_SPEC.md specifies (ids 101-119, the ``aero-*`` names) plus
three WP-08 columns (ids 120-122: ``wp08-nusweep``, ``wp08-melt``,
``wp08-freeze``) that the SAME ``build_aero.sh`` invocation produced in the
same format -- see tools/thompson_wrf461_oracle/build_aero.sh:154 and
README-AEROSOL.md:177-179.  They pin every reachable ``nu_c`` (3..15) and both
branches of the terminal phase cleanup (:3947-3953 MELT, :3956-3965 FREEZE),
so dropping them from the gate would lose real coverage.
The published narrative -- the registry, PHYSICS.md and the column-evidence
page -- still says "nineteen", which is the size of the SPEC'd set; this gate
drives all twenty-two and its counts below are stated over all twenty-two,
with the aero-only subtotal given alongside so the two can be reconciled.

TOLERANCE POLICY (MP28_PORT_SPEC.md, "Validation plan")
-------------------------------------------------------
``activ_ncloud`` selects a NEAREST 10 K temperature bin and ``idx_d``/
``idx_c``/``idx_n`` are INT truncations, so activated ``nc`` is a STEP
function of state: near a bin edge an FP32 GPU port and the Fortran reference
can land in different bins and differ by tens of percent in ``nc`` while every
mass field agrees.  The committed fixtures are chosen away from bin edges.
That behaviour is documented here and pinned by
:func:`test_activation_bin_edge_policy_is_documented_not_absorbed`; it is NOT
absorbed into a loose global tolerance, and no bound in
:data:`_END_TO_END_BOUNDS` exists to accommodate it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

_REPO = Path(__file__).parents[1]
_ORACLE_AERO = _REPO / "gpuwm" / "data" / "thompson" / "oracle-aero"

f32 = np.float32
_INF = f32(np.inf)

#: module_mp_thompson.F:183.
R1 = 1.0e-12


# ===========================================================================
# The recorded call graph (host, no device).
# ===========================================================================

#: Aerosol launchers, keyed by the module that owns them.  Taken from the
#: facade so a launcher that moves owners cannot silently escape the recorder.
def _aerosol_launcher_map():
    from gpuwm.core.thompson_aerosol import AEROSOL_LAUNCHERS
    return dict(AEROSOL_LAUNCHERS)


class _HostState:
    """A NumPy-backed stand-in with exactly the mp=28 DomainState surface.

    Deliberately NOT a DomainState: the host gates must run with no CUDA
    device, and the point of these tests is the adapter's statement order,
    not its arithmetic.  Every attribute below is one the adapter reads; a
    missing one raises AttributeError rather than being silently skipped.
    """

    def __init__(self, nz: int = 3, ny: int = 1, nx: int = 1) -> None:
        shape = (nz, ny, nx)
        face = (nz + 1, ny, nx)
        self.p = np.full(shape, 80000.0, f32)
        self.thb = np.full((nz,), 300.0, f32)
        self.thp = np.zeros(shape, f32)
        self.phb = np.asarray(
            [0.0] + [9810.0 * (k + 1) for k in range(nz)], f32)
        self.php = np.zeros(face, f32)
        self.w = np.zeros(face, f32)
        self.qv = np.full(shape, 0.005, f32)
        self.qc = np.full(shape, 2.0e-4, f32)
        self.qr = np.full(shape, 1.0e-4, f32)
        self.nr = np.full(shape, 2.0e5, f32)
        self.qi = np.full(shape, 1.0e-6, f32)
        self.ni = np.full(shape, 1.0e4, f32)
        self.qs = np.full(shape, 2.0e-5, f32)
        self.qg = np.full(shape, 3.0e-5, f32)
        self.nc = np.full(shape, 1.0e8, f32)
        self.nwfa = np.full(shape, 1.0e8, f32)
        self.nifa = np.full(shape, 1.0e5, f32)
        self.nwfa2d = np.full((ny, nx), 1.0e4, f32)
        self.nifa2d = np.zeros((ny, nx), f32)
        self.effc = np.zeros(shape, f32)
        self.effi = np.zeros(shape, f32)
        self.effs = np.zeros(shape, f32)
        self.h_diabatic = np.zeros(shape, f32)
        self._scratch: dict[str, np.ndarray] = {}

    def scratch(self, shape, slot, dtype=None):
        shape = tuple(shape)
        want = np.dtype(np.float32 if dtype is None else dtype)
        value = self._scratch.get(slot)
        if value is None:
            value = np.zeros(shape, want)
            self._scratch[slot] = value
        else:
            assert value.shape == shape, (slot, value.shape, shape)
            assert value.dtype == want, (slot, value.dtype, want)
        return value

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def _record_adapter_call(monkeypatch, *, refl_due: bool = False,
                         state: _HostState | None = None,
                         delegate: tuple[str, ...] = (
                             "zero_aerosol_accumulators",)):
    """Run the real adapter with every launcher replaced by a recorder.

    ``delegate`` names launchers that are recorded AND actually executed;
    ``zero_aerosol_accumulators`` is the default because several gates need
    to observe its real effect on the host arrays.
    """
    import gpuwm.core.microphysics_aerosol as adapter
    import gpuwm.core.refl as refl
    import gpuwm.core.thompson as thompson
    import gpuwm.core.thompson_aerosol_runtime as aerosol_runtime
    import gpuwm.core.thompson_runtime as classic_runtime

    calls: list[tuple[str, tuple, dict]] = []

    def spy(name, real=None):
        def launch(*args, **kwargs):
            calls.append((name, args, dict(kwargs)))
            if real is not None:
                return real(*args, **kwargs)
            return None
        return launch

    for module_name, names in _aerosol_launcher_map().items():
        module = sys.modules[module_name] if module_name in sys.modules else (
            __import__(module_name, fromlist=["_"]))
        for name in names:
            real = getattr(module, name) if name in delegate else None
            monkeypatch.setattr(module, name, spy(name, real))

    # EVERY public classic launcher is recorded, not just the eight the
    # adapter is supposed to use.  That is what makes the Cooper gate a
    # statement about the whole call graph rather than about a chosen list.
    for name in thompson.__all__:
        if callable(getattr(thompson, name, None)):
            monkeypatch.setattr(thompson, name, spy(name))

    classic_owner = SimpleNamespace(
        name="classic-table-owner", roundtrip_verified=True,
        arrays={"tnc_wev": "tnc_wev-array"})
    aerosol_owner = SimpleNamespace(
        name="aerosol-table-owner",
        ccn_activation_table="tnccn_act-array")
    monkeypatch.setattr(
        classic_runtime, "load_classic_device_tables",
        spy("load_classic_device_tables",
            lambda _root: classic_owner))
    monkeypatch.setattr(
        aerosol_runtime, "load_aerosol_device_tables",
        spy("load_aerosol_device_tables", lambda _root: aerosol_owner))
    monkeypatch.setattr(
        aerosol_runtime, "device_drop_evaporation_number_table",
        spy("device_drop_evaporation_number_table",
            lambda owner: owner.arrays["tnc_wev"]))

    monkeypatch.setattr(adapter, "cp", np)
    monkeypatch.setattr(
        adapter, "save_pre_mp_theta", spy("save_pre_mp_theta"))
    monkeypatch.setattr(
        adapter, "moist_physics_finish", spy("moist_physics_finish"))
    monkeypatch.setattr(
        refl, "compute_and_stash_refl_10cm", spy("reflectivity"))
    monkeypatch.setattr(
        adapter, "_thompson_table_root", lambda: "host-call-graph-fixture")

    state = _HostState() if state is None else state
    cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0)
    diagnostics = adapter._apply_thompson_aerosol(
        state, cfg, 10.0, refl_10cm_due=refl_due)
    return calls, state, diagnostics, classic_owner, aerosol_owner


def _names(calls):
    return [name for name, _, _ in calls]


def _one(calls, name):
    matches = [call for call in calls if call[0] == name]
    assert len(matches) == 1, (name, _names(calls))
    return matches[0]


def _index(calls, name):
    return _names(calls).index(name)


# ---------------------------------------------------------------------------
# (3) LAUNCH ORDER.
# ---------------------------------------------------------------------------

#: MP28_PORT_SPEC.md's WP-09 order, which is WRF ``mp_thompson``'s order.
#: Every placement below is annotated in gpuwm/core/microphysics_aerosol.py
#: with the module_mp_thompson.F line that forces it.
_EXPECTED_ORDER = (
    "load_classic_device_tables",
    "load_aerosol_device_tables",
    "device_drop_evaporation_number_table",
    "zero_aerosol_accumulators",
    "save_pre_mp_theta",
    "launch_aerosol_entry_snapshot",
    "launch_aerosol_entry_cloud_number",
    "launch_classic_graupel_number_init",
    "launch_aa_cold_network_from_owner",
    "launch_aerosol_warm_source_network_from_owner",
    "launch_ncten_balance",
    "launch_hydrometeor_column_mask",
    "launch_hydrometeor_column_mask",
    "launch_graupel_fallout_column_mask",
    "launch_tau1_density",
    "launch_aerosol_working_number",
    "launch_aerosol_saturation_adjust",
    "launch_aerosol_rain_evaporation",
    "launch_aa_cloud_sedimentation",
    "launch_ice_sedimentation",
    "launch_snow_sedimentation",
    "launch_graupel_sedimentation",
    "launch_rain_sedimentation",
    "launch_aa_final_phase_cleanup",
    "launch_classic_graupel_number_finalize",
    "launch_aerosol_state_finalize",
    "launch_aerosol_effective_radius",
    "launch_aerosol_surface_emission",
    "moist_physics_finish",
)


def test_launcher_call_order_is_exactly_the_wrf_driver_order(monkeypatch):
    """The full ordered call graph, pinned as a receipt.

    An equality on the whole list, not a set of pairwise "A before B"
    assertions: the failure modes this must catch include a launcher that
    silently disappears (a set membership test would still pass if the
    remaining order were right) and one that is issued twice.
    """
    calls, _, _, _, _ = _record_adapter_call(monkeypatch)
    assert _names(calls) == list(_EXPECTED_ORDER)


def test_reflectivity_is_issued_only_when_due_and_never_reorders_the_call(
        monkeypatch):
    """REFL_10CM is a cadence decision, never a physics decision.

    module_mp_thompson.F:5710 ``calc_refl10cm`` takes no ``nc`` argument and
    never re-reads ``rc`` after :5764, so cloud water and droplet number
    contribute exactly zero to it.  It must therefore slot in between the
    terminal apply and the effective radii without perturbing anything else.
    """
    due, _, _, _, _ = _record_adapter_call(monkeypatch, refl_due=True)
    names = _names(due)
    assert names.count("reflectivity") == 1
    assert (names.index("launch_aerosol_state_finalize")
            < names.index("reflectivity")
            < names.index("launch_aerosol_effective_radius"))
    without = [n for n in names if n != "reflectivity"]
    assert without == list(_EXPECTED_ORDER)


def test_ncten_balance_runs_once_between_the_warm_network_and_condensation(
        monkeypatch):
    """module_mp_thompson.F:2996-3019 runs ONCE per column, after every
    ``ncten`` source and before the saturation adjustment's ``pnc_wcd``.

    The limiter OVERWRITES ``ncten`` where a clamp fires -- it backs the
    tendency out against ``nc1d*rho`` rather than adding to it -- so a second
    application is not idempotent.  Calling it from inside both networks (the
    obvious "put it where it is used" refactor) leaves every value finite and
    plausible and is wrong.
    """
    calls, _, _, _, _ = _record_adapter_call(monkeypatch)
    names = _names(calls)
    assert names.count("launch_ncten_balance") == 1
    assert (names.index("launch_aa_cold_network_from_owner")
            < names.index("launch_aerosol_warm_source_network_from_owner")
            < names.index("launch_ncten_balance")
            < names.index("launch_aerosol_saturation_adjust"))

    # It must receive the ENTRY cloud mass and the CURRENT one as two
    # distinct arrays, the ENTRY droplet number, and the ENTRY density
    # (:1802) -- not the TAU+1 density of :3193, which does not exist yet at
    # this point in the call and is a different quantity.
    _, args, _ = _one(calls, "launch_ncten_balance")
    qc_entry, qc_after, nc_entry, density, ncten, _dt = args
    _, state, _, _, _ = _record_adapter_call(monkeypatch)
    assert qc_entry is not qc_after
    assert nc_entry is not ncten
    assert density is not state._scratch["mp_thompson_aero_tau1_density"]


def test_surface_emission_is_the_last_aerosol_write_and_follows_the_finalize(
        monkeypatch):
    """mp_gt_driver:1310-1327 emits AFTER ``mp_thompson`` returns.

    WRF has already applied its terminal 9999.E6 ceiling by then, so the
    emission is deliberately unclamped and the aerosol fields may legitimately
    exceed the ceiling between here and the next call's entry pack
    (:1805-1806).  Ordering it before the finalize would let the terminal
    clamp eat the emission on every step -- a silent, monotone drain of the
    boundary-layer aerosol budget.
    """
    calls, _, _, _, _ = _record_adapter_call(monkeypatch)
    names = _names(calls)
    aerosol_writes = [
        "launch_aerosol_state_finalize", "launch_aerosol_surface_emission"]
    assert [n for n in names if n in aerosol_writes] == aerosol_writes
    assert names.index("launch_aerosol_surface_emission") == (
        len(names) - 2), "only moist_physics_finish may follow the emission"


# ---------------------------------------------------------------------------
# (1) THE ACCUMULATOR CONTRACT.
# ---------------------------------------------------------------------------

def test_accumulators_are_zeroed_before_any_kernel_can_read_them(monkeypatch):
    """``zero_aerosol_accumulators`` precedes every launcher that touches
    ``ncten``/``nwfaten``/``nifaten``.

    The three slots are persistent scratch.  ``state.scratch`` keeps its
    buffer for the lifetime of the domain (``gpuwm/core/state.py:710-735``),
    so without the explicit zeroing the first thing the cold network sees is
    the previous STEP's tendency.  The result stays bounded -- the terminal
    clamp at :3972-4021 bounds it -- and drifts.
    """
    calls, state, _, _, _ = _record_adapter_call(monkeypatch)
    names = _names(calls)
    zero_at = names.index("zero_aerosol_accumulators")
    first_reader = min(
        names.index(name) for name in (
            "launch_aa_cold_network_from_owner",
            "launch_aerosol_warm_source_network_from_owner",
            "launch_ncten_balance",
            "launch_aerosol_working_number",
            "launch_aerosol_saturation_adjust",
            "launch_aerosol_rain_evaporation",
            "launch_aa_cloud_sedimentation",
            "launch_aa_final_phase_cleanup",
            "launch_aerosol_state_finalize"))
    assert zero_at < first_reader

    _, args, _ = _one(calls, "zero_aerosol_accumulators")
    assert [id(a) for a in args] == [
        id(state._scratch[slot]) for slot in (
            "mp_thompson_aero_ncten",
            "mp_thompson_aero_nwfaten",
            "mp_thompson_aero_nifaten")]


def test_the_three_accumulators_are_distinct_buffers_and_never_alias_state(
        monkeypatch):
    """Distinct slots, and none of them aliases ``nc``/``nwfa``/``nifa``.

    An accumulator aliasing its own entry state would make the "read-only
    entry state" guarantee silently false: the cold network would be adding
    a tendency to the very array the warm network then reads as ``nc1d``.
    """
    _, state, _, _, _ = _record_adapter_call(monkeypatch)
    accumulators = [state._scratch[slot] for slot in (
        "mp_thompson_aero_ncten", "mp_thompson_aero_nwfaten",
        "mp_thompson_aero_nifaten")]
    assert len({id(a) for a in accumulators}) == 3
    entry = [state.nc, state.nwfa, state.nifa]
    assert not ({id(a) for a in accumulators} & {id(e) for e in entry})


def test_entry_state_is_read_only_until_the_single_terminal_apply(monkeypatch):
    """``state.nc``/``nwfa``/``nifa`` reach every kernel unchanged.

    module_mp_thompson.F freezes ``nc1d``/``nwfa1d``/``nifa1d`` at :1795-1830
    and applies the three accumulators exactly ONCE at :3972-4021, with the
    only clamps in the scheme.  Nine launchers take one of those arrays; all
    nine must receive the SAME object, and only two launchers may write it:
    the entry droplet diagnosis (:1844-1845's zeroing, which is what MAKES it
    the entry value) and the terminal apply.
    """
    calls, state, _, _, _ = _record_adapter_call(monkeypatch)
    identity = {"nc": id(state.nc), "nwfa": id(state.nwfa),
                "nifa": id(state.nifa)}

    def positional(name):
        return _one(calls, name)[1]

    # cold network: (..., nc_entry, nwfa_entry, nifa_entry, ncten, ...)
    cold = positional("launch_aa_cold_network_from_owner")
    assert [id(a) for a in cold[10:13]] == [
        identity["nc"], identity["nwfa"], identity["nifa"]]
    warm = positional("launch_aerosol_warm_source_network_from_owner")
    assert [id(a) for a in warm[11:14]] == [
        identity["nc"], identity["nwfa"], identity["nifa"]]
    assert id(positional("launch_ncten_balance")[2]) == identity["nc"]
    assert id(positional("launch_aerosol_working_number")[0]) == (
        identity["nwfa"])
    assert id(positional("launch_aerosol_saturation_adjust")[4]) == (
        identity["nc"])
    assert id(positional("launch_aa_cloud_sedimentation")[1]) == (
        identity["nc"])
    assert id(positional("launch_aa_final_phase_cleanup")[4]) == (
        identity["nc"])
    finalize = positional("launch_aerosol_state_finalize")
    assert [id(a) for a in finalize[1:4]] == [
        identity["nc"], identity["nwfa"], identity["nifa"]]
    # The terminal apply writes the same three arrays it read (documented as
    # legal: every thread reads its own element before writing it).
    assert [id(a) for a in finalize[9:12]] == [
        identity["nc"], identity["nwfa"], identity["nifa"]]

    # The entry snapshot receives the per-kilogram state, never a per-m3
    # scratch: :1805-1806 is where the multiplication by rho happens.
    snapshot = positional("launch_aerosol_entry_snapshot")
    assert [id(a) for a in snapshot[3:5]] == [
        identity["nwfa"], identity["nifa"]]


# The complementary statement -- that NOTHING ELSE writes the three entry
# arrays -- cannot be made from a recorded host call graph, because a recorder
# never executes the kernel that would do the writing.  It is made on real
# device kernels instead, by
# test_the_terminal_apply_is_the_only_writer_of_nc_nwfa_and_nifa below.


# ---------------------------------------------------------------------------
# (2) WARM / COLD ENTRY-MASK DISJOINTNESS.
# ---------------------------------------------------------------------------

def test_warm_entry_mask_is_captured_before_the_cold_network(monkeypatch):
    """The mask the warm network consumes must hold the ENTRY temperature.

    ``thompson_aerosol_cold.cu:311`` returns for ``temperature >= 273.15``
    reading the LIVE array, and ``thompson_aerosol_warm.cu:602-608`` returns
    unless its held mask is set.  The cold network then writes latent heating
    into that same live temperature array.  So the two gates are exact
    complements only while the mask predates the cold launch; capture it
    afterwards and any cell the cold network warmed across 0 C executes BOTH
    halves, double-counting the four :2160-2230 number/aerosol rates that
    WRF evaluates once per level.

    The test drives the real adapter with a MIXED column -- two sub-freezing
    levels and two warm ones -- and a cold-network stand-in that heats the
    whole column above freezing.  The three possible adapters are then
    distinguishable, which an all-cold column would not make them:

        mask captured at entry (correct)   ->  [0, 1, 0, 1]
        mask captured after the cold call  ->  [1, 1, 1, 1]
        mask never captured at all         ->  [0, 0, 0, 0]

    An assertion that only said "the mask is all zeros" would pass for the
    third of those, which is why it says ``mask == (entry T >= 273.15)``.
    """
    import gpuwm.core.thompson_aerosol_cold as cold_module

    state = _HostState(nz=4)
    state.thb = np.zeros((4,), f32)
    pii = np.power(state.p / f32(1.0e5), f32(2.0 / 7.0)).astype(f32)
    entry_targets = np.asarray(
        [260.0, 280.0, 265.0, 290.0], f32).reshape(4, 1, 1)
    state.thp[...] = (entry_targets / pii).astype(f32)

    def heating_cold_network(*args, **kwargs):
        temperature = args[7]
        temperature[...] = f32(285.0)    # every level crosses 0 C
        return None

    # Installed BEFORE the recorder, so the recorder wraps the stand-in and
    # monkeypatch's own teardown restores the real launcher.
    monkeypatch.setattr(
        cold_module, "launch_aa_cold_network_from_owner",
        heating_cold_network)
    calls, state, _, _, _ = _record_adapter_call(
        monkeypatch, state=state,
        delegate=("zero_aerosol_accumulators",
                  "launch_aa_cold_network_from_owner"))

    # The stand-in really ran and really did carry every level across 0 C.
    temperature = state._scratch["mp_thompson_temperature"]
    assert float(temperature.min()) >= 273.15

    _, warm_args, _ = _one(
        calls, "launch_aerosol_warm_source_network_from_owner")
    mask = warm_args[6]
    assert mask is state._scratch["mp_thompson_graupel_melt_marker"]
    expected = (entry_targets >= f32(273.15)).astype(f32)
    assert expected.any() and not expected.all(), (
        "the fixture no longer straddles freezing; this test's whole "
        "discriminating power comes from that")
    np.testing.assert_array_equal(mask, expected)


def test_the_entry_mask_and_the_cold_gate_are_exact_complements(monkeypatch):
    """The seam value 273.15 K belongs to the WARM half, in both kernels.

    ``thompson_aerosol_cold.cu:311`` is ``if (temperature >= 273.15f)
    return;`` and the adapter's mask is ``temperature >= 273.15``.  Both use
    ``>=``, so 273.15 K itself is serviced exactly once, by the warm network.
    A ``>`` on either side would double-service it and a ``<`` would drop it;
    both are invisible in any single-kernel test.
    """
    state = _HostState()
    state.thb = np.zeros((3,), f32)
    # Pick thp so the derived temperature straddles the seam exactly.
    pii = np.power(state.p / f32(1.0e5), f32(2.0 / 7.0)).astype(f32)
    targets = np.asarray([273.15, 273.0, 274.0], f32).reshape(3, 1, 1)
    state.thp[...] = (targets / pii).astype(f32)
    calls, state, _, _, _ = _record_adapter_call(monkeypatch, state=state)
    temperature = state._scratch["mp_thompson_temperature"]
    mask = _one(calls, "launch_aerosol_warm_source_network_from_owner")[1][6]
    np.testing.assert_array_equal(
        mask.astype(bool), temperature >= f32(273.15))

    source = (_REPO / "gpuwm" / "core" / "kernels"
              / "thompson_aerosol_cold.cu").read_text(encoding="utf-8")
    assert "if (temperature[idx] >= 273.15f) return;" in source, (
        "the cold kernel's gate changed; the adapter's entry mask is its "
        "exact complement and both must move together")


# ---------------------------------------------------------------------------
# (4) NO COOPER ON THE mp=28 PATH.
# ---------------------------------------------------------------------------

def test_no_cooper_bearing_classic_launcher_is_ever_called(monkeypatch):
    """mp=28 SUBSTITUTES iceDeMott for Cooper; it does not add it.

    module_mp_thompson.F:2537-2551 chooses one or the other on
    ``is_aerosol_aware``.  gpuwm/core/kernels/thompson.cu carries Cooper's
    literal ``MIN(250.E3, 5.0*EXP(0.304*(T_0-temp)))`` at :5931, :6208, :6641,
    :7190 and :7635, inside ``thompson_frozen_vapor_network``,
    ``thompson_frozen_vapor_cloud_network`` and ``thompson_ice_nucleation``.
    Launching any of them here would give an mp=28 column TWO deposition
    nucleation sources -- stable, bounded, and wrong by roughly the Cooper
    curve.

    Every public classic launcher is recorded, so this is a statement about
    the whole call graph.
    """
    from gpuwm.core.thompson_aerosol import (
        COOPER_BEARING_CLASSIC_LAUNCHERS, REUSED_CLASSIC_LAUNCHERS)
    import gpuwm.core.thompson as thompson

    calls, _, _, _, _ = _record_adapter_call(monkeypatch, refl_due=True)
    used = {name for name in _names(calls) if name in set(thompson.__all__)}
    assert not (used & set(COOPER_BEARING_CLASSIC_LAUNCHERS)), sorted(used)
    assert used == set(REUSED_CLASSIC_LAUNCHERS), sorted(used)


def test_the_cooper_launcher_inventory_matches_the_frozen_cuda_source():
    """The list this gate defends is derived from thompson.cu, not asserted.

    A future kernel that grows a Cooper term, or a launcher rename, must
    update ``COOPER_BEARING_CLASSIC_LAUNCHERS`` or fail here.
    """
    from gpuwm.core.thompson_aerosol import COOPER_BEARING_CLASSIC_KERNELS

    source = (_REPO / "gpuwm" / "core" / "kernels" / "thompson.cu").read_text(
        encoding="utf-8")
    lines = source.splitlines()
    starts = [(i, line.split("void ", 1)[1].split("(", 1)[0])
              for i, line in enumerate(lines)
              if line.startswith('extern "C" __global__ void ')]
    owner_of = {}
    for index, (line_no, name) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(lines)
        owner_of[name] = (line_no, end)

    carriers = set()
    for i, line in enumerate(lines):
        if "0.304f" not in line:
            continue
        for name, (start, end) in owner_of.items():
            if start <= i < end:
                carriers.add(name)
    assert carriers, "Cooper's 0.304 coefficient vanished from thompson.cu"
    assert carriers == set(COOPER_BEARING_CLASSIC_KERNELS), sorted(carriers)


def test_the_adapter_source_never_names_a_cooper_bearing_launcher():
    """Belt and braces: the import list itself is checked.

    The recorded-call gate proves it for one state; this proves it for every
    state, including branches a host fixture cannot reach.
    """
    from gpuwm.core.thompson_aerosol import COOPER_BEARING_CLASSIC_LAUNCHERS

    import ast

    tree = ast.parse(
        (_REPO / "gpuwm" / "core" / "microphysics_aerosol.py").read_text(
            encoding="utf-8"))
    # Identifiers only: the module docstring NAMES these launchers, which is
    # the point -- it explains why they are absent.  A string comparison over
    # raw source would make the explanation impossible to write.
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.alias):
            identifiers.add(node.name.split(".")[-1])
            if node.asname:
                identifiers.add(node.asname)
    for name in COOPER_BEARING_CLASSIC_LAUNCHERS:
        assert name not in identifiers, name


# ---------------------------------------------------------------------------
# Guards, scratch budget, facade.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "missing",
    ["qc", "qi", "ni", "qs", "qg", "qr", "nr", "nc", "nwfa", "nifa",
     "nwfa2d", "nifa2d"])
def test_required_state_fields_fail_closed(monkeypatch, missing):
    """An absent mp=28 field raises instead of running a partial scheme.

    ``nwfa2d``/``nifa2d`` are in the guard even though microphysics never
    writes them: they are INTENT(IN) to ``mp_gt_driver`` and the surface
    emission at :1310-1327 reads them unconditionally, so a state without
    them would fail deep inside the last launcher of the call, after every
    field had already been mutated.
    """
    import gpuwm.core.microphysics_aerosol as adapter

    state = _HostState()
    setattr(state, missing, None)
    monkeypatch.setattr(adapter, "cp", np)
    monkeypatch.setattr(
        adapter, "_thompson_table_root", lambda: "host-guard-fixture")
    with pytest.raises(ValueError, match=f"mp=28 state lacks .*{missing}"):
        adapter._apply_thompson_aerosol(
            state, SimpleNamespace(mp_physics=28), 10.0)


def test_scratch_slots_match_the_preflight_registry_exactly(monkeypatch):
    """The adapter's slot set equals what preflight budgets for mp=28.

    Both directions matter.  A slot the adapter draws that preflight does not
    budget is an unpriced allocation behind gpuwm's arena gate -- it appears
    at runtime on a machine sized from the registry.  A slot preflight
    budgets that nothing draws is reserved memory no forecast ever uses, and
    it hides the first kind of defect by making the totals look plausible.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core import preflight
    from gpuwm.core.microphysics_aerosol import (
        AEROSOL_INT_SCRATCH_SLOTS, AEROSOL_SCRATCH_SLOTS)

    _, state, _, _, _ = _record_adapter_call(monkeypatch)
    drawn = set(state._scratch)
    cfg = RunConfig(nx=1, ny=1, nz=3, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=1.0, run_seconds=10.0, moist=True, mp_physics=28)
    budgeted = set(preflight.scratch_slot_registry(cfg))
    assert drawn <= budgeted, sorted(drawn - budgeted)

    aerosol_drawn = {s for s in drawn if s.startswith("mp_thompson_aero_")}
    aerosol_budgeted = {
        s for s in budgeted if s.startswith("mp_thompson_aero_")}
    assert aerosol_drawn == aerosol_budgeted == set(AEROSOL_SCRATCH_SLOTS)
    for slot in AEROSOL_INT_SCRATCH_SLOTS:
        assert state._scratch[slot].dtype == np.int32
    for slot in set(AEROSOL_SCRATCH_SLOTS) - set(AEROSOL_INT_SCRATCH_SLOTS):
        assert state._scratch[slot].dtype == np.float32

    # refl_t/refl_10cm only appear on a due call; everything else must be
    # drawn on every call or the budget is describing a different program.
    due, due_state, _, _, _ = _record_adapter_call(monkeypatch, refl_due=True)
    del due
    assert set(due_state._scratch) <= budgeted


def test_reused_classic_launchers_receive_the_mp8_argument_shape(monkeypatch):
    """The eight frozen launchers get the arguments mp=8 gives them.

    module_mp_thompson.F:3790-3936 has no ``is_aerosol_aware`` branch and no
    nc/nwfa/nifa reference, so "reused unchanged" has to mean the argument
    tuples too.  This compares the mp=28 adapter's calls against the mp=8
    adapter's calls on the same host state, launcher by launcher, by POSITION
    and by which state array or scratch slot each position holds.
    """
    import gpuwm.core.microphysics as microphysics
    import gpuwm.core.refl as refl
    import gpuwm.core.thompson as thompson
    import gpuwm.core.thompson_runtime as classic_runtime

    reused = (
        "launch_classic_graupel_number_init",
        "launch_classic_graupel_number_finalize",
        "launch_hydrometeor_column_mask",
        "launch_graupel_fallout_column_mask",
        "launch_ice_sedimentation",
        "launch_snow_sedimentation",
        "launch_graupel_sedimentation",
        "launch_rain_sedimentation",
    )

    def label(state, value):
        for name in ("qv", "qc", "qr", "qi", "ni", "qs", "qg", "nr", "nc",
                     "nwfa", "nifa", "p", "effc", "effi", "effs"):
            if getattr(state, name, None) is value:
                return f"state.{name}"
        for slot, buf in state._scratch.items():
            if buf is value:
                return f"scratch.{slot}"
        if isinstance(value, np.ndarray):
            return f"array{value.shape}"
        return repr(value)

    aerosol_calls, aerosol_state, _, _, _ = _record_adapter_call(monkeypatch)

    # Now the mp=8 adapter over the same skeleton.
    classic_calls: list = []

    def spy(name):
        def launch(*args, **kwargs):
            classic_calls.append((name, args, dict(kwargs)))
        return launch

    for name in thompson.__all__:
        if callable(getattr(thompson, name, None)):
            monkeypatch.setattr(thompson, name, spy(name))
    monkeypatch.setattr(
        classic_runtime, "load_classic_device_tables", lambda _root: object())
    monkeypatch.setattr(microphysics, "cp", np)
    monkeypatch.setattr(microphysics, "save_pre_mp_theta", spy("prep"))
    monkeypatch.setattr(microphysics, "moist_physics_finish", spy("finish"))
    monkeypatch.setattr(refl, "compute_and_stash_refl_10cm", spy("refl"))
    monkeypatch.setattr(
        microphysics, "_thompson_table_root", lambda: "host-fixture")
    classic_state = _HostState()
    microphysics._apply_thompson(
        classic_state, SimpleNamespace(no_mp_heating=0, mp_tend_lim=10.0),
        10.0)

    def profile(calls, state, name):
        return [(tuple(label(state, a) for a in args),
                 {k: label(state, v) for k, v in kwargs.items()})
                for call_name, args, kwargs in calls if call_name == name]

    for name in reused:
        got = profile(aerosol_calls, aerosol_state, name)
        want = profile(classic_calls, classic_state, name)
        assert got == want, name


def test_the_facade_reexports_are_the_owners_own_objects():
    """``thompson_aerosol`` must be a view, never a copy.

    A facade that wrapped or re-implemented a launcher would make every
    monkeypatch-based gate in this file test the wrapper instead of the
    launcher, and every per-package oracle measurement would stop applying.
    """
    import importlib

    from gpuwm.core import thompson_aerosol as facade

    for module_name, names in facade.AEROSOL_LAUNCHERS.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert getattr(facade, name) is getattr(module, name), name

    import gpuwm.core.thompson as thompson
    for name in facade.REUSED_CLASSIC_LAUNCHERS:
        assert hasattr(thompson, name), name
        assert not hasattr(facade, name), (
            f"{name} is a frozen mp=8 launcher and must be called from its "
            "own module so the adapter shows which lines are reused")
    for name in facade.COOPER_BEARING_CLASSIC_LAUNCHERS:
        assert not hasattr(facade, name), name


def test_init_fill_is_not_a_per_step_call_and_refuses_other_schemes():
    """``thompson_aerosol_init_fill`` is a domain-construction operation."""
    from gpuwm.core.microphysics_aerosol import thompson_aerosol_init_fill

    state = _HostState()
    with pytest.raises(ValueError, match="mp_physics=28"):
        thompson_aerosol_init_fill(state, SimpleNamespace(mp_physics=8))

    source = (_REPO / "gpuwm" / "core" / "microphysics_aerosol.py").read_text(
        encoding="utf-8")
    body = source.split("def _apply_thompson_aerosol", 1)[1].split(
        "def thompson_aerosol_init_fill", 1)[0]
    assert "init_profile" not in body and "init_fill" not in body, (
        "the per-step adapter must never run thompson_init's profile fill")


# ===========================================================================
# The oracle harness (device).
# ===========================================================================

_FIXTURES = tuple(sorted(
    path.name[:-len("-column.csv")]
    for path in _ORACLE_AERO.glob("*-column.csv")))

#: PROGNOSTIC COLUMN fields the fixtures carry that a complete mp=28 call is
#: responsible for.  This tuple is the registry's published field count
#: (``tests/test_physics_registry.py`` asserts ``compared_fields ==
#: len(_END_TO_END_FIELDS) + 1``, the +1 being ``rainnc_mm``); the SURFACE and
#: REFLECTIVITY inventories below are separate for that reason and are
#: compared just as hard.
_END_TO_END_FIELDS = (
    "qv", "qc", "qr", "qi", "qs", "qg", "ni_per_kg", "nr_per_kg",
    "nc_per_kg", "nwfa_per_kg", "nifa_per_kg", "temp_k",
    "effc_m", "effi_m", "effs_m",
)

#: THE SURFACE HALF OF THE GATE (WP-12a).  Every column of the fixtures'
#: ``*-surface.csv`` that mp=28 is responsible for, mapped to the scratch slot
#: the adapter leaves it in.  Before WP-12a the gate compared ``rainnc_mm``
#: and NOTHING ELSE here, although the fixtures carry real signal in all
#: seven: ``snownc`` reaches 9.26e-04 mm and ``graupelnc`` 6.59e-03 mm on
#: aero-scav-frozen, and ``sr`` is a genuine mixed-phase fraction on five
#: fixtures.  Widening it immediately found a defect -- see
#: ``test_the_sr_diagnostic_is_wrfs_epsilon_quotient_not_a_guarded_ratio``.
#:
#: WRF's own definitions, all in one block: RAINNCV/RAINNC at
#: mp_gt_driver:1298-1299, SNOWNCV/SNOWNC at :1301-1302 (``pptsnow + pptice``,
#: i.e. snow AND cloud ice fall out into the "snow" bucket), GRAUPELNCV/
#: GRAUPELNC at :1305-1306, SR at :1308.
_END_TO_END_SURFACE_FIELDS: dict[str, str] = {
    "rainnc_mm": "mp_rainnc",
    "rainncv_mm": "mp_rainncv",
    "snownc_mm": "mp_snownc",
    "snowncv_mm": "mp_snowncv",
    "graupelnc_mm": "mp_graupelnc",
    "graupelncv_mm": "mp_graupelncv",
    "sr": "mp_sr",
}

#: THE REFLECTIVITY HALF (WP-12a).  The fixtures' 23rd column, written by
#: unmodified WRF ``calc_refl10cm`` through the oracle's own
#: ``diagflag=.true., do_radar_ref=1`` call
#: (tools/thompson_wrf461_oracle/run_column_aero.F90:296).  It reaches 51.98
#: dBZ on aero-cold-overlap and 44.30 on aero-scav-frozen, and the G3 gate
#: compared it nowhere before WP-12a.
#:
#: THE METRIC IS ABSOLUTE dB, NOT RELATIVE, and that is not a softening.
#: dBZ is stored in the LOG domain, so its float32 resolution is ~1.9e-06 dB
#: near 25 dBZ and a relative gate on a quantity that crosses zero and runs
#: negative is meaningless.  Both existing reflectivity comparisons in the
#: port are absolute dB for the same reason: the mp=8 composition gate uses
#: 5.0e-02 dB (tests/test_thompson_adapter_composition.py:754) and WP-11a's
#: focused mp=28 gate uses 2.0e-04 dB
#: (tests/test_mp28_runnable.py::_REFL_DBZ_GATE).  This gate is the STRICTER
#: of those two, transcribed.
#:
#: THE PER-FIXTURE CARVE-OUT IS GONE.  It went 1.0e-02 -> 1.0e-03 dB when
#: WP-12a shrank aero-reduces-to-classic's residual from 7.736e-03 dB to
#: 5.283e-04 dB, and WP-13a deleted it outright: restoring WRF's level-wise
#: :3237-vs-:3568 sedimentation density (thompson_aerosol_sat.cu, the
#: `reference_density` write) took that residual to 3.242e-05 dB, INSIDE the
#: flat 2.0e-04 dB gate, so the dict buys nothing and
#: :func:`test_every_g3_allowance_is_named_and_buys_exactly_one_fixture`
#: required its removal.  It is kept as an empty dict, not deleted, so
#: :func:`_g3_bound` keeps one code path and a future carve-out has to be
#: added deliberately.
_REFL_FIELD = "refl_dbz_db"
_REFL_DB_GATE = 2.0e-4
_REFL_DB_BOUNDS: dict[str, float] = {}

#: Columns of the fixtures' ``*-column.csv`` / ``*-surface.csv`` that the gate
#: legitimately does not compare, each with the reason.  Named explicitly so
#: ``test_the_g3_gate_leaves_no_fixture_column_uncompared`` can prove the
#: complement is covered rather than trusting a hand-maintained list.
_UNCOMPARED_FIXTURE_COLUMNS: dict[str, str] = {
    "phase": "row label ('before'/'after'), not data",
    "k": "level index",
    "z_m": "the oracle's synthetic height axis; the adapter derives dz from "
           "the reconstructed geopotential and dz is pinned bitwise by "
           "test_entry_state_reconstruction_is_exact_in_temperature_and_dz",
    "p_pa": "INPUT: reconstructed BITWISE by _reconstruct_entry_state on "
            "every fixture, pinned level by level and fixture by fixture by "
            "test_entry_state_reconstruction_is_exact_in_temperature_and_dz "
            "against _ENTRY_STATE_PERTURBATION, which is uniformly (0, 0.0)",
    "pii": "INPUT: a function of p_pa alone",
    "w_m_s": "INPUT: entry vertical velocity, never written by mp=28",
    "dz_m": "INPUT: pinned bitwise by the reconstruction test",
    "theta_k": "INPUT: temp_k/pii; temp_k itself IS compared",
    "scenario": "surface row label, not data",
    "dt_s": "INPUT: the timestep the adapter is driven with",
    "nwfa2d_kg_s": "INPUT: INTENT(IN) surface emission constant "
                   "(mp_gt_driver:1318); microphysics never writes it",
    "nifa2d_kg_s": "INPUT: INTENT(IN) surface emission constant "
                   "(mp_gt_driver:1319)",
}

#: WRF's mp_gt_driver:1475-1477 clamps, applied to the oracle's raw metres so
#: it can be compared against gpuwm's radiation-facing micron state contract.
_EFFECTIVE_RADIUS_CLAMPS = {
    "effc_m": (f32(2.49e-6), f32(50.0e-6)),
    "effi_m": (f32(4.99e-6), f32(125.0e-6)),
    "effs_m": (f32(9.99e-6), f32(999.0e-6)),
}


def _oracle_case(name):
    with (_ORACLE_AERO / f"{name}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    with (_ORACLE_AERO / f"{name}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    assert len(rows) % 2 == 0 and rows[0]["phase"] == "before"
    half = len(rows) // 2
    return rows[:half], rows[half:], surface


def _column(rows, key):
    return np.asarray([float(row[key]) for row in rows], f32)


def _pii_of(xp, pressure):
    from gpuwm.core import constants as c
    return xp.power(xp.asarray(pressure) / f32(c.P0), f32(c.RCP)).get()


def _solve_theta(target, pii, window: int = 8):
    """float32 ``x`` with ``float32(x*pii) == target``, or ``None``."""
    seed = f32(np.float64(target) / np.float64(pii))
    x = seed
    for _ in range(window):
        if f32(x * pii) == target:
            return x
        x = np.nextafter(x, _INF, dtype=np.float32)
    x = seed
    for _ in range(window):
        x = np.nextafter(x, -_INF, dtype=np.float32)
        if f32(x * pii) == target:
            return x
    return None


def _geopotential_preimage(target, gravity):
    """float32 ``phi`` with ``float32(phi/G) == target``, or ``None``."""
    seed = f32(np.float64(target) * np.float64(gravity))
    candidate = seed
    for _ in range(3):
        if f32(candidate / gravity) == target:
            return candidate
        candidate = np.nextafter(candidate, _INF, dtype=np.float32)
    candidate = seed
    for _ in range(3):
        candidate = np.nextafter(candidate, -_INF, dtype=np.float32)
        if f32(candidate / gravity) == target:
            return candidate
    return None


def _solve_geopotential(dz, gravity, tries: int = 200000):
    """Geopotential column whose float32 round trip reproduces ``dz`` exactly.

    Searched over a base height offset because the requirement is that every
    ``z8w`` level lands in the image of ``phi -> float32(phi/G)``, and whether
    it does depends on the level's binade.  Measured: all 22 committed
    fixtures solve, 19 at the first offset tried (z0 = 0.5000002980232239 m);
    aero-cold-overlap and aero-warm-overlap need z0 = 1788.0225830078125 m and
    wp08-nusweep z0 = 944.457275390625 m.
    """
    z0 = f32(0.0)
    for trial in range(tries):
        levels = [z0]
        phis = [_geopotential_preimage(z0, gravity)]
        ok = phis[0] is not None
        if ok:
            for step in dz:
                nxt = f32(levels[-1] + step)
                if f32(nxt - levels[-1]) != step:
                    ok = False
                    break
                phi = _geopotential_preimage(nxt, gravity)
                if phi is None:
                    ok = False
                    break
                levels.append(nxt)
                phis.append(phi)
        if ok:
            return np.asarray(phis, f32), z0
        z0 = (np.nextafter(z0, _INF, dtype=np.float32) if trial % 2 == 0
              else f32(z0 + f32(0.03125)))
    raise AssertionError("no base height offset reproduces dz exactly")


def _reconstruct_entry_state(xp, before, max_pressure_ulps: int = 64):
    """Solve for the prognostic fields the adapter derives T and dz from.

    Returns ``(pressure, theta, geopotential, report)``.  ``report`` carries
    the measured cost of the reconstruction so a test can publish it.
    """
    from gpuwm.core import constants as c

    pressure = _column(before, "p_pa")
    target_t = _column(before, "temp_k")
    dz = _column(before, "dz_m")
    nz = target_t.size

    candidates = [pressure.copy()]
    up = down = pressure.copy()
    for _ in range(max_pressure_ulps):
        up = np.nextafter(up, _INF, dtype=np.float32)
        down = np.nextafter(down, -_INF, dtype=np.float32)
        candidates.append(up.copy())
        candidates.append(down.copy())
    stacked = np.stack(candidates)
    pii = _pii_of(xp, stacked)

    chosen_p = pressure.copy()
    theta = np.empty(nz, f32)
    perturbed = []
    for k in range(nz):
        solved = None
        for index in range(stacked.shape[0]):
            value = _solve_theta(target_t[k], pii[index, k])
            if value is not None:
                solved = (index, value)
                break
        assert solved is not None, (
            f"no exact (p, theta) pair reproduces temp_k at level {k + 1}")
        index, value = solved
        theta[k] = value
        chosen_p[k] = stacked[index, k]
        if index:
            perturbed.append(
                (k + 1, float(abs(stacked[index, k] - pressure[k])
                              / pressure[k])))

    geopotential, z0 = _solve_geopotential(dz, f32(c.G))
    report = {
        "pressure_perturbed_levels": len(perturbed),
        "max_pressure_relative_perturbation": (
            max((rel for _, rel in perturbed), default=0.0)),
        "base_height_offset_m": float(z0),
    }
    return chosen_p, theta, geopotential, report


class _ColumnState:
    """A device-backed stand-in with the mp=28 ``DomainState`` surface.

    A real ``DomainState`` would be the ideal driver, but it derives its
    prognostic fields from a base state and a vertical coordinate that this
    harness has to invert anyway; using it would add a second inversion
    without adding evidence.  Every attribute the adapter reads is present
    with the real dtype, shape and staggering, so the adapter cannot tell the
    difference -- and the scratch protocol below is
    ``gpuwm/core/state.py:710-745``'s, including the shape/dtype pinning that
    makes a slot collision an error rather than a silent reshape.
    """

    def __init__(self, xp, before, surface, pressure, theta, geopotential):
        self._xp = xp
        nz = len(before)
        shape = (nz, 1, 1)

        def volume(name):
            return xp.asarray(
                _column(before, name).reshape(shape).copy())

        self.p = xp.asarray(pressure.reshape(shape).copy())
        self.thb = xp.zeros((nz,), xp.float32)
        self.thp = xp.asarray(theta.reshape(shape).copy())
        self.phb = xp.zeros((nz + 1,), xp.float32)
        self.php = xp.asarray(geopotential.reshape(nz + 1, 1, 1).copy())
        w = _column(before, "w_m_s")
        self.w = xp.asarray(
            np.concatenate([w, w[-1:]]).reshape(nz + 1, 1, 1).copy())
        self.qv, self.qc, self.qr = volume("qv"), volume("qc"), volume("qr")
        self.qi, self.qs, self.qg = volume("qi"), volume("qs"), volume("qg")
        self.ni, self.nr = volume("ni_per_kg"), volume("nr_per_kg")
        self.nc = volume("nc_per_kg")
        self.nwfa, self.nifa = volume("nwfa_per_kg"), volume("nifa_per_kg")
        self.effc, self.effi = volume("effc_m"), volume("effi_m")
        self.effs = volume("effs_m")
        self.nwfa2d = xp.full(
            (1, 1), f32(surface["nwfa2d_kg_s"]), xp.float32)
        self.nifa2d = xp.full(
            (1, 1), f32(surface["nifa2d_kg_s"]), xp.float32)
        self.h_diabatic = xp.zeros(shape, xp.float32)
        # ``gpuwm.core.refl.stash_refl_10cm`` hands the due REFL_10CM frame to
        # the physics driver and refuses to overwrite an unconsumed one.  The
        # fixtures were produced with ``diagflag=.true., do_radar_ref=1``
        # (tools/thompson_wrf461_oracle/run_column_aero.F90:296), so every G3
        # call is a DUE call and needs somewhere to put the frame.
        self.physics = SimpleNamespace(refl_10cm=None, state=self)
        self._scratch: dict = {}

    def scratch(self, shape, slot, dtype=None):
        xp = self._xp
        shape = tuple(shape)
        want = np.dtype(np.float32 if dtype is None else dtype)
        value = self._scratch.get(slot)
        if value is None:
            value = xp.zeros(shape, dtype=want)
            self._scratch[slot] = value
        else:
            if value.shape != shape:
                raise ValueError(
                    f"scratch slot {slot!r} has shape {value.shape}, "
                    f"requested {shape}")
            if value.dtype != want:
                raise ValueError(
                    f"scratch slot {slot!r} has dtype {value.dtype}, "
                    f"requested {want}")
        return value

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def _build_case(xp, name):
    from gpuwm.physics_compat import thompson_table_root

    before, after, surface = _oracle_case(name)
    pressure, theta, geopotential, report = _reconstruct_entry_state(
        xp, before)
    state = _ColumnState(
        xp, before, surface, pressure, theta, geopotential)
    cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0)
    del thompson_table_root
    return state, cfg, float(surface["dt_s"]), before, after, surface, report


def _adapter_outputs(xp, state, before):
    """Read the fields the fixture records back off the adapter's own state."""
    temperature = state.existing_scratch("mp_thompson_temperature")
    out = {
        "qv": state.qv, "qc": state.qc, "qr": state.qr,
        "qi": state.qi, "qs": state.qs, "qg": state.qg,
        "ni_per_kg": state.ni, "nr_per_kg": state.nr,
        "nc_per_kg": state.nc, "nwfa_per_kg": state.nwfa,
        "nifa_per_kg": state.nifa, "temp_k": temperature,
        "effc_m": state.effc, "effi_m": state.effi, "effs_m": state.effs,
    }
    del before
    return {key: xp.asnumpy(value).ravel().astype(np.float64)
            for key, value in out.items()}


def _oracle_expectation(after, field):
    """The oracle column, in the adapter's own units.

    Only the three effective radii need a transform: the fixture records
    ``calc_effectRad``'s raw metres, while gpuwm's state contract is WRF's
    own ``mp_gt_driver:1475-1477`` clamped MICRON value.  Applying the driver
    clamp to the oracle -- rather than reading a second, unclamped kernel --
    keeps the comparison against exactly what a forecast consumes.
    """
    values = _column(after, field)
    clamp = _EFFECTIVE_RADIUS_CLAMPS.get(field)
    if clamp is None:
        return values.astype(np.float64)
    low, high = clamp
    return (np.maximum(low, np.minimum(values, high)).astype(f32)
            * f32(1.0e6)).astype(np.float64)


def _max_relative(got, want):
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    both_zero = (got == 0.0) & (want == 0.0)
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1.0e-30)
    return float(np.where(both_zero, 0.0, rel).max())


#: The working end-to-end bound.  2e-6 is one part in five hundred thousand
#: across an eighteen-launcher composition and it is NOT slack: gpuwm applies
#: mass tendencies in place where WRF accumulates them and applies once at
#: :3975, so ``(q + d1) + d2`` is being compared against ``q + (d1 + d2)`` at
#: every level, which costs about one float32 ulp per process.
#:
#: This is the SAME default and the SAME per-field exception wave 2's
#: gpuwm/core/thompson_aerosol_sed.py gate uses, transcribed unchanged.  An
#: auditor diffing the two will find no widening.
_END_TO_END_DEFAULT_BOUND = 2.0e-6

#: Measured, localised and attributed exceptions.  Each entry names the level
#: and the mechanism; none of them is an activation-bin-edge allowance.
#:
#: WP-12a TIGHTENED THIS 25x, from 2.5e-03 to 1.0e-04, after closing the
#: mechanism the old bound existed for.  Nothing here was widened; the entry
#: below is the ONLY entry, as it was before.
_END_TO_END_BOUNDS: dict[str, dict[str, float]] = {
    # EMPTY, and that is the WHOLE point of this entry: the one relative
    # bound this dict ever carried was retired by the 1.4.1 merge, not
    # narrowed again.
    #
    # It read {"aero-reduces-to-classic": {"nr_per_kg": 1.0e-5}} and existed
    # for 0-based level 5 of that column, measured at 5.700e-06 -- the one
    # level where module_mp_thompson.F:3500-3574 removes 49.75% of the rain
    # number without emptying it.  The sequence was 2.5e-03 -> 1.0e-04 ->
    # 1.0e-05 -> GONE.
    #
    # What closed it belongs to the mp=8 lane, not to this port.  Merging
    # integration/release-1.4.1 brought in 5e4af4e3 ("the rain MVD bound
    # belongs to TAU+1, not to sedimentation") and cb765336 ("the
    # rain-presence gate is a mass concentration, floor included"), both in
    # the byte-frozen thompson.cu that mp=28 shares for rain sedimentation.
    # RE-MEASURED on the merged tree, level 5: nr_per_kg 4.146e-07, which is
    # inside the flat 2.0e-06 gate by a factor of 4.8 -- so the bound has
    # nothing left to buy and is deleted rather than kept as a formality.
}

#: Levels (0-based) held out of the RELATIVE qr/nr comparison because the
#: step drives the level into catastrophic cancellation, plus the absolute
#: bound that replaces the relative one there.  This used to be a bare
#: exclusion list justified as "WRF's own qr survives at a value BOTH ArWen
#: ports drive to exactly zero"; after WP-12a's rain-density fix that
#: sentence is no longer true (mp=28 now produces 1.3384e-10 there against
#: WRF's 1.3426e-10, where it used to produce exactly 0.0), so the level is
#: bounded rather than skipped.
#:
#: WHY RELATIVE IS THE WRONG METRIC AT THIS ONE LEVEL.  aero-reduces-to-
#: classic level 6 enters with qr = 3.1695777e-07 kg/kg and evaporates
#: 99.958% of it in one 10 s step, so the surviving qr is the difference of
#: two nearly equal float32 numbers.  Relative error in the DIFFERENCE is the
#: relative error in the rate amplified by 1/(1 - 0.99958) = 2370, which puts
#: a 2e-06 relative gate below the float32 resolution of the entry value
#: itself.  The bound below is therefore stated in ULPS OF THE ENTRY VALUE,
#: which is the quantity the cancellation is measured against.
#: MEASURED: qr 14.9 ulp (4.2361e-13 kg/kg against ulp 2.842e-14),
#: nr_per_kg 4.05 ulp (0.1266 against ulp 0.03125).
_NEAR_CANCELLATION_LEVELS: dict[str, tuple[int, ...]] = {
    "aero-reduces-to-classic": (6,)}
_NEAR_CANCELLATION_ULPS = 32.0


#: EVERY DEPARTURE FROM THE FLAT GATE, NAMED, WITH WHAT IT BUYS.
#:
#: An auditor found that ``aero-reduces-to-classic`` was being counted clean
#: through THREE simultaneous allowances with no single place that said so.
#: This is that place.  Each entry is
#: ``(name, the object that implements it, the fixture(s) it applies to,
#:   the WRF justification)`` and
#: :func:`test_every_g3_allowance_is_named_and_buys_exactly_one_fixture`
#: proves the list is complete and that removing all of them costs exactly the
#: fixtures :data:`_G3_ALLOWANCE_ONLY_CLEAN` names -- so the honest,
#: unexceptioned count can never again be inferred from prose.
#:
#: NOTHING HERE WAS WIDENED.  ``_END_TO_END_BOUNDS`` went 2.5e-03 -> 1.0e-04
#: (25x stricter) when WP-12a closed the density mechanism it existed for;
#: ``_REFL_DB_BOUNDS`` went 1.0e-02 -> 1.0e-03 dB (10x stricter) for the same
#: reason and WP-13a DELETED it (the residual it covered is now 3.242e-05 dB,
#: inside the flat gate); ``_NEAR_CANCELLATION_LEVELS`` replaced a bare SKIP of
#: the level with a 32-ulp absolute bound, which is strictly more than the skip
#: asserted.  The list was three entries, then two.
#:
#: IT IS NOW ONE.  The 1.4.1 merge inherited the mp=8 lane's two
#: sedimentation reconciliations (5e4af4e3, cb765336) in the byte-frozen
#: thompson.cu this port shares for rain fallout, and re-measuring
#: aero-reduces-to-classic on the merged tree puts level 5's nr_per_kg at
#: 4.146e-07 against the 5.700e-06 the relative bound existed for.  The
#: bound bought nothing and was deleted; the near-cancellation bound at
#: level 6 is still load-bearing and is the only allowance left in the
#: port.
_G3_ALLOWANCES = (
    ("0-based level 6 held to 32 ulps of the entry value instead of a "
     "relative bound",
     "_NEAR_CANCELLATION_LEVELS",
     ("aero-reduces-to-classic",),
     "the level enters with qr = 3.1695777e-07 kg/kg and :3500-3574 "
     "evaporates 99.958% of it in one 10 s step, so the surviving value is "
     "the difference of two nearly equal float32 numbers and a relative gate "
     "there measures the rate's error amplified by 1/(1-0.99958) = 2370.  "
     "Bounded in ulps of the entry value, which is what the cancellation is "
     "measured against; MEASURED 14.9 ulp (qr) and 4.05 ulp (nr)."),
)


def _require_device():
    import cupy as cp
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")


def _tables_or_skip():
    """Skip only when CCN_ACTIVATE.BIN is genuinely absent.

    ``CCN_ACTIVATE.BIN`` ships as of 2026-08-01 (MP28_PORT_SPEC.md blocking
    unknown 1, reversed), so this guard does not fire on a clean checkout and
    the oracle gates run.  It stays as defence for a tree missing the file.
    The skip is narrow on purpose: it names the one missing asset instead of
    swallowing every load failure, which is how the wave-2 cold gates
    silently skipped seven tests.
    """
    from gpuwm.core.thompson_aerosol_contract import (
        MissingAerosolTableAsset, resolve_ccn_activation_path,
        resolve_aerosol_table_root)
    try:
        resolve_ccn_activation_path(None, resolve_aerosol_table_root(None))
    except MissingAerosolTableAsset as exc:
        pytest.skip(f"CCN_ACTIVATE.BIN unavailable: {exc}")


# ---------------------------------------------------------------------------
# G3.
# ---------------------------------------------------------------------------

#: THE EXACT COST OF THE ENTRY-STATE RECONSTRUCTION, PER FIXTURE.
#: ``fixture -> (number of levels whose float32 pressure had to move,
#: max |dp|/p over those levels)``.  Recorded rather than left implicit so a
#: reader can tell, fixture by fixture, which G3 residuals could possibly be
#: harness reconstruction and which cannot: a fixture with a zero here is
#: being driven from the oracle's OWN pressure, bit for bit, and every
#: residual it shows is physics.
#:
#: MEASURED ON THIS TREE: every entry is (0, 0.0).  It was not always: three
#: fixtures used to need up to 15 ulps at 10 of their 24 levels, because
#: run_column_aero.F90 built its Exner function from ``287.0/1004.0`` while
#: gpuwm/core/constants.py:32 uses ``RD/CP``, which is
#: share/module_model_constants.F:20's ``cp = 7.*r_d/2.`` and is exactly
#: ``2./7.`` in float32.  That was fixed IN THE HARNESS and the fixtures were
#: regenerated (run_column_aero.F90:33-46), which is why this table is now
#: uniformly zero.  It is asserted for EQUALITY, not as an upper bound: a
#: fixture that starts needing a perturbation fails here.
_ENTRY_STATE_PERTURBATION: dict[str, tuple[int, float]] = {
    "aero-ccn-activate": (0, 0.0),
    "aero-ccn-sweep": (0, 0.0),
    "aero-cloud-freeze-nc": (0, 0.0),
    "aero-cold-overlap": (0, 0.0),
    "aero-drop-evap": (0, 0.0),
    "aero-ice-demott-dep": (0, 0.0),
    "aero-ice-demott-idxin": (0, 0.0),
    "aero-ice-koop": (0, 0.0),
    "aero-init-profile": (0, 0.0),
    "aero-nc-accrete": (0, 0.0),
    "aero-nc-auto": (0, 0.0),
    "aero-nc-cap": (0, 0.0),
    "aero-nc-effrad": (0, 0.0),
    "aero-nc-sed": (0, 0.0),
    "aero-reduces-to-classic": (0, 0.0),
    "aero-scav-frozen": (0, 0.0),
    "aero-scav-rain": (0, 0.0),
    "aero-sfc-emit": (0, 0.0),
    "aero-warm-overlap": (0, 0.0),
    "wp08-freeze": (0, 0.0),
    "wp08-melt": (0, 0.0),
    "wp08-nusweep": (0, 0.0),
}


@requires_gpu
def test_entry_state_reconstruction_is_exact_in_temperature_and_dz():
    """The harness must not be the thing being measured.

    Publishes, per fixture, the exact-reconstruction receipt: temperature and
    layer depth bit-exact, and the float32 pressure perturbation the exact
    temperature cost.  A future change that starts approximating the entry
    state fails here rather than quietly inflating every number in the G3
    table.

    The perturbation is asserted against :data:`_ENTRY_STATE_PERTURBATION`
    EXACTLY -- level count and magnitude -- in both directions.  Every entry
    is currently ``(0, 0.0)``: G3 drives all twenty-two fixtures from the
    oracle's own float32 pressure, unmodified, so no G3 residual anywhere in
    this file can be attributed to the reconstruction.
    """
    import cupy as cp

    from gpuwm.core import constants as c

    assert set(_ENTRY_STATE_PERTURBATION) == set(_FIXTURES), (
        "the perturbation table and the fixture set disagree: "
        f"{sorted(set(_FIXTURES) ^ set(_ENTRY_STATE_PERTURBATION))}")

    report_lines = []
    for name in _FIXTURES:
        before, _, surface = _oracle_case(name)
        pressure, theta, geopotential, report = _reconstruct_entry_state(
            cp, before)
        want_levels, want_max = _ENTRY_STATE_PERTURBATION[name]
        assert report["pressure_perturbed_levels"] == want_levels, (
            f"{name}: the reconstruction now perturbs "
            f"{report['pressure_perturbed_levels']} pressure levels, "
            f"_ENTRY_STATE_PERTURBATION records {want_levels}")
        assert report["max_pressure_relative_perturbation"] == want_max, (
            f"{name}: max |dp|/p is "
            f"{report['max_pressure_relative_perturbation']:.3e}, "
            f"_ENTRY_STATE_PERTURBATION records {want_max:.3e}")
        # The strongest form of the same statement, level by level: the
        # array handed to the adapter IS the fixture's own pressure column.
        assert np.array_equal(pressure, _column(before, "p_pa")), (
            f"{name}: the reconstructed pressure is no longer bitwise the "
            "fixture's own")
        nz = len(before)
        thp = cp.asarray(theta.reshape(nz, 1, 1))
        p = cp.asarray(pressure.reshape(nz, 1, 1))
        pii = cp.power(p / f32(c.P0), f32(c.RCP))
        temperature = cp.asnumpy(
            (cp.zeros((nz, 1, 1), cp.float32) + thp) * pii).ravel()
        z8w = cp.asnumpy(
            (cp.zeros((nz + 1, 1, 1), cp.float32)
             + cp.asarray(geopotential.reshape(nz + 1, 1, 1)))
            / f32(c.G)).ravel()
        dz = (z8w[1:] - z8w[:-1]).astype(f32)
        assert np.array_equal(temperature, _column(before, "temp_k")), name
        assert np.array_equal(dz, _column(before, "dz_m")), name
        report_lines.append(
            f"{name:28s} dt={float(surface['dt_s']):5.1f} "
            f"p-perturbed levels={report['pressure_perturbed_levels']:2d} "
            f"max|dp|/p={report['max_pressure_relative_perturbation']:.2e}")
    print("\nENTRY-STATE RECONSTRUCTION\n" + "\n".join(report_lines))

    # The branch-critical family -- the exactly-saturated columns, where a
    # one-ulp qvs error flips module_mp_thompson.F:3401's condensation gate
    # -- must reconstruct with the pressure completely untouched.  This is now
    # subsumed by the table above (every fixture is untouched), but it is kept
    # because it is the check that would still hold if a future fixture
    # rebuild reintroduced a perturbation somewhere harmless: this family is
    # where a perturbation is never harmless.  Saturation is decided against
    # the FMA-FREE HOST transcription of WRF's RSLF, not against the device
    # fit, so a device regression cannot make it silently vacuous.
    saturated = []
    for name in _FIXTURES:
        before, _, _ = _oracle_case(name)
        qvs = _fma_free_saturation(
            _column(before, "p_pa"), _column(before, "temp_k"),
            _RSLF_COEFFICIENTS, ice=False)
        if not np.array_equal(qvs, _column(before, "qv")):
            continue
        saturated.append(name)
        _, _, _, report = _reconstruct_entry_state(cp, before)
        assert report["pressure_perturbed_levels"] == 0, (
            f"{name} is exactly saturated and must reconstruct with the "
            "fixture pressure untouched")
    print("EXACTLY SATURATED FIXTURES (pressure untouched): "
          + ", ".join(saturated))
    assert len(saturated) >= 6, saturated


#: The FMA-free float32 transcription of WRF's RSLF/RSIF Horner chains.
#: ``tools/thompson_wrf461_oracle/build_aero.sh`` compiles the oracle with
#: plain ``gfortran -O2`` on baseline x86-64, which has NO fma instruction, so
#: every REAL(4) multiply and add in module_mp_thompson.F:5378-5446 is
#: separately rounded.  WP-08 verified this transcription independently: the
#: ``aero-nc-sed`` entry column's ``qv`` IS ``rslf(p,T)`` from the Fortran,
#: bit for bit, and this function reproduces it at all 24 levels.
_RSLF_COEFFICIENTS = (
    0.611583699e3, 0.444606896e2, 0.143177157e1, 0.264224321e-1,
    0.299291081e-3, 0.203154182e-5, 0.702620698e-8, 0.379534310e-11,
    -0.321582393e-13)
_RSIF_COEFFICIENTS = (
    0.609868993e3, 0.499320233e2, 0.184672631e1, 0.402737184e-1,
    0.565392987e-3, 0.521693933e-5, 0.307839583e-7, 0.105785160e-9,
    0.161444444e-12)


def _fma_free_saturation(pressure, temperature, coefficients, *, ice: bool):
    coefficients = [f32(value) for value in coefficients]
    pressure = np.asarray(pressure, np.float32)
    x = np.maximum(f32(-80.0),
                   (np.asarray(temperature, np.float32) - f32(273.16))
                   ).astype(np.float32)
    es = np.full(x.shape, coefficients[8], np.float32)
    for coefficient in reversed(coefficients[:8]):
        es = (coefficient + (x * es).astype(np.float32)).astype(np.float32)
    es = np.minimum(es, (pressure * f32(0.15)).astype(np.float32))
    denominator = (pressure - es).astype(np.float32)
    if ice:
        denominator = np.maximum(f32(1.0e-4), denominator).astype(np.float32)
    return (f32(0.622) * es / denominator).astype(np.float32)


#: The in-tree spelling of the FMA-CONTRACTED Horner chain that
#: ``gpuwm/core/kernels/thompson_aerosol_common.cuh`` currently carries, and
#: the contraction-pinned replacement.  Used ONLY to build an in-memory source
#: string; no file is written.  See
#: ``test_g3_receipt_with_the_shared_saturation_fit_repaired_in_memory``.
def _contracted_chain(variable: str) -> str:
    return (f"    float {variable} = c0 + x * (c1 + x * (c2 + x * (c3 + x * "
            "(c4\n        + x * (c5 + x * (c6 + x * (c7 + x * c8)))))));")


def _pinned_chain(variable: str) -> str:
    inner = "c8"
    for index in range(7, -1, -1):
        inner = f"thompson_aa_add(c{index}, thompson_aa_mul(x, {inner}))"
    return f"    float {variable} = {inner};"


@requires_gpu
def test_the_shared_saturation_fit_must_be_bitwise_before_g3_can_mean_anything(
):
    """G3's PRECONDITION, stated as its own gate so its failure has one owner.

    ``module_mp_thompson.F:3401`` opens the whole condensation/CCN-activation
    block on ``ssatw(k) .gt. eps`` with ``eps = 1.E-15`` (:185), and
    :3197-3198 sets ``ssatw`` from ``qv/qvs`` with a snap to zero only inside
    that same 1e-15 band.  A saturation fit that is ONE float32 ulp low
    therefore does not perturb a rate -- it FLIPS A BRANCH, and every
    cloud-free level of a saturated column condenses water WRF does not.

    ``gpuwm/core/kernels/thompson_aerosol_common.cuh:716-717`` (RSLF) and
    :735-736 (RSIF) evaluate their Horner chains as plain ``a + x*b``, which
    nvrtc contracts into FMAs because ``--fmad=true`` is the default.  The
    Fortran oracle has no FMA instruction at all.  MEASURED on the RTX 5090:
    the device RSLF is one ulp low at 23 of 24 levels of the ``aero-nc-sed``
    entry column, ``ssatw`` becomes +2.4e-7 instead of exactly 0, and the
    end-to-end mp=28 call then produces ``nc`` wrong by a factor of 1e36 at
    the cloud-free levels while consuming up to 72 percent of the column CCN.

    THIS IS NOT THIS PACKAGE'S FILE.  The fix belongs to the header's owner
    and is one line each: evaluate both chains through ``thompson_aa_add`` /
    ``thompson_aa_mul``, exactly as the header already does for
    ``activ_ncloud``, ``iceKoop`` and ``Eff_aero`` and for the same reason.
    ``tests/test_thompson_aerosol_sed_gpu.py::
    test_device_saturation_fit_reproduces_the_fortran_horner_bitwise``
    is the same finding on one fixture; this is the whole fixture set and
    both fits.
    """
    import cupy as cp

    _require_device()
    from gpuwm.core.thompson_aerosol_launch import probe_saturation

    broken = []
    for name in _FIXTURES:
        before, _, _ = _oracle_case(name)
        pressure = _column(before, "p_pa")
        temperature = _column(before, "temp_k")
        nz = pressure.size
        device_l, device_i = probe_saturation(
            cp.asarray(pressure.reshape(nz, 1, 1)),
            cp.asarray(temperature.reshape(nz, 1, 1)))
        got_l = cp.asnumpy(device_l).ravel().astype(f32)
        got_i = cp.asnumpy(device_i).ravel().astype(f32)
        want_l = _fma_free_saturation(
            pressure, temperature, _RSLF_COEFFICIENTS, ice=False)
        want_i = _fma_free_saturation(
            pressure, temperature, _RSIF_COEFFICIENTS, ice=True)
        bad_l = int((got_l != want_l).sum())
        bad_i = int((got_i != want_i).sum())
        if bad_l or bad_i:
            broken.append(f"{name}: RSLF {bad_l}/{nz}, RSIF {bad_i}/{nz}")
    assert not broken, (
        "gpuwm/core/kernels/thompson_aerosol_common.cuh:716 and :735 are "
        "FMA-contracted; the Fortran oracle has no FMA instruction, so the "
        "mp=28 condensation gate at module_mp_thompson.F:3401 fires where "
        "WRF's does not.  Levels differing:\n  " + "\n  ".join(broken))


def _ulps(value):
    """float32 ulp of ``value``, for the near-cancellation bound."""
    magnitude = f32(abs(np.float64(value)))
    if magnitude == 0.0:
        return float(np.nextafter(f32(0.0), _INF, dtype=np.float32))
    return float(np.nextafter(magnitude, _INF, dtype=np.float32) - magnitude)


def _run_g3(cp, scenario, *, widened=False):
    """One complete adapter call, reduced to per-field differences.

    ``widened=False`` returns EXACTLY the 15 prognostic column residuals plus
    ``rainnc_mm``: that is the dictionary
    ``tests/test_physics_registry.py::
    test_mp28_published_residuals_still_equal_a_live_adapter_measurement``
    consumes and the shape the registry's ``column_oracle_evidence`` publishes,
    so it is a contract and is kept stable on purpose.

    ``widened=True`` adds the six surface diagnostics the fixtures carry that
    nothing compared before WP-12a, and the reflectivity column, and is what
    the G3 gate itself asserts.  Both modes run the SAME call -- reflectivity
    is issued on every G3 call because the oracle produced these fixtures with
    ``do_radar_ref=1``, and ``test_a_due_reflectivity_call_perturbs_no_
    prognostic_field`` proves that costs the compared state nothing.
    """
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    state, cfg, dt, before, after, surface, report = _build_case(cp, scenario)
    _apply_thompson_aerosol(state, cfg, dt, refl_10cm_due=True)
    cp.cuda.Stream.null.synchronize()
    got = _adapter_outputs(cp, state, before)
    for field, value in got.items():
        assert np.all(np.isfinite(value)), f"{scenario}: {field} is not finite"

    excluded = list(_NEAR_CANCELLATION_LEVELS.get(scenario, ()))
    measured = {}
    for field in _END_TO_END_FIELDS:
        want = _oracle_expectation(after, field)
        keep = np.ones(want.size, bool)
        if field in ("qr", "nr_per_kg"):
            keep[excluded] = False
        measured[field] = _max_relative(got[field][keep], want[keep])

    def surface_relative(key):
        slot = _END_TO_END_SURFACE_FIELDS[key]
        mine = float(cp.asnumpy(state._scratch[slot]).ravel()[0])
        theirs = float(surface[key])
        if not (mine or theirs):
            return 0.0
        return abs(mine - theirs) / max(abs(theirs), 1.0e-30)

    measured["rainnc_mm"] = surface_relative("rainnc_mm")
    if not widened:
        return measured, report

    for key in _END_TO_END_SURFACE_FIELDS:
        if key != "rainnc_mm":
            measured[key] = surface_relative(key)
    assert state.physics.refl_10cm is not None, (
        f"{scenario}: a due mp=28 call stashed no REFL_10CM frame")
    refl = cp.asnumpy(state.physics.refl_10cm).ravel().astype(np.float64)
    want_refl = _column(after, "refl_dbz").astype(np.float64)
    assert refl.shape == want_refl.shape, scenario
    assert np.all(np.isfinite(refl)), scenario
    # mp_gt_driver:1461 clamps to -35 dBZ; it is a floor for gpuwm too, and
    # every level WRF reports AT the floor must be bitwise -35.0 here rather
    # than a value a dB tolerance would forgive.
    assert float(refl.min()) >= -35.0, scenario
    at_floor = want_refl == -35.0
    assert np.all(refl[at_floor] == -35.0), (
        f"{scenario}: {int((refl[at_floor] != -35.0).sum())} of "
        f"{int(at_floor.sum())} oracle-floor levels are not bitwise -35.0")
    measured[_REFL_FIELD] = float(np.abs(refl - want_refl).max())
    return measured, report


def _g3_bound(scenario, field):
    """The bound this (scenario, field) is held to, and its units."""
    if field == _REFL_FIELD:
        return _REFL_DB_BOUNDS.get(scenario, _REFL_DB_GATE)
    return _END_TO_END_BOUNDS.get(scenario, {}).get(
        field, _END_TO_END_DEFAULT_BOUND)


def _g3_failures(scenario, measured):
    return [f"{field}={value:.3e}>{_g3_bound(scenario, field):.1e}"
            for field, value in measured.items()
            if not value <= _g3_bound(scenario, field)]


def _g3_columns(table):
    ordered = list(_END_TO_END_FIELDS) + list(_END_TO_END_SURFACE_FIELDS)
    ordered.append(_REFL_FIELD)
    present = set().union(*(set(row) for row in table.values()))
    return [field for field in ordered if field in present]


def _print_g3_table(title, table):
    columns = _g3_columns(table)
    header = (f"{'fixture':28s} "
              + " ".join(f"{field:>11s}" for field in columns))
    rows = [f"{name:28s} " + " ".join(
        f"{table[name][field]:11.2e}" for field in columns)
        for name in _FIXTURES]
    print(f"\n{title}\n{header}\n" + "\n".join(rows)
          + f"\n(all columns are max relative difference except "
            f"{_REFL_FIELD}, which is max |dBZ - WRF| in dB)")


@requires_gpu
def test_g3_end_to_end_against_all_nineteen_oracle_fixtures():
    """G3, as the tree stands: every fixture through the real adapter.

    This is the gate the port's maturity claim rests on.  ``aero-warm-overlap``
    (117) and ``aero-cold-overlap`` (118) are the stated acceptance criterion
    for the cross-network ``ncten``/``nwfaten`` reconciliation holding across
    package boundaries; ``aero-reduces-to-classic`` (119) is the bridge to the
    model-validated mp=8 port.

    All the fixtures run in ONE test on purpose: the failure message is
    then the whole table, which is what a reader needs in order to tell a
    composition defect (many fixtures, one field) from a kernel defect (one
    fixture, one field).  The name says "nineteen" because that is the size
    of MP28_PORT_SPEC.md's specified set (ids 101-119) and the number the
    registry and the public documents quote; the glob resolves to TWENTY-TWO
    and all twenty-two are gated -- the three extras are WP-08's ids 120-122
    (``wp08-nusweep`` / ``wp08-melt`` / ``wp08-freeze``) from the same
    ``build_aero.sh`` run.  The name is deliberately NOT changed here:
    docs/public/PHYSICS.md:76 and the column-evidence page cite this test by
    name, and a rename would break the citation instead of the count.

    WHAT THE GATE COVERS (WP-12a widened it).  Twenty-three quantities per
    fixture, not sixteen: the 15 prognostic column fields, all SEVEN surface
    diagnostics WRF's driver writes at mp_gt_driver:1298-1308 (only
    ``rainnc_mm`` of which was compared before), and the ``refl_dbz`` column
    unmodified WRF ``calc_refl10cm`` wrote into every fixture -- 51.98 dBZ at
    its peak and, before WP-12a, compared by nothing in this file.  The
    ``-35.0`` dBZ floor is additionally required to be BITWISE exact, which no
    tolerance can express.

    THE BOUND IS NOT NEGOTIABLE HERE.  ``_END_TO_END_DEFAULT_BOUND`` is 2e-6
    and ``_END_TO_END_BOUNDS`` still carries exactly ONE entry -- now 25x
    STRICTER than the 2.5e-3 wave 2 wrote (``tests/
    test_thompson_aerosol_sed_gpu.py``, which WP-12c has since tightened to
    match), because WP-12a closed the mechanism that bound existed for.
    Nothing in this file widens a wave-2 tolerance, the reflectivity gate is
    the stricter of the port's two existing dB gates, and no bound here
    exists to accommodate an ``activ_ncloud`` bin-edge disagreement -- see
    ``test_activation_bin_edge_policy_is_documented_not_absorbed``.  Every
    departure from the flat 2e-6 gate is enumerated in
    :data:`_G3_ALLOWANCES` with its WRF justification.

    STILL RED, HONESTLY, AND THE COUNT IS STATED TWICE.
    MEASURED ON THIS TREE (RTX 5090, cupy 14.1.1):

      * UNEXCEPTIONED -- a flat 2.0e-06 relative / 2.0e-04 dB gate on all
        twenty-three quantities, no bounds dict, no excluded levels, no
        per-fixture carve-out: **17 of 22 clean** (16 of the 19 ``aero-*``
        fixtures, plus ``wp08-melt``).  That table is the truth and it is
        printed by
        :func:`test_the_unexceptioned_g3_table_is_printed_and_its_count_pinned`
        below.
      * AS GATED -- with the two allowances in :data:`_G3_ALLOWANCES`
        applied: **18 of 22** (17 of 19).  The allowances buy exactly ONE
        fixture, ``aero-reduces-to-classic``, and both are needed for that one
        fixture.  A third allowance, ``_REFL_DB_BOUNDS``, was RETIRED by
        WP-13a: the reflectivity residual it covered fell from 5.283e-04 dB to
        3.242e-05 dB, inside the flat gate.

    SO THIS TEST FAILS, and it fails on FOUR fixtures (it was six before
    WP-13a closed ``aero-drop-evap`` and ``aero-ice-demott-idxin`` outright):
    aero-cloud-freeze-nc, aero-cold-overlap, wp08-freeze and wp08-nusweep.
    Five miss the unexceptioned gate; aero-reduces-to-classic is the fifth and
    is the one the allowances cover.

    Both counts are pinned as data (:data:`_G3_UNEXCEPTIONED_CLEAN`,
    :data:`_G3_GATED_CLEAN`) so neither can drift from the measurement, and
    :data:`_G3_RESIDUALS` carries every surviving number with its attribution.

    AND EVERY RESIDUAL IS ALSO PUBLISHED IN FLOAT32 ULPS (WP-14).  The
    relative number above is not the whole truth about a residual and at two
    cells it is actively misleading: ``aero-cold-overlap`` qc reads 1.000e+00
    for a ONE-ULP disagreement, because WRF's surviving value there IS one
    ulp.  :func:`test_every_g3_residual_is_published_in_ulps_as_well_as_
    relative` prints the same 22 x 23 table denominated in ulps of
    ``max(|entry|, |WRF after|)`` and pins every cell of it.  The two units
    disagree about which cell is worst, and that disagreement is the point:
    four of the ten above-gate cells are 0.2 to 1.8 ulps (rounding), three are
    14 to 60 ulps (two named mechanisms in the frozen mp=8 kernel), and one --
    ``aero-cold-overlap`` effc_m -- is 1.1e+07 ulps, which is a branch that
    flipped and nothing to do with rounding at all.  Nothing there can make a
    fixture pass; :func:`test_the_ulp_companion_metric_cannot_relax_the_
    relative_gate` asserts that arithmetically.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()

    table = {}
    failures = {}
    for scenario in _FIXTURES:
        measured, _ = _run_g3(cp, scenario, widened=True)
        table[scenario] = measured
        bad = _g3_failures(scenario, measured)
        if bad:
            failures[scenario] = bad
    _print_g3_table("G3 (tree as it stands)", table)

    # The widening is part of the assertion, not part of the prose: if a
    # future edit narrows _run_g3 back to the sixteen fields this gate
    # started with, that is a silently weakened gate and it fails here.
    compared = set(table[_FIXTURES[0]])
    assert compared == (set(_END_TO_END_FIELDS)
                        | set(_END_TO_END_SURFACE_FIELDS)
                        | {_REFL_FIELD}), sorted(compared)
    assert len(compared) == 23, len(compared)      # 15 column + 7 surface + 1

    assert not failures, "\n".join(
        f"{name}: {', '.join(bad)}" for name, bad in failures.items())


#: THE HONEST COUNT, PART 1: the fixtures that clear a FLAT gate.
#:
#: Fixtures whose worst residual, over all twenty-three compared quantities,
#: is inside 2.0e-06 relative (2.0e-04 dB for reflectivity) with NO bounds
#: dict, NO excluded levels and NO per-fixture carve-out of any kind.  This is
#: the number the port may publish as "clean" without a footnote.
#:
#: MEASURED ON THIS TREE: 17 of 22 -- sixteen of the nineteen ``aero-*``
#: fixtures MP28_PORT_SPEC.md specifies, plus ``wp08-melt``.
#:
#: WP-13a ADDED TWO: ``aero-drop-evap`` and ``aero-ice-demott-idxin``.  Both
#: were held out solely by the sedimentation-density defect described on
#: :data:`_G3_RESIDUALS`, and both went to 0.000e+00 on every surface
#: accumulator when it was fixed.
_G3_UNEXCEPTIONED_CLEAN = (
    "aero-ccn-activate",
    "aero-ccn-sweep",
    "aero-drop-evap",
    "aero-ice-demott-dep",
    "aero-ice-demott-idxin",
    "aero-ice-koop",
    "aero-init-profile",
    "aero-nc-accrete",
    "aero-nc-auto",
    "aero-nc-cap",
    "aero-nc-effrad",
    "aero-nc-sed",
    "aero-scav-frozen",
    "aero-scav-rain",
    "aero-sfc-emit",
    "aero-warm-overlap",
    "wp08-melt",
)

#: THE HONEST COUNT, PART 2: the fixtures that clear the gate AS GATED, i.e.
#: with the one allowance in :data:`_G3_ALLOWANCES` applied.  18 of 22.
#: The difference between this and the tuple above is exactly
#: :data:`_G3_ALLOWANCE_ONLY_CLEAN`, and that identity is asserted, so the two
#: counts can never be conflated again.
_G3_GATED_CLEAN = tuple(sorted(
    set(_G3_UNEXCEPTIONED_CLEAN) | {"aero-reduces-to-classic"}))

#: The fixtures that are clean ONLY because an allowance exists.  Exactly one,
#: and it now rests on ONE allowance rather than two: dropping the surviving
#: near-cancellation bound puts it back over the gate, which
#: :func:`test_every_g3_allowance_is_named_and_buys_exactly_one_fixture`
#: proves by removing them one at a time.  (A third, ``_REFL_DB_BOUNDS``, was
#: retired by WP-13a and the same test now asserts the flat dB gate holds
#: without it.)
_G3_ALLOWANCE_ONLY_CLEAN = ("aero-reduces-to-classic",)


#: THE RESIDUAL RATCHET.  Every (fixture, quantity) that is ABOVE the gate as
#: gated, with the value measured on this tree.  Asserted in both directions
#: by :func:`test_the_g3_residual_ratchet_holds_in_both_directions`: no
#: recorded residual may grow, the SET of quantities above the gate may not
#: change, and a fixture may neither leave nor silently join the clean set.
#:
#: WHAT MOVED IN WP-13, AND WHY.  Three production changes, all in mp=28-owned
#: kernels, and one recorded growth.
#:
#:   * WP-13a, THE SEDIMENTATION DENSITY.  WRF forms the working rain mass and
#:     number sedimentation consumes TWICE: at :3237-3238 from the :3193 TAU+1
#:     density, for every level with L_qr, and again at :3568/:3570 from the
#:     :3490 POST-condensation density -- but only inside the :3501-3502 gate.
#:     thompson_aerosol_sat.cu wrote the post-condensation density into its
#:     `reference_density` output unconditionally, so ArWen gave EVERY level
#:     the :3568 answer, including the levels WRF never rewrote, and
#:     microphysics_aerosol.py hands that same buffer to
#:     `launch_rain_sedimentation`.  The write is now the :3237 density by
#:     default and the :3568 one only after all three of WRF's gates pass.
#:     WHAT IT CLOSED, measured: ``aero-drop-evap`` (rainnc/rainncv 5.165e-04
#:     -> 0.000e+00, qr 3.533e-05 -> 7.35e-08, nr 2.258e-05 -> 3.92e-07) and
#:     ``aero-ice-demott-idxin`` (sr and rainnc/rainncv 1.279e-04 -> 3.5e-07,
#:     qr 2.894e-05 -> 1.35e-06) left this table entirely;
#:     ``aero-cloud-freeze-nc`` lost five of its six rows (qr 2.800e-05 ->
#:     8.97e-08, nr 1.797e-05 -> 2.59e-07, rainnc/rainncv/sr 1.162e-05 ->
#:     0.000e+00); ``aero-reduces-to-classic``'s reflectivity went 5.283e-04
#:     -> 3.242e-05 dB, retiring ``_REFL_DB_BOUNDS``.
#:   * WP-13b, CONTRACTION PINNING OF THE SOURCE-NETWORK APPLY.  WRF's
#:     :3973-4023 terminal apply is `q1d(k) = q1d(k) + qten(k)*DT` and the
#:     gfortran -O2 baseline-x86-64 oracle has no FMA, so qten*DT is rounded to
#:     REAL(4) first; nvrtc contracted the same expression in
#:     thompson_aerosol_cold.cu and thompson_aerosol_warm.cu and never rounded
#:     it.  Both now use thompson_aa_add/sub/mul, which
#:     thompson_aerosol_sat.cu's rain-evaporation apply already did.
#:     VERIFIED AT THE STAGE, not just end to end: the post-source-network qc
#:     at wp08-freeze level 0 is now 2.7275411412119865e-05 where it was the
#:     fused 2.7275422326056287e-05, and at aero-cold-overlap level 4 it is
#:     1.5486410120502114e-05 where it was the fused 1.5486406482523307e-05 --
#:     both of which are the rounded-product values WRF itself produces.
#:     WHAT IT CLOSED: ``aero-nc-cap`` qc 1.17e-07 -> 0.000e+00 and nc
#:     9.43e-08 -> 0.000e+00 (bitwise), ``aero-ice-demott-idxin`` rainnc /
#:     rainncv 3.49e-07 -> 0.000e+00 and sr 3.35e-07 -> 0.000e+00 (bitwise),
#:     its qr 1.35e-06 -> 6.42e-07, ``aero-warm-overlap`` nr 4.44e-07 ->
#:     4.20e-07, ``aero-cold-overlap`` nr 1.340e-04 -> 1.261e-04.
#:   * WHAT IT COST, RECORDED RATHER THAN ABSORBED.  ``aero-cold-overlap``
#:     qr at level 6 GREW, 3.667e-05 -> 4.443e-05.  BISECTED to a single line,
#:     the cold network's qr apply: reverting that one pin restores 3.667e-05
#:     and simultaneously loses all four aero-ice-demott-idxin improvements
#:     above, two of which are exact.  In ulps of the entry value -- the scale
#:     :data:`_RESIDUAL_ATTRIBUTION` records this cell at, because 99.5% of the
#:     level's rain is consumed in the step -- the move is 1.477 -> 1.789 ulp.
#:     The pin is kept because it makes the instruction sequence WRF's at every
#:     cell rather than at the cells that flatter this table; the growth is
#:     recorded here and the published matrix in
#:     tests/test_thompson_aerosol_gpu.py is updated with it.
#:
#: WHAT MOVED IN THE WAVE BEFORE.  Four things.
#:   * ``aero-ice-koop`` LEFT this table entirely.  Its qi 1.612e-03 /
#:     ni 1.764e-03 -- described by three auditors as the port's largest
#:     genuine physics gap -- now measure 1.534e-07 and 3.396e-07, inside the
#:     flat gate.  WP-06 closed it; this file only re-measures it.
#:   * ``aero-cold-overlap`` GAINED a qc / nc_per_kg / effc_m row at 0-based
#:     level 4, and its nc/effc worst level moved from 6 to 4.  MEASURED: WRF
#:     ends the step there with qc = 1.4551915228366852e-11 kg/kg -- exactly
#:     2**-36, and exactly 1.0000 float32 ulp of the 2.325216e-04 kg/kg the
#:     level entered with -- and nc = 1.833336 per kg, while gpuwm ends at
#:     exactly zero.  The relative metric therefore reports 1.0 on an absolute
#:     difference of one ulp (nc: 0.229 ulp).  Recorded as a MISS, not
#:     allowanced.
#:     WHAT IS AND IS NOT CLAIMED ABOUT ITS ORIGIN.  The fixture deck was
#:     regenerated at the same time the harness's Exner constant was corrected
#:     (run_column_aero.F90:33-46), and this entry column carries a one-ulp
#:     temperature offset at exactly 0-based levels 4, 5, 13, 14, 15 and 19
#:     (239.99998 K, and 240.00002 K at 19, against 240.0 K elsewhere) -- the
#:     theta = T/pii round-trip artifact that constant governs, and levels 4
#:     and 5 are two of the six.  The PREVIOUS fixture's after-state cannot be
#:     re-measured from this tree, so no claim is made that it was zero there;
#:     what is claimed is only that the previous wave's residual table carried
#:     no qc row for this fixture and this one does, and that the difference
#:     today is one ulp.
#:   * ``aero-cloud-freeze-nc``'s effc_m (5.018e-06) and
#:     ``aero-ice-demott-idxin``'s qc (6.031e-06) fell inside the gate and are
#:     gone from the table.
#:
#: Every residual below was located to its worst level and measured there by
#: :func:`test_every_surviving_residual_is_located_and_its_regime_is_stated`.
#: Every one of them now sits where the field is either CREATED FROM ZERO
#: inside the step or driven to near-total consumption -- the regimes where a
#: fixed relative gate measures the amplified rounding of a difference rather
#: than the rate that produced it.  That is stated, not used: no bound below
#: is relaxed for it, and after WP-06 closed aero-ice-koop there is no longer
#: any surviving residual in the "rate disagreement" class at all.
#:
#:   aero-cloud-freeze-nc  qc 4.926e-06 at level 4, where 98.2% of the entry
#:       cloud water is frozen away and the surviving 1.477e-06 kg/kg differs
#:       by exactly 1.0000 ulp of the 8.247212e-05 entry value.  This is the
#:       ONE cell in the table whose mechanism the port could name and could
#:       not reach: it is the SECOND float32 rounding of qc inside one step.
#:       WRF rounds ONCE, at :3975, from a qcten that carries the source
#:       network and the condensation together; ArWen applies the source
#:       network to state.qc and then applies the condensation to the already-
#:       rounded value.  MEASURED, and pinned by
#:       :func:`test_the_qc_residual_is_exactly_one_extra_rounding`: qc at this
#:       level is 8.205620542867109e-05 entering the condensation stage,
#:       1.4771576388739049e-06 leaving it, and NOTHING AFTER THAT TOUCHES IT
#:       -- cloud sedimentation, the final phase cleanup and the terminal
#:       apply all leave it bit-identical -- so the whole 4.926e-06 is the one
#:       extra rounding and nothing else.
#:       WHY IT WAS NOT CLOSED.  Carrying a qcten accumulator from the source
#:       networks through the condensation needs (a) a scratch slot, and
#:       gpuwm/core/preflight.py enumerates the mp=28 aerosol working set with
#:       test_scratch_slots_match_the_preflight_registry_exactly asserting
#:       equality in both directions, and (b) resequencing the adapter so the
#:       ncten balance limiter (:2996-3019), the :3646 cloud column mask and
#:       gpuwm/core/kernels/thompson_aerosol_sed.cu's cloud sedimentation all
#:       read qc_entry + qcten*dt instead of state.qc.  Neither preflight.py
#:       nor thompson_aerosol_sed.cu was WP-13's to edit.  Recorded,
#:       attributed, measured, not absorbed.
#:       Its qr / nr / rainnc / rainncv / sr rows are GONE: WP-13a closed them.
#:   aero-cold-overlap  qc / nc_per_kg 1.000e+00 and effc_m 8.102e-01 at level
#:       4 -- the sub-ulp remainder described above.  THE CHAIN, CITED: the
#:       one extra rounding leaves ArWen's qc at exactly 0.0 where WRF's is
#:       1.4551915228366852e-11; :4006-4008 is
#:       ``if (qc1d(k) .le. R1) then qc1d = 0.0 ; nc1d = 0.0``, and
#:       R1 = 1.E-12 (:183), so WRF takes the ELSE branch and :4010-4020's
#:       nu_c / lamc / D0c size bound hands back nc = 1.833336 per kg while
#:       ArWen takes the THEN branch and zeroes nc.  :5625-5647 then does the
#:       same thing again for the radius.  So all three rows of this fixture
#:       at this level are ONE branch, taken on a one-ulp difference, and the
#:       ULP companion table reads them 1.000, 0.229 and 1.114e+07
#:       respectively -- which is exactly the shape a branch flip has and a
#:       rounding does not.  Also at this fixture: nr 1.261e-04 and qr
#:       4.443e-05 at level 6, where the rain number falls from 255.41 to
#:       0.0739 per kg (99.97% consumed) and the difference is 0.611 ulp of
#:       the entry value (qr: 1.789 ulp).  The cold half of the shared
#:       ncten/nwfaten accumulator contract; the warm half (aero-warm-overlap)
#:       is clean at 4.204e-07.
#:   aero-reduces-to-classic  qr / nr 3.155e-03 at level 6 UNEXCEPTIONED -- the
#:       one fixture the two surviving allowances cover, so it is gated-clean
#:       and appears here only through
#:       tests/test_thompson_aerosol_gpu.py's published matrix.  MECHANISM,
#:       traced but NOT fixable inside mp=28: WRF rewrites rr/nr at :3568-3570
#:       WITHOUT re-applying the :3240-3250 mvd_r clamp, so :3616-3627 derives
#:       the fall speed from an unclamped pair (mvd_r 12.901 um, vtrk
#:       8.016012e-02 m/s from WRF's own dump).  gpuwm/core/kernels/
#:       thompson.cu:438-459 re-applies the clamp, gets 37.5 um and vtrk
#:       2.314977e-01 m/s -- 2.888x too fast -- and that kernel is the frozen,
#:       model-validated mp=8 one.  See the integration note in the WP-13
#:       report.
#:   wp08-freeze  nr 2.724e-06 at level 0, created from exactly zero and
#:       reaching 23.808 per kg.  1.36x the gate.  MECHANISM, likewise traced
#:       and likewise in the frozen kernel: thompson.cu:438 gates rain
#:       sedimentation on `qr > 1.0e-12` -- a MIXING RATIO -- where WRF's
#:       :3616 tests `rr(k) > R1`, a MASS CONCENTRATION.  At this fixture's
#:       level 1 qr = 8.5265e-13 kg/kg but rr = 1.1748e-12 kg/m3, so WRF gives
#:       the level a real number fall speed and ArWen treats it as rain-free.
#:   wp08-nusweep  qr 4.642e-06 at level 12, created from exactly zero and
#:       reaching 2.242e-11 kg/kg.  2.3x the gate.  |got - want| is 1.04e-16
#:       kg/kg, 2.502e-08 of the column's own peak rain.  No mechanism is
#:       claimed for this one beyond the regime: it is the smallest absolute
#:       difference in the table by nine decades.
#: THE TWO wp08 CELLS, ADJUDICATED AT THE 1.4.1 MERGE.  Both were carried
#: for waves as "one explained, one not".  Re-measuring on the merged tree
#: swapped them, and the swap is recorded here rather than in prose because
#: it changes which cell a future wave should chase.
#:
#: wp08-nusweep qr @ level 12 -- WAS the deck's one unexplained cell.  It is
#: EXPLAINED now, and by conditioning rather than by a mechanism: the cell is
#: ill-conditioned to a degree that puts the 2.0e-06 gate below what FP32 can
#: deliver there.  MEASURED by perturbing the level's entry state by exactly
#: one float32 ULP and re-running the whole adapter call:
#:
#:     entry qc  +1 ulp  ->  exit qr moves 128 ulp
#:     entry qc  -1 ulp  ->  exit qr moves  32 ulp
#:     entry nc  +1 ulp  ->  exit qr moves 256 ulp
#:     entry nc  -1 ulp  ->  exit qr moves 256 ulp
#:
#: The disagreement is 60 ulp -- SMALLER than a single-ulp input change
#: produces -- so it is consistent with a sub-ulp difference in an
#: intermediate, which is a difference FP32 cannot represent and therefore
#: cannot be coded away.  The gate at that level is about 26 ulp, i.e. below
#: the cell's own condition number.  The regime is visible in the fixture
#: too: level 12 is the ONLY level of this column where nc RISES across the
#: step (1.417475e+08 -> 1.428941e+08 per kg), the number-weighted cloud
#: sedimentation feeding it from level 13 while autoconversion drains it, and
#: the exit qr there is 2.242e-11 kg/kg against 6.02e-10 one level below.
#:
#: wp08-freeze nr @ level 0 -- EXPLAINED, un-explained, and now MEASURED.
#: 29 of its 34 ulp are thompson.cu:438 gating rain presence on a mixing
#: ratio (qr > 1.0e-12f) where module_mp_thompson.F:3616 gates on a mass
#: concentration (rr .gt. R1).  At 0-based level 1 of this column the two
#: disagree, so WRF computes a real number-weighted fall speed of 0.3165 m/s
#: there and ArWen inherits 1.6887 from the level above.  Forcing the gate
#: open on the frozen kernel moves level 0's nr from 34 ulp away from WRF to
#: 5; a mass change of the same size that does NOT flip the gate leaves the
#: output bit-identical.  test_the_wp08_freeze_residual_is_the_presence_
#: gates_units_measured runs all three arms.
#:
#: THIS WAS PUBLISHED AS FALSIFIED FOR PART OF 2026-08-01, WRONGLY.  The
#: falsification read qr1d + qrten*DT at the END of the step (8.521e-13,
#: below R1) and took it for the value :3236 tested; the rain evaporation
#: block at :3501 subtracts from qrten in between.  Instrumented WRF
#: (tools/thompson_wrf461_oracle/instrument_sedimentation_entry.py) records
#: L_qr = .true. at that level, and L_qr is written nowhere between
#: :3239/:3254 and the fall-speed loop, so :3236 took its TRUE branch and
#: :3568 left rr at 1.174815e-12, above R1.  cb765336 did not move the
#: residual because it reconciled the sedimentation DENSITY, not the gate's
#: UNITS.  The cell is well conditioned either way -- a one-ulp entry
#: perturbation moves the exit by 3-7 ulp (qc) or 0-4 ulp (nc) -- which is
#: why 34 ulp was findable and, unlike wp08-nusweep, was found.
_G3_RESIDUALS: dict[str, dict[str, float]] = {
    "aero-cloud-freeze-nc": {"qc": 4.9256e-06},
    "aero-cold-overlap": {
        "qc": 1.0000e+00, "nc_per_kg": 1.0000e+00, "effc_m": 8.1018e-01,
        "nr_per_kg": 1.2613e-04, "qr": 4.4426e-05},
    "wp08-freeze": {"nr_per_kg": 2.7239e-06},
    "wp08-nusweep": {"qr": 4.6424e-06},
}


_G3_TABLE: dict[str, dict[str, float]] = {}
_G3_TABLE_UNEXCEPTIONED: dict[str, dict[str, float]] = {}


def _g3_table(cp):
    """The widened G3 table AS GATED, measured once and cached.

    "As gated" means exactly what :func:`_run_g3` does for the real gate: the
    :data:`_NEAR_CANCELLATION_LEVELS` entries ARE held out of the relative
    qr/nr comparison.  The per-fixture bounds are NOT applied here -- they are
    applied by :func:`_g3_bound` at comparison time -- so this table is the
    right input for the residual ratchet and for the gated verdict.
    """
    if not _G3_TABLE:
        for scenario in _FIXTURES:
            _G3_TABLE[scenario], _ = _run_g3(cp, scenario, widened=True)
    return _G3_TABLE


def _run_g3_without_level_exclusions(cp, scenario):
    """``_run_g3`` with :data:`_NEAR_CANCELLATION_LEVELS` neutralised."""
    saved = dict(_NEAR_CANCELLATION_LEVELS)
    _NEAR_CANCELLATION_LEVELS.clear()
    try:
        return _run_g3(cp, scenario, widened=True)
    finally:
        _NEAR_CANCELLATION_LEVELS.update(saved)


def _g3_table_unexceptioned(cp):
    """The widened G3 table with NOTHING held out anywhere.

    This is the one that earns the word "unexceptioned": every level of every
    field of every fixture is in the comparison, including the levels
    :data:`_NEAR_CANCELLATION_LEVELS` removes for the real gate.  Keeping it
    SEPARATE from :func:`_g3_table` is the whole point -- the first version of
    this helper reused the gated table and quietly inherited its level
    exclusion, which is precisely the bookkeeping error this package exists to
    remove.
    """
    if not _G3_TABLE_UNEXCEPTIONED:
        for scenario in _FIXTURES:
            _G3_TABLE_UNEXCEPTIONED[scenario], _ = (
                _run_g3_without_level_exclusions(cp, scenario))
    return _G3_TABLE_UNEXCEPTIONED


def _flat_failures(scenario, measured):
    """Failures under the FLAT gate: no bounds dict, no carve-outs."""
    return {field: value for field, value in measured.items()
            if not value <= (_REFL_DB_GATE if field == _REFL_FIELD
                             else _END_TO_END_DEFAULT_BOUND)}


@requires_gpu
def test_the_unexceptioned_g3_table_is_printed_and_its_count_pinned():
    """THE UNEXCEPTIONED TABLE.  No bounds dict, no exclusions, no carve-outs.

    Everything else this file says about G3 is a claim ABOUT this table, so
    this is where the table is produced and the count is pinned.  The metric
    is a flat 2.0e-06 maximum relative difference on the 22 scalar/column
    quantities and a flat 2.0e-04 dB on the reflectivity column, applied to
    every level of every fixture with nothing held out.

    The count is asserted for SET EQUALITY in both directions: a fixture that
    stops clearing the flat gate fails here, and one that starts clearing it
    fails here too, because a port whose evidence understates it is a port
    whose evidence nobody re-read.

    MEASURED: 17 of 22 clean.  The five that miss and every number they miss
    by are in :data:`_G3_RESIDUALS` (four of them) and, for the one the
    allowances cover, in tests/test_thompson_aerosol_gpu.py's published matrix.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()

    table = _g3_table_unexceptioned(cp)
    _print_g3_table("G3 UNEXCEPTIONED (flat 2.0e-06 / 2.0e-04 dB, "
                    "no level held out)", table)

    clean, missing = [], {}
    for scenario in _FIXTURES:
        bad = _flat_failures(scenario, table[scenario])
        if bad:
            missing[scenario] = bad
        else:
            clean.append(scenario)
    print("UNEXCEPTIONED CLEAN: "
          f"{len(clean)}/{len(_FIXTURES)} -- {sorted(clean)}")
    for scenario, bad in sorted(missing.items()):
        print(f"  MISS {scenario:24s} " + ", ".join(
            f"{field}={value:.3e}"
            for field, value in sorted(bad.items(), key=lambda kv: -kv[1])))

    assert set(clean) == set(_G3_UNEXCEPTIONED_CLEAN), (
        "the unexceptioned clean set moved.  Newly clean: "
        f"{sorted(set(clean) - set(_G3_UNEXCEPTIONED_CLEAN))}; no longer "
        f"clean: {sorted(set(_G3_UNEXCEPTIONED_CLEAN) - set(clean))}.  "
        "Update _G3_UNEXCEPTIONED_CLEAN, _G3_RESIDUALS, the G3 docstring and "
        "tests/test_thompson_aerosol_gpu.py in ONE change.")
    assert len(clean) == 17 and len(_FIXTURES) == 22, (
        len(clean), len(_FIXTURES))
    # ...and the aero-only subtotal the public documents quote.
    aero = [name for name in clean if name.startswith("aero-")]
    assert len(aero) == 16, aero
    assert len([n for n in _FIXTURES if n.startswith("aero-")]) == 19, (
        "the spec'd fixture set (ids 101-119) is no longer nineteen columns")


# ---------------------------------------------------------------------------
# THE COMPANION METRIC (WP-14): the same table, denominated in float32 ULPS.
# ---------------------------------------------------------------------------
#
# WHAT THIS IS FOR, AND WHAT IT IS EMPHATICALLY NOT.
#
# It is NOT a second, looser gate.  The relative gate above is unchanged --
# still a flat 2.0e-06 / 2.0e-04 dB, still two allowances, still five
# fixtures missing it -- and nothing below can make a fixture pass.  Every
# assertion here is an ADDITIONAL upper bound on a number the relative gate
# does not bound at all.
#
# WHY IT IS NEEDED.  A relative metric divides by WRF's own surviving value,
# and at some levels that value is one float32 ulp above zero.  The clearest
# case in the deck is ``aero-cold-overlap`` qc at 0-based level 4: WRF ends
# the step at 1.4551915228366852e-11 kg/kg, which is 2**-36 and is EXACTLY
# 1.0000 float32 ulp of the 2.3252160e-04 kg/kg the level entered with, and
# gpuwm ends at exactly 0.0.  The relative metric reports 1.000e+00 -- the
# largest number it can report -- for an absolute disagreement of one ulp, at
# a level where no float32 implementation could report anything else.  Read
# alone, that row is indistinguishable from a scheme that lost all its cloud
# water.  In ulps it reads 1.000, and the same column's ``effc_m`` reads
# 1.114e+07, which is the honest statement that ONE of those two rows is a
# rounding artefact and the other is a branch that flipped.
#
# THE DENOMINATOR, STATED EXACTLY.  For each level the scale is
# ``max(|value the level entered the step with|, |value WRF left it at|)``
# and the unit is the float32 ulp of that scale.  Two consequences, both
# deliberate:
#
#   * wherever the level is CONSUMED (WRF's answer is smaller than the entry
#     value, which is every row this port has above the gate except three),
#     the scale IS the entry value and the number is literally "ulps of the
#     entry value" as the residual attribution table records it.  That
#     identity is asserted per row below, not assumed.
#   * wherever the field is CREATED FROM ZERO the entry value is 0 and its
#     ulp is the smallest subnormal, which would make every such row read
#     1e+30.  The scale is then WRF's own answer, which is the only
#     non-degenerate resolution the difference can be measured against.
#     Those rows are marked "after" and there are exactly three of them.
#
# WHAT THE MEASUREMENT SETTLES.  Wave 5 proposed that the whole residual set
# is ArWen rounding mass fields once per stage where module_mp_thompson.F
# accumulates qcten/qrten/... and applies them once at :3971-4021.  In these
# units that hypothesis is testable, and the answer is PARTLY:
#
#   * it holds for the two 1.000e+00 relative rows and their consequences.
#     Both qc rows are 1.000 ulp EXACTLY, and
#     ``test_the_qc_residual_is_exactly_one_extra_rounding`` locates both to
#     the second of exactly two stages that write qc.  ``nc_per_kg`` at the
#     same level is 0.229 ulp -- SUB-ulp, i.e. below the resolution the
#     comparison is carried at.
#   * it is consistent with ``aero-cold-overlap`` qr / nr at level 6 (1.789
#     and 0.611 ulp: one to two roundings).
#   * it does NOT explain ``wp08-freeze`` nr (34 ulps, at the column maximum,
#     created from zero) or ``aero-reduces-to-classic`` qr / nr (14.9 and
#     27.5 ulps).  Those three are attributed to two NAMED mechanisms in the
#     frozen mp=8 kernel and are pinned by
#     ``test_the_two_residuals_that_live_in_the_frozen_kernel_are_measured_
#     here``: gpuwm/core/kernels/thompson.cu:438 gates rain sedimentation on
#     ``qr > 1.0e-12`` (a MIXING RATIO) where module_mp_thompson.F:3616 tests
#     ``rr(k) > R1`` (a MASS CONCENTRATION), and thompson.cu:443-455
#     re-applies the :3240-3250 mvd_r clamp inside the :3616-3627 fall-speed
#     block, where WRF does not.
#   * ``wp08-nusweep`` qr is 60 ulps and no mechanism is claimed for it; the
#     absolute difference is 1.04e-16 kg/kg.
#
# So the hypothesis is a real mechanism that accounts for four of the ten
# rows and is not the whole story.  Recorded, not asserted away.


def _ulp_column(values):
    """Per-element float32 ulp of ``values``, in float64.

    ``ulp(0)`` is the smallest positive float32 subnormal, which is the
    correct answer to "what is the resolution here" and the wrong denominator
    for a physical residual -- see :func:`_ulp_residual`, which never hands
    this function a zero scale unless the difference is zero too.
    """
    magnitude = np.abs(np.asarray(values, np.float64)).astype(f32)
    step = (np.nextafter(magnitude, _INF, dtype=np.float32)
            - magnitude).astype(np.float64)
    tiny = float(np.nextafter(f32(0.0), _INF, dtype=np.float32))
    return np.where(magnitude == f32(0.0), tiny, step)


def _ulp_residual(got, want, entry):
    """``|got - want|`` per level, in float32 ulps of the level's own scale.

    The scale is ``max(|entry|, |want|)``; see the block comment above for
    why, and :data:`_G3_ULP_PINS` for which side dominates on every row the
    port has above the relative gate.
    """
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    entry = np.asarray(entry, np.float64)
    difference = np.abs(got - want)
    scale = np.maximum(np.abs(entry), np.abs(want))
    return np.where(difference == 0.0, 0.0,
                    difference / _ulp_column(scale))


def _entry_expectation(before, field):
    """The value the level ENTERED the step with, in the compared units.

    Same transform :func:`_oracle_expectation` applies to the after-state, so
    an effective radius is compared -- and denominated -- in the clamped
    microns a forecast consumes rather than in the fixture's raw metres.
    """
    return _oracle_expectation(before, field)


def _run_g3_in_ulps(cp, scenario):
    """One complete adapter call, reduced to per-level ULP residuals.

    Returns ``field -> (ulps per level, primary metric per level)`` where the
    primary metric is the relative difference for every quantity except
    ``refl_dbz_db``, which is absolute dB exactly as the gate measures it.
    The second array exists so the ULP table can report the ULP count AT THE
    LEVEL THE RELATIVE GATE FAILS ON, which is the number a reader comparing
    the two tables needs, as well as the worst ULP count anywhere.

    Nothing is held out: no bounds dict, no ``_NEAR_CANCELLATION_LEVELS``.
    """
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    state, cfg, dt, before, after, surface, _report = _build_case(cp, scenario)
    _apply_thompson_aerosol(state, cfg, dt, refl_10cm_due=True)
    cp.cuda.Stream.null.synchronize()
    got = _adapter_outputs(cp, state, before)

    rows: dict[str, tuple] = {}
    for field in _END_TO_END_FIELDS:
        want = _oracle_expectation(after, field)
        entry = _entry_expectation(before, field)
        mine = got[field]
        both_zero = (mine == 0.0) & (want == 0.0)
        relative = np.where(both_zero, 0.0,
                            np.abs(mine - want)
                            / np.maximum(np.abs(want), 1.0e-30))
        rows[field] = (_ulp_residual(mine, want, entry), relative)

    for key, slot in _END_TO_END_SURFACE_FIELDS.items():
        mine = np.asarray(
            [float(cp.asnumpy(state._scratch[slot]).ravel()[0])], np.float64)
        theirs = np.asarray([float(surface[key])], np.float64)
        relative = (np.zeros(1) if not (mine[0] or theirs[0])
                    else np.abs(mine - theirs)
                    / np.maximum(np.abs(theirs), 1.0e-30))
        # A surface accumulator enters the step at zero by construction
        # (mp_gt_driver:1298-1308 writes the step's own total), so the scale
        # is WRF's own value.
        rows[key] = (_ulp_residual(mine, theirs, np.zeros(1)), relative)

    refl = cp.asnumpy(state.physics.refl_10cm).ravel().astype(np.float64)
    want_refl = _column(after, "refl_dbz").astype(np.float64)
    entry_refl = _column(before, "refl_dbz").astype(np.float64)
    rows[_REFL_FIELD] = (_ulp_residual(refl, want_refl, entry_refl),
                         np.abs(refl - want_refl))
    return rows


_G3_ULP_ROWS: dict[str, dict[str, tuple]] = {}


def _g3_ulp_rows(cp):
    if not _G3_ULP_ROWS:
        for scenario in _FIXTURES:
            _G3_ULP_ROWS[scenario] = _run_g3_in_ulps(cp, scenario)
    return _G3_ULP_ROWS


#: EVERY RESIDUAL ABOVE THE FLAT RELATIVE GATE, PINNED IN ULPS.  Ten rows,
#: one per (fixture, quantity) cell that
#: :func:`test_the_unexceptioned_g3_table_is_printed_and_its_count_pinned`
#: reports as a MISS -- so this tuple's key set is asserted to EQUAL that
#: miss set and a residual cannot appear without a ULP pin or vice versa.
#:
#: Each row is
#: ``(fixture, field, worst-relative level, ULPs there, which side of
#:   max(|entry|, |after|) is the scale there, worst ULPs anywhere in the
#:   column, the level that sits at, which side is the scale there)``.
#:
#: MEASURED ON THIS TREE (RTX 5090, cupy 14.1.1, nvrtc defaults).  READ THE
#: TWO ULP COLUMNS TOGETHER: ``aero-cold-overlap`` qr is 1.789 ulp at the
#: level the relative gate fails on and 18 ulps at level 1, so the relative
#: metric and the ULP metric do not agree about where this column's worst
#: disagreement is -- which is itself the reason for publishing both.
_G3_ULP_PINS = (
    ("aero-cloud-freeze-nc", "qc", 4, 1.000, "entry", 1.000, 4, "entry"),
    ("aero-cold-overlap", "qc", 4, 1.000, "entry", 1.000, 4, "entry"),
    ("aero-cold-overlap", "qr", 6, 1.789, "entry", 18.00, 1, "after"),
    ("aero-cold-overlap", "nr_per_kg", 6, 0.6109, "entry", 16.00, 1, "after"),
    ("aero-cold-overlap", "nc_per_kg", 4, 0.2292, "entry", 0.2292, 4,
     "entry"),
    ("aero-cold-overlap", "effc_m", 4, 1.11439e+07, "after", 1.11439e+07, 4,
     "after"),
    # RE-MEASURED after the 1.4.1 merge inherited the mp=8 lane's two
    # sedimentation reconciliations.  Both rows TIGHTENED, and by a lot:
    # qr level 6 went 14.90 -> 0.5850 entry-ulps and its worst level moved
    # off 6 entirely; nr_per_kg level 6 went 4.052 -> 0.1593, and its worst
    # went 27.50 ulps at level 5 -- the cell the retired relative bound
    # existed for -- to 3.000 ulps at level 2.  The rows stay because both
    # cells are still above the FLAT gate once level 6 is put back in.
    ("aero-reduces-to-classic", "qr", 6, 0.5850, "entry", 1.000, 0, "after"),
    ("aero-reduces-to-classic", "nr_per_kg", 6, 0.1593, "entry", 3.000, 2,
     "entry"),
    ("wp08-freeze", "nr_per_kg", 0, 34.00, "after", 34.00, 0, "after"),
    ("wp08-nusweep", "qr", 12, 60.00, "after", 60.00, 12, "after"),
)

#: THE CEILING ON EVERYTHING ELSE.  Every one of the 496 (fixture, quantity)
#: cells that is INSIDE the flat relative gate is also inside this many
#: float32 ulps of the level's own scale, at EVERY level -- not just at the
#: level the relative metric happens to peak on.
#:
#: MEASURED: the worst is exactly 20.0 ulps, ``aero-cloud-freeze-nc``
#: ``effc_m`` at 0-based level 4, and that is not an unrelated cell: it is the
#: derived consequence of the 1.000-ulp qc row at the same level of the same
#: fixture (:5646-5647 takes a cube root of a ratio in qc, and the level's
#: cloud water is 98.2% consumed).  Second worst is 9.0 ulps.
#:
#: This bound exists because the relative gate does NOT constrain these cells
#: in ULP terms at all: a level whose values are large can drift many ulps
#: and stay far inside 2e-06.  A regression that doubles the rounding error
#: of a currently-clean quantity fails here.
_G3_CLEAN_ULP_CEILING = 20.0

#: THE WORST ULP ANYWHERE IN EACH FIXTURE, over all 23 quantities and all
#: levels, above-gate cells included.  A per-fixture ratchet: it bounds the
#: cells the relative gate leaves unbounded AND the cells
#: :data:`_G3_ULP_PINS` bounds, in one number a reader can scan.
#:
#: FOUR FIXTURES ARE BIT-EXACT AGAINST WRF ON EVERY COMPARED QUANTITY AT
#: EVERY LEVEL -- 0.0 ulps, not "0.0 to four figures": aero-ccn-activate,
#: aero-ccn-sweep, aero-init-profile and aero-sfc-emit.  That is stated here
#: because it is the strongest single fact the deck contains and the relative
#: table renders it as an unremarkable column of ``0.00e+00``.  It is asserted
#: for EQUALITY (``measured == 0.0``), not as an upper bound.
_G3_WORST_ULP_BY_FIXTURE = {
    "aero-ccn-activate": 0.0,
    "aero-ccn-sweep": 0.0,
    "aero-cloud-freeze-nc": 20.0,
    "aero-cold-overlap": 1.11439e+07,
    "aero-drop-evap": 4.0,
    "aero-ice-demott-dep": 5.0,
    "aero-ice-demott-idxin": 3.0,
    "aero-ice-koop": 4.0,
    "aero-init-profile": 0.0,
    "aero-nc-accrete": 5.0,
    "aero-nc-auto": 3.0,
    "aero-nc-cap": 4.0,
    "aero-nc-effrad": 3.0,
    "aero-nc-sed": 3.0,
    "aero-reduces-to-classic": 4.0,
    "aero-scav-frozen": 3.0,
    "aero-scav-rain": 4.0,
    "aero-sfc-emit": 0.0,
    "aero-warm-overlap": 5.0,
    "wp08-freeze": 34.0,
    "wp08-melt": 2.0,
    "wp08-nusweep": 60.0,
}

#: Round-trip slack on the four-significant-figure literals above.  It is NOT
#: a physics tolerance: the whole ULP table was measured twice end to end on
#: this device and every cell was bit-identical across the repeats.
_G3_ULP_ROUND_TRIP = 1.01


@requires_gpu
def test_every_g3_residual_is_published_in_ulps_as_well_as_relative():
    """THE COMPANION TABLE, printed, and every cell of it pinned.

    Three assertions, none of which can make a fixture pass the relative gate:

    1.  The set of cells :data:`_G3_ULP_PINS` covers is EXACTLY the set of
        cells above the flat relative gate.  A new residual therefore cannot
        appear without a ULP pin, and a closed one cannot leave a stale pin
        behind.
    2.  Every pinned cell is inside its recorded ULP count, at the level the
        relative gate fails on AND at its own worst level, and the recorded
        "which side is the scale" flag is re-derived from the fixture rather
        than trusted.
    3.  Every OTHER cell is inside :data:`_G3_CLEAN_ULP_CEILING` ulps at every
        level, and every fixture is inside its
        :data:`_G3_WORST_ULP_BY_FIXTURE` entry.

    WHAT THIS BUYS THE READER.  ``aero-cold-overlap`` qc reads 1.000e+00
    relative and 1.000 ulp; ``aero-cold-overlap`` effc_m reads 8.102e-01
    relative and 1.114e+07 ulp.  Those two numbers are the same order of
    magnitude in the relative column and eight decades apart here, and the
    ULP column is the one that is right: the first is a one-ulp remainder,
    the second is module_mp_thompson.F:5638's ``rc <= R1`` CYCLE taking a
    different branch.  A reader with only the relative table cannot tell
    those apart, and neither could five waves of this port.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()

    rows = _g3_ulp_rows(cp)
    columns = list(_END_TO_END_FIELDS) + list(_END_TO_END_SURFACE_FIELDS)
    columns.append(_REFL_FIELD)
    header = (f"{'fixture':28s} "
              + " ".join(f"{field:>11s}" for field in columns))
    body = "\n".join(
        f"{name:28s} " + " ".join(
            f"{rows[name][field][0].max():11.4g}" for field in columns)
        for name in _FIXTURES)
    print("\nG3 IN FLOAT32 ULPS OF max(|entry|, |WRF after|), worst level of "
          f"each column\n{header}\n{body}\n"
          "(companion to the relative table; the relative gate is unchanged)")

    above = {(fixture, field)
             for fixture in _FIXTURES
             for field, value in _g3_table_unexceptioned(cp)[fixture].items()
             if not value <= (_REFL_DB_GATE if field == _REFL_FIELD
                              else _END_TO_END_DEFAULT_BOUND)}
    pinned = {(row[0], row[1]) for row in _G3_ULP_PINS}
    assert pinned == above, (
        "the ULP pins and the cells above the relative gate have diverged.  "
        f"Above the gate with no ULP pin: {sorted(above - pinned)}; pinned "
        f"but no longer above the gate: {sorted(pinned - above)}")
    assert len(_G3_ULP_PINS) == 10, len(_G3_ULP_PINS)

    for (fixture, field, level, at_level, scale_at_level,
         worst, worst_level, scale_at_worst) in _G3_ULP_PINS:
        ulps, primary = rows[fixture][field]
        assert int(np.argmax(primary)) == level, (
            f"{fixture}.{field}: the relative gate's worst level moved from "
            f"{level} to {int(np.argmax(primary))}; the ULP pin is anchored "
            "to it and is now describing a different cell")
        assert ulps[level] <= at_level * _G3_ULP_ROUND_TRIP, (
            f"{fixture}.{field} at level {level}: {ulps[level]:.6g} ulps, "
            f"pinned at {at_level:.6g}")
        assert int(np.argmax(ulps)) == worst_level, (
            f"{fixture}.{field}: the worst ULP level moved from {worst_level} "
            f"to {int(np.argmax(ulps))}")
        assert ulps.max() <= worst * _G3_ULP_ROUND_TRIP, (
            f"{fixture}.{field}: worst {ulps.max():.6g} ulps at level "
            f"{int(np.argmax(ulps))}, pinned at {worst:.6g}")
        # ...and the recorded denominator is re-derived, not trusted.
        if field in _END_TO_END_FIELDS:
            before, after, _surface = _oracle_case(fixture)
            entry = _entry_expectation(before, field)
            final = _oracle_expectation(after, field)
            for index, side in ((level, scale_at_level),
                                (worst_level, scale_at_worst)):
                dominant = ("entry" if abs(entry[index]) >= abs(final[index])
                            else "after")
                assert dominant == side, (
                    f"{fixture}.{field} at level {index}: the ULP scale is "
                    f"the {dominant} value ({entry[index]:.8g} entry vs "
                    f"{final[index]:.8g} after), not the recorded {side}")

    worst_clean = []
    for fixture in _FIXTURES:
        for field, (ulps, _primary) in rows[fixture].items():
            if (fixture, field) in pinned:
                continue
            worst_clean.append((float(ulps.max()), fixture, field,
                                int(np.argmax(ulps))))
    worst_clean.sort(reverse=True)
    assert worst_clean[0][0] <= _G3_CLEAN_ULP_CEILING, (
        "a quantity INSIDE the flat relative gate drifted past the ULP "
        f"ceiling: {worst_clean[0][1]}.{worst_clean[0][2]} is "
        f"{worst_clean[0][0]:.6g} ulps at level {worst_clean[0][3]}, ceiling "
        f"{_G3_CLEAN_ULP_CEILING}")
    print(f"worst ULP inside the relative gate: {worst_clean[0][0]:.6g} "
          f"({worst_clean[0][1]}.{worst_clean[0][2]} at level "
          f"{worst_clean[0][3]}); ceiling {_G3_CLEAN_ULP_CEILING}")

    assert set(_G3_WORST_ULP_BY_FIXTURE) == set(_FIXTURES), sorted(
        set(_G3_WORST_ULP_BY_FIXTURE) ^ set(_FIXTURES))
    for fixture in _FIXTURES:
        measured = max(float(ulps.max())
                       for ulps, _primary in rows[fixture].values())
        pin = _G3_WORST_ULP_BY_FIXTURE[fixture]
        assert measured <= pin * _G3_ULP_ROUND_TRIP, (
            f"{fixture}: worst ULP over all 23 quantities grew from {pin:.6g} "
            f"to {measured:.6g}")
        if pin == 0.0:
            assert measured == 0.0, (
                f"{fixture} is published as BIT-EXACT against WRF on every "
                f"compared quantity and now measures {measured:.6g} ulps")

    exact = sorted(name for name, pin in _G3_WORST_ULP_BY_FIXTURE.items()
                   if pin == 0.0)
    assert exact == ["aero-ccn-activate", "aero-ccn-sweep",
                     "aero-init-profile", "aero-sfc-emit"], exact
    print(f"BIT-EXACT ON EVERY COMPARED QUANTITY: {len(exact)}/22 -- {exact}")


@requires_gpu
def test_the_ulp_companion_metric_cannot_relax_the_relative_gate():
    """HARD RULE 7, asserted against this file's own new machinery.

    The ULP table is additional published information.  It must be impossible
    for it to buy a fixture, so this proves three things arithmetically:

    * the gate function the G3 test calls (:func:`_g3_failures`) does not
      consult any ULP object -- asserted over the module source, because a
      future edit that wires one in is exactly the failure being prevented;
    * the flat relative gate and its two allowances are the values this file
      has always carried;
    * every fixture the ULP pins describe is STILL a miss under the relative
      gate, i.e. the pins document residuals rather than excusing them.
    """
    import cupy as cp

    source = Path(__file__).read_text(encoding="utf-8")
    start = source.index("def _g3_bound(")
    end = source.index("def _g3_columns(")
    gate_source = source[start:end]
    for forbidden in ("_ULP", "_ulp"):
        assert forbidden not in gate_source, (
            "the G3 bound/failure functions now reference a ULP object; the "
            "companion metric must never be able to widen the gate")

    assert _END_TO_END_DEFAULT_BOUND == 2.0e-6
    assert _REFL_DB_GATE == 2.0e-4
    assert _NEAR_CANCELLATION_ULPS == 32.0
    # BOTH relative carve-outs are now empty.  _REFL_DB_BOUNDS was retired
    # by WP-13a; _END_TO_END_BOUNDS was retired by the 1.4.1 merge, whose
    # inherited mp=8 sedimentation fixes took the 5.700e-06 it covered to
    # 4.146e-07.  Asserted as EMPTY so a re-added entry fails here rather
    # than passing silently through _g3_bound.
    assert _END_TO_END_BOUNDS == {}
    assert _REFL_DB_BOUNDS == {}

    _require_device()
    _tables_or_skip()
    table = _g3_table(cp)
    for fixture, field, *_rest in _G3_ULP_PINS:
        bound = _g3_bound(fixture, field)
        value = table[fixture][field]
        if fixture in _G3_GATED_CLEAN:
            # aero-reduces-to-classic: the allowances cover it, and the ULP
            # pins record what they cover rather than replacing them.
            assert value <= bound, (fixture, field, value, bound)
        else:
            assert not value <= bound, (
                f"{fixture}.{field} carries a ULP pin but is no longer above "
                "the relative gate; remove the pin in the same change that "
                "removes its _G3_RESIDUALS row")


@requires_gpu
def test_every_g3_allowance_is_named_and_buys_exactly_one_fixture():
    """The surviving allowance, measured, plus both retired ones held down.

    An auditor's finding: ``aero-reduces-to-classic`` is counted clean through
    three simultaneous allowances and no single place said so.  This is that
    place, and it is arithmetic rather than prose.  Two of the three are now
    retired -- the reflectivity carve-out by WP-13a, the relative bound by
    the 1.4.1 merge -- and each retirement inverted its assertion rather than
    removing it, so a carve-out cannot quietly come back:

    * the gated clean set minus the unexceptioned clean set is exactly
      :data:`_G3_ALLOWANCE_ONLY_CLEAN`;
    * removing ANY ONE of the three allowances puts that fixture back over the
      gate, so none of them is redundant padding;
    * no allowance touches any other fixture -- for every fixture except the
      one named, the gated verdict and the flat verdict are identical.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    gated_table = _g3_table(cp)
    flat_table = _g3_table_unexceptioned(cp)

    gated = {name for name in _FIXTURES
             if not _g3_failures(name, gated_table[name])}
    flat = {name for name in _FIXTURES
            if not _flat_failures(name, flat_table[name])}
    assert gated == set(_G3_GATED_CLEAN), sorted(gated)
    assert flat == set(_G3_UNEXCEPTIONED_CLEAN), sorted(flat)
    assert gated - flat == set(_G3_ALLOWANCE_ONLY_CLEAN), sorted(gated - flat)
    assert not flat - gated, sorted(flat - gated)

    # The registry names every object that implements an allowance, and every
    # such object is in the registry.  A fourth allowance added without a row
    # here fails this.
    named = {entry[1] for entry in _G3_ALLOWANCES}
    assert named == {"_NEAR_CANCELLATION_LEVELS"}, named
    for _label, _object, fixtures, justification in _G3_ALLOWANCES:
        assert set(fixtures) <= set(_FIXTURES), fixtures
        assert set(fixtures) == set(_G3_ALLOWANCE_ONLY_CLEAN), fixtures
        assert "module_mp_thompson.F" in justification or ":" in justification
    for name, allowance in (("_NEAR_CANCELLATION_LEVELS",
                             _NEAR_CANCELLATION_LEVELS),):
        assert set(allowance) == set(_G3_ALLOWANCE_ONLY_CLEAN), (
            f"{name} now applies to {sorted(allowance)}, but the port's "
            f"allowances are declared to buy only "
            f"{sorted(_G3_ALLOWANCE_ONLY_CLEAN)}; a second allowanced "
            "fixture must be added to _G3_ALLOWANCE_ONLY_CLEAN, to "
            "_G3_ALLOWANCES with its WRF justification, and to the G3 "
            "docstring's two counts in the same change")
    # THE RETIRED ONE STAYS RETIRED.  WP-13a's sedimentation-density fix put
    # aero-reduces-to-classic's reflectivity residual inside the flat dB gate,
    # so the carve-out buys nothing; re-adding an entry here without also
    # adding a row to _G3_ALLOWANCES fails this line rather than passing
    # silently through _g3_bound.
    assert _REFL_DB_BOUNDS == {}, (
        "_REFL_DB_BOUNDS was retired by WP-13a because the residual it "
        "covered is now inside the flat gate; a new entry needs a new "
        "_G3_ALLOWANCES row with its WRF justification")

    # NONE OF THE THREE IS REDUNDANT.  Drop one, re-measure, require the
    # fixture to fail.  Nothing is written; the measurement is re-run against
    # a modified bound view.
    scenario = _G3_ALLOWANCE_ONLY_CLEAN[0]
    measured, _ = _run_g3(cp, scenario, widened=True)
    assert not _g3_failures(scenario, measured), measured

    # (a) THE RELATIVE-BOUND ALLOWANCE IS NO LONGER NEEDED EITHER.  This
    #     used to assert the opposite -- that dropping _END_TO_END_BOUNDS put
    #     the fixture back over the gate on nr_per_kg.  The 1.4.1 merge
    #     closed the mechanism (5e4af4e3 and cb765336 in the frozen mp=8
    #     kernel this port shares for rain sedimentation), so the assertion
    #     is INVERTED rather than deleted, exactly as (b) below was: with
    #     _NEAR_CANCELLATION_LEVELS still applied, every field of this
    #     fixture must clear the FLAT gate on its own.
    without_bounds = [
        f"{field}={value:.3e}" for field, value in measured.items()
        if not value <= (_REFL_DB_GATE if field == _REFL_FIELD
                         else _END_TO_END_DEFAULT_BOUND)]
    assert not without_bounds, (
        "a relative carve-out is needed again for "
        f"{scenario}: {without_bounds}; re-adding one needs a new "
        "_G3_ALLOWANCES row with its WRF justification")
    # (b) THE REFLECTIVITY CARVE-OUT IS NO LONGER NEEDED AT ALL.  This used to
    #     assert the opposite -- that dropping _REFL_DB_BOUNDS put the fixture
    #     back over the gate.  WP-13a closed the mechanism, so the assertion is
    #     inverted rather than deleted: the flat 2.0e-04 dB gate must hold on
    #     its own, and if it ever stops holding this fails instead of a
    #     carve-out quietly reappearing.
    assert measured[_REFL_FIELD] <= _REFL_DB_GATE, (
        f"{scenario} reflectivity {measured[_REFL_FIELD]:.3e} dB no longer "
        f"clears the flat {_REFL_DB_GATE:.1e} dB gate WP-13a put it inside")
    # (c) without the near-cancellation level exclusion.
    raw = flat_table[scenario]
    excluded = _NEAR_CANCELLATION_LEVELS[scenario]
    assert excluded, scenario
    for field in ("qr", "nr_per_kg"):
        assert not raw[field] <= _g3_bound(scenario, field), (
            f"_NEAR_CANCELLATION_LEVELS is not buying anything for {field}")


@requires_gpu
def test_the_g3_residual_ratchet_holds_in_both_directions():
    """Nothing above the gate may grow, appear or silently disappear.

    Strictly stronger than the pass/fail gate: it pins the MAGNITUDE of every
    residual, so a change that moves aero-nc-sed from 2.2e-07 to 1.9e-06 still
    passes G3 and still fails here once it crosses, and a change that doubles
    a recorded residual fails here immediately.  ``* 1.01`` is a printing
    tolerance on a four-significant-figure literal, not a physics tolerance.

    WHY A 1% RATCHET IS LEGITIMATE RATHER THAN FLAKY.  The whole table was
    measured three times end to end on this device and all 506 cells (22
    fixtures x 23 quantities) were BIT-IDENTICAL across the repeats, so there
    is no run-to-run jitter for a tolerance to absorb; the 1% is purely the
    round-trip of the literals above.  If that ever stops being true the right
    response is to record the spread, not to widen this.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    table = _g3_table(cp)

    assert set(_G3_RESIDUALS) | set(_G3_GATED_CLEAN) == set(_FIXTURES), (
        "every fixture must be either gated-clean or carry a residual row: "
        f"{sorted(set(_FIXTURES) - set(_G3_RESIDUALS)
                  - set(_G3_GATED_CLEAN))}")
    assert not set(_G3_RESIDUALS) & set(_G3_GATED_CLEAN)

    for scenario, recorded in _G3_RESIDUALS.items():
        measured = table[scenario]
        offenders = {field: value for field, value in measured.items()
                     if not value <= _g3_bound(scenario, field)}
        assert set(offenders) == set(recorded), (
            f"{scenario}: the set of quantities above the gate changed; "
            f"measured { {k: f'{v:.4e}' for k, v in offenders.items()} }, "
            f"recorded {sorted(recorded)}")
        for field, bound in recorded.items():
            assert offenders[field] <= bound * 1.01, (
                f"{scenario}.{field} grew from {bound:.4e} to "
                f"{offenders[field]:.4e}")

    # The ratchet must cover the widened gate, not the sixteen fields G3
    # started from, or a surface/reflectivity regression could hide here.
    assert set(_g3_columns(table)) == (
        set(_END_TO_END_FIELDS) | set(_END_TO_END_SURFACE_FIELDS)
        | {_REFL_FIELD})


@requires_gpu
def test_the_contraction_pinned_saturation_fit_is_load_bearing_for_g3():
    """Remove the pin IN MEMORY and require the gate to COLLAPSE.

    THIS TEST REPLACES A STALE ONE.  Its predecessor measured G3 "with the
    shared saturation fit repaired in memory", i.e. it substituted the pinned
    Horner chain IN for a contracted one it expected to find in
    ``gpuwm/core/kernels/thompson_aerosol_common.cuh``.  The header has since
    been pinned in the tree, so the substitution found nothing, the test
    measured the tree twice and its premise was false by design.  Running the
    substitution in the OTHER direction cannot go stale: it asserts that the
    pin currently in the tree is doing work, and it fails the moment someone
    "restores consistency with mp=8" by unpinning it.

    NO FILE IS WRITTEN.  ``gpuwm/core/kernels/__init__.py::module_source``
    assembles the exact string nvrtc compiles; this replaces the two pinned
    chains in that string for the six ``thompson_aerosol_*`` translation units
    ONLY.  ``thompson.cu`` is byte-frozen and is asserted below to contain no
    pinned chain at all, so mp=8 cannot be touched even accidentally.

    WHY THE PIN EXISTS (module_mp_thompson.F).  :3401 opens the whole
    condensation / CCN-activation block on ``ssatw(k) .gt. eps`` with
    ``eps = 1.E-15`` (:185), and :3197-3198 sets ssatw from ``qv/qvs`` with a
    snap to zero only inside that same 1e-15 band.  nvrtc defaults to
    ``--fmad=true`` and contracts a plain ``a + x*b`` Horner chain into FMAs;
    ``tools/thompson_wrf461_oracle/build_aero.sh`` compiles the oracle with
    ``gfortran -O2`` on baseline x86-64, which has NO fma instruction.  One
    float32 ulp therefore does not perturb a rate -- it FLIPS A BRANCH, and
    every cloud-free level of a saturated column condenses water WRF does not.

    MEASURED (RTX 5090, cupy 14.1.1), worst value over ALL 22 fixtures,
    tree as it stands -> same tree with the pin removed:

        unexceptioned clean          15/22  ->   4/22
        worst nc_per_kg residual  1.00e+00  ->  4.84e+36
        worst qc residual         1.00e+00  ->  8.86e+20
        worst qi residual         2.86e-07  ->  7.06e-03  (aero-ice-koop)
        worst nwfa_per_kg         0.00e+00  ->  7.22e-01

    and the four fixtures that survive unpinning are aero-ice-demott-dep,
    aero-scav-frozen, aero-sfc-emit and wp08-melt -- the ones with no live
    condensation at any level.  The nc and qc figures are not typos: nc and
    qc are created out of nothing at every level the flipped branch opens,
    so the relative difference against a fixture whose value there is exactly
    zero is unbounded.  (The 1.00e+00 entries in the left column are
    aero-cold-overlap's one-ulp remainder at level 4, which is a completely
    different phenomenon and is documented on :data:`_G3_RESIDUALS`.)
    """
    import cupy as cp

    import gpuwm.core.kernels as kernels

    _require_device()
    _tables_or_skip()

    substitutions = [(_pinned_chain(name), _contracted_chain(name))
                     for name in ("esl", "esi")]
    real_source = kernels.module_source

    # PRECONDITION 1: the pin is actually in the tree, in every aerosol unit.
    aerosol_modules = ("thompson_aerosol_sat", "thompson_aerosol_warm",
                       "thompson_aerosol_cold", "thompson_aerosol_state",
                       "thompson_aerosol_sed", "thompson_aerosol_probe")
    for module in aerosol_modules:
        source = real_source(module)
        for pinned, _ in substitutions:
            assert source.count(pinned) == 1, (
                f"{module} no longer carries exactly one contraction-pinned "
                "Horner chain; this test cannot measure what it claims to")
    # PRECONDITION 2: mp=8 has none, so the substitution cannot reach it.
    frozen = real_source("thompson")
    for pinned, _ in substitutions:
        assert pinned not in frozen

    def unpinned(name):
        source = real_source(name)
        if not name.startswith("thompson_aerosol"):
            return source          # thompson.cu is byte-frozen, never touched
        for pinned, contracted in substitutions:
            source = source.replace(pinned, contracted)
        return source

    kernels.module_source = unpinned
    kernels.load_module.cache_clear()
    kernels.get_kernel.cache_clear()
    try:
        broken = {}
        for scenario in _FIXTURES:
            broken[scenario], _ = _run_g3_without_level_exclusions(
                cp, scenario)
    finally:
        kernels.module_source = real_source
        kernels.load_module.cache_clear()
        kernels.get_kernel.cache_clear()
    _print_g3_table("G3 WITH THE CONTRACTION PIN REMOVED", broken)

    collapsed = [name for name in _FIXTURES
                 if not _flat_failures(name, broken[name])]
    print(f"UNPINNED CLEAN: {len(collapsed)}/{len(_FIXTURES)} -- "
          f"{sorted(collapsed)}")

    pinned_clean = set(_G3_UNEXCEPTIONED_CLEAN)
    assert len(collapsed) <= 5, (
        "removing the contraction pin no longer collapses G3, so either the "
        "pin has stopped mattering or the substitution stopped matching: "
        f"{len(collapsed)} fixtures still clean")
    assert len(pinned_clean) - len(collapsed) >= 10, (
        len(pinned_clean), len(collapsed))
    # The specific, huge signature: nc is manufactured where WRF makes none.
    worst_nc = max(row["nc_per_kg"] for row in broken.values())
    assert worst_nc > 1.0e30, worst_nc
    assert max(row["qc"] for row in broken.values()) > 1.0e15
    # ...and every fixture that the pin makes clean is genuinely one it
    # rescues, or the collapse above is being carried by a handful of them.
    rescued = pinned_clean - set(collapsed)
    assert len(rescued) >= 10, sorted(rescued)


# ---------------------------------------------------------------------------
# WP-12a: the widened gate, and the two defects widening it found.
# ---------------------------------------------------------------------------

def test_the_g3_gate_leaves_no_fixture_column_uncompared():
    """Every column the oracle writes is either compared or excused by name.

    THIS IS THE TEST THAT MAKES THE WIDENING DURABLE.  Before WP-12a the gate
    compared 15 of the 23 column fields and 1 of the 7 surface fields, and
    nothing said so: the omission was invisible because the inventory was a
    hand-written tuple with no relationship to the fixtures.  Here the
    fixtures' own CSV headers are the authority.  A future oracle rebuild that
    adds a column, or an edit that quietly drops one from
    ``_END_TO_END_FIELDS``/``_END_TO_END_SURFACE_FIELDS``, fails here instead
    of silently shrinking the gate.

    Runs on the host: it reads headers, not arrays, so it holds on a machine
    with no GPU and cannot be skipped away.
    """
    assert _FIXTURES, "no aerosol column fixtures present"

    compared = (set(_END_TO_END_FIELDS) | set(_END_TO_END_SURFACE_FIELDS)
                | {"refl_dbz"})
    excused = set(_UNCOMPARED_FIXTURE_COLUMNS)
    assert not (compared & excused), sorted(compared & excused)

    seen: set[str] = set()
    for name in _FIXTURES:
        for suffix in ("column", "surface"):
            with (_ORACLE_AERO / f"{name}-{suffix}.csv").open(
                    newline="", encoding="ascii") as stream:
                header = next(csv.reader(stream))
            seen.update(header)
            missing = set(header) - compared - excused
            assert not missing, (
                f"{name}-{suffix}.csv carries {sorted(missing)}, which the "
                "G3 gate neither compares nor excuses by name")

    # ...and nothing is claimed to be compared that the fixtures do not have.
    assert compared <= seen, sorted(compared - seen)
    assert excused <= seen, sorted(excused - seen)
    # Non-vacuous: the six quantities WP-12a added really are in there.
    for added in ("snownc_mm", "snowncv_mm", "graupelnc_mm", "graupelncv_mm",
                  "sr", "refl_dbz"):
        assert added in seen, added


def test_the_widened_gate_is_not_vacuous():
    """Each newly compared quantity carries real signal somewhere in the deck.

    A gate that compares six more fields proves nothing if all six are zero
    in all nineteen fixtures.  These are the magnitudes that make the
    widening worth having, read from the committed fixtures themselves so a
    future oracle rebuild that flattened them would fail here rather than
    quietly turn six comparisons into six no-ops.

    MEASURED on the committed deck:
        snownc/snowncv      9.2608e-04 mm (aero-scav-frozen), nonzero on 4
        graupelnc/graupelncv 6.5944e-03 mm (aero-scav-frozen), nonzero on 1
        sr                  1.0000 (aero-scav-frozen), nonzero on 5
        refl_dbz            51.98 dBZ (aero-cold-overlap); 44.30 on
                            aero-scav-frozen; 89 levels above the -35 floor
    """
    assert _FIXTURES
    peak = {key: 0.0 for key in _END_TO_END_SURFACE_FIELDS}
    nonzero = {key: 0 for key in _END_TO_END_SURFACE_FIELDS}
    max_dbz = -1.0e30
    signal_levels = floor_levels = 0
    for name in _FIXTURES:
        _, after, surface = _oracle_case(name)
        for key in _END_TO_END_SURFACE_FIELDS:
            value = abs(float(surface[key]))
            peak[key] = max(peak[key], value)
            nonzero[key] += value > 0.0
        refl = _column(after, "refl_dbz")
        max_dbz = max(max_dbz, float(refl.max()))
        signal_levels += int((refl != f32(-35.0)).sum())
        floor_levels += int((refl == f32(-35.0)).sum())

    assert peak["snownc_mm"] > 1.0e-4 and nonzero["snownc_mm"] >= 4
    assert peak["snowncv_mm"] == peak["snownc_mm"]
    assert peak["graupelnc_mm"] > 1.0e-3 and nonzero["graupelnc_mm"] >= 1
    assert peak["graupelncv_mm"] == peak["graupelnc_mm"]
    assert peak["sr"] >= 1.0 and nonzero["sr"] >= 5
    assert peak["rainnc_mm"] > 1.0e-2
    # Reflectivity: a real storm return, and a real floor population, so
    # both halves of the refl assertion in _run_g3 have something to bite on.
    assert max_dbz > 50.0, max_dbz
    assert signal_levels >= 80, signal_levels
    assert floor_levels >= 300, floor_levels


@requires_gpu
def test_the_sr_diagnostic_is_wrfs_epsilon_quotient_not_a_guarded_ratio():
    """SR is ``frozen/(RAINNCV + 1.e-12)``, epsilon included (:1308).

    WRF's mp_gt_driver:1308 is, verbatim::

        SR(i,j) = (pptsnow + pptgraul + pptice)/(RAINNCV(i,j)+1.e-12)

    -- no threshold, no MIN, and the epsilon is part of the VALUE, not a
    divide-by-zero guard.  RAINNCV is ``pptrain`` plus those same three
    frozen terms (:1298), so an entirely frozen column has SR strictly below
    1 in WRF while a guarded ``f/f`` returns exactly 1.  The adapter used to
    compute ``where(rainncv > 1e-12, min(1, frozen/rainncv), 0)``, which is
    that guarded ratio.

    WHAT THIS COSTS, MEASURED.  aero-ice-demott-dep (RAINNCV 2.7022e-09 mm,
    entirely frozen) and aero-ice-koop (3.2127e-09 mm, entirely frozen) are
    the two fixtures where the epsilon is resolvable, and the guarded form
    misses WRF by 3.701e-04 and 3.113e-04 -- more than two decades above the
    port's 2e-06 gate, on a field the WRF history stream publishes.  Both
    fixtures were in the CLEAN set before WP-12a's widening, because nothing
    compared SR at all.

    Fails on the pre-WP-12a adapter with exactly those two numbers.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    # The two fixtures where 1e-12 is a resolvable fraction of RAINNCV.
    sensitive = ("aero-ice-demott-dep", "aero-ice-koop")
    checked = 0
    for scenario in _FIXTURES:
        state, cfg, dt, _, _, surface, _ = _build_case(cp, scenario)
        _apply_thompson_aerosol(state, cfg, dt)
        cp.cuda.Stream.null.synchronize()

        def slot(name):
            return float(cp.asnumpy(state._scratch[name]).ravel()[0])

        got = slot("mp_sr")
        rainncv = slot("mp_rainncv")
        frozen = slot("mp_snowncv") + slot("mp_graupelncv")
        want = float(surface["sr"])
        want_rainncv = float(surface["rainncv_mm"])

        # WRF's own expression, recomputed here from gpuwm's own accumulators
        # so the assertion is about the FORM, not just the fixture value.
        # This holds on EVERY fixture, including the two whose RAINNCV itself
        # disagrees with WRF (aero-cloud-freeze-nc, aero-ice-demott-idxin):
        # SR is exactly as right as its own inputs, which is the property
        # this test owns.
        assert got == pytest.approx(
            frozen / (rainncv + 1.0e-12), rel=1.0e-6, abs=1.0e-9), scenario

        # Against WRF's own SR, wherever the input RAINNCV agrees with WRF.
        # Where it does not, the SR difference is that same residual read
        # through :1308 and is recorded in _G3_RESIDUALS, not here.
        inputs_agree = (
            abs(rainncv - want_rainncv) <= 2.0e-6 * max(abs(want_rainncv),
                                                        1.0e-30))
        if inputs_agree:
            assert abs(got - want) <= 2.0e-6 * max(abs(want), 1.0), scenario

        if scenario in sensitive:
            checked += 1
            # The fixture really is the discriminating case: entirely frozen
            # fallout, small enough that 1e-12 is a resolvable fraction.
            assert rainncv > 0.0 and frozen == rainncv, scenario
            guarded = min(1.0, frozen / rainncv)
            assert guarded == 1.0, scenario
            assert abs(guarded - want) / want > 1.0e-4, (
                f"{scenario} no longer discriminates the guarded form")
    assert checked == len(sensitive), checked


def test_the_adapter_hands_rain_evaporation_wrfs_pre_condensation_density(
        monkeypatch):
    """:3242-3243's density is not :3490's, and the adapter must pass it.

    module_mp_thompson.F diagnoses rho(k) at :3193, builds the working rain
    mass and number from it at :3242-3243, and freezes ``ilamr``/``N0_r`` from
    those at :3384-3388.  The condensation loop then OVERWRITES rho(k) at
    :3490, and the rain-evaporation loop at :3505-3520 reads that newer one
    for ``orho``/``rhof``/``vsc2``/``rvs``.  ``prv_rev`` therefore scales with
    the PRE-condensation density while everything around it scales with the
    post-condensation one.

    ``launch_aerosol_rain_evaporation`` exposes the older density as its
    ``entry_density`` argument and documents that
    ``launch_aerosol_saturation_adjust`` has already written it into the
    caller's ``reference_density`` buffer.  This pins that the adapter passes
    it and passes THAT buffer -- a host statement about the call graph, so it
    holds for every state, not just the fixtures.
    """
    calls, _, _, _, _ = _record_adapter_call(monkeypatch)
    _, _, sat_kwargs = _one(calls, "launch_aerosol_saturation_adjust")
    _, _, evap_kwargs = _one(calls, "launch_aerosol_rain_evaporation")

    assert "entry_density" in evap_kwargs, (
        "the rain evaporation is running on the post-condensation density "
        "alone; :3242-3243 uses the :3193 one")
    assert evap_kwargs["entry_density"] is sat_kwargs["reference_density"], (
        "entry_density must be the buffer the saturation adjustment wrote, "
        "which is WRF's :3193 density")
    # ...and NOT the buffer this same call writes its own post-condensation
    # density into, which is the mistake the argument exists to prevent.
    assert evap_kwargs["entry_density"] is not evap_kwargs[
        "reference_density"]


@requires_gpu
def test_the_pre_condensation_density_is_wrfs_3193_density_bitwise():
    """The buffer passed is the SAME number ``launch_tau1_density`` produces.

    The adapter has two independent routes to WRF's :3193 density: the
    explicit ``launch_tau1_density`` call at step 7 (which the working-number
    refresh needs) and the saturation adjustment's ``reference_density`` side
    output.  They must be the same value, or one of them is not :3193 and the
    argument above is misdirected.  MEASURED: bitwise identical on all 22
    fixtures at every level.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    for scenario in _FIXTURES:
        state, cfg, dt, _, _, _, _ = _build_case(cp, scenario)
        _apply_thompson_aerosol(state, cfg, dt)
        cp.cuda.Stream.null.synchronize()
        tau1 = state._scratch["mp_thompson_aero_tau1_density"]
        reference = state._scratch["mp_thompson_frozen_reference_density"]
        assert bool(cp.array_equal(tau1, reference)), scenario


@requires_gpu
def test_suppressing_the_pre_condensation_density_reproduces_the_old_defect():
    """The fails-before half: what the missing argument was worth, measured.

    Runs aero-reduces-to-classic twice -- once as the adapter stands, once
    with ``entry_density`` stripped from the call, which is exactly the
    pre-WP-12a adapter -- and shows that the difference between them is the
    density ratio WRF's :3193/:3490 split predicts, at the one level where
    the whole cloud evaporates inside the step.

    Numbers on this tree, 0-based level 5 (entry temperature 274.125 K,
    entry cloud water 2.6186e-04 kg/kg, all of which goes):

        without entry_density   qr 1.915e-03   nr 1.922e-03
        with    entry_density   qr 1.788e-07   nr 5.700e-06
        rho(:3490)/rho(:3193)   1.001953668
        evaporated-mass ratio   1.001944368 (qr) / 1.001941119 (nr)

    The last two lines are the attribution: the defect is not "about
    1e-03", it IS the density ratio, to five significant figures.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    import gpuwm.core.thompson_aerosol_sat as sat_module
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    scenario = "aero-reduces-to-classic"
    level = 5

    def run(*, suppress):
        state, cfg, dt, before, after, _, _ = _build_case(cp, scenario)
        real = sat_module.launch_aerosol_rain_evaporation
        if suppress:
            def without(*args, **kwargs):
                kwargs.pop("entry_density", None)
                return real(*args, **kwargs)
            sat_module.launch_aerosol_rain_evaporation = without
        try:
            _apply_thompson_aerosol(state, cfg, dt)
            cp.cuda.Stream.null.synchronize()
        finally:
            sat_module.launch_aerosol_rain_evaporation = real
        pull = (lambda field:
                cp.asnumpy(field).ravel().astype(np.float64)[level])
        return {
            "qr": pull(state.qr), "nr": pull(state.nr),
            "rho_pre": pull(state._scratch["mp_thompson_aero_tau1_density"]),
            "rho_post": pull(
                state._scratch["mp_thompson_rain_reference_density"]),
            "before": before, "after": after,
        }

    fixed = run(suppress=False)
    broken = run(suppress=True)

    want_qr = _column(fixed["after"], "qr").astype(np.float64)[level]
    want_nr = _column(fixed["after"], "nr_per_kg").astype(np.float64)[level]
    entry_qr = _column(fixed["before"], "qr").astype(np.float64)[level]
    entry_nr = _column(fixed["before"], "nr_per_kg").astype(np.float64)[level]

    def relative(got, want):
        return abs(got - want) / abs(want)

    # 1. The defect is real and is the size the old carve-out was sized for.
    assert 1.5e-3 < relative(broken["qr"], want_qr) < 2.5e-3
    assert 1.5e-3 < relative(broken["nr"], want_nr) < 2.5e-3
    # 2. The fix closes it by three decades on qr.
    assert relative(fixed["qr"], want_qr) < 1.0e-6
    assert relative(fixed["nr"], want_nr) < 1.0e-5
    assert relative(broken["qr"], want_qr) > 1000.0 * relative(
        fixed["qr"], want_qr)
    # 3. THE ATTRIBUTION.  The broken run's excess evaporation is the ratio
    #    of the two densities WRF keeps in play, not an unexplained epsilon.
    ratio = fixed["rho_post"] / fixed["rho_pre"]
    assert ratio > 1.0
    for key, entry, want in (("qr", entry_qr, want_qr),
                             ("nr", entry_nr, want_nr)):
        evaporated_ratio = (entry - broken[key]) / (entry - want)
        assert abs(evaporated_ratio - ratio) <= 1.0e-4 * ratio, (
            f"{key}: evaporated-mass ratio {evaporated_ratio!r} is not the "
            f"density ratio {ratio!r}")
    # 4. And the two densities really are different here, so (3) is not
    #    vacuously satisfied by rho_post == rho_pre.
    assert ratio - 1.0 > 1.0e-3


@requires_gpu
def test_rain_sedimentation_gets_wrfs_level_wise_working_rain_density():
    """module_mp_thompson.F:3237-3238 vs :3568-3570, end to end (WP-13a).

    The test above covers the density RAIN EVAPORATION reads.  This one covers
    the density RAIN SEDIMENTATION reads, which is a different WRF decision
    made in a different place and was wrong for every fixture in the deck.

    WRF forms the working rain mass and number sedimentation consumes
    (``rr(k)``/``nr(k)`` at :3794-3795) in exactly two places:

      * :3237-3238, from the :3193 TAU+1 density, at EVERY level with L_qr;
      * :3568/:3570, from the :3490 post-condensation density, and ONLY at the
        levels where the :3501-3502 rain-evaporation gate passed.

    ``thompson_aerosol_sat.cu`` used to write the post-condensation density
    into its ``reference_density`` output unconditionally, before its own
    gates, and ``microphysics_aerosol.py`` hands that same buffer to
    ``launch_rain_sedimentation`` -- so ArWen gave every level the :3568
    answer including the levels WRF never rewrote.

    ``aero-drop-evap`` is the cleanest witness: the :3501-3502 gate fires at
    NO level of that column (the condensation block runs first and vetoes it
    through :3502), so WRF's sedimentation sees the :3237 pair everywhere and
    the buffer must be the TAU+1 density at every level.  It measures
    RAINNC/RAINNCV 5.165e-04 wrong when it is not, and exactly 0.000e+00 when
    it is -- one of the two fixtures WP-13a moved into
    :data:`_G3_UNEXCEPTIONED_CLEAN`.

    Asserted three ways: the buffer identity, the surface consequence, and
    the reconstruction of the defect by feeding sedimentation the single
    density the frozen mp=8 kernel diagnoses for itself.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    import gpuwm.core.thompson as classic_module
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    scenario = "aero-drop-evap"

    def run(*, single_density):
        state, cfg, dt, before, after, surface, _ = _build_case(cp, scenario)
        real = classic_module.launch_rain_sedimentation
        if single_density:
            # thompson.cu:430-431 falls back to the density it diagnoses from
            # the state it is handed when reference_density is NULL, which on
            # this column is the post-condensation one -- i.e. exactly what the
            # adapter used to hand it at every level.
            def without(*args, **kwargs):
                kwargs.pop("reference_density", None)
                return real(*args, **kwargs)
            classic_module.launch_rain_sedimentation = without
        try:
            _apply_thompson_aerosol(state, cfg, dt)
            cp.cuda.Stream.null.synchronize()
        finally:
            classic_module.launch_rain_sedimentation = real
        return state, surface

    fixed, surface = run(single_density=False)
    tau1 = cp.asnumpy(
        fixed._scratch["mp_thompson_frozen_reference_density"]).ravel()
    handed = cp.asnumpy(
        fixed._scratch["mp_thompson_rain_reference_density"]).ravel()

    # 1. THE BUFFER.  The :3501-3502 gate fires nowhere on this column, so
    #    every level must carry WRF's :3193 TAU+1 density, bit for bit.
    np.testing.assert_array_equal(handed, tau1)
    # ...and this is not vacuous: the post-condensation density really is a
    # different number here.  The rain-evaporation kernel diagnoses it from
    # the same state, so reproduce :3490 on the host and require it to differ
    # somewhere.
    temperature = cp.asnumpy(
        fixed._scratch["mp_thompson_temperature"]).ravel().astype(np.float32)
    pressure = cp.asnumpy(fixed.p).ravel().astype(np.float32)
    qv = np.maximum(np.float32(1.0e-10),
                    cp.asnumpy(fixed.qv).ravel().astype(np.float32))
    post = (np.float32(0.622) * pressure
            / (np.float32(287.04) * temperature
               * (qv + np.float32(0.622)))).astype(np.float32)
    assert np.any(post != tau1), (
        "the two densities coincide everywhere on this column, so the "
        "identity above proves nothing; pick a different witness fixture")

    # 2. THE SURFACE CONSEQUENCE.  RAINNC and RAINNCV are bit-exact against
    #    WRF, which is the strongest statement available and needs no bound.
    for key, slot in (("rainnc_mm", "mp_rainnc"), ("rainncv_mm", "mp_rainncv")):
        mine = float(cp.asnumpy(fixed._scratch[slot]).ravel()[0])
        assert mine == float(surface[key]), (key, mine, float(surface[key]))
        assert mine > 0.0, key

    # 3. THE DEFECT, RECONSTRUCTED.  Hand sedimentation one density instead of
    #    WRF's level-wise mixture and the surface accumulation moves off WRF
    #    by the 5.165e-04 this fixture used to publish.
    broken, _ = run(single_density=True)
    for key, slot in (("rainnc_mm", "mp_rainnc"), ("rainncv_mm", "mp_rainncv")):
        mine = float(cp.asnumpy(broken._scratch[slot]).ravel()[0])
        residual = abs(mine - float(surface[key])) / abs(float(surface[key]))
        assert 4.0e-4 < residual < 7.0e-4, (key, residual)


#: (fixture, 0-based level, WRF's post-source-network qc, the FUSED value the
#: same kernel produced before WP-13b pinned the multiply-add).
#:
#: PROVENANCE, stated because these are the only two hard-coded WRF numbers in
#: this file that did not come out of a committed fixture.  The fixtures record
#: WRF's state after the WHOLE mp_thompson call, not after the source network,
#: so a stage-level value cannot be read from them.  The WP-13 forensics
#: package measured these from an instrumented copy of the pristine module
#: (WRITE statements only, verified inert by reproducing all 44 committed CSVs
#: byte for byte).  Both are re-derivable here without trusting that: WRF's
#: :3975 is ``qc1d(k) = qc1d(k) + qcten(k)*DT`` on a compiler with no FMA, so
#: the value must satisfy ``f32(entry) + f32(tendency*DT)`` EXACTLY, which is
#: the constant-free property asserted first below.  The fused column is what
#: this tree measured before the pin and is here only to prove the property
#: discriminates.
_SOURCE_APPLY_CELLS = (
    ("wp08-freeze", 0, 2.7275411412119865e-05, 2.7275422326056287e-05),
    ("aero-cold-overlap", 4, 1.5486410120502114e-05, 1.5486406482523307e-05),
)


@requires_gpu
def test_the_source_networks_round_the_tendency_before_adding_it():
    """module_mp_thompson.F:3973-4023 has no FMA, and nvrtc gave it one.

    WRF applies every accumulated tendency as ``q1d(k) = q1d(k) + qten(k)*DT``
    and the ``gfortran -O2`` baseline-x86-64 build the oracle comes from has no
    fused multiply-add instruction: ``qten*DT`` is rounded to REAL(4) and only
    then added.  ``thompson_aerosol_cold.cu`` and ``thompson_aerosol_warm.cu``
    wrote the same expression plainly, nvrtc (``--fmad=true`` by default)
    contracted it into a single ``fma``, and the product was never rounded.

    The difference is at most one ulp of the state -- and one ulp of the ENTRY
    value is a 1e-5 RELATIVE error at a level that is nearly emptied in the
    step, which is precisely where this port's surviving residuals live.
    ``thompson_aerosol_sat.cu`` already pinned this exact multiply-add for the
    rain-evaporation apply, so the source networks were the inconsistent half.

    THE DISCRIMINATOR IS A PROPERTY, NOT A CONSTANT.  If the product was
    rounded to float32 before the add, then at a level where the add itself is
    exact (a sink: the survivor is far smaller than the entry, so the exact sum
    lands on the entry's own ulp grid) the difference ``post - entry`` IS a
    float32 number.  If the product was fused, the difference is a multiple of
    the ulp of the SURVIVOR, needs more mantissa than float32 has, and fails to
    round-trip.  Measured on the two cells below, the fused values fail this
    and the rounded values pass it; no tolerance is involved.

    THE SECOND AND THIRD ASSERTIONS anchor it to WRF: the measured value must
    BE WRF's, bit for bit, and must not be the fused value this tree produced
    before the pin.  See :data:`_SOURCE_APPLY_CELLS` for where WRF's numbers
    come from.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    import gpuwm.core.thompson_aerosol_warm as warm_module
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    def post_source_qc(scenario):
        state, cfg, dt, before, _after, _s, _r = _build_case(cp, scenario)
        captured = {}
        real = warm_module.launch_aerosol_warm_source_network_from_owner

        def capture(*args, **kwargs):
            result = real(*args, **kwargs)
            cp.cuda.Stream.null.synchronize()
            captured["qc"] = cp.asnumpy(state.qc).ravel().astype(np.float64)
            return result

        warm_module.launch_aerosol_warm_source_network_from_owner = capture
        try:
            _apply_thompson_aerosol(state, cfg, dt)
            cp.cuda.Stream.null.synchronize()
        finally:
            warm_module.launch_aerosol_warm_source_network_from_owner = real
        assert "qc" in captured, scenario
        return captured["qc"], _column(before, "qc").astype(np.float64)

    for scenario, level, wrf, fused in _SOURCE_APPLY_CELLS:
        post, entry = post_source_qc(scenario)
        got = float(post[level])
        start = float(entry[level])
        # The regime the property needs: a sink that removes most of the mass,
        # so the add is exact and the only question is whether the product was
        # rounded.
        assert start > 0.0 and 0.0 < got < start / 8.0, (scenario, start, got)

        # 1. THE PROPERTY.  post - entry must round-trip through float32.
        delta = got - start
        assert float(np.float32(delta)) == delta, (
            f"{scenario} level {level}: the source network's qc apply left "
            f"post - entry = {delta!r}, which is not a float32; the "
            "tendency*dt product was fused into the add instead of being "
            "rounded first, and WRF's :3975 cannot do that")
        # ...and the property really does discriminate: the fused value this
        # kernel used to produce fails it.
        fused_delta = fused - start
        assert float(np.float32(fused_delta)) != fused_delta, (
            scenario, fused_delta)

        # 2. and 3. WRF's value, bitwise; and not the fused one.
        assert got == wrf, (
            f"{scenario} level {level}: post-source qc is {got!r}, WRF's is "
            f"{wrf!r}")
        assert got != fused, (scenario, got)

    # THE WARM HALF, which the two cells above cannot reach: both are
    # sub-freezing columns and the two source networks own disjoint cells by
    # entry temperature.  ``aero-nc-cap`` enters at 285 K at ALL 24 levels, so
    # every cell of it goes through thompson_aerosol_warm.cu -- and with that
    # file's apply pinned, its qc and nc columns are BITWISE equal to WRF's
    # after-state.  They measured 1.174e-07 and 9.428e-08 before the pin.
    state, cfg, dt, before, after, _s, _r = _build_case(cp, "aero-nc-cap")
    entry_temperature = _column(before, "temp_k").astype(np.float64)
    assert float(entry_temperature.min()) >= 273.15, float(
        entry_temperature.min())
    _apply_thompson_aerosol(state, cfg, dt)
    cp.cuda.Stream.null.synchronize()
    warm = _adapter_outputs(cp, state, before)
    for field in ("qc", "nc_per_kg"):
        want = _oracle_expectation(after, field)
        np.testing.assert_array_equal(warm[field], want, err_msg=(
            f"aero-nc-cap {field} is no longer bitwise equal to WRF; the warm "
            "network's terminal multiply-add is the only thing that made it "
            "so"))
        assert float(np.abs(want).max()) > 0.0, field


@requires_gpu
def test_the_qc_residual_is_exactly_one_extra_rounding():
    """The last column residual the port could name, LOCATED to one stage.

    ``aero-cloud-freeze-nc`` qc at 0-based level 4 and ``aero-cold-overlap`` qc
    at 0-based level 4 are the two cells where the port's answer and WRF's
    differ by exactly ONE float32 ulp of the value the level entered with.  The
    mechanism claimed for them on :data:`_G3_RESIDUALS` is the port's
    apply-per-stage against WRF's :3971-4021 apply-once, and this test measures
    that claim rather than asserting it.

    module_mp_thompson.F accumulates ``qcten`` across the whole network and
    evaluates ``qc1d(k) = qc1d(k) + qcten(k)*DT`` ONCE at :3975.  ArWen's
    mp=28 does that for ncten / nwfaten / nifaten (zeroed once, written by
    every kernel, applied once by thompson_aerosol_state.cu) but not for the
    mass fields, so qc is rounded once per stage that touches it.

    THE MEASUREMENT.  If the claim is right, then at these two cells qc is
    written by EXACTLY TWO stages -- the source network and the condensation --
    and every stage after the condensation must leave it bit-identical.  If a
    third stage moved it, "one extra rounding" would be the wrong story and the
    residual would need a different one.  So: snapshot qc around cloud
    sedimentation, the final phase cleanup and the terminal apply, and require
    all three to be inert here.

    THIS IS EVIDENCE, NOT A GATE ON THE FIX.  Closing it needs a qcten
    accumulator, which needs a scratch slot ``gpuwm/core/preflight.py`` budgets
    and a resequencing of readers in
    ``gpuwm/core/kernels/thompson_aerosol_sed.cu`` -- neither file belonged to
    the package that measured this.  Pinning the mechanism here is what stops
    the attribution drifting away from the numbers while it waits.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    import gpuwm.core.thompson_aerosol_sat as sat_module
    import gpuwm.core.thompson_aerosol_sed as sed_module
    import gpuwm.core.thompson_aerosol_state as state_module
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    for scenario, level in (("aero-cloud-freeze-nc", 4),
                            ("aero-cold-overlap", 4)):
        state, cfg, dt, before, after, _s, _r = _build_case(cp, scenario)
        trace: list[tuple[str, float]] = []
        restore = []

        def instrument(module, name, tag):
            real = getattr(module, name)
            restore.append((module, name, real))

            def wrapped(*args, **kwargs):
                cp.cuda.Stream.null.synchronize()
                trace.append((f"before {tag}",
                              float(cp.asnumpy(state.qc).ravel()[level])))
                result = real(*args, **kwargs)
                cp.cuda.Stream.null.synchronize()
                trace.append((f"after {tag}",
                              float(cp.asnumpy(state.qc).ravel()[level])))
                return result

            setattr(module, name, wrapped)

        instrument(sat_module, "launch_aerosol_saturation_adjust", "condense")
        instrument(sed_module, "launch_aa_cloud_sedimentation", "cloud sed")
        instrument(sed_module, "launch_aa_final_phase_cleanup", "cleanup")
        instrument(state_module, "launch_aerosol_state_finalize", "finalize")
        try:
            _apply_thompson_aerosol(state, cfg, dt)
            cp.cuda.Stream.null.synchronize()
        finally:
            for module, name, real in restore:
                setattr(module, name, real)

        seen = dict(trace)
        entry = float(_column(before, "qc").astype(np.float64)[level])
        want = float(_oracle_expectation(after, "qc")[level])
        final = float(cp.asnumpy(state.qc).ravel()[level])

        # The source network moved it, and so did the condensation: two
        # writes, which is one more rounding than WRF's :3975 performs.
        assert seen["before condense"] != entry, (scenario, trace)
        assert seen["after condense"] != seen["before condense"], (
            scenario, trace)
        # And NOTHING after the condensation touches it, so the residual
        # cannot be attributed to any later stage.
        for tag in ("cloud sed", "cleanup", "finalize"):
            assert seen[f"after {tag}"] == seen[f"before {tag}"], (
                f"{scenario}: the {tag} stage now moves qc at level {level}; "
                "the 'one extra rounding' attribution on _G3_RESIDUALS is no "
                f"longer the whole story.  Trace: {trace}")
        assert final == seen["after finalize"], (scenario, final, trace)

        # ...and the gap really is one ulp of the ENTRY value, which is the
        # signature of a second rounding at the entry scale rather than a rate
        # disagreement.
        ulp = _ulps(entry)
        assert abs(final - want) <= 1.0001 * ulp, (
            f"{scenario}: |got - want| is {abs(final - want) / ulp:.4f} ulp "
            "of the entry value, so this is no longer a one-rounding gap")
        assert abs(final - want) >= 0.99 * ulp, (
            f"{scenario}: the gap shrank to "
            f"{abs(final - want) / ulp:.4f} ulp; if it closed, remove the row "
            "from _G3_RESIDUALS and _RESIDUAL_ATTRIBUTION")


def _state_entering_rain_sedimentation(cp, scenario):
    """qr, nr and the reference density as ``launch_rain_sedimentation`` sees.

    The frozen mp=8 kernel is where the last two mechanisms live, so the only
    way to measure them is to read its arguments.  Nothing is modified: the
    wrapper snapshots and delegates.
    """
    import gpuwm.core.thompson as classic_module
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    state, cfg, dt, before, after, surface, _ = _build_case(cp, scenario)
    seen: dict[str, np.ndarray] = {}
    real = classic_module.launch_rain_sedimentation

    def wrap(qr, nr, temperature, pressure, qv, dz, rainnc, rainncv, dt_,
             **kwargs):
        cp.cuda.Stream.null.synchronize()
        seen["qr"] = cp.asnumpy(qr).ravel().astype(np.float32).copy()
        seen["nr"] = cp.asnumpy(nr).ravel().astype(np.float32).copy()
        seen["rho"] = cp.asnumpy(
            kwargs["reference_density"]).ravel().astype(np.float32).copy()
        return real(qr, nr, temperature, pressure, qv, dz, rainnc, rainncv,
                    dt_, **kwargs)

    classic_module.launch_rain_sedimentation = wrap
    try:
        _apply_thompson_aerosol(state, cfg, dt)
        cp.cuda.Stream.null.synchronize()
    finally:
        classic_module.launch_rain_sedimentation = real
    assert seen, scenario
    return seen, before, after, surface


@requires_gpu
def test_the_two_residuals_that_live_in_the_frozen_kernel_are_measured_here():
    """The last two mechanisms, MEASURED rather than asserted -- and NOT fixed.

    Two of this port's surviving residuals were traced to
    ``gpuwm/core/kernels/thompson.cu``, the byte-frozen, model-validated mp=8
    rain-sedimentation kernel that ``microphysics_aerosol.py`` reuses.  mp=28
    cannot fix either one: the file is frozen, ``gpuwm/core/microphysics.py``
    wires it the same way for mp=8, and changing it would move a trajectory
    validated against 92 classic fixtures that this package did not re-measure.

    Leaving them as prose would make them exactly the kind of claim that rots.
    So both are measured here, confined to the levels they occur at, and cited
    to WRF.  If either kernel-side condition is ever corrected, this test fails
    and the correction has to arrive with an updated record.

    ONE.  THE SEDIMENTATION PRESENCE GATE IS IN THE WRONG UNITS.
    module_mp_thompson.F:3616 opens rain fall-speed calculation on
    ``rr(k) .gt. R1`` -- a MASS CONCENTRATION in kg/m3, built at :3237 as
    ``qr*rho``.  thompson.cu:438 tests ``qr[idx] > 1.0e-12f``, a MIXING RATIO.
    They disagree wherever ``qr`` straddles R1/rho, and ``wp08-freeze`` has
    such a level: 0-based 1 carries qr = 8.526513e-13 kg/kg (below the mixing-
    ratio threshold, so ArWen gives it the level-above velocities at
    thompson.cu:469-470) and rr = 1.175581e-12 kg/m3 (above R1, so WRF gives it
    a real fall speed).  That is 29 of the 34 ULP of this fixture's published
    ``nr_per_kg`` 2.724e-06, measured on the frozen kernel by the sibling
    test below.

    THIS ATTRIBUTION WAS PUBLISHED AS FALSIFIED FOR PART OF 2026-08-01 AND
    THE FALSIFICATION WAS ITSELF WRONG.  It reasoned that WRF's :3236 test
    fails at this level too, so WRF floors rr to R1 and closes its own gate.
    It read ``qr1d + qrten*DT`` at the END of the step -- 8.521e-13, below R1
    -- and took that for the value :3236 tested.  It is not: the rain
    evaporation block at :3501 runs in between and SUBTRACTS from ``qrten``.
    Instrumented WRF (tools/thompson_wrf461_oracle/
    instrument_sedimentation_entry.py, proved inert by reproducing all 44
    committed aerosol CSVs byte for byte) records ``L_qr = .true.`` at that
    level, and ``L_qr`` is written nowhere between :3239/:3254 and here --
    so :3236 took the TRUE branch, and :3568's
    ``rr = MAX(R1, (qr1d + DT*qrten)*rho)`` left rr at
    1.17481534483293570E-12, above R1.  WRF opens the gate.

    Why cb765336 did not move the residual: that merge reconciled the
    sedimentation DENSITY (:3237 against :3568, level-wise), not the gate's
    UNITS.  A different mechanism landing without effect was never evidence
    against this one.

    TWO.  THE mvd_r CLAMP IS RE-APPLIED WHERE WRF NO LONGER CARRIES IT.
    WRF bounds the mean volume diameter at :3240-3250, in the :3236 TAU+1
    block only.  At the levels where the :3501-3502 rain-evaporation gate
    passes, :3568-3570 REBUILDS rr(k) and nr(k) and does NOT re-apply the
    clamp, so :3616-3627 derives the fall speed from an unclamped pair.
    thompson.cu:449-453 re-applies it unconditionally.  Measured on
    ``aero-reduces-to-classic``: exactly one level of the column, 0-based 6,
    has mvd_r below the 37.5 um lower bound -- 12.9007 um -- and it is exactly
    the level carrying that fixture's published 3.155e-03 qr/nr residual.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()

    r1 = np.float32(1.0e-12)

    # ONE.  wp08-freeze, the units disagreement, level by level.
    seen, _before, _after, _surface = _state_entering_rain_sedimentation(
        cp, "wp08-freeze")
    qr, rho = seen["qr"], seen["rho"]
    mass_concentration = (qr * rho).astype(np.float32)
    arwen_open = qr > r1                               # thompson.cu:438
    wrf_open = mass_concentration > r1                 # :3616
    disagree = sorted(np.nonzero(arwen_open != wrf_open)[0].tolist())
    assert disagree == [1], (
        f"the sedimentation presence gate now disagrees with WRF's at "
        f"{disagree} instead of [1]; the port's wp08-freeze attribution needs "
        "re-deriving")
    assert not arwen_open[1] and wrf_open[1]
    assert float(qr[1]) < 1.0e-12 < float(mass_concentration[1])
    assert float(rho[1]) > 1.0, float(rho[1])

    # TWO.  aero-reduces-to-classic, the clamp that WRF stopped carrying.
    seen, _before, _after, _surface = _state_entering_rain_sedimentation(
        cp, "aero-reduces-to-classic")
    qr, nr, rho = seen["qr"], seen["nr"], seen["rho"]
    # thompson.cu:439-443, transcribed on the host.
    am_r = np.float32(3.1415926536) * np.float32(1000.0) / np.float32(6.0)
    clamped = []
    present = []
    for level in range(qr.size):
        if not qr[level] > r1:
            continue
        present.append(level)
        rain_mass = np.float32(qr[level] * rho[level])
        rain_number = max(np.float32(1.0e-6), np.float32(nr[level] * rho[level]))
        lam = float(np.cbrt(float(am_r * np.float32(6.0) * rain_number
                                  / rain_mass)))
        mvd = 3.672 / lam
        if mvd < 37.5e-6 or mvd > 2.5e-3:
            clamped.append((level, mvd))
    assert len(present) >= 6, present
    assert [level for level, _ in clamped] == [6], clamped
    (_level, mvd), = clamped
    # 12.9007 um, an order below the 37.5 um bound -- not a borderline case.
    assert 12.5e-6 < mvd < 13.5e-6, mvd
    assert mvd < 37.5e-6 / 2.0
    # And it is the level the published residual sits at.
    assert "aero-reduces-to-classic" in _G3_ALLOWANCE_ONLY_CLEAN
    assert _NEAR_CANCELLATION_LEVELS["aero-reduces-to-classic"] == (6,)


#: WRF v4.6.1's own numbers at the moment rain sedimentation begins, for
#: ``wp08-freeze``, 0-based level 1 -- the level where ArWen's mixing-ratio
#: presence gate and WRF's mass-concentration one disagree.  From the
#: instrumented build described in
#: ``tools/thompson_wrf461_oracle/instrument_sedimentation_entry.py``, whose
#: inertness is proved by ``build_aero.sh`` reproducing every committed
#: fixture byte for byte afterwards (measured 2026-08-02: it did).
#:
#: ``L_qr`` is the field that settles the argument.  WRF writes it in exactly
#: two places before this point, :1887/:1904 and :3239/:3254, so ``.true.``
#: here means :3236 took its TRUE branch and ``rr`` was never floored to R1.
_WP08_FREEZE_WRF_SEDIMENTATION_ENTRY = {
    "level": 1,
    "L_qr": True,
    "rr": 1.17481534483293570e-12,          # > R1, so :3616 OPENS
    "nr": 1.58222410827875137e-02,
    "vtrk": 5.02253711223602295e-01,
    "vtnrk": 3.16547304391860962e-01,
    "qrten": 8.52096171399807645e-14,
    "nstep": 1,
    "ksed1_rain": 6,
    # The level above, whose velocities ArWen's closed gate inherits
    # (thompson.cu:469-470).  5.3x faster in number.
    "vtrk_above": 2.59040689468383789e+00,
    "vtnrk_above": 1.68873035907745361e+00,
}

#: The three-arm experiment below, as measured on the RTX 5090.
#: (mode, nr at 0-based level 0, ULP from WRF)
_WP08_FREEZE_GATE_ARMS = {
    "unmodified": (23.80756950378418, 34),
    "gate_opened": (23.80764389038086, 5),
}
_WP08_FREEZE_WRF_NR_LEVEL0 = 23.807634353637695


@requires_gpu
def test_the_wp08_freeze_residual_is_the_presence_gates_units_measured():
    """THE ATTRIBUTION, DEMONSTRATED ON THE FROZEN KERNEL ITSELF.

    The sibling test above shows the two gates disagree at exactly one level
    of this column.  A disagreement at a level is not yet a cause of a
    residual two levels' worth of arithmetic later, and this residual has
    already been attributed once, un-attributed once, and both times on
    reasoning rather than on a measurement.  So this measures it.

    THREE RUNS OF THE SAME FIXTURE THROUGH THE SAME FROZEN KERNEL.  Nothing
    in gpuwm is modified and ``thompson.cu`` stays byte-frozen:
    ``launch_rain_sedimentation`` is wrapped, and the wrapper edits ONE
    element of ``qr`` -- 0-based level 1, the disagreeing level -- before
    delegating.

      A  unmodified                         gate CLOSED (ArWen's own answer)
      B  qr[1] nudged just above 1.0e-12    gate OPENS, as WRF's is
      C  qr[1] nudged DOWN by the same absolute amount, gate stays CLOSED

    C IS THE CONTROL AND IT IS WHAT MAKES THIS A MEASUREMENT.  B changes the
    mass at that level by 1.5e-13 kg/kg as well as flipping the gate, so B
    alone cannot separate the two.  C applies a mass change of the same size
    in the other direction without flipping the gate.  If the gate is the
    cause, C is bit-identical to A.

    MEASURED: it is.  C reproduces A exactly, and B moves the level-0 ``nr``
    from 34 ULP away from WRF to 5 -- 29 of the 34 ULP, 85% of the published
    residual, bought by the gate alone.

    NOT FIXED, AND DELIBERATELY.  The gate is ``thompson.cu:438``, in the
    byte-frozen model-validated mp=8 rain-sedimentation kernel that
    ``gpuwm/core/microphysics.py`` wires the same way for mp=8.  Correcting
    it is an mp=8 change that has to arrive with the 92 classic fixtures
    re-measured; it is not mp=28's to make.  The remaining <=5 ULP is not
    separately attributed and is not claimed to be: B is a forced gate flip,
    not the correction, so 5 ULP is an upper bound on what is left over.
    """
    import cupy as cp

    import gpuwm.core.thompson as classic_module
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    _require_device()
    _tables_or_skip()

    level = _WP08_FREEZE_WRF_SEDIMENTATION_ENTRY["level"]
    threshold = np.float32(1.0e-12)

    def run(mode):
        state, cfg, dt, before, after, _surface, _ = _build_case(
            cp, "wp08-freeze")
        real = classic_module.launch_rain_sedimentation
        seen: dict[str, object] = {}

        def wrap(qr, nr, temperature, pressure, qv, dz, rainnc, rainncv,
                 dt_, **kwargs):
            cp.cuda.Stream.null.synchronize()
            flat = qr.ravel()
            original = np.float32(cp.asnumpy(flat)[level])
            seen["qr_entry"] = original
            seen["rho"] = np.float32(cp.asnumpy(
                kwargs["reference_density"]).ravel()[level])
            if mode == "up":
                used = np.float32(np.nextafter(threshold, np.float32(1.0)))
            elif mode == "down":
                used = np.float32(original - (threshold - original))
            else:
                used = original
            seen["qr_used"] = used
            seen["gate_open"] = bool(used > threshold)
            flat[level] = cp.float32(used)
            return real(qr, nr, temperature, pressure, qv, dz, rainnc,
                        rainncv, dt_, **kwargs)

        classic_module.launch_rain_sedimentation = wrap
        try:
            _apply_thompson_aerosol(state, cfg, dt)
            cp.cuda.Stream.null.synchronize()
        finally:
            classic_module.launch_rain_sedimentation = real
        got = _adapter_outputs(cp, state, before)["nr_per_kg"]
        want = _oracle_expectation(after, "nr_per_kg")
        return got, want, seen

    def ulp_gap(got, want):
        pair = [int(np.frombuffer(np.float32(value).tobytes(),
                                  dtype=np.int32)[0])
                for value in (got, want)]
        return abs(pair[0] - pair[1])

    plain, want, entry = run("none")
    opened, _want, opened_seen = run("up")
    control, _want, control_seen = run("down")

    # The premise: ArWen's gate really is closed at this level and WRF's
    # instrumented rr really is above R1.
    assert not entry["gate_open"]
    assert float(entry["qr_entry"]) < 1.0e-12
    assert float(np.float32(entry["qr_entry"] * entry["rho"])) > 1.0e-12
    assert _WP08_FREEZE_WRF_SEDIMENTATION_ENTRY["rr"] > 1.0e-12
    assert _WP08_FREEZE_WRF_SEDIMENTATION_ENTRY["L_qr"] is True
    assert opened_seen["gate_open"] and not control_seen["gate_open"]

    # WRF's number-weighted fall speed there against the one ArWen inherits.
    inherited = _WP08_FREEZE_WRF_SEDIMENTATION_ENTRY["vtnrk_above"]
    assert inherited / _WP08_FREEZE_WRF_SEDIMENTATION_ENTRY["vtnrk"] > 5.0

    # THE CONTROL.  A mass change of B's size, without the gate flip, does
    # nothing at all -- so the movement B produces is not the mass.
    assert control[0] == plain[0], (
        f"the control arm moved level-0 nr from {plain[0]!r} to "
        f"{control[0]!r}; a mass change alone is not supposed to, and if it "
        "does then arm B does not isolate the gate")
    assert float(control_seen["qr_used"]) < float(entry["qr_entry"])

    # THE MEASUREMENT.
    published, published_ulp = _WP08_FREEZE_GATE_ARMS["unmodified"]
    assert plain[0] == published, (plain[0], published)
    assert want[0] == _WP08_FREEZE_WRF_NR_LEVEL0, want[0]
    assert ulp_gap(plain[0], want[0]) == published_ulp

    expected, expected_ulp = _WP08_FREEZE_GATE_ARMS["gate_opened"]
    assert opened[0] == expected, (opened[0], expected)
    assert ulp_gap(opened[0], want[0]) == expected_ulp

    # 29 of 34, and the direction is toward WRF, not merely different.
    assert published_ulp - expected_ulp == 29
    assert abs(opened[0] - want[0]) < abs(plain[0] - want[0])

    # And the published residual is still the one the registry carries, so
    # this test and that number cannot drift apart.
    assert _G3_RESIDUALS["wp08-freeze"]["nr_per_kg"] == 2.7239e-06
    measured = abs(plain[0] - want[0]) / abs(want[0])
    assert f"{measured:.4e}" == "2.7239e-06", measured


@requires_gpu
def test_the_reduces_to_classic_residual_is_the_classic_paths():
    """What is left on aero-reduces-to-classic, and whose it is.

    THE ANSWER CHANGED IN WP-13a AND THE TEST CHANGED WITH IT.  It used to
    assert that mp=28's qr was BITWISE IDENTICAL to the frozen mp=8 pipeline's
    at 0-based levels 2 and 3, and read that identity as proof the residual was
    inherited from the classic warm-rain/fallout path (:3790-3936 carries no
    ``is_aerosol_aware`` branch, so both schemes run the same code there).

    That identity is gone, and it is gone because mp=28 got NEARER TO WRF, not
    further.  Restoring WRF's level-wise :3237-vs-:3568 sedimentation density
    made mp=28 BIT-EXACT against WRF at qr levels 1, 2 and 4, where the mp=8
    pipeline is 7.61e-05, 4.00e-05 and 6.27e-05 away.  So the surviving
    residual is NOT the classic path's and is no longer claimed to be.

    What this test asserts instead is strictly stronger than the old bitwise
    claim, and it is the claim the port actually wants to be able to make:

      1. at EVERY level of the column, mp=28 is at least as near WRF as the
         frozen, model-validated mp=8 pipeline is;
      2. at three qr levels mp=28 is bit-exact against WRF and mp=8 is not;
      3. mp=8 is strictly further away at seven of the eight levels that carry
         any rain at all, so (1) is not vacuous;
      4. the two schemes really are different runs (nc is prognostic in one and
         pinned at Nt_c in the other), so none of the above is a tautology.

    A regression that moves mp=28 back onto mp=8's answer fails (1) or (2).
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics import _apply_thompson
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    scenario = "aero-reduces-to-classic"
    state, cfg, dt, _, after, _, _ = _build_case(cp, scenario)
    _apply_thompson_aerosol(state, cfg, dt)
    cp.cuda.Stream.null.synchronize()
    aerosol_qr = cp.asnumpy(state.qr).ravel().astype(np.float64)

    classic_state, _, _, _, _, _, _ = _build_case(cp, scenario)
    _apply_thompson(
        classic_state,
        SimpleNamespace(mp_physics=8, no_mp_heating=0, mp_tend_lim=10.0), dt)
    cp.cuda.Stream.null.synchronize()
    classic_qr = cp.asnumpy(classic_state.qr).ravel().astype(np.float64)

    want = _column(after, "qr").astype(np.float64)

    def distance(got, level):
        reference = abs(want[level]) or 1.0
        return abs(got[level] - want[level]) / reference

    # (1) mp=28 is never further from WRF than mp=8, anywhere in the column.
    worse = [level for level in range(want.size)
             if distance(aerosol_qr, level) > distance(classic_qr, level)]
    assert not worse, (
        "mp=28 is further from WRF than the frozen mp=8 pipeline at 0-based "
        f"levels {worse}: "
        + "; ".join(f"{lev}: {distance(aerosol_qr, lev):.4e} vs "
                    f"{distance(classic_qr, lev):.4e}" for lev in worse))

    # (2) and it is BIT-EXACT against WRF where mp=8 is not.
    exact = [level for level in range(want.size)
             if want[level] != 0.0 and aerosol_qr[level] == want[level]]
    assert set(exact) >= {1, 2, 4}, exact
    for level in (1, 2, 4):
        assert classic_qr[level] != want[level], (
            f"level {level}: mp=8 is bit-exact too, so mp=28 being bit-exact "
            "is not a finding about the aerosol port")
        assert distance(classic_qr, level) > 1.0e-5, (
            level, distance(classic_qr, level))

    # (3) non-vacuous: over the levels that carry rain, mp=8 is STRICTLY
    #     further away almost everywhere -- this is a real ordering, not a tie.
    rain_levels = [level for level in range(want.size)
                   if want[level] != 0.0 or classic_qr[level] != 0.0]
    assert len(rain_levels) == 7, rain_levels
    strictly = [level for level in rain_levels
                if distance(classic_qr, level) > distance(aerosol_qr, level)]
    assert len(strictly) == len(rain_levels), (strictly, rain_levels)

    # (4) Non-vacuous: the two schemes are genuinely different everywhere the
    # aerosol physics is live, so the ordering above is a finding.
    aerosol_nc = float(cp.asnumpy(state.nc).ravel()[0])
    classic_nc = float(cp.asnumpy(classic_state.nc).ravel()[0])
    assert aerosol_nc != classic_nc


@requires_gpu
def test_the_near_cancellation_level_is_bounded_in_ulps_not_excluded():
    """The one held-out level, held to an absolute bound instead.

    aero-reduces-to-classic 0-based level 6 enters with 3.1695777e-07 kg/kg
    of rain and evaporates 99.958% of it in one step, so its surviving qr is
    the difference of two nearly equal float32 numbers and a relative gate
    there is a gate on the fifth significant digit of a cancellation.  It is
    excluded from the RELATIVE comparison in :func:`_run_g3` and bounded here
    in ulps of the entry value instead.

    This also replaces the exclusion's old justification, which was that
    "WRF's own qr survives at a value BOTH ArWen ports drive to exactly
    zero".  After WP-12a's rain-density fix that is no longer true: mp=28 now
    produces 1.3384e-10 kg/kg there against WRF's 1.3426e-10, and the test
    below requires it to be non-zero so the old sentence cannot silently
    come back.

    MEASURED: qr 14.9 ulp of its entry value, nr_per_kg 4.05 ulp of its.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    for scenario, levels in _NEAR_CANCELLATION_LEVELS.items():
        state, cfg, dt, before, after, _, _ = _build_case(cp, scenario)
        _apply_thompson_aerosol(state, cfg, dt)
        cp.cuda.Stream.null.synchronize()
        for level in levels:
            for field, key in ((state.qr, "qr"), (state.nr, "nr_per_kg")):
                entry = float(_column(before, key)[level])
                want = float(_column(after, key)[level])
                got = float(cp.asnumpy(field).ravel()[level])
                # It really is a near-total consumption, so the carve-out
                # cannot be reused for a level that merely disagrees.
                assert entry > 0.0 and want > 0.0
                assert (entry - want) / entry > 0.999, (scenario, key)
                # The surviving value is non-zero: the port no longer just
                # collapses this level, which is what the old justification
                # assumed.
                assert got > 0.0, (scenario, key)
                ulp = _ulps(entry)
                assert abs(got - want) <= _NEAR_CANCELLATION_ULPS * ulp, (
                    f"{scenario} level {level} {key}: "
                    f"{abs(got - want) / ulp:.1f} ulp of the entry value "
                    f"exceeds {_NEAR_CANCELLATION_ULPS}")


#: WHERE EACH SURVIVING RESIDUAL LIVES AND WHAT REGIME IT LIVES IN, so the
#: narrative on ``_G3_RESIDUALS`` is checkable rather than assertive.
#: ``(fixture, field, 0-based level, regime, scale)`` with regime one of:
#:
#:   "created"  -- the field enters the level at EXACTLY zero.  ``scale`` is
#:                 |got - want| as a fraction of the column peak of the
#:                 oracle's own after-state for that field, which is the only
#:                 non-degenerate denominator available.
#:   "consumed" -- at least 98% of the entry value is removed in the step.
#:                 ``scale`` is |got - want| in ULPS OF THE ENTRY VALUE, which
#:                 is the resolution the difference is actually carried at.
#:   "partial"  -- neither of the above; ``scale`` is again ulps of the entry
#:                 value, and it is the honest statement that the residual is
#:                 the accumulated rounding of a chain of processes rather
#:                 than a threshold or a cancellation.
#:   "derived"  -- the quantity is a diagnostic FUNCTION of other quantities
#:                 at the same level that already appear in this table.
#:                 ``scale`` is the relative difference; the attribution is
#:                 the dependency, not the size.
#:   "surface"  -- a column-integrated surface accumulation; it has no level.
#:
#: THE HEADLINE, AND IT IS AN IMPROVEMENT.  Every column residual left in the
#: port is now "created", "consumed", "partial" or "derived".  The "rate"
#: regime -- a disagreement in a process rate itself, at values far from any
#: threshold -- is EMPTY.  It used to contain aero-ice-koop's qi 1.612e-03 and
#: ni 1.764e-03, which three auditors called the port's largest genuine
#: physics gap; WP-06 closed it and this file now measures 1.534e-07 / 
#: 3.396e-07 there, inside the flat gate, so aero-ice-koop is in
#: :data:`_G3_UNEXCEPTIONED_CLEAN` and has no row here at all.
#:
#: THE TWO EXTREME ROWS, IN WRF'S OWN TERMS.
#:   aero-cold-overlap qc at level 4 reads a relative 1.000e+00 because WRF
#:   ends the step with 1.4551915e-11 kg/kg -- exactly 2**-36, and exactly
#:   1.000 float32 ulp of the 2.325216e-04 kg/kg the level entered with --
#:   while gpuwm ends at exactly zero.  The two implementations remove the
#:   same cloud water to within one ulp; the relative metric divides by the
#:   remainder.  The same level's nc is 0.229 ulp of its entry value.
#:   aero-cold-overlap effc_m at level 4 follows from that single ulp through
#:   a BRANCH, not a rate: :5625 sets rc = MAX(R1, qc*rho) and :5638 CYCLEs
#:   the whole cloud-radius loop when rc <= R1 with R1 = 1.E-12 (:183), so
#:   WRF's 1.455e-11 clears the threshold and computes 13.118 micron from
#:   :5646-5647 while gpuwm's exact zero leaves the RE_QC_BG background that
#:   mp_gt_driver:1475 clamps to 2.49 micron.  Recorded as a MISS, not
#:   allowanced.
_RESIDUAL_ATTRIBUTION = (
    ("aero-cloud-freeze-nc", "qc", 4, "consumed", 1.0),
    ("aero-cold-overlap", "qc", 4, "consumed", 1.0),
    ("aero-cold-overlap", "nc_per_kg", 4, "consumed", 0.2292),
    ("aero-cold-overlap", "effc_m", 4, "derived", 0.8102),
    ("aero-cold-overlap", "nr_per_kg", 6, "consumed", 0.6108),
    ("aero-cold-overlap", "qr", 6, "consumed", 1.789),
    ("wp08-freeze", "nr_per_kg", 0, "created", 2.724e-06),
    ("wp08-nusweep", "qr", 12, "created", 2.502e-08),
)

#: Which quantities a "derived" row depends on, at the same level.  Named so
#: the dependency is asserted rather than asserted-about.
_DERIVED_INPUTS = {"effc_m": ("qc", "nc_per_kg"),
                   "effi_m": ("qi", "ni_per_kg"),
                   "effs_m": ("qs",)}


@requires_gpu
def test_every_surviving_residual_is_located_and_its_regime_is_stated():
    """The residual narrative, asserted level by level and scale by scale.

    :data:`_G3_RESIDUALS` ratchets the MAGNITUDE of what is left.  This
    ratchets the STORY: which level each residual sits at, whether the field
    there is being created from zero, consumed almost entirely, carried
    through a chain of processes, or derived from another row -- and, for
    every one of them, the SCALE the difference is actually carried at.
    Without it the attribution prose drifts away from the numbers silently,
    which is exactly what happened to the 2.5e-03 carve-out an earlier wave
    replaced, whose comment named the wrong level and a mechanism that
    measurement did not support.

    COMPLETENESS IS ASSERTED, not assumed: every (fixture, field) in
    :data:`_G3_RESIDUALS` must have a row here and vice versa, so a new
    residual cannot appear with no story attached to it.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    recorded = {(fixture, field) for fixture, fields in _G3_RESIDUALS.items()
                for field in fields}
    stated = {(row[0], row[1]) for row in _RESIDUAL_ATTRIBUTION}
    assert stated == recorded, (
        f"attribution rows with no residual: {sorted(stated - recorded)}; "
        f"residuals with no attribution: {sorted(recorded - stated)}")

    attribution: dict[str, list] = {}
    for fixture, field, level, regime, scale in _RESIDUAL_ATTRIBUTION:
        attribution.setdefault(fixture, []).append(
            (field, level, regime, scale))

    seen_regimes = set()
    for fixture, expectations in sorted(attribution.items()):
        state, cfg, dt, before, after, _, _ = _build_case(cp, fixture)
        _apply_thompson_aerosol(state, cfg, dt)
        cp.cuda.Stream.null.synchronize()
        got = _adapter_outputs(cp, state, before)
        for field, level, regime, scale in expectations:
            seen_regimes.add(regime)
            if regime == "surface":
                assert level == -1 and field in _END_TO_END_SURFACE_FIELDS
                continue
            want = _oracle_expectation(after, field)
            relative = np.where(
                (got[field] == 0.0) & (want == 0.0), 0.0,
                np.abs(got[field] - want)
                / np.maximum(np.abs(want), 1.0e-30))
            assert int(np.argmax(relative)) == level, (
                f"{fixture}.{field}: worst residual moved from 0-based level "
                f"{level} to {int(np.argmax(relative))}")
            entry = float(_column(before, field)[level])
            final = float(want[level])
            difference = abs(float(got[field][level]) - final)
            if regime == "created":
                assert entry == 0.0, (fixture, field, entry)
                assert final > 0.0, (fixture, field, final)
                measured_scale = difference / float(np.abs(want).max())
            elif regime == "consumed":
                assert entry > 0.0 and final >= 0.0
                assert (entry - final) / entry >= 0.98, (
                    fixture, field, (entry - final) / entry)
                measured_scale = difference / _ulps(entry)
            elif regime == "partial":
                assert entry > 0.0 and final > 0.0
                assert (entry - final) / entry < 0.98, (
                    f"{fixture}.{field} is now a near-total consumption; "
                    "move it to the 'consumed' regime")
                measured_scale = difference / _ulps(entry)
            elif regime == "derived":
                inputs = _DERIVED_INPUTS[field]
                for source in inputs:
                    assert source in _G3_RESIDUALS[fixture], (
                        f"{fixture}.{field} is recorded as derived from "
                        f"{source}, but {source} is not itself above the "
                        "gate; the attribution is wrong")
                measured_scale = difference / max(abs(final), 1.0e-30)
            else:                                        # pragma: no cover
                raise AssertionError(f"unknown regime {regime!r}")
            assert measured_scale <= scale * 1.01, (
                f"{fixture}.{field} at level {level}: {regime} scale grew "
                f"from {scale:.4g} to {measured_scale:.4g}")

    # THE REGIMES THAT ARE NOW EMPTY, AND WHY THAT IS A RESULT RATHER THAN A
    # WEAKENING.  This used to require all five regimes to be exercised.  Two
    # of them no longer are, because WP-13a closed every residual in them:
    #
    #   "surface" -- there were FIVE surface rows (aero-cloud-freeze-nc's
    #       rainnc/rainncv/sr, aero-drop-evap's rainnc/rainncv,
    #       aero-ice-demott-idxin's sr/rainnc/rainncv).  The port now has NO
    #       surface residual above the gate on ANY fixture, which is asserted
    #       positively below rather than left as an absence.
    #   "partial" -- the single row was aero-ice-demott-idxin nr_per_kg at
    #       level 5 (4.594e-06, 23.0 ulps of entry); it now measures 3.61e-07.
    #
    # So the assertion becomes: every regime PRESENT is one the table defines
    # (the elif chain above already raises otherwise), the two live regimes
    # are still both exercised, and the empty ones are empty for a reason that
    # is itself checked.
    assert seen_regimes <= {"created", "consumed", "partial", "derived",
                            "surface"}, sorted(seen_regimes)
    assert {"created", "consumed"} <= seen_regimes, sorted(seen_regimes)
    # ...and the regime that would be a genuine physics gap is EMPTY.
    assert not any(row[3] == "rate" for row in _RESIDUAL_ATTRIBUTION)

    # NO SURFACE DIAGNOSTIC IS ABOVE THE GATE ANYWHERE.  All seven of WRF's
    # mp_gt_driver:1298-1308 surface writes, on all 22 fixtures, inside the
    # flat 2.0e-06 gate with no allowance -- 154 cells.  Stated as an
    # assertion so "there are no surface rows" cannot become true again by a
    # row being deleted instead of a residual being closed.
    flat_table = _g3_table_unexceptioned(cp)
    above = {f"{fixture}.{field}": flat_table[fixture][field]
             for fixture in _FIXTURES
             for field in _END_TO_END_SURFACE_FIELDS
             if flat_table[fixture][field] > _END_TO_END_DEFAULT_BOUND}
    assert not above, (
        "a surface diagnostic is above the flat gate again; add its row to "
        f"_RESIDUAL_ATTRIBUTION and _G3_RESIDUALS: {above}")
    assert len(_END_TO_END_SURFACE_FIELDS) == 7, _END_TO_END_SURFACE_FIELDS


@requires_gpu
def test_the_koop_rate_gap_the_auditors_called_the_largest_is_closed():
    """aero-ice-koop, re-measured: the port's largest gap is no longer there.

    Three adversarial auditors recorded ``aero-ice-koop`` qi 1.612e-03 and ni
    1.764e-03 at 0-based level 14 as the port's ONE surviving rate
    disagreement -- neither a threshold residual nor a near-cancellation, ice
    created from nothing to 6.032e-08 kg/kg against WRF's 6.042e-08.  It is
    closed on this tree.  This is the re-measurement, kept as its own gate so
    the closure cannot silently regress into the residual table.

    Homogeneous haze freezing (``iceKoop``, :2569-2596 in mp_gt_driver's
    callee) fires only below 238 K with ssati >= 0.4, so this is the only
    fixture that reaches it; a regression here would be invisible everywhere
    else in the suite.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    state, cfg, dt, before, after, _, _ = _build_case(cp, "aero-ice-koop")
    _apply_thompson_aerosol(state, cfg, dt)
    cp.cuda.Stream.null.synchronize()

    level = 14
    mine = float(cp.asnumpy(state.qi).ravel()[level])
    theirs = float(_column(after, "qi")[level])
    # Both values are four decades above WRF's R1 floor, so no threshold or
    # cancellation is available to explain away a disagreement here.
    assert mine > 1.0e4 * R1 and theirs > 1.0e4 * R1, (mine, theirs)
    assert abs(mine - theirs) / theirs <= 2.0e-6, (
        f"the Koop rate gap has reopened: qi {mine!r} against WRF's "
        f"{theirs!r}")
    mine_n = float(cp.asnumpy(state.ni).ravel()[level])
    theirs_n = float(_column(after, "ni_per_kg")[level])
    assert abs(mine_n - theirs_n) / theirs_n <= 2.0e-6, (mine_n, theirs_n)
    # Non-vacuous: the level really is ice created from nothing in the step.
    assert float(_column(before, "qi")[level]) == 0.0
    assert theirs > 0.0
    # And the whole fixture clears the flat gate, which is the claim that
    # replaced the old 1.6e-03 row.
    assert "aero-ice-koop" in _G3_UNEXCEPTIONED_CLEAN


@requires_gpu
def test_a_due_reflectivity_call_perturbs_no_prognostic_field():
    """REFL_10CM is a diagnostic, and G3 issues it on every call.

    The oracle produced all 22 fixtures with ``diagflag=.true.,
    do_radar_ref=1`` (run_column_aero.F90:296), so the G3 gate runs the
    adapter with ``refl_10cm_due=True`` in order to compare the fixtures'
    ``refl_dbz`` column.  That is only legitimate if a due call and a
    not-due call leave identical state -- WRF's ``calc_refl10cm`` (:5710) is
    INTENT(OUT) in ``dBZ`` and touches nothing else, and gpuwm's branch takes
    the same arguments (gpuwm/core/refl.py:549-552).

    Bit-identity on every compared field of every fixture, not a tolerance.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    for scenario in _FIXTURES:
        outputs = []
        for due in (False, True):
            state, cfg, dt, before, _, _, _ = _build_case(cp, scenario)
            _apply_thompson_aerosol(state, cfg, dt, refl_10cm_due=due)
            cp.cuda.Stream.null.synchronize()
            assert (state.physics.refl_10cm is not None) is due, scenario
            surfaces = {slot: cp.asnumpy(state._scratch[slot]).copy()
                        for slot in _END_TO_END_SURFACE_FIELDS.values()}
            outputs.append((_adapter_outputs(cp, state, before), surfaces))
        (plain, plain_surface), (due_out, due_surface) = outputs
        for field in _END_TO_END_FIELDS:
            assert np.array_equal(plain[field], due_out[field]), (
                scenario, field)
        for slot in _END_TO_END_SURFACE_FIELDS.values():
            assert np.array_equal(
                plain_surface[slot], due_surface[slot]), (scenario, slot)


@requires_gpu
def test_accumulators_are_zeroed_at_entry_not_carried_between_calls():
    """(1) on real kernels: poison the three slots and require bit-identity.

    ``state.scratch`` is persistent by design, so the pre-fix failure mode is
    that step N+1 starts with step N's ``ncten``/``nwfaten``/``nifaten``.  A
    unit test cannot see this -- it allocates a fresh zeroed accumulator every
    time.  Here the SAME state object is reused, the three slots are filled
    with a value the size of a real tendency, the entry fields are restored,
    and the whole call is re-run: the outputs must be bit-identical.

    Without the ``zero_aerosol_accumulators`` call at adapter entry this
    fails on ``nc``, ``nwfa`` and ``nifa`` immediately, because the terminal
    apply at :3972-4021 integrates the poisoned tendency straight into state.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    scenario = "aero-warm-overlap"
    state, cfg, dt, before, _, surface, _ = _build_case(cp, scenario)
    entry = {name: getattr(state, name).copy() for name in
             ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
              "nwfa", "nifa", "thp", "h_diabatic", "effc", "effi", "effs")}
    _apply_thompson_aerosol(state, cfg, dt)
    cp.cuda.Stream.null.synchronize()
    first = _adapter_outputs(cp, state, before)

    for name, value in entry.items():
        getattr(state, name)[...] = value
    for slot in ("mp_thompson_aero_ncten", "mp_thompson_aero_nwfaten",
                 "mp_thompson_aero_nifaten"):
        state._scratch[slot][...] = f32(1.0e5)
    _apply_thompson_aerosol(state, cfg, dt)
    cp.cuda.Stream.null.synchronize()
    second = _adapter_outputs(cp, state, before)

    del surface
    for field in _END_TO_END_FIELDS:
        assert np.array_equal(first[field], second[field]), (
            f"{field} depends on the previous call's accumulator contents; "
            "the entry zeroing is missing or incomplete")


@requires_gpu
def test_the_terminal_apply_is_the_only_writer_of_nc_nwfa_and_nifa():
    """(1) on real kernels: state moves exactly once, and only there.

    Runs the real adapter twice on the same entry column -- once complete,
    once with ``launch_aerosol_state_finalize`` replaced by a recorder that
    does nothing -- and requires ``nc``/``nwfa``/``nifa`` to be UNCHANGED from
    entry in the second run except for the surface emission's lowest level.
    That is the strongest available statement that no other kernel writes
    them, and it is exactly the property four other packages assumed.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    import gpuwm.core.thompson_aerosol_state as state_module
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    state, cfg, dt, before, _, _, _ = _build_case(cp, "aero-cold-overlap")
    entry = {name: getattr(state, name).copy()
             for name in ("nc", "nwfa", "nifa")}
    # module_mp_thompson.F:1844-1845 zeroes nc1d wherever qc1d <= R1, and the
    # entry droplet diagnosis is what applies it.  That rewrite is PART of
    # forming the entry value, so the expectation below carries it.
    entry["nc"] = cp.where(
        state.qc > np.float32(R1), entry["nc"], np.float32(0.0))

    real = state_module.launch_aerosol_state_finalize
    try:
        state_module.launch_aerosol_state_finalize = (
            lambda *args, **kwargs: None)
        _apply_thompson_aerosol(state, cfg, dt)
        cp.cuda.Stream.null.synchronize()
    finally:
        state_module.launch_aerosol_state_finalize = real

    # nc is still the ENTRY value everywhere: :1844-1845's zeroing is part of
    # forming the entry value, and it happened before the copy above.
    assert cp.array_equal(state.nc, entry["nc"])
    # nwfa/nifa move only at the lowest level, and only by the surface
    # emission (mp_gt_driver:1310-1327), which is deliberately outside the
    # terminal clamp.
    for name in ("nwfa", "nifa"):
        got = getattr(state, name)
        assert cp.array_equal(got[1:], entry[name][1:]), name
    surface_delta = float(cp.asnumpy(state.nwfa - entry["nwfa"]).ravel()[0])
    expected = float(cp.asnumpy(state.nwfa2d).ravel()[0]) * dt
    assert abs(surface_delta - expected) <= 1.0e-6 * max(abs(expected), 1.0)
    del before


@requires_gpu
def test_the_cuda_symbol_inventory_of_one_mp28_call_is_pinned():
    """(4) at the symbol level: every kernel a real mp=28 call launches.

    Records every ``get_kernel(module, symbol)`` request the adapter makes.
    This is the strongest form of the Cooper gate -- it does not depend on
    which Python name a launcher has -- and it doubles as an inventory
    receipt: a kernel appearing or disappearing from an mp=28 call is a
    change to the scheme, not a refactor.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    import gpuwm.core.kernels as kernels
    import gpuwm.core.thompson  # noqa: F401  (bind before patching)
    import gpuwm.core.thompson_aerosol_cold  # noqa: F401
    import gpuwm.core.thompson_aerosol_launch  # noqa: F401
    import gpuwm.core.thompson_aerosol_sat  # noqa: F401
    import gpuwm.core.thompson_aerosol_sed  # noqa: F401
    import gpuwm.core.thompson_aerosol_state  # noqa: F401
    import gpuwm.core.thompson_aerosol_warm  # noqa: F401
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol
    from gpuwm.core.thompson_aerosol import COOPER_BEARING_CLASSIC_KERNELS

    requested: list[tuple[str, str]] = []
    real = kernels.get_kernel

    def recording(module, func):
        requested.append((module, func))
        return real(module, func)

    patched = [kernels]
    for module in list(sys.modules.values()):
        if getattr(module, "get_kernel", None) is real:
            patched.append(module)
    for module in patched:
        module.get_kernel = recording
    try:
        state, cfg, dt, _, _, _, _ = _build_case(cp, "aero-reduces-to-classic")
        _apply_thompson_aerosol(state, cfg, dt)
        cp.cuda.Stream.null.synchronize()
        not_due = sorted(set(requested))
        requested.clear()
        # AND AGAIN WITH REFLECTIVITY DUE, which is how the G3 gate runs the
        # adapter (the fixtures were produced with do_radar_ref=1).  The
        # Cooper statement has to hold on the path the acceptance gate
        # actually exercises, not only on the cheaper one.
        due_state, cfg, dt, _, _, _, _ = _build_case(
            cp, "aero-reduces-to-classic")
        _apply_thompson_aerosol(due_state, cfg, dt, refl_10cm_due=True)
        cp.cuda.Stream.null.synchronize()
        due = sorted(set(requested))
    finally:
        for module in patched:
            module.get_kernel = real

    # The due call launches the SAME set through the Thompson module loader
    # and nothing more: gpuwm/core/refl.py:453 loads calc_refl10cm through its
    # own ``_column_kernel`` specialiser, not ``kernels.get_kernel``, so the
    # reflectivity kernel is deliberately absent from this inventory.  What
    # matters here is the negative: making the call due does not reach for a
    # single extra Thompson kernel, Cooper-bearing or otherwise.
    assert set(due) == set(not_due), sorted(
        set(due) ^ set(not_due))
    assert due_state.physics.refl_10cm is not None, (
        "the due call stashed no REFL_10CM frame, so the comparison above "
        "did not actually exercise the reflectivity path")

    launched = not_due
    print("\nCUDA SYMBOLS LAUNCHED BY ONE mp=28 CALL:\n  "
          + "\n  ".join(f"{m}::{f}" for m, f in launched))
    symbols = {func for _, func in launched}
    assert not (symbols & set(COOPER_BEARING_CLASSIC_KERNELS)), sorted(
        symbols & set(COOPER_BEARING_CLASSIC_KERNELS))

    modules = {module for module, _ in launched}
    assert modules <= {
        "thompson", "thompson_aerosol_state", "thompson_aerosol_sat",
        "thompson_aerosol_cold", "thompson_aerosol_warm",
        "thompson_aerosol_sed"}, sorted(modules)
    # The frozen translation unit contributes ONLY the graupel-number
    # diagnostic, the two column masks and the four reused fallout kernels.
    classic = sorted(f for m, f in launched if m == "thompson")
    for func in classic:
        assert ("sediment" in func or "column_mask" in func
                or "classic_graupel_number" in func), func


@requires_gpu
def test_preflight_module_inventory_covers_every_launched_module():
    """Every CUDA module an mp=28 call compiles is priced by preflight.

    ``gpuwm/core/preflight.py``'s mp=28 row lists six modules.  A module the
    adapter launches that preflight does not name is device memory the VRAM
    estimate never saw.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core import preflight

    del cp
    listed = None
    for attribute in dir(preflight):
        value = getattr(preflight, attribute)
        if isinstance(value, dict) and 28 in value and isinstance(
                value.get(28), tuple):
            listed = set(value[28])
            break
    assert listed == set(
        preflight._MICROPHYSICS_KERNEL_MODULES[28])
    assert listed is not None, "preflight has no mp=28 kernel-module row"
    assert listed == {
        "thompson", "thompson_aerosol_state", "thompson_aerosol_sat",
        "thompson_aerosol_cold", "thompson_aerosol_warm",
        "thompson_aerosol_sed"}, sorted(listed)


def _real_domain_state(cp, *, aerosol_present: bool):
    """A REAL ``DomainState`` for mp=28, filled with a plausible column.

    The oracle harness above deliberately duck-types the state so it can
    invert the fixture's entry column exactly.  That leaves one thing
    unproven: that the adapter's attribute, shape and staggering expectations
    match the state gpuwm actually allocates.  ``DomainState`` is the only
    thing that can answer it, so these two tests use it.
    """
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState

    cfg = RunConfig(nx=4, ny=3, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=10.0, run_seconds=10.0, moist=True, mp_physics=28)
    state = DomainState(cfg)
    nz = cfg.nz
    state.p[...] = cp.asarray(
        np.linspace(95000.0, 30000.0, nz, dtype=np.float32)[:, None, None])
    state.thb[...] = cp.asarray(np.full(nz, 300.0, np.float32))
    state.thp[...] = 0.0
    state.phb[...] = cp.asarray(
        np.linspace(0.0, 9.81 * 10000.0, nz + 1, dtype=np.float32))
    state.php[...] = 0.0
    state.qv[...] = 0.006
    state.qc[...] = 2.0e-4
    state.qr[...] = 1.0e-4
    state.nr[...] = 2.0e5
    state.qi[...] = 1.0e-6
    state.ni[...] = 1.0e4
    state.qs[...] = 2.0e-5
    state.qg[...] = 3.0e-5
    if aerosol_present:
        state.nc[...] = 1.0e8
        state.nwfa[...] = 1.0e8
        state.nifa[...] = 1.0e5
        state.nwfa2d[...] = 1.0e4
    return state, cfg


@requires_gpu
def test_the_adapter_runs_on_a_real_domainstate_and_holds_wrfs_own_bounds():
    """The duck-typed oracle harness cannot prove this; only DomainState can.

    Checks the three things a forecast would notice first: every field
    finite, WRF's own terminal bounds honoured, and the two surface emission
    constants untouched.  ``nwfa2d``/``nifa2d`` are INTENT(IN) to
    ``mp_gt_driver`` -- it reads them at :1310-1327 and never writes them --
    so an adapter that wrote them would corrupt a cross-step constant.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol
    from gpuwm.core.thompson_aerosol_state import (
        AEROSOL_CEILING, NIFA_FLOOR, NT_C_MAX, NWFA_FLOOR)

    state, cfg = _real_domain_state(cp, aerosol_present=True)
    emission = (state.nwfa2d.copy(), state.nifa2d.copy())
    diagnostics = _apply_thompson_aerosol(state, cfg, 10.0)
    cp.cuda.Stream.null.synchronize()

    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
                 "nwfa", "nifa", "effc", "effi", "effs", "thp",
                 "h_diabatic"):
        value = getattr(state, name)
        assert bool(cp.all(cp.isfinite(value))), name
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni"):
        assert float(getattr(state, name).min()) >= 0.0, name

    assert cp.array_equal(state.nwfa2d, emission[0])
    assert cp.array_equal(state.nifa2d, emission[1])

    # :3979-3982 clamp the PER-KILOGRAM aerosol against the per-m3 constants
    # with no density at all -- one of WRF's unit inconsistencies, reproduced
    # literally.  The lowest level is exempt from the ceiling because
    # mp_gt_driver:1310-1327 emits AFTER the clamp, deliberately unclamped.
    for value, floor in ((state.nwfa, NWFA_FLOOR), (state.nifa, NIFA_FLOOR)):
        assert float(value.min()) >= floor * (1.0 - 1.0e-6)
        assert float(value[1:].max()) <= AEROSOL_CEILING * (1.0 + 1.0e-6)

    # :4020 caps the rediagnosed droplet number at Nt_c_max/rho, i.e. the
    # volumetric cap converted; compare against the adapter's own entry
    # density so the assertion is WRF's inequality and not an approximation.
    rho = state.existing_scratch("mp_thompson_aero_entry_density")
    assert float((state.nc * rho).max()) <= NT_C_MAX * (1.0 + 1.0e-5)

    # Radiation-facing micron contract, WRF's mp_gt_driver:1475-1477 bands.
    # The 1e-6 slack is the float32 round trip of the metre band edges through
    # the kernel's final ``* 1.0e6f`` (4.99e-6 -> 4.9899998, 125e-6 ->
    # 125.00001), not a physics allowance.
    edge = 1.0e-6
    for value, low, high in ((state.effc, 2.49, 50.0),
                             (state.effi, 4.99, 125.0),
                             (state.effs, 9.99, 999.0)):
        assert float(value.min()) >= low * (1.0 - edge)
        assert float(value.max()) <= high * (1.0 + edge)
    assert 0.0 <= float(diagnostics.sr.min())
    assert float(diagnostics.sr.max()) <= 1.0


@requires_gpu
def test_thompson_aerosol_init_fill_is_idempotent_and_derives_nwfa2d():
    """``thompson_init``:493-551 on a real state, once per domain.

    Three properties, each of which WRF's own text forces:

    * the two fills are INDEPENDENT decisions (:493 and :530 are separate
      ``MAXVAL`` reductions), so a state with CCN but no IN gets exactly one;
    * ``nwfa2d`` is DERIVED from the filled profile at :510 and is the only
      2-D product -- ``nifa2d`` is not even a ``thompson_init`` dummy
      argument (:424-444) and stays exactly zero, which is WRF's behaviour and
      not an ArWen shortcut;
    * running it again is a NO-OP, because the guard now sees aerosol.  That
      is what makes it safe to call from an init path that may be re-entered,
      and it is the property that would break if it were ever moved into the
      per-step adapter.
    """
    import cupy as cp

    _require_device()
    from gpuwm.core.microphysics_aerosol import thompson_aerosol_init_fill
    from gpuwm.core.thompson_aerosol_state import (
        NA_CCN0, NA_CCN1, NA_IN0, NA_IN1)

    state, cfg = _real_domain_state(cp, aerosol_present=False)
    assert float(state.nwfa.max()) == 0.0 and float(state.nifa.max()) == 0.0
    assert float(state.nwfa2d.max()) == 0.0

    first = thompson_aerosol_init_fill(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert first == {"ccn": True, "in": True}
    # naCCN1 <= nwfa <= naCCN1 + naCCN0 and the IN counterpart (:511, :548).
    assert float(state.nwfa.min()) >= NA_CCN1 * (1.0 - 1.0e-6)
    assert float(state.nwfa.max()) <= (NA_CCN0 + NA_CCN1) * (1.0 + 1.0e-6)
    assert float(state.nifa.min()) >= NA_IN1 * (1.0 - 1.0e-6)
    assert float(state.nifa.max()) <= (NA_IN0 + NA_IN1) * (1.0 + 1.0e-6)
    assert float(state.nwfa2d.min()) > 0.0
    assert float(state.nifa2d.max()) == 0.0, (
        "thompson_init never assigns nifa2d; it is not one of its dummy "
        "arguments (module_mp_thompson.F:424-444)")

    filled = (state.nwfa.copy(), state.nifa.copy(), state.nwfa2d.copy())
    second = thompson_aerosol_init_fill(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert second == {"ccn": False, "in": False}
    assert cp.array_equal(state.nwfa, filled[0])
    assert cp.array_equal(state.nifa, filled[1])
    assert cp.array_equal(state.nwfa2d, filled[2])

    # One fill without the other: the decisions are separate reductions.
    partial, cfg = _real_domain_state(cp, aerosol_present=False)
    partial.nifa[...] = 5.0e5
    assert thompson_aerosol_init_fill(partial, cfg) == {
        "ccn": True, "in": False}
    assert float(partial.nifa.max()) == 5.0e5


@requires_gpu
def test_activation_bin_edge_policy_is_documented_not_absorbed():
    """The tolerance policy is a decision, so it is pinned as one.

    ``activ_ncloud`` (:5178-5253) picks the NEAREST 10 K temperature bin and
    ``idx_d``/``idx_c``/``idx_n`` are INT truncations, so activated ``nc`` is
    a STEP function of state.  Near a bin edge an FP32 GPU port and the
    Fortran reference can select DIFFERENT bins and differ by tens of percent
    in ``nc`` while every mass field agrees to a few ulps.

    The policy is: fixtures are chosen away from bin edges, and no bound in
    ``_END_TO_END_BOUNDS`` exists to accommodate bin-edge disagreement.  This
    test demonstrates the effect exists and is large -- so that nobody later
    "fixes" a bin-edge failure by widening the global tolerance -- and pins
    that the committed fixtures sit away from the edges.
    """
    import cupy as cp

    _require_device()
    _tables_or_skip()
    from gpuwm.core.thompson_aerosol_launch import probe_activ_ncloud
    from gpuwm.core.thompson_aerosol_runtime import load_aerosol_device_tables

    tables = load_aerosol_device_tables()
    tnccn = tables.ccn_activation_table

    # WRF: ta_Na bins are 10 K apart from 243.15 K, chosen by NINT, so the
    # decision boundary sits at the midpoint between two bins.
    shape = (3, 1, 1)
    edge = f32(248.15)
    below = np.nextafter(edge, f32(-np.inf), dtype=np.float32)
    temperature = cp.asarray(
        np.asarray([below, edge, f32(250.0)], f32).reshape(shape))
    w = cp.full(shape, f32(1.0), cp.float32)
    nccn = cp.full(shape, f32(1.0e9), cp.float32)
    activated = cp.asnumpy(
        probe_activ_ncloud(temperature, w, nccn, tnccn)).ravel()
    step = abs(activated[1] - activated[0]) / max(abs(activated[0]), 1e-30)
    print(f"\nactiv_ncloud bin-edge step at 248.15 K: {step:.3e} "
          f"({activated[0]:.6e} -> {activated[1]:.6e}) for a "
          f"{abs(float(edge) - float(below)):.3e} K change")
    assert step > 1.0e-3, (
        "the nearest-bin temperature selector no longer produces a step; "
        "the documented tolerance policy rests on it")

    # And the committed fixtures stay away from those edges.
    for name in _FIXTURES:
        before, _, _ = _oracle_case(name)
        t = _column(before, "temp_k")
        offset = np.abs(((t - f32(243.15)) / f32(10.0)) % f32(1.0) - f32(0.5))
        assert float(offset.min()) > 1.0e-3, (
            f"{name} has a level within 0.01 K of an activ_ncloud bin edge")

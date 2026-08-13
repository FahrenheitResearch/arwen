"""Correlation-coefficient (RhoHV) quality control for radar superobbing.

Dual-pol RhoHV separates meteorological from non-meteorological echo:
rain and snow return 0.97-0.99+ (Kumjian 2013, J. Operational Meteor.;
Ryzhkov & Zrnic 2019), while biota, chaff and ground clutter return
0.3-0.8 and almost never exceed 0.9 (Zrnic & Ryzhkov 1998, BAMS; Park
et al. 2009, WAF).  A bird layer that survives into the superob grid is
assimilated as a storm, and a bird layer in the velocity field corrupts
every wind it touches.  This mask exists to remove that echo before it
becomes an observation.

**The rule is per moment, and the two moments are not symmetric.**
Severe weather legitimately depresses RhoHV: hail cores run 0.85-0.95
and giant hail lower (Balakrishnan & Zrnic 1990; Kumjian 2013), the
melting layer dips to 0.90-0.97 (Giangrande et al. 2008, JAMC), and the
tornadic debris signature falls below 0.8 -- often below 0.7 -- exactly
at the mesocyclone couplet (Ryzhkov et al. 2005, JAM; Bodine et al.
2013, WAF).  All three are *genuine echo*: hail, melting ice and lofted
debris really are up there, and deleting them from the reflectivity
field would be deleting the storm.  Whether their motion is an
observation of the *wind* is a separate question with a separate
answer, so this mask answers it separately.

**Reflectivity is shielded.**  A reflectivity gate is dropped only when
its RhoHV is low **and** its co-located reflectivity is below
``ref_shield_dbz``.  That is operational practice: MRMS drops low-RhoHV
echo only where the compound evidence says non-precipitation (Tang et
al. 2014, WAF 29, 1106-1119), and the WSR-88D ORPG classifies with
fuzzy logic (Park et al. 2009) rather than a bare threshold.  Hail
cores, debris signatures and the bright-band peak all carry
reflectivity above any plausible shield, so they survive in this moment
by construction, and the tests that prove it live in
``tests/test_cc_qc.py``.

**Velocity carries no reflectivity shield.**  A velocity gate whose
RhoHV is low is dropped whatever its reflectivity -- including,
deliberately and explicitly, the velocity at the 45 dBZ core of a
tornadic debris signature.  This is an owner ruling (2026-08-05,
relayed from consulting meteorologists) and it withdraws the
reflectivity shield from this moment entirely.  The reason is physical
rather than procedural: a Doppler velocity is an observation of the
*air* only if the scatterer moves with the air, and debris does not.
Lofted boards, soil and vegetation are large, irregular, non-Rayleigh
scatterers centrifuged outward by the vortex, so their radial velocity
carries a systematic outward bias and an enormous spectrum width --
Ryzhkov et al. (2005) and Bodine et al. (2013) identify the TDS by
exactly that non-meteorological scattering, and the property that makes
the signature diagnostic is the property that makes its velocity wrong.
Assimilated, it would place a confidently mis-specified wind at the one
point in the domain where the analysis matters most.  A missing
observation beats a confidently wrong one, so the debris core is left
to the surrounding meteorological gates instead of being filled in with
debris motion.  The same argument covers hail cores and the melting
layer: their reflectivity is kept because the ice is real, their
velocity is dropped because tumbling, fast-falling and mixed-phase
scatterers are not tracers of the wind.  The cost is counted rather
than hidden -- ``gates_dropped_shielded_z`` records, per sweep and per
moment, exactly how many velocity gates the reflectivity shield would
have kept -- and ``rho_min_velocity`` is the one knob that would
revisit the ruling wholesale without a code change.

**One narrow band is exempted from that rule, by name and by
measurement: the debris-signature fringe.**  Owner ruling 2026-08-12,
closing the seam the 2026-08-05 verification found.  A TDS is admitted
in the literature down to about 30 dBZ (Ryzhkov et al. 2005; Bodine et
al. 2013), so the weak or distant signature -- the one a forecaster
most needs a wind analysis to see -- lives in the 30-35 dBZ band, below
the reflectivity shield.  Under the unshielded velocity rule that whole
band lost its velocity, and with it the rotational couplet of every
tornado too far away or too weak to paint a 35 dBZ debris core.  The
exemption keeps those velocities, and it is deliberately conjunctive:
**low RhoHV alone is never enough.**  A gate is exempted only when all
five of these hold, each with a stated threshold and a receipt counter
for the gates it turns away:

1. the strict rule would drop it (RhoHV below the velocity threshold);
2. RhoHV is at or above ``tds_rho_floor`` (0.50) -- a debris floor.
   Measured TDS RhoHV runs 0.5-0.8; below 0.5 the return is
   receiver-noise or thin biota, and no amount of surrounding rotation
   makes noise into a wind observation;
3. reflectivity is at or above ``tds_ref_min_dbz`` (30.0) -- the
   literature's TDS admission threshold, and above the 25-30 dBZ band
   where roosting-bird echo peaks;
4. reflectivity is *below* ``ref_shield_dbz`` (35.0).  The exemption is
   a band, not a new shield: at and above 35 dBZ the 2026-08-05 ruling
   stands untouched, because the debris core is exactly where the
   centrifuging bias is largest and where the surrounding
   meteorological gates are densest;
5. the gate sits inside the rotation neighbourhood of an accepted
   **velocity-couplet seed** in the same sweep (see below).

Criterion 5 is what separates a debris fringe from a field of birds
that happen to return 32 dBZ.  Debris is lofted *by a vortex*; if there
is no vortex signature in the velocity field beside the gate, whatever
depressed its RhoHV was not a tornado.  ``tds_fringe_exempt`` is on by
default because a correctness remedy that has to be switched on is a
workaround.  Setting it False restores the 2026-08-05 verdict on every
gate -- the couplet scan does not even run -- and the receipt says
which ruling was in force.  The observation arrays are then identical
to a pre-ruling build; the receipt is not, because it gains this
paragraph's counters reading zero, which is the difference between a
mask that did nothing and a mask that was not there.

**The couplet instrument, and what it cannot do.**  Seeds are found on
the sweep's own raw velocity plane, the same signature the dealiasing
work uses to locate couplets: an azimuthally adjacent radial pair, at
the same gate and within ``tds_couplet_max_azimuth_gap_deg`` of each
other, whose velocities have opposite sign, whose magnitudes both reach
``tds_couplet_lobe_ms``, and whose difference reaches
``tds_couplet_delta_v_ms`` -- 30 m/s across one beam pair, the
operational gate-to-gate delta-V band for a tornadic vortex signature.
A lone pair is noise, so a seed only counts inside a cluster of at
least ``tds_couplet_min_seeds`` within a
``tds_couplet_cluster_radius``-gate box, and the exemption region is
that cluster grown by ``tds_rotation_radius_radials`` radials and
``tds_rotation_radius_gates`` gates -- roughly 3 degrees by 3 km on a
super-res cut, the scale over which a debris fringe surrounds its
couplet.

This runs *before* dealiasing, on raw folded velocities, and that is a
real limit rather than an oversight: a Nyquist-crossing straight-line
jet also produces adjacent opposite-signed gates, so the instrument
cannot in principle distinguish a compact couplet from a compact fold
seam.  What bounds the damage is the conjunction -- such a seam must
*also* sit in the 30-35 dBZ band with RhoHV between 0.50 and the
velocity threshold before a single gate is exempted -- and the count:
``gates_exempt_tds_fringe`` is the exact number of velocity gates this
rule kept that the 2026-08-05 ruling would have deleted, per sweep and
per moment, so the exemption is auditable as a number rather than
argued as a policy.

**Split cuts are paired, because the real volumes force it.**  On a
2026 WSR-88D volume (measured, KDMX VCP with SAILS revisits) the
split-cut tilts publish dual-pol moments on the surveillance (CS) half
only: CS cuts carry REF+RHO and no VEL, the adjacent Doppler (CD) cuts
carry REF+VEL and no RHO.  A same-sweep-only mask would therefore never
QC velocity at the lowest tilts -- the tilts where both the biota and
the couplet live -- and would leave the CD half's reflectivity
uncleaned beside the cleaned CS half, so the cell-level maximum kept
the bird echo anyway.  The ORPG's own split-cut recombination pairs
exactly these adjacent cuts (WSR-88D ICD 2620002; the halves are
seconds apart), and this module does the same: a sweep without RHO
borrows its immediate neighbour's, preceding sweep first (in every
WSR-88D VCP the CS half precedes its CD half), guarded by an elevation
tolerance that keeps a distinctly different tilt from qualifying, and
matched radial-by-radial by nearest azimuth.  The ICD's stated
effective azimuthal resolution of the dual-pol variables is 1.0 degree
even on 0.5-degree radials, which is where the azimuth tolerance
default comes from.

**Absent data never fabricates a verdict.**  Dual-pol absence is
legitimate at three granularities -- whole volume (the fleet retrofit
completed mid-2013; earlier archives have no RHO at all), whole sweep
(a cut with no usable RHO source), and gate (RHO censored at the
receiver, or a reflectivity gate beyond RHO's extent).  In every case
the gate passes through exactly as it would have in 2012, and the
absence is counted in provenance.  An absent RhoHV is not a failed
test: the gate's velocity is precisely as trustworthy as it was before
the radar grew a second polarization.  (Contrast the Nyquist mask in
:mod:`gpuwm.obs.superob`, which fails closed -- a velocity without an
alias test is untrustworthy in itself.  A velocity without a RhoHV is
not.)

**Known limits, stated rather than hidden.**  The RhoHV estimator is
biased low at low SNR (Melnikov & Zrnic 2007, JTECH), and the pack
carries no SNR plane, so weak far-range precipitation shows depressed,
noisy CC.  In reflectivity the shield is the mitigation and the
residual risk sits at stratiform bright-band edges near 25-30 dBZ,
below any shield that still catches heavy biota.  In velocity there is
no mitigation and the cost is larger by construction: far-range weak
precipitation with a noisy low CC loses its velocity outright.  That is
the price of the purity ruling, it is paid where the observations are
weakest rather than where they are best, and ``rho_min_velocity`` is
the knob that buys some of it back.  ``rho_floor`` -- an unconditional
drop below a second, lower threshold -- survives as a tunable but is
now effectively a *reflectivity* knob, because velocity already drops
unconditionally below its own threshold; it defaults **off** because
the debris and hail reflectivity it would delete is exactly what the
shield exists to keep.

Two smaller honesties, since a docstring that overstates is worse than
one that omits.  Companion pairing is guarded by an elevation
tolerance, but adjacency alone does not make a different tilt
impossible: a Doppler cut whose preceding neighbour lacks RhoHV can
borrow a *following* cut within the tolerance.  No real WSR-88D VCP
orders its cuts that way -- every pairing on the six measured volumes
is a true CS/CD half -- but the guard is a tolerance, not a proof.  And
the thresholds are compared in float64 against values that arrive as
float32, so a stored CC of exactly ``rho_min`` reads a hair below it
and dies; real Level-II RhoHV is quantized in ~1/300 steps, so the
coincidence is not reachable in practice, but the comparison is not
exact and does not pretend to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

#: Moment tokens as the ``gpuwm-obs.radar-sweeps.v1`` pack spells them.
CORRELATION = "RHO"
_REFLECTIVITY = "REF"
_VELOCITY = "VEL"

#: Sweep-plan reasons, spelled once.  ``colocated``/``companion`` mean
#: the mask ran; the rest say exactly why it could not.
REASON_COLOCATED = "colocated"
REASON_COMPANION = "companion"
REASON_NO_RHO = "no-rho"
REASON_NO_REF = "no-ref"
REASON_NO_TARGET = "no-target"

#: Why a *gate* died, per moment, spelled once and counted disjointly so
#: the reasons sum to the moment's drop total.  ``DROP_LOW_RHO`` is the
#: unshielded velocity rule; ``DROP_LOW_RHO_BELOW_SHIELD`` is the
#: compound reflectivity rule; ``DROP_RHO_FLOOR`` is the optional
#: unconditional floor, counted only where it kills something the
#: moment's own rule would have kept.
DROP_LOW_RHO = "low-rho"
DROP_LOW_RHO_BELOW_SHIELD = "low-rho-below-shield"
DROP_RHO_FLOOR = "rho-floor"

#: Why a candidate for the debris-fringe exemption was *not* exempted,
#: spelled once.  These are disjoint tests applied in order, so a gate
#: appears against the first criterion it fails and the four counters
#: plus ``gates_exempt_tds_fringe`` account for every low-RhoHV velocity
#: gate the strict rule would drop.
TDS_TURNED_AWAY_RHO_BELOW_FLOOR = "rho-below-debris-floor"
TDS_TURNED_AWAY_BELOW_Z_BAND = "below-debris-reflectivity"
TDS_TURNED_AWAY_ABOVE_Z_BAND = "at-or-above-shield"
TDS_TURNED_AWAY_NO_ROTATION = "no-couplet-nearby"


class CcQcParamsError(ValueError):
    """A CC-QC parameter that cannot mean what the mask needs it to mean."""


@dataclass(frozen=True)
class CcQcParams:
    """Every CC-QC tunable, in one hashable place -- it goes into provenance.

    The (``rho_min``, ``ref_shield_dbz``) pair is the science surface of
    this mask and deliberately exposed rather than hardwired: 0.95 is the
    MRMS non-precipitation discriminator (Tang et al. 2014) and 35 dBZ
    sits in the operational 35-40 dBZ shield band -- above nearly all
    biota (roost rings peak near 30-35 dBZ; Zrnic & Ryzhkov 1998) and
    below hail cores, debris signatures and the bright-band peak, which
    are the echoes the shield exists to protect *in reflectivity*.  The
    shield does not reach velocity at all; see ``rho_min_velocity`` and
    the module docstring for the ruling that put it that way.
    """

    #: Drop a *reflectivity* gate whose RhoHV is below this AND whose
    #: reflectivity is below ``ref_shield_dbz`` -- neither condition
    #: drops alone.  Also the default *velocity* threshold, where it
    #: drops alone and always (``rho_min_velocity``).
    rho_min: float = 0.95
    #: Reflectivity at or above this is never CC-dropped **in the
    #: reflectivity moment**.  The shield is what makes the melting
    #: layer, hail cores and the tornadic debris signature survive as
    #: echo; lowering it protects more weather and passes more biota,
    #: raising it the reverse.  A gate whose reflectivity is censored
    #: (NaN) or beyond the REF plane's extent counts as *below* the
    #: shield: the shield protects strong echo, and an absent echo is
    #: not strong echo.  The shield has no effect on velocity.
    ref_shield_dbz: float = 35.0
    #: The velocity threshold, or ``None`` to use ``rho_min``.  This is
    #: the single knob that revisits the 2026-08-05 purity ruling
    #: without a code change: at ``None`` every velocity gate below
    #: ``rho_min`` dies regardless of reflectivity, which costs the
    #: hail-core (0.85-0.95) and melting-layer velocities along with the
    #: debris and the biota.  Set it below those bands -- 0.80, the TDS
    #: criterion of Ryzhkov et al. (2005) and Bodine et al. (2013) --
    #: and hail-core velocity survives while debris and biota still die.
    #: Changing it is a decision about what a wind observation is, and
    #: the default is the ruling, not an accident.
    rho_min_velocity: float | None = None
    #: Optional unconditional floor: RhoHV below this is dropped in both
    #: moments regardless of reflectivity.  ``None`` (the default) means
    #: no floor, and the default is a safety property, not an omission:
    #: the tornadic debris signature runs below 0.8 -- often below 0.7 --
    #: at the couplet itself (Ryzhkov et al. 2005; Bodine et al. 2013),
    #: so any floor at or above those values deletes the debris *echo*,
    #: which the ruling keeps.  Since velocity now drops unconditionally
    #: below its own threshold, this is in practice a reflectivity knob;
    #: it can only bite velocity when ``rho_min_velocity`` is set below
    #: it.  Enabling it is a decision about a specific volume, never a
    #: default.
    rho_floor: float | None = None
    #: Borrow RhoHV from the adjacent split-cut companion when the sweep
    #: itself has none (CS half preceding, CD half following -- see the
    #: module docstring).  Off restricts the mask to sweeps carrying
    #: their own RHO plane, which on modern split-cut VCPs means the
    #: surveillance reflectivity only.
    pair_companion_sweeps: bool = True
    #: An adjacent sweep is only a companion if its elevation agrees to
    #: within this.  Measured CS/CD halves of the same tilt differ by up
    #: to ~0.26 degrees (antenna wobble between revisits); distinct tilts
    #: in the tightest modern VCPs differ by ~0.4.  0.5 accepts every
    #: true pair on the measured volume while adjacency (only the two
    #: neighbouring sweeps are ever candidates) keeps a different tilt
    #: from qualifying.
    companion_elevation_tolerance_deg: float = 0.5
    #: A target radial with no companion radial within this azimuth
    #: distance passes open.  1.0 degree is the ICD's stated effective
    #: azimuthal resolution of the dual-pol variables on super-res cuts.
    companion_azimuth_tolerance_deg: float = 1.0
    #: Keep the velocity of debris-signature *fringe* gates -- the
    #: 30-35 dBZ band, RhoHV above the debris floor, beside a velocity
    #: couplet -- that the unshielded velocity rule would otherwise
    #: delete.  Owner ruling 2026-08-12; on by default because a
    #: correctness remedy behind a flag is a workaround.  False restores
    #: the 2026-08-05 behaviour exactly.
    tds_fringe_exempt: bool = True
    #: The debris floor.  A TDS depresses RhoHV to 0.5-0.8 (Ryzhkov et
    #: al. 2005; Bodine et al. 2013); below this the return is receiver
    #: noise or thin biota and is never exempted however much rotation
    #: surrounds it.  Raising it exempts less and trusts the CC estimate
    #: more; lowering it walks toward "low CC alone", which is the
    #: failure mode the whole conjunction exists to prevent.
    tds_rho_floor: float = 0.50
    #: The debris reflectivity floor: the literature's TDS admission
    #: threshold.  Below it a low-RhoHV gate is biota, chaff or clutter,
    #: not lofted debris, whatever the velocity field is doing.
    tds_ref_min_dbz: float = 30.0
    #: A couplet seed needs this much velocity difference across one
    #: adjacent-radial pair at the same gate.  30 m/s is the operational
    #: gate-to-gate delta-V band for a tornadic vortex signature.
    tds_couplet_delta_v_ms: float = 30.0
    #: ...and both members of the pair must reach this magnitude, so a
    #: one-sided spike beside a zero does not read as rotation.
    tds_couplet_lobe_ms: float = 10.0
    #: Radial pairs further apart than this in azimuth are not adjacent
    #: beams and cannot seed a couplet -- a sector boundary or a data
    #: gap is not shear.
    tds_couplet_max_azimuth_gap_deg: float = 2.0
    #: A seed only counts inside a cluster: at least this many seeds
    #: within the cluster box.  A lone opposite-signed pair is noise.
    tds_couplet_min_seeds: int = 3
    #: Half-width, in radials and gates, of that cluster box.
    tds_couplet_cluster_radius: int = 3
    #: How far the exemption reaches from an accepted seed, in radials.
    #: 6 radials is ~3 degrees on a 0.5-degree super-res cut.
    tds_rotation_radius_radials: int = 6
    #: ...and in gates.  12 gates is 3 km on a 250 m cut -- the scale
    #: over which a debris fringe surrounds its couplet.
    tds_rotation_radius_gates: int = 12

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "CcQcParams":
        def _real(name, value):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise CcQcParamsError(
                    f"{name} is {value!r} ({type(value).__name__}); it must "
                    "be a real number -- a string or flag that happens to "
                    "survive float() is the caller not saying what they "
                    "meant")
            return float(value)

        rho_min = _real("rho_min", self.rho_min)
        if not np.isfinite(rho_min) or not (0.0 < rho_min <= 1.0):
            raise CcQcParamsError(
                f"rho_min is {rho_min!r}; RhoHV is a correlation magnitude "
                "and a usable threshold lies in (0, 1]. At 0 nothing is "
                "ever low; above 1 everything is")
        shield = _real("ref_shield_dbz", self.ref_shield_dbz)
        if not np.isfinite(shield):
            raise CcQcParamsError(
                f"ref_shield_dbz is {shield!r}; it must be finite. A NaN "
                "shield protects nothing (every comparison is False) and "
                "an infinite one either disables the mask or the shield "
                "entirely, silently")
        if self.rho_min_velocity is not None:
            velocity = _real("rho_min_velocity", self.rho_min_velocity)
            if not np.isfinite(velocity) or not (0.0 < velocity <= 1.0):
                raise CcQcParamsError(
                    f"rho_min_velocity is {velocity!r}; it is a RhoHV "
                    "threshold like rho_min and lies in (0, 1], or is None "
                    "to say velocity uses rho_min. At 0 the velocity mask "
                    "silently never fires, which is a decision that has to "
                    "be made by turning cc_qc off, not by a threshold")
        if self.rho_floor is not None:
            floor = _real("rho_floor", self.rho_floor)
            if not np.isfinite(floor) or not (0.0 < floor <= 1.0):
                raise CcQcParamsError(
                    f"rho_floor is {floor!r}; a usable unconditional floor "
                    "lies in (0, 1], or is None for no floor at all")
            if floor > rho_min:
                raise CcQcParamsError(
                    f"rho_floor {floor!r} is above rho_min {rho_min!r}; the "
                    "floor is the unconditional subset of the compound "
                    "test, and a floor above the compound threshold would "
                    "quietly turn the reflectivity shield off for the band "
                    "between them")
        if not isinstance(self.pair_companion_sweeps, bool):
            raise CcQcParamsError(
                f"pair_companion_sweeps is {self.pair_companion_sweeps!r} "
                f"({type(self.pair_companion_sweeps).__name__}); it is a "
                "yes/no decision about borrowing the split-cut "
                "companion's RhoHV and must be a bool")
        for name in ("companion_elevation_tolerance_deg",
                     "companion_azimuth_tolerance_deg"):
            value = _real(name, getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise CcQcParamsError(
                    f"{name} is {value!r}; a matching tolerance must be "
                    "finite and strictly positive -- at or below zero no "
                    "companion can ever match and the pairing silently "
                    "never happens, which is a decision "
                    "pair_companion_sweeps exists to make loudly")
        if not isinstance(self.tds_fringe_exempt, bool):
            raise CcQcParamsError(
                f"tds_fringe_exempt is {self.tds_fringe_exempt!r} "
                f"({type(self.tds_fringe_exempt).__name__}); it is a "
                "yes/no decision about the debris-fringe ruling and must "
                "be a bool")
        floor = _real("tds_rho_floor", self.tds_rho_floor)
        if not np.isfinite(floor) or not (0.0 < floor <= 1.0):
            raise CcQcParamsError(
                f"tds_rho_floor is {floor!r}; it is a RhoHV threshold and "
                "lies in (0, 1]. At 0 the debris floor stops separating "
                "debris from receiver noise, which is the conjunction's "
                "whole purpose")
        velocity_threshold = (rho_min if self.rho_min_velocity is None
                              else float(self.rho_min_velocity))
        if floor >= velocity_threshold:
            raise CcQcParamsError(
                f"tds_rho_floor {floor!r} is at or above the velocity "
                f"threshold {velocity_threshold!r}; the exemption band is "
                "the RhoHV interval between them, and an empty band means "
                "the ruling silently never fires. Turn it off with "
                "tds_fringe_exempt=False if that is what was meant")
        ref_min = _real("tds_ref_min_dbz", self.tds_ref_min_dbz)
        if not np.isfinite(ref_min):
            raise CcQcParamsError(
                f"tds_ref_min_dbz is {ref_min!r}; it must be finite. A NaN "
                "floor admits nothing (every comparison is False) and an "
                "infinite one either admits everything or nothing, "
                "silently")
        if ref_min >= shield:
            raise CcQcParamsError(
                f"tds_ref_min_dbz {ref_min!r} is at or above "
                f"ref_shield_dbz {shield!r}; the exemption is the "
                "reflectivity band between them and an empty band means "
                "the ruling silently never fires")
        for name in ("tds_couplet_delta_v_ms", "tds_couplet_lobe_ms",
                     "tds_couplet_max_azimuth_gap_deg"):
            value = _real(name, getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise CcQcParamsError(
                    f"{name} is {value!r}; a couplet criterion must be "
                    "finite and strictly positive -- at or below zero "
                    "every pair of gates reads as rotation and the "
                    "conjunction loses the criterion that separates a "
                    "debris fringe from a field of birds")
        if (2.0 * float(self.tds_couplet_lobe_ms)
                > float(self.tds_couplet_delta_v_ms)):
            raise CcQcParamsError(
                f"tds_couplet_lobe_ms {self.tds_couplet_lobe_ms!r} is more "
                f"than half tds_couplet_delta_v_ms "
                f"{self.tds_couplet_delta_v_ms!r}; two opposite-signed "
                "lobes of that magnitude already differ by more than the "
                "delta-V criterion, so the delta-V test would never bind "
                "and the seed rule would be the lobe rule alone")
        for name in ("tds_couplet_min_seeds", "tds_couplet_cluster_radius",
                     "tds_rotation_radius_radials",
                     "tds_rotation_radius_gates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CcQcParamsError(
                    f"{name} is {value!r} "
                    f"({type(value).__name__}); it counts radials, gates or "
                    "seeds and must be an int")
            if value < 0:
                raise CcQcParamsError(
                    f"{name} is {value!r}; a count of radials, gates or "
                    "seeds cannot be negative")
        if self.tds_couplet_min_seeds < 1:
            raise CcQcParamsError(
                f"tds_couplet_min_seeds is {self.tds_couplet_min_seeds!r}; "
                "at 0 every gate in the sweep is inside a cluster of zero "
                "seeds and the rotation criterion silently never binds")
        return self

    def to_payload(self) -> dict:
        return {
            "rho_min": float(self.rho_min),
            "ref_shield_dbz": float(self.ref_shield_dbz),
            "rho_min_velocity": (None if self.rho_min_velocity is None
                                 else float(self.rho_min_velocity)),
            "rho_floor": (None if self.rho_floor is None
                          else float(self.rho_floor)),
            "pair_companion_sweeps": bool(self.pair_companion_sweeps),
            "companion_elevation_tolerance_deg": float(
                self.companion_elevation_tolerance_deg),
            "companion_azimuth_tolerance_deg": float(
                self.companion_azimuth_tolerance_deg),
            "tds_fringe_exempt": bool(self.tds_fringe_exempt),
            "tds_rho_floor": float(self.tds_rho_floor),
            "tds_ref_min_dbz": float(self.tds_ref_min_dbz),
            "tds_couplet_delta_v_ms": float(self.tds_couplet_delta_v_ms),
            "tds_couplet_lobe_ms": float(self.tds_couplet_lobe_ms),
            "tds_couplet_max_azimuth_gap_deg": float(
                self.tds_couplet_max_azimuth_gap_deg),
            "tds_couplet_min_seeds": int(self.tds_couplet_min_seeds),
            "tds_couplet_cluster_radius": int(
                self.tds_couplet_cluster_radius),
            "tds_rotation_radius_radials": int(
                self.tds_rotation_radius_radials),
            "tds_rotation_radius_gates": int(self.tds_rotation_radius_gates),
        }

    @property
    def velocity_threshold(self) -> float:
        """The RhoHV below which a velocity gate is dropped."""

        return float(self.rho_min if self.rho_min_velocity is None
                     else self.rho_min_velocity)


def _nearest_azimuth_rows(target_az: np.ndarray, source_az: np.ndarray,
                          tolerance_deg: float) -> np.ndarray:
    """For each target radial, the nearest source radial by azimuth.

    Circular in azimuth (359.8 and 0.1 are 0.3 degrees apart), vectorized,
    and honest about failure: a target radial with no source radial
    within ``tolerance_deg`` maps to ``-1``, which the masker treats as
    "no RhoHV here" -- pass open, counted.
    """

    target = np.asarray(target_az, dtype=np.float64) % 360.0
    source = np.asarray(source_az, dtype=np.float64) % 360.0
    order = np.argsort(source, kind="stable")
    sorted_az = source[order]
    # Wrap one radial around each end so the seam at 0/360 has neighbours.
    ext_az = np.concatenate((sorted_az[-1:] - 360.0, sorted_az,
                             sorted_az[:1] + 360.0))
    ext_rows = np.concatenate((order[-1:], order, order[:1]))
    pos = np.searchsorted(ext_az, target)
    left = np.clip(pos - 1, 0, ext_az.size - 1)
    right = np.clip(pos, 0, ext_az.size - 1)
    take_right = (np.abs(ext_az[right] - target)
                  < np.abs(target - ext_az[left]))
    chosen = np.where(take_right, right, left)
    rows = ext_rows[chosen].astype(np.intp)
    rows[np.abs(ext_az[chosen] - target) > float(tolerance_deg)] = -1
    return rows


def _box_count(mask: np.ndarray, half_rows: int,
               half_cols: int) -> np.ndarray:
    """Count of True in a (2h+1) x (2w+1) box centred on each cell.

    Radials wrap (a sweep is a circle, and a couplet may straddle north);
    gates do not (range 0 has no neighbour below it).  Integral-image, so
    the cost is one pass whatever the box size.
    """

    rows, cols = mask.shape
    if rows == 0 or cols == 0:
        return np.zeros(mask.shape, dtype=np.int64)
    block = mask.astype(np.int64)
    if half_rows > 0:
        # Wrap by index, so a box taller than the sweep still works.
        block = block[np.arange(-half_rows, rows + half_rows) % rows]
    if half_cols > 0:
        pad = np.zeros((block.shape[0], half_cols), dtype=np.int64)
        block = np.concatenate((pad, block, pad), axis=1)
    integral = np.zeros((block.shape[0] + 1, block.shape[1] + 1),
                        dtype=np.int64)
    integral[1:, 1:] = block.cumsum(axis=0).cumsum(axis=1)
    height = 2 * half_rows + 1
    width = 2 * half_cols + 1
    return (integral[height:height + rows, width:width + cols]
            - integral[0:rows, width:width + cols]
            - integral[height:height + rows, 0:cols]
            + integral[0:rows, 0:cols])


def couplet_seed_mask(velocity: np.ndarray, azimuth_deg: np.ndarray,
                      params: CcQcParams) -> np.ndarray:
    """``(radials, gates)`` bool: gates that seed a velocity couplet.

    A seed is one azimuthally adjacent radial pair, at the same gate,
    whose two velocities have opposite sign, each reach
    ``tds_couplet_lobe_ms``, and differ by at least
    ``tds_couplet_delta_v_ms``; both members of the pair are marked.
    Radials further apart than ``tds_couplet_max_azimuth_gap_deg`` are
    not adjacent beams and cannot seed, so a sector boundary or a data
    gap never reads as shear.  The sweep is treated as a circle, so a
    couplet straddling north is found like any other.

    This runs on raw, possibly folded velocities -- CC QC is upstream of
    every alias defence on purpose -- so a Nyquist-crossing straight-line
    shear seam produces seeds too.  It is a *couplet proximity* test, not
    a tornado detection, and the module docstring says what bounds the
    consequence.
    """

    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return np.zeros(values.shape, dtype=bool)
    azimuth = np.asarray(azimuth_deg, dtype=np.float64)
    nearer = np.roll(values, -1, axis=0)
    gap = (np.roll(azimuth, -1) - azimuth) % 360.0
    adjacent = (gap > 0.0) & (gap <= params.tds_couplet_max_azimuth_gap_deg)
    lobe = float(params.tds_couplet_lobe_ms)
    pair = (np.isfinite(values) & np.isfinite(nearer)
            & adjacent[:, None]
            & (values * nearer < 0.0)
            & (np.abs(values) >= lobe) & (np.abs(nearer) >= lobe)
            & (np.abs(values - nearer)
               >= float(params.tds_couplet_delta_v_ms)))
    seeds = pair | np.roll(pair, 1, axis=0)
    return seeds


def rotation_association_mask(velocity: np.ndarray, azimuth_deg: np.ndarray,
                              params: CcQcParams) -> np.ndarray:
    """``(radials, gates)`` bool: the neighbourhood of a *clustered* couplet.

    Seeds alone are not rotation -- one opposite-signed pair is two noisy
    gates.  A seed counts only where at least ``tds_couplet_min_seeds``
    seeds share a ``tds_couplet_cluster_radius`` box, and the region
    returned is those surviving seeds grown by
    ``tds_rotation_radius_radials`` radials and
    ``tds_rotation_radius_gates`` gates.
    """

    seeds = couplet_seed_mask(velocity, azimuth_deg, params)
    if not seeds.any():
        return seeds
    radius = int(params.tds_couplet_cluster_radius)
    clustered = seeds & (_box_count(seeds, radius, radius)
                         >= int(params.tds_couplet_min_seeds))
    if not clustered.any():
        return clustered
    return _box_count(clustered,
                      int(params.tds_rotation_radius_radials),
                      int(params.tds_rotation_radius_gates)) > 0


class CcSweepMasker:
    """The CC drop mask for one sweep, whatever moment is being read.

    Built once per sweep from the RhoHV source (the sweep's own RHO
    plane, or the paired companion's) and the sweep's own REF plane (the
    shield is always judged against co-located reflectivity from the
    *target* sweep -- borrowing Z across cuts would judge the shield
    against a different pulse train than the gate being dropped).
    ``drop_mask`` is then evaluated against each target moment's actual
    gate ranges, because REF and VEL do not owe each other a gate count:
    matching is by slant range, never by index.  The rule it applies is
    per moment -- shielded for reflectivity, unshielded for velocity --
    and the module docstring holds the ruling that made it so.
    """

    def __init__(self, *, rho, rho_rows, ref, params: CcQcParams,
                 reason: str, companion_sweep_index: int | None = None,
                 velocity=None, azimuth_deg=None):
        self._rho = rho              # Moment ("RHO" plane, source sweep)
        self._rows = rho_rows        # target radial -> source radial, -1 open
        self._ref = ref              # Moment ("REF" plane, target sweep)
        self._velocity = velocity    # Moment ("VEL" plane, target sweep)
        self._azimuth_deg = azimuth_deg
        self._rotation_plane: np.ndarray | None = None
        self._params = params
        self.reason = reason
        self.companion_sweep_index = companion_sweep_index
        self.gates_tested: dict[str, int] = {}
        self.gates_rho_missing: dict[str, int] = {}
        self.gates_dropped: dict[str, int] = {}
        #: ``{product: {reason: gates}}``, disjoint, summing to
        #: ``gates_dropped[product]``.
        self.gates_dropped_reason: dict[str, dict[str, int]] = {}
        #: ``{product: gates}`` -- dropped gates whose co-located
        #: reflectivity was at or above the shield, i.e. the gates the
        #: reflectivity rule would have kept.  For velocity this is the
        #: measured price of the purity ruling; for reflectivity it can
        #: only be non-zero through ``rho_floor``.
        self.gates_dropped_shielded_z: dict[str, int] = {}
        #: ``{product: gates}`` -- gates the debris-fringe ruling kept
        #: that the strict velocity rule would have deleted.  This is
        #: the exemption's whole footprint, and it is the number to
        #: argue about if the ruling is ever revisited.
        self.gates_exempt_tds_fringe: dict[str, int] = {}
        #: ``{product: {criterion: gates}}`` -- of the gates the strict
        #: rule would drop, which conjunct turned each one away.  The
        #: four criteria are disjoint and, with
        #: ``gates_exempt_tds_fringe``, account for every such gate.
        self.gates_tds_turned_away: dict[str, dict[str, int]] = {}
        #: Couplet seeds this sweep's velocity plane produced, after
        #: clustering.  Zero means the exemption could not fire here
        #: whatever the dual-pol moments said.
        self.couplet_seed_gates: int = 0
        #: Per product, the masks the counting above needs, kept from
        #: the one ``drop_mask`` call that moment gets per sweep.
        self._reasons: dict[str, dict[str, np.ndarray]] = {}
        self._shielded: dict[str, np.ndarray] = {}
        self._exempt: dict[str, np.ndarray] = {}
        self._turned_away: dict[str, dict[str, np.ndarray]] = {}

    def _sample(self, moment, ranges: np.ndarray, rows: np.ndarray,
                fill: float) -> np.ndarray:
        """Moment values at (rows, nearest gate by slant range), else fill."""

        index = np.rint((np.asarray(ranges, dtype=np.float64)
                         - moment.first_gate_range_m)
                        / moment.gate_size_m).astype(np.intp)
        in_gates = (index >= 0) & (index < moment.gate_count)
        data = np.asarray(moment.data, dtype=np.float64)
        values = data[np.clip(rows, 0, data.shape[0] - 1)][
            :, np.clip(index, 0, moment.gate_count - 1)]
        values[rows < 0, :] = fill
        values[:, ~in_gates] = fill
        return values

    def _rotation_at(self, ranges: np.ndarray) -> np.ndarray:
        """The rotation-association mask, sampled at ``ranges``.

        Computed once per sweep, on the sweep's *own* velocity plane and
        its own radials: a couplet is a property of this cut's wind
        field, and borrowing one across cuts the way RhoHV is borrowed
        would associate a fringe with rotation the beam never saw.
        """

        moment = self._velocity
        if moment is None or self._azimuth_deg is None:
            return np.zeros((self._ref.data.shape[0], len(ranges)),
                            dtype=bool)
        if self._rotation_plane is None:
            seeds = couplet_seed_mask(moment.data, self._azimuth_deg,
                                      self._params)
            radius = int(self._params.tds_couplet_cluster_radius)
            clustered = seeds & (
                _box_count(seeds, radius, radius)
                >= int(self._params.tds_couplet_min_seeds))
            self.couplet_seed_gates = int(clustered.sum())
            self._rotation_plane = (
                clustered if not clustered.any() else
                _box_count(clustered,
                           int(self._params.tds_rotation_radius_radials),
                           int(self._params.tds_rotation_radius_gates)) > 0)
        plane = self._rotation_plane
        index = np.rint((np.asarray(ranges, dtype=np.float64)
                         - moment.first_gate_range_m)
                        / moment.gate_size_m).astype(np.intp)
        in_gates = (index >= 0) & (index < moment.gate_count)
        rows = min(plane.shape[0], self._ref.data.shape[0])
        sampled = np.zeros((self._ref.data.shape[0], len(ranges)), dtype=bool)
        sampled[:rows] = plane[:rows][
            :, np.clip(index, 0, moment.gate_count - 1)]
        sampled[:, ~in_gates] = False
        return sampled

    def drop_mask(self, product: str, ranges: np.ndarray) -> np.ndarray:
        """``(radials, len(ranges))`` bool: True where the gate is dropped.

        Per moment.  Reflectivity is compound -- low RhoHV AND
        reflectivity below the shield -- so genuine echo with a
        depressed RhoHV (hail, the bright band, debris) stays in the
        reflectivity field.  Velocity is unshielded: low RhoHV drops it
        whatever the reflectivity, because a scatterer that is not a
        tracer of the air does not carry a wind observation however
        brightly it returns.  A gate with no co-located finite RhoHV is
        never dropped in either moment (pass open, counted); a
        reflectivity gate with no co-located reflectivity is
        shield-eligible (an absent echo is not a hail core).
        """

        params = self._params
        rho = self._sample(self._rho, ranges, self._rows, np.nan)
        identity = np.arange(self._ref.data.shape[0], dtype=np.intp)
        z = self._sample(self._ref, ranges, identity, np.nan)
        has_rho = np.isfinite(rho)
        shielded = z >= params.ref_shield_dbz           # NaN -> False
        if product == _VELOCITY:
            primary = has_rho & (rho < params.velocity_threshold)
            primary_reason = DROP_LOW_RHO
            if params.tds_fringe_exempt:
                primary = self._apply_fringe_exemption(
                    product, primary, rho, z, ranges)
        else:
            primary = has_rho & (rho < params.rho_min) & ~shielded
            primary_reason = DROP_LOW_RHO_BELOW_SHIELD
        reasons = {primary_reason: primary}
        drop = primary
        if params.rho_floor is not None:
            floor_only = has_rho & (rho < params.rho_floor) & ~primary
            reasons[DROP_RHO_FLOOR] = floor_only
            drop = primary | floor_only
        self._reasons[product] = reasons
        self._shielded[product] = shielded
        self.gates_tested[product] = (self.gates_tested.get(product, 0)
                                      + int(has_rho.sum()))
        self.gates_rho_missing[product] = (
            self.gates_rho_missing.get(product, 0)
            + int((~has_rho).sum()))
        return drop

    def _apply_fringe_exemption(self, product: str, would_drop: np.ndarray,
                                rho: np.ndarray, z: np.ndarray,
                                ranges: np.ndarray) -> np.ndarray:
        """Remove debris-fringe gates from ``would_drop``, and say why not.

        The five conjuncts are applied in order and the gates each one
        turns away are kept as disjoint masks, so the receipt can say
        which criterion decided each gate rather than only how many
        survived.  Returns the surviving drop mask.
        """

        params = self._params
        above_floor = would_drop & (rho >= params.tds_rho_floor)
        in_band = above_floor & (z >= params.tds_ref_min_dbz)
        fringe = in_band & (z < params.ref_shield_dbz)
        rotation = self._rotation_at(ranges) if fringe.any() else np.zeros(
            fringe.shape, dtype=bool)
        exempt = fringe & rotation
        self._exempt[product] = exempt
        self._turned_away[product] = {
            TDS_TURNED_AWAY_RHO_BELOW_FLOOR: would_drop & ~above_floor,
            TDS_TURNED_AWAY_BELOW_Z_BAND: above_floor & ~in_band,
            TDS_TURNED_AWAY_ABOVE_Z_BAND: in_band & ~fringe,
            TDS_TURNED_AWAY_NO_ROTATION: fringe & ~rotation,
        }
        return would_drop & ~exempt

    def count_block(self, product: str, finite: np.ndarray,
                    start: int) -> int:
        """Account for one radial block's fringe exemptions and refusals.

        Called for every block, dropped gates or not, because an
        exemption is invisible in the drop counts by construction: the
        gate it saved is the gate that is no longer there to count.
        Only finite gates are counted -- exempting a hole saves nothing.
        Returns the number of gates exempted in this block.
        """

        stop = start + int(finite.shape[0])
        turned_away = self._turned_away.get(product)
        if turned_away is not None:
            per_criterion = self.gates_tds_turned_away.setdefault(product, {})
            for criterion, mask in turned_away.items():
                per_criterion[criterion] = (
                    per_criterion.get(criterion, 0)
                    + int((finite & mask[start:stop]).sum()))
        exempt = self._exempt.get(product)
        if exempt is None:
            return 0
        saved = int((finite & exempt[start:stop]).sum())
        self.gates_exempt_tds_fringe[product] = (
            self.gates_exempt_tds_fringe.get(product, 0) + saved)
        return saved

    def count_dropped(self, product: str, hits: np.ndarray,
                      start: int) -> int:
        """Account for one radial block's actual drops; returns the count.

        ``hits`` is the block's dropped-AND-finite mask, so a gate that
        was already a hole is never counted as a rejection, and the
        reason split is taken against the same mask rather than against
        the rule -- what is counted is what was removed.
        """

        stop = start + int(hits.shape[0])
        dropped = int(hits.sum())
        self.gates_dropped[product] = (self.gates_dropped.get(product, 0)
                                       + dropped)
        per_reason = self.gates_dropped_reason.setdefault(product, {})
        for reason, mask in self._reasons.get(product, {}).items():
            per_reason[reason] = (per_reason.get(reason, 0)
                                  + int((hits & mask[start:stop]).sum()))
        shielded = self._shielded.get(product)
        if shielded is not None:
            self.gates_dropped_shielded_z[product] = (
                self.gates_dropped_shielded_z.get(product, 0)
                + int((hits & shielded[start:stop]).sum()))
        return dropped


@dataclass
class CcSweepPlan:
    """What CC QC will do for one sweep, and why."""

    reason: str
    masker: CcSweepMasker | None = None

    @property
    def applicable(self) -> bool:
        return self.masker is not None

    def record(self, sweep) -> dict:
        """The per-sweep provenance entry, applied or not."""

        entry = {
            "sweep_index": int(sweep.sweep_index),
            "elevation_angle_deg": float(sweep.elevation_angle_deg),
            "applied": self.applicable,
            "reason": self.reason,
        }
        if self.masker is not None:
            entry["rho_companion_sweep_index"] = (
                self.masker.companion_sweep_index)
            entry["gates_tested"] = dict(self.masker.gates_tested)
            entry["gates_rho_missing"] = dict(
                self.masker.gates_rho_missing)
            entry["gates_dropped"] = dict(self.masker.gates_dropped)
            # Per moment, why: the reasons are disjoint and sum to that
            # moment's gates_dropped, and gates_dropped_shielded_z says
            # how many of them the reflectivity shield would have kept.
            entry["gates_dropped_reason"] = {
                product: dict(reasons) for product, reasons
                in self.masker.gates_dropped_reason.items()}
            entry["gates_dropped_shielded_z"] = dict(
                self.masker.gates_dropped_shielded_z)
            # The debris-fringe ruling, per sweep: how many velocity
            # gates it kept, how many couplet seeds made that possible,
            # and which conjunct turned every other candidate away.
            entry["gates_exempt_tds_fringe"] = dict(
                self.masker.gates_exempt_tds_fringe)
            entry["gates_tds_turned_away"] = {
                product: dict(criteria) for product, criteria
                in self.masker.gates_tds_turned_away.items()}
            entry["couplet_seed_gates"] = int(
                self.masker.couplet_seed_gates)
        return entry


def build_cc_plans(sweeps, params: CcQcParams) -> list[CcSweepPlan]:
    """One :class:`CcSweepPlan` per sweep, in sweep order.

    Planned for the whole volume at once because companion pairing needs
    the neighbours: a sweep the superobber later skips (elevation) can
    still lend its RHO plane to the cut beside it.
    """

    params.validate()
    plans: list[CcSweepPlan] = []
    for position, sweep in enumerate(sweeps):
        if (_REFLECTIVITY not in sweep.moments
                and _VELOCITY not in sweep.moments):
            plans.append(CcSweepPlan(reason=REASON_NO_TARGET))
            continue
        ref = sweep.moments.get(_REFLECTIVITY)
        if ref is None:
            # The compound rule is the safety property; without co-located
            # reflectivity there is no shield, and CC-alone is the failure
            # mode this module exists to prevent.  Pass open, counted.
            plans.append(CcSweepPlan(reason=REASON_NO_REF))
            continue
        # The couplet instrument reads this sweep's own velocity plane:
        # rotation is a property of the cut being masked, never borrowed.
        velocity = sweep.moments.get(_VELOCITY)
        rho = sweep.moments.get(CORRELATION)
        if rho is not None:
            rows = np.arange(rho.data.shape[0], dtype=np.intp)
            plans.append(CcSweepPlan(
                reason=REASON_COLOCATED,
                masker=CcSweepMasker(rho=rho, rho_rows=rows, ref=ref,
                                     params=params,
                                     reason=REASON_COLOCATED,
                                     velocity=velocity,
                                     azimuth_deg=sweep.azimuth_deg)))
            continue
        companion = None
        if params.pair_companion_sweeps:
            # Preceding neighbour first: in every WSR-88D VCP the CS half
            # (which carries the dual-pol moments) precedes its CD half.
            for offset in (-1, 1):
                neighbour_pos = position + offset
                if not (0 <= neighbour_pos < len(sweeps)):
                    continue
                neighbour = sweeps[neighbour_pos]
                near = neighbour.moments.get(CORRELATION)
                if near is None:
                    continue
                apart = abs(float(neighbour.elevation_angle_deg)
                            - float(sweep.elevation_angle_deg))
                if apart <= params.companion_elevation_tolerance_deg:
                    companion = (neighbour, near)
                    break
        if companion is None:
            plans.append(CcSweepPlan(reason=REASON_NO_RHO))
            continue
        neighbour, near = companion
        rows = _nearest_azimuth_rows(
            sweep.azimuth_deg, neighbour.azimuth_deg,
            params.companion_azimuth_tolerance_deg)
        plans.append(CcSweepPlan(
            reason=REASON_COMPANION,
            masker=CcSweepMasker(
                rho=near, rho_rows=rows, ref=ref, params=params,
                reason=REASON_COMPANION,
                companion_sweep_index=int(neighbour.sweep_index),
                velocity=velocity, azimuth_deg=sweep.azimuth_deg)))
    return plans

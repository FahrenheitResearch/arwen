# WDM6 (mp_physics=16): known deltas before the oracle campaign

Read this before quoting any ULP or relative-agreement number for mp=16.

**Status of the scheme:** implemented-unverified. No oracle comparison against
`phys/module_mp_wdm6.F` has been run. This file is not evidence that one was;
it is the list of places where a comparison is expected to disagree, or was
suspected to and does not, written down before the campaign so nobody spends a
night rediscovering them.

The physics registry's `wdm6-mp16` warnings cite this file.

---

## 1. `_rgmma` runs in float64 where the Fortran runs in REAL(4) — REAL, unbounded a priori

`gpuwm/core/wsm6_constants.py::_rgmma` is WRF's own truncated Weierstrass
product, ported verbatim, and WDM6's coefficient block reuses it
(`gpuwm/core/wdm6_constants.py`). Getting the *truncation* right is the thing
that matters and the port has it: `wdm6init`'s inline comment at
`module_mp_wdm6.F:2130` claims `g4pbr = 17.837825` (the true `Gamma(4.8)`),
while the code's own 10000-term product returns 17.8173. Porting to the comment
would have been a 0.1-0.7% error in every fall speed. The port reproduces the
code.

What is NOT reproduced is the arithmetic width. The Fortran accumulates the
product in default `REAL` (single precision); gpuwm accumulates in Python
float64 and only then rounds to FP32. With 10^4 sequential multiply-accumulates
the accumulated single-precision rounding can reach the 5th-6th significant
digit, so every derived coefficient can differ in its last FP32 digits — and
these coefficients multiply *every* fall speed and collection rate.

Consequence for the campaign: a per-column ULP comparison against a
`gfortran -O0` oracle will show a nonzero floor that is NOT a transcription
error. Establish that floor first, by driving the oracle's own `wdm6init` and
diffing its coefficients against `gpuwm/core/wdm6_constants.py`, before
attributing any process-rate difference to the process.

This is inherited from WSM6, which shares `_rgmma`. It is deliberate: float64 is
the defensible arithmetic, and ArWen's standing rule is not to be bit-exact to a
rounding artifact.

## 2. The PLM remap's `kt` clamp — REAL divergence, believed unreachable

`gpuwm/core/kernels/wdm6.cu` clamps the remap top index (`kt = (kt > 0) ? kt - 1
: 0`) where `nislfv_rain_plmr:2629` and `nislfv_rain_plm6:2891` decrement
unconditionally. On the one input where they differ, the Fortran drops a
layer's mass and the port conserves it. The port takes the defined behaviour, as
the standing rule requires.

Believed unreachable for `nz >= 2`, because the `con1 = 0.05` limiter forces
`dza >= 0.95*dz`, and `nz = 1` is below the adapter's floor of 2. "Believed" is
an argument, not a measurement. If a column oracle ever shows a single-layer
mass difference in sedimentation with everything else matching, this is it. The
full reasoning is in a comment at the site.

## 3. The rain-slope density inconsistency — REAL, and it is WRF's

`wdm62D` consumes `ncr` as a VOLUMETRIC number (`:2251`) while `refl10cm_wdm6`
forms `nr = nr1d*rho` against `rr = qr1d*rho`, so the densities cancel and
`nr1d` enters as a PER-MASS number (`:3005`). The two rain slope definitions
differ by a factor `rho` inside a cube root, about 3% in `lamr` near the
surface. WRF does this; gpuwm reproduces both as written. An oracle will
therefore agree — this is listed so nobody "fixes" it into a disagreement.

## 4. The -35 dBZ floor is NOT a delta — checked, and the suspicion is wrong

Recorded because it looks like one on a partial read, and a reviewer raised it.

`refl10cm_wdm6` initialises `dBZ(k) = -35.0` and then, in its final loop
(`:3126-3128`), overwrites every level unconditionally with
`10.*log10((ze_rain+ze_snow+ze_graupel)*1.d18)`. With the `1.e-22` floor on each
of the three `ze` accumulators, a hydrometeor-free column leaves the routine at
`10*log10(3e-22*1e18) = -35.23` dBZ, not -35.0. So the ROUTINE's floor really is
-35.23.

But the routine is not the boundary. The caller stores
`refl_10cm(i,k,j) = max(-35., dBZ(k))` (`:294`), and gpuwm applies exactly that
clamp inside its kernel (`fmaxf(-35.0f, ...)`). The published field is identical
on both sides, including in hydrometeor-free columns. There is no 0.23 dB
inheritance.

Compare against `refl_10cm`, the field WRF publishes — not against `dBZ`, the
routine's intermediate — and this reads as agreement.

---

## What an oracle campaign for this scheme still has to build

A `tools/wdm6_wrf461_oracle` harness driving the byte-frozen Fortran at
`gfortran -O0` over a column fixture set, on the Shin-Hong / Grell-Freitas
pattern. Both `hail_opt` arms, and both `xland` arms — WDM6 is the first gpuwm
microphysics whose PROCESS RATES read a surface field
(`module_mp_wdm6.F:607-614`), so a land/sea mask that differs between the two
sides is a trajectory difference with no other symptom.

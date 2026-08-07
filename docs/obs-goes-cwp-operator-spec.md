# Spec: the CWP observation operator (one page)

**Status: DESIGN ONLY**, paired with `obs-goes-cwp-bridge-design.md`.

## The observable

Satellite CWP is a column integral. The model equivalent is the same
integral over the hydrometeor mixing ratios the retrieval can see:

```
H(x) = sum_k  rho_d(k) * q_cond(k) * dz(k)      [kg/m^2 -> g/m^2]
```

with `rho_d` dry-air density and `dz` the layer thickness from the
model's own geopotential — both already on the DA lanes' state, no new
diagnostics required.

## Phase-aware composition of `q_cond`

The retrieval's phase is cloud-TOP phase, so the operator does not get
to integrate a phase-matched column naively. v1 rule, deliberately
simple and symmetric with what WoFS does:

* obs phase liquid/supercooled: `q_cond = qc` (cloud liquid; `qr` is
  excluded — rain is not what DCOMP sees).
* obs phase ice/mixed: `q_cond = qc + qi` (an ice-topped deep column
  hides liquid below its top; comparing model ice alone against a
  retrieval that integrated the whole optical column biases H(x) low).
  `qs`/`qg` excluded in v1: large precipitating ice contributes little
  to tau per unit mass. Revisit with the scorecard, not by argument.
* clear-sky zero (CWP_obs = 0): `q_cond = qc + qi` — the zero must be
  allowed to remove any cloud the model invented, whatever its phase.

Vertical bounds: full column in v1. ACHA height / CTP pressure ship in
the pack for a later bounded-integral v2 (integrate only below the
retrieved top), which is where the cloud-top products earn their seat.

## Per-type observation error

Assigned at superob time, per observation, recorded in `cwp_err`:

| obs type | error model |
| --- | --- |
| clear-sky zero | small constant floor (removing false cloud is the highest-confidence statement the retrieval makes) |
| liquid CWP | max(rel_liq * CWP, floor_liq) |
| ice / mixed CWP | max(rel_ice * CWP, floor_ice), rel_ice > rel_liq (PROVISIONAL ice coefficient upstream; the error model must not pretend otherwise) |
| DCOMP thin-cloud (512) / thick-cloud (256) bits set | inflate by a stated factor instead of gating — the pack carries the gated inputs and DQF policy, so the superob layer can do this without refetching |

The thin/thick row is not in tension with the upstream DQF gate: bits 256
and 512 are outside the condemn mask, so those pixels arrive as
observations and the operator inflates them. What the gate condemns —
missing/fill DQF, snow/sea-ice (8), twilight (16), glint (64), ~8% of
retrievals — never reaches this table at all, by Drew's 2026-08-05
ruling ("no point contam it with the 8% that might hurt it"). Moving any
of those causes from gating to inflation is a future A/B, not the v1
behaviour.

Numbers (rel_*, floors, inflation) are tuning constants set at build
time from the WoFS CWP literature and then judged by the obs-skill
scoreboard (MRMS/ASOS twin-instrument referee), per the 2026-08-03
model-quality ruling — not frozen in this spec.

## Fail-closed rules

No observation without: a finite CWP or a phase-confirmed clear zero;
its look geometry (pixel lat/lon from the pack navigation); its error;
and grid-identity agreement with the target. Mixed clear/cloud target
cells are masked and counted, never averaged into a compromise value.
Every drop is a count in the output, never silence.

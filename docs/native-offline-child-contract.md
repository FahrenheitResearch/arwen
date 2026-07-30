# Native offline-child/downscaling contract

This is gpuwm's CUDA-native replacement for the scientific role of WRF
`ndown.exe`. It does not invoke WPS, `real.exe`, `wrf.exe`, or `ndown.exe`.
It is intentionally separate from RW-WPS: RW-WPS starts a forecast from a
large-scale analysis/forecast product, while offline-child starts a new,
standalone child forecast from an archived completed parent simulation.

## Required parent evidence

The preparation transaction must fail before allocating a child unless all of
the following are true:

1. At least two unique, strictly increasing parent states exist.
2. Every state has identical horizontal/vertical geometry, projection, static
   terrain identity, and transported-field inventory.
3. Every state contains `T/U/V/W/PH/PHB/MU/MUB`, dry pressure, water vapor,
   model-top pressure, and the complete active parent microphysics inventory.
4. A companion setup record supplies any base-state, map-factor, static land,
   or surface state omitted from history output. No field is guessed from a
   filename or silently zero-filled.
5. The caller declares a maximum acceptable lateral-boundary interval, and no
   parent interval exceeds it. Hourly d03 history is mechanically readable,
   but it is not accepted for a production 500-m child under a 15-minute
   forcing contract. Five-minute parent archives are preferred for convective
   reruns; 15 minutes is the outer planned default for a 500-m child.
6. The requested child footprint, including the interpolation stencil and
   specified/relaxation frame, is fully covered by the parent.

The current metadata gate lives in `gpuwm.offline_child`. It accepts both
gpuwm and stock-WRF history NetCDF, verifies the invariant grid using a SHA-256
geometry fingerprint, reports an advisory microphysics inventory, and fails
on sparse or irregular forcing. Inventory inference is not authoritative:
the source scheme must be bound from the parent setup/restart/namelist because
WRF history can retain dormant scheme variables.

`bind_parent_physics_from_gpuwm_restart` verifies the restart's resolved
physics fingerprint and binds its domain/config identity. The stock-WRF path
uses `bind_parent_physics_from_wrf_namelist`, binding the chosen domain column
and exact namelist bytes. Production cold-start/LBC builders reject a bare
scheme integer; that lower-level argument remains only for isolated conversion
and interpolation tests.

## Native preparation result

The completed in-memory executable slice currently produces:

- a standalone child initial state using SINT-inherited parent terrain and
  base fields;
- time-varying child lateral value/tendency tables in gpuwm's existing coupled
  WRF units;
- per-frame conversion receipts binding source files, valid times, grid
  placement, interpolation policy, source/target microphysics, and moment
  diagnosis.

The high-resolution static-geography join and compact checksummed on-disk
boundary container are the next publication steps. Until those land, the
receipt explicitly says `sint-parent-inherited`; it does not imply that fine
terrain or land state was synthesized.

Initial and boundary interpolation reuse gpuwm's landed `register_nest`,
`sint`, `bdy_interp1`, and specified-boundary CUDA paths. The child executes as
a standalone specified domain; no live parent model is allocated or stepped.

## Physics-change rules

Changing physics between parent and child is supported only through an
explicit conversion policy. A change starts a new physics trajectory and has
a documented spin-up period; it is not a continuation bit-identical to an
online same-physics nest.

The child clock starts a new standalone forecast. Parent `h_diabatic`, held
radiation/cumulus/PBL tendencies, physics scheduler deadlines, and radiation
averages are not recoverable from ordinary history and are not inherited.
They start through the normal child cold-start policy. A gpuwm restart/setup
companion can supply stronger continuity evidence in a later mode, but an
ordinary history-driven physics change must be interpreted with this spin-up.

For the requested Thompson/Morrison to unified NSSL mp18 path:

- `qv/qc/qr/qi/qs` copy directly.
- Thompson graupel maps to NSSL graupel; NSSL hail starts absent.
- Morrison `qg` maps according to the bound `morr_rimed_ice` receipt:
  graupel mode maps to NSSL graupel and hail mode maps to NSSL hail. An absent
  receipt is an error.
- Existing compatible rain/ice number moments may be carried, with their
  source distribution semantics recorded.
- Any active category lacking its target number moment must run the official
  NSSL `calcnfromq` initializer. Active mass plus a fabricated zero number is
  forbidden.
- NSSL aerosol number starts from its WRF homogeneous background
  (`0.5e9/1.225 # kg-1`) minus newly diagnosed droplet number.
- Initial graupel/hail volume moments use the official initialization
  densities (700/900 kg m-3) when no positive compatible volume is present.
- NSSL three-moment fields are a separate contract and cannot be zero-filled
  when that option is active.

The pure-CPU conversion gate is implemented now. It intentionally raises
`MomentDiagnosisRequired` until the exact NSSL CUDA `calcnfromq` routine is
provided by the mp18 port.

## Compatibility boundaries

Gpuwm history plus a gpuwm setup/restart archive is the first complete path.
Stock-WRF `wrfout` is a supported compatibility input, but ordinary history
streams frequently omit initialization-only base and surface fields. Such a
stream must be paired with a compatible `wrfinput`/restart/setup record and the
relevant namelist physics identity. The tool fails closed when that companion
state is absent.

This contract does not claim that hourly legacy history can recover sub-hourly
parent evolution. It can build a mechanically valid coarse-forced child only
when the caller explicitly permits that cadence; it cannot manufacture the
missing forcing information.

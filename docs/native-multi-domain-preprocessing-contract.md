# Native multi-domain preprocessing contract

Status: design interface only. The current direct source adapters and direct
WRF exporter remain certified for one target domain. This document defines the
next interface without claiming that nested export is implemented.

## Goal

One source decode must be able to initialize an arbitrary, validated nest tree
for either gpuwm or unchanged CPU WRF. Geometry, vertical coordinates, physics
initialization, source identity, and forcing times must be explicit inputs.
Nothing may fall back to a hidden 49-level or real74-specific default.

## Inputs

`NativeNestRequest` owns:

- a run clock and ordered forcing times;
- one `SourceFrameContract` per source time, plus immutable source hashes;
- one `DomainTarget` per domain;
- one explicit shared `VerticalConfig` in version 1;
- one `PhysicsInitializationContract` per domain; and
- identities for static geography, interpolation policy, code, and decoder.

Each `DomainTarget` contains the projection, mass-grid dimensions, spacing,
reference point, time step, boundary widths, and the WRF nesting tuple
`(grid_id, parent_id, parent_grid_ratio, parent_time_step_ratio,
i_parent_start, j_parent_start)`. The root must be `specified=true`; children
must be `nested=true` and `specified=false`.

Version 1 deliberately requires every domain to have the same `nz`, explicit
`eta_levels`, `p_top`, `hybrid_opt`, and `etac`. gpuwm currently has one shared
vertical contract, and accepting different child levels would be a false
capability claim. A later version may add a separately validated vertical-nest
operator instead of weakening this check.

## Validation

The request is rejected before allocation unless all of the following hold:

1. Domain IDs are unique, there is exactly one root, parent references form an
   acyclic tree, and parent IDs precede their children.
2. Ratios and starts are positive integers, child spacing agrees with parent
   spacing divided by the grid ratio, and every child footprint fits inside
   its parent with the required interpolation halo.
3. Projection parameters, coordinates, grid sizes, time steps, boundary
   widths, and vertical values are finite and internally consistent.
4. The explicit eta array is strictly decreasing from exactly 1 to 0 and has
   `nz + 1` interfaces (`e_vert = nz + 1`).
5. Every requested source time exists and source pressure coverage reaches the
   requested model top. Extrapolation above the source atmosphere is refused.
6. Every physics scheme has a registered initializer describing its required
   state. Unsupported schemes or missing prognostic state fail closed; they
   are never silently substituted.

## Execution interfaces

The source adapter emits a canonical source frame independent of the target
model and source product. Static geography, horizontal interpolation weights,
and vertical coordinate terms are cached by content identity.

At the initial time, each domain may be initialized independently after its
geometry/static prerequisites are ready. Work can therefore be scheduled in
parallel across domains and fields. The result for each domain is a
`PreparedDomainState` containing:

- complete mass and staggered atmospheric state;
- surface and land state;
- explicit coordinate arrays whose shapes derive from live `nz`;
- scheme-specific initialized fields or a documented validated cold start;
- map factors, Coriolis terms, rotation terms, and static fields; and
- hashes binding source, geometry, vertical, physics, and code contracts.

Only the root receives external lateral-boundary forcing records. Child
domains receive parent-interface/nest metadata and are forced by the native
nesting machinery during integration. Directly initializing a child from the
source at the start time is allowed, but does not turn later source frames into
child external LBCs.

## Consumers

The gpuwm consumer receives a tree prepared-cache with one domain-state entry
per domain, one root LBC sequence, and explicit parent/child interface
metadata.

The unchanged-WRF consumer receives:

- `wrfinput_d01` through `wrfinput_dNN` at the initial time;
- `wrfbdy_d01` only, with all requested external forcing times; and
- a manifest binding every file to the same nest, source, vertical, physics,
  static, and code identities.

The WRF files must use dimensions derived from each live domain and the shared
vertical contract. Compatibility is not complete until an unchanged stock WRF
run accepts the entire tree and advances beyond initialization.

## Implementation gates

1. Typed request/tree validation with synthetic 1-, 2-, and 4-domain cases.
2. Parallel per-domain initial-state preparation with deterministic 1/4/8
   worker equivalence.
3. Tree prepared-cache serialization, shape/dtype rejection, and restart
   identity binding.
4. Direct WRF multi-domain export and structural NetCDF validation.
5. Unchanged stock-WRF startup gate for two domains, then four domains.
6. Short gpuwm nested integration with interface conservation and health
   checks.

Until all gates pass, public capability text must continue to say
"single-domain direct export."

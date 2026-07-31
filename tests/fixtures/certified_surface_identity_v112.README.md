# Certified surface-isolation baseline

`certified_surface_identity_v112.json` was written from the real two-step
trajectory produced by `tools/certified_surface_identity.py` against detached
worktree commit `8a33aa2a` (`v1.1.2`). The same unchanged harness produced the
candidate values.

It hashes every device array on `DomainState`, every Noah surface field, and
every active microphysics diagnostic for:

- WSM6 + Dudhia
- Thompson + Dudhia
- Morrison + RTE/RRTMGP
- NSSL-2 + RTE/RRTMGP

The synthetic grid is flat because the balanced-moist ideal-state constructor
has no terrain path; all physics selectors named above match their certified
profiles. The field-inventory digest separately proves this lane did not add
surface carriers to Noah.

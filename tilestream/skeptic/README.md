# Skeptic verification of the cuda-graphs workstream

Everything here was run against `tilestream-graphcap` (1086579) from a
separate worktree, on machines checked idle before each timing.  Scripts are
the ones that produced the logs beside them; nothing in `tilestream/` or
`gpuwm/` was modified on this branch.

* `digest_probe.py` -- per-carrier sha256 of the RESIDENT monolithic path, so
  a change that moves the tiled and monolithic sides together cannot hide.
  `digest_BASE.json` (f518024) vs `digest_HEAD.json` are identical on nine
  rungs; `digest_PERTURB.json` is the sensitivity control (G + 1e-8 relative,
  which moves both digests).
* `remote_mutate.py` -- breaks one defence at a time and runs
  `test_gate --graph-only`.  `evidence_4090_mutsweep.log` is the result.
* `probe_verify.py` -- constructs the case where the cached graph's topology
  really has moved, to show `verify_topology` can fail rather than only pass.
* `probe_settle.py` -- counts how often the settling capture and the host
  drift branches are actually taken by the gate's own cases.
* `probe_census.py` -- counts how many of the fourteen rungs capture.
* `probe_crash.py` -- isolates the illegal memory access; the deterministic
  reproducer is in `evidence_4090_iso.log` (warmup=2, steps=20).

# Boundary-time kernel frame recording

The moist WRF boundary time-law change `3eb49b2fd21d426cac79c60825ecc3e421e5a9fe`
added `lbc_time.cu` without a local-memory row or runtime module selection.
The source-coverage gate correctly found this omission. The operation is
now included in `CORE_KERNEL_MODULES`, and both entry points have actual
compiler/driver readings. No kernel arithmetic or compilation option changed.

The measurement used `tools/vram_reserve_probe.py::mode_frames`' procedure,
restricted to this module: production `load_module`, every declared entry
point, and each driver's `local_size_bytes` attribute. Compilation and symbol
lookup failures were fatal; no exception was converted to a zero frame.
The process read its fresh stack limit before compiling the module.

| Local RTX3080, sm86 | Windows11 | WSL2/Linux |
|---|---:|---:|
| NVRTC build |13.0.48|12.8.93|
| CuPy |14.0.1|14.2.0|
| Python |3.13.7|3.12.3|
| Fresh stack limit |1024B|1024B|
| `evaluate_linear_boundary` local / registers |0B /16|0B /16|
| `evaluate_rational_boundary` local / registers |0B /36|0B /36|

Both symbols also reported0B shared storage. The assembled source SHA-256
was `1d9040dabb757c4d7fef1bd76913fd1dc7fc495b1a409bb87093ff9aa16f104a`
in both processes, measured from base `73fc2406f172e672c0012f02991e236bf053c041`.
Full source hash, compile options, library hashes and driver attributes are in
[Windows receipt](measurements/lbc-time-2026-09-05/windows.json) and
[Linux receipt](measurements/lbc-time-2026-09-05/linux.json).

The existing sm86/NVRTC13.0.48 recording is extended only with its own measured
row. NVRTC12.8.93 has a new partial recording containing just this module.
The historical sm120/NVRTC13.0.88 recording predates this source and is now
explicitly incomplete; all of its prior values remain unchanged and retain
their exact-equality checks. No value was invented for an unmeasured platform.

The recorded maximum is0B, below the default stack, so adding this operation
changes no current reservation or forecast-envelope bytes. Including it in
runtime selection ensures future measured frame growth can affect admission.
These are compiler metadata measurements, not a forecast peak measurement.
The focused GPU control covers every symbol; the CPU control deliberately
raises the frame to prove that pricing observes it. Existing boundary-time
trajectory tests retain their original numerical tolerances.

The completeness review also exposed a pre-existing missing `ntiedtke` row
in the sm86/NVRTC13.0.48 table. All21 entry points were independently read
on that compiler: each local frame was0B (register counts34–174). The
[full receipt](measurements/lbc-time-2026-09-05/ntiedtke-windows.json) binds
its assembled source `337e17d2c573d4e67e54458393675e3743297efa0409208609fbbcc8f5ad7e0d`.
This fills the hole with an actual same-platform measurement and preserves
the complete table. A new CPU control checks every complete recording's
source membership before a GPU regeneration run is needed. Its existing
shipped ceiling was already0B; this adds no reservation.

Validation on the repaired source: the unchanged full Windows
`test_the_recorded_local_frames_match_the_driver` passed in31.87s,
regenerating all standalone frames, matching every applicable row and
checking the ceiling. Focused Windows controls passed20, including actual
GPU boundary evaluation; Linux passed19 before the added CPU membership
control, including the same actual GPU evaluator. The optional external
real.exe moist-boundary fixture was unavailable on both systems.

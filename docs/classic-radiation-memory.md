# Classic longwave column memory

`experiment.column_chunk` bounds classic RRTM longwave (WRF selector 1) as
well as modern RTE+RRTMGP. It applies to the 1/1 pair and mixed spectra through
the shared factory. Each call uses at most the smaller of the selected cap
and its actual domain or tile-window column count. Legacy RRTMG selector 4
retains its separate engine chunking contract.

Previously the experiment setting was ignored by classic longwave, whose
adapter chose a chunk from available GPU memory. Consequently, even an
experiment that omits the setting now has classic workspace bounded by the
existing shared default. This can change throughput: smaller chunks dispatch
more calls. It does not change per-column arithmetic. An explicitly chosen
cap is honored; it is not silently reduced to pass admission.

Low-level `RRTMLongwaveRadiation(column_chunk=None)` still auto-sizes at its
first eager solve and caches the result for graph capture. The auto-sizer now
uses the shared allocation inventory instead of a coefficient fitted to one
53-layer measurement. A smaller-than-512 block is allowed when necessary to
fit its allowance. Cold graph capture retains its existing query-free
fallback, and explicit pins bypass the automatic selection.

`gpuwm.core.rrtm_inventory` accounts for full-window column packing, the
actual packaged coefficient tables, and the largest simultaneous profile,
band absorption, or transfer live set. It includes all eleven retained
140-g-point grids, Planck and flux arrays, row operands, and allocator request
rounding. The radiation depth includes the Cavallo buffer derived from the
declared pressure top. Preflight applies its existing allocator headroom
separately; the inventory does not claim that every phase is resident at once.

The tile planner already reserves an empirical radiation allowance. It adds
only a selected classic window's excess above that allowance, so the final
reservation is the maximum of the existing allowance and the classic call
estimate. It does not add the same reservation twice or multiply a serial
radiation call by the number of tile buffers. The global and per-domain tile
options receive their cap and pressure top from the validated experiment.

A wrapped legacy/ideal `RunConfig` uses pressure top zero to mean “the
initializer supplies this later.” That placeholder remains an explicitly
unpriced classic workspace before initialization. The estimator does not
invent buffer layers or reject an otherwise executable legacy route.

Validation covers CPU cap/window/reservation controls, real CUDA allocation
peaks at 43, 53 and 65 radiation layers, and byte-identical output for explicit
caps 1, 7, the whole window, and automatic selection. These are allocation and
chunk-invariance checks; they do not claim a new WRF scientific comparison.

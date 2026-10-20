# Family products from a saved ensemble

`gpuwm ensemble-products RUN --out OUT` reads an existing ensemble receipt and
its retained diagnostics. It launches no forecast. It reduces on the GPU and
renders through `gpuwm.rustwx` and `rw_wrfbatch`.

`RUN` can name a mass launcher's `ensemble-set.json`, a native
`ensemble-manifest.json`, or a folder containing one manifest. Select an explicit
manifest for a run with several domains. Native replay needs the deferred typed
diagnostic packs and their coordinate file, or retained member histories. An
aggregate ensemble mean cannot recover family distributions. Finish this pass
before retiring the retained diagnostic inputs.
Histories and packs can coexist in one manifest. A retained history takes
precedence for the same member and valid time; packs fill the other members and
hours, with their committed identities and duplicate membership checked.

Each member's provenance supplies `parent_family` and `family_weight`. Source
and model names never select a reducer. A recorded family mass is divided among
its members in proportion to their existing member weights; changing the number
of members in another family does not change that mass. A family with zero
combined weight still has its own conditional products. Older receipts require
an explicit `--member-map member-map.json`, for example:

```json
{"0": {"family": "ai", "family_weight": 0.7},
 "1": {"family": "physics", "family_weight": 0.3}}
```

Supply the actual parents' families. A map labels existing members; it does not
create an AI forecast or establish missing lineage. If every family weight is
absent, equal family masses are the recorded default. Partial or inconsistent
recorded weights are refused.

For every whole output hour the command writes per-family and combined mean,
spread, exceedance probabilities and existing PMM fields. Thresholds come from
the shared ensemble product table, with 30 and 40 dBZ added for verification.
PMM samples the member-weighted reflectivity distribution using the weighted
mean's spatial ranks. Tied ranks share their matched value. Combined PMM is
computed from the weighted member distribution, not from averaging family PMMs.
Product NetCDF attributes retain family masses, member probability weights,
the full input provenance and the PMM definition. A nonfinite member masks the
cell rather than changing its probability denominator.

`family-products.json` records each saved frame, products, panel paths, missing
diagnostics, MRMS object identity, scan time offset, regrid contract and scores.
`scores.csv` repeats the numeric score rows. FSS and CSI use the engine's Rust
verification kernels and a common observation/forecast validity mask. They score
both ensemble mean and PMM at 30 and 40 dBZ. Default FSS box widths are 1, 3, 7,
11 and 21 cells (3, 9, 21, 33 and 63 km with `--dx-m 3000`). Undefined values are
JSON null, and no common observed cells are labelled explicitly.

Each comparison sheet has MRMS followed by each family and the combined
product. Rows show each reflectivity exceedance threshold, then PMM. MRMS comes
through the shared `MrmsCompositeCache` and native regrid path, with the chosen
scan's offset in the receipt. `--offline` requires an existing matching cache.
An absent or unreachable observation is recorded as unavailable and exits 2.

Use `--valid-time YYYY-MM-DDTHH:MM:SS` repeatedly to select saved instants,
including sub-hour frames. History precipitation is relative to the earliest
retained history for that domain, which is recorded in its product attributes.
Preserve the initial frame if a full forecast accumulation is required.

On a host whose card is shared, run it under that host's card scheduler with a VRAM request that covers its real peak:

```sh
python -m gpuwm.cli ensemble-products RUN --out OUT
```

The 1536 MiB value is the measured small-control envelope, not a universal
capacity estimate. Choose an envelope that covers the saved grid and roster.

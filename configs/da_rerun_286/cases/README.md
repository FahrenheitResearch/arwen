These five case plans contain public acquisition manifests for e4, r3s8,
z5, c3 and b3. They contain actual object keys, listing sizes, ETags and
listing modification times from the unsigned inventory. They contain no
weather-data hashes or prepared-authority hashes invented before retrieval.

`case.json` records the historical event center, radar, HRRR cycle, forecast
fork, proposed cycling window and the exact 1, 3 and 6 hour scoring windows.
`fetch-inventory.json` can be consumed by the deck's `fetch_public.py fetch`
command. Fetch once through a single process into the shared input copy.
For several cases, use the deduplicated all-case inventory from the deck,
rather than fetching the same MRMS objects again through separate case files.

The run manifests are blocked. Each `case.json` lists these missing inputs:

* Original timestamped radar slots and the approved public Vr/Z QC recipe.
  The extra-case `da_start` values are proposed two-hour windows, not exact
  recovered first-volume timestamps.
* Actual prepared bundles, full Lambert geometry and eta coordinates,
  terrain/static data, resolved current physics and source/authority hashes.
  `shared.moist_cq` must be explicitly true for both matched arms.
* A pinned surface station table, approved age/elevation/completeness policy
  and decoded surface seam record. The AWS inventory supplies no surface data.
* Tracked observed forecast-hour footprint masks, their tracking recipe and
  QC receipts. An event center is insufficient to define these masks.
* Decoded native MRMS valid/QC masks, declared product height support and a
  complete common-support receipt. Object presence is not continuous coverage.
* A lead decision on the remaining historical research-only features. No
  private source was read or copied to make these manifests.

These are historical tuning cases. Their measurements do not satisfy the
specification's unseen-event qualification requirement.

Rebuild the metadata files from the deck inventory, without retrieving data:

```sh
python3 tools/da_case_manifest.py \
  --cases regional-cases.json \
  --inventory receipts/all-regional-public-inventory.json \
  --out configs/da_rerun_286/cases
```

The command is stdlib-only. The preparation interfaces it names are existing
public doors: `tools.da_background_ab.prepare_case` binds a real source
manifest; `gpuwm.source_cli --source hrrr` prepares the native source;
`gpuwm.hrrr_route_inputs.write_hrrr_route_inputs` authors nested geometry.
They still require the missing geometry and actual retrieved source bytes.

Validation command: `python -m pytest -q tests/test_da_case_manifest.py`.
Result: 4 passed. The tests check exact HRRR identities, radar-site selection,
midnight truth windows, missing-forcing refusal, explicit missing run inputs
and timezone/horizon refusal. No model, GPU, data retrieval, preparation or
native truth decode ran for these case plans.

The retained acquisition inventory uses its recorded echo products. For the
new primary QC composite scorer, regenerate an inventory that also contains
`MergedReflectivityQCComposite_00.50` using the updated deck fetcher. These
metadata files do not establish that primary echo coverage.

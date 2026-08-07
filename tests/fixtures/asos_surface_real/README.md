# Real-seam ASOS surface fixture

Every file here is either a **real archived artifact** of the obs battery
or the **real `rw_asos` binary's own output** over one — nothing was typed
by hand and nothing was produced by a Python METAR parser (none exists in
this repo, on purpose).

Provenance chain:

1. `observations_subset.csv` — a row subset of the obs battery's archived
   IEM fetch artifact
   `cache/obsbattery/battery/asos/2024-05-21/observations.csv`
   (fetched from mesonet.agron.iastate.edu by `rw_asos fetch`,
   report_type routine+SPECI).  Selection was purely textual: the header
   line plus every row whose station is one of
   DSM, IKV, AMW, BNW, I75, OXV, TNU, PRO, MIW, PEA and whose valid time
   lies in 2024-05-21 11:30–14:30 UTC.  89 rows.  No field of any row was
   edited.  sha256
   `c67c94c6f54ca6d1bf38c7c1d8f8934cdecf4a04d01a5ca691c148587de42462`.
2. `stations.json` — the obs battery's frozen, hash-pinned station table
   for the same case, byte-for-byte
   (`cache/obsbattery/battery/asos/2024-05-21/stations.json`,
   station_table rows digest
   `e10ec778da19bf368f146a92e43f4ab2cc94836ed3b344d15039c0ca3efb1ed1`).
3. `surface_subset.v1.json` — `gpuwm-obs.asos-surface.v1`, written by the
   REAL writer:

   ```
   rw_asos decode --stations stations.json --obs observations_subset.csv \
       --start 2024-05-21T12:00:00Z --end 2024-05-21T14:00:00Z \
       --step-hours 1 --out surface_subset.v1.json
   ```

   rw_asos 0.1.0, built from the vendored BowEcho rustwx workspace on
   integration/obs-battery (`tools/rustwx/crates/rw-obs/src/bin/asos.rs`),
   binary sha256
   `edc9b1300e80ce6cf7445e0b3f24d87626910c0e85842e620bc780e4ba46daf7`.
   8 stations survived, 24 reports at 3 hourly valid times; PEA and PRO
   were dropped by the writer's own completeness rule — real refusals,
   kept because tests should see them.

`tests/test_da_obs_surface.py` re-runs step 3 against this very CSV when
`GPUWM_RW_ASOS` points at an `rw_asos` binary, and compares the seam
fields with the committed record — the reader is tested against the other
lane's real writer, never against its own fixture-making code.

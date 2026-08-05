# B5 — the observation archive of record, pulled

2026-08-04, against `0279f0f2` (`integration/obs-battery`). Every number below
was produced by the front doors' own fetch records; the manifests beside this
file (`OBS-ARCHIVE-MANIFEST.json` and `manifests/obs-YYYYMMDD.json`) carry the
per-object URL, SHA-256, byte count and retrieval instant that these totals sum.

**The objects are not in the repository.** They live at
`~/gpuwm/cache/obsbattery/battery` on the box that pulled them; the
manifests are what the archive-of-record mirror is verified against, and what a
re-fetch is compared with when an archive quietly re-issues a file. No mirror
was attempted here and nothing was copied off this box.

No forecast was run for this document, on either venue, and it makes no claim
about skill, of any model, against anything.

---

## 1. What was pulled

All seven case days of the spec's section 1.2 menu, whole-day, in one pass.

| case day | MRMS objects | MRMS bytes | Stage-IV objects | Stage-IV bytes | ASOS stations frozen | ASOS bytes |
|---|---|---|---|---|---|---|
| 2021-12-10 | 718 | 1.221 GB | 46 | 14.0 MB | 799 | 8.5 MB |
| 2023-03-31 | 708 | 1.552 GB | 46 | 15.5 MB | 743 | 7.9 MB |
| 2024-04-27 | 720 | 1.510 GB | 46 | 16.5 MB | 575 | 6.0 MB |
| 2024-05-21 | 720 | 1.216 GB | 46 | 12.6 MB | 642 | 6.8 MB |
| 2024-07-15 | 720 | 1.011 GB | 46 | 11.8 MB | 698 | 7.4 MB |
| 2025-03-14 | 720 | 1.190 GB | 46 | 14.1 MB | 775 | 7.9 MB |
| 2025-05-16 | 720 | 1.403 GB | 46 | 11.8 MB | 802 | 8.4 MB |
| **total** | **5026** | **9.103 GB** | **322** | **96.2 MB** | — | **53.0 MB** |

Battery total: **5369 objects, 9.252 GB**, every byte SHA-256'd at fetch.

* **MRMS** — `MergedReflectivityQCComposite_00.50`, whole UTC day, from
  `noaa-mrms-pds` (spec section 2.1: the day is fetched and kept even though
  scoring reads the hourly-bracketing frames, because the delta is under a
  gigabyte and buys re-registration freedom).
* **Stage-IV** — the Iowa archive (`WAVE-ERRATA-20260804.md` section 1), all
  three accumulation windows `01h`/`06h`/`24h`, over `D 00 UTC .. D+1 12 UTC`
  so a 24 h forecast from a 12 UTC init is covered end to end with slack:
  37 + 7 + 2 objects per case day, every one present, no 404 on any day.
* **ASOS** — one bounded IEM window per case day over that case's own
  480 x 400 x 3 km domain box, `D 12 UTC .. D+1 12 UTC` with the driver's
  60 min slack, station table frozen and hashed before the observations were
  requested.

**Wall clock.** MRMS 2135 s of stream time (261-372 s per case day), pulled at
most four concurrent streams and about 13 minutes elapsed; single-stream rate
measured on the 720-object 2024-05-21 day: **4.66 MB/s**. Stage-IV: all 21
window fetches in **75 s**. ASOS: all seven case days in **~40 s**. Nothing was
hammered: MRMS never exceeded four streams, and the ASOS front door now paces
itself (section 3).

**Sizing, measured against the errata.** `WAVE-ERRATA-20260804.md` section 4
records B1's projection of **8.6 GB, ~30 minutes**. Measured here: **9.252 GB**
and ~36 minutes of stream time. The projection was low by 7.6 %, which is what
a projection scaled from seven 21 UTC probe frames should be — 21 UTC is peak
convective coverage and compresses worst, so a per-file mean taken there
overestimates the day and the day count carried the rest. The errata's headline
conclusion is unchanged and now measured rather than projected: **the
observation side is not a storage or scheduling constraint**; the WRF arm's
staged HRRR (errata section 3, 25-37 GB per case) remains the storage long pole.

## 2. Archive vintage: two days are not 720 frames

The MRMS archive publishes a nominal 2-minute cadence. Two case days are short,
and both are the archive's own gaps, not fetch failures — `matched_frames`
equals the file count on every day, so nothing listed was missed:

* **2021-12-10: 718 frames.** Two single missing slots, at
  `12:18:40 -> 12:22:36` and `23:12:42 -> 23:16:40` UTC.
* **2023-03-31: 708 frames.** Twelve gaps, ten of them clustered between
  **10:24 and 12:11 UTC** — the archive was thin that morning. The longest is
  `11:59:14 -> 12:04:29` (315 s), and it straddles 12 UTC.

The 12 UTC straddle is the only one that touches a registered instant: for a
12 UTC init on that case, the nearest frame to `12:00:00` is `11:59:14`, an
offset of **-46 s**, comfortably inside the registered +/- 240 s matching
window. Nothing here disqualifies a case; it is recorded so that a case entry
receipt on 2023-03-31 states it rather than rediscovers it.

Every other day is 720 for 720.

## 3. Two front-door defects, found by real case boxes and fixed

Neither could be found on a probe-sized query, and both were found by the first
pull at battery shape. Both fixes are in
`tools/rustwx/crates/rw-obs/src/bin/asos.rs` and are covered by a test.

### 3.1 HTTP 414 — the bounded query outgrows the endpoint

`rw_asos fetch` names every frozen station in one `station=` list. A battery
case box is 1440 x 1200 km, and over the dense Midwest or Southeast that
freezes **575 to 802** ASOS/AWOS sites — far above the spec's "~40-120
stations/case" estimate, which was written for a station set already thinned by
the interior mask and the admission rules rather than for the box.

Measured directly: a 642-station table answered **200**; a 698-station table
answered **HTTP 414 URI Too Long**. Five of the seven case days failed on the
first pass, and all five failed for that one reason.

Fixed by chunking: `fetch` splits the frozen table into groups of
`--stations-per-request` (default **400**, well inside the measured ceiling),
requests each group with the identical pinned parameter set, and concatenates
the CSV bodies under one header — refusing outright if a later chunk answers
with a different header, because concatenating columns that do not line up
would shift every field. The query stays bounded by the frozen table, which is
the property the bin exists to keep; the digest and the decoder see the
assembled file. The record now carries `requests`, `request_urls` and
`stations_per_request` beside the fields it already carried.

### 3.2 HTTP 429 — seven case boxes back to back is a burst

With chunking in, seven case days went out as twenty-one requests in about
thirty seconds and the twenty-first answered **HTTP 429**. The archive is free
and is asking to be paced.

Fixed by pacing: `--request-pause-ms`, default **2000**, between the chunk
requests of one window. The default is the pace rather than a flag an operator
has to remember; the re-run of the failed day answered 200 on every chunk.

## 4. What the pull proves about the observations themselves

* **The `-99` ruling holds on whole days, not just probe frames.** Every one of
  the 36 decoded case-box frames reports `observed_fraction` **1.0000** — the
  scored interior is fully observed at every scored lead, against the spec's
  "> 10 % masked interior fails the case" bar (section 3.1). With `-99` masked
  instead of valued, B1 measured that same box at 0.238.
* **Stage-IV is complete on every case day** at all three accumulations, which
  is the presence half of every case's entry receipt (spec section 1.1).
* **Every observation re-hashed clean.** The 34 archive objects behind the
  shakedown case's scored fields were re-hashed at scoring against the digest
  taken at fetch: 34 of 34 match (`B5-OBS-CONTROLS-20240521.json`,
  `observation_rehash`). That is the observation half of the promotion rule's
  integrity clause R4, exercised before any run exists.

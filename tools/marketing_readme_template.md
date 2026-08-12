# ArWen — 2019-05-28 Kansas dryline outbreak, streamed out of core

**Every image in this folder is a SIMULATION, not an observation.** The
reflectivity is the model's own `REFL_10CM`, not a radar mosaic.

## The forecast

| | |
|---|---|
| Event | 2019-05-28 central/eastern Kansas dryline supercells; the Lawrence–Linwood EF4 was on the ground 2306Z–0000Z |
| Initial condition | operational **HRRR 15Z analysis** (f00, 50 native hybrid levels), NOAA archive `noaa-hrrr-bdp-pds` |
| Lateral boundaries | the same HRRR cycle's f01…f09, real hourly cadence |
| Grid | **1200 × 900 × 49 = 52 920 000 cells**, Lambert conformal, Δx **2.999 km** |
| Coverage | 3599 × 2699 km, 25.0°N–50.7°N, 120.7°W–72.0°W |
| Physics | Morrison mp10 · MYNN surface layer + PBL · Noah-MP · RTE-RRTMGP (12-min cadence) · **no cumulus parameterisation** |
| Integration | Δt 15 s, F+00 → F+09 (15Z 5/28 → 00Z 5/29), history every 15 forecast minutes |
| Hardware | ONE NVIDIA RTX 4090, 24 GB |

**Operational HRRR runs at 3 km.** This forecast is at the same spacing, on a
domain of comparable size, on a single consumer card.

## Why "out of core" is the point

The domain does not fit on the card. The model state plus its physics driver
at this size leaves **2.5 GiB free of 23.5 GiB** on the 4090 — before the
dycore asks for a single byte of per-step scratch — so a resident integration
is refused. ArWen instead keeps the whole domain in **pinned host RAM**
(11.6 GiB over 213 carriers, 231 B per cell), gathers one 400×300 tile plus
its 16-cell halo to the GPU, steps it, and scatters the interior back. Nine
tiles per sweep, two buffers deep.

## What the outer frame is

The outermost 44 cells (20 hard + 24 cells of cosine Davies taper) are
prescribed from the analysis every step. **Nothing in that frame is a
forecast** and it is excluded from every number quoted here.

## UP_HELI_MAX

Where updraft helicity appears it is the **instantaneous 2–5 km updraft
helicity at the plotted valid time**, not WRF's running maximum between
history writes. Each wrfout carries `GPUWM_UP_HELI_MAX_SEMANTICS` saying so.

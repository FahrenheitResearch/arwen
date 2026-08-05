# cross-reader receipt: the science core against an independent read

- schema: `gpuwm.obs-cross-reader/v1`
- evaluator commit: `8193cdad54b08aa009a4e396b32a6bc8cd65dafd`
- scope: `reader-qualification` (no registered battery case named; this receipt qualifies the reader on the writers it was given and claims nothing about a battery case)
- registration sha256: `44b0fddbb4e4964c01317d60f9c052ccd57fe2e1b1d3ef41f4a5f988cabb5c26`
- science core: `wrf-rust==0.2.35` (pin 0.2.35, pinned)
- science core module: `~/wrf-rust-diamomd/python/wrf/__init__.py`
- pairing: paired
- verdict: **PASS**

| side | quantity | units | kind | tol | max abs diff | rms diff | differing elements | verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| arwen | refl_10cm_column_max | dBZ | reduction | 0 | 0 | 0 | 0 | PASS |
| arwen | t2 | K | passthrough | 0 | 0 | 0 | 0 | PASS |
| arwen | uvmet10_u | m s-1 | derived | 1e-09 | 0 | 0 | 0 | PASS |
| arwen | uvmet10_v | m s-1 | derived | 1e-09 | 0 | 0 | 0 | PASS |
| wrf461 | refl_10cm_column_max | dBZ | reduction | 0 | 0 | 0 | 0 | PASS |
| wrf461 | t2 | K | passthrough | 0 | 0 | 0 | 0 | PASS |
| wrf461 | uvmet10_u | m s-1 | derived | 1e-09 | 0 | 0 | 0 | PASS |
| wrf461 | uvmet10_v | m s-1 | derived | 1e-09 | 0 | 0 | 0 | PASS |

## pairing: the stored georeference across the sides

| field | max abs diff (deg) | tol (deg) | verdict |
|---|---:|---:|---|
| XLAT | 2.86102e-05 | 0.0001 | PASS |
| XLONG | 3.05176e-05 | 0.0001 | PASS |

## diagnostics (recorded, never gated)

| side | quantity | alternate recipe | max abs diff |
|---|---|---|---:|
| arwen | uvmet10_u | `U10*COSALPHA + V10*SINALPHA` | 5.94294 |
| arwen | uvmet10_v | `V10*COSALPHA - U10*SINALPHA` | 4.53925 |
| wrf461 | uvmet10_u | `U10*COSALPHA + V10*SINALPHA` | 6.01059 |
| wrf461 | uvmet10_v | `V10*COSALPHA - U10*SINALPHA` | 4.51845 |

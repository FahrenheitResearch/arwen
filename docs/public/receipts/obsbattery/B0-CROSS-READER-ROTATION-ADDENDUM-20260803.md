# B0 addendum: the earth-rotation sign, adjudicated

Addendum to `B0-CROSS-READER-20260803.json` / `.md`.
Date: 2026-08-03. Scope: the `uvmet10` rotation convention only. The scored
verdict of the receipt this amends is unchanged.

The cross-reader receipt as issued carried the opposite-sign rotation beside
the scored one and called it an open question about which convention is
right. **That framing was wrong, and this receipt retracts it.** There is no
convention split. The rotation the receipt registered and scored is correct,
its PASS stands exactly as issued, and the opposite-sign number it published
is not a rival convention at all — it is the cost of a sign error.

---

## 1. The settled formula

Earth-relative 10 m wind from the grid-relative components and the frame's
own stored rotation fields:

```
u_earth = U10 * COSALPHA - V10 * SINALPHA
v_earth = U10 * SINALPHA + V10 * COSALPHA
```

This is character-for-character WRF v4.6.1's own `earth_u` / `earth_v`
(`share/wrf_timeseries.F:418-419`). It is what the science core computes
(`crates/wrf-core/src/diag/wind.rs`, `rotate_to_earth` / `compute_uvmet10`),
and it is what this battery's registered control computes. The science core
is correct and needs no change.

## 2. The angle those fields carry

```
alpha = cone * (STAND_LON - XLONG)
```

positive **west** of the standard meridian, written by WPS geogrid's
`get_rotang` (`process_tile_module.F:1951-1961`) and stored as `SINALPHA` /
`COSALPHA` in every frame. Both writers in the amended receipt store it that
way; verified on both files below.

## 3. Why wrf-python is not the opposite

wrf-python does **not** apply the opposite rotation. It recomputes the
*negative* angle, `cone * (XLONG - STAND_LON)`, and applies the sign-flipped
formula shape to it (NCAR wrf-python `develop`: `fortran/wrf_user.f90`,
`src/wrf/g_uvmet.py`). The two negations cancel and the resulting rotation
matrix is identical to §1.

WRF, WPS, wrf-python and the science core therefore all compute one
rotation. Nothing needs reconciling.

## 4. What the published diagnostic actually is

The amended receipt's diagnostic applied the **stored** angle inside
wrf-python's **sign-flipped formula shape**. That combination takes the
negation from one lineage and the angle from the other, so the cancellation
in §3 does not happen and the result is rotated by `2*alpha`. No shipping
WRF post-processor computes it.

It is retained in the instrument, relabelled: it is a **magnitude control on
the rotation sign**, and its number is what a sign error would cost on the
domain being read. On a domain whose `|alpha|` reaches 0.27 rad that is
about 6 m/s — which is precisely why the control is worth keeping.

## 5. Geometric ground truth

Ground truth taken from the grid axes' true azimuths and nothing else: at
each mass point the azimuth `A` of the grid `+y` axis is derived from
central differences of the stored `XLAT` / `XLONG` on a spherical earth, and
the earth-relative components follow by projecting the grid-relative wind
onto east and north,

```
u_east  =  U10 * cos(A) + V10 * sin(A)
v_north = -U10 * sin(A) + V10 * cos(A)
```

with no rotation formula and no stored `SINALPHA` / `COSALPHA` assumed.
Measured on both frames of the amended receipt's pair, `d01`, valid
1974-04-03_14:00:00, on the interior with the two outermost rows and columns
of the differencing stencil trimmed:

| quantity | `wrf461` side | `arwen` side |
|---|---:|---:|
| `max abs(stored alpha - cone*(STAND_LON - XLONG))` | 8.25e-08 rad | 5.95e-08 rad |
| `max abs(geometric azimuth A + stored alpha)` | 1.23e-04 rad | 3.05e-05 rad |
| **registered rotation vs geometric truth** | **1.51e-03 m/s** | **3.76e-04 m/s** |
| sign-error control vs geometric truth | 6.0095 m/s | 5.9429 m/s |

The registered rotation sits at the noise floor of the ground truth itself:
`XLAT` / `XLONG` are stored FP32, and differencing them amplifies that
storage noise, which is what the third row's 1e-4 rad and the fourth row's
1e-3 m/s are made of. The sign-error control reproduces the amended
receipt's published figures (6.01059 and 5.94294 m/s) to within that same
floor.

The second row is the mechanism in one line: the geometric azimuth is
*minus* the stored alpha, which is exactly wrf-python's angle, and applying
the sign-flipped shape to it recovers §1.

## 6. No surface score is sensitive to any of this

Rotation is orthogonal, so wind **speed** is invariant under it. Measured on
the same frames, `max abs(speed(rotated) - speed(grid))`:

| rotation | `wrf461` side | `arwen` side |
|---|---:|---:|
| registered | 1.060e-06 m/s | 4.355e-07 m/s |
| sign-error control | 1.060e-06 m/s | 4.355e-07 m/s |

Identical under both, at FP round-off. The science core's `wspd10` is
`hypot(U10, V10)` and touches no rotation angle at all. The battery's
surface guardrail scalars are bias and RMSE of `T2`, 2 m dewpoint and
`wspd10`; **none of them is convention-sensitive**. The rotation matters
only where a score reads wind *direction* or a vector component, and no
registered score does today.

## 7. Scope of this adjudication

Northern-hemisphere Lambert conformal, `MAP_PROJ = 1`. That covers every
case in the battery's proposed set. **Not audited:** the southern-hemisphere
sign flip, and every other projection. A battery case outside that scope
re-opens this receipt rather than inheriting it.

## 8. Effect on the instrument

The measurement is untouched: same registered recipe, same tolerances, same
scored numbers, same PASS. What changed is the wording carried beside the
diagnostic, and the registration now states the stored angle, the settled
status, this citation and the scope above.

| | registration sha256 |
|---|---|
| as issued in `B0-CROSS-READER-20260803.json` | `44b0fddbb4e4964c01317d60f9c052ccd57fe2e1b1d3ef41f4a5f988cabb5c26` |
| after this addendum | `61d4a77214700c504db01b562f73b08db1c700215053964855d12d5d69f58eec` |

The issued receipt is **not** rewritten. It was scored under the contract it
carries, and editing that contract after the fact would misstate which
contract was in force. Both hashes are recorded here; receipts issued from
now on carry the second.

# Explicit acoustic comparisons

Both routines were compared with compiled WRF v4.7.1 on 20 canonical cases: six real-state or stress crops with mapped and identity-map launches, plus periodic, open, top-lid, zero-reference, later-substep pressure-history and three moist-loading launches. The moist face factors come from the engine's own `prepare_moist_cq` provider and are supplied identically to Fortran. Four diagnostic cases isolate contraction, the scalar coordinate and reference-flux cancellation. No kernel changes were made for these two routines.

The packaged fixture contains the actual compiled Fortran answers and every engine output word. Regression tests assert the complete engine words and the exact measured discrepancies. None of the measurements below is an acceptance tolerance or a general error bound.

| Routine | Status | Explanation for a non-specialist |
| --- | --- | --- |
| `advance_uv` | DIFFERENT, explained FP32 operation order | Rounding the half-level geopotential before its horizontal difference changes the wind update's last bits. |
| `advance_mu_t`, `advance_mu_th`, `advance_mu_th_msf` | DIFFERENT, explained scalar coordinate and reference-flux operation order | Subtracting nearly equal total and reference fluxes in WRF gives different small residuals from advancing the residual directly. |

`advance_mu_th` and `advance_mu_th_msf` are the identity-map and mapped engine paths for the same WRF routine, not two additional WRF routines.

## Every direct output

Maximum distances across the 20 canonical cases are recorded below. Large ULP values involving zero or a sign crossing do not establish a useful accuracy bound.

| WRF output or engine field | Largest measured ULP distance |
| --- | ---: |
| `u` / `u_pp` | 4096 |
| `v` / `v_pp`, excluding the zero-input cancellation family | 10240 |
| `v` / `v_pp`, zero-input cancellation family | 154035783 |
| `mu` / `mu_pp` | 20654 |
| `mudf` | 10327 |
| `muave`, captured from the real implicit kernel register | 45439 |
| `muts`, captured from the real implicit kernel register | 0 |
| `t_ave`, captured from the explicit driver's old-state buffer | 0 |
| canonical `t` / transformed `th_pp` | 2089485341 |
| `ww` / `ww_pp` | 1866547809 |

The compiled harness also checks that `ww_1`, `uam`, `vam` and `wwam` remain unchanged. These arguments are never assigned by WRF's `advance_mu_t`. The diagnostic stores for `muave` and `muts` are trusted only after the instrumented implicit launch reproduces every native `w`, geopotential, pressure and inverse-density output word.

## Quantified causes

At the real initial case's upper-level v face `(48,4,6)`, WRF's supplied, pre-rounded half-level geopotential difference is `-0.21875`; the engine's difference of full-level sums is `-0.25`. The `-0.03125` difference multiplied by the native terrain coefficient `71.25118255615234`, inverse spacing `1/3000` and substep `0.25` predicts a coupled-momentum difference of `0.00018554996`. The measured largest difference is `0.00018548965`. Disabling FMA retains this difference. The relevant expressions are `acoustic.cu:pgrad_face` and WRF `advance_uv`'s `php` term at source lines 935-936.

The later-substep case uses `first=False` in the actual driver and nonzero `smdiv=0.1`, with different current and previous pressure arrays. WRF receives its float32 weighted pressure input. Its largest coupled-momentum difference is `0.00018644333`; the overall maximum ULP values in the table are unchanged by this case.

The zero-input family retains 17 v words of `2.0992578721129284e-33` where WRF writes zero, even with FMA disabled. This is a normal float32 value, not a flushed subnormal. WRF adds the large-step tendency first and then subtracts the pressure tendency; the engine combines the two tendencies before its final update. Ordinary nonzero-case discrepancies must not be described by this zero-to-tiny-number ULP distance.

At the real initial case's omega point `(19,1,4)`, compiled WRF forms total omega `-0.14318422973155975` and reference omega `-0.14318618178367615`, then stores their float32 difference `+1.952052116394043e-6`. The engine's perturbation-only recurrence writes `-1.388411874359008e-6`. The code order is WRF `advance_mu_t` lines 1090-1119 versus the separate reference and perturbation reductions and perturbation-only recurrence in `acoustic.cu`. This explains the sign crossing; the native cancellation is observable, not inferred from a NumPy mirror.

The existing full-theta coordinate and face-mass distinctions are declared in `gpuwm/core/ieva.py:32-50`. The acoustic source also states its contraction policy at `acoustic.cu:36-39` and the perturbation-only omega recurrence at `acoustic.cu:321-324`. The full-theta, zero-reference, FMA-disabled diagnostic has zero ULP distance in `mu`, `mudf`, `muave` and `ww`, and two ULP in theta. Four mass words differ only in the sign of zero. These controls identify the rounding amplifiers; they do not replace the canonical comparisons.

## Tests and reproduction

The test names are:

* `test_horizontal_compiled_fixture_pins`
* `test_advance_uv_against_compiled_wrf471`
* `test_advance_mu_t_against_compiled_wrf471`
* `test_horizontal_word_gate_rejects_missing_pressure_term`

Run the packaged gate with `python -m pytest -q tests/test_smallstep_horizontal_wrf471_parity.py`. The mutation test removes the native terrain pressure term in a diagnostic compile and requires the complete-word gate to reject it.

Rebuild the fixture with `python tools/smallstep_wrf471_oracle/horizontal_measure.py --output tests/data/wrf471_smallstep/horizontal-baseline.json --fixture tests/data/wrf471_smallstep/horizontal.npz`, with `GPUWM_SMALLSTEP_ORACLE_LIB` pointing to the pinned library. Run the intermediate witness with `python tools/smallstep_wrf471_oracle/horizontal_cause.py tests/data/wrf471_smallstep/horizontal.npz --library <compiled-library> --case real_initial_map`.

This comparison establishes the recorded answers and the identified arithmetic differences for these cases. It does not establish canonical bit identity, a uniform small ULP bound, or forecast qualification.

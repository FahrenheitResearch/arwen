# Coordinate-surface diffusion oracle

The build extracts unchanged WRF v4.7.1 routine bytes at commit
`f52c197ed39d12e087d02c50f412d90d418f6186`. SHA-256 pins, compiler flags,
routine line ranges and wrapper bytes accompany the fixture archive.
The configuration type and C ABI wrapper adapt storage only. Constants
are compiled from WRF's own `module_model_constants.F`.

```sh
python tools/wrf_diffopt1_oracle/build.py WRF_SOURCE BUILD_DIRECTORY
python tools/wrf_diffopt1_oracle/fixtures.py BUILD_DIRECTORY/oracle.so FIXTURE_DIRECTORY
python tools/wrf_diffopt1_oracle/capture.py FIXTURE_DIRECTORY OUTPUT_PREFIX --mode km2
python tools/wrf_diffopt1_oracle/capture.py FIXTURE_DIRECTORY MERGED_OUTPUT_PREFIX --mode all
python -m pytest -q -o addopts= tests/test_diff_opt1.py tests/test_diff_opt1_wrf471.py
```

The fixture set is entirely synthetic. Forty horizontal cases cover all
C-grid staggers, initial-theta subtraction, flat and projected maps,
periodic and open boundaries, and retained nonzero tendency words. Their
27,792 output words match the compiled coordinate routines exactly.
Eight coefficient cases match all 10,368 km4 output words exactly.
Thirty-two km2 cases cover both mixing-length modes, both surface-flux
seed switches, zero and near-zero TKE, projection and boundary combinations.
The horizontal Km/Kh comparison covers 41,472 words with a measured maximum
of five ULP on the recorded RTX 5090 corpus; exact production output words
are separately pinned. Every returned TKE input word remains unchanged.
The captured vertical coefficient differences are diagnostics: diff_opt=1
does not consume that pair or call `vertical_diffusion_2` or `tke_rhs`.
The same compiled module also supplies eight deformation probes and eight
stability probes. All 20,736 captured N2 words match exactly. D11 and D22
match exactly across their complete arrays. D12 has a maximum absolute
difference of `2**-31` across its complete arrays, including physical tensor
rows now copied from WRF's evaluation-point donors. Complete output words
and WRF distances remain recorded for those probes.

`merged-gpu.json` and its archive measure every one of the 88 coordinate
cases after the diffusion corrections. `merged-attribution.json` preserves
prior/current distances and every changed word. Only 492 D12 boundary words
move from the historical production capture. A tools-only control removes
exactly the donor mapping added in `83fde6032`; all 216 prior coefficient
and tensor arrays then return to their original words. The GPU regression
repeats this control in an isolated process. It neither changes production
kernel compilation nor widens a numerical bound:

```sh
python tools/wrf_diffopt1_oracle/capture.py FIXTURE_DIRECTORY CONTROL_OUTPUT_PREFIX \
  --mode all --revert-deformation-donor
```

A second tools-only control retains those donors and rounds/weights the
meridional and zonal D12 terms separately, as WRF does. All 24 deformation
arrays then match all 15,552 compiled WRF words. This explains the remaining
bounded default arithmetic difference without changing the production path:

```sh
python tools/wrf_diffopt1_oracle/capture.py FIXTURE_DIRECTORY ORDER_OUTPUT_PREFIX \
  --mode all --deformation-reference-order
```

WRF's `module_em.F` always uses `horizontal_diffusion_3dmp` with `t_init`
for theta in this branch, independently of `mix_full_fields`. The logical
changes only vertical-shear tensors and vertical diffusion routines that
these km2/km4 coordinate operators do not consume. Horizontal moisture and
TKE scalars use Kh and full fields. The coordinate v meridional flux
retains the missing dry-mass factor in WRF's pinned source.

The previous branch `a4177ebbf342252405df3f6ed8309704daee94fc` produced the
historical diff_opt=2 reference and legacy checkpoints, retained unchanged.
The merged metric baseline separately accounts for the authorized diffusion,
advection and large-step oracle corrections through source-isolated captures.
The acoustic correction is measured as inert on this corpus. Eight production
cases run three complete RK3 steps and compare
every retained state and mixing array, 101,808 words in 160 arrays.
Four coordinate cases compare uninterrupted four-step runs with a two-step
checkpoint plus two-step continuation. Two genuine previous-branch metric
checkpoints compare against an independent continuation under the merged
operator. These are operator and continuation checks, not observation-based
skill. `diff2-merged-attribution.json` records the exact baseline movement;
`merged_metric_capture.py` and `merged_metric_attribution.py` reproduce it.

For metric attribution, export `gpuwm/core/dycore.py` and the `smag2d`,
`diff6`, `diff6_seam`, `dycore`, `advection`, `openbc` and `acoustic` CUDA
files from `a4177ebbf342252405df3f6ed8309704daee94fc` into a control directory
by basename. The source-isolated captures restore successive default-on
oracle groups while leaving the merged configuration and synthetic inputs
unchanged. Restoring all groups must return every historical metric and
legacy continuation word exactly. No numerical tolerance is used:

```sh
python -m tools.wrf_diffopt1_oracle.merged_metric_capture FIXTURE_DIRECTORY \
  CAPTURE_DIRECTORY/control-0
for level in 1 2 3 4; do
  python -m tools.wrf_diffopt1_oracle.merged_metric_capture FIXTURE_DIRECTORY \
    CAPTURE_DIRECTORY/control-$level --control-level "$level" \
    --control-source CONTROL_DIRECTORY
done
python -m tools.wrf_diffopt1_oracle.merged_metric_attribution FIXTURE_DIRECTORY \
  CAPTURE_DIRECTORY OUTPUT_DIRECTORY --merged-commit MERGED_REFERENCE_COMMIT
```

The compact summary and exact merged archives are regression fixtures. The
separately emitted `diff2-merged-attribution-words.json` is the complete
index, original and current hexadecimal words, and source transition record
for every moved word. Its SHA-256 is sealed in the compact summary.

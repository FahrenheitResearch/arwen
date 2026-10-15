WSM5 (mp_physics=4) uses WRF 4.6.1 single-moment five-class microphysics.

Source: WRF v4.6.1 commit d66e442fccc04111067e29274c9f9eaccc3cef28,
phys/module_mp_wsm5.F and the radar helper in phys/module_mp_radar.F.
WRF LICENSE.txt declares these scheme sources public domain. The plain WSM5
module is the source; the alternate accelerator module is not used.
WOOF arithmetic helpers remain WOOF code.

The shared column scheme table owns qv, qc, qr, qi and qs. No graupel or
number moment is allocated. Both default and GPUWM_WRF_EXACT=1 use the shared
unflushed loader with --fmad=false --ftz=false.

Two corrections run by default and are declared in the physics registry:
1. The sedimentation top search visits the last arrival interface and keeps
   the top cell at or above the bottom cell. This prevents upper-cell water loss.
2. Melting snow evaporation uses the WSM6 expression. Stock WSM5 uses psdep,
   which is zero wherever its warm, subsaturated evaporation branch runs.
The oracle builds each correction separately and a defined arm with both.
The stock discrepancy mask must equal the two measured probe sets.

Build: bash tools/wsm5_wrf461_column_oracle/build.sh "$WRF_SOURCE_ROOT" ORACLE
Check: python -m tools.wsm5_wrf461_column_oracle.compare_arms ORACLE --out arms.json
GPU: GPUWM_WRF_EXACT=1 python -m tools.wsm5_wrf461_column_oracle.run_gpu check-fixture tests/data/oracles/wsm5
GPU default: env -u GPUWM_WRF_EXACT python -m tools.wsm5_wrf461_column_oracle.run_gpu check-fixture tests/data/oracles/wsm5
Negative control: python -m tools.wsm5_wrf461_column_oracle.run_gpu plant tests/data/oracles/wsm5
Transport: python -m tools.wsm5_wrf461_column_oracle.run_gpu remap-probe
Focused suite: python -m pytest -q tests/test_wsm5.py

Every device command must use the local card queue. Column agreement is
implementation evidence. It establishes no observation-based forecast skill.

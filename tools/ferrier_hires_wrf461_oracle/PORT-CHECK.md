Ferrier acceptance requires a strict bitwise pass on every output word.

Run on a GPU box through its lane mutex:

```bash
GPUWM_WRF_EXACT=1 python -m pytest -q tests/test_fer_hires_port.py -m gpu
```

This grades the stock WRF -O0 referee on 72 original, 65 refute and 22 new
coverage columns. Nine level outputs and three surface outputs are compared
without a tolerance or a finite-value mask. The coverage columns deliberately
reach non-convergence fallbacks and can develop negative absolute temperature
and NaNs. Their strict output words are still compared. The first two steps
were bitwise equal before fixing later-step MAX/MIN NaN operand choices.

Run the ordinary arithmetic mode with run_woof_ferrier.py and report its
compare_ferrier.py result separately. Default arithmetic allows contraction
and is not bitwise equal to the -O0 referee.

Run check_port.py with the committed fixture directory, a directory containing
strict/{lane,refute,coverage} and default/{lane,refute,coverage}, and the real
forecast receipt. A missing result fails. It plants one bit in the first th
word of the actual strict candidate and requires the comparator to catch it.
The first recorded check failed with all GPU outputs and the forecast absent.

```bash
python tools/ferrier_hires_wrf461_oracle/check_port.py \
  tests/data/oracles/ferrier_hires LANE --forecast LANE/forecast.json \
  --out LANE/mechanical.json
```

The forecast receipt records a completed ordinary CLI run with dx=12000,
mp_physics=5, bl_pbl_physics=2 and sf_sfclay_physics=2. It must carry full-field
hashes and the SHA-256 of the Rust-rendered sheet. This is an implementation
check, not a stock-WRF trajectory match or a measured forecast-skill result.

The new coverage generator reaches WRF lines 2213, 2307-2308, 2621-2622,
2685-2686 and 3292. Line 1627 remains unreachable: RFmax=1.1**40 caps RimeF1
before XRF=10.492*log(RimeF1); binary32 XRF is 39.9997863770 (word 421FFFC8), below 40, so INT(XRF) cannot be
Nrime. The rain table and init scalars use WRF arithmetic order. CBFR comes
from init DTPH; call-dependent rates come from DT.

Modern Ferrier-Aligo (5) carries total ice in qi and no qs. The RRTMG wrappers
remap that total ice outside ICLOUD. cal_cldfra1 uses qc+qi and qi/(qc+qi),
with ice weight one below -40 C. The F_ICE_PHY arm with qs is the old Eta
package (95), which remains refused at admission. Cross-scheme nests are
refused by name because total-ice and rime-history closures are not ported.

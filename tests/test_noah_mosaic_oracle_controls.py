"""Independent CPU controls proving D1/D2 from actual WRF receipts.

D1 accumulation runs inside the reverse tile loop. Its defective index may
read a lower tile's pre-step state, so only the time-aware control matches WRF.
The naive final-output-only control must differ. Corrected port reduction uses
final tile outputs at 4*t+ns, in WRF's reverse tile accumulation order.
"""
from pathlib import Path
import numpy as np
from gpuwm.verify.noah_mosaic_oracle import (
    load, soil_reduction, wrf_defective_soil_reduction, weighted_twin_increments,
)

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/noah_mosaic"


def test_d1_control():
    different = 0
    final_only_failures = 0
    for family in ("base", "cats", "lcz"):
        for fixture in load(ROOT / family).values():
            mask = (fixture["xland_in"] < np.float32(1.5)) & (fixture["xice_in"] < np.float32(.5))
            mc = int(fixture["mosaic_cat"])
            fractions = fixture["landusef2_in"][:, :mc, :]
            for field in ("tslb", "smois", "sh2o"):
                before = fixture[field + "_mosaic_in"]
                after = fixture[field + "_mosaic_out"]
                actual = fixture[field + "_out"]
                defective = wrf_defective_soil_reduction(before, after, fractions)
                assert np.array_equal(defective.transpose(1, 0, 2)[:, mask].view(np.uint32),
                                      actual.transpose(1, 0, 2)[:, mask].view(np.uint32))
                corrected = soil_reduction(after, fractions)
                different += np.count_nonzero(corrected.view(np.uint32) != defective.view(np.uint32))
                final_only = wrf_defective_soil_reduction(after, after, fractions)
                final_only_failures += np.count_nonzero(final_only.view(np.uint32) != defective.view(np.uint32))
    assert different > 0, "D1 control needs distinct soil states to expose the wrong layer index"
    assert final_only_failures > 0, "D1 final-output-only control must expose the in-loop state dependency"


def test_d2_control():
    multi = load(ROOT / "base")["v1_s1"]
    raw = load(ROOT / "twins")
    twins = [raw[f"v{t+8}_s1"] for t in range(3)]
    # Non-glacial columns, no water or sea ice, RDLAI2D false.
    subset = np.r_[0:14, 28:42]
    f = multi["landusef2_in"][:, :3, :]
    different = 0
    for field in ("sfcrunoff", "udrunoff", "potevp", "acsnom", "snopcx"):
        control = multi[field + "_in"].copy()
        for t in range(2, -1, -1):
            value = twins[t][field + "_out"]
            control = control - (-value) if field == "snopcx" else control + value
        assert np.array_equal(control[subset].view(np.uint32),
                              multi[field + "_out"][subset].view(np.uint32)), field
        weighted = weighted_twin_increments(twins, f, field)
        different += np.count_nonzero(weighted[subset].view(np.uint32) != control[subset].view(np.uint32))
    assert different > 0, "D2 control needs nonzero increments to expose duplicated cell accumulation"
    dt = np.float32(multi["dt"])
    frozen = multi["t3d_in"][:, 0, :] <= np.float32(273.15)
    increment = np.where(frozen, (multi["rainbl_in"] / dt) * dt, np.float32(0))
    control = multi["acsnow_in"].copy()
    for t in range(2, -1, -1):
        control = control + increment
        assert np.array_equal(twins[t]["acsnow_out"][subset].view(np.uint32),
                              increment[subset].view(np.uint32))
    assert np.array_equal(control[subset].view(np.uint32), multi["acsnow_out"][subset].view(np.uint32))


def test_all_step_increment_controls():
    for family in ("base", "cats", "lcz", "twins"):
        for name, fixture in load(ROOT/family).items():
            for field in ("sfcrunoff", "udrunoff", "potevp", "acsnom", "snopcx", "acsnow"):
                control = fixture[field + "_in"].copy()
                inc = fixture["increment_" + field]
                for t in range(int(fixture["mosaic_cat"])-1, -1, -1):
                    control = control - (-inc[:, t, :]) if field == "snopcx" else control + inc[:, t, :]
                assert np.array_equal(control.view(np.uint32), fixture[field + "_out"].view(np.uint32)), (family, name, field)

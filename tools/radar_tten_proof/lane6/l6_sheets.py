"""Lane 6 sheets and scores from the cycled arms on box L (CPU only, niced).

    python l6_sheets.py ARM [ARM ...]

Rows (all the model's own REFL_10CM, one product):
    21Z  end of the last observed (forced) leg, the 21Z prior
    f01  22Z, f02  23Z, f03  00Z  from the 21Z analysis
Columns: MRMS | WOOF DA member 0 | WOOF DA mean (probability-matched mean of
the 32 member composites) | WOOF no DA (the 'off' arm's unanalysed control) | HRRR (21Z cycle).

Every weather panel is drawn by rw_compare (the Rust renderer) and cut apart
by the da-tune harness's own cutter; only labels are drawn here.
Scores: coverage bias at 15/25/35/45 dBZ over MRMS-covered cells (model cells
over MRMS cells), FSS 35 dBZ at 27 km, 35 dBZ object count ratio, spurious
echo fraction (>= 20 dBZ where MRMS < 5 dBZ), for member 0, the PMM and the
member mean of each statistic.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/work/da-l6/cycled")
REFS = Path("/work/da-l6/refs-2100Z")
OUT = Path("/work/da-l6/sheets")
os.environ.setdefault("DA_TUNE_RW_COMPARE",
                      "/work/da-e/engine/tools/rustwx/target/release/rw_compare")
sys.path.insert(0, "/work/da-l6/harness/tools")
from da_tune import pics, sheets  # noqa: E402
from da_tune.common import fss  # noqa: E402

#: (title, frame prefix, MRMS hour index).  Free-forecast frames are the
#: driver 2-minute history captures (seconds from the 18Z start).
ROWS = (("21Z prior: end of the last observed leg", "leg02", 0),
        ("analysis + 2 min (21:02Z, first model step)", "history000000010920t1", 0),
        ("f01 from the 21Z analysis", "history000000014400t1", 1),
        ("f02 from the 21Z analysis", "history000000018000t1", 2),
        ("f03 from the 21Z analysis", "history000000021600t1", 3))
VALID = ("2026-10-01 21Z", "2026-10-01 21:02Z", "2026-10-01 22Z", "2026-10-01 23Z",
         "2026-10-02 00Z")
CYCLE = "2026100121"
THRESHOLDS = (15, 25, 35, 45)


def members(arm: str, leg: str):
    comp = ROOT / arm / "da" / "composites"
    files = sorted((p for p in comp.glob(f"{leg}_*.npz")
                    if p.stem.rsplit("_", 1)[1].isdigit()),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return comp, files


def frame(path: Path) -> np.ndarray:
    with np.load(path) as z:
        if str(z["refl_product"]) != "native_refl_10cm":
            raise ValueError(f"{path} carries {z['refl_product']}, not native REFL_10CM")
        return np.asarray(z["refl_colmax"], np.float32)


def pmm(fields: list) -> np.ndarray:
    """Probability-matched mean: the ensemble mean's spatial ranking, the
    pooled member amplitude distribution."""
    stack = np.stack(fields)
    mean = stack.mean(axis=0)
    pooled = np.sort(stack.ravel())[::-1][::len(fields)]
    order = np.argsort(mean.ravel())[::-1]
    out = np.empty(mean.size, np.float32)
    out[order] = pooled[:mean.size]
    return out.reshape(mean.shape)


def write_like(template: Path, target: Path, field, title: str) -> Path:
    """A copy of ``template`` (optionally with a new composite field), its
    valid time floored to the hour: MRMS and the reference model are matched
    on whole hours, so the first model step after an analysis is drawn at
    the analysis hour (the label says when it was taken)."""
    import netCDF4
    shutil.copyfile(template, target)
    with netCDF4.Dataset(target, "a") as d:
        if field is not None:
            d.variables["REFL_10CM"][0, 0, :, :] = field
        d.setncattr("TITLE", title)
        stamp = d.variables["Times"][0].tobytes().decode()
        if stamp[14:19] != "00:00":
            fixed = stamp[:14] + "00:00"
            d.variables["Times"][0] = np.frombuffer(fixed.encode(), dtype="S1")
            if "XTIME" in d.variables:
                minutes = float(d.variables["XTIME"][0])
                d.variables["XTIME"][0] = minutes - (minutes % 60.0)
    return target


def scores(model: np.ndarray, ref: np.ndarray, covered: np.ndarray) -> dict:
    from scipy import ndimage
    # MRMS on the grid is NaN for both "no echo" and "no radar"; inside
    # radar coverage a NaN is an observation of no echo
    ok = covered
    m = np.where(ok & np.isfinite(model), model, -99.0)
    o = np.where(ok & np.isfinite(ref), ref, -99.0)
    out = {}
    for th in THRESHOLDS:
        n_o = int((o >= th).sum())
        out[f"bias{th}"] = float((m >= th).sum() / n_o) if n_o else float("nan")
    fm, fo = (m >= 35).astype(float), (o >= 35).astype(float)
    out["fss35_27km"] = fss(fm, fo, 3)
    n_m = ndimage.label(fm)[1]
    n_o = ndimage.label(fo)[1]
    out["objects35_ratio"] = float(n_m / n_o) if n_o else float("nan")
    clear = ok & (o < 5.0)
    out["spurious20"] = float((m[clear] >= 20).mean()) if clear.any() else float("nan")
    return out


def run(arms: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    refdir = OUT / "rwref"
    refdir.mkdir(exist_ok=True)
    table = {}
    for arm in arms:
        blocks, table[arm] = [], {}
        for (title, leg, k), valid in zip(ROWS, VALID):
            comp, files = members(arm, leg)
            if not files:
                print(f"{arm} {leg}: no member frames", flush=True)
                continue
            fields = [frame(p) for p in files]
            ref = np.load(REFS / f"mrms_t{k}.npy")
            cov = np.load(REFS / f"covered_t{k}.npy").astype(bool)
            mean = pmm(fields)
            row = {"pmm": scores(mean, ref, cov),
                   "member0": scores(fields[0], ref, cov)}
            per = [scores(f, ref, cov) for f in fields]
            row["members_mean"] = {key: float(np.nanmean([p[key] for p in per]))
                                   for key in per[0]}
            noda = ROOT / "off" / "da" / "composites" / f"{leg}_control.npz"
            if noda.exists():
                row["noda"] = scores(frame(noda), ref, cov)
            row["n_members"] = len(fields)
            table[arm][leg] = row
            # panels
            m0 = comp / f"wrfout_{leg}_0.nc"
            pm = write_like(m0, work / f"wrfout_{arm}_{leg}_pmm.nc", mean,
                            f"WOOF DA mean (PMM of {len(fields)} members), {arm}")
            noda_src = ROOT / "off" / "da" / "composites" / f"wrfout_{leg}_control.nc"
            cols = [("member", write_like(m0, work / f"wrfout_{arm}_{leg}_m0.nc", None,
                                           f"WOOF DA member 0, {arm}"), f"{arm}: DA member 0"),
                    ("mean", pm, f"{arm}: DA mean (PMM)"),
                    ("noda", write_like(noda_src, work / f"wrfout_{arm}_{leg}_noda.nc", None,
                                         "WOOF no DA") if noda_src.exists() else noda_src,
                     "no DA (unanalysed control)")]
            drawn, header, refs = {}, None, None
            for col, path, label in cols:
                if not Path(path).exists():
                    continue
                h, (woof, mrms_p, hrrr_p) = pics._panels(pics._compare(
                    Path(path), f"{label}, {title}", CYCLE, work, refdir))
                drawn[col] = woof
                if refs is None:
                    header, refs = h, (mrms_p, hrrr_p)
            if refs is None:
                continue
            size = refs[0].size
            panels = [refs[0]] + [drawn.get(c) or sheets.placeholder(size, "not available")
                                  for c, _p, _l in cols] + [refs[1]]
            from PIL import Image
            width = sum(p.width for p in panels)
            b35 = row["pmm"]["bias35"]
            strip = sheets.text_strip(
                width, f"lane 6 {arm}  |  {title}, valid {valid}  |  MRMS | DA member 0 | "
                f"DA mean (PMM) | no DA | HRRR 21Z  |  composite reflectivity, WOOF = model "
                f"REFL_10CM  |  35 dBZ coverage bias (PMM) {b35:.2f}")
            height = max(p.height for p in panels)
            block = Image.new("RGB", (width, strip.height + header.height + height), (255, 255, 255))
            block.paste(strip, (0, 0))
            block.paste(header, (0, strip.height))
            x = 0
            for p in panels:
                block.paste(p, (x, strip.height + header.height))
                x += p.width
            blocks.append(block)
        if blocks:
            from PIL import Image
            width = max(b.width for b in blocks)
            canvas = Image.new("RGB", (width, sum(b.height for b in blocks)), (255, 255, 255))
            y = 0
            for b in blocks:
                canvas.paste(b, (0, y))
                y += b.height
            target = OUT / f"S1-lane6-{arm}.png"
            canvas.save(target, optimize=True)
            print(f"sheet {target}", flush=True)
    previous = {}
    if (OUT / "scores.json").exists():
        previous = json.loads((OUT / "scores.json").read_text())
    previous.update(table)
    (OUT / "scores.json").write_text(json.dumps(previous, indent=1))
    print(json.dumps({a: {l: {"pmm_bias35": r["pmm"]["bias35"], "pmm_fss35": r["pmm"]["fss35_27km"],
                              "noda_bias35": r.get("noda", {}).get("bias35")}
                          for l, r in t.items()} for a, t in table.items()}, indent=1))


if __name__ == "__main__":
    run(sys.argv[1:])

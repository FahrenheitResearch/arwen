"""Write the column inputs for the revised MM5 surface-layer oracle.

    python tools/sfclayrev_wrf461_oracle/make_inputs.py [OUT_DIR]

Writes ``sfclayrev-inputs.hex`` (what ``run_sfclayrev`` reads: a case id and
21 float32 words in hex) and ``sfclayrev-cases.csv`` (the same columns in
decimal with a regime label, for people).  Deterministic: a fixed seed and
fixed regime recipes, so rerunning it reproduces the committed bytes.

The columns are chosen to reach every branch of ``sf_sfclayrev_run`` rather
than to look like one forecast: daytime convective and night stable land,
calm very stable (``br > 250``), the previously-unstable clamp that makes
``br`` exactly zero (regime 3), open ocean, lakes, hurricane winds, snow and
sea ice, high terrain, hot desert, tall roughness on thin layers (the
``0.9*gz1oz0`` limiters and the full-function branch past the 1000-entry
tables), ``ust < 0.001``, ``br < -250``, zero wind, dry air, zero ``ust``,
``qsfc <= 0`` on land, ``-0.0`` inputs and one-ULP temperature contrasts.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

FIELDS = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "pblh", "mavail",
          "xland", "lakemask", "dx", "znt", "ust", "ustm", "mol", "hfx", "qfx",
          "qsfc", "zol")

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "sfclayrev"


def _f32(x) -> np.float32:
    return np.float32(x)


def _column(label, rng, **kw):
    """One column; anything not given gets a plausible jittered default."""
    t = kw.get("t", rng.uniform(280.0, 300.0))
    p = kw.get("p", rng.uniform(95000.0, 100500.0))
    dz = kw.get("dz8w", rng.uniform(20.0, 60.0))
    rho = p / (287.0 * t)
    col = {
        "u": rng.uniform(1.0, 8.0), "v": rng.uniform(-4.0, 4.0), "t": t,
        "qv": rng.uniform(0.003, 0.012), "p": p, "dz8w": dz,
        "psfc": p + 0.5 * dz * rho * 9.81, "tsk": t, "pblh": rng.uniform(200.0, 1500.0),
        "mavail": rng.uniform(0.3, 1.0), "xland": 1.0, "lakemask": 0.0,
        "dx": 3000.0, "znt": rng.uniform(0.02, 0.4), "ust": rng.uniform(0.1, 0.5),
        "ustm": rng.uniform(0.1, 0.5), "mol": 0.0, "hfx": 0.0, "qfx": 0.0,
        "qsfc": 0.0, "zol": 0.0,
    }
    for key, value in kw.items():
        col[key] = value
    if "psfc" not in kw:
        rho = col["p"] / (287.0 * col["t"])
        col["psfc"] = col["p"] + 0.5 * col["dz8w"] * rho * 9.81
    return label, {k: _f32(col[k]) for k in FIELDS}


def build_columns():
    rng = np.random.default_rng(20261007)
    cols = []
    add = lambda label, **kw: cols.append(_column(label, rng, **kw))
    dxs = (1000.0, 3000.0, 12000.0, 27000.0)

    for i in range(8):           # daytime convective land
        t = rng.uniform(293.0, 310.0)
        add("convective-land", t=t, tsk=t + rng.uniform(3.0, 15.0),
            u=rng.uniform(1.0, 9.0), mol=-rng.uniform(0.05, 0.6),
            hfx=rng.uniform(80.0, 420.0), qfx=rng.uniform(3e-5, 2e-4),
            qsfc=(0.0 if i % 3 == 0 else rng.uniform(0.008, 0.02)),
            pblh=rng.uniform(800.0, 2800.0), dx=dxs[i % 4],
            ust=rng.uniform(0.2, 0.7), ustm=rng.uniform(0.15, 0.6))
    for i in range(8):           # night stable land
        t = rng.uniform(270.0, 295.0)
        add("stable-night-land", t=t, tsk=t - rng.uniform(0.5, 8.0),
            u=rng.uniform(0.5, 6.0), v=rng.uniform(-2.0, 2.0),
            mol=rng.uniform(0.02, 0.6), hfx=-rng.uniform(5.0, 60.0),
            qfx=rng.uniform(-1e-5, 1e-5), qsfc=rng.uniform(0.003, 0.012),
            pblh=rng.uniform(40.0, 300.0), dx=dxs[(i + 1) % 4],
            ust=rng.uniform(0.03, 0.3))
    for i in range(4):           # calm, very stable: br clipped at 250
        t = rng.uniform(255.0, 285.0)
        add("very-stable-calm", t=t, tsk=t - rng.uniform(8.0, 15.0),
            u=rng.uniform(0.0, 0.08), v=rng.uniform(0.0, 0.05),
            mol=rng.uniform(0.2, 1.0), hfx=-rng.uniform(1.0, 20.0),
            dz8w=rng.uniform(40.0, 90.0), dx=1000.0, ust=rng.uniform(0.01, 0.1))
    for i in range(4):           # previously unstable, now stable: br -> 0
        t = rng.uniform(285.0, 300.0)
        add("prev-unstable-br-zero", t=t, tsk=t - rng.uniform(0.5, 3.0),
            mol=-rng.uniform(0.01, 0.2), hfx=rng.uniform(5.0, 50.0),
            dx=dxs[i % 4], xland=(2.0 if i == 3 else 1.0),
            znt=(2e-4 if i == 3 else rng.uniform(0.02, 0.3)))
    for i in range(8):           # open ocean, unstable
        t = rng.uniform(280.0, 300.0)
        add("ocean-unstable", t=t, tsk=t + rng.uniform(0.3, 4.0), xland=2.0,
            znt=rng.uniform(1e-5, 2e-3), u=rng.uniform(2.0, 18.0),
            v=rng.uniform(-8.0, 8.0), mavail=1.0, ust=rng.uniform(0.08, 0.6),
            qsfc=rng.uniform(0.005, 0.02), dx=dxs[i % 4],
            hfx=rng.uniform(0.0, 60.0), qfx=rng.uniform(0.0, 1e-4),
            mol=-rng.uniform(0.0, 0.2))
    for i in range(4):           # ocean, warm air over cold water
        t = rng.uniform(280.0, 295.0)
        add("ocean-stable", t=t, tsk=t - rng.uniform(0.5, 5.0), xland=2.0,
            znt=rng.uniform(5e-5, 1e-3), u=rng.uniform(2.0, 12.0), mavail=1.0,
            ust=rng.uniform(0.05, 0.4), mol=rng.uniform(0.01, 0.3),
            dx=dxs[(i + 2) % 4])
    for i in range(4):           # hurricane-force winds over warm sea
        add("hurricane-ocean", t=rng.uniform(298.0, 301.0), tsk=302.5,
            xland=2.0, znt=rng.uniform(1e-3, 2.8e-3),
            u=rng.uniform(25.0, 60.0), v=rng.uniform(-30.0, 30.0),
            qv=rng.uniform(0.016, 0.02), mavail=1.0,
            ust=rng.uniform(1.2, 3.2), ustm=rng.uniform(1.2, 3.0),
            mol=-rng.uniform(0.0, 0.05), dx=dxs[i % 4])
    for i in range(4):           # lakes (no salinity factor)
        t = rng.uniform(275.0, 295.0)
        add("lake", t=t, tsk=t + rng.uniform(-3.0, 3.0), xland=2.0,
            lakemask=1.0, znt=rng.uniform(1e-4, 1e-3), mavail=1.0,
            ust=rng.uniform(0.05, 0.3), dx=dxs[i % 4])
    for i in range(6):           # snow and ice-covered land
        t = rng.uniform(240.0, 270.0)
        add("snow-ice-land", t=t, tsk=t + rng.uniform(-6.0, 3.0),
            qv=rng.uniform(1e-4, 2e-3), znt=rng.uniform(5e-4, 1e-2),
            mavail=1.0, qsfc=rng.uniform(1e-4, 2e-3), dx=dxs[i % 4],
            mol=(rng.uniform(0.02, 0.3) if i % 2 == 0 else -rng.uniform(0.01, 0.1)))
    for i in range(2):           # sea ice handed to the layer as land
        add("sea-ice-as-land", t=rng.uniform(245.0, 262.0), tsk=258.0,
            znt=1e-3, qv=5e-4, mavail=1.0, dx=12000.0, qsfc=6e-4)
    for i in range(6):           # high terrain
        p = rng.uniform(55000.0, 75000.0)
        t = rng.uniform(260.0, 295.0)
        add("high-terrain", p=p, t=t, tsk=t + rng.uniform(-6.0, 12.0),
            qv=rng.uniform(5e-4, 6e-3), znt=rng.uniform(0.05, 1.0),
            pblh=rng.uniform(100.0, 3000.0), dx=dxs[i % 4],
            hfx=rng.uniform(-30.0, 300.0), mol=rng.uniform(-0.3, 0.3))
    for i in range(4):           # hot dry desert
        add("hot-desert", t=rng.uniform(305.0, 315.0), tsk=rng.uniform(328.0, 342.0),
            qv=rng.uniform(1e-3, 4e-3), mavail=rng.uniform(0.01, 0.05),
            znt=rng.uniform(0.01, 0.05), hfx=rng.uniform(300.0, 500.0),
            mol=-rng.uniform(0.3, 1.0), pblh=rng.uniform(2500.0, 4500.0),
            dx=dxs[i % 4], ust=rng.uniform(0.3, 0.6))
    for i in range(4):           # tall roughness, thin lowest layer
        t = rng.uniform(285.0, 305.0)
        add("rough-thin-layer", t=t,
            tsk=t + (rng.uniform(4.0, 12.0) if i < 2 else -rng.uniform(3.0, 8.0)),
            znt=rng.uniform(1.0, 2.5), dz8w=rng.uniform(8.0, 20.0),
            u=rng.uniform(0.5, 4.0), mol=(-0.2 if i < 2 else 0.4),
            ust=rng.uniform(0.2, 0.8), dx=dxs[i % 4])
    for i in range(2):           # stable thin layer: |zol*10/za| past 10
        t = rng.uniform(270.0, 285.0)
        add("thin-layer-full-function", t=t, tsk=t - rng.uniform(6.0, 12.0),
            dz8w=rng.uniform(4.0, 7.0), u=rng.uniform(0.1, 0.5),
            znt=rng.uniform(0.01, 0.1), mol=0.5, dx=1000.0, ust=0.05)
    add("ust-below-1e-3-land", ust=5e-4, tsk=300.0, t=292.0, mol=-0.1, hfx=50.0)
    add("ust-below-1e-3-ocean", ust=2e-4, xland=2.0, znt=1e-4, tsk=295.0,
        t=290.0, mavail=1.0)
    add("br-below-minus-250", u=0.0, v=0.0, t=295.0, tsk=330.0, hfx=0.0,
        qfx=0.0, pblh=1000.0, dz8w=80.0, dx=1000.0)
    add("br-below-minus-250-b", u=0.05, v=0.0, t=300.0, tsk=340.0, hfx=-5.0,
        qfx=0.0, pblh=500.0, dz8w=60.0, dx=1000.0, ust=0.3)
    add("zero-wind-stable", u=0.0, v=0.0, t=285.0, tsk=280.0, mol=0.2, dx=1000.0)
    add("zero-wind-ocean", u=0.0, v=0.0, t=290.0, tsk=292.0, xland=2.0,
        znt=1e-4, mavail=1.0, dx=12000.0)
    add("qsfc-zero-land", qsfc=0.0, tsk=303.0, t=298.0)
    add("qsfc-negative-land", qsfc=-1e-4, tsk=296.0, t=297.0)
    add("fluxc-floor-negative-fluxes", hfx=-200.0, qfx=-3e-4, tsk=301.0, t=298.0)
    add("ocean-huge-ust", xland=2.0, znt=2.8e-3, ust=6.0, u=40.0, tsk=300.0,
        t=299.0, mavail=1.0)
    add("ocean-tiny-ust", xland=2.0, znt=1e-6, ust=1e-3, u=0.3, tsk=291.0,
        t=290.0, mavail=1.0)
    add("dry-air", qv=0.0, qsfc=0.0, tsk=305.0, t=300.0)
    add("zero-ust-land", ust=0.0, ustm=0.0, tsk=280.0, t=284.0, mol=0.1)
    add("tiny-roughness-land", znt=1e-6, tsk=290.0, t=288.0)
    add("negative-zero-v", v=-0.0, u=3.0, tsk=289.0, t=291.0)
    t1 = np.float32(290.0)
    add("one-ulp-warmer-ground", t=t1, tsk=np.nextafter(t1, np.float32(400.0)),
        p=100000.0, psfc=100000.0, qv=1e-8, qsfc=1e-8)
    add("one-ulp-colder-ground", t=t1, tsk=np.nextafter(t1, np.float32(0.0)),
        p=100000.0, psfc=100000.0, qv=1e-8, qsfc=1e-8)
    add("pressure-equal-psfc", p=98000.0, psfc=98000.0, t=290.0, tsk=290.0)

    for i in range(24):          # mixed sweep
        water = i % 3 == 0
        t = rng.uniform(250.0, 310.0)
        add("mixed-sweep", t=t, tsk=t + rng.uniform(-10.0, 15.0),
            xland=2.0 if water else 1.0,
            lakemask=1.0 if (water and i % 2 == 0) else 0.0,
            znt=rng.uniform(1e-5, 2e-3) if water else 10 ** rng.uniform(-3.0, 0.3),
            u=rng.uniform(-15.0, 15.0), v=rng.uniform(-15.0, 15.0),
            p=rng.uniform(60000.0, 102000.0), qv=rng.uniform(0.0, 0.02),
            dz8w=rng.uniform(10.0, 120.0), dx=dxs[i % 4],
            ust=10 ** rng.uniform(-2.5, 0.3), ustm=10 ** rng.uniform(-2.5, 0.3),
            mol=rng.uniform(-0.8, 0.8), hfx=rng.uniform(-100.0, 400.0),
            qfx=rng.uniform(-5e-5, 3e-4), qsfc=rng.uniform(0.0, 0.02),
            pblh=rng.uniform(30.0, 3500.0), mavail=rng.uniform(0.0, 1.0))
    return cols


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = build_columns()
    with open(out_dir / "sfclayrev-inputs.hex", "w", newline="\n") as fh:
        fh.write("# case " + " ".join(FIELDS) + "\n")
        for case, (_, col) in enumerate(cols, start=1):
            words = " ".join(f"{int(np.float32(col[k]).view(np.uint32)):08X}" for k in FIELDS)
            fh.write(f"{case} {words}\n")
    with open(out_dir / "sfclayrev-cases.csv", "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("case", "label") + FIELDS)
        for case, (label, col) in enumerate(cols, start=1):
            writer.writerow((case, label) + tuple(repr(float(col[k])) for k in FIELDS))
    print(f"{len(cols)} columns -> {out_dir}")


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)

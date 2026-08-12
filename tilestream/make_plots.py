"""Render the measured out-of-core results as figures.

Every number below is MEASURED, not projected, and each figure names the box it
came from.  Where a value is derived rather than observed the caption says so.

Deliberately plain in the style of the CM1 and MPAS-A scaling figures this work
is meant to sit alongside: a reference line for the ideal, one idea per panel,
and units on every axis.  The one departure is colour, used only to separate
lanes that must not be confused (resident vs paging vs streamed).

    python make_plots.py [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

INK = "#16202c"
MUTED = "#6b7785"
GRID = "#dfe4ea"
RESIDENT = "#1f6f8b"     # what fits today
PAGING = "#c1442e"       # what happens when you overrun it
STREAMED = "#2e7d52"     # what this work adds
ACCENT = "#b8860b"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def finish(fig, ax, title, subtitle, source, path):
    # Subtitle sits between the title and the axes, so the title needs enough
    # pad to clear both it and any wrapped second line.
    ax.set_title(title, pad=34, loc="left")
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5,
            color=MUTED, va="bottom", wrap=True)
    fig.text(0.01, 0.012, source, fontsize=7.2, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------- 1. the curve
def plot_outofcore(outdir):
    """The one figure that makes the case.

    Three regimes on one axis: what fits, what happens 2.5% past what fits, and
    what streaming does instead.  The paging points are the honest alternative
    to this work -- not "run a smaller domain" but "fall off a cliff".
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    res_x = [250, 1536, 1950]
    res_y = [3.685, 3.71, 3.707]
    pag_x = [1950, 2000, 2040]
    pag_y = [3.707, 16.436, 33.688]
    str_x = [1950, 3276]
    str_y = [4.084, 3.886]

    ax.plot(res_x, res_y, "o-", color=RESIDENT, lw=2, ms=6,
            label="resident in VRAM (today)")
    ax.plot(pag_x, pag_y, "s--", color=PAGING, lw=2, ms=6,
            label="past VRAM, driver paging")
    ax.plot(str_x, str_y, "D-", color=STREAMED, lw=2.4, ms=7,
            label="tiled, streamed from host RAM")

    ax.axvline(1950, color=MUTED, lw=0.9, ls=":")
    ax.text(1960, 45, "largest domain\nthat fits (1950²)", fontsize=8,
            color=MUTED, va="top")

    ax.annotate("2000² — only 2.5% bigger,\n4.4× slower", xy=(2000, 16.436),
                xytext=(2180, 22), fontsize=8.5, color=PAGING,
                arrowprops=dict(arrowstyle="->", color=PAGING, lw=1.1))
    ax.annotate("3276² — 2.5× the card's memory,\nand FASTER per cell than 1950²",
                xy=(3276, 3.886), xytext=(2150, 6.6), fontsize=8.5,
                color=STREAMED,
                arrowprops=dict(arrowstyle="->", color=STREAMED, lw=1.1))

    ax.set_yscale("log")
    ax.set_xlabel("domain side (grid points, nz = 49)")
    ax.set_ylabel("nanoseconds per useful cell per step")
    ax.set_ylim(3, 60)
    ax.set_xlim(0, 3600)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    finish(fig, ax, "Streaming beats the memory wall — and the wall is a cliff",
           "Lower is faster. Red is what ArWen does today when a domain just exceeds VRAM.",
           "MEASURED — RTX 5090, WSL2, dry dynamics, 49 levels. Every point carries a "
           "SHA-256 that matches the monolithic run.",
           outdir / "01-out-of-core-curve.png")


# ---------------------------------------------------------------- 2. the ladder
def plot_ladder(outdir):
    """Where the 4.7% actually goes, isolated one term at a time."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))

    labels = ["monolithic\nresident", "tiled,\nstore in VRAM",
              "tiled, store in\nhost RAM", "3276² —\nexceeds VRAM"]
    vals = [3.712, 3.895, 4.084, 3.886]
    cols = [RESIDENT, ACCENT, ACCENT, STREAMED]

    bars = ax.bar(labels, vals, color=cols, width=0.6)
    ax.axhline(3.712, color=MUTED, ls="--", lw=1)
    ax.text(3.44, 3.74, "baseline", fontsize=8, color=MUTED)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.045, f"{v:.3f}",
                ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v + 0.135,
                f"{v / 3.712:.3f}×", ha="center", fontsize=8, color=MUTED)

    ax.annotate("", xy=(1, 3.90), xytext=(0, 3.90),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.text(0.5, 3.93, "cost of tiling\n(halo + launches)", ha="center",
            fontsize=8, color=MUTED)
    ax.annotate("", xy=(2, 4.13), xytext=(1, 4.13),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.text(1.5, 4.16, "cost of PCIe", ha="center", fontsize=8, color=MUTED)

    ax.set_ylabel("ns per useful cell per step")
    ax.set_ylim(3.4, 4.35)

    finish(fig, ax, "Where the 4.7% goes",
           "Same domain, same seed, same SHA-256 across the first three bars.",
           "MEASURED — RTX 5090, WSL2, 1950²×49 dry for bars 1–3; bar 4 is 3276²×49, "
           "which cannot fit in the 32 GB card at all.",
           outdir / "02-cost-ladder.png")


# ---------------------------------------------------------------- 3. halo trap
def plot_halo(outdir):
    """Why the halo must come from theory, not from a passing test.

    The point of this figure is the halo=13 row: bit-exact for three steps,
    then silently divergent.  A short test certifies it AND it is faster.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    steps = [1, 2, 3, 5, 8, 12, 16, 24]
    series = {
        11: [7.8e-3, 4.4e-1, 4.9e-1, 7.3e-1, 7.2e-1, None, None, None],
        12: [1e-16, 2.1e-1, 3.7e-1, 7.0e-1, 7.0e-1, None, None, None],
        13: [1e-16, 1e-16, 1e-16, 3.8e-6, 4.9e-1, 7.1e-1, 7.3e-1, 7.3e-1],
        14: [1e-16] * 8,
        16: [1e-16] * 8,
    }
    colors = {11: PAGING, 12: "#d4763f", 13: ACCENT, 14: STREAMED, 16: RESIDENT}

    for halo, ys in series.items():
        xs = [s for s, y in zip(steps, ys) if y is not None]
        vs = [y for y in ys if y is not None]
        lw = 2.6 if halo == 13 else 1.6
        ax.plot(xs, vs, "o-", color=colors[halo], lw=lw, ms=5,
                label=f"halo = {halo}" + ("  ← the trap" if halo == 13 else ""))

    ax.axhline(1e-16, color=MUTED, ls=":", lw=0.9)
    ax.text(24.4, 1.4e-16, "bit-exact", fontsize=8, color=MUTED, va="center")
    ax.annotate("passes every short test,\nthen diverges", xy=(8, 4.9e-1),
                xytext=(9.5, 2e-3), fontsize=8.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))

    ax.set_yscale("log")
    ax.set_xlabel("steps between tile gathers (N)")
    ax.set_ylabel("max |tiled − monolithic|")
    ax.set_ylim(3e-17, 5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    finish(fig, ax, "The halo cannot be tuned on a short test",
           "Two probes concluded 12 was the minimum. Both were wrong — and too-narrow is FASTER.",
           "MEASURED — RTX 5090, WSL2, 192²×49 periodic. Minimum passing halo also "
           "differs by architecture: 14 on Ada vs 13 on Blackwell at N=3.",
           outdir / "03-halo-trap.png")


# ------------------------------------------------------------- 4. managed memory
def plot_managed(outdir):
    """Is there a zero-effort alternative to explicit tiling?  Measured: no."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    labels = ["device-resident\n(reference)", "managed,\nFITS in VRAM",
              "explicit pinned\ntransfer", "managed 1.49×,\nchunked prefetch",
              "managed 1.49×,\n+advice", "managed 1.49×,\nnaive",
              "managed 1.49×,\n+full prefetch"]
    vals = [957.8, 957.7, 26.3, 11.3, 4.29, 3.06, 2.84]
    cols = [MUTED, MUTED, STREAMED, ACCENT, PAGING, PAGING, PAGING]

    bars = ax.barh(range(len(labels))[::-1], vals, color=cols, height=0.62)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels, fontsize=8.5)
    for b, v in zip(bars, vals):
        ax.text(v * 1.12, b.get_y() + b.get_height() / 2, f"{v:g}",
                va="center", fontsize=8.5, color=INK)

    ax.set_xscale("log")
    ax.set_xlabel("GB/s (log scale)")
    ax.set_xlim(1.5, 3000)

    finish(fig, ax, "There is no free lunch — the explicit path wins by 2.2×",
           "Managed that fits is perfect. Oversubscribed, even hand-tuned, it loses.",
           "MEASURED — RTX 4090, native Linux, NUMA-bound. Prefetching the whole "
           "oversubscribed array is WORSE than doing nothing.",
           outdir / "04-managed-memory.png")


# ------------------------------------------------------------------- 5. duplex
def plot_duplex(outdir):
    """Full duplex is real on Linux -- but only if you bind to the GPU's node."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9))

    labels = ["WSL2 / Windows\n(RTX 5090)", "Linux, WRONG\nNUMA node",
              "Linux, bound to\nGPU's NUMA node"]
    vals = [1.00, 0.84, 1.639]
    cols = [PAGING, PAGING, STREAMED]

    bars = ax.bar(labels, vals, color=cols, width=0.55)
    ax.axhline(1.0, color=MUTED, ls="--", lw=1)
    ax.text(2.42, 1.02, "no benefit", fontsize=8, color=MUTED)
    ax.axhline(1.5, color=MUTED, ls=":", lw=1)
    ax.text(2.42, 1.52, "real duplex", fontsize=8, color=MUTED)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.035, f"{v:.3f}×",
                ha="center", fontsize=10, color=INK, fontweight="bold")

    ax.set_ylabel("concurrent H2D+D2H ÷ best one-way")
    ax.set_ylim(0, 1.95)

    finish(fig, ax, "Simultaneous upload and download: a Windows limitation, not hardware",
           "Middle bar is the trap: unbound on a dual-socket box, Linux mimics Windows.",
           "MEASURED — 2 GiB pinned buffers, 15 reps. Same-direction control gives "
           "1.000× (serialised), which is what makes the 1.639× interpretable.",
           outdir / "05-duplex-numa.png")


# --------------------------------------------------------- 6. rings vs shadow
def plot_rings(outdir):
    """The capacity unlock: one store plus a ring, instead of two stores.

    The shadow scheme is correct but costs a whole second copy of the domain,
    and the host store is what bounds the largest runnable domain -- so the
    shadow halves it.  The ring keeps only the band another tile can read.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    # Dry carrier set, measured: 32.26 B/cell/store, nz = 49.
    per_cell, nz = 32.26, 49
    budgets = [b for b in range(8, 481, 2)]  # GiB of pinned host store

    def mcell(gib, mult):
        return gib * (1 << 30) / (per_cell * mult) / 1e6

    def edge(gib, mult):
        return (mcell(gib, mult) * 1e6 / nz) ** 0.5

    ax.plot(budgets, [mcell(b, 2.00) for b in budgets], color=PAGING, lw=2.2,
            label="shadow — two full stores (2.00×)")
    ax.plot(budgets, [mcell(b, 1.052) for b in budgets], color=STREAMED, lw=2.6,
            label="ring — one store + 5.2% (MEASURED)")

    for gib, note, tx in ((44.14, "this box\n(WSL2 pinned ceiling)", 240),
                          (100.0, "128 GB Linux box", 240)):
        ax.axvline(gib, color=MUTED, ls=":", lw=0.9)
        for mult, col in ((2.00, PAGING), (1.052, STREAMED)):
            ax.plot([gib], [mcell(gib, mult)], "o", color=col, ms=6)

    ax.annotate("44 GiB (this box):\n3,872² → 5,339²\n735 → 1,397 Mcell",
                xy=(44.14, 1397), xytext=(150, 3100), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("100 GiB (128 GB Linux box):\n5,828² → 8,035²\n1,664 → 3,164 Mcell",
                xy=(100, 3164), xytext=(150, 7600), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

    ax.set_xlabel("pinned host store (GiB)")
    ax.set_ylabel("largest domain (millions of cells, nz = 49)")
    ax.set_xlim(0, 480)
    ax.set_ylim(0, 16000)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    finish(fig, ax, "Halo rings give back 1.90× of domain, at identical results",
           "The host store is what bounds domain size, and the shadow scheme spent "
           "half of it on a second copy.",
           "MEASURED — ring cost 5.2% at 1950² tile 650 and 5.8% at 3276² tile 546, "
           "dry carrier set at 32.26 B/cell/store, nz=49. The gate demands identical "
           "SHA-256 digests from the ring and shadow paths. The full-physics carrier "
           "set costs more per cell, which shifts both curves down and leaves the "
           "1.90× unchanged.",
           outdir / "06-rings-vs-shadow.png")


# ------------------------------------------------------------- 7. geography
def plot_geography(outdir):
    """Why geography must be SLICED from the parent, never rebuilt per tile.

    Rebuilding re-centres the projection on each tile, so every tile but the
    middle one thinks it is somewhere else on Earth.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    km = [[1022.1, 615.3, 1022.1],    # top row of tiles
          [816.5, 13.6, 816.5],
          [1022.0, 615.3, 1022.0]]
    cor = [[12.665, 12.165, 12.665],
           [2.447, 0.165, 2.447],
           [20.587, 17.148, 20.587]]

    im = ax.imshow(km, cmap="OrRd", vmin=0, vmax=1100)
    for j in range(3):
        for i in range(3):
            dark = km[j][i] > 700
            ax.text(i, j - 0.13, f"{km[j][i]:,.0f} km", ha="center",
                    fontsize=11, fontweight="bold",
                    color="white" if dark else INK)
            ax.text(i, j + 0.17, f"Coriolis {cor[j][i]:.1f}% off", ha="center",
                    fontsize=8.5, color="white" if dark else MUTED)

    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("tile centre displaced (great-circle km)", fontsize=8.5)
    cb.outline.set_visible(False)

    finish(fig, ax, "Geography cannot be rebuilt per tile — 1,022 km of error",
           "Only the middle tile is nearly right, because it shares the parent's centre.",
           "MEASURED — real74_d01 geometry: 250×200 mass, dx=12 km, CONUS Lambert "
           "(29.5/49.5, ref 39.5N/98.5W), split 3×3 with halo 16. The earlier "
           "'472 km' figure was latitude-only displacement, not great-circle.",
           outdir / "07-geography-error.png")


# ------------------------------------------------- 8. vanilla vs streaming
def plot_vanilla(outdir):
    """The only comparison a user cares about: against ArWen as it ships.

    Same card, same physics, same answer to the bit -- the only difference is
    where the domain lives.  Drawn in kilometres at 1 km spacing, because
    "1808 vs 5712 cells" is a number about the benchmark and "does CONUS fit"
    is a number about the forecast.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    rows = [
        ("RTX 4090, 24 GB\n96 GiB host RAM", 1808, 5712, True),
        ("RTX 5090, 32 GB\n44 GiB host RAM (WSL2)", 1950, 3276, False),
    ]
    y = range(len(rows))
    h = 0.34
    for i, (_, van, strm, _meas) in enumerate(rows):
        ax.barh(i + h / 2, van, height=h, color=RESIDENT)
        ax.barh(i - h / 2, strm, height=h, color=STREAMED)
        ax.text(van + 90, i + h / 2, f"{van:,} km", va="center", fontsize=9,
                color=RESIDENT)
        ax.text(strm + 90, i - h / 2, f"{strm:,} km", va="center", fontsize=9,
                color=STREAMED, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()

    # CONUS is the line that decides whether this is a percentage or a
    # capability.
    ax.axvline(4500, color=ACCENT, ls="--", lw=1.4)
    ax.text(4560, -0.62, "CONUS ≈ 4,500 km", fontsize=8.5, color=ACCENT,
            va="center")

    ax.set_xlabel("largest square domain at 1 km spacing (km per side)")
    ax.set_xlim(0, 6900)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=RESIDENT, label="ArWen as it ships"),
        plt.Rectangle((0, 0), 1, 1, color=STREAMED, label="with tiled streaming"),
    ], frameon=False, fontsize=9, loc="lower right")

    finish(fig, ax, "CONUS at 1 km does not fit on a 4090 today. Streamed, it does.",
           "Same card, same code path, bit-identical answers — the domain just "
           "lives in host RAM instead of VRAM.",
           "MEASURED, dry dynamics. 4090: 1808² is the hard resident limit (1824² "
           "allocates, then OOMs on the first step's scratch); 5712² ran at 0.988× "
           "the resident cost per cell. The 5090 row is WSL2, capped by the Windows "
           "2 GiB pinned-block artefact rather than by the card. Full physics costs "
           "8.7× more bytes per cell, so those domains are ~2.9× smaller per side.",
           outdir / "08-vanilla-vs-streamed.png")


# --------------------------------------------------------- 9. output scaling
#: Written by ``tilestream.make_output_json`` from the raw benchmark blocks
#: ``tilestream.bench_ooc_output`` dumps.  Kept out of this file so the
#: figures cannot drift from the numbers that were actually measured.
OUTPUT_JSON = Path(__file__).with_name("output-scaling.json")

#: The immovable majority of a frame.  Grey on purpose: it is the part no
#: choice in this lane can move, so the colour belongs to the slivers that
#: the hypothesis is actually about.
DISK = "#9aa4b0"


def _output_data():
    import json
    return json.loads(OUTPUT_JSON.read_text())


def _src(text, width=118):
    """Hard-wrap a source line before handing it to :func:`finish`.

    ``finish`` saves with ``bbox_inches="tight"``, which grows the canvas to
    contain every artist -- including a one-line ``fig.text``.  A 400-character
    provenance note therefore stretches the PNG to 4000 px wide and squashes
    the axes to a strip.  Wrapping here fixes it without touching ``finish``,
    which the other figures in this file share.
    """
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


def plot_output_scaling(outdir):
    """The CM1 framing: does writing history cost this model its scaling?

    CM1's published curves fall off a cliff the moment they write -- netCDF
    by ~5,000 cores, raw binary by ~20,000.  These are the same axes for
    ArWen and the answer is the opposite: every cadence of 10 steps or
    sparser lies ON the no-output line.

    The cadence curves being near-coincident with the reference IS the
    result, which makes them impossible to tell apart on the main axes -- so
    the inset carries what the main axes cannot resolve.
    """
    d = _output_data()
    c = d["cadence_curves"]
    xs = c["sizes"]

    fig, ax = plt.subplots(figsize=(7.6, 5.3))
    ax.plot(xs, c["no output"], "--", color=MUTED, lw=2.2, zorder=5,
            label="no output (reference)")

    # Colour runs red (worst) to green (best) with cadence, but a marker and
    # a direct label carry the same identity -- #2e7d52 and #c1442e sit at
    # deutan dE 7.7, which is only legal with that secondary encoding.
    series = [(1, PAGING, "o", "every step"),
              (10, ACCENT, "s", "every 10"),
              (60, RESIDENT, "^", "every 60"),
              (240, STREAMED, "D", "every 240 (hourly)")]
    for cad, col, mk, lab in series:
        ys = c[f"monolithic C={cad}"]
        ax.plot(xs, ys, marker=mk, color=col, lw=2.0, ms=5, zorder=6,
                label=f"output {lab}")
    # Only TWO direct labels: the three sparse cadences are coincident to
    # within 2%, so labelling each would be three labels on one line.
    ax.annotate("every step", xy=(xs[-1], c["monolithic C=1"][-1]),
                xytext=(9, 0), textcoords="offset points", fontsize=8,
                color=PAGING, fontweight="bold", va="center")
    ax.annotate("every 10 / 60 / 240\n(coincident)",
                xy=(xs[-1], c["monolithic C=240"][-1]), xytext=(9, 0),
                textcoords="offset points", fontsize=8, color=STREAMED,
                fontweight="bold", va="center")

    ax.annotate("writing EVERY step: 9.3x slower —\nthe only cadence that hurts",
                xy=(2400, c["monolithic C=1"][5]), xytext=(1080, 0.40),
                fontsize=8.5, color=PAGING,
                arrowprops=dict(arrowstyle="->", color=PAGING, lw=1.1))
    ax.annotate("every cadence of 10 steps or sparser\nlies ON the no-output reference",
                xy=(724, c["monolithic C=10"][1]), xytext=(505, 26),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("domain side (grid points, nz = 49)")
    ax.set_ylabel("timesteps per second")
    ax.set_xlim(480, 7000)
    ax.set_ylim(0.035, 90)
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    # Minor log ticks would print a second, overlapping set of x labels.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    ins = ax.inset_axes((0.60, 0.585, 0.375, 0.30))
    for cad, col, mk, _lab in series:
        pct = [100.0 * (1.0 - m / nn) for m, nn
               in zip(c[f"monolithic C={cad}"], c["no output"])]
        ins.plot(xs, pct, marker=mk, color=col, lw=1.5, ms=3.5)
    ins.set_xscale("log")
    ins.set_yscale("log")
    ins.set_xlim(470, 3600)
    ins.set_ylim(0.02, 400)
    ins.set_title("output as % of wall time", fontsize=7.5, pad=4,
                  color=MUTED, fontweight="normal")
    ins.set_xticks([512, 1024, 2048])
    ins.set_yticks([0.1, 1, 10, 100])
    ins.tick_params(labelsize=6.5)
    ins.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ins.xaxis.set_minor_formatter(NullFormatter())
    ins.yaxis.set_minor_formatter(NullFormatter())
    for s in ("top", "right"):
        ins.spines[s].set_visible(False)

    m = d["meta"]
    finish(fig, ax,
           "Output does not threaten this design, at any cadence anyone runs",
           "CM1 falls off a cliff when it writes. On this path every cadence "
           "of 10 steps or sparser sits on the no-output line.",
           _src(f"MEASURED — RTX 5090, WSL2, ext4 on the WSL2 VHD (not "
                f"/mnt/c). Frames land at {m['frame_throughput_GBs']:.2f} GB/s "
                f"through netCDF and fsync, under a {m['disk_ceiling_GBs']} "
                f"GB/s dd ceiling; solver 3.325 ns/cell. Curves DERIVED from "
                f"those measured per-cell costs.", 124),
           outdir / "06-output-scaling.png")


# ------------------------------------------------------ 10. frame breakdown
def plot_output_breakdown(outdir):
    """Where one history frame's time goes, and what streaming actually moves.

    The hypothesis was that an out-of-core run writes for free, because its
    authoritative state already sits in pinned host RAM.  The transfer it
    removes is real -- and it is 2% of a frame.  Worse, the store is NOT
    quiescent while tiles run, so a concurrent writer needs a host-side
    snapshot, and that copy costs more than the PCIe transfer it replaced.
    """
    d = _output_data()
    b = d["breakdown"]
    mono, strm = b["monolithic"], b["streamed"]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    segs = [
        ("transfer / copy", PAGING,
         [mono["transfer D2H"], 0.0, strm["snapshot (host copy)"]]),
        ("derive T, P, PSFC", ACCENT,
         [mono["derive (device)"], strm["derive (host)"],
          strm["derive (host)"]]),
        ("netCDF encode", RESIDENT, [mono["netCDF encode"]] * 3),
        ("disk: close + fsync", DISK, [mono["disk (close+fsync)"]] * 3),
    ]
    labels = ["MONOLITHIC\nD2H, then write",
              "STREAMED\nstore quiescent",
              "STREAMED\nwhile tiles run"]
    ypos = [2, 1, 0]
    left = [0.0, 0.0, 0.0]
    for name, col, vals in segs:
        ax.barh(ypos, vals, left=left, color=col, height=0.55,
                edgecolor="white", linewidth=2, label=name, zorder=3)
        for y, v, l in zip(ypos, vals, left):
            if v > 95:                      # direct-label only what fits
                ax.text(l + v / 2, y, f"{v:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold",
                        zorder=4)
        left = [l + v for l, v in zip(left, vals)]

    for y, tot in zip(ypos, left):
        ax.text(tot + 20, y, f"{tot:.0f} ms", va="center", fontsize=9,
                color=INK, fontweight="bold")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("milliseconds for one history frame")
    ax.set_xlim(0, 1760)
    ax.set_ylim(-1.02, 2.78)

    ax.annotate(f"{mono['transfer D2H']:.0f} ms of PCIe — the whole quantity "
                "the hypothesis targets",
                xy=(16, 2.30), xytext=(210, 2.60), fontsize=8, color=PAGING,
                arrowprops=dict(arrowstyle="->", color=PAGING, lw=1.0))
    ax.annotate(f"{strm['snapshot (host copy)']:.0f} ms of host memcpy "
                "replaces it — 2.4x DEARER\nthan the PCIe copy it saved",
                xy=(36, -0.30), xytext=(120, -0.62), fontsize=8, color=PAGING,
                arrowprops=dict(arrowstyle="->", color=PAGING, lw=1.0))
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)

    m = d["meta"]
    finish(fig, ax, "Streaming removes 2% of a frame, and pays 5% to do it",
           "1024² × 49, 15 fields, 1.67 GB payload. Both paths write a "
           "BYTE-IDENTICAL wrfout — whole-file SHA-256 verified.",
           _src("MEASURED — RTX 5090, WSL2, ext4 on the WSL2 VHD. "
                "Transfer/copy are MINIMA over 20 reps (a second lane shared "
                "the GPU); encode and disk are medians of 5, through fsync. "
                "'While tiles run' needs a snapshot: the ring store is TORN "
                "mid-sweep.", 124),
           outdir / "07-output-breakdown.png")


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1
                  else "tilestream-plots")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"rendering to {outdir}")
    plot_outofcore(outdir)
    plot_ladder(outdir)
    plot_halo(outdir)
    plot_managed(outdir)
    plot_duplex(outdir)
    plot_rings(outdir)
    plot_geography(outdir)
    plot_vanilla(outdir)
    plot_output_scaling(outdir)
    plot_output_breakdown(outdir)
    print("done")


if __name__ == "__main__":
    main()

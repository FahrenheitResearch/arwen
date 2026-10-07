import sys, json
sys.path.insert(0, '/work/meeting-perf-a13/hex/tree/src')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hexcore import device_admission as da
MIB = da.MIB
R = [("PRO 6000\nN1", 188, 1, 655362, 74115, 66086.1),
     ("PRO 6000\nN2", 188, 2, 331877, 57987, 36290.8),
     ("PRO 6000\nN4", 188, 4, 168322, 32183, 21226.1),
     ("PRO 6000\nN8*", 188, 8, 85295, 20932, None),
     ("RTX 5090\nN4", 170, 4, 168321, 31268, 20858.6),
     ("RTX 5090\nN8", 170, 8, 85295, 17396, 13204.6),
     ("conus3km LAM\nN4 PRO 6000", 188, 4, 590671, 97228, 62770.7)]
labels, before, after, peak = [], [], [], []
for lab, sms, n, cells, pk, old in R:
    conf = "limited-area" if "LAM" in lab else "global"
    m = da.model_for_card(da.card_profile_from_attributes(lab, {"MultiProcessorCount": sms, "MaxThreadsPerMultiProcessor": 1536}), conf)
    margin = m.margin_bytes() / MIB
    if old is None:   # old price: single-card row at the local cells, no new terms
        old = (m.core_bytes + m.bytes_per_cell * cells + m.tiled_bytes(cells)) / MIB
    new = (m.predict_bytes(cells) if n == 1 else m.predict_rank_bytes(cells)) / MIB
    labels.append(lab); before.append((old + margin) / 1024); after.append((new + margin) / 1024); peak.append(pk / 1024)
fig, ax = plt.subplots(figsize=(11.5, 5.6))
x = range(len(labels)); w = 0.27
ax.bar([i - w for i in x], before, w, label="Admission price before", color="#9aa5b1")
ax.bar([i for i in x], peak, w, label="Measured peak per card", color="#d1495b")
ax.bar([i + w for i in x], after, w, label="Admission price after", color="#2e86ab")
ax.axhline(31.4, color="k", lw=0.8, ls="--"); ax.text(-0.45, 32.3, "32 GB card", fontsize=8)
ax.axhline(95.6, color="k", lw=0.8, ls=":"); ax.text(-0.45, 96.5, "96 GB card", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("GiB per card"); ax.set_title("Hex admission price per card against the measured peak (30 km global, and conus3km limited-area)")
ax.legend(fontsize=8, loc="upper center")
fig.text(0.01, 0.01, "Peaks: whole-device nvidia-smi; global 2026-10-04 boxes, conus3km box D 2026-10-05 (largest rank). *PRO 6000 N8 ran to step 30 (crash since fixed).", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(sys.argv[1], dpi=130)
print(json.dumps({"labels": [l.replace("\n", " ") for l in labels], "before_gib": before, "peak_gib": peak, "after_gib": after}, indent=1))

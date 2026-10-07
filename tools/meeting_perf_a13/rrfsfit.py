import sys
sys.path.insert(0, '/work/meeting-perf-a13/hex/tree/src')
from hexcore import device_admission as da
MIB = da.MIB
for sms in (188, 170):
    m = da.model_for_card(da.card_profile_from_attributes("c", {"MultiProcessorCount": sms, "MaxThreadsPerMultiProcessor": 1536}), "limited-area")
    print(sms, "core", m.core_bytes / MIB, "margin", m.margin_bytes() / MIB)
    for cells, pk in ((590588, 91342), (591077, 90598), (591590, 96490), (590671, 97228)):
        p = m.predict_rank_bytes(cells)
        extra = pk * MIB - m.margin_bytes() - p
        print(f"  {cells} pred {p/MIB:.1f} required {(p+m.margin_bytes())/MIB:.1f} peak {pk} extra {extra/MIB:.1f} MiB = {extra/cells:.0f} B/cell")

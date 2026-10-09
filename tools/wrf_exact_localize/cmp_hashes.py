"""cmp_hashes.py A/hashes.json B/hashes.json : per-frame list of fields whose hashes differ."""
import json, sys
a = json.load(open(sys.argv[1]))["fields"]; b = json.load(open(sys.argv[2]))["fields"]
skip = {"RQIBLTEN"}
common = sorted(set(a) & set(b) - skip)
n = min(len(next(iter(a.values()))["sha256"]), len(next(iter(b.values()))["sha256"]))
print("fields common", len(common), "only A", len(set(a) - set(b)), "only B", len(set(b) - set(a)), "frames", n)
for f in range(n):
    d = [k for k in common if a[k]["sha256"][f] != b[k]["sha256"][f]]
    print(f"frame {f}: {len(d)} differ" + (": " + " ".join(d[:30]) if d else ""))

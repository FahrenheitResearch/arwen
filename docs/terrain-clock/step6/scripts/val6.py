import json, sys
r = json.loads(open(sys.argv[1]).readline())
ok = r["peak_w"] == 34.655418395996094
print("VALIDATE card", sys.argv[2], "OK" if ok else "MISMATCH", repr(r["peak_w"]))
sys.exit(0 if ok else 3)

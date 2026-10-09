#!/usr/bin/env python3
"""Cut a raw-array overlay for `gpuwm verify-exact` out of a full recording.

    python verify_exact_raw_subset.py RECORDING_ROOT NEEDS.json OVERLAY_DIR

NEEDS.json is what ``gpuwm verify-exact --write-needs`` writes: a list of
``{build, case, combo, step, fields}``.  For each entry this copies only those
fields' words at that step from ``RECORDING_ROOT/<build>/<case>/<combo>/stepNNNN.bin``
into ``OVERLAY_DIR`` in the same layout, so a copy of the recording without raw
arrays can still measure a digest miss in ULP (``--raw-overlay OVERLAY_DIR``).

numpy and the standard library only: it runs on the box that holds the full
recording, which has no gpuwm install.  Every copied field's SHA-256 is checked
against the recording's own index before it is written.
"""

import hashlib
import json
import os
import sys


def main(argv):
    root, needs_path, overlay = argv[1:4]
    needs = json.load(open(needs_path))
    wrote = 0
    for need in needs:
        rel = os.path.join(need["build"], need["case"], need["combo"])
        name = "step%04d" % int(need["step"])
        index_path = os.path.join(root, rel, name + ".json")
        if not os.path.exists(index_path):
            print("not recorded raw:", rel, name)
            continue
        index = json.load(open(index_path))
        by_name = {f["name"]: f for f in index["fields"]}
        out_dir = os.path.join(overlay, rel)
        os.makedirs(out_dir, exist_ok=True)
        out_index = os.path.join(out_dir, name + ".json")
        have = json.load(open(out_index))["fields"] if os.path.exists(out_index) else []
        got = {f["name"] for f in have}
        blob_path = os.path.join(out_dir, name + ".bin")
        with open(os.path.join(root, rel, name + ".bin"), "rb") as src, open(blob_path, "ab") as dst:
            offset = dst.tell()
            for field in need["fields"]:
                if field in got or field not in by_name:
                    continue
                entry = by_name[field]
                src.seek(entry["offset"])
                payload = src.read(entry["nbytes"])
                if entry.get("sha256") and hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise SystemExit("%s %s %s: bytes do not match the recording's digest" % (rel, name, field))
                dst.write(payload)
                have.append(dict(entry, offset=offset))
                offset += len(payload)
                wrote += 1
        json.dump({"step": int(need["step"]), "fields": have}, open(out_index, "w"))
    print("overlay", overlay, "fields", wrote)


if __name__ == "__main__":
    main(sys.argv)

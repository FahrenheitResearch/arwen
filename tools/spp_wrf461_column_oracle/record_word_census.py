"""Record the full SPP MYNN output census without changing reference bytes."""
from pathlib import Path
import argparse, hashlib, json
import spp_column_fixtures as columns
import spp_column_grade as grade

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rows = []
    for mixlength in (1, 2):
        path = columns.FIXTURE_DIR / f"mynn-spp-ml{mixlength}.npz"
        for replay in (True, False):
            result = grade.grade_mynn(path, mixlength, replay=replay, backend="cpu")
            assert result["max_ulp"] == 0 and result["words_differ"] == 0, result
            rows.append({"mixlength": mixlength, "replay": replay,
                         "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "words": result["words"], "words_differ": result["words_differ"],
                         "fields": {k: v["words"] for k, v in result["fields"].items()}})
    counts = {row["words"] for row in rows}
    assert len(counts) == 1
    words = counts.pop()
    if args.write:
        root = Path(__file__).resolve().parents[2]
        for filename in ("test_spp_wrf461_column_oracle.py", "test_spp_wrf461_column_oracle_gpu.py"):
            path = root / "tests" / filename
            text = path.read_text()
            import re
            text, n = re.subn(r'assert graded\["words"\] == [0-9]+',
                             'assert graded["words"] == ' + str(words), text)
            assert n == 1
            text = text.replace("26 output fields,", str(len(rows[0]["fields"])) + " output fields,")
            text = text.replace("332,352 words per run", f"{words:,} words per run")
            path.write_text(text)
    args.out.write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    print(json.dumps({"runs": len(rows), "fields_per_run": len(rows[0]["fields"]),
                      "words_per_run": words, "words_differ": 0}))

if __name__ == "__main__":
    main()

"""Aurora rows fetch only the files their loader opens.

WOOF owner canary run_867b9bec5bedcfd7 (2026-10-26): a row without weights.files fetched the whole microsoft/aurora
repository (21 files, 41.2 GB) and the box ran out of disk ('No space left on device')."""
import json
from pathlib import Path

TABLE = json.loads((Path(__file__).resolve().parents[1] / "gpuwm/authorities/earth2-models.v1.json").read_text(encoding="utf-8"))


def test_aurora_1p5_rows_name_their_checkpoint_and_statics():
    for name, checkpoint in (("aurora15", "aurora-0.25-v1.5.ckpt"), ("aurora15-6h", "aurora-0.25-v1.5.ckpt"),
                             ("aurora15-ensemble", "aurora-0.25-v1.5-ensemble.ckpt"),
                             ("aurora15-ensemble-6h", "aurora-0.25-v1.5-ensemble.ckpt")):
        files = TABLE["models"][name]["weights"]["files"]
        assert [f["path"] for f in files] == [checkpoint, "aurora-0.25-v1.5-static.pickle"], name
        assert sum(f["size"] for f in files) < 6e9, name


def test_aurora_v1_names_why_it_cannot_load():
    assert "static-npy" in TABLE["models"]["aurora"]["runtime_refusal"]

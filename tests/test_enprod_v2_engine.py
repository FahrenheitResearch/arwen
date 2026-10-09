"""The real ``rw_ensbatch`` reduces a ``gpuwm go --members`` run folder.

The same histories laid out the two ways an ensemble is written -- the
``member_NNN/`` + ``gpuwm-ensemble-manifest.v1`` root and the
``<run>/<domain>/ensemble-manifest.json`` (``gpuwm-ensemble-output.v2``)
run folder ``go --members --keep-member-files`` writes -- give the same
panels through the engine this checkout resolves.  Before 2.8.8 the v2 run
folder failed with the engine's usage line (tests/test_enprod_v2_manifest.py
names the breakage).  Skips, saying why, where no usable engine resolves.
"""
from __future__ import annotations

import json
import shutil

import pytest

from gpuwm import cli, render_layout, rustwx_lanes
from gpuwm.da import enprod


def _engine():
    try:
        path = rustwx_lanes.find_ensemble_bin()
    except FileNotFoundError as error:
        return None, str(error)
    if path is None:
        return None, "rw_ensbatch is not built"
    usable, evidence = rustwx_lanes.probe_ensemble_bin(path)
    return (path if usable else None), evidence


_ENGINE, _WHY = _engine()
pytestmark = pytest.mark.skipif(_ENGINE is None, reason=f"no usable rw_ensbatch: {_WHY}")


def _as_go_run(v1_root, run):
    """Lay the v1 fixture's members out as go --members writes them."""
    manifest = json.loads((v1_root / enprod.MANIFEST_FILENAME).read_text(
        encoding="utf-8"))
    files, order = [], []
    for record in manifest["members"]:
        member = int(record["index"])
        order.append(member)
        for source in sorted((v1_root / record["member_dir"]).rglob("wrfout*")):
            relative = f"members/member-{member:04d}/wrfout/{source.name}"
            target = run / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append({"bytes": target.stat().st_size, "domain": "d02",
                          "member_id": member, "path": relative})
    (run / "d02").mkdir(parents=True, exist_ok=True)
    (run / "d02" / enprod.MANIFEST_FILENAME).write_text(json.dumps({
        "schema": enprod.V2_MANIFEST_SCHEMA, "members_requested": len(order),
        "member_order": order, "member_metadata": [],
        "keep_member_files": True, "member_files": files, "frames": []}),
        encoding="utf-8")
    return run


def test_the_engine_draws_the_same_panels_from_both_layouts(tmp_path, capsys):
    v1 = tmp_path / "v1"
    enprod.write_synthetic_ensemble(v1, n_members=3, domain_id=2)
    run = _as_go_run(v1, tmp_path / "run")
    counts = {}
    for name, root in (("v1", v1), ("v2", run)):
        out = tmp_path / f"png-{name}"
        rc = cli.main(["enprod", str(root), "--field", "refl", "--products",
                       "mean,spread,prob", "--out", str(out),
                       "--engine", "rust"])
        captured = capsys.readouterr()
        assert rc == 0, captured.out + captured.err
        assert "rw_ensbatch --list-fields" not in captured.err
        delivered = render_layout.iter_rendered(out)
        assert delivered, captured.out + captured.err
        counts[name] = sorted(path.relative_to(out).parts[1]
                              for path in delivered)
        assert "members n=3" in captured.out
    assert counts["v1"] == counts["v2"]

"""New forecast's Physics step starts every set the engine's check says runs (create.js, run in Node).

The step used to start a picked set only when it was one of the data source's named sets: of 80 two-pick GFS
mixes the check ran 63 and the page could start 5, with Next held off under "Runs." for the rest.  A set that
runs is startable now; the named set it makes, when the source offers one, is what goes into the plan by name.

create.js is loaded as the module the browser loads, with each page module it imports replaced by a stub that
exports the same names, so only its own code runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

JS = Path(__file__).resolve().parents[1] / "gpuwm" / "gui" / "static" / "js"
IMPORT = re.compile(r'^import\s+(?:\{([^}]*)\}|\*\s+as\s+\w+)\s+from\s+"\./([\w.-]+)";', re.M)

SCRIPT = r"""
import { startable, offeredSet } from "./create.js";
const gfs = {id: "gfs", profiles: [{id: "morrison-set"}, {id: "thompson-set"}]};
console.log(JSON.stringify({
  mix: startable({valid: true, named_suite: null}),
  named: startable({valid: true, named_suite: "thompson-set"}),
  refused: startable({valid: false, named_suite: null}),
  unchecked: startable(null),
  offered: offeredSet({valid: true, named_suite: "thompson-set"}, gfs),
  not_offered: offeredSet({valid: true, named_suite: "other-set"}, gfs),
  no_name: offeredSet({valid: true, named_suite: null}, gfs),
  refused_name: offeredSet({valid: false, named_suite: "thompson-set"}, gfs),
}));
"""


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    folder = tmp_path_factory.mktemp("create")
    text = (JS / "create.js").read_text(encoding="utf-8")
    (folder / "create.js").write_text(text, encoding="utf-8")
    for names, module in IMPORT.findall(text):
        exported = [name.strip().split(" as ")[-1] for name in names.split(",") if name.strip()]
        (folder / module).write_text("".join(f"export const {name} = () => {{}};\n" for name in exported) or
                                     "export {};\n", encoding="utf-8")
    (folder / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    (folder / "t.js").write_text(SCRIPT, encoding="utf-8")
    done = subprocess.run([node, "t.js"], cwd=folder, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_a_set_that_runs_is_startable_whether_or_not_a_named_set_matches_it(page):
    assert page["mix"] is True and page["named"] is True
    assert page["refused"] is False and page["unchecked"] is False


def test_only_a_set_the_source_offers_goes_into_the_plan_by_name(page):
    assert page["offered"] == "thompson-set"
    assert page["not_offered"] == page["no_name"] == page["refused_name"] == ""

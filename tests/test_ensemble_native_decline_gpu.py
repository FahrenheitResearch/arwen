"""A source-audit drift in the native member pack declines to the ordinary runner.

The native pack rewrites CUDA text and the Python source of stock physics
calls under counted source audits. The audits run while a pack prepares its
launches, after the run was admitted. A drift there used to end an admitted
run as a failure ("no retry is permitted") although no member had advanced.

This gate drives the real session and its automatic native path on a card,
which no other shipped test does, in two arms over the same real inputs:

* unchanged sources: the members run as one native pack, and the run's step
  log says what the pack advanced (it used to close with "0 steps");
* one audited kernel line spelled differently: the real audit refuses, the
  session declines with the audit's own sentence as its named reason, every
  member runs through the ordinary runner, and the products are the same
  bytes the native pack wrote.
"""
from dataclasses import replace
import gc
import hashlib
import inspect
import os
from pathlib import Path
import re

import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

_MEMBERS = 4
_AUDIT_SENTENCE = "Thompson base-load indexing changed; shared terrain adapter needs a new source audit"


@pytest.fixture(scope="module")
def decline_real(tmp_path_factory):
    directory = os.environ.get("GPUWM_TEST_WRF_REAL_DIRECTORY")
    if not directory:
        pytest.skip("set GPUWM_TEST_WRF_REAL_DIRECTORY to retained real WRF inputs")
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.wrfinput_forecast import prepare_wrf_run
    from gpuwm.prepared_domain_tree_forecast import _with_terrain_acoustics
    run = resolve_wrfinput_run(Path(directory))
    text, count = re.subn(r"(?m)^(\s*history_interval_s)\s*=.*$",
                         r"\1 = 12.0", run.toml_text)
    assert count, "the imported configuration must contain history authority"
    inputs = prepare_wrf_run(replace(run, toml_text=text),
                             tmp_path_factory.mktemp("decline-ensemble-source"), run_seconds=24)
    inputs = _with_terrain_acoustics(inputs)
    assert inputs.experiment.root.run.mp_physics == 8, "the audit this gate drifts is Thompson's"
    return inputs


def _renderer():
    from gpuwm import rustwx
    renderer = (Path(os.environ["GPUWM_ENSEMBLE_RENDERER"])
                if "GPUWM_ENSEMBLE_RENDERER" in os.environ else rustwx.find_renderer())
    if renderer is None:
        pytest.fail("the production gate requires the built native ensemble renderer")
    return renderer


def _session(out):
    """The production session, as a door builds it, for copies of one forecast.

    A native pack is reachable for members of one prepared input. A session
    that asks its caller to state that purpose is told it.
    """
    from gpuwm.ensemble.production import PreparedEnsembleSession
    stated = {}
    if "identical_members" in inspect.signature(PreparedEnsembleSession.__init__).parameters:
        stated["identical_members"] = ("native decline gate: every member runs the one prepared "
                                       "input so the native pack and the ordinary runner can be compared")
    return PreparedEnsembleSession({"members": _MEMBERS}, output_directory=out,
                                   renderer=_renderer(), **stated)


def _run(inputs, out):
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    from gpuwm.wrfinput_forecast import WrfInitialization
    # The wrfinput door's own call: the real runner and its own initialization.
    report = _session(out).run_prepared(run_prepared_tree, inputs, io_mode="history",
                                        initialization=WrfInitialization(inputs))
    gc.collect()
    return report


def _product_bytes(out):
    files = sorted(path for pattern in ("d01/ensemble/products/*.nc", "maps/**/*.png")
                   for path in out.glob(pattern))
    assert files, "the run wrote no aggregate product"
    return {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def test_a_native_source_audit_drift_declines_to_the_ordinary_runner(decline_real, tmp_path,
                                                                     monkeypatch, capfd):
    from gpuwm.ensemble import batch_physics

    # Arm 1, unchanged sources: one native pack carries the roster.
    native = _run(decline_real, tmp_path / "native")
    said = capfd.readouterr()
    assert native["status"] == "PASS" and native["members_completed"] == list(range(_MEMBERS))
    assert native["automatic_admission"]["native_admitted"] is True
    assert not native["automatic_admission"]["fallback_reasons"]
    (pack,) = native["member_results"]
    assert pack["global_member_ids"] == list(range(_MEMBERS))
    assert pack["result"]["executor"]["steps"] == 2
    # The run's step log follows the pack: its two steps, the roster the
    # pack carried, and a closing count that is the pack's own.
    log = said.out
    assert len(re.findall(r"(?m)^Timing for main: .* on domain +1: ", log)) == 2, log[-3000:]
    assert f"gpuwm: phase native ensemble forecast, {_MEMBERS} members, 2 steps each:" in log, log[-3000:]
    assert "gpuwm: SUCCESS COMPLETE SIMULATION, 2 steps, " in log, log[-3000:]
    assert "SUCCESS COMPLETE SIMULATION, 0 steps" not in log
    assert "the native member pack declined" not in said.err

    # Arm 2: the kernel text the Thompson audit reads has one audited
    # base-state load spelled differently, as an edit to that kernel would.
    audit, refused = batch_physics._thompson_base_source, []

    def drifted(source, **options):
        changed = source.replace("thb[thb_full ? idx : k]", "thb[(thb_full ? idx : k)]")
        assert changed != source, "the stock kernel no longer has the line this gate drifts"
        try:
            return audit(changed, **options)
        except ValueError as refusal:
            refused.append(str(refusal))
            raise

    monkeypatch.setattr(batch_physics, "_thompson_base_source", drifted)
    declined = _run(decline_real, tmp_path / "declined")
    said = capfd.readouterr()
    assert refused == [_AUDIT_SENTENCE], "the real audit is what refused, once, in the first pack"
    assert declined["status"] == "PASS" and declined["members_completed"] == list(range(_MEMBERS))
    admission = declined["automatic_admission"]
    assert admission["native_admitted"] is False and admission["fallback"] == "ordinary_member_runner"
    (reason,) = admission["fallback_reasons"]
    assert "before any member advanced" in reason and _AUDIT_SENTENCE in reason
    assert all(batch["execution_mode"] == "ordinary_member" for batch in declined["packing"]["batches"])
    assert ("ensemble: the native member pack declined and every member runs through the "
            "ordinary runner: ") in said.err, said.err[-3000:]
    assert _AUDIT_SENTENCE in said.err
    # Every member then ran its own ordinary forecast from the start.
    assert said.out.count("gpuwm: SUCCESS COMPLETE SIMULATION, 2 steps, ") == _MEMBERS, said.out[-3000:]
    # The decline cost the run nothing but time: the same product bytes.
    assert _product_bytes(tmp_path / "declined") == _product_bytes(tmp_path / "native")

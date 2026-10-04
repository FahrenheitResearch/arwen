"""Members with different inputs, packed on one card, against their own ordinary runs.

Every member starts from the retained real WRF inputs but owns its own
initialized words at the bootstrap seam: the existing seeded wind increment
plus a seeded skin and soil temperature offset written into its initialized
physics driver. Each member's ordinary run is the reference. The native pack
is built the way the production handoff builds it for time-lagged and
multi-model members: the first member stays live as the root, every other
member is bootstrapped through its own ordinary initializer, snapshotted to
host, released, and bound to the root as a member source. Every state,
physics and output word of every member must match its ordinary run, and
the aggregate products must match byte for byte.
"""
from datetime import timedelta
import gc
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from test_ensemble_production_forecast_gpu import (
    production_real, _WordCapture, _CapturedProducts, _collector, _renderer,
    _product_words, _available_bytes, _SEED,
)

pytestmark = [pytest.mark.gpu, requires_gpu]

#: Skin and soil temperature are member fields already; deep soil temperature
#: is a shared static field that becomes member-owned when members disagree.
_SURFACE_FIELDS = ("tsk", "tslb", "tmn")


def _initialization(inputs, member):
    """The member's own initialization: seeded winds and a seeded surface offset."""
    from gpuwm.wrfinput_forecast import WrfInitialization
    from gpuwm.forecast_initialization import DomainInitialization
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds

    class MemberInitialization(WrfInitialization):
        def restore_domain(self, domain, grid, bundle, **options):
            initialized = super().restore_domain(domain, grid, bundle, **options)
            def initialize():
                result = initialized.initialize_physics()
                initialize_member_winds(state=initialized.state, cfg=domain.run,
                    member_indices=(member,), seeds=(_SEED + member,),
                    phase="after_physics_before_step")
                fields = initialized.state.physics.fields
                touched = [name for name in _SURFACE_FIELDS if name in fields]
                assert touched, "the retained real driver carries no skin or soil temperature field"
                for name in touched:
                    fields[name] += np.float32(0.05 * (member + 1))
                return result
            return DomainInitialization(initialized.initial_result, initialize)
    return MemberInitialization(inputs)


def _ordinary(inputs, out, member, capture):
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    Path(out).mkdir(parents=True, exist_ok=True)
    def initialized(*, node, **kwargs):
        capture.words.nodes[member] = node
        capture.words.state("initialized", member=member, state=node.state,
                            driver=node.state.physics, clock=node.clock)
        return None
    with member_output_scope(MemberOutputCapture(capture.submit, member, capture.keep_member_files)):
        report = run_prepared_tree(inputs, output_directory=out, io_mode="history",
            initialization=_initialization(inputs, member), ensemble_bootstrap=initialized)
    node = capture.words.nodes.pop(member)
    capture.words.state("final", member=member, state=node.state,
                        driver=node.state.physics, clock=node.clock)
    assert node.clock.step_count == 2 and node.clock.elapsed_seconds == 24
    del node
    return report


@pytest.mark.parametrize("members", [2, 4])
def test_members_with_their_own_bootstraps_pack_byte_identically_to_their_ordinary_runs(production_real, members, tmp_path):
    import cupy as cp
    from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
    from gpuwm.ensemble.native_forecast import run_initialized_native_ensemble
    from gpuwm.ensemble.prepared_batch import (NativeMemberSources, bind_member_sources,
                                               native_prepared_eligibility, snapshot_member_source)
    inputs, renderer = production_real, _renderer()
    cfg = inputs.experiment.root.run
    control_words = _WordCapture(tmp_path / "ordinary-words", cfg)
    control = _CapturedProducts(_collector(tmp_path / "ordinary-products", inputs, members, renderer), control_words)
    for member in range(members):
        _ordinary(inputs, tmp_path / "ordinary" / f"member-{member:04d}", member, control)
        gc.collect()
    expected_products = _product_words(control.collector)
    # The members really differ, in the dycore state and in the land surface.
    initialized = {member: control_words.records[f"member-{member:04d}/initialized"] for member in range(members)}
    assert initialized[0]["state"]["u"] != initialized[1]["state"]["u"]
    surface = [path for path in initialized[0]["physics"] if path.split("/")[-1] in _SURFACE_FIELDS]
    assert surface and all(initialized[0]["physics"][path] != initialized[1]["physics"][path] for path in surface)

    native_words = _WordCapture(tmp_path / "native-words", cfg, reference=control_words, members=members)
    native = _CapturedProducts(_collector(tmp_path / "native-products", inputs, members, renderer),
                               native_words, native=True)
    bootstraps = []

    def native_bootstrap(*, node, **kwargs):
        pool = cp.get_default_memory_pool()
        snapshots = []
        for member in range(1, members):
            holder = {}
            def capture(*, node, **unused):
                holder["source"] = snapshot_member_source(node, member_id=member, receipt={"seed": _SEED + member})
                return {"status": "PASS", "ensemble_bootstrap_only": True,
                        "completed_seconds": 0.0, "forecast_steps": 0}
            before = int(pool.used_bytes())
            out = tmp_path / "member-bootstraps" / f"member-{member:04d}"
            out.mkdir(parents=True)
            report = run_prepared_tree(inputs, output_directory=out, io_mode="history",
                initialization=_initialization(inputs, member), ensemble_bootstrap=capture)
            assert isinstance(report, dict) and report["forecast_steps"] == 0 and "source" in holder
            del report
            gc.collect()
            cp.cuda.get_current_stream().synchronize()
            bootstraps.append({"member_id": member, "pool_live_before_bytes": before,
                               "pool_live_after_release_bytes": int(pool.used_bytes()),
                               "host_snapshot_bytes": holder["source"].nbytes})
            snapshots.append(holder.pop("source"))
        sources = NativeMemberSources(0, snapshots)
        assert sources.compatibility_reasons(node) == ()
        bind_member_sources(node, sources)
        eligibility = native_prepared_eligibility(inputs, node, members=members)
        assert eligibility.eligible, eligibility.receipt()

        def validation(*, phase, owners, clock, member_ids, **kw):
            for local_member, member in enumerate(member_ids):
                label = (f"history-{(inputs.experiment.start_time + timedelta(seconds=clock.elapsed_seconds)).strftime('%Y-%m-%d_%H:%M:%S')}"
                         if phase == "history" else phase)
                native_words.state(label, member=member, state=owners.batch,
                    driver=owners.physics.driver, clock=clock, packed_member=local_member)
            return {"members_checked": list(member_ids)}
        report = run_initialized_native_ensemble(inputs, node, members=members,
            member_ids=tuple(range(members)), collector=native,
            available_bytes=_available_bytes, validation_callback=validation)
        assert report is not None and report["backend"] == "native_member_batched"
        assert report["executor"]["steps"] == 2 and report["executor"]["member_steps"] == 2 * members
        allocations = report["native_allocations"]
        assert allocations["ordinary_initialized_bootstraps"] == members
        assert allocations["member_sources"][0]["source"].startswith("live ordinary bootstrap")
        assert all(row["source"].startswith("own ordinary bootstrap") for row in allocations["member_sources"][1:])
        assert allocations["physics"]["member_bootstraps"][0] == "live bootstrap"
        return report

    bootstrap_out = tmp_path / "native-bootstrap"
    bootstrap_out.mkdir()
    native_report = run_prepared_tree(inputs, output_directory=bootstrap_out, io_mode="history",
        initialization=_initialization(inputs, 0), ensemble_bootstrap=native_bootstrap)
    assert set(native_words.records) == set(control_words.records)
    assert _product_words(native.collector) == expected_products
    assert not list(bootstrap_out.rglob("wrfout_*"))
    cp.cuda.get_current_stream().synchronize()
    receipt = {"schema": "gpuwm-ensemble-member-sources-identity.v1", "members": members,
               "member_inputs": "same real inputs, per-member seeded winds and surface temperature offset",
               "identity": "all state, physics and output words per member; aggregate products byte for byte",
               "member_bootstraps": bootstraps, "native_forecast": native_report,
               "ordinary_words": control_words.flush(), "native_words": native_words.flush()}
    (tmp_path / "member-sources-identity-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")

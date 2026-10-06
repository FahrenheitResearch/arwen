"""Run the real prepared-tree CLI and inject one health fault after a restart.

GPU proof tool, never part of production dispatch. Pass the ordinary prepared
tree runner's arguments. Configure at least one restart before the stop time.
The injector changes one child's actual w cell after the first durable
checkpoint, asks the real full-state health gate to refuse it, and then lets
default checkpoint recovery and the real solver finish the remaining run.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@contextmanager
def health_fault_after_checkpoint(output_directory, *, delay_after_checkpoint_s=0):
    """Wrap a real ``run_prepared_tree`` call and yield its injection ledger."""
    from gpuwm.core import model as model_module
    from gpuwm.core.health import HealthCheckError, health_validator_for_domain
    from gpuwm.supervisor import atomic_write_json
    from gpuwm.io.restart import read_restart_header

    output_directory = Path(output_directory)
    if delay_after_checkpoint_s < 0:
        raise ValueError("health fault delay must be nonnegative")
    injected = []
    origin = []
    original = model_module.execute_experiment

    def execute(tree, **kwargs):
        progress = kwargs.get("progress_callback")

        def inject(**event):
            if progress is not None:
                progress(**event)
            checkpoint = getattr(tree, "_last_checkpoint", None)
            if injected or checkpoint is None or event["model_elapsed_seconds"] <= 0:
                return
            if not origin:
                header = read_restart_header(checkpoint)
                origin.append({"checkpoint": str(checkpoint),
                    "model_seconds": float(header["elapsed_seconds"])})
            threshold = origin[0]["model_seconds"] + delay_after_checkpoint_s
            if event["model_elapsed_seconds"] < threshold:
                return
            if str(checkpoint) != origin[0]["checkpoint"]:
                raise ValueError("a later checkpoint replaced the injection "
                                 "origin; shorten the requested fault delay")
            candidates = [node for node in tree.walk_parent_first()
                          if node.parent is not None and node._started
                          and getattr(node.state, "_streamed_domain", None) is None]
            if not candidates:
                raise ValueError("health injection needs an active resident child")
            node = candidates[-1]
            node.state.w[0, 0, 0] = 239.79
            evidence = {
                "schema": "gpuwm.stability-fault-injection/v1",
                "variable": "w", "value": 239.79, "index": [0, 0, 0],
                "domain": f"d{int(node.cfg.grid_id):02d}",
                "checkpoint": str(checkpoint),
                "origin_checkpoint_model_seconds": origin[0]["model_seconds"],
                "delay_after_checkpoint_s": delay_after_checkpoint_s,
                "model_seconds": event["model_elapsed_seconds"],
                "purpose": "prove bounded default recovery uses real checkpoint "
                           "state and continues the real GPU solver",
            }
            injected.append(evidence)
            path = output_directory / "evidence" / "stability-fault-injection.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                health_validator_for_domain(tree, node).require_healthy(
                    phase=f"post-d01-sync.d{int(node.cfg.grid_id):02d}")
            except HealthCheckError as error:
                evidence["error"] = str(error)
                atomic_write_json(path, evidence)
                raise
            raise AssertionError("the real health gate accepted w=239.79")

        return original(tree, **{**kwargs, "progress_callback": inject})

    model_module.execute_experiment = execute
    try:
        yield injected
    finally:
        model_module.execute_experiment = original


def main(argv=None):
    from gpuwm import prepared_domain_tree_forecast as runner

    argv = list(sys.argv[1:] if argv is None else argv)
    args = runner.build_parser().parse_args(argv)
    with health_fault_after_checkpoint(args.outdir) as injected:
        code = runner.main(argv)
    if code != 0:
        return code
    if len(injected) != 1:
        raise AssertionError("forecast finished without exercising a restart retry")
    receipt_path = args.outdir / "evidence" / "stability-recovery.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "RECOVERED" or len(receipt["attempts"]) != 1:
        raise AssertionError(f"fault injection did not recover once: {receipt}")
    print("stability retry proof: PASS, one real health failure, one restored "
          "checkpoint, real solver finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

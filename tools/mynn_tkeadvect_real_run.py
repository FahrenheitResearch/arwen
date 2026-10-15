"""Run the production prepared forecast while recording finite scalar TKE checks."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--split-checkpoint-dir", type=Path)
    p.add_argument("--final-checkpoint", type=Path)
    a, forecast_args = p.parse_known_args()
    if forecast_args[:1] == ["--"]:
        forecast_args = forecast_args[1:]
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.io.restart import write_restart, write_tree_restart
    original = dycore.step
    readings = []
    split_checkpoint = None

    def recorded_step(state, cfg, *args, **kwargs):
        result = original(state, cfg, *args, **kwargs)
        seconds = float(state.elapsed_seconds)
        if seconds == 3600. and a.final_checkpoint is not None:
            write_restart(a.final_checkpoint, state, cfg)
        if cfg.bl_pbl_physics == 5 and cfg.bl_mynn_tkeadvect:
            qke = state.qke_adv
            finite = bool(cp.isfinite(qke).all().item())
            assert finite, "QKE_ADV became non-finite in the production RK forecast"
            readings.append(dict(seconds=float(state.elapsed_seconds),
                                 finite=finite, minimum=float(qke.min().item()),
                                 maximum=float(qke.max().item())))
            if float(state.elapsed_seconds) >= 3600.:
                readings[-1]["sha256"] = hashlib.sha256(cp.asnumpy(qke).tobytes()).hexdigest()
        return result

    dycore.step = recorded_step
    # Import after binding the observation wrapper. The wrapper calls the
    # original production step, changing no operands or numerical returns.
    import gpuwm.prepared_single_domain_forecast as prepared
    from gpuwm.core import model as model_module
    execute = model_module.execute_experiment

    def observed_execute(model, **kwargs):
        progress = kwargs.get("progress_callback")

        def observed_progress(**reading):
            nonlocal split_checkpoint
            if progress is not None:
                progress(**reading)
            if (a.split_checkpoint_dir is not None and split_checkpoint is None
                    and reading.get("model_elapsed_seconds") == 1800.):
                io_manager = getattr(model, "_io_manager", None)
                if io_manager is not None:
                    io_manager.drain()
                    model._runtime_status.pending_d2h = int(io_manager.pending)
                valid = model.schedule.clock.start_time + timedelta(seconds=1800.)
                split_checkpoint = str(write_tree_restart(
                    a.split_checkpoint_dir, model, valid))

        kwargs["progress_callback"] = observed_progress
        return execute(model, **kwargs)

    model_module.execute_experiment = observed_execute
    forecast = prepared.main
    rc = None
    try:
        rc = forecast(forecast_args)
    finally:
        a.receipt.write_text(json.dumps(dict(exit_code=rc, steps=len(readings),
            final_seconds=(readings[-1]["seconds"] if readings else None),
            split_checkpoint=split_checkpoint,
            readings=readings), indent=2) + "\n", encoding="utf-8")
    assert rc == 0, rc
    assert readings and readings[-1]["seconds"] == 3600., readings[-1:]
    print(a.receipt, flush=True)


if __name__ == "__main__":
    main()

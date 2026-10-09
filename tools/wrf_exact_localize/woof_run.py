"""Run WOOF's WRF-file door on <stage dir> (wrfinput_d01, wrfbdy_d01, namelist.input); history every step to <out>.
usage: woof_run.py STAGE OUT SECONDS [--resolve-only]"""
import sys, time, json
from pathlib import Path
stage, out, seconds = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
import os
if os.environ.get("WOOF_LOCDUMP"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import woof_hook  # noqa: F401
    import woof_hook2  # noqa: F401
from gpuwm.wrfinput_door import resolve_wrfinput_run
t0 = time.time()
resolved = resolve_wrfinput_run(stage)
(out).mkdir(parents=True, exist_ok=True)
(out / "resolved.toml").write_text(resolved.toml_text)
print("resolved in", round(time.time() - t0, 2), "s", flush=True)
if "--resolve-only" in sys.argv:
    sys.exit(0)
from gpuwm.prepared_domain_tree_forecast import run_prepared_tree
from gpuwm.progress_log import ProgressOptions
from gpuwm.wrfinput_forecast import WrfInitialization, prepare_wrf_run
inputs = prepare_wrf_run(resolved, out / "input", run_seconds=seconds)
run_prepared_tree(inputs, output_directory=out, io_mode="history",
                  initialization=WrfInitialization(inputs),
                  progress_options=ProgressOptions(frame_markers=False))
print("done", round(time.time() - t0, 2), "s", flush=True)

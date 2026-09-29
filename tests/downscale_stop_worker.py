"""Process bodies for tests/test_downscale_stop.py.

Two programs in one file, chosen by the first argument:

``child OUTDIR CONFIG PRODUCTS``
    The real engine door (:func:`gpuwm.offline_child_run.main`, and so
    ``run``, its stop handling and its records) with a stub stepper in
    place of the CUDA forecast: it publishes the run manifest, arms the
    renders, commits frames and emits one ``model_progress`` per step,
    printing ``child_step N`` so a test knows it is integrating.
    ``STUB_STEPS`` sets how many steps it runs; the default is far more
    than any test waits for, and a smaller one lets the forecast finish
    and reach its finalize render.

``render --out DIR FRAME``
    A stand-in for ``gpuwm render`` that does what the real one does to
    its output folder: the engine draws every picture FLAT under its own
    staging name (``rustwx_wrf_..._<product>.png``, ``..._var_wrf_*``),
    then the render files each one into ``<domain>/<product>/<day>/`` and
    publishes its invocation receipt.  ``STUB_HOLD`` names a frame whose
    render stops between the two steps (after writing
    ``STUB_MARK/staged-<frame>``), which is where the Stop landed when a
    stopped child counted the renderer's staging files as its pictures.
    A Ctrl-C exits 130, as ``gpuwm render`` does.
    Each render writes its pid to ``STUB_MARK/pid-<frame>``, so a test
    can tell whether a stop ended it.

Kept in its own module because a test module's name depends on how
pytest collected it.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

START = "2023-06-21T18:00:00Z"
STEP_SECONDS = 1.25
#: Steps the stub forecast runs unless ``STUB_STEPS`` says otherwise;
#: far more than any test waits for.
TOTAL_STEPS = 4000
#: The step at which the second frame is committed.
SECOND_FRAME_STEP = 5
_PRODUCTS = ("composite_reflectivity", "2m_temperature")


def _frame_name(minutes: int) -> str:
    return f"wrfout_d02_2023-06-21_18_{minutes:02d}_00"


def child(outdir: str, config: str, products: str) -> int:
    from datetime import datetime, timedelta

    from gpuwm import go_cli, offline_child_run

    worker = Path(__file__).resolve()

    def render_command(plan, frames=None, **_options):
        return [sys.executable, "-u", str(worker), "render",
                "--out", str(plan["render"]),
                *(str(frame) for frame in frames or ())]

    go_cli.render_command = render_command
    # The stand-in is this child's renderer, so the finalize stage draws
    # with it whether or not this machine has the real one.
    go_cli.render_extra_missing = lambda: None

    def stub_run(args, progress):
        root = Path(args.outdir)
        root.mkdir(parents=True, exist_ok=True)
        start = datetime(2023, 6, 21, 18)
        progress.start(outdir=root, child_config=Path(args.child_config),
                       ratio=12, start_time=start,
                       parent={"run_dir": str(root.parent), "frames": 3},
                       name="Downscale of stop test")
        progress.arm_render(outdir=root,
                            render_products=args.render_products)
        progress.emit("stage_started", stage="initialize",
                      phase="preprocess")
        progress.emit("stage_started", stage="forecast", phase="integrate")

        def commit(minutes: int) -> None:
            frame = root / _frame_name(minutes)
            frame.write_bytes(f"CDF stub child frame {minutes}".encode())
            valid = start + timedelta(minutes=minutes)
            progress.output_committed(
                domain=2, valid_time=valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                path=str(frame), bytes=frame.stat().st_size)

        commit(0)
        began = time.perf_counter()
        steps = int(os.environ.get("STUB_STEPS") or TOTAL_STEPS)
        for step in range(1, steps + 1):
            time.sleep(0.05)
            progress.emit("model_progress", domain=2,
                          model_seconds=step * STEP_SECONDS,
                          run_seconds=steps * STEP_SECONDS,
                          outer_step=step, total_steps=steps,
                          wall_seconds=time.perf_counter() - began)
            print(f"child_step {step}", flush=True)
            if step == SECOND_FRAME_STEP:
                commit(15)
        return {"result": "PASS",
                "outputs": [str(path) for path in sorted(
                    root.glob("wrfout_d02_*"))]}

    offline_child_run._run = stub_run
    parent = Path(outdir).parent
    return offline_child_run.main([
        "--parent-history", str(parent / "wrfout_d01_parent"),
        "--parent-restart", str(parent / "gpuwmrst_d01_parent.npz"),
        "--child-config", config,
        "--parent-grid-ratio", "12",
        "--i-parent-start", "10", "--j-parent-start", "10",
        "--max-boundary-interval-seconds", "3600",
        "--render-products", products,
        "--outdir", outdir])


def render(arguments: list[str]) -> int:
    out = Path(arguments[arguments.index("--out") + 1])
    frame = Path(arguments[-1])
    hold = os.environ.get("STUB_HOLD", "")
    mark = os.environ.get("STUB_MARK", "")
    try:
        if mark:
            (Path(mark) / f"pid-{frame.name}").write_text(
                str(os.getpid()), encoding="utf-8")
        out.mkdir(parents=True, exist_ok=True)
        minutes = int(frame.name.split("_")[-2])
        stem = f"rustwx_wrf_20230621_18z_f{minutes:03d}_d02-250m"
        staged = {product: out / f"{stem}_{product}.png"
                  for product in _PRODUCTS}
        staged["var_wrf_isltyp"] = (
            out / f"{stem}_var_wrf_isltyp_b4233990277fc8e6.png")
        for product, path in staged.items():
            path.write_bytes(b"\x89PNG stub " + f"{frame.name} {product}"
                             .encode())
        if hold and hold == frame.name:
            if mark:
                (Path(mark) / f"staged-{frame.name}").write_text(
                    "staged\n", encoding="utf-8")
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                time.sleep(0.05)
        from gpuwm.render_receipts import publish_invocation

        written = []
        for product, path in staged.items():
            target = (out / "d02-250m" / product / "20230621"
                      / path.name.replace(f"{stem}_", f"f{minutes:03d}_"))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
            written.append(target)
        publish_invocation(root=out, engine="rust", requested_spec="all",
                           written=written, failures=[], skipped=[],
                           layout="nested", inputs=[frame])
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    if sys.argv[1] == "child":
        raise SystemExit(child(*sys.argv[2:5]))
    raise SystemExit(render(sys.argv[2:]))

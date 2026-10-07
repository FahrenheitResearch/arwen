"""Nowcast frame producers: the runner side of ``gpuwm-obs.nowcast-frames.v1``.

``frames`` is the receipt writer, input ledger and causality check
(standard library and numpy only, testable without any model package).
``run_nowcast`` is the orchestration over a third-party nowcast model; it
runs in its own virtual environment and is never imported by the engine.
"""

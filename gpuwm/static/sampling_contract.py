"""Identity of the arithmetic used to sample a prepared static footprint."""
PORTABLE_SAMPLING_CONTRACT = "wps-sampling-portable-v1"


def current_sampling_contract():
    from gpuwm.static import rust_bridge
    if rust_bridge.python_fallback_requested() or rust_bridge.unavailable_reason() is not None:
        return "python-platform-sampling"
    return PORTABLE_SAMPLING_CONTRACT


def require_relocation_sampling_contract(recorded, *, same_process=False):
    """Require matching arithmetic, and portability for statics read from disk.

    An in-memory preparation and its moves share one process, so the Python
    sampler is also valid when both sides use it. A stored tree must carry the
    portable contract because its preparation may have used another machine.
    """
    running = current_sampling_contract()
    if (recorded is None or recorded != running
            or (not same_process and running != PORTABLE_SAMPLING_CONTRACT)):
        raise ValueError(
            "Moving-nest statics have an incompatible sampling contract "
            f"(prepared={recorded!r}, running={running!r}). The first move "
            "would fail the overlap-statics equality check because its "
            "terrain and climatologies would differ from the prepared footprint. "
            "Use the current Rust static-fields bridge, disable GPUWM_STATIC_PYTHON, "
            "and re-prepare the moving-nest tree: gpuwm go CONFIG --outdir NEW_DIRECTORY (omit --prepared-root)")

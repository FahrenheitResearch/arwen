"""Can ``verify_topology`` actually FIRE?

The gate runs ``graph_verify_topology=True`` as a POSITIVE only -- on a case
where the cached graph is expected to still describe the step, so a mode that
was hard-wired to agree would pass it.  ``verify_topology`` is what graphcap
offers as the proof that :func:`cadence_key` is COMPLETE rather than merely
plausible, so a mode that cannot fail would leave that claim resting on
nothing.

The construction that must trip it: ``graph_key="none"`` with
``graph_reuse="run"`` on the fast-cadence rung.  One graph is then cached for
the whole run, and cumulus is due on alternate steps at cudt=0.1 min, so the
step's true launch sequence really does differ from the cached graph's on
every second step.  ``verify_topology`` recaptures and compares structural
fingerprints, so it must raise GraphCaptureError.  If instead the case merely
comes out numerically wrong -- which is what the same settings without
verify_topology already do -- then verify_topology added nothing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def one(label, **kw):
    from tilestream import graphcap
    from tilestream import test_gate as tg

    try:
        rec = tg.graph_case(kind="physics", rung="full fast cadence",
                            tile_nx=48, tile_ny=40, nsteps=3, **kw)
    except graphcap.GraphCaptureError as exc:
        print(f"{label:46s} RAISED GraphCaptureError -- {str(exc)[:110]}",
              flush=True)
        return "raised"
    except Exception as exc:                                # noqa: BLE001
        print(f"{label:46s} raised {type(exc).__name__}: {str(exc)[:90]}",
              flush=True)
        return "other"
    print(f"{label:46s} completed, bitexact={rec['bitexact']}, "
          f"differing={len(rec.get('differing', []))}, "
          f"maxabs={rec.get('max_abs', 0.0):.3e}", flush=True)
    return "ok"


def main() -> int:
    # Control: the same broken key WITHOUT verify_topology.  This is the
    # gate's own negative control and it is caught only as a digest
    # mismatch, i.e. after the wrong kernels have already run.
    a = one("key=none reuse=run, verify OFF",
            graph_key="none", graph_reuse="run")
    # The claim under test: with verify_topology on, the same run must be
    # REFUSED before it can produce a wrong number.
    b = one("key=none reuse=run, verify ON",
            graph_key="none", graph_reuse="run",
            graph_verify_topology=True)
    print()
    print("VERDICT:", "verify_topology CAN fire" if b == "raised"
          else "verify_topology did NOT fire where the topology really moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

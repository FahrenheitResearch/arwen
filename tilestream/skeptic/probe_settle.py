"""Does the gate ever EXERCISE the mechanisms graphcap says it needs?

Removing the one-shot ``settle`` graph entirely leaves ``test_gate
--graph-only`` green, which has two very different explanations:

  * the gate cannot see the difference (a defence with no control), or
  * the settling case never arises in any gated configuration, so the
    mutation was a no-op (a defence the gate never exercises).

This counts, per gate graph case, how many captures actually produced a
settling graph and how many produced host DRIFT -- the two branches of
:func:`tilestream.graphcap.capture_step` that the module docstring rests on.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

COUNTS = {"captures": 0, "settled": 0, "drifted": 0, "verify_host": 0}


def main() -> int:
    from tilestream import graphcap

    real = graphcap.capture_step

    def counting(*a, **kw):
        got = real(*a, **kw)
        COUNTS["captures"] += 1
        COUNTS["verify_host"] += int(kw.get("verify_host", True))
        COUNTS["settled"] += int(got.settle is not None)
        COUNTS["drifted"] += int(bool(got.drift))
        return got

    graphcap.capture_step = counting
    import tilestream.test_gate as tg
    tg.graphcap = graphcap

    for label, kwargs, expect in tg.GRAPH_CASES:
        for k in COUNTS:
            COUNTS[k] = 0
        try:
            rec = tg.graph_case(**kwargs)
            ok = rec["bitexact"] == expect and rec["graph_ok"]
        except Exception as exc:                            # noqa: BLE001
            print(f"{'ERR':5s} {label[:58]:58s} {type(exc).__name__}",
                  flush=True)
            continue
        print(f"{'ok' if ok else 'BAD':5s} {label[:58]:58s} "
              f"captures={COUNTS['captures']:3d} "
              f"verify_host={COUNTS['verify_host']:3d} "
              f"SETTLED={COUNTS['settled']:3d} "
              f"drifted={COUNTS['drifted']:3d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

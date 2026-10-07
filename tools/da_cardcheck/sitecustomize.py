"""Measurement shim (Lane 0 identity): every device analysis the cycle runs on
several cards is solved again on card 0 alone, in the same process, from the
same inputs, and the two are compared byte for byte (increments, positivity
receipt, filter diagnostics).  Read-only for the run: the production result is
returned unchanged.  Active only when this directory is on PYTHONPATH, with
CARDCHECK_OUT naming where call-NNN.json records go (check1b on box L,
2026-10-06: both analyses identical on 5 cards and on card 0)."""
import importlib.abc, json, os, sys, time

_OUT = os.environ.get("CARDCHECK_OUT", "/tmp/cardcheck")
_TARGET = "gpuwm.da.radar_assimilation"


def _patch(ra):
    # The route shadow (tools.da_letkf_route_shadow) replaces
    # ra._execute_analysis after import and calls ra._analysis_attempt
    # directly, so the check sits on _analysis_attempt, which every route
    # reaches (check1 of 2026-10-06 patched _execute_analysis and never fired).
    original = ra._analysis_attempt
    calls = {"n": 0}

    def wrapped(solver, prior, batches, geometry, config, *, namespace, storage,
                supports_staging, progress, diagnostics=None, device_options=None):
        out = original(solver, prior, batches, geometry, config, namespace=namespace,
                       storage=storage, supports_staging=supports_staging,
                       progress=progress, diagnostics=diagnostics,
                       device_options=device_options)
        opts = dict(device_options or {})
        if storage != "cuda-obs-sparse":
            return out
        calls["n"] += 1
        from gpuwm.da.letkf import LetkfDiagnostics
        from gpuwm.da.positivity import DevicePositivity
        prod_diag = out[1]
        one = dict(opts, devices=(0,))
        hook = opts.get("chunk_hook")
        if isinstance(hook, DevicePositivity):
            one["chunk_hook"] = DevicePositivity(hook.policy, fields=hook.fields)
        t0 = time.time()
        again = original(solver, prior, batches, geometry, config, namespace=namespace,
                         storage=storage, supports_staging=supports_staging,
                         progress=None, diagnostics=LetkfDiagnostics(),
                         device_options=one)
        diag1 = again[1]
        rows = {}
        for name, value in out[0].items():
            other = again[0][name]
            rows[name] = bool(value.dtype == other.dtype and value.shape == other.shape
                              and value.tobytes() == other.tobytes())
        keys = ("prior_spread", "posterior_spread", "mean_increment_rms", "active_points",
                "max_local_obs", "chunk_hook_result", "withheld")
        same = {k: getattr(prod_diag, k, None) == getattr(diag1, k, None) for k in keys}
        record = {"call": calls["n"], "fields": list(out[0]),
                  "devices_production": getattr(prod_diag, "devices", None),
                  "device_chunk_counts": getattr(prod_diag, "device_chunk_counts", None),
                  "one_card_devices": getattr(diag1, "devices", None),
                  "increments_identical": rows, "all_identical": all(rows.values()),
                  "diagnostics_identical": same,
                  "production_solve_seconds": getattr(prod_diag, "solve_seconds", None),
                  "one_card_solve_seconds": getattr(diag1, "solve_seconds", None),
                  "recheck_wall_seconds": round(time.time() - t0, 3)}
        os.makedirs(_OUT, exist_ok=True)
        with open(os.path.join(_OUT, f"call-{calls['n']:03d}.json"), "w") as f:
            json.dump(record, f, indent=1, default=str)
        print(f"cardcheck call {calls['n']}: identical={record['all_identical']} "
              f"diag={all(same.values())} devices={record['devices_production']}", flush=True)
        del again
        return out

    ra._analysis_attempt = wrapped


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            spec = f.find_spec(name, path, target) if hasattr(f, "find_spec") else None
            if spec is not None and spec.loader is not None:
                orig = spec.loader.exec_module

                def exec_module(module, _o=orig):
                    _o(module)
                    _patch(module)
                spec.loader.exec_module = exec_module
                return spec
        return None


sys.meta_path.insert(0, _Finder())

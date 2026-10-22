"""CUDA driver for explicitly labelled saved-frame FRAM accumulations."""
from pathlib import Path
import numpy as np
from gpuwm.certify.kernel_manifest import record_module

FIELDS = ("freezing_rain_total", "ice_accretion_horizontal", "ice_accretion")
CONTRACT = {
    "method": "saved-frame-right-endpoint",
    "precipitation": "interval difference of RAINNC + RAINC + RAINSH, mm",
    "precipitation_type": "lowest-level microphysics fall flux, liquid fraction >= 0.7, rate >= 0.1 mm/h",
    "surface_gate": "T2 < 273.15 K or TSK < 273.15 K",
    "wetbulb": "pressure-aware psychrometric bisection over water, endpoint T2/Q2/PSFC",
    "ice_model": "FRAM equations (1)-(8), doi:10.1175/WAF-D-15-0118.1",
    "ice_accretion": "equivalent radial ice, mm, 0.394 * elevated horizontal ice",
    "calibration": "FRAM hourly continuous freezing rain >= 0.5 mm/h; other rates/intervals are extrapolation",
    "losses": "gross accretion; no melt, shedding, sublimation or ground icing",
    "missing": "no zero fill across missing inputs or unobserved start",
}


def memory_plan(shape):
    """One serialized callback workspace per device and geometry."""
    from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
    names = ("T2", "Q2", "PSFC", "U10", "V10", "QRAIN", "QSNOW", "TSK", "QGRAUP",
             "zero", "previous_rain", *FIELDS, "liquid_out", "horizontal_out", "radial_out",
             "wetbulb_out", "rate_out")
    return BatchMemoryPlan(tuple(BatchArraySpec(f"ice:{n}", tuple(shape), "shared") for n in names), reserved_bytes=0)


class SavedFrameIce:
    """Only CUDA launch and storage here; weather arithmetic lives in CUDA."""
    def __init__(self, array_module):
        self.xp = array_module
        self.history = {}
        self.kernels = {}
        self.intervals = set()
        self.last_interval = None
        self.workspaces = {}

    def invalidate(self, key):
        self.last_interval = None
        prior = self.history.get(key)
        if prior is not None:
            for name in FIELDS:
                prior[name].fill(np.nan)

    def _kernel(self, device):
        if device not in self.kernels:
            source = (Path(__file__).parents[1] / "core/kernels/ice_accretion.cu").read_text()
            module = self.xp.RawModule(code=source, options=("--std=c++17",))
            self.kernels[device] = module.get_function("saved_frame_ice")
            record_module("gpuwm.ensemble.ice_accretion:saved-frame", source=source,
                          options=("--std=c++17",), module=module)
        return self.kernels[device]

    def submit(self, key, tick, fields, rain):
        xp = self.xp
        # The sampling interval THIS submission accumulated over (None for a
        # member's first frame): a frame's product records its own members'
        # intervals, not the collector's running set, which depends on the
        # order members reach their frames (native pack vs ordinary runner).
        self.last_interval = None
        required = ("T2", "Q2", "PSFC", "U10", "V10", "QRAIN", "QSNOW")
        if not all(name in fields for name in required):
            # Returning no diagnostic lets the native manifest name it as
            # unavailable rather than constructing a false zero ice hazard.
            self.invalidate(key)
            return {}
        shape = rain.shape
        device = int(xp.cuda.runtime.getDevice())
        workspace_key = (device, tuple(shape))
        if workspace_key not in self.workspaces:
            from gpuwm.ensemble.batch_storage import BatchStorage
            self.workspaces[workspace_key] = BatchStorage(memory_plan(shape), 1, array_module=xp,
                available_bytes=int(xp.cuda.runtime.memGetInfo()[0]))
        buffers = self.workspaces[workspace_key].arrays
        get = lambda name: buffers[f"ice:{name}"]
        def surface(name):
            value = fields[name]
            if value.ndim == 3:
                value = value[0]
            if value.shape != shape:
                raise ValueError(f"saved-frame ice {name} grid differs from precipitation")
            target = get(name)
            if isinstance(value, np.ndarray):
                target.set(np.ascontiguousarray(value, dtype=np.float32))
            else:
                xp.copyto(target, value)
            return target
        inputs = {name: surface(name) for name in required}
        zero = get("zero")
        zero.fill(0)
        skin = surface("TSK") if "TSK" in fields else inputs["T2"]
        graupel = surface("QGRAUP") if "QGRAUP" in fields else zero
        prior = self.history.get(key)
        if prior is not None and tick <= prior["tick"]:
            raise ValueError("saved-frame ice clock must advance; duplicate output would double accretion")
        if prior is None:
            # A missing t=0 frame cannot be recovered from the first hourly
            # precipitation/type sample. Keep that uncertainty visible.
            previous_rain = get("previous_rain")
            xp.copyto(previous_rain, rain)
            previous = tuple(get(name) for name in FIELDS)
            for value in previous:
                value.fill(0 if tick == 0 else xp.nan)
            hours = 0.0
        else:
            hours = (tick-prior["tick"]) / 3_600_000_000
            self.intervals.add(hours)
            self.last_interval = hours
            previous_rain = get("previous_rain")
            previous_rain.set(prior["rain"])
            previous = tuple(get(name) for name in FIELDS)
            for name, value in zip(FIELDS, previous):
                value.set(prior[name])
        outputs = tuple(get(name+"_out") for name in ("liquid", "horizontal", "radial", "wetbulb", "rate"))
        count = int(rain.size)
        self._kernel(device)(
            ((count+255)//256,), (256,),
            (rain, previous_rain, *previous,
             inputs["T2"], skin, inputs["Q2"], inputs["PSFC"], inputs["U10"], inputs["V10"],
             inputs["QRAIN"], inputs["QSNOW"], graupel, *outputs,
             np.uint64(count), np.float64(hours), np.int32("TSK" in fields)))
        self.history[key] = {"tick": tick, "rain": xp.asnumpy(rain),
                            **{name: xp.asnumpy(array) for name, array in zip(FIELDS, outputs)}}
        return {**dict(zip(FIELDS, outputs)), "freezing_rain_rate": outputs[4]}


def main(argv=None):
    """Public observation-binding door; parsing and scoring stay in Rust."""
    import argparse
    import os
    import subprocess
    from gpuwm import bridges
    from gpuwm.rustwx import crate_dir
    parser = argparse.ArgumentParser(description="Bind native ice products to ASOS and mPING reports")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("request", type=Path)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args(argv)
    name = bridges.executable_name("rw_ice_verify")
    found = None
    if not os.environ.get("GPUWM_ICE_VERIFY"):
        found = next((p for p in (crate_dir()/"target"/"release"/name,
                                  crate_dir()/"target"/"debug"/name) if p.is_file()), None)
    found = found or bridges.find_artifact("GPUWM_ICE_VERIFY", name)
    if found is None:
        parser.error("rw_ice_verify is missing; observation reports cannot be matched or scored. "
                     "Stage it with gpuwm fetch-bridges, or build tools/rustwx with "
                     "cargo build --release -p rw-obs --bin rw_ice_verify")
    usable, detail = bridges.bridge_abi_matches("rw_ice_verify", found)
    if not usable:
        parser.error(f"{found}: {detail}; this binary cannot bind the native ice contract")
    return subprocess.run([str(found), str(args.request), str(args.outdir)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

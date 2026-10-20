"""Earth2Studio's data/coordinate/IO APIs, with diagnostic identity from a row.

Executed in GPUWM_EARTH2_PYTHON. No gpuwm imports are required there.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    row = request["row"]
    # Load the same policy without requiring gpuwm in the E2S environment.
    import importlib.util
    policy_spec = importlib.util.spec_from_file_location(
        "earth2_licensing", Path(__file__).with_name("earth2_licensing.py"))
    policy = importlib.util.module_from_spec(policy_spec)
    policy_spec.loader.exec_module(policy)
    policy.require_licence(request["model"], row, request.get("accept_licence", ()))
    # Preserve non-licence refusals even under explicit licence acceptance.
    if row.get("runtime_refusal"):
        raise ValueError(row["runtime_refusal"])
    if row.get("weights", {}).get("account_needed"):
        raise ValueError("diagnostic weights require an account")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import numpy as np
    import torch
    import xarray as xr
    from earth2studio.data import InferenceOutputSource
    from earth2studio.data.utils import fetch_data
    from earth2studio.io import NetCDF4Backend
    from earth2studio.models import dx
    from earth2studio.models.auto import Package
    from earth2studio.utils.coords import map_coords, split_coords
    start = time.monotonic()
    cls = getattr(dx, row["earth2studio_class"])
    uri = row.get("weights", {}).get("uri")
    if row.get("initialization") == "constructor":
        model = cls(**row.get("constructor_kwargs", {}))
    else:
        package = Package(uri) if uri else cls.load_default_package()
        model = cls.load_model(package, **row.get("load_model_kwargs", {}))
    device = torch.device(request.get("device", "cuda"))
    model = model.to(device)
    model.eval()
    coords = model.input_coords()
    dataset = xr.open_dataset(request["input"], engine="zarr" if Path(request["input"]).suffix == ".zarr" else None)
    # Exporters may store one tensor or separate lexicon variables. Only
    # reshape the tensor's named variable axis; E2S owns time and data access.
    if "variable" in dataset.dims:
        tensors = [name for name, value in dataset.data_vars.items() if "variable" in value.dims]
        if len(tensors) != 1:
            raise ValueError("Earth2Studio export must contain one tensor with a variable axis")
        dataset = dataset[tensors[0]].to_dataset(dim="variable")
    source = InferenceOutputSource(dataset)
    times = source.da.coords["time"].values
    io = None
    try:
        for moment in times:
            # Empty spatial coordinates mean the model accepts the export's
            # grid. Earth2Studio owns coordinate selection and batch packing.
            x, actual = fetch_data(source=source, time=[moment], variable=coords["variable"],
                lead_time=np.asarray([np.timedelta64(0, "h")]), device=device)
            x, actual = map_coords(x, actual, coords)
            if not torch.isfinite(x).all().item() and row.get("missing_input_policy") != "propagate":
                raise ValueError("diagnostic input does not cover the model's complete finite input grid")
            if not torch.isfinite(x).any().item():
                raise ValueError("diagnostic input contains no finite data")
            with torch.inference_mode():
                x, actual = model(x, actual)
            actual["time"] = actual["time"].astype("datetime64[us]")
            actual["lead_time"] = actual["lead_time"].astype("timedelta64[h]")
            if io is None:
                output = actual.copy()
                variables = output.pop("variable")
                output["time"] = times.astype("datetime64[us]")
                io = NetCDF4Backend(request["output"], backend_kwargs={"mode": "w"})
                io.add_array(output, variables)
                for name, attributes in row.get("output_attributes", {}).items():
                    if name in io.root.variables:
                        for attribute, value in attributes.items():
                            io.root[name].setncattr(attribute, value)
                for key, units in (("lat", "degrees_north"), ("lon", "degrees_east")):
                    if key in io.root.variables:
                        io.root[key].setncattr("units", units)
                io.root.setncattr("Conventions", "CF-1.8")
                io.root.setncattr("source", "Earth2Studio " + row["earth2studio_class"])
            io.write(*split_coords(x, actual))
        if io is None:
            raise ValueError("diagnostic export contains no valid times")
    finally:
        if io is not None:
            io.close()
        dataset.close()
    report = dict(schema="gpuwm-earth2-diagnostic-v1", model=request["model"], row=row,
        request_sha256=hashlib.sha256(Path(request_path).read_bytes()).hexdigest(),
        output=request["output"], output_sha256=hashlib.sha256(Path(request["output"]).read_bytes()).hexdigest(),
        output_bytes=Path(request["output"]).stat().st_size,
        valid_times=[str(t) for t in times], frames=len(times), wall_seconds=time.monotonic()-start,
        peak_vram_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0)
    Path(request["output"]).with_suffix(".receipt.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main(sys.argv[1])

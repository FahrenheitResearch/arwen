"""Coordinate compatibility for Earth2Studio 0.19's pregridded IO path.

The exporter has already regridded these fields in Rust. Earth2Studio 0.19
otherwise tries another interpolation, using coordinate names its own HRRR
source does not supply. Keep its loader and tensor conversion; select a
matching target window and repair only the spatial coordinate dictionary.
"""
from contextlib import contextmanager
from collections import OrderedDict


@contextmanager
def pregridded_conditioning():
    """Use around Earth2Studio inference with an already matching 2-D grid.

    Target index coordinates and both geographic coordinates must agree.
    Other inputs use Earth2Studio's original conversion unchanged. No model
    class, channel list or weather-field transformation lives in this hook.
    The installed Earth2Studio environment is never modified.
    """
    import torch
    from earth2studio.data import utils

    original = utils.prep_data_array

    def prepare(da, device="cpu", interp_to=None, interp_method="nearest"):
        if interp_to is None or "lat" in da.dims:
            return original(da, device, interp_to, interp_method)
        lat_name = "lat" if "lat" in da.coords else "_lat"
        lon_name = "lon" if "lon" in da.coords else "_lon"
        if lat_name not in da.coords or lon_name not in da.coords:
            return original(da, device, interp_to, interp_method)
        spatial = da[lat_name].dims
        if (len(spatial) != 2 or da[lon_name].dims != spatial
                or any(d not in interp_to for d in spatial)
                or "_lat" not in interp_to or "_lon" not in interp_to):
            return original(da, device, interp_to, interp_method)
        try:
            selected = da.sel({d: interp_to[d] for d in spatial})
        except KeyError:
            return original(da, device, interp_to, interp_method)
        # Coordinate checks use Torch's parallel implementation. Values are
        # left to Earth2Studio's own loader and tensor conversion below.
        for name, target in ((lat_name, "_lat"), (lon_name, "_lon")):
            source_coord = torch.as_tensor(selected[name].values, device=device)
            target_coord = torch.as_tensor(interp_to[target], device=device)
            if (source_coord.shape != target_coord.shape
                    or not torch.allclose(source_coord, target_coord.to(source_coord.dtype),
                                          rtol=0, atol=1e-5)):
                return original(da, device, interp_to, interp_method)
        tensor, coords = original(selected, device, None, interp_method)
        # The legacy inference contract names the last two tensor axes
        # lat/lon, while retaining the real 2-D geography under _lat/_lon.
        coords = OrderedDict((d, coords[d]) for d in selected.dims
                             if d not in spatial)
        coords["lat"] = selected[spatial[0]].values
        coords["lon"] = selected[spatial[1]].values
        coords["_lat"] = selected[lat_name].values
        coords["_lon"] = selected[lon_name].values
        return tensor, coords

    utils.prep_data_array = prepare
    try:
        yield
    finally:
        utils.prep_data_array = original

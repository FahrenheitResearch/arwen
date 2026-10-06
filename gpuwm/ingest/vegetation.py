"""Select analyzed initial vegetation with the existing climatology fallback."""
import numpy as np


def initial_vegetation_fraction(met, static, valid_time):
    """Use the analyzed fraction in percent when the source carries one."""
    field = met.fields.get("VEGFRA")
    if field is None:
        from gpuwm.static.build import monthly_interp_to_date
        return 100.0 * monthly_interp_to_date(static["GREENFRAC"], valid_time)
    host = field.get() if hasattr(field, "get") else field
    values = np.asarray(host)
    if (values.shape != np.shape(static["LANDMASK"])
            or not np.isfinite(values).all()
            or (values < 0.0).any() or (values > 100.0).any()):
        raise ValueError("analyzed VEGFRA must be a finite mass-grid fraction "
                         "in percent between 0 and 100; other values would "
                         "initialize a different vegetated area")
    return field

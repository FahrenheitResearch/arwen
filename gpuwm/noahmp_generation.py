"""Numerical generations for the Noah-MP snow-aging call placement.

This selector changes only call placement. It does not select the complete
Noah-MP 5.1.2 model shipped with WRF v4.8.0.
"""

NOAHMP_SNOW_AGE_GENERATIONS = ("wrf_470", "wrf_461")


def snow_age_generation(value):
    if value not in NOAHMP_SNOW_AGE_GENERATIONS:
        raise ValueError(
            f"noahmp_snow_age={value!r} must be 'wrf_470' (night aging) or "
            "'wrf_461' (daylight-only reproduction); substituting a generation "
            "would change TAUSS and the following dawn's BATS albedo")
    return value


def radiation_kernel(value):
    return ("noahmp_radiation_batch_night_age" if
            snow_age_generation(value) == "wrf_470" else "noahmp_radiation_batch")

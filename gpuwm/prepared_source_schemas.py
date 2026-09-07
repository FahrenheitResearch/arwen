"""Prepared input identities shared by source staging and forecast readers."""
from gpuwm.source_adapters import packaged_profile_sources
from gpuwm.source_authorities import packaged_profile


def composed_packaged_profiles():
    return {
        source: profile_id
        for source, profile_id in packaged_profile_sources().items()
        if packaged_profile(profile_id)["composition_state"] == "composed"
    }

# All mapped bundles share the same preparation and forecast implementation.
# A packaged name is an exact identity claim; caller-authored authorities are
# reported as ``mapped`` and bound to their own sealed input/cache receipts.
def mapped_sources():
    return frozenset({"mapped", *composed_packaged_profiles()})

def source_schemas(sources=None):
    return {
        # Any composed mapped preparation writes this one; the packaged
        # profile, not the schema, is what says WHICH source it is.
        **{source: "gpuwm-mapped-composition-inputs-v1"
           for source in (mapped_sources() if sources is None else sources)},
        "gfs": "gpuwm-gfs-direct-input-manifest-v1",
        "era5": "gpuwm-era5-direct-input-manifest-v1",
        "20crv3": "gpuwm-20crv3-grib2-inputs-v1",
        "hrrr": "gpuwm-hrrr-native-input-manifest-v1",
    }

"""Near-surface HRRR field transfers shared by cached forecast runners."""

from __future__ import annotations


def surface_fields_to_device(met, array_module):
    """Upload restored or freshly mapped near-surface fields to a device."""

    return {
        name: array_module.asarray(
            met.fields[name], dtype=array_module.float32)
        for name in ("T2", "U10", "V10")
    }


__all__ = ["surface_fields_to_device"]

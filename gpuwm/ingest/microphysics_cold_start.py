"""Exact initial values for prognostic species absent from analyzed input.

These are state-allocation contracts, independent of the source name and of
radiation, turbulence, surface or cumulus presets. WRF v4.6.1 Registry.EM_COMMON
lines 3025/3031 bind Milbrandt/WDM6 qnc to QNCLOUD (line 542); NSSL
uses the distinct qndrop/QNDROP declaration at lines 521-522.
An analyzed mass inventory is validated separately by initialize_real and its correspondence receipt.
"""
from __future__ import annotations

from types import SimpleNamespace
import struct


def source_absent_microphysics(cfg):
    """Return (WRF field names, native FP32 initial values) for active extras."""
    mp = int(cfg.mp_physics)
    if mp in (0, 1, 6):
        return (), {}
    if mp == 8:
        return ("QNICE", "QNRAIN"), {"ni": 0.0, "nr": 0.0}
    if mp == 9:
        return (("QHAIL", "QNCLOUD", "QNRAIN", "QNICE", "QNSNOW",
                 "QNGRAUPEL", "QNHAIL"),
                dict.fromkeys(("qh", "nc", "nr", "ni", "ns", "ng", "nh"), 0.0))
    if mp == 10:
        return (("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"),
                dict.fromkeys(("nc", "nr", "ni", "ns", "ng"), 0.0))
    if mp == 16:
        return (("QNCLOUD", "QNRAIN", "QNCCN"),
                {"nc": 0.0, "nr": 0.0, "nn": float(cfg.wdm6_ccn_conc)})
    if mp == 18:
        from gpuwm.core.microphysics_transition import NSSL2_BACKGROUND_CCN_PER_KG
        values = dict.fromkeys(("qh", "qndrop", "qnr", "qni", "qns", "qng",
                                "qnh", "qnn", "qvolg", "qvolh"), 0.0)
        values["qnn"] = NSSL2_BACKGROUND_CCN_PER_KG
        return (("QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL",
                 "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL"), values)
    if mp == 50:
        return (("QNICE", "QNRAIN", "QIR", "QIB"),
                dict.fromkeys(("ni", "nr", "qir", "qib"), 0.0))
    if mp == 28:
        raise ValueError("analyzed input lacks the aerosol-aware Thompson boundary "
                         "species; supply a source with the required aerosol forcing")
    raise ValueError(f"no native prognostic-species initialization for mp_physics={mp}")


def cold_start_contract(selection):
    """Wire-level expected FP32 values and bits, computed from active settings."""
    cfg = SimpleNamespace(**selection) if isinstance(selection, dict) else selection
    fields, values = source_absent_microphysics(cfg)
    expected = {}
    for name, value in values.items():
        packed = struct.pack("<f", value)
        expected[name] = (struct.unpack("<f", packed)[0], struct.unpack("<I", packed)[0])
    return fields, expected

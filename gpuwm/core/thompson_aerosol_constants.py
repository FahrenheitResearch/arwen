"""The three ``mp_physics=28`` scalars a mixed nest edge seeds from.

WRF v4.6.1 ``phys/module_mp_thompson.F`` fixes each of them once, and the
line numbers are carried so the transcription stays checkable:

``NT_C``        :88    the constant droplet number classic Thompson runs
                        with, m-3, and the ELSE-branch ``nc`` when the
                        driver is not aerosol aware (:1248-1255);
``NWFA_FLOOR``  :1805  ``naCCN1*0.222``, the water-friendly aerosol floor;
``NIFA_FLOOR``  :1806  ``naIN1*0.01``, the ice-friendly aerosol floor.

They live in a module of their own, with no imports, because two readers
need them and only one of them may reach the scheme.
:mod:`gpuwm.core.thompson_aerosol_contract` re-exports them beside the rest
of the mp=28 table contract; :mod:`gpuwm.core.microphysics_transition`
reads them for the entry closure of a nest edge whose target is mp=28.
The transition module is staged into the standalone RW-WPS preparation
wheel and the contract module is not (it carries a correctly rounded libm
the preparation wheel has no use for), so the edge reads the constants
from here and the wheel's staging gate has nothing unresolved to refuse.
"""

from __future__ import annotations

NT_C = 100.0e6            # :88   the mp=8 constant droplet number, m-3
NWFA_FLOOR = 11.1e6       # :1805 == naCCN1*0.222
NIFA_FLOOR = 5.0e3        # :1806 == naIN1*0.01

__all__ = ["NIFA_FLOOR", "NT_C", "NWFA_FLOOR"]

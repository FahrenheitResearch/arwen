"""Physical constants, matching WRF share/module_model_constants.F exactly.

``module_model_constants.F`` declares its derived constants as ``REAL
PARAMETER``, and gfortran folds a parameter expression *in the declared kind*:
every operation in the chain is rounded to float32, not just the result.  A
Python double expression rounded once at the end is a third function --
neither the float32 chain nor the correctly rounded double -- so a constant
WRF derives by float32 arithmetic has to be spelled the same way here.

Verified rather than assumed, on the pinned v4.6.1 tree
(``d66e442fccc04111067e29274c9f9eaccc3cef28``).  A probe compiled at ``-O0``
against the byte-unmodified module prints ``transfer(rvovrd, i)`` as
``3FCDDED2``; the same program routes ``r_v`` and ``r_d`` through an opaque
subroutine so the divide must happen on the hardware and prints ``3FCDDED2``
again; and it prints ``real(461.6d0/287.0d0)`` as ``3FCDDED1``.  All eight
derived constants below were checked against that probe.  Seven agree either
way -- ``cp``, ``cv``, ``cpovcv``, ``rcp``, ``rcv``, ``reradius``, ``EP_2``.
``rvovrd`` does not, because 461.6 is not representable in either format, and
its two spellings land one ULP apart.  ``tests/test_constants.py`` holds all
eight to the float32 chain so the next one cannot slip in unnoticed.
"""
import numpy as _np

G = 9.81
RD = 287.0
RV = 461.6
CP = 7.0 * RD / 2.0        # 1004.5
CV = CP - RD               # 717.5
P0 = 1.0e5
T0 = 300.0
GAMMA = CP / CV
RCP = RD / CP
RCV = RD / CV
#: WRF ``rvovrd = r_v/r_d`` (module_model_constants.F:41): a float32 divide of
#: two float32 values, not a double divide rounded once.  The divide is
#: spelled here rather than its answer, so this is still derived and never
#: hardcoded.  Read by the EOS diagnostic (WRF ``calc_p_rho_phi``'s
#: ``qvf = 1.+rvovrd*qv``) and by the real-data initialization; ``EP_1``
#: consumers -- YSU, MYNN, the surface layer -- form ``rvovrd - 1`` and
#: inherit twice the error, because the subtract keeps the absolute gap while
#: dropping the exponent.
RVOVRD = float(_np.float32(_np.float32(RV) / _np.float32(RD)))
RERADIUS = 1.0 / 6370.0e3  # reciprocal earth radius (m^-1); curvature terms

# Microphysics constants (module_model_constants.F; Kessler consumes them
# through the WRF microphysics driver's argument list).
XLV = 2.5e6                # latent heat of vaporization at 0 C (J/kg)
SVP1 = 0.6112              # Teten saturation vapor pressure constants
SVP2 = 17.67
SVP3 = 29.65               # (K)
SVPT0 = 273.15             # (K)
RHOWATER = 1000.0          # density of liquid water (kg/m^3)
EP2 = RD / RV              # 0.6217... (WRF ep_2; never hardcoded)

CUDA_DEFINES = {
    "G": G, "RD": RD, "RV": RV, "CP": CP, "CV": CV,
    "P0": P0, "T0": T0, "GAMMA": GAMMA, "RCP": RCP, "RCV": RCV,
    "RVOVRD": RVOVRD, "RERADIUS": RERADIUS,
    "XLV": XLV, "SVP1": SVP1, "SVP2": SVP2, "SVP3": SVP3, "SVPT0": SVPT0,
    "RHOWATER": RHOWATER, "EP2": EP2,
}

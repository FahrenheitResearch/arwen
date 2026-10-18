"""WRF v4.6.1 WSM6 coefficient initialization.

The values and operation order below follow ``mp_wsm6_init`` in
``phys/physics_mmm/mp_wsm6.F90``.  Keeping the derived constants on the
host makes the hail/graupel selection explicit and lets restart identity
bind the actual coefficient set instead of only the namelist integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math


@dataclass(frozen=True)
class WSM6RimedIceConstants:
    n0g: float
    deng: float
    avtg: float
    bvtg: float
    lamdagmax: float


@dataclass(frozen=True)
class WSM6Coefficients:
    rimed: WSM6RimedIceConstants
    xlv1: float
    qc0: float
    qck1: float
    pidnc: float
    pidn0r: float
    pidn0s: float
    pidn0g: float
    pvtr: float
    pvts: float
    pvtg: float
    pacrr: float
    pacrs: float
    pacrg: float
    pacrc: float
    precr1: float
    precr2: float
    precs1: float
    precs2: float
    precg1: float
    precg2: float
    roqimax: float
    rslopermax: float
    rslopesmax: float
    rslopegmax: float
    rsloperbmax: float
    rslopesbmax: float
    rslopegbmax: float
    rsloper2max: float
    rslopes2max: float
    rslopeg2max: float
    rsloper3max: float
    rslopes3max: float
    rslopeg3max: float


def _rgmma(x: float) -> float:
    """WSM6's Weierstrass-product Gamma implementation, verbatim."""
    if x == 1.0:
        return 0.0
    value = x * math.exp(0.577215664901532 * x)
    for index in range(1, 10001):
        y = float(index)
        value *= (1.0 + x / y) * math.exp(-x / y)
    return 1.0 / value


def rimed_ice_constants(hail_opt: int) -> WSM6RimedIceConstants:
    """Return WRF's five ``hail_opt``-selected WSM6 constants."""
    if int(hail_opt) == 0:
        return WSM6RimedIceConstants(4.0e6, 500.0, 330.0, 0.8, 6.0e4)
    if int(hail_opt) == 1:
        return WSM6RimedIceConstants(4.0e4, 700.0, 285.0, 0.8, 2.0e4)
    raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")


@lru_cache(maxsize=2)
def coefficients(hail_opt: int = 0) -> WSM6Coefficients:
    """Reproduce ``mp_wsm6_init`` for liquid density 1000 kg m-3."""
    r = rimed_ice_constants(hail_opt)
    pi = 4.0 * math.atan(1.0)
    den0, denr, dens = 1.28, 1000.0, 100.0
    cliq, cpv = 4190.0, 1846.4
    n0r, n0s = 8.0e6, 2.0e6
    avtr, bvtr, avts, bvts = 841.9, 0.8, 11.72, 0.41

    g3r, g4r = _rgmma(3.0 + bvtr), _rgmma(4.0 + bvtr)
    g5r2 = _rgmma(2.5 + 0.5 * bvtr)
    g3s, g4s = _rgmma(3.0 + bvts), _rgmma(4.0 + bvts)
    g5s2 = _rgmma(2.5 + 0.5 * bvts)
    g3g, g4g = _rgmma(3.0 + r.bvtg), _rgmma(4.0 + r.bvtg)
    g5g2 = _rgmma(2.5 + 0.5 * r.bvtg)

    rr, rs, rg = 1.0 / 8.0e4, 1.0 / 1.0e5, 1.0 / r.lamdagmax
    return WSM6Coefficients(
        rimed=r,
        xlv1=cliq - cpv,
        qc0=(4.0 / 3.0) * pi * denr * (0.8e-5 ** 3) * 3.0e8 / den0,
        qck1=(0.104 * 9.8 * 0.55 / ((3.0e8 * denr) ** (1.0 / 3.0))
              / 1.718e-5 * den0 ** (4.0 / 3.0)),
        pidnc=pi * denr / 6.0,
        pidn0r=pi * denr * n0r,
        pidn0s=pi * dens * n0s,
        pidn0g=pi * r.deng * r.n0g,
        pvtr=avtr * g4r / 6.0,
        pvts=avts * g4s / 6.0,
        pvtg=r.avtg * g4g / 6.0,
        pacrr=pi * n0r * avtr * g3r * 0.25,
        pacrs=pi * n0s * avts * g3s * 0.25,
        pacrg=pi * r.n0g * r.avtg * g3g * 0.25,
        pacrc=pi * n0s * avts * g3s * 0.25,
        precr1=2.0 * pi * n0r * 0.78,
        precr2=2.0 * pi * n0r * 0.31 * math.sqrt(avtr) * g5r2,
        precs1=4.0 * n0s * 0.65,
        precs2=4.0 * n0s * 0.44 * math.sqrt(avts) * g5s2,
        precg1=2.0 * pi * r.n0g * 0.78,
        precg2=2.0 * pi * r.n0g * 0.31 * math.sqrt(r.avtg) * g5g2,
        roqimax=2.08e22 * 500.0e-6 ** 8,
        rslopermax=rr, rslopesmax=rs, rslopegmax=rg,
        rsloperbmax=rr ** bvtr, rslopesbmax=rs ** bvts,
        rslopegbmax=rg ** r.bvtg,
        rsloper2max=rr * rr, rslopes2max=rs * rs,
        rslopeg2max=rg * rg,
        rsloper3max=rr ** 3, rslopes3max=rs ** 3,
        rslopeg3max=rg ** 3,
    )


#: The float32 words WRF v4.6.1's ``mp_wsm6_init`` leaves in the mp_wsm6
#: module (phys/physics_mmm/mp_wsm6.F90:73-189), by ``hail_opt``, in the
#: order ``kernels/wsm6.cu``'s ``Wsm6Coef`` enum reads them.
#:
#: They are WRF's own float32 arithmetic -- the Weierstrass-product
#: ``rgmma`` run in float32 through the libm ``expf`` included -- which is
#: why they are frozen bit patterns and not recomputed here:
#: :func:`coefficients` above evaluates the same formulas in float64, and
#: e.g. its ``pvtr`` (2500.0682) is not WRF's (0x451C4105 = 2500.0637).
#: Dumped by tools/wsm6_wrf461_column_oracle/dump_wsm6_init.F90 (gfortran
#: 13.3.0 -O0, glibc 2.39) into ``wsm6-init-f32.txt``, and PROVEN by the
#: column oracle: a wrong word cannot reproduce WRF's outputs bit for bit
#: (tests/test_wsm6_wrf461_column_exact.py).
WRF_INIT_F32 = {
    0: (
        ("pi", 0x40490FDB),  # 3.141592741E+00
        ("xlv1", 0x4512799A),  # 2.343600098E+03
        ("qc0", 0x3A03C498),  # 5.026548170E-04
        ("qck1", 0x40D8C3BE),  # 6.773894310E+00
        ("pidnc", 0x4402E653),  # 5.235988159E+02
        ("pidn0r", 0x50BB40E7),  # 2.513274266E+10
        ("pidn0s", 0x4E15CD85),  # 6.283185280E+08
        ("pidn0g", 0x4FBB40E7),  # 6.283185664E+09
        ("pvtr", 0x451C4105),  # 2.500063721E+03
        ("pvts", 0x41A069C1),  # 2.005163765E+01
        ("pvtg", 0x4474FCE3),  # 9.799513550E+02
        ("pacrr", 0x50B8DFD0),  # 2.481340416E+10
        ("pacrc", 0x4C537EB0),  # 5.544211200E+07
        ("pacrg", 0x4F90EE38),  # 4.863062016E+09
        ("precr1", 0x4C159029),  # 3.920707600E+07
        ("precr2", 0x4E44E61A),  # 8.258535680E+08
        ("precs1", 0x4A9EB100),  # 5.200000000E+06
        ("precs2", 0x4B8E882C),  # 1.868194400E+07
        ("precg1", 0x4B959029),  # 1.960353800E+07
        ("precg2", 0x4D768C0C),  # 2.585233280E+08
        ("g6pbr", 0x43F7BADA),  # 4.954597778E+02
        ("roqimax", 0x38AA64C8),  # 8.125003660E-05
        ("n0g", 0x4A742400),  # 4.000000000E+06
        ("bvtg", 0x3F4CCCCD),  # 8.000000119E-01
        ("rslopermax", 0x3751B717),  # 1.249999968E-05
        ("rslopesmax", 0x3727C5AC),  # 9.999999747E-06
        ("rslopegmax", 0x378BCF65),  # 1.666666685E-05
        ("rsloperbmax", 0x38FAB3B8),  # 1.195440418E-04
        ("rslopesbmax", 0x3C1205C6),  # 8.912509307E-03
        ("rslopegbmax", 0x391DCA2F),  # 1.504800603E-04
        ("rsloper2max", 0x2F2BCC76),  # 1.562499852E-10
        ("rslopes2max", 0x2EDBE6FE),  # 9.999999440E-11
        ("rslopeg2max", 0x2F98B5BF),  # 2.777777730E-10
        ("rsloper3max", 0x270CBCCB),  # 1.953124780E-15
        ("rslopes3max", 0x26901D7C),  # 9.999998977E-16
        ("rslopeg3max", 0x27A6CCCC),  # 4.629629674E-15
    ),
    1: (
        ("pi", 0x40490FDB),  # 3.141592741E+00
        ("xlv1", 0x4512799A),  # 2.343600098E+03
        ("qc0", 0x3A03C498),  # 5.026548170E-04
        ("qck1", 0x40D8C3BE),  # 6.773894310E+00
        ("pidnc", 0x4402E653),  # 5.235988159E+02
        ("pidn0r", 0x50BB40E7),  # 2.513274266E+10
        ("pidn0s", 0x4E15CD85),  # 6.283185280E+08
        ("pidn0g", 0x4CA7C777),  # 8.796460000E+07
        ("pvtr", 0x451C4105),  # 2.500063721E+03
        ("pvts", 0x41A069C1),  # 2.005163765E+01
        ("pvtg", 0x44539495),  # 8.463215942E+02
        ("pacrr", 0x50B8DFD0),  # 2.481340416E+10
        ("pacrc", 0x4C537EB0),  # 5.544211200E+07
        ("pacrg", 0x4C2036D0),  # 4.199916800E+07
        ("precr1", 0x4C159029),  # 3.920707600E+07
        ("precr2", 0x4E44E61A),  # 8.258535680E+08
        ("precs1", 0x4A9EB100),  # 5.200000000E+06
        ("precs2", 0x4B8E882C),  # 1.868194400E+07
        ("precg1", 0x483F70D8),  # 1.960353750E+05
        ("precg2", 0x4A12A338),  # 2.402510000E+06
        ("g6pbr", 0x43F7BADA),  # 4.954597778E+02
        ("roqimax", 0x38AA64C8),  # 8.125003660E-05
        ("n0g", 0x471C4000),  # 4.000000000E+04
        ("bvtg", 0x3F4CCCCD),  # 8.000000119E-01
        ("rslopermax", 0x3751B717),  # 1.249999968E-05
        ("rslopesmax", 0x3727C5AC),  # 9.999999747E-06
        ("rslopegmax", 0x3851B717),  # 4.999999874E-05
        ("rsloperbmax", 0x38FAB3B8),  # 1.195440418E-04
        ("rslopesbmax", 0x3C1205C6),  # 8.912509307E-03
        ("rslopegbmax", 0x39BDFF22),  # 3.623897792E-04
        ("rsloper2max", 0x2F2BCC76),  # 1.562499852E-10
        ("rslopes2max", 0x2EDBE6FE),  # 9.999999440E-11
        ("rslopeg2max", 0x312BCC76),  # 2.499999763E-09
        ("rsloper3max", 0x270CBCCB),  # 1.953124780E-15
        ("rslopes3max", 0x26901D7C),  # 9.999998977E-16
        ("rslopeg3max", 0x2A0CBCCB),  # 1.249999859E-13
    ),
}

#: Names in kernel order, for the device upload and the table tests.
WRF_INIT_NAMES = tuple(name for name, _ in WRF_INIT_F32[0])


def wrf_init_words(hail_opt: int):
    """``mp_wsm6_init``'s words for ``hail_opt`` as a float32 numpy array."""
    import numpy as np

    hail = int(hail_opt)
    if hail not in WRF_INIT_F32:
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")
    return np.asarray([bits for _, bits in WRF_INIT_F32[hail]],
                      dtype=np.uint32).view(np.float32)



#: mp_radar's ``radar_init`` words as ``refl10cm_wsm6`` reads them after
#: ``mp_wsm6_init`` (phys/physics_mmm/mp_radar.F90:72-189), float32 part,
#: by ``hail_opt``, in ``kernels/wsm6.cu``'s ``Wsm6RadarF32`` order (``pi``
#: and ``n0g`` are mp_wsm6's).  WRF's own float32 arithmetic: e.g. ``xcrg4``
#: = WGAMMA(7.) is 0x44340001 = 720.00006, not 720.  Dumped with
#: :data:`WRF_INIT_F32` by tools/wsm6_wrf461_column_oracle/dump_wsm6_init.F90.
WRF_RADAR_F32 = {
    0: (
        ("pi", 0x40490FDB),  # 3.141592741E+00
        ("n0g", 0x4A742400),  # 4.000000000E+06
        ("xam_r", 0x4402E653),  # 5.235988159E+02
        ("xam_s", 0x42517084),  # 5.235987854E+01
        ("xam_g", 0x4382E653),  # 2.617994080E+02
        ("xcre1", 0x40800000),  # 4.000000000E+00
        ("xcre4", 0x40E00000),  # 7.000000000E+00
        ("xcrg3", 0x40C00000),  # 6.000000000E+00
        ("xcrg4", 0x44340001),  # 7.200000610E+02
        ("xcse1", 0x40800000),  # 4.000000000E+00
        ("xcse4", 0x40E00000),  # 7.000000000E+00
        ("xcsg3", 0x40C00000),  # 6.000000000E+00
        ("xcsg4", 0x44340001),  # 7.200000610E+02
        ("xcge1", 0x40800000),  # 4.000000000E+00
        ("xcge4", 0x40E00000),  # 7.000000000E+00
        ("xcgg3", 0x40C00000),  # 6.000000000E+00
        ("xcgg4", 0x44340001),  # 7.200000610E+02
        ("xbm_s", 0x40400000),  # 3.000000000E+00
        ("xbm_g", 0x40400000),  # 3.000000000E+00
        ("xmu_s", 0x00000000),  # 0.000000000E+00
        ("xmu_g", 0x00000000),  # 0.000000000E+00
        ("xocms", 0x3E88DBAA),  # 2.673009038E-01
        ("xobms", 0x3EAAAAAB),  # 3.333333433E-01
        ("xocmg", 0x3E2011F6),  # 1.563185155E-01
        ("xobmg", 0x3EAAAAAB),  # 3.333333433E-01
    ),
    1: (
        ("pi", 0x40490FDB),  # 3.141592741E+00
        ("n0g", 0x471C4000),  # 4.000000000E+04
        ("xam_r", 0x4402E653),  # 5.235988159E+02
        ("xam_s", 0x42517084),  # 5.235987854E+01
        ("xam_g", 0x43B74274),  # 3.665191650E+02
        ("xcre1", 0x40800000),  # 4.000000000E+00
        ("xcre4", 0x40E00000),  # 7.000000000E+00
        ("xcrg3", 0x40C00000),  # 6.000000000E+00
        ("xcrg4", 0x44340001),  # 7.200000610E+02
        ("xcse1", 0x40800000),  # 4.000000000E+00
        ("xcse4", 0x40E00000),  # 7.000000000E+00
        ("xcsg3", 0x40C00000),  # 6.000000000E+00
        ("xcsg4", 0x44340001),  # 7.200000610E+02
        ("xcge1", 0x40800000),  # 4.000000000E+00
        ("xcge4", 0x40E00000),  # 7.000000000E+00
        ("xcgg3", 0x40C00000),  # 6.000000000E+00
        ("xcgg4", 0x44340001),  # 7.200000610E+02
        ("xbm_s", 0x40400000),  # 3.000000000E+00
        ("xbm_g", 0x40400000),  # 3.000000000E+00
        ("xmu_s", 0x00000000),  # 0.000000000E+00
        ("xmu_g", 0x00000000),  # 0.000000000E+00
        ("xocms", 0x3E88DBAA),  # 2.673009038E-01
        ("xobms", 0x3EAAAAAB),  # 3.333333433E-01
        ("xocmg", 0x3E0F1658),  # 1.397336721E-01
        ("xobmg", 0x3EAAAAAB),  # 3.333333433E-01
    ),
}

#: The float64 part (identical for both ``hail_opt``): k_w, pi5, lamda4,
#: the refractive indices m_w_0 and m_i_0 (real, imaginary), the 50 snow
#: and graupel size bins and widths, and the 51 Simpson weights, in
#: ``Wsm6RadarF64`` order.  Bit patterns as IEEE binary64.
WRF_RADAR_F64 = (
    ("k_w", 0x3FEDE4B5A805B813),  # 9.34168651741631328E-01
    ("pi5", 0x4073204BA0000000),  # 3.06018463134765625E+02
    ("lamda4", 0x3F1A36E305532621),  # 1.00000005960464603E-04
    ("m_w_0_re", 0x402213D1A0000000),  # 9.03870868682861328E+00
    ("m_w_0_im", 0xBFF634D6E0000000),  # -1.38790023326873779E+00
    ("m_i_0_re", 0x3FFC91DAE74D3B24),  # 1.78560915329189474E+00
    ("m_i_0_im", 0xBF21ACB622AE704E),  # -1.34846909841781386E-04
    ("xxds(1)", 0x3F1BA409DB297486),  # 1.05441190308268746E-04
    ("xxds(2)", 0x3F1EBB079752770D),  # 1.17227876972294569E-04
    ("xxds(3)", 0x3F2115387D5665E6),  # 1.30332132056306521E-04
    ("xxds(4)", 0x3F22FE142AD6D844),  # 1.44901239236440893E-04
    ("xxds(5)", 0x3F251D95692C1748),  # 1.61098946215238467E-04
    ("xxds(6)", 0x3F2779D809403699),  # 1.79107305143967594E-04
    ("xxds(7)", 0x3F2A19A6AB6F8CF9),  # 1.99128718775566459E-04
    ("xxds(8)", 0x3F2D048E4A0CE200),  # 2.21388215345688644E-04
    ("xxds(9)", 0x3F302179F98CDD44),  # 2.46135977750101343E-04
    ("xxds(10)", 0x3F31EF1677DAB285),  # 2.73650155444817294E-04
    ("xxds(11)", 0x3F33F04CCF611FB4),  # 3.04239990672967907E-04
    ("xxds(12)", 0x3F362AE1A802BFA6),  # 3.38249294155263508E-04
    ("xxds(13)", 0x3F38A53EBAD47393),  # 3.76060309308639737E-04
    ("xxds(14)", 0x3F3B668545D00F11),  # 4.18098008424503764E-04
    ("xxds(15)", 0x3F3E76A28F903D41),  # 4.64834869092951213E-04
    ("xxds(16)", 0x3F40EF335B0E2AF3),  # 5.16796184556993199E-04
    ("xxds(17)", 0x3F42D3CF0531FEA0),  # 5.74565967681867767E-04
    ("xxds(18)", 0x3F44EE96A0B0304C),  # 6.38793514896382388E-04
    ("xxds(19)", 0x3F47459865FC6260),  # 7.10200703880902590E-04
    ("xxds(20)", 0x3F49DF8FD7ECB26E),  # 7.89590107023464263E-04
    ("xxds(21)", 0x3F4CC3F922C031BE),  # 8.77854011834204367E-04
    ("xxds(22)", 0x3F4FFB26A57CE929),  # 9.75984449701959860E-04
    ("xxds(23)", 0x3F51C72C71CE74E4),  # 1.08508434571002402E-03
    ("xxds(24)", 0x3F53C3EC91F79B6B),  # 1.20637991482804951E-03
    ("xxds(25)", 0x3F55F98B84A26D3D),  # 1.34123444380558556E-03
    ("xxds(26)", 0x3F586E64BD092F83),  # 1.49116361366716309E-03
    ("xxds(27)", 0x3F5B29899AD9698A),  # 1.65785253502423674E-03
    ("xxds(28)", 0x3F5E32D5C0476AC2),  # 1.84317468767029912E-03
    ("xxds(29)", 0x3F60C982D70B9635),  # 2.04921297732843033E-03
    ("xxds(30)", 0x3F62A9E7F3542E0C),  # 2.27828314621603654E-03
    ("xxds(31)", 0x3F64C000702C7F67),  # 2.53295980054695251E-03
    ("xxds(32)", 0x3F6711CD0BD3ECB9),  # 2.81610534750384141E-03
    ("xxds(33)", 0x3F69A5FA4D3E7318),  # 3.13090216691448361E-03
    ("xxds(34)", 0x3F6C83F3B7FE38FF),  # 3.48088837922155793E-03
    ("xxds(35)", 0x3F6FB3F925B2E709),  # 3.86999761175566394E-03
    ("xxds(36)", 0x3F719F9B413630EB),  # 4.30260321025975178E-03
    ("xxds(37)", 0x3F7397EF18262E6F),  # 4.78356739257500187E-03
    ("xxds(38)", 0x3F75C8A32F2A4F82),  # 5.31829589694499290E-03
    ("xxds(39)", 0x3F783804D36655C9),  # 5.91279873915113389E-03
    ("xxds(40)", 0x3F7AED15A98C9F98),  # 6.57375776135173812E-03
    ("xxds(41)", 0x3F7DEF9FD6AEEC8D),  # 7.30860173183168777E-03
    ("xxds(42)", 0x3F80A42634FAA2EF),  # 8.12558983973718334E-03
    ("xxds(43)", 0x3F82805E23DC0717),  # 9.03390452322442965E-03
    ("xxds(44)", 0x3F8491D1EED7D837),  # 1.00437546743529071E-02
    ("xxds(45)", 0x3F86DE74F7F81201),  # 1.11664903806820721E-02
    ("xxds(46)", 0x3F896CE4EBA7CA80),  # 1.24147304931956359E-02
    ("xxds(47)", 0x3F8C447CC9DFCB0F),  # 1.38025044543375313E-02
    ("xxds(48)", 0x3F8F6D6A1011EBCA),  # 1.53454099802185114E-02
    ("xxds(49)", 0x3F917862205C08B5),  # 1.70607883692432578E-02
    ("xxds(50)", 0x3F936C53861D22D4),  # 1.89679194075179219E-02
    ("xdts(1)", 0x3EE77161F03B8D68),  # 1.11784461362454533E-05
    ("xdts(2)", 0x3EEA103E6F958E30),  # 1.24280227164547585E-05
    ("xdts(3)", 0x3EECFA18DA335F10),  # 1.38172825416139997E-05
    ("xdts(4)", 0x3EF01BA99BC1EBD0),  # 1.53618400280211326E-05
    ("xdts(5)", 0x3EF1E89FB9BEE440),  # 1.70790550410896639E-05
    ("xdts(6)", 0x3EF3E91D17CBBA40),  # 1.89882280094372333E-05
    ("xdts(7)", 0x3EF622E44991CBA8),  # 2.11108168496999609E-05
    ("xdts(8)", 0x3EF89C5CB86E2E28),  # 2.34706781401650794E-05
    ("xdts(9)", 0x3EFB5CA5107DD508),  # 2.60943352538749792E-05
    ("xdts(10)", 0x3EFE6BA7BCF49940),  # 2.90112764648406147E-05
    ("xdts(11)", 0x3F00E918D7584E60),  # 3.22542863779001331E-05
    ("xdts(12)", 0x3F02CD05D7491C10),  # 3.58598144072839716E-05
    ("xdts(13)", 0x3F04E70B4239FE30),  # 3.98683844453597480E-05
    ("xdts(14)", 0x3F073D3521E19EF8),  # 4.43250503259755443E-05
    ("xdts(15)", 0x3F09D63C8BE3D3E0),  # 4.92799022015278417E-05
    ("xdts(16)", 0x3F0CB99AF9DCBC90),  # 5.47886295251205678E-05
    ("xdts(17)", 0x3F0FEF9FCAFB9C00),  # 6.09131469653731439E-05
    ("xdts(18)", 0x3F11C0C416875130),  # 6.77222902887894117E-05
    ("xdts(19)", 0x3F13BCCCD8E5CA20),  # 7.52925900309527225E-05
    ("xdts(20)", 0x3F15F19FEE6907A0),  # 8.37091316521474706E-05
    ("xdts(21)", 0x3F1865967FB87488),  # 9.30665118450015429E-05
    ("xdts(22)", 0x3F1B1FBF605CDEE8),  # 1.03469901742477368E-04
    ("xdts(23)", 0x3F1E27F35D7F9700),  # 1.15036228975986393E-04
    ("xdts(24)", 0x3F20C375E8E694D8),  # 1.27895491869234627E-04
    ("xdts(25)", 0x3F22A32DDFB908E0),  # 1.42192220538523394E-04
    ("xdts(26)", 0x3F24B885DC355DB0),  # 1.58087101321353115E-04
    ("xdts(27)", 0x3F27097C72B987D8),  # 1.75758782790912419E-04
    ("xdts(28)", 0x3F299CBBC26DC8E0),  # 1.95405883654914923E-04
    ("xdts(29)", 0x3F2C79ACA2426CC0),  # 2.17249225106330365E-04
    ("xdts(30)", 0x3F2FA88BF2AD43B0),  # 2.41534312716255081E-04
    ("xdts(31)", 0x3F31994128BF2170),  # 2.68534095763792689E-04
    ("xdts(32)", 0x3F3390DF39FF1AB0),  # 2.98552035016201416E-04
    ("xdts(33)", 0x3F35C0C93994F538),  # 3.31925513439152147E-04
    ("xdts(34)", 0x3F382F4A2F2B1680),  # 3.69029628171403667E-04
    ("xdts(35)", 0x3F3AE36138FA9020),  # 4.10281406383329952E-04
    ("xdts(36)", 0x3F3DE4D5AD57CE60),  # 4.56144492402920862E-04
    ("xdts(37)", 0x3F409E26BE2ABE10),  # 5.07134358789631048E-04
    ("xdts(38)", 0x3F4279B308F1B160),  # 5.63824099925318983E-04
    ("xdts(39)", 0x3F448A680000F378),  # 6.26850873238645433E-04
    ("xdts(40)", 0x3F46D636E051CC08),  # 6.96923060458210820E-04
    ("xdts(41)", 0x3F4963BAF3DE1CE0),  # 7.74828229382597961E-04
    ("xdts(42)", 0x3F4C3A4C93F09170),  # 8.61441985652562289E-04
    ("xdts(43)", 0x3F4F62164B6F3720),  # 9.57737814013735062E-04
    ("xdts(44)", 0x3F5172162AF7B9A8),  # 1.06479801967971667E-03
    ("xdts(45)", 0x3F5365535F972F98),  # 1.18382589276941862E-03
    ("xdts(46)", 0x3F55905F370BA5E0),  # 1.31615923253957517E-03
    ("xdts(47)", 0x3F57F976B77487E8),  # 1.46328538341621885E-03
    ("xdts(48)", 0x3F5AA7896CB11678),  # 1.62685795182096297E-03
    ("xdts(49)", 0x3F5DA24D5D1F3A40),  # 1.80871539167849760E-03
    ("xdts(50)", 0x3F60792A9CB73498),  # 2.01090166749524554E-03
    ("xxdg(1)", 0x3F1BE52C6D4D4E0B),  # 1.06411778074737433E-04
    ("xxdg(2)", 0x3F1F9648B0C9DD5D),  # 1.20495020562096667E-04
    ("xxdg(3)", 0x3F21E23CFF932616),  # 1.36442133032141962E-04
    ("xxdg(4)", 0x3F2440271C8B315B),  # 1.54499792435545648E-04
    ("xxdg(5)", 0x3F26EE420EDFFC82),  # 1.74947322591354675E-04
    ("xxdg(6)", 0x3F29F72AC2B16E9F),  # 1.98101014890696294E-04
    ("xxdg(7)", 0x3F2D66E5B77868E7),  # 2.24319020831149229E-04
    ("xxdg(8)", 0x3F30A5874B5B6693),  # 2.54006892061655807E-04
    ("xxdg(9)", 0x3F32D9870B7F02AB),  # 2.87623853633826420E-04
    ("xxdg(10)", 0x3F35582B8057A06B),  # 3.25689907497045655E-04
    ("xxdg(11)", 0x3F382B55A30B2B62),  # 3.68793876117370673E-04
    ("xxdg(12)", 0x3F3B5E35203D655F),  # 4.17602510642453172E-04
    ("xxdg(13)", 0x3F3EFD74A3FC3BD5),  # 4.72870804501588249E-04
    ("xxdg(14)", 0x3F418BB6013C913B),  # 5.35453671976194603E-04
    ("xxdg(15)", 0x3F43DE2C8217D90B),  # 6.06319172389986421E-04
    ("xxdg(16)", 0x3F467F4FDB31B216),  # 6.86563484476434725E-04
    ("xxdg(17)", 0x3F4979899F513F38),  # 7.77427862553615144E-04
    ("xxdg(18)", 0x3F4CD8A428D880E8),  # 8.80317836792013537E-04
    ("xxdg(19)", 0x3F5054FCA51A4B10),  # 9.96824954573486549E-04
    ("xxdg(20)", 0x3F527E539615BDB2),  # 1.12875139924626918E-03
    ("xxdg(21)", 0x3F54F0E615B10661),  # 1.27813786708976689E-03
    ("xxdg(22)", 0x3F57B6655115131D),  # 1.44729513370229286E-03
    ("xxdg(23)", 0x3F5AD9CAD5844970),  # 1.63883979809450672E-03
    ("xxdg(24)", 0x3F5E678405F4A72A),  # 1.85573475739393224E-03
    ("xxdg(25)", 0x3F6136D1A88CC628),  # 2.10133503824107629E-03
    ("xxdg(26)", 0x3F637E0BF55E34BB),  # 2.37943969381734571E-03
    ("xxdg(27)", 0x3F6612767286FB31),  # 2.69435056927083043E-03
    ("xxdg(28)", 0x3F68FE4851E6E59C),  # 3.05093884454939006E-03
    ("xxdg(29)", 0x3F6C4D12E21BF0E1),  # 3.45472038395487667E-03
    ("xxdg(30)", 0x3F7005F7AE866566),  # 3.91194105795853418E-03
    ("xxdg(31)", 0x3F7224D963359EBD),  # 4.42967335707294702E-03
    ("xxdg(32)", 0x3F748B9453CA2EFE),  # 5.01592579224641004E-03
    ("xxdg(33)", 0x3F7743AAC8B1AEFC),  # 5.67976677403314242E-03
    ("xxdg(34)", 0x3F7A57E135A9F49C),  # 6.43146488675685676E-03
    ("xxdg(35)", 0x3F7DD468DD15F31B),  # 7.28264772749012256E-03
    ("xxdg(36)", 0x3F80E3880BF712D4),  # 8.24648176684076445E-03
    ("xxdg(37)", 0x3F831FBC80BE7278),  # 9.33787601371104115E-03
    ("xxdg(38)", 0x3F85A7ABAFB253FE),  # 1.05737126344056485E-02
    ("xxdg(39)", 0x3F88855B5D8A8DFB),  # 1.19731080934064506E-02
    ("xxdg(40)", 0x3F8BC424E122D453),  # 1.35577088552542354E-02
    ("xxdg(41)", 0x3F8F70E214646E8E),  # 1.53520262215842938E-02
    ("xxdg(42)", 0x3F91CD101BE97692),  # 1.73838154827224492E-02
    ("xxdg(43)", 0x3F94282CC91FF6B1),  # 1.96845052487250628E-02
    ("xxdg(44)", 0x3F96D31B586ABF0A),  # 2.22896836009445556E-02
    ("xxdg(45)", 0x3F99D86C24ECE8BB),  # 2.52396485841266220E-02
    ("xxdg(46)", 0x3F9D4415736099DA),  # 2.85800315542930963E-02
    ("xxdg(47)", 0x3FA091D16837085E),  # 3.23625030246297157E-02
    ("xxdg(48)", 0x3FA2C33559F20BA0),  # 3.66455719277134495E-02
    ("xxdg(49)", 0x3FA53EE59E9C3495),  # 4.14954906574191221E-02
    ("xxdg(50)", 0x3FA80EB77CE9713F),  # 4.69872798900916530E-02
    ("xdtg(1)", 0x3EEBC14E866A2140),  # 1.32346651302717036E-05
    ("xdtg(2)", 0x3EEF6DAB983EB268),  # 1.49862287413756622E-05
    ("xdtg(3)", 0x3EF1CB3E70A34428),  # 1.69696059309534764E-05
    ("xdtg(4)", 0x3EF4261D7CA34128),  # 1.92154764498418988E-05
    ("xdtg(5)", 0x3EF6D0C642A7D4A8),  # 2.17585804111643385E-05
    ("xdtg(6)", 0x3EF9D5C8097B0790),  # 2.46382556656831266E-05
    ("xdtg(7)", 0x3EFD4117DCEB0DB0),  # 2.78990462969760304E-05
    ("xdtg(8)", 0x3F00901FF3A518E8),  # 3.15913916489206814E-05
    ("xdtg(9)", 0x3F02C14A87960F38),  # 3.57724065436479713E-05
    ("xdtg(10)", 0x3F053CB9D6D8B830),  # 4.05067647587384816E-05
    ("xdtg(11)", 0x3F080C4226E5E400),  # 4.58676994296650098E-05
    ("xdtg(12)", 0x3F0B3B04C25B0230),  # 5.19381358521402636E-05
    ("xdtg(13)", 0x3F0ED59C0BF1F4E0),  # 5.88119742070765308E-05
    ("xdtg(14)", 0x3F117526B35E1998),  # 6.65955420498853595E-05
    ("xdtg(15)", 0x3F13C4A0DCA10FD8),  # 7.54092390318755875E-05
    ("xdtg(16)", 0x3F166262B5A79FB0),  # 8.53893992950316687E-05
    ("xdtg(17)", 0x3F1958C86DE8EDE0),  # 9.66904003484795986E-05
    ("xdtg(18)", 0x3F1CB38D36DF1978),  # 1.09487051047718153E-04
    ("xdtg(19)", 0x3F203FFCDC37B48C),  # 1.23977295614894809E-04
    ("xdtg(20)", 0x3F22668C559F03B8),  # 1.40385275527092060E-04
    ("xdtg(21)", 0x3F24D5F9349D7A20),  # 1.58964796635311691E-04
    ("xdtg(22)", 0x3F2797E8302391E8),  # 1.80003255145013601E-04
    ("xdtg(23)", 0x3F2AB744B9043188),  # 2.03826083187041507E-04
    ("xdtg(24)", 0x3F2E406C37ADDA40),  # 2.30801782744998255E-04
    ("xdtg(25)", 0x3F3120AF81772098),  # 2.61347625805993713E-04
    ("xdtg(26)", 0x3F3364FBE8D28898),  # 2.95936108907332487E-04
    ("xdtg(27)", 0x3F35F6154179D6A8),  # 3.35102261920775749E-04
    ("xdtg(28)", 0x3F38DE259AC34AD0),  # 3.79451924129953384E-04
    ("xdtg(29)", 0x3F3C28AF63BE1408),  # 4.29671115618924116E-04
    ("xdtg(30)", 0x3F3FE2BAFED72E80),  # 4.86536648932590909E-04
    ("xdtg(31)", 0x3F420D852ED55AE8),  # 5.50928145154863831E-04
    ("xdtg(32)", 0x3F447129B8AFB4A0),  # 6.23841640274531289E-04
    ("xdtg(33)", 0x3F4725C12B7B19A8),  # 7.06404992308053321E-04
    ("xdtg(34)", 0x3F4A360222CC0320),  # 7.99895327503550441E-04
    ("xdtg(35)", 0x3F4DAE0E33CD3220),  # 9.05758795491321683E-04
    ("xdtg(36)", 0x3F50CDD0FB8E0B38),  # 1.02563293896258505E-03
    ("xdtg(37)", 0x3F530725B71CEB38),  # 1.16137202390005460E-03
    ("xdtg(38)", 0x3F558BD3CDFC5510),  # 1.31507572217987381E-03
    ("xdtg(39)", 0x3F5865D421E79EA8),  # 1.48912159021989156E-03
    ("xdtg(40)", 0x3F5BA0717221DE28),  # 1.68620184606805591E-03
    ("xdtg(41)", 0x3F5F48751296F8F0),  # 1.90936501381560805E-03
    ("xdtg(42)", 0x3F61B62CC6FCF5B8),  # 2.16206307950869356E-03
    ("xdtg(43)", 0x3F640E41FDB0A0C0),  # 2.44820488798688785E-03
    ("xdtg(44)", 0x3F66B5C27535FA78),  # 2.77221660661491551E-03
    ("xdtg(45)", 0x3F69B730F38289C8),  # 3.13911019118615423E-03
    ("xdtg(46)", 0x3F6D1E745BF18448),  # 3.55456091305986557E-03
    ("xdtg(47)", 0x3F707C8368599540),  # 4.02499514675486170E-03
    ("xdtg(48)", 0x3F72AB15888706A8),  # 4.55768977593754704E-03
    ("xdtg(49)", 0x3F75239473E3D5A8),  # 5.16088475545953967E-03
    ("xdtg(50)", 0x3F77EFC8CC80FDB0),  # 5.84391057060378027E-03
    ("simpson(1)", 0x3FD5555555555555),  # 3.33333333333333315E-01
    ("simpson(2)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(3)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(4)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(5)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(6)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(7)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(8)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(9)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(10)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(11)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(12)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(13)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(14)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(15)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(16)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(17)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(18)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(19)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(20)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(21)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(22)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(23)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(24)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(25)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(26)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(27)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(28)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(29)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(30)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(31)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(32)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(33)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(34)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(35)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(36)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(37)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(38)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(39)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(40)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(41)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(42)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(43)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(44)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(45)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(46)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(47)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(48)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(49)", 0x3FE5555555555555),  # 6.66666666666666630E-01
    ("simpson(50)", 0x3FF5555555555555),  # 1.33333333333333326E+00
    ("simpson(51)", 0x3FD5555555555555),  # 3.33333333333333315E-01
)


WRF_SCALAR_RADAR_F64_OVERRIDES = {
    "k_w": 0x3FEDE4B5A79D9DE5,
    "m_w_0_im": 0xBFF634D6C0000000,
    "m_i_0_im": 0xBF21ACB622AE704D,
}


def wrf_radar_words(hail_opt: int, *, initialization="column_oracle"):
    """``refl10cm_wsm6``'s radar words: (float32 array, float64 array)."""
    import numpy as np

    hail = int(hail_opt)
    if hail not in WRF_RADAR_F32:
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")
    f32 = np.asarray([b for _, b in WRF_RADAR_F32[hail]], np.uint32).view(np.float32)
    if initialization not in ("column_oracle", "scalar_build"):
        raise ValueError("WSM6 radar initialization must be column_oracle or scalar_build")
    # The existing column oracle was compiled at -O0. Stock scalar WRF
    # compiles radar_init at -O2 and produces three different words through
    # its complex initialization. Every other binary64 word is identical.
    # Keep the column entry point unchanged; the forecast wrapper selects
    # the scalar build it is replaying. Both hail options share these words.
    overrides = WRF_SCALAR_RADAR_F64_OVERRIDES if initialization == "scalar_build" else {}
    f64 = np.asarray([overrides.get(n,b) for n,b in WRF_RADAR_F64], np.uint64).view(np.float64)
    return f32, f64

#: ``kernels/wsm6.cu``'s ``WSM6_KMAX`` ladder.  One CUDA thread owns a whole
#: column, so the bound sizes the per-thread work arrays and the driver
#: reserves a local-memory backing store for the device's entire resident
#: -thread capacity at first launch -- the law at the head of
#: ``gpuwm/core/preflight.py``.  ``gpuwm/core/wsm6.py`` therefore compiles the
#: SHALLOWEST tier that holds the configuration's ``nz``, and compiles a TIER
#: rather than ``nz`` itself so a nested tree with several level counts loads
#: at most two WSM6 modules.
#:
#: 64 is the ``#ifndef`` default baked into the source, which is why it is
#: also the value the driver-regeneration gate reads back for the
#: unspecialized module and therefore the value recorded in
#: ``KERNEL_MAX_LOCAL_SIZE_BYTES``.  The 80 tier prices ABOVE that row;
#: ``preflight.WSM6_TIER_FRAME`` is what carries it.
#:
#: This ladder lives in the constants module, not in ``gpuwm/core/wsm6.py``,
#: for one reason: the adapter imports CuPy at module scope and the memory
#: estimator must price a configuration on a host with no device at all.
WSM6_SHALLOW_KMAX = 64
WSM6_DEEP_KMAX = 80
WSM6_KERNEL_LEVEL_TIERS = (WSM6_SHALLOW_KMAX, WSM6_DEEP_KMAX)
#: ``(minimum, maximum)`` levels WSM6 admits: 2 (the scheme needs a layer
#: above and below to sediment between) up to the deepest compiled tier.
WSM6_VERTICAL_LEVEL_BOUNDS = (2, WSM6_DEEP_KMAX)


def wsm6_level_tier(nz: int) -> int:
    """Return the ``WSM6_KMAX`` tier ``wsm6.cu`` compiles for ``nz``."""
    nz = int(nz)
    minimum, maximum = WSM6_VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WSM6 requires {minimum} <= nz <= {maximum}, got {nz}")
    return next(tier for tier in WSM6_KERNEL_LEVEL_TIERS if nz <= tier)


# ---------------------------------------------------------------------------
# WSM5 (mp_physics = 4), phys/module_mp_wsm5.F.  The WSM family shares one
# constants module: WSM5's process constants are WSM6's rain and snow
# constants (same intercepts, fall laws and densities; no graupel), and
# kernels/wsm5.cu compiles at WSM6's level ladder.
# ---------------------------------------------------------------------------

#: The float32 words ``wsm5init`` leaves in module_mp_wsm5 (WRF v4.6.1, called
#: as module_physics_init.F:4485 calls it), in kernels/wsm5.cu's ``Wsm5Coef``
#: order, copied from the scalar WRF build's own object by
#: tools/wsm5_wrf461_column_oracle/dump_wsm5_init.F90, not recomputed: the
#: rgmma Weierstrass product rounds in float32 (pvtr is 2500.0637, where a
#: float64 Gamma gives 2500.0682), and a wrong word cannot reproduce WRF's
#: outputs bit for bit, which the column oracle checks.
WSM5_INIT_F32: tuple[tuple[str, int], ...] = (
    ("pi", 0x40490FDB),
    ("xlv1", 0x4512799A),
    ("qc0", 0x3A03C498),
    ("qck1", 0x40D8C3BE),
    ("pidnc", 0x4402E653),
    ("pidn0r", 0x50BB40E7),
    ("pidn0s", 0x4E15CD85),
    ("pvtr", 0x451C4105),
    ("pvts", 0x41A069C1),
    ("pacrr", 0x50B8DFD0),
    ("pacrc", 0x4C537EB0),
    ("precr1", 0x4C159029),
    ("precr2", 0x4E44E61A),
    ("precs1", 0x4A9EB100),
    ("precs2", 0x4B8E882C),
    ("roqimax", 0x38AA64C8),
    ("rslopermax", 0x3751B717),
    ("rslopesmax", 0x3727C5AC),
    ("rsloperbmax", 0x38FAB3B8),
    ("rslopesbmax", 0x3C1205C6),
    ("rsloper2max", 0x2F2BCC76),
    ("rslopes2max", 0x2EDBE6FE),
    ("rsloper3max", 0x270CBCCB),
    ("rslopes3max", 0x26901D7C),
)

#: module_mp_radar's radar_init words ``refl10cm_wsm5`` reads, after
#: wsm5init set xam_r/xam_s and the m(D) exponents: float32 in kernels/
#: wsm5.cu's ``Wsm5RadarF32`` order (xcrg(4) = WGAMMA(7.) is 720.00006),
#: then float64 in its ``Wsm5RadarF64`` order (k_w, pi5 -- a float32
#: product held in binary64 --, lamda4, the two refractive indices, the 50
#: snow bins and their widths, the 51 Simpson weights).
WSM5_RADAR_F32: tuple[tuple[str, int], ...] = (
    ("pi", 0x40490FDB),
    ("xam_r", 0x4402E653),
    ("xam_s", 0x42517084),
    ("xcre1", 0x40800000),
    ("xcre4", 0x40E00000),
    ("xcrg3", 0x40C00000),
    ("xcrg4", 0x44340001),
    ("xcse1", 0x40800000),
    ("xcse4", 0x40E00000),
    ("xcsg3", 0x40C00000),
    ("xcsg4", 0x44340001),
    ("xbm_s", 0x40400000),
    ("xmu_s", 0x00000000),
    ("xocms", 0x3E88DBAA),
    ("xobms", 0x3EAAAAAB),
)
WSM5_RADAR_F64: tuple[tuple[str, int], ...] = (
    ("k_w", 0x3FEDE4B5A79D9DE5),
    ("pi5", 0x4073204BA0000000),
    ("lamda4", 0x3F1A36E305532621),
    ("m_w_0_re", 0x402213D1A0000000),
    ("m_w_0_im", 0xBFF634D6C0000000),
    ("m_i_0_re", 0x3FFC91DAE74D3B24),
    ("m_i_0_im", 0xBF21ACB622AE704D),
    ("xxds1", 0x3F1BA409DB297486),
    ("xxds2", 0x3F1EBB079752770D),
    ("xxds3", 0x3F2115387D5665E6),
    ("xxds4", 0x3F22FE142AD6D844),
    ("xxds5", 0x3F251D95692C1748),
    ("xxds6", 0x3F2779D809403699),
    ("xxds7", 0x3F2A19A6AB6F8CF9),
    ("xxds8", 0x3F2D048E4A0CE200),
    ("xxds9", 0x3F302179F98CDD44),
    ("xxds10", 0x3F31EF1677DAB285),
    ("xxds11", 0x3F33F04CCF611FB4),
    ("xxds12", 0x3F362AE1A802BFA6),
    ("xxds13", 0x3F38A53EBAD47393),
    ("xxds14", 0x3F3B668545D00F11),
    ("xxds15", 0x3F3E76A28F903D41),
    ("xxds16", 0x3F40EF335B0E2AF3),
    ("xxds17", 0x3F42D3CF0531FEA0),
    ("xxds18", 0x3F44EE96A0B0304C),
    ("xxds19", 0x3F47459865FC6260),
    ("xxds20", 0x3F49DF8FD7ECB26E),
    ("xxds21", 0x3F4CC3F922C031BE),
    ("xxds22", 0x3F4FFB26A57CE929),
    ("xxds23", 0x3F51C72C71CE74E4),
    ("xxds24", 0x3F53C3EC91F79B6B),
    ("xxds25", 0x3F55F98B84A26D3D),
    ("xxds26", 0x3F586E64BD092F83),
    ("xxds27", 0x3F5B29899AD9698A),
    ("xxds28", 0x3F5E32D5C0476AC2),
    ("xxds29", 0x3F60C982D70B9635),
    ("xxds30", 0x3F62A9E7F3542E0C),
    ("xxds31", 0x3F64C000702C7F67),
    ("xxds32", 0x3F6711CD0BD3ECB9),
    ("xxds33", 0x3F69A5FA4D3E7318),
    ("xxds34", 0x3F6C83F3B7FE38FF),
    ("xxds35", 0x3F6FB3F925B2E709),
    ("xxds36", 0x3F719F9B413630EB),
    ("xxds37", 0x3F7397EF18262E6F),
    ("xxds38", 0x3F75C8A32F2A4F82),
    ("xxds39", 0x3F783804D36655C9),
    ("xxds40", 0x3F7AED15A98C9F98),
    ("xxds41", 0x3F7DEF9FD6AEEC8D),
    ("xxds42", 0x3F80A42634FAA2EF),
    ("xxds43", 0x3F82805E23DC0717),
    ("xxds44", 0x3F8491D1EED7D837),
    ("xxds45", 0x3F86DE74F7F81201),
    ("xxds46", 0x3F896CE4EBA7CA80),
    ("xxds47", 0x3F8C447CC9DFCB0F),
    ("xxds48", 0x3F8F6D6A1011EBCA),
    ("xxds49", 0x3F917862205C08B5),
    ("xxds50", 0x3F936C53861D22D4),
    ("xdts1", 0x3EE77161F03B8D68),
    ("xdts2", 0x3EEA103E6F958E30),
    ("xdts3", 0x3EECFA18DA335F10),
    ("xdts4", 0x3EF01BA99BC1EBD0),
    ("xdts5", 0x3EF1E89FB9BEE440),
    ("xdts6", 0x3EF3E91D17CBBA40),
    ("xdts7", 0x3EF622E44991CBA8),
    ("xdts8", 0x3EF89C5CB86E2E28),
    ("xdts9", 0x3EFB5CA5107DD508),
    ("xdts10", 0x3EFE6BA7BCF49940),
    ("xdts11", 0x3F00E918D7584E60),
    ("xdts12", 0x3F02CD05D7491C10),
    ("xdts13", 0x3F04E70B4239FE30),
    ("xdts14", 0x3F073D3521E19EF8),
    ("xdts15", 0x3F09D63C8BE3D3E0),
    ("xdts16", 0x3F0CB99AF9DCBC90),
    ("xdts17", 0x3F0FEF9FCAFB9C00),
    ("xdts18", 0x3F11C0C416875130),
    ("xdts19", 0x3F13BCCCD8E5CA20),
    ("xdts20", 0x3F15F19FEE6907A0),
    ("xdts21", 0x3F1865967FB87488),
    ("xdts22", 0x3F1B1FBF605CDEE8),
    ("xdts23", 0x3F1E27F35D7F9700),
    ("xdts24", 0x3F20C375E8E694D8),
    ("xdts25", 0x3F22A32DDFB908E0),
    ("xdts26", 0x3F24B885DC355DB0),
    ("xdts27", 0x3F27097C72B987D8),
    ("xdts28", 0x3F299CBBC26DC8E0),
    ("xdts29", 0x3F2C79ACA2426CC0),
    ("xdts30", 0x3F2FA88BF2AD43B0),
    ("xdts31", 0x3F31994128BF2170),
    ("xdts32", 0x3F3390DF39FF1AB0),
    ("xdts33", 0x3F35C0C93994F538),
    ("xdts34", 0x3F382F4A2F2B1680),
    ("xdts35", 0x3F3AE36138FA9020),
    ("xdts36", 0x3F3DE4D5AD57CE60),
    ("xdts37", 0x3F409E26BE2ABE10),
    ("xdts38", 0x3F4279B308F1B160),
    ("xdts39", 0x3F448A680000F378),
    ("xdts40", 0x3F46D636E051CC08),
    ("xdts41", 0x3F4963BAF3DE1CE0),
    ("xdts42", 0x3F4C3A4C93F09170),
    ("xdts43", 0x3F4F62164B6F3720),
    ("xdts44", 0x3F5172162AF7B9A8),
    ("xdts45", 0x3F5365535F972F98),
    ("xdts46", 0x3F55905F370BA5E0),
    ("xdts47", 0x3F57F976B77487E8),
    ("xdts48", 0x3F5AA7896CB11678),
    ("xdts49", 0x3F5DA24D5D1F3A40),
    ("xdts50", 0x3F60792A9CB73498),
    ("simpson1", 0x3FD5555555555555),
    ("simpson2", 0x3FF5555555555555),
    ("simpson3", 0x3FE5555555555555),
    ("simpson4", 0x3FF5555555555555),
    ("simpson5", 0x3FE5555555555555),
    ("simpson6", 0x3FF5555555555555),
    ("simpson7", 0x3FE5555555555555),
    ("simpson8", 0x3FF5555555555555),
    ("simpson9", 0x3FE5555555555555),
    ("simpson10", 0x3FF5555555555555),
    ("simpson11", 0x3FE5555555555555),
    ("simpson12", 0x3FF5555555555555),
    ("simpson13", 0x3FE5555555555555),
    ("simpson14", 0x3FF5555555555555),
    ("simpson15", 0x3FE5555555555555),
    ("simpson16", 0x3FF5555555555555),
    ("simpson17", 0x3FE5555555555555),
    ("simpson18", 0x3FF5555555555555),
    ("simpson19", 0x3FE5555555555555),
    ("simpson20", 0x3FF5555555555555),
    ("simpson21", 0x3FE5555555555555),
    ("simpson22", 0x3FF5555555555555),
    ("simpson23", 0x3FE5555555555555),
    ("simpson24", 0x3FF5555555555555),
    ("simpson25", 0x3FE5555555555555),
    ("simpson26", 0x3FF5555555555555),
    ("simpson27", 0x3FE5555555555555),
    ("simpson28", 0x3FF5555555555555),
    ("simpson29", 0x3FE5555555555555),
    ("simpson30", 0x3FF5555555555555),
    ("simpson31", 0x3FE5555555555555),
    ("simpson32", 0x3FF5555555555555),
    ("simpson33", 0x3FE5555555555555),
    ("simpson34", 0x3FF5555555555555),
    ("simpson35", 0x3FE5555555555555),
    ("simpson36", 0x3FF5555555555555),
    ("simpson37", 0x3FE5555555555555),
    ("simpson38", 0x3FF5555555555555),
    ("simpson39", 0x3FE5555555555555),
    ("simpson40", 0x3FF5555555555555),
    ("simpson41", 0x3FE5555555555555),
    ("simpson42", 0x3FF5555555555555),
    ("simpson43", 0x3FE5555555555555),
    ("simpson44", 0x3FF5555555555555),
    ("simpson45", 0x3FE5555555555555),
    ("simpson46", 0x3FF5555555555555),
    ("simpson47", 0x3FE5555555555555),
    ("simpson48", 0x3FF5555555555555),
    ("simpson49", 0x3FE5555555555555),
    ("simpson50", 0x3FF5555555555555),
    ("simpson51", 0x3FD5555555555555),
)

#: ``kernels/wsm5.cu``'s ``WSM5_KMAX`` ladder: WSM6's, for the same reason
#: (one thread per column, per-thread work arrays sized by the tier).
WSM5_SHALLOW_KMAX = WSM6_SHALLOW_KMAX
WSM5_DEEP_KMAX = WSM6_DEEP_KMAX
WSM5_KERNEL_LEVEL_TIERS = WSM6_KERNEL_LEVEL_TIERS
#: ``(minimum, maximum)`` levels WSM5 admits: 2 (nislfv_rain_plm needs an
#: interface above and below) up to the deepest compiled tier.
WSM5_VERTICAL_LEVEL_BOUNDS = (2, WSM5_DEEP_KMAX)


def wsm5_level_tier(nz: int) -> int:
    """Return the ``WSM5_KMAX`` tier ``wsm5.cu`` compiles for ``nz``."""
    nz = int(nz)
    minimum, maximum = WSM5_VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WSM5 requires {minimum} <= nz <= {maximum}, got {nz}")
    return next(tier for tier in WSM5_KERNEL_LEVEL_TIERS if nz <= tier)


def _words(table, dtype):
    import numpy as np
    store = np.uint32 if dtype == "float32" else np.uint64
    return np.asarray([bits for _, bits in table], dtype=store).view(dtype)


def wsm5_init_words():
    """``wsm5init``'s float32 words as the array kernels/wsm5.cu reads."""
    return _words(WSM5_INIT_F32, "float32")


def wsm5_radar_words():
    """radar_init's words for ``refl10cm_wsm5``: (float32, float64) arrays."""
    return _words(WSM5_RADAR_F32, "float32"), _words(WSM5_RADAR_F64, "float64")


#: WSM6R (WRF ``mp_physics = 106``, phys/module_mp_wsm6r.F) -- the revised
#: WSM6 shares WSM6's species set and therefore this family module.  These
#: are the float32 words ``wsm6rinit`` (:332-451) leaves in the module's
#: SAVE variables, as gfortran 13.3.0 computes them from WRF v4.6.1's
#: unmodified source (its rgmma Gamma product and every runtime ``**``
#: included), dumped by tools/wsm6r_wrf461_column_oracle/dump_wsm6r_init.F90
#: and committed as tests/data/oracles/wsm6r/wsm6r-init-f32.txt.  They are
#: COPIED, never recomputed on the host: a float64 recomputation of a
#: Gamma-product constant is a different word (the WSM6 lesson, pvtr
#: 2500.0682 against WRF's 2500.0637).  The order is kernels/wsm6r.cu's
#: ``Wsm6rCoef`` enum; tests/test_wsm6r.py holds the two together and holds
#: these words to the committed dump.
WSM6R_INIT_F32: tuple[tuple[str, int], ...] = (
    ("pi", 0x40490FDB),
    ("qc0", 0x3A03C498),
    ("qck1", 0x40D8C3BE),
    ("roqimax", 0x38AA64C8),
    ("xlv1", 0x4512799A),
    ("vt2r_a", 0x41BC30CE),
    ("vt2s_a", 0x4036055D),
    ("vt2g_a", 0x4142AAE2),
    ("vt2i_a", 0x40533333),
    ("falli_a", 0x40533333),
    ("prevp_a", 0x43774FC6),
    ("prevp_b", 0x45A667C6),
    ("psdep_a", 0x434F7332),
    ("psdep_b", 0x459295AE),
    ("psevp_a", 0x434F7332),
    ("psevp_b", 0x459295AE),
    ("pgdep_a", 0x43774FC6),
    ("pgdep_b", 0x458E5137),
    ("pgevp_a", 0x43774FC6),
    ("pgevp_b", 0x458E5137),
    ("psmlt_a", 0x36042BA5),
    ("psmlt_b", 0x383A9415),
    ("pgmlt_a", 0x35C89EE5),
    ("pgmlt_b", 0x37E71445),
    ("praci_a", 0x4ABFBF63),
    ("praci_b", 0x330815BC),
    ("praci_c", 0x32AFD39D),
    ("praci_d", 0x31E3378A),
    ("psaci_a", 0x49BFBF63),
    ("psaci_b", 0x350747CA),
    ("psaci_c", 0x340B00CE),
    ("psaci_d", 0x328EDAE7),
    ("pgaci_a", 0x4A3FBF63),
    ("pgaci_b", 0x33C073FF),
    ("pgaci_c", 0x332FD39D),
    ("pgaci_d", 0x3220AAA9),
    ("pracs_a", 0x5A6068A9),
    ("pracs_b", 0x2665D0DD),
    ("pracs_c", 0x25123670),
    ("pracs_d", 0x23688EAA),
    ("psacr_a", 0x5C0C416A),
    ("psacr_b", 0x23123670),
    ("psacr_c", 0x23131507),
    ("psacr_d", 0x22B8F22E),
    ("pgacr_a", 0x5C8C416A),
    ("pgacr_b", 0x22A4715D),
    ("pgacr_c", 0x223A0BBB),
    ("pgacr_d", 0x21838DE4),
    ("pgacs_a", 0x59E068A9),
    ("pgacs_b", 0x26A2811D),
    ("pgacs_c", 0x25923670),
    ("pgacs_d", 0x2424715E),
    ("pidep_a", 0x48AA8AC0),
    ("diffac_a", 0x43EC5EB8),
    ("diffac_b", 0x4631AC00),
    ("pgfrz_a", 0x38D0C06B),
    ("piacr_a", 0x482A43F3),
    ("pracw_a", 0x406CB983),
    ("psacw_a", 0x3FFDA21B),
    ("pgacw_a", 0x402D268B),
)

#: mp_wsm6_init's two words (physics_mmm/mp_wsm6.F90:107 pidnc = pi*denr/6.,
#: :139 pidn0s = pi*dens*n0s, at module_physics_init.F's WSM6 arguments)
#: that WSM6's effective-radius routine reads.  gpuwm diagnoses WSM6's radii
#: on the WSM6R state for RTE+RRTMGP cloud optics (kernels/wsm6r.cu
#: ``wsm6r_effective_radii``); dumped beside WSM6R_INIT_F32.
WSM6R_RADII_F32: tuple[tuple[str, int], ...] = (
    ("wsm6_pidnc", 0x4402E653),
    ("wsm6_pidn0s", 0x4E15CD85),
)

#: ``(minimum, maximum)`` levels WSM6R admits.  Its kernel keeps the column
#: in the caller's global arrays plus one (3, nz, ny, nx) fall-speed
#: scratch, so the thread frame does not grow with nz and there is no
#: compiled tier ladder: like P3 the row is a floor with no declared
#: ceiling.  The floor is one level, which WRF's wsm6r also runs (kte = kts:
#: fallk's top-level block is then the surface level).
WSM6R_VERTICAL_LEVEL_BOUNDS = (1, None)


def wsm6r_init_words() -> tuple[int, ...]:
    """The WSM6R coefficient words in kernel order, as uint32 bit patterns."""
    return tuple(bits for _, bits in WSM6R_INIT_F32)


__all__ = ["WRF_INIT_F32", "WRF_INIT_NAMES", "WRF_RADAR_F32", "WRF_RADAR_F64",
           "WSM5_DEEP_KMAX", "WSM5_INIT_F32", "WSM5_KERNEL_LEVEL_TIERS",
           "WSM5_RADAR_F32", "WSM5_RADAR_F64", "WSM5_SHALLOW_KMAX",
           "WSM5_VERTICAL_LEVEL_BOUNDS", "WSM6Coefficients", "WSM6R_INIT_F32",
           "WSM6R_RADII_F32", "WSM6R_VERTICAL_LEVEL_BOUNDS",
           "WSM6RimedIceConstants", "WSM6_DEEP_KMAX",
           "WSM6_KERNEL_LEVEL_TIERS", "WSM6_SHALLOW_KMAX",
           "WSM6_VERTICAL_LEVEL_BOUNDS", "coefficients", "rimed_ice_constants",
           "wrf_init_words", "wrf_radar_words", "wsm5_init_words",
           "wsm5_level_tier", "wsm5_radar_words", "wsm6_level_tier",
           "wsm6r_init_words"]

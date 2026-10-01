"""Constants and tables of the WRF v4.7.1 UW moist-turbulence PBL port.

Every value here is the exact binary64 word the WRF oracle
(tools/uwpbl_wrf471_oracle, gfortran 15.2.0 -O0, glibc 2.43) holds at run
time, written as a hexadecimal float so no decimal conversion stands
between the word and the port.  They are recorded from the oracle's
``const/`` fixture records rather than recomputed, for two reasons:

* ``physconst`` / ``shr_const_mod`` build their values as Fortran PARAMETER
  expressions that gfortran folds at compile time with MPFR; the words are
  what the compiler produced, and copying the words is exact where a
  re-derivation would have to reproduce the folding.
* The saturation vapour pressure table (``gestbl`` via ``gffgch``,
  module_cam_wv_saturation.F / module_cam_gffgch.F, called from esinti with
  tmn=173.16, tmx=375.16, trice=20, ip=.true.) is built at initialization
  with glibc's binary64 ``log10``, ``log``, ``exp`` and ``pow``.  Taking the
  oracle's words keeps a CPU transcendental out of the path entirely (no
  host libm, no portable_math route to agree or disagree with glibc).
  tests/test_uwpbl_wrf471_parity.py re-reads the words from the committed
  oracle fixture, so a changed table cannot pass silently.
"""
from __future__ import annotations

from typing import Final

#: module_cam_physconst.F / module_cam_shr_const_mod.F, as held at run time.
CPAIR: Final[float] = float.fromhex("0x1.f651eb851eb85p+9")    # 1004.64
GRAVIT: Final[float] = float.fromhex("0x1.39cc100e6afcdp+3")   # 9.80616
RAIR: Final[float] = float.fromhex("0x1.1f0ad4eae9222p+8")     # RGAS/MWDAIR
ZVIR: Final[float] = float.fromhex("0x1.3730a75507baep-1")     # RWV/RDAIR-1
LATVAP: Final[float] = float.fromhex("0x1.314c400000000p+21")  # 2.501e6
LATICE: Final[float] = float.fromhex("0x1.45e1000000000p+18")  # 3.337e5
KARMAN: Final[float] = float.fromhex("0x1.999999999999ap-2")   # 0.4
EPSILO: Final[float] = float.fromhex("0x1.3e72edbda50a2p-1")   # MWWV/MWDAIR
RH2O: Final[float] = float.fromhex("0x1.cd81301343d8ap+8")     # RGAS/MWWV
TMELT: Final[float] = float.fromhex("0x1.1126666666666p+8")    # 273.15
#: constituents qmin(1) (Q, 1.E-12_r8); qmin/qmincg of the other four are 0.
QMIN_Q: Final[float] = float.fromhex("0x1.19799812dea11p-40")
#: eddy_diff init: b123 = b1**(2._r8/3._r8), b1 = 5.8, through glibc pow.
B123: Final[float] = float.fromhex("0x1.9d339a3d1e83dp+1")

#: wv_saturation table parameters (esinti -> gestbl).
ESTBL_TMIN: Final[float] = 173.16
ESTBL_TMAX: Final[float] = 375.16
ESTBL_TTRICE: Final[float] = 20.0
#: gestbl pcf(1..5), the degree-5 water-ice transition fit (pcf(6) unset).
PCF: Final[tuple[float, ...]] = (5.04469588506e-01, -5.47288442819e+00,
                                 -3.67471858735e-01, -8.95963532403e-03,
                                 -7.78053686625e-05)

#: estbl(1..250): 204 Goff-Gratch entries from 173.16 K in 1 K steps, then
#: gestbl's -99999 fill.  Index i of this tuple is Fortran estbl(i+1).
_ESTBL_HEX: Final[tuple[str, ...]] = (
    "0x1.7051e74897e06p-10", "0x1.c32944cd69aa3p-10", "0x1.13afe36d4dad8p-9",
    "0x1.502ae1ffca58dp-9", "0x1.990311948a9a4p-9", "0x1.f08fc6837f60cp-9",
    "0x1.2cc853eee88d8p-8", "0x1.6b9f1be9e83a9p-8", "0x1.b6ae5b788690dp-8",
    "0x1.08146b7db5825p-7", "0x1.3d4f1c92c759ap-7", "0x1.7c8480761b1c0p-7",
    "0x1.c76f2739d5647p-7", "0x1.1007f1dee49aep-6", "0x1.445bf45eea764p-6",
    "0x1.820a197157c36p-6", "0x1.ca9d32a7cc0f5p-6", "0x1.0fedd06e56c2dp-5",
    "0x1.41e7ba3a9fd3ep-5", "0x1.7c67c52e32d33p-5", "0x1.c0c4e03755845p-5",
    "0x1.0843979bb8a32p-4", "0x1.36b734c8dd50fp-4", "0x1.6cbc79fc4f6cbp-4",
    "0x1.ab75e40adbd34p-4", "0x1.f42d9f189d925p-4", "0x1.242d33a7fac5cp-3",
    "0x1.54d2789e97b9bp-3", "0x1.8cf6f33dc4cefp-3", "0x1.cdaa422cb831fp-3",
    "0x1.0c0f7ac848260p-2", "0x1.36d74b723fa16p-2", "0x1.67ef0cbd0c90fp-2",
    "0x1.a0315e583bbdep-2", "0x1.e093b4c9a5d56p-2", "0x1.1514a58f63d52p-1",
    "0x1.3f132e839ca42p-1", "0x1.6ef1d79ea8d96p-1", "0x1.a57136420ccfbp-1",
    "0x1.e3682ef418d62p-1", "0x1.14e32157132bap+0", "0x1.3ccb088e5a027p+0",
    "0x1.6a0009e7c3097p+0", "0x1.9d26cb4a28a23p+0", "0x1.d6f5f8ba978c2p+0",
    "0x1.0c1c02454a544p+1", "0x1.30e6869a255b3p+1", "0x1.5a5675c32f995p+1",
    "0x1.88f4b5f29e45dp+1", "0x1.bd585ac66de2fp+1", "0x1.f827f1723e04ap+1",
    "0x1.1d0d738750ca3p+2", "0x1.41fd855f5c762p+2", "0x1.6b5316b8e9295p+2",
    "0x1.9987f08e214dap+2", "0x1.cd21acac53092p+2", "0x1.03595bde4fc94p+3",
    "0x1.236db30a0c6f8p+3", "0x1.47258e3338132p+3", "0x1.6ee0c6c12c652p+3",
    "0x1.9b0807257f204p+3", "0x1.cc0d80972f333p+3", "0x1.0136d6994037fp+4",
    "0x1.1f580f9517815p+4", "0x1.40b42ede9374ep+4", "0x1.659b68554283ep+4",
    "0x1.8e64dd7229753p+4", "0x1.bb6f22a4162a7p+4", "0x1.ed20cd15e4a50p+4",
    "0x1.11f484a490673p+5", "0x1.30201d797093bp+5", "0x1.5154534616563p+5",
    "0x1.75d792855b8e5p+5", "0x1.9df5f11d1cb24p+5", "0x1.ca0193191308fp+5",
    "0x1.fa53153379af0p+5", "0x1.17a4feb66ad98p+6", "0x1.34a698faf931bp+6",
    "0x1.5465bb586d125p+6", "0x1.771df8c7a335ep+6", "0x1.9d1ab0658eac2p+6",
    "0x1.cb24d636462bap+6", "0x1.fd4baa11ce6dep+6", "0x1.19ea2333f3ec2p+7",
    "0x1.3782f2e1518cap+7", "0x1.5794d736a811fp+7", "0x1.7a4539b81ddb9p+7",
    "0x1.9fba34eeb974ep+7", "0x1.c81a68ada5a0ap+7", "0x1.f38cc764bd602p+7",
    "0x1.111c2df348588p+8", "0x1.2a22037fd1898p+8", "0x1.44eb1a19fdaf7p+8",
    "0x1.618a42779ddb9p+8", "0x1.8011b8b5300aap+8", "0x1.a092ef407ae19p+8",
    "0x1.c31e53d30f294p+8", "0x1.e7c30e1000060p+8", "0x1.07475ba999ae0p+9",
    "0x1.1bc6859dd2e97p+9", "0x1.3163cf541bcadp+9", "0x1.484eef36eea99p+9",
    "0x1.60bcdab8d3cebp+9", "0x1.7ac3541ed6732p+9", "0x1.96791c30785bep+9",
    "0x1.b3f5fabea86c1p+9", "0x1.d352c7510a97ep+9", "0x1.f4a971f8fc8c6p+9",
    "0x1.0c0a8624dba77p+10", "0x1.1ed8e93a71cb9p+10", "0x1.32ce9a4575f00p+10",
    "0x1.47faefef268b1p+10", "0x1.5e6de4448e1bcp+10", "0x1.76381989f6d70p+10",
    "0x1.8f6adf1ea0971p+10", "0x1.aa18367045b6ap+10", "0x1.c652d7fdf6fdap+10",
    "0x1.e42e3869d240ep+10", "0x1.01df46cc8975dp+11", "0x1.128c69f0fbdf4p+11",
    "0x1.242969a37a666p+11", "0x1.36c1925f528c6p+11", "0x1.4a609bc3f5ba0p+11",
    "0x1.5f12ab44d7737p+11", "0x1.74e456decda9dp+11", "0x1.8be2a7d2a6382p+11",
    "0x1.a41b1d64a2d51p+11", "0x1.bd9bafa08bac0p+11", "0x1.d872d22207716p+11",
    "0x1.f4af76e0e6997p+11", "0x1.09308880872d0p+12", "0x1.18cbcbd2d7a12p+12",
    "0x1.2931c463b9118p+12", "0x1.3a6af606a2c91p+12", "0x1.4c802ad3bac4cp+12",
    "0x1.5f7a74965b107p+12", "0x1.73632e3c650a4p+12", "0x1.8843fd4636b2fp+12",
    "0x1.9e26d337162fep+12", "0x1.b515ef05e77e2p+12", "0x1.cd1bde8e006c0p+12",
    "0x1.e6437fffef143p+12", "0x1.004c01a90381fp+13", "0x1.0e1275d844d0dp+13",
    "0x1.1c7b08769f3a2p+13", "0x1.2b8bd07728271p+13", "0x1.3b4b108e225f3p+13",
    "0x1.4bbf37e771ae2p+13", "0x1.5ceee2dc71a73p+13", "0x1.6ee0dba91abdbp+13",
    "0x1.819c1b2061606p+13", "0x1.9527c95fbaea6p+13", "0x1.a98b3e81b4b04p+13",
    "0x1.bece034f899efp+13", "0x1.d4f7d1f1a3650p+13", "0x1.ec10969ef4688p+13",
    "0x1.021038258c130p+14", "0x1.0e97d8a994236p+14", "0x1.1ba370149a585p+14",
    "0x1.29375bff26eb8p+14", "0x1.375814b5884d7p+14", "0x1.460a2d8b51e11p+14",
    "0x1.5552552e0bbffp+14", "0x1.653555f70c0f2p+14", "0x1.75b8163c729e3p+14",
    "0x1.86df98a13fe83p+14", "0x1.98b0fc6480c59p+14", "0x1.ab317daf88781p+14",
    "0x1.be6675e332eabp+14", "0x1.d2555be429766p+14", "0x1.e703c466249adp+14",
    "0x1.fc77623625944p+14", "0x1.095b0341d16cep+15", "0x1.14e2d09451948p+15",
    "0x1.20d6207861557p+15", "0x1.2d3809ef8f491p+15", "0x1.3a0bb3b991b2dp+15",
    "0x1.475454761d728p+15", "0x1.551532c62b45fp+15", "0x1.6351a56caa10bp+15",
    "0x1.720d136e9d162p+15", "0x1.814af432a517cp+15", "0x1.910ecf9ff3b04p+15",
    "0x1.a15c3e3ca82ddp+15", "0x1.b236e94b9589dp+15", "0x1.c3a28ae971276p+15",
    "0x1.d5a2ee296a3ebp+15", "0x1.e83bef312a065p+15", "0x1.fb717b543cb19p+15",
    "0x1.07a3c89771e68p+16", "0x1.11e1206029264p+16", "0x1.1c72d5c229952p+16",
    "0x1.275b02462d9e4p+16", "0x1.329bc84c03bb0p+16", "0x1.3e3753168443dp+16",
    "0x1.4a2fd6d73ea1cp+16", "0x1.568790b9dea19p+16", "0x1.6340c6ef4aaa2p+16",
    "0x1.705dc8b87bb4ap+16", "0x1.7de0ee7110054p+16", "0x1.8bcc999999984p+16",
    "0x1.9a2334e1a960ap+16", "0x1.a8e73431987acp+16", "0x1.b81b14b410954p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16", "-0x1.869f000000000p+16", "-0x1.869f000000000p+16",
    "-0x1.869f000000000p+16",
)
ESTBL: Final[tuple[float, ...]] = tuple(float.fromhex(h) for h in _ESTBL_HEX)
assert len(ESTBL) == 250

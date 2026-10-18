// Apache-2.0. Shared GFS kind_phys=8 constants from the pinned CCPP physcons.
// CCPP 1d9b076503c27cd4cfa8b22a977a71e889cbb149, physics/hooks/physcons.F90.
// Explicit kind_phys literals retain binary64 precision.
#pragma once
namespace gfs_physcons {
constexpr double con_g=9.80665,con_rd=287.05,con_cp=1004.6,con_rv=461.5;
constexpr double con_hvap=2500000.,con_hfus=333580.;
constexpr double con_cvap=1846.,con_cliq=4185.5,con_csol=2106.;
constexpr double con_psat=610.78,con_ttp=273.16;
constexpr double con_eps=con_rd/con_rv,con_epsm1=con_eps-1.;
constexpr double con_fvirt=con_rv/con_rd-1.;
}

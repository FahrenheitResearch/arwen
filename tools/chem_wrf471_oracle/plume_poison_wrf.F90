! Controlled experiment only: no poison values enter published reference fixtures.
module plume_poison
use module_zero_plumegen_coms
implicit none
private
public :: poison_state
contains
subroutine poison_state
character(64) :: label
call get_command_argument(2,label)
if(label=='all' .or. label=='w') w=-1.e10
if(label=='all' .or. label=='t') t=-1.e10
if(label=='all' .or. label=='qv') qv=-1.e10
if(label=='all' .or. label=='qc') qc=-1.e10
if(label=='all' .or. label=='qh') qh=-1.e10
if(label=='all' .or. label=='qi') qi=-1.e10
if(label=='all' .or. label=='sc') sc=-1.e10
if(label=='all' .or. label=='vth') vth=-1.e10
if(label=='all' .or. label=='vti') vti=-1.e10
if(label=='all' .or. label=='rho') rho=-1.e10
if(label=='all' .or. label=='txs') txs=-1.e10
if(label=='all' .or. label=='est') est=-1.e10
if(label=='all' .or. label=='qsat') qsat=-1.e10
if(label=='all' .or. label=='qpas') qpas=-1.e10
if(label=='all' .or. label=='qtotal') qtotal=-1.e10
if(label=='all' .or. label=='wc') wc=-1.e10
if(label=='all' .or. label=='wt') wt=-1.e10
if(label=='all' .or. label=='tt') tt=-1.e10
if(label=='all' .or. label=='qvt') qvt=-1.e10
if(label=='all' .or. label=='qct') qct=-1.e10
if(label=='all' .or. label=='qht') qht=-1.e10
if(label=='all' .or. label=='qit') qit=-1.e10
if(label=='all' .or. label=='sct') sct=-1.e10
if(label=='all' .or. label=='vctr1') vctr1=-1.e10
if(label=='all' .or. label=='vctr2') vctr2=-1.e10
if(label=='all' .or. label=='vt3dc') vt3dc=-1.e10
if(label=='all' .or. label=='vt3df') vt3df=-1.e10
if(label=='all' .or. label=='vt3dk') vt3dk=-1.e10
if(label=='all' .or. label=='vt3dg') vt3dg=-1.e10
if(label=='all' .or. label=='scr1') scr1=-1.e10
if(label=='all' .or. label=='pke') pke=-1.e10
if(label=='all' .or. label=='the') the=-1.e10
if(label=='all' .or. label=='thve') thve=-1.e10
if(label=='all' .or. label=='thee') thee=-1.e10
if(label=='all' .or. label=='pe') pe=-1.e10
if(label=='all' .or. label=='te') te=-1.e10
if(label=='all' .or. label=='qvenv') qvenv=-1.e10
if(label=='all' .or. label=='rhe') rhe=-1.e10
if(label=='all' .or. label=='dne') dne=-1.e10
if(label=='all' .or. label=='sce') sce=-1.e10
if(label=='all' .or. label=='ucon') ucon=-1.e10
if(label=='all' .or. label=='vcon') vcon=-1.e10
if(label=='all' .or. label=='wcon') wcon=-1.e10
if(label=='all' .or. label=='thtcon') thtcon=-1.e10
if(label=='all' .or. label=='rvcon') rvcon=-1.e10
if(label=='all' .or. label=='picon') picon=-1.e10
if(label=='all' .or. label=='tmpcon') tmpcon=-1.e10
if(label=='all' .or. label=='dncon') dncon=-1.e10
if(label=='all' .or. label=='prcon') prcon=-1.e10
if(label=='all' .or. label=='zcon') zcon=-1.e10
if(label=='all' .or. label=='zzcon') zzcon=-1.e10
if(label=='all' .or. label=='scon') scon=-1.e10
if(label=='all' .or. label=='dqsdz') dqsdz=1.e10
if(label=='all' .or. label=='visc') visc=-1.e10
if(label=='all' .or. label=='viscosity') viscosity=1.e10
if(label=='all' .or. label=='tstpf') tstpf=1.e10
if(label=='all' .or. label=='n') n=100000
if(label=='all' .or. label=='nm1') nm1=100000
if(label=='all' .or. label=='l') l=100000
if(label=='all' .or. label=='advw') advw=-1.e10
if(label=='all' .or. label=='advt') advt=-1.e10
if(label=='all' .or. label=='advv') advv=1.e10
if(label=='all' .or. label=='advc') advc=1.e10
if(label=='all' .or. label=='advh') advh=1.e10
if(label=='all' .or. label=='advi') advi=1.e10
if(label=='all' .or. label=='cvh') cvh=-1.e10
if(label=='all' .or. label=='cvi') cvi=-1.e10
if(label=='all' .or. label=='adiabat') adiabat=-1.e10
if(label=='all' .or. label=='wbar') wbar=1.e10
if(label=='all' .or. label=='alast') alast=-1.e10
if(label=='all' .or. label=='vhrel') vhrel=1.e10
if(label=='all' .or. label=='virel') virel=1.e10
if(label=='all' .or. label=='zbase') zbase=1.e10
if(label=='all' .or. label=='ztop') ztop=1.e10
if(label=='all' .or. label=='lbase') lbase=100000
if(label=='all' .or. label=='area') area=1.e10
if(label=='all' .or. label=='rsurf') rsurf=1.e10
if(label=='all' .or. label=='alpha') alpha=1.e10
if(label=='all' .or. label=='radius') radius=-1.e10
if(label=='all' .or. label=='heating') heating=-1.e10
if(label=='all' .or. label=='fmoist') fmoist=-1.e10
if(label=='all' .or. label=='bload') bload=1.e10
if(label=='all' .or. label=='dt') dt=-1.e10
if(label=='all' .or. label=='time') time=1.e10
if(label=='all' .or. label=='tdur') tdur=1.e10
if(label=='all' .or. label=='mintime') mintime=100000
if(label=='all' .or. label=='mdur') mdur=100000
if(label=='all' .or. label=='maxtime') maxtime=100000
if(label=='all' .or. label=='w_vmd') w_vmd=-1.e10
if(label=='all' .or. label=='vmd') vmd=-1.e10
if(label=='all' .or. label=='upe') upe=-1.e10
if(label=='all' .or. label=='vpe') vpe=-1.e10
if(label=='all' .or. label=='vel_e') vel_e=-1.e10
if(label=='all' .or. label=='vel_p') vel_p=-1.e10
if(label=='all' .or. label=='rad_p') rad_p=-1.e10
if(label=='all' .or. label=='vel_t') vel_t=-1.e10
if(label=='all' .or. label=='rad_t') rad_t=-1.e10
if(label=='all' .or. label=='ztop_') ztop_=-1.e10
if(label=='cvi_1') cvi(1)=-1.e10
if(label=='cvi_tail') cvi(2:nkp)=-1.e10
end subroutine
end module

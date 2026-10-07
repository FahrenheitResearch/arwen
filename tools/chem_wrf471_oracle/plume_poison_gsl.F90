! Controlled experiment only: no poison values enter published reference fixtures.
module plume_poison
use module_zero_plumegen_coms
implicit none
private
public :: poison_state
contains
subroutine poison_state(coms)
character(64) :: label
type(plumegen_coms), pointer :: coms
call get_command_argument(2,label)
if(label=='all' .or. label=='w') coms%w=-1.e10
if(label=='all' .or. label=='t') coms%t=-1.e10
if(label=='all' .or. label=='qv') coms%qv=-1.e10
if(label=='all' .or. label=='qc') coms%qc=-1.e10
if(label=='all' .or. label=='qh') coms%qh=-1.e10
if(label=='all' .or. label=='qi') coms%qi=-1.e10
if(label=='all' .or. label=='sc') coms%sc=-1.e10
if(label=='all' .or. label=='vth') coms%vth=-1.e10
if(label=='all' .or. label=='vti') coms%vti=-1.e10
if(label=='all' .or. label=='rho') coms%rho=-1.e10
if(label=='all' .or. label=='txs') coms%txs=-1.e10
if(label=='all' .or. label=='est') coms%est=-1.e10
if(label=='all' .or. label=='qsat') coms%qsat=-1.e10
if(label=='all' .or. label=='wc') coms%wc=-1.e10
if(label=='all' .or. label=='wt') coms%wt=-1.e10
if(label=='all' .or. label=='tt') coms%tt=-1.e10
if(label=='all' .or. label=='qvt') coms%qvt=-1.e10
if(label=='all' .or. label=='qct') coms%qct=-1.e10
if(label=='all' .or. label=='qht') coms%qht=-1.e10
if(label=='all' .or. label=='qit') coms%qit=-1.e10
if(label=='all' .or. label=='sct') coms%sct=-1.e10
if(label=='all' .or. label=='vctr1') coms%vctr1=-1.e10
if(label=='all' .or. label=='vctr2') coms%vctr2=-1.e10
if(label=='all' .or. label=='vt3dc') coms%vt3dc=-1.e10
if(label=='all' .or. label=='vt3df') coms%vt3df=-1.e10
if(label=='all' .or. label=='vt3dk') coms%vt3dk=-1.e10
if(label=='all' .or. label=='vt3dg') coms%vt3dg=-1.e10
if(label=='all' .or. label=='scr1') coms%scr1=-1.e10
if(label=='all' .or. label=='pke') coms%pke=-1.e10
if(label=='all' .or. label=='the') coms%the=-1.e10
if(label=='all' .or. label=='thve') coms%thve=-1.e10
if(label=='all' .or. label=='thee') coms%thee=-1.e10
if(label=='all' .or. label=='pe') coms%pe=-1.e10
if(label=='all' .or. label=='te') coms%te=-1.e10
if(label=='all' .or. label=='qvenv') coms%qvenv=-1.e10
if(label=='all' .or. label=='dne') coms%dne=-1.e10
if(label=='all' .or. label=='ucon') coms%ucon=-1.e10
if(label=='all' .or. label=='vcon') coms%vcon=-1.e10
if(label=='all' .or. label=='thtcon') coms%thtcon=-1.e10
if(label=='all' .or. label=='rvcon') coms%rvcon=-1.e10
if(label=='all' .or. label=='picon') coms%picon=-1.e10
if(label=='all' .or. label=='tmpcon') coms%tmpcon=-1.e10
if(label=='all' .or. label=='zcon') coms%zcon=-1.e10
if(label=='all' .or. label=='zzcon') coms%zzcon=-1.e10
if(label=='all' .or. label=='dqsdz') coms%dqsdz=1.e10
if(label=='all' .or. label=='visc') coms%visc=-1.e10
if(label=='all' .or. label=='viscosity') coms%viscosity=1.e10
if(label=='all' .or. label=='tstpf') coms%tstpf=1.e10
if(label=='all' .or. label=='n') coms%n=100000
if(label=='all' .or. label=='nm1') coms%nm1=100000
if(label=='all' .or. label=='l') coms%l=100000
if(label=='all' .or. label=='cvh') coms%cvh=-1.e10
if(label=='all' .or. label=='cvi') coms%cvi=-1.e10
if(label=='all' .or. label=='adiabat') coms%adiabat=1.e10
if(label=='all' .or. label=='wbar') coms%wbar=1.e10
if(label=='all' .or. label=='vhrel') coms%vhrel=1.e10
if(label=='all' .or. label=='virel') coms%virel=1.e10
if(label=='all' .or. label=='ztop') coms%ztop=1.e10
if(label=='all' .or. label=='area') coms%area=1.e10
if(label=='all' .or. label=='rsurf') coms%rsurf=1.e10
if(label=='all' .or. label=='alpha') coms%alpha=1.e10
if(label=='all' .or. label=='radius') coms%radius=-1.e10
if(label=='all' .or. label=='heating') coms%heating=-1.e10
if(label=='all' .or. label=='fmoist') coms%fmoist=1.e10
if(label=='all' .or. label=='bload') coms%bload=1.e10
if(label=='all' .or. label=='dt') coms%dt=1.e10
if(label=='all' .or. label=='time') coms%time=1.e10
if(label=='all' .or. label=='tdur') coms%tdur=1.e10
if(label=='all' .or. label=='mintime') coms%mintime=100000
if(label=='all' .or. label=='mdur') coms%mdur=100000
if(label=='all' .or. label=='maxtime') coms%maxtime=100000
if(label=='all' .or. label=='upe') coms%upe=-1.e10
if(label=='all' .or. label=='vpe') coms%vpe=-1.e10
if(label=='all' .or. label=='vel_e') coms%vel_e=-1.e10
if(label=='all' .or. label=='vel_p') coms%vel_p=-1.e10
if(label=='all' .or. label=='rad_p') coms%rad_p=-1.e10
if(label=='all' .or. label=='vel_t') coms%vel_t=-1.e10
if(label=='all' .or. label=='rad_t') coms%rad_t=-1.e10
if(label=='all' .or. label=='ztop_') coms%ztop_=-1.e10
if(label=='all' .or. label=='testval') coms%testval=100000
end subroutine
end module

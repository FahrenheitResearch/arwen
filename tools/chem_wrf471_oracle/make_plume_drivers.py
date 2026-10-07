"""Write explicit original/observed drivers for the pinned plume solvers.

Synthetic sounding families are fixture inputs, not model defaults. The
observed build must match the original driver's emissions word for word.
"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
common='''
  implicit none
  integer, parameter :: maxnz=120, nr=3
  real :: t_in(1,maxnz,1),p_in(1,maxnz,1),rho_in(1,maxnz,1)
  real :: u_in(1,maxnz,1),v_in(1,maxnz,1),w_in(1,maxnz,1)
  real :: z_in(1,maxnz+1,1),zw_in(1,maxnz+1,1),q_in(1,maxnz,1,1)
  real :: ebu(1,maxnz,1,nr),baseline(1,maxnz,1,nr),incoming(1,1,1,nr)
  real :: dz0,agl,temperature,pressure,humidity,terrain,psurf,speed,gradient
  real :: prop(4),size(4),windavg(1,1),pblheight(1,1),frp_values(11)
  integer :: nz,iz,family,fire,c,k,r,pass,errflg,mode
  character(1024) :: root,errmsg
  character(64) :: case_name
  call oracle_root(root)
  do iz=1,2
    nz=49; if(iz==2) nz=59
    do family=1,6
      terrain=0.; psurf=100000.; humidity=.006; speed=0.; gradient=.004
      select case(family)
      case(1); humidity=.001; gradient=.002
      case(2); humidity=.001; gradient=.012
      case(3); humidity=.014; gradient=.002
      case(4); humidity=.008; speed=8.; gradient=.006
      case(5); terrain=2500.; psurf=75000.; humidity=.004; speed=4.
      case(6); humidity=.020; gradient=.003; speed=2.
      end select
      dz0=150.; if(iz==2) dz0=110.
      do k=1,nz+1
        agl=real(k-1)*dz0+real((k-1)**2)*4.5
        zw_in(1,k,1)=terrain+agl
      enddo
      do k=1,nz
        z_in(1,k,1)=.5*(zw_in(1,k,1)+zw_in(1,k+1,1))
        agl=z_in(1,k,1)-terrain
        pressure=psurf*exp(-agl/8000.)
        temperature=max(205.,300.-.0065*agl)
        if(family==1 .or. family==3) then
          temperature=300.*exp(-min(agl,2000.)/8000.*(287./1004.5))
          if(agl>2000.) temperature=max(205.,temperature-.0065*(agl-2000.))
        endif
        if(family==2) temperature=temperature+8.*(1.-exp(-agl/500.))
        p_in(1,k,1)=pressure; t_in(1,k,1)=temperature
        q_in(1,k,1,1)=humidity*exp(-agl/2500.)
        u_in(1,k,1)=speed+speed*.001*agl
        v_in(1,k,1)=speed*.0005*agl
        w_in(1,k,1)=0.
        rho_in(1,k,1)=pressure/(287.*temperature)*(1.+q_in(1,k,1,1))
      enddo
      windavg=speed; pblheight=2500.
'''
dump='''
        call oracle_open(trim(case_name))
        call oracle_put('nz',nz)
        call oracle_put('t_phy',t_in(1,1:nz,1))
        call oracle_put('p_phy',p_in(1,1:nz,1))
        call oracle_put('rho',rho_in(1,1:nz,1))
        call oracle_put('qv_in',q_in(1,1:nz,1,1))
        call oracle_put('u_phy',u_in(1,1:nz,1))
        call oracle_put('v_phy',v_in(1,1:nz,1))
        call oracle_put('z',z_in(1,1:nz,1))
        call oracle_put('z_at_w',zw_in(1,1:nz+1,1))
        call oracle_put('ebu_in',incoming(1,1,1,:))
        call oracle_put('ebu',ebu(1,1:nz,1,:))
#ifdef PLUME_TRACE
        call oracle_put('ztopmax',trace_tops)
        call oracle_put('steps',trace_steps)
        call oracle_put('solver_k_min',trace_k1)
        call oracle_put('solver_k_max',trace_k2)
'''
profiles_wrf='''
        call oracle_put('w',w); call oracle_put('t',t); call oracle_put('qv',qv)
        call oracle_put('qc',qc); call oracle_put('qh',qh); call oracle_put('qi',qi)
        call oracle_put('radius',radius)
'''
profiles_gsl='''
        call oracle_put('w',real(coms%w,4)); call oracle_put('t',real(coms%t,4))
        call oracle_put('qv',real(coms%qv,4)); call oracle_put('qc',real(coms%qc,4))
        call oracle_put('qh',real(coms%qh,4)); call oracle_put('qi',real(coms%qi,4))
        call oracle_put('radius',real(coms%radius,4))
'''
wrf='''! WRF f52c197e: real generic column driver with three emitted rows.
program run_plumerise_wrfchem
  use oracle_io
  use module_configure
  use module_zero_plumegen_coms
  use module_chem_plumerise_scalar, only: set_grid
  use plume_column_driver, only: original_driver=>plumerise_driver
#ifdef PLUME_TRACE
  use plume_trace
  use plume_column_driver_observed, only: observed_driver=>plumerise_driver
#endif
'''+common.replace('  implicit none','  type(grid_config_rec_type) :: cfg\n  real :: emis_ant(1,1,1,1)\n  implicit none',1)
# IMPLICIT NONE must precede declarations.
wrf=wrf.replace('  type(grid_config_rec_type) :: cfg\n  real :: emis_ant(1,1,1,1)\n  implicit none','  implicit none\n  type(grid_config_rec_type) :: cfg\n  real :: emis_ant(1,1,1,1)')
wrf+='''
      do fire=1,9
        prop=0.; size=0.
        if(fire<=4) then
          prop(fire)=.6; size(fire)=100000.
        elseif(fire==5) then
          prop=(/.3,.2,.4,.1/); size=(/100000.,200000.,300000.,50000./)
        elseif(fire==6) then
          prop=(/.1,.4,.2,.3/); size=(/200000.,50000.,100000.,150000./)
        elseif(fire==7) then
          size=100000.
        elseif(fire==8) then
          prop=.2
        else
          prop=.2; size=100000.
        endif
        incoming(1,1,1,:)=(/1.234567,2.5,.012345/)
        if(fire==9) incoming=0.
        do pass=1,2
#ifndef PLUME_TRACE
          if(pass==2) exit
#endif
          call zero_plumegen_coms; call set_grid
          ebu=0.; ebu(1,1,1,:)=incoming(1,1,1,:)
          emis_ant=0.
#ifdef PLUME_TRACE
          call trace_clear
          if(pass==2) then
            call observed_driver(1,1,36.,ebu,incoming,prop(1:1),prop(2:2),prop(3:3),prop(4:4), &
                size(1:1),size(2:2),size(3:3),size(4:4),cfg,t_in,q_in,rho_in,w_in,u_in,v_in,p_in, &
                emis_ant,zw_in,z_in,.false.,1,1,1,1,1,nz,1,1,1,1,1,maxnz,1,1,1,1,1,nz)
          else
#endif
            call original_driver(1,1,36.,ebu,incoming,prop(1:1),prop(2:2),prop(3:3),prop(4:4), &
                size(1:1),size(2:2),size(3:3),size(4:4),cfg,t_in,q_in,rho_in,w_in,u_in,v_in,p_in, &
                emis_ant,zw_in,z_in,.false.,1,1,1,1,1,nz,1,1,1,1,1,maxnz,1,1,1,1,1,nz)
#ifdef PLUME_TRACE
          endif
#endif
          if(pass==1) baseline=ebu
          if(any(transfer(ebu,[0],size(ebu))/=transfer(baseline,[0],size(baseline)))) error stop 'observer changed emissions'
        enddo
        write(case_name,'("nz",I2.2,"_s",I1,"_f",I2.2)') nz,family,fire
'''+dump+profiles_wrf+'''#endif
        call oracle_put('mean_fct',prop); call oracle_put('firesize',size)
        call oracle_close()
      enddo
    enddo
  enddo
end program
'''
# SIZE intrinsic is hidden by the four-group size array; use fixed transfer extents.
wrf=wrf.replace('size(ebu)','maxnz*nr').replace('size(baseline)','maxnz*nr')
gsl='''! GSL 3e6660c6: ebu_driver with the real solver live, debug disabled.
program gsl_run_plumerise_frp
  use machine, only: kind_phys
  use oracle_io
  use rrfs_smoke_config
  use module_zero_plumegen_coms
  use module_plumerise, only: original_driver=>ebu_driver
#ifdef PLUME_TRACE
  use plume_trace
  use module_plumerise_observed, only: observed_driver=>ebu_driver
#endif
'''+common.replace('  implicit none','''  implicit none
  type(plumegen_coms), pointer :: coms
  real(kind_phys) :: theta(1,maxnz,1),pi_in(1,maxnz,1)
  real(kind_phys) :: ee(1,maxnz,1),base(1,maxnz,1),fla(1,1),power(1,1),emis(1,1)
  real(kind_phys) :: xlat(1,1)=0.,xlong(1,1)=0.
  real(kind_phys) :: tt(1,maxnz,1),pp(1,maxnz,1),rr(1,maxnz,1)
  real(kind_phys) :: uu(1,maxnz,1),vv(1,maxnz,1),ww(1,maxnz,1),qq(1,maxnz,1)
  real(kind_phys) :: zz(1,maxnz+1,1),zw(1,maxnz+1,1)
  integer :: kpbl(1,1),kmin(1,1),kmax(1,1)
''',1)
# Dimensions need constants declared before these additions.
gsl=gsl.replace('  implicit none\n','  implicit none\n  integer, parameter :: maxnz=120, nr=3\n',1).replace('  integer, parameter :: maxnz=120, nr=3\n  real :: t_in','  real :: t_in',1)
gsl+='''
      dbg_opt=.false.; coms=>get_thread_coms()
      frp_values=(/0.,9999999.,1.e7,10000001.,1.e8,999999936.,1.e9,1000000064.,5.e9,2.e10,3.e10/)
      do fire=1,11
        incoming(1,1,1,:)=(/1.234567,2.5,.012345/)
        power=min(real(frp_values(fire),kind_phys),real(2.e10,kind_phys))
        kpbl=24
        tt=real(t_in,kind_phys); pp=real(p_in,kind_phys); rr=real(rho_in,kind_phys)
        uu=real(u_in,kind_phys); vv=real(v_in,kind_phys); ww=0.; qq=real(q_in(:,:,:,1),kind_phys)
        zz=real(z_in,kind_phys); zw=real(zw_in,kind_phys)
        do k=1,nz
          pi_in(1,k,1)=real(1004.5,kind_phys)*(pp(1,k,1)/real(100000.,kind_phys))**(real(287.,kind_phys)/real(1004.5,kind_phys))
          theta(1,k,1)=tt(1,k,1)/pi_in(1,k,1)*real(1004.5,kind_phys)
        enddo
        do pass=1,2
#ifndef PLUME_TRACE
          if(pass==2) exit
#endif
          call coms%set_to_zero()
          ee=0.; emis=real(incoming(1,1,1,1),kind_phys); errmsg=''; errflg=0
#ifdef PLUME_TRACE
          call trace_clear
          if(pass==2) then
            call observed_driver(fla,emis,ee,theta,qq,rr,ww,uu,vv,pi_in,ww,zw,zz, &
                real(9.81,kind_phys),real(1004.5,kind_phys),real(287.,kind_phys),power,kmin,kmax,1,kpbl,kpbl, &
                real(0.,kind_phys),xlat,xlong,windavg,pblheight,0,real(.05,kind_phys), &
                real(1.e7,kind_phys),real(1.e9,kind_phys),real(2000.,kind_phys),real(5.,kind_phys), &
                1,1,1,1,1,nz,1,1,1,1,1,maxnz,1,1,1,1,1,nz,errmsg,errflg)
          else
#endif
            call original_driver(fla,emis,ee,theta,qq,rr,ww,uu,vv,pi_in,ww,zw,zz, &
                real(9.81,kind_phys),real(1004.5,kind_phys),real(287.,kind_phys),power,kmin,kmax,1,kpbl,kpbl, &
                real(0.,kind_phys),xlat,xlong,windavg,pblheight,0,real(.05,kind_phys), &
                real(1.e7,kind_phys),real(1.e9,kind_phys),real(2000.,kind_phys),real(5.,kind_phys), &
                1,1,1,1,1,nz,1,1,1,1,1,maxnz,1,1,1,1,1,nz,errmsg,errflg)
#ifdef PLUME_TRACE
          endif
#endif
          if(errflg/=0) error stop 'plume environment failed'
          if(pass==1) base=ee
          if(any(transfer(ee,[0],maxnz*kind_phys/4)/=transfer(base,[0],maxnz*kind_phys/4))) error stop 'observer changed emissions'
        enddo
        ebu=0.; ebu(1,:,1,1)=real(ee(1,:,1),4)
        do r=2,nr
          ! Species-independent driver replay for each emitted row.
          do k=1,nz
            ebu(1,k,1,r)=real(fla(1,1)*real(incoming(1,1,1,r),kind_phys)*(zw(1,k+1,1)-zw(1,k,1))/(zw(1,kmax(1,1),1)-zw(1,kmin(1,1),1)),4)
            if(k<kmin(1,1) .or. k>=kmax(1,1)) ebu(1,k,1,r)=0.
          enddo
          ebu(1,1,1,r)=real((1.-fla(1,1))*real(incoming(1,1,1,r),kind_phys),4)
        enddo
        write(case_name,'("nz",I2.2,"_s",I1,"_f",I2.2)') nz,family,fire
'''+dump+profiles_gsl+'''#endif
        call oracle_put('frp_inst',real(power(1,1),4))
        call oracle_put('frp_raw',frp_values(fire))
        call oracle_put('flam_frac',real(fla(1,1),4))
        call oracle_put('k_min',kmin(1,1)); call oracle_put('k_max',kmax(1,1))
        call oracle_put('kpbl',kpbl(1,1)); call oracle_put('uspdavg2d',windavg(1,1))
        call oracle_put('hpbl2d',pblheight(1,1))
        call oracle_close()
      enddo
    enddo
  enddo
end program
'''
for name,src in [('run_plumerise_wrfchem',wrf),('gsl_run_plumerise_frp',gsl)]:
    src=src.replace('character(64) :: case_name','character(64) :: case_name, state_mode')
    src=src.replace('  call oracle_root(root)','  call get_command_argument(2,state_mode)\n  call oracle_root(root)')
    src=src.replace('do pass=1,2','do pass=1,2\n          if(state_mode==\'carry\' .and. pass==1) cycle')
    src=src.replace('call zero_plumegen_coms; call set_grid',"if(state_mode/='carry') call zero_plumegen_coms\n          call set_grid")
    src=src.replace('call coms%set_to_zero()',"if(state_mode/='carry') call coms%set_to_zero()")
    src=src.replace('if(any(transfer(',"if(state_mode=='' .and. any(transfer(")
    (HERE/(name+'.F90')).write_text(src)

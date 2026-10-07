! WRF f52c197e: real generic column driver with three emitted rows.
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

  implicit none
  type(grid_config_rec_type) :: cfg
  real :: emis_ant(1,1,1,1)
  integer, parameter :: maxnz=120, nr=3
  real :: t_in(1,maxnz,1),p_in(1,maxnz,1),rho_in(1,maxnz,1)
  real :: u_in(1,maxnz,1),v_in(1,maxnz,1),w_in(1,maxnz,1)
  real :: z_in(1,maxnz+1,1),zw_in(1,maxnz+1,1),q_in(1,maxnz,1,1)
  real :: ebu(1,maxnz,1,nr),baseline(1,maxnz,1,nr),incoming(1,1,1,nr)
  real :: dz0,agl,temperature,pressure,humidity,terrain,psurf,speed,gradient
  real :: prop(4),size(4),windavg(1,1),pblheight(1,1),frp_values(11)
  integer :: nz,iz,family,fire,c,k,r,pass,errflg,mode
  character(1024) :: root,errmsg
  character(64) :: case_name, state_mode
  call get_command_argument(2,state_mode)
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
          if(state_mode=='carry' .and. pass==1) cycle
#ifndef PLUME_TRACE
          if(pass==2) exit
#endif
          if(state_mode/='carry') call zero_plumegen_coms
          call set_grid
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
          if(state_mode=='' .and. any(transfer(ebu,[0],maxnz*nr)/=transfer(baseline,[0],maxnz*nr))) error stop 'observer changed emissions'
        enddo
        write(case_name,'("nz",I2.2,"_s",I1,"_f",I2.2)') nz,family,fire

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

        call oracle_put('w',w); call oracle_put('t',t); call oracle_put('qv',qv)
        call oracle_put('qc',qc); call oracle_put('qh',qh); call oracle_put('qi',qi)
        call oracle_put('radius',radius)
#endif
        call oracle_put('mean_fct',prop); call oracle_put('firesize',size)
        call oracle_close()
      enddo
    enddo
  enddo
end program

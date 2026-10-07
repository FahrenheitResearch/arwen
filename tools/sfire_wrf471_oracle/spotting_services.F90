! Standalone driver records and WRF clock/grid services. No field math.
module module_domain
implicit none
type domain
 integer::id=1,itimestep=0,sr_x=2,sr_y=3,use_theta_m=0
 integer::nx=6,ny=4,nz=7,halo=4,domain_clock=0,alarms(3)=[1,2,3]
 logical::this_is_an_ideal_run=.true.,fuel_crosswalk=.false.
 real::dt=.125,rdx=.01,rdy=.01,p_top=50000.
 real,allocatable::u_2(:,:,:),v_2(:,:,:),w_2(:,:,:),p(:,:,:),pb(:,:,:),t_2(:,:,:)
 real,allocatable::phb(:,:,:),ph_2(:,:,:),moist(:,:,:,:),al(:,:,:),alb(:,:,:)
 real,allocatable::msftx(:,:),msfty(:,:),muts(:,:),znw(:),c1h(:),c2h(:),dnw(:),fnm(:),fnp(:)
 real,allocatable::burnt_area_dt(:,:),fmc_g(:,:),nfuel_cat(:,:),fgip(:,:),fire_area(:,:)
end type
contains
subroutine get_ijk_from_grid(grid,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,ips,ipe,jps,jpe,kps,kpe)
type(domain),intent(in)::grid
integer,intent(out)::ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,ips,ipe,jps,jpe,kps,kpe
ids=1;ide=grid%nx+1;jds=1;jde=grid%ny+1;kds=1;kde=grid%nz+1
ims=1-grid%halo;ime=grid%nx+grid%halo;jms=1-grid%halo;jme=grid%ny+grid%halo;kms=1;kme=grid%nz+1
ips=1;ipe=grid%nx;jps=1;jpe=grid%ny;kps=1;kpe=grid%nz+1
end subroutine
subroutine get_ijk_from_subgrid(grid,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,ips,ipe,jps,jpe,kps,kpe)
type(domain),intent(in)::grid
integer,intent(out)::ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,ips,ipe,jps,jpe,kps,kpe
ids=1;ide=(grid%nx+1)*grid%sr_x;jds=1;jde=(grid%ny+1)*grid%sr_y;kds=1;kde=1
ims=1;ime=grid%nx*grid%sr_x;jms=1;jme=grid%ny*grid%sr_y;kms=1;kme=1
ips=1;ipe=ime;jps=1;jpe=jme;kps=1;kpe=1
end subroutine
logical function is_alarm_tstep(clock,alarm)
integer,intent(in)::clock,alarm
is_alarm_tstep=alarm==1.and.mod(clock,4)==0
end function
subroutine domain_get_time_since_sim_start()
end subroutine
subroutine domain_clock_get()
end subroutine
end module
module module_configure
implicit none
type grid_config_rec_type
 integer::fs_array_maxsize=256,fs_firebrand_gen_levels=3,fs_firebrand_gen_lim=7,fs_firebrand_gen_dt=2
 integer::fs_firebrand_max_life_dt=200,fs_firebrand_gen_levrand_seed=1,fs_firebrand_gen_mom3d_dt=1
 real::fs_firebrand_gen_maxhgt=50.,fs_firebrand_land_hgt=.15
 real::fs_firebrand_gen_prop_diam=10.,fs_firebrand_gen_prop_effd=10.
 real::fs_firebrand_gen_prop_temp=900.,fs_firebrand_gen_prop_tvel=0.
 real::fs_firebrand_dens=513000.,fs_firebrand_dens_char=299000.
 logical::fs_firebrand_gen_levrand=.false.,trackember=.false.
end type
end module
module module_symbols_util
implicit none
type WRFU_TimeInterval
 integer::unused=0
end type
contains
subroutine WRFU_TimeIntervalGet()
end subroutine
subroutine WRFU_TimeIntervalSet()
end subroutine
end module
module module_domain_type
implicit none
integer,parameter::HISTORY_ALARM=1,restart_alarm=2,AUXHIST23_ALARM=3
end module
module module_state_description
implicit none
integer,parameter::p_qv=1,num_moist=3,param_first_scalar=1
end module
module module_utility
implicit none
type WRFU_Alarm
 integer::unused=0
end type
end module
subroutine wrf_debug(level,message)
integer,intent(in)::level
character(len=*),intent(in)::message
end subroutine
subroutine wrf_error_fatal(message)
character(len=*),intent(in)::message
print *,message
error stop 'native firebrand driver fatal condition'
end subroutine

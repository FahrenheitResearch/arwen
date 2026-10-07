program run_moisture_driver
use moisture_driver_control
use module_fr_fire_phys
use oracle_io
implicit none
integer,parameter::mx=6,my=4,nc=5,rx=2,ry=3,fx=mx*rx,fy=my*ry,steps=10
type(clock_record)::clock
type(settings_record)::settings
real::rainc(mx,my),rainnc(mx,my),t2(mx,my),q2(mx,my),psfc(mx,my)
real::rain_old(mx,my),t2_old(mx,my),q2_old(mx,my),psfc_old(mx,my),rh(mx,my)
real::fmcgc(mx,nc,my),fmep(mx,2,my),equi(mx,nc,my),lag(mx,nc,my)
real::classes(0:mx+1,nc,0:my+1),cat(0:fx+1,0:fy+1),fine(0:fx+1,0:fy+1)
real::dt_moisture
integer::arm,c,s,i,j,k
logical::run_advance,run_fuel,fire_run,initialize,initialized
character(len=1024)::root
character(len=64)::name,label
call oracle_root(root)
fire_print_msg=0;fire_print_file=0;fire_fmc_read=0
call init_fuel_cats(.true.)
do arm=0,1
  label='original'
  if(arm==1)label='corrected'
  do c=1,12
    settings=settings_record(.true.,.true.,.false.,0,600.)
    select case(c)
      case(2);settings%fmoist_dt=275.5
      case(3);settings%fmoist_dt=.1
      case(4);settings%fmoist_freq=1
      case(5);settings%fmoist_freq=3
      case(6);settings%fmoist_freq=5
      case(7);settings%fmoist_only=.true.;settings%fmoist_freq=3
      case(8);settings%fmoist_interp=.false.;settings%fmoist_freq=3
      case(9);settings%fmoist_run=.false.
      case(10);settings%fmoist_run=.false.;settings%fmoist_only=.true.
      case(11);settings%fmoist_interp=.false.;settings%fmoist_only=.true.
      case(12);settings%fmoist_freq=3
    end select
    moisture_classes=5
    if(c==12)moisture_classes=3
    clock=clock_record(0,137.25,0.,0.)
    rain_old=0.;t2_old=0.;q2_old=0.;psfc_old=0.;rh=0.;equi=0.;lag=0.
    do j=1,my
      do i=1,mx
        fmep(i,1,j)=.001*real(mod(i,3)-1)
        fmep(i,2,j)=.002*real(mod(j,3)-1)
        do k=1,nc
          fmcgc(i,k,j)=.03*real(k)+.001*real(i*i+j)
        enddo
      enddo
    enddo
    do j=0,fy+1
      do i=0,fx+1
        cat(i,j)=real(1+mod(i+3*j,14))
      enddo
    enddo
    fine=.08;initialized=.false.;dt_moisture=-999.
    ! The ifun 1/2 initialization pass initializes clocks only. The model
    ! call is inside ifun 3 and therefore cannot execute during this pass.
    call decide(1,2)
    do s=1,steps
      clock%itimestep=s
      do j=1,my
        do i=1,mx
          rainc(i,j)=.013*real(s*mod(i+j,3))
          rainnc(i,j)=.041*real(s*mod(i+2*j,4))
          t2(i,j)=275.+real(i*3+j)+.125*real(s*mod(i,3))
          q2(i,j)=.001+.0005*real(mod(i+2*j,12))+.00001*real(s)
          psfc(i,j)=80000.+100.*real(i*i+7*j+s)
        enddo
      enddo
      write(name,'(a,a,a,i0,a,i0)')'moisture_driver/',trim(label),'_',c,'_',s
      call oracle_open(trim(name))
      call oracle_put('case',c);call oracle_put('step',s)
      call oracle_put('atmosphere_dt',clock%dt)
      call oracle_put('fmoist_run',merge(1,0,settings%fmoist_run))
      call oracle_put('fmoist_interp',merge(1,0,settings%fmoist_interp))
      call oracle_put('fmoist_only',merge(1,0,settings%fmoist_only))
      call oracle_put('fmoist_freq',settings%fmoist_freq);call oracle_put('fmoist_dt',settings%fmoist_dt)
      call oracle_put('active_classes',moisture_classes)
      call oracle_put('rainc',rainc);call oracle_put('rainnc',rainnc)
      call oracle_put('t2',t2);call oracle_put('q2',q2);call oracle_put('psfc',psfc)
      call oracle_put('nfuel_cat',cat)
      call save_state('_in')
      call decide(3,6)
      initialize=.false.
      if(run_advance)then
        initialize=s==1
        if(arm==1)initialize=.not.initialized
        call advance_moisture(initialize,1,mx,1,my,1,mx,1,my,nc,dt_moisture,999999., &
          rainc,rainnc,t2,q2,psfc,rain_old,t2_old,q2_old,psfc_old,rh,fmcgc,fmep,equi,lag)
        initialized=.true.
      endif
      if(run_fuel.and.fire_run)then
        classes=0.;classes(1:mx,:,1:my)=fmcgc
        call fuel_moisture(0,nc,1,mx,1,my,0,mx+1,0,my+1,1,mx,1,my,1,mx,1,my, &
          1,fx,1,fy,0,fx+1,0,fy+1,1,fx,1,fy,rx,ry,cat,classes,fine)
      endif
      call oracle_put('run_advance',merge(1,0,run_advance))
      call oracle_put('run_fuel',merge(1,0,run_fuel.and.fire_run))
      call oracle_put('initialize',merge(1,0,initialize))
      call oracle_put('initialized',merge(1,0,initialized))
      call oracle_put('dt_moisture',dt_moisture)
      call save_state('_out')
      call oracle_close()
    enddo
  enddo
enddo
contains
subroutine decide(first,last)
integer,intent(in)::first,last
if(arm==0)then
 call decide_original(clock,settings,first,last,dt_moisture,run_advance,run_fuel,fire_run)
else
 call decide_clock_corrected(clock,settings,first,last,dt_moisture,run_advance,run_fuel,fire_run)
endif
end subroutine
subroutine save_state(suffix)
character(len=*),intent(in)::suffix
call oracle_put('rain_old'//suffix,rain_old);call oracle_put('t2_old'//suffix,t2_old)
call oracle_put('q2_old'//suffix,q2_old);call oracle_put('psfc_old'//suffix,psfc_old)
call oracle_put('rh_fire'//suffix,rh);call oracle_put('fmc_gc'//suffix,fmcgc)
call oracle_put('fmep'//suffix,fmep);call oracle_put('fmc_equi'//suffix,equi);call oracle_put('fmc_lag'//suffix,lag)
call oracle_put('fmc_g'//suffix,fine)
call oracle_put('lasttime'//suffix,clock%fmoist_lasttime)
call oracle_put('nexttime'//suffix,clock%fmoist_nexttime)
end subroutine
end program

program run_gocart_settling
  use module_configure
  use module_state_description
  use module_data_gocart_dust, only: reff_dust, den_dust, dyn_visc
  use module_data_gocart_seas, only: reff_seas, den_seas
  use module_model_constants, only: g
  use module_gocart_settling
  use oracle_io
  implicit none
  integer, parameter :: nx=3, ny=2, nz=48
  type(grid_config_rec_type) :: cfg
  real :: t(nx,nz,ny), p(nx,nz,ny), pw(nx,nz,ny), dz(nx,nz,ny), rho(nx,nz,ny)
  real :: moist(nx,nz,ny,num_moist), chem(nx,nz,ny,num_chem)
  real :: accum(nx,ny,5), vel(nx,ny,5), unused(nx,ny,5)
  real :: dt, rh, z, requested_rh(nx,ny)
  real, parameter :: dts(3)=(/7.5,36.0,60.0/), rhs(6)=(/0.05,0.30,0.80,0.95,0.99,0.10/)
  integer :: i,j,k,r,ic,it,step,arm,ns(nx,ny,9), mode
  real(8) :: rr(9),den(9),growth,v,dtmax
  character(len=1024) :: root
  character(len=80) :: name
  call oracle_root(root)
  rr(1:5)=reff_dust; rr(6:9)=reff_seas
  den(1:5)=den_dust; den(6:9)=den_seas
  unused=0.
  do arm=1,2
    cfg%chem_opt=300
    if (arm==2) cfg%chem_opt=401
    do mode=0,1
      do it=1,3
        dt=dts(it)
        do j=1,ny
          do i=1,nx
            ic=i+(j-1)*nx
            rh=rhs(ic)
            requested_rh(i,j)=rh
            do k=1,nz
              z=real(k-1)*160.
              t(i,k,j)=max(235.,294.-0.006*z+0.2*real(ic))
              p(i,k,j)=100800.*exp(-z/8200.)
              pw(i,k,j)=p(i,k,j)+300.
              dz(i,k,j)=40.+4.*real(k)+real(ic)
              if (mode==1 .and. k==1) dz(i,k,j)=0.05
              rho(i,k,j)=p(i,k,j)/(287.*t(i,k,j))
              moist(i,k,j,:)=0.
              moist(i,k,j,p_qv)=rh*(3.80*exp(17.27*(t(i,k,j)-273.)/(t(i,k,j)-36.))/(.01*p(i,k,j)))
              do r=1,num_chem
                chem(i,k,j,r)=real(r)*0.7*exp(-z/4000.)+0.05*real(k)
              enddo
              if (mode==1 .and. mod(k,7)==0) chem(i,k,j,p_dust_1:p_seas_4)=-0.25
            enddo
          enddo
        enddo
        accum=2.e-10; vel=-99.
        do r=1,9
          growth=1.
          if (r>5) growth=3.
          v=4.0/9.0*g*den(r)*(growth*rr(r))**2/dyn_visc
          do j=1,ny
            do i=1,nx
              dtmax=minval(real(dz(i,:,j),8))/v
              ns(i,j,r)=min(12,max(1,int(int(dt)/dtmax)))
            enddo
          enddo
        enddo
        do step=1,3
          write(name,'("arm",I0,"_mode",I0,"_dt",I0,"_step",I0)') cfg%chem_opt,mode,it,step
          call oracle_open(trim(name))
          call oracle_put('chem_in',chem)
          call oracle_put('accum_in',accum)
          call oracle_put('temp',t); call oracle_put('pressure',p)
          call oracle_put('dz',dz); call oracle_put('rho',rho)
          call oracle_put('qv',moist(:,:,:,p_qv))
          call oracle_put('rh_requested',requested_rh)
          call oracle_put('dt',dt); call oracle_put('gravity',g)
          call oracle_put('dyn_visc',dyn_visc)
          call oracle_put('radius_bits',reshape(transfer(rr,(/0/)),(/2,9/)))
          call oracle_put('density_bits',reshape(transfer(den,(/0/)),(/2,9/)))
          call oracle_put('nsteps',ns)
          call oracle_put('chem_opt',cfg%chem_opt)
          call gocart_settling_driver(dt,cfg,t,moist,chem,rho,dz,pw,p,unused,unused, &
            6000.,g,accum(:,:,1),accum(:,:,2),accum(:,:,3),accum(:,:,4),accum(:,:,5), &
            vel(:,:,1),vel(:,:,2),vel(:,:,3),vel(:,:,4),vel(:,:,5),0, &
            1,nx,1,ny,1,nz+1,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
          call oracle_put('chem_out',chem)
          call oracle_put('accum_out',accum); call oracle_put('velocity',vel)
          call oracle_close()
        enddo
      enddo
    enddo
  enddo
end program run_gocart_settling

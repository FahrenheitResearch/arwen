program run_gocart_optics
 use optics_module
 use module_configure
 use module_data_sorgam
 use module_data_gocart_seas, only: ra,rb
 use module_data_gocart_dust, only: ndust,ra_dust,rb_dust
 use module_aer_opt_out
 use rrtmg_module
 use oracle_io
 implicit none
 integer, parameter :: nz=50
 integer, parameter :: nc=3
 real :: chem(1,nz,1,20),alt(1,nz,1),rh(1,nz,1),dz(1,nz,1)
 real :: radius(1,nz,1,9),core(1,nz,1,9),number(1,nz,1,9)
 complex :: swri(1,nz,1,9,4),swcore(1,nz,1,9,4),swshell(1,nz,1,9,4)
 complex :: lwri(1,nz,1,9,16),lwcore(1,nz,1,9,16),lwshell(1,nz,1,9,16)
 real :: rcol(9,nz),ncol(9,nz),swsize(4,nz),ext(4,nz),ssa(4,nz),asm(4,nz),tau(4,nz),bsc(4,nz)
 complex :: srcol(9,nz,4),lrcol(9,nz,16)
 real :: lext(16,nz),ltau(16,nz),mom(4,nz,6)
 real :: ec(1,nz,1,5),bc(1,nz,1,3),asym(1,nz,1,3)
 real :: btau(1,nz,14),bssa(1,nz,14),basm(1,nz,14),aod(1)
 real :: chem_all(nc,nz,1,20),alt_all(nc,nz,1),rh_all(nc,nz,1),dz_all(nc,nz,1)
 real :: radius_all(nc,nz,1,9),number_all(nc,nz,1,9),sr_all(9,nz,4,nc),si_all(9,nz,4,nc)
 real :: lr_all(9,nz,16,nc),li_all(9,nz,16,nc),tau_all(4,nz,nc),ext_all(4,nz,nc)
 real :: ssa_all(4,nz,nc),asm_all(4,nz,nc),bs_all(4,nz,nc),lt_all(16,nz,nc),le_all(16,nz,nc)
 real :: mom_all(4,nz,6,nc),ec_all(nc,nz,1),aod_all(nc),bt_all(nc,nz,14),bsw_all(nc,nz,14),ba_all(nc,nz,14)
 real :: mass_i(9),mass_j(9),mass_c(9),nn(9),diam(9),dtemp
 integer :: c,k,n,ix,jy,ic
 real :: bma,bpa,xr,rs(200),rmin,rmax,pie
 integer,parameter :: nsiz=200
 integer :: nrefr,nrefi
 character(1024) :: root
 character(24) :: cases(33)
 cases(1:13)=[character(24) :: 'dust_light','dust_haboob','salt_030','salt_080','salt_095','salt_099', &
 'carbon_sulfate','all_species','nearly_clean','threshold','lite_zero','lite_negative_zero','negative_zero']
 do n=14,32
  write(cases(n),'(a,i2.2)')'isolated_row_',n-13
 enddo
 cases(33)='dry_mix'
 call oracle_root(root)
 if(trim(root)=='--badindex')then
  chem=0.;chem(:,:,:,p_msa)=1.e-4;alt=1.;rh=.05;dz=100.
  call prepare_gocart(chem,alt,rh,core,radius,number,swri,swcore,swshell,lwri,lwcore,lwshell,nz)
  rcol=transpose(radius(1,:,1,:));ncol=transpose(number(1,:,1,:))
  do n=1,4
   srcol(:,:,n)=transpose(swri(1,:,1,:,n))
  enddo
  do n=1,16
   lrcol(:,:,n)=transpose(lwri(1,:,1,:,n))
  enddo
  call evaluate_mie(ncol,rcol,srcol,lrcol,dz(1,:,1),swsize,ext,ssa,asm,tau,lext,ltau, &
   mom(:,:,1),mom(:,:,2),mom(:,:,3),mom(:,:,4),mom(:,:,5),mom(:,:,6),bsc,nz)
  error stop 'invalid index did not trigger the WRF fatal path'
 endif
 if(trim(root)=='--negative')then
  tau=0.3;tau(2,:)=-0.3;ssa=.9;asm=.7
  call convert_bands(reshape(tau(1,:),[1,nz,1]),reshape(tau(2,:),[1,nz,1]), &
   reshape(tau(3,:),[1,nz,1]),reshape(tau(4,:),[1,nz,1]), &
   reshape(ssa(1,:),[1,nz,1]),reshape(ssa(2,:),[1,nz,1]),reshape(ssa(3,:),[1,nz,1]),reshape(ssa(4,:),[1,nz,1]), &
   reshape(asm(1,:),[1,nz,1]),reshape(asm(2,:),[1,nz,1]),reshape(asm(3,:),[1,nz,1]),reshape(asm(4,:),[1,nz,1]), &
   btau,bssa,basm,nz)
  error stop 'negative column did not trigger the WRF fatal path'
 endif
 do c=1,size(cases)
 do ic=1,nc
  chem=0.
  do k=1,nz
   alt(1,k,1)=0.85*exp(real(k-1)*0.055)
   dz(1,k,1)=40.+real(k-1)*12.
   rh(1,k,1)=0.3+0.5*real(mod(k,7))/6.
   select case(c)
   case(1,2)
    do n=11,15
     chem(1,k,1,n)=real(n-10)*0.3*exp(-real(k-1)/6.)
     if(c==2)chem(1,k,1,n)=chem(1,k,1,n)*6000.
    enddo
   case(3:6)
    do n=16,19
     chem(1,k,1,n)=real(n-15)*2.*exp(-real(k-1)/5.)
    enddo
    if(c==3)rh(1,k,1)=0.3
    if(c==4)rh(1,k,1)=0.8
    if(c==5)rh(1,k,1)=0.95
    if(c==6)rh(1,k,1)=0.99
   case(7)
    chem(1,k,1,3)=0.001*exp(-real(k-1)/12.)
    chem(1,k,1,7:10)=3.*exp(-real(k-1)/10.)
   case(8,11,12,33)
    chem(1,k,1,2:20)=0.2*exp(-real(k-1)/9.)
    chem(1,k,1,3)=chem(1,k,1,3)*0.005
    chem(1,k,1,5)=chem(1,k,1,5)*0.001
    if(c==11.or.c==12)then
     chem(1,k,1,4:6)=0.;chem(1,k,1,20)=0.
    endif
    if(c==12)then
     chem(1,k,1,4:6)=-0.;chem(1,k,1,20)=-0.
    endif
    if(c==33)rh(1,k,1)=0.
   case(9)
    chem=1.e-12
   case(10)
    chem=0.
   case(13)
    chem=-0.
   case(14:32)
    chem(1,k,1,c-12)=0.1*exp(-real(k-1)/8.)
    if(c==15.or.c==17)chem(1,k,1,c-12)=chem(1,k,1,c-12)*.001
   end select
  enddo
  ! Three independent columns per case, with modest density and mass changes.
  alt=alt*(1.+real(ic-1)*0.02)
  chem=chem*(1.+real(ic-1)*0.05)
  ! prep reaches kts..kte. The oracle supplies all mass levels explicitly.
  call prepare_gocart(chem,alt,rh,core,radius,number,swri,swcore,swshell,lwri,lwcore,lwshell,nz)
  rcol=transpose(radius(1,:,1,:));ncol=transpose(number(1,:,1,:))
  do n=1,4
   srcol(:,:,n)=transpose(swri(1,:,1,:,n))
  enddo
  do n=1,16
   lrcol(:,:,n)=transpose(lwri(1,:,1,:,n))
  enddo
  call evaluate_mie(ncol,rcol,srcol,lrcol,dz(1,:,1),swsize,ext,ssa,asm,tau,lext,ltau, &
   mom(:,:,1),mom(:,:,2),mom(:,:,3),mom(:,:,4),mom(:,:,5),mom(:,:,6),bsc,nz)
  call clamp_outputs(tau,ext,ssa,asm,bsc,ltau,lext,nz)
  call aer_opt_out(dz,ec,bc,asym,reshape(tau(1,:),[1,nz,1]),reshape(tau(2,:),[1,nz,1]), &
   reshape(tau(3,:),[1,nz,1]),reshape(tau(4,:),[1,nz,1]), &
   reshape(asm(1,:),[1,nz,1]),reshape(asm(2,:),[1,nz,1]),reshape(asm(3,:),[1,nz,1]),reshape(asm(4,:),[1,nz,1]), &
   reshape(ssa(1,:),[1,nz,1]),reshape(ssa(2,:),[1,nz,1]),reshape(ssa(3,:),[1,nz,1]),reshape(ssa(4,:),[1,nz,1]), &
   5,3,3,1,1,1,1,1,nz,1,1,1,1,1,nz,1,1,1,1,1,nz)
  call convert_bands(reshape(tau(1,:),[1,nz,1]),reshape(tau(2,:),[1,nz,1]), &
   reshape(tau(3,:),[1,nz,1]),reshape(tau(4,:),[1,nz,1]), &
   reshape(ssa(1,:),[1,nz,1]),reshape(ssa(2,:),[1,nz,1]),reshape(ssa(3,:),[1,nz,1]),reshape(ssa(4,:),[1,nz,1]), &
   reshape(asm(1,:),[1,nz,1]),reshape(asm(2,:),[1,nz,1]),reshape(asm(3,:),[1,nz,1]),reshape(asm(4,:),[1,nz,1]), &
   btau,bssa,basm,nz)
  aod=0.
  do k=1,nz
   aod(1)=aod(1)+ec(1,k,1,2)*dz(1,k,1)*1.e-3
  enddo
  chem_all(ic,:,:,:)=chem(1,:,:,:);alt_all(ic,:,:)=alt(1,:,:);rh_all(ic,:,:)=rh(1,:,:);dz_all(ic,:,:)=dz(1,:,:)
  radius_all(ic,:,:,:)=radius(1,:,:,:);number_all(ic,:,:,:)=number(1,:,:,:)
  sr_all(:,:,:,ic)=real(srcol);si_all(:,:,:,ic)=aimag(srcol)
  lr_all(:,:,:,ic)=real(lrcol);li_all(:,:,:,ic)=aimag(lrcol)
  tau_all(:,:,ic)=tau;ext_all(:,:,ic)=ext;ssa_all(:,:,ic)=ssa;asm_all(:,:,ic)=asm;bs_all(:,:,ic)=bsc
  lt_all(:,:,ic)=ltau;le_all(:,:,ic)=lext;mom_all(:,:,:,ic)=mom
  ec_all(ic,:,:)=ec(1,:,:,2);aod_all(ic)=aod(1)
  bt_all(ic,:,:)=btau(1,:,:);bsw_all(ic,:,:)=bssa(1,:,:);ba_all(ic,:,:)=basm(1,:,:)
 enddo ! independent columns
  call oracle_open(trim(cases(c)))
  call oracle_put('chem',chem_all);call oracle_put('alt',alt_all);call oracle_put('relhum',rh_all);call oracle_put('dz8w',dz_all)
  call oracle_put('radius',radius_all);call oracle_put('number',number_all)
  call oracle_put('swri_real',sr_all);call oracle_put('swri_imag',si_all)
  call oracle_put('lwri_real',lr_all);call oracle_put('lwri_imag',li_all)
  call oracle_put('tauaer',tau_all);call oracle_put('extaer',ext_all);call oracle_put('waer',ssa_all)
  call oracle_put('gaer',asm_all);call oracle_put('bscoef',bs_all)
  call oracle_put('tauaerlw',lt_all);call oracle_put('extaerlw',le_all);call oracle_put('moments',mom_all)
  call oracle_put('EXTCOF55',ec_all);call oracle_put('AOD5502D',aod_all)
  call oracle_put('band_tau',bt_all);call oracle_put('band_ssa',bsw_all);call oracle_put('band_asm',ba_all)
  call oracle_close()
 enddo
 call oracle_open('tables')
 call oracle_put('extpsw',extpsw);call oracle_put('abspsw',abspsw);call oracle_put('ascatpsw',ascatpsw)
 call oracle_put('asmpsw',asmpsw);call oracle_put('sbackpsw',sbackpsw)
 call oracle_put('pmom2psw',pmom2psw);call oracle_put('pmom3psw',pmom3psw);call oracle_put('pmom4psw',pmom4psw)
 call oracle_put('pmom5psw',pmom5psw);call oracle_put('pmom6psw',pmom6psw);call oracle_put('pmom7psw',pmom7psw)
 call oracle_put('extplw',extplw);call oracle_put('absplw',absplw);call oracle_put('ascatplw',ascatplw)
 call oracle_put('asmplw',asmplw)
 call oracle_put('refrtabsw',refrtabsw);call oracle_put('refitabsw',refitabsw)
 call oracle_put('refrtablw',refrtablw);call oracle_put('refitablw',refitablw)
 call oracle_put('wavmidsw',wavmidsw);call oracle_put('wavmidlw',wavmidlw)
 call oracle_put('rmmin',rmmin);call oracle_put('rmmax',rmmax)
 call oracle_put('xrmin',log(rmmin));call oracle_put('xrmax',log(rmmax))
 call oracle_put('pie',4.*atan(1.))
 call sect02(dginin*1.e6,sginin,1.8,2,1.,9,0.0390625,20.,nn,mass_i)
 call sect02(dginia*1.e6,sginia,1.8,2,1.,9,0.0390625,20.,nn,mass_j)
 call sect02(dginic*1.e6,sginic,1.8,2,1.,9,0.0390625,20.,nn,mass_c)
 call oracle_put('mass_i',mass_i);call oracle_put('mass_j',mass_j);call oracle_put('mass_c',mass_c)
 dtemp=0.0390625
 do n=1,9
  diam(n)=(dtemp+dtemp*2.)/2.*1.e-4
  dtemp=dtemp*2.
 enddo
 call oracle_put('diam_cm',diam)
 rmin=rmmin;rmax=rmmax;pie=4.*atan(1.)
 include 'fit_limits.inc'
 do n=1,200
  include 'fit_sizes.inc'
 enddo
 call oracle_put('rs',rs);call oracle_put('bma',bma);call oracle_put('bpa',bpa)
 ! Every grid uses both full extents; the final LW grid controls the SAVE values.
 nrefr=prefr;nrefi=prefi
 call oracle_put('nrefr',nrefr);call oracle_put('nrefi',nrefi)
 call dump_fractions()
 call oracle_put('refrwsw',refrwsw);call oracle_put('refiwsw',refiwsw)
 call oracle_put('refrwlw',refrwlw);call oracle_put('refiwlw',refiwlw)
 call oracle_put('refrsw_sulf',refrsw_sulf);call oracle_put('refisw_sulf',refisw_sulf)
 call oracle_put('refrlw_sulf',refrlw_sulf);call oracle_put('refilw_sulf',refilw_sulf)
 call oracle_put('refrsw_dust',refrsw_dust);call oracle_put('refisw_dust',refisw_dust)
 call oracle_put('refrlw_dust',refrlw_dust);call oracle_put('refilw_dust',refilw_dust)
 call oracle_put('refrsw_oc',refrsw_oc);call oracle_put('refisw_oc',refisw_oc)
 call oracle_put('refrlw_oc',refrlw_oc);call oracle_put('refilw_oc',refilw_oc)
 call oracle_put('refrsw_seas',refrsw_seas);call oracle_put('refisw_seas',refisw_seas)
 call oracle_put('refrlw_seas',refrlw_seas);call oracle_put('refilw_seas',refilw_seas)
 call oracle_close()
 ! Isolated RRTMG branch probes. k=1 is at threshold, k=2 has one endpoint
 ! below it, k=3 passes, and k=4..nz exercise both extrapolation limits.
 tau=0.3;ssa=0.9;asm=0.7
 tau(1,1)=1.e-9;tau(4,2)=1.e-10
 ssa(2,3)=0.1;ssa(3,3)=1.2;asm(2,3)=0.1;asm(3,3)=1.2
 call convert_bands(reshape(tau(1,:),[1,nz,1]),reshape(tau(2,:),[1,nz,1]), &
  reshape(tau(3,:),[1,nz,1]),reshape(tau(4,:),[1,nz,1]), &
  reshape(ssa(1,:),[1,nz,1]),reshape(ssa(2,:),[1,nz,1]),reshape(ssa(3,:),[1,nz,1]),reshape(ssa(4,:),[1,nz,1]), &
  reshape(asm(1,:),[1,nz,1]),reshape(asm(2,:),[1,nz,1]),reshape(asm(3,:),[1,nz,1]),reshape(asm(4,:),[1,nz,1]), &
  btau,bssa,basm,nz)
 call oracle_open('rrtmg_edges')
 call oracle_put('tauaer',tau);call oracle_put('waer',ssa);call oracle_put('gaer',asm)
 call oracle_put('band_tau',btau);call oracle_put('band_ssa',bssa);call oracle_put('band_asm',basm)
 call oracle_close()
contains
 subroutine dump_fractions()
  integer,parameter :: nbin_o=9
  integer :: m,n
  real :: dlo_um,dhi_um,dlo,dhi,xlo,xhi,dxbin,dlo_sectm(9),dhi_sectm(9)
  real :: seasfrc_goc9bin(4,9),dustfrc_goc9bin(5,9)
  real*8 :: dlogoc,dhigoc
  dlo_um=0.0390625;dhi_um=20.
  include 'section_fractions.inc'
  call oracle_put('seasfrc_goc9bin',seasfrc_goc9bin)
  call oracle_put('dustfrc_goc9bin',dustfrc_goc9bin)
  call oracle_put('dlo_sectm',dlo_sectm);call oracle_put('dhi_sectm',dhi_sectm)
 end subroutine
end program

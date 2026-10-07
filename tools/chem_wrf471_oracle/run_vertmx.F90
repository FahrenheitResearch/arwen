program run_vertmx
 use oracle_vertmx
 use module_model_constants, only: mwdry
 use oracle_io
 implicit none
 integer, parameter :: param_first_scalar=2, num_chem=3,numgas=2,p_o3=2
 integer, parameter :: RADM2SORG_AQ=10,RADM2SORG_AQCHEM=11,RACMSORG_AQ=12,RACMSORG_AQCHEM_KPP=13,RACM_ESRLSORG_AQCHEM_KPP=14,CBMZ_MOSAIC_4BIN_AQ=15,CBMZ_MOSAIC_8BIN_AQ=16,CBMZSORG_AQ=17,CBMZ_MOSAIC_DMS_4BIN=18,CBMZ_MOSAIC_DMS_8BIN=19,CBMZ_MOSAIC_DMS_4BIN_AQ=20,CBMZ_MOSAIC_DMS_8BIN_AQ=21,CRI_MOSAIC_8BIN_AQ_KPP=22,CRI_MOSAIC_4BIN_AQ_KPP=23,MOZART_MOSAIC_4BIN_AQ_KPP=24,RACM_SOA_VBS_AQCHEM_KPP=25,SAPRC99_MOSAIC_8BIN_VBS2_AQ_KPP=26,CB05_SORG_AQ_KPP=27,CB05_SORG_VBS_AQ_KPP=28
 real, parameter :: epsilc=1.e-16
 type config_type
 integer :: chem_opt=0
 logical :: mynn_chem_vertmx=.false.
 end type
 type(config_type) :: config_flags
 integer :: i=1,j=1,k,nv,kts=1,kte,nz,nn,family,step,sf_urban_physics,p_e_co,p_e_pm25i,p_e_pm25j,p_e_pm_25,p_ebu_in_co
 logical :: is_CAMMGMP_used=.false.,vertMixAero(3)=.true.,is_aerosol(3)=[.false.,.false.,.true.]
 real :: dtstep,old,new,fac,ddmassn(3),dep_vel_o3(1,1),ddvel(1,1,3),emis_ant(1,1,1,5),ebu_in(1,1,1,2),accum(3)
 real, allocatable :: chem(:,:,:,:),initial(:,:,:,:),z_at_w(:,:,:),z(:,:,:),alt(:,:,:),dz8w(:,:,:),exch_h(:,:,:),pblst(:),dryrho_1d(:),zzfull(:),zz(:),ekmfull(:)
 character(1024) :: out
 character(80) :: name
 call oracle_root(out)
 do nn=1,2
 nz=49+10*(nn-1); kte=nz
 allocate(chem(1,nz,1,3),initial(1,nz,1,3),z_at_w(1,nz+1,1),z(1,nz,1),alt(1,nz,1),dz8w(1,nz,1),exch_h(1,nz,1),pblst(nz),dryrho_1d(nz),zzfull(nz+1),zz(nz),ekmfull(nz+1))
 do family=1,18
 dtstep=18.; if(mod(family,3)==1)dtstep=36.; if(mod(family,3)==2)dtstep=60.
 ddvel=0.; if(mod(family,4)>0)ddvel=.001
 if(mod(family,4)==2)ddvel=.01
 if(mod(family,4)==3)ddvel=.05
 p_e_co=2;p_e_pm25i=3;p_e_pm25j=4;p_e_pm_25=5;p_ebu_in_co=2
 sf_urban_physics=0; emis_ant=0.;ebu_in=0.
 select case(family)
 case(5);emis_ant(1,1,1,2)=1.
 case(6);emis_ant(1,1,1,2)=200.
 case(7);emis_ant(1,1,1,2)=200.00002
 case(8);emis_ant(1,1,1,3)=8.19e-4*200
 case(9);emis_ant(1,1,1,3)=.1;emis_ant(1,1,1,4)=.1
 case(10);ebu_in=1.
 case(11);emis_ant=300.;sf_urban_physics=1
 case(12);emis_ant=300.;p_e_co=1;p_ebu_in_co=1
 case(13);emis_ant=300.;p_e_pm25i=2;p_e_pm_25=2
 case(14);emis_ant(1,1,1,5)=.2
 case(15);emis_ant=300.;ebu_in=1.;sf_urban_physics=1
 end select
 do k=1,nz+1
 z_at_w(1,k,1)=1000.+real(k-1)*60.+real(k-1)**2*.5
 enddo
 do k=1,nz
 z(1,k,1)=.5*(z_at_w(1,k,1)+z_at_w(1,k+1,1))
 dz8w(1,k,1)=z_at_w(1,k+1,1)-z_at_w(1,k,1)
 alt(1,k,1)=.8+real(k)*.025
 exch_h(1,k,1)=0.
 if(mod(family,4)==1)exch_h(1,k,1)=1.e-5
 if(mod(family,4)==2)exch_h(1,k,1)=100.
 if(mod(family,4)==3 .and. k<12)exch_h(1,k,1)=20.
 chem(1,k,1,:)=.01+real(mod(k,5))*10.
 if(family>=16)chem(1,k,1,:)=epsilc*real(mod(k,4))
 enddo
 initial=chem;accum=0.
 do step=1,5
 write(name,'("family",I0,"_nz",I0,"_step",I0)')family,nz,step
 call oracle_open(trim(name))
 call oracle_put('input',chem(1,:,1,2:3)); call oracle_put('alt',alt(1,:,1));call oracle_put('z_at_w',z_at_w(1,:,1));call oracle_put('z',z(1,:,1));call oracle_put('dz8w',dz8w(1,:,1));call oracle_put('exch_h',exch_h(1,:,1))
 call oracle_put('dt',dtstep);call oracle_put('vd',ddvel(1,1,2:3));call oracle_put('anth',emis_ant(1,1,1,:));call oracle_put('fire',ebu_in(1,1,1,2));call oracle_put('urban',sf_urban_physics)
 call oracle_put('indices',[p_e_co,p_e_pm25i,p_e_pm_25,p_ebu_in_co]);call oracle_put('accum_in',accum(2:3))
 ! Verbatim dry_dep_driver.F:675-805. No edits inside the block.
 ! Enclosing tile loops are replaced by i=j=1 and the case/step loops.
 ! Declarations and branch constants are supplied above; num_chem=3 includes
 ! one dummy, one gas, one aerosol. chem_opt=0 selects CASE DEFAULT.
      pblst=0.
      ddmassn(:) = 0.0
!
!
!-- start with vertical mixing
!
      do k=kts,kte+1
         zzfull(k)=z_at_w(i,k,j)-z_at_w(i,kts,j)
      enddo
      do k=kts,kte
         ekmfull(k)=max(1.e-6,exch_h(i,k,j))
      enddo
      ekmfull(kts)=0.
      ekmfull(kte+1)=0.

!!$! UNCOMMENT THIS AND FINE TUNE LEVELS TO YOUR DOMAIN IF YOU WANT TO
!!$! FORCE MIXING ESPECIALLY OVER URBAN AREAS TO A CERTAIN DEPTH:
!!$!
!!$! --- Mix the emissions up several layers in urban areas if no urban surface physics
!!$!     if e_co > 0., the grid cell should not be over water
!!$!     if e_co > 200, the grid cell should be over a large urban region
!!$!
! Do NOT increase mixing at surface/kts, where exch==0
! this code is wrong - doesn't work if e_co is == param_first_scalar
! (like it happened to be the case for MOZCART)
!     if (p_e_co > param_first_scalar )then
     if (p_e_co >= param_first_scalar )then
       if (sf_urban_physics .eq. 0 ) then
         if (emis_ant(i,kts,j,p_e_co) .gt. 0) then
          ekmfull(kts+1:kts+10) = max(ekmfull(kts+1:kts+10),1.)
         endif
         if (emis_ant(i,kts,j,p_e_co) .gt. 200) then
          ekmfull(kts+1:kte/2) = max(ekmfull(kts+1:kte/2),2.)
         endif
         if (p_e_pm25i > param_first_scalar )then
          if (emis_ant(i,kts,j,p_e_pm25i)+ emis_ant(i,kts,j,p_e_pm25j) .GT. 8.19e-4*200) then
           ekmfull(kts+1:kte/2) = max(ekmfull(kts+1:kte/2),2.)
          endif
         endif
         if (p_e_pm_25 > param_first_scalar )then
          if (emis_ant(i,kts,j,p_e_pm_25) .GT. 8.19e-4*200) then
           ekmfull(kts+1:kte/2) = max(ekmfull(kts+1:kte/2),2.)
          endif
         endif
       endif
     endif
!!$! --- Mix the emissions up several layers when satellite data shows a wildfire
!!$!     if ebu_in_e_co > 0., a wildfire exists so increase vertical mixing
!     if (p_ebu_in_co > param_first_scalar )then
     if (p_ebu_in_co >= param_first_scalar )then
         if (ebu_in(i,1,j,p_ebu_in_co) .gt. 0) then
          ekmfull(kts+1:kte/2) = max(ekmfull(kts+1:kte/2),2.)
         endif
     endif

     do k=kts,kte
        zz(k)=z(i,k,j)-z_at_w(i,kts,j)
     enddo
!
!   vertical mixing routine (including deposition)
!   need to be careful here with that dumm tracer in spot 1
!   do not need lho,lho2
!   (03-may-2006 rce - calc dryrho_1d and pass it to vertmx)
!
      dep_vel_o3(i,j)=ddvel(i,j,p_o3)
      do nv=2,num_chem-0
         if(is_CAMMGMP_used .and. .not.vertMixAero(nv))cycle !Balwinder.Singh@pnnl.gov: Do mix constituents which are already mixed by CAMMGMP microphysics
         do k=kts,kte
            pblst(k)=max(epsilc,chem(i,k,j,nv))
            dryrho_1d(k) = 1./alt(i,k,j)
         enddo

         mix_select: SELECT CASE(config_flags%chem_opt)
         CASE (RADM2SORG_AQ, RADM2SORG_AQCHEM, RACMSORG_AQ, RACMSORG_AQCHEM_KPP, RACM_ESRLSORG_AQCHEM_KPP, CBMZ_MOSAIC_4BIN_AQ, &
              CBMZ_MOSAIC_8BIN_AQ, CBMZSORG_AQ, CBMZ_MOSAIC_DMS_4BIN, CBMZ_MOSAIC_DMS_8BIN, CBMZ_MOSAIC_DMS_4BIN_AQ,  &
              CBMZ_MOSAIC_DMS_8BIN_AQ, CRI_MOSAIC_8BIN_AQ_KPP, CRI_MOSAIC_4BIN_AQ_KPP,     &
              MOZART_MOSAIC_4BIN_AQ_KPP, RACM_SOA_VBS_AQCHEM_KPP,                          &
              SAPRC99_MOSAIC_8BIN_VBS2_AQ_KPP,                                             &
              CB05_SORG_AQ_KPP, CB05_SORG_VBS_AQ_KPP)
            if(.not.is_aerosol(nv))then ! mix gases not aerosol
               if (.not.config_flags%mynn_chem_vertmx) then
                  call vertmx(dtstep,pblst,ekmfull,dryrho_1d, &
                           zzfull,zz,ddvel(i,j,nv),kts,kte)
               endif
            endif
!The default case below does turbulent mixing for all gas and aerosol species
!If aqueous phase is not activated
!It requires ddvel array as a boundary condition near surface for flux of species
!The top boundary condition for eddy diffusivity is zero
         CASE DEFAULT
            if (.not.config_flags%mynn_chem_vertmx) then
               call vertmx(dtstep,pblst,ekmfull,dryrho_1d, &
                        zzfull,zz,ddvel(i,j,nv),kts,kte)
            endif
         END SELECT mix_select

         ! chem is in ppmv
         ! dry deposition is combined with vertical mixing, but column independent.
         ! Hence, all molecules lost per column must be dry deposited.

         ! old and new column totals (mol/m2 or ug/m2)
         old = 0.0
         new = 0.0

         do k=kts,kte-1
           fac = 1.0
           if (nv <= numgas) then
             ! from ppmv to mol/m2
             ! fac     = 1e-6 * rho * 1/mw_air * dz
             !                 kg/m3   mol/kg    m
             fac = 1e-6 * dryrho_1d(k) * 1./(mwdry*1.e-3) * dz8w(i,k,j)
           else
             ! from ug/kg to ug/m2
             ! fac     = rho * dz
             !          kg/m3  m
             fac = dryrho_1d(k) * dz8w(i,k,j)
           endif

           old = old + max(epsilc,chem(i,k,j,nv)) * fac
           new = new + max(epsilc,pblst(k)) * fac
         enddo

         ! we ignore (spurious) and add new dry deposition to
         ! existing field (accumulated deposition!)
         ddmassn(nv) =  max( 0.0, (old - new) )

         do k=kts,kte-1
            chem(i,k,j,nv)=max(epsilc,pblst(k))
         enddo
      enddo


 accum=accum+ddmassn
 call oracle_put('output',chem(1,:,1,2:3));call oracle_put('mixed',pblst);call oracle_put('ekmfull',ekmfull);call oracle_put('ddmassn',ddmassn(2:3));call oracle_put('accum',accum(2:3))
 call oracle_close()
 enddo
 enddo
 deallocate(chem,initial,z_at_w,z,alt,dz8w,exch_h,pblst,dryrho_1d,zzfull,zz,ekmfull)
 enddo
end program

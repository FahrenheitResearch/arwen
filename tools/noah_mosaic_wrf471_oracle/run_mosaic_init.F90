program run_mosaic_init
 use module_sf_noahdrv, only: lsm_mosaic_init
 use oracle_io
 implicit none
 integer, parameter :: nx=18, ny=2
 integer :: nlcat, mc, frac, i,j,k,a,b,c
 integer, allocatable :: ivgtyp(:,:), mosaic_cat_index(:,:,:)
 real, allocatable :: landusef(:,:,:),landusef2(:,:,:)
 character(len=1024) :: root
 character(len=80) :: label
 real :: tsk(nx,ny)
 real :: snow(nx,ny)
 real :: snowc(nx,ny)
 real :: snowh(nx,ny)
 real :: canwat(nx,ny)
 real :: albedo(nx,ny)
 real :: albbck(nx,ny)
 real :: emiss(nx,ny)
 real :: embck(nx,ny)
 real :: znt(nx,ny)
 real :: xland(nx,ny)
 real :: xice(nx,ny)
 real :: tslb(nx,4,ny)
 real :: smois(nx,4,ny)
 real :: sh2o(nx,4,ny)
 real, allocatable :: tsk_mosaic(:,:,:)
 real, allocatable :: canwat_mosaic(:,:,:)
 real, allocatable :: snow_mosaic(:,:,:)
 real, allocatable :: snowh_mosaic(:,:,:)
 real, allocatable :: snowc_mosaic(:,:,:)
 real, allocatable :: albedo_mosaic(:,:,:)
 real, allocatable :: albbck_mosaic(:,:,:)
 real, allocatable :: emiss_mosaic(:,:,:)
 real, allocatable :: embck_mosaic(:,:,:)
 real, allocatable :: znt_mosaic(:,:,:)
 real, allocatable :: z0_mosaic(:,:,:)
 real, allocatable :: tr_urb2d_mosaic(:,:,:)
 real, allocatable :: tb_urb2d_mosaic(:,:,:)
 real, allocatable :: tg_urb2d_mosaic(:,:,:)
 real, allocatable :: tc_urb2d_mosaic(:,:,:)
 real, allocatable :: qc_urb2d_mosaic(:,:,:)
 real, allocatable :: sh_urb2d_mosaic(:,:,:)
 real, allocatable :: lh_urb2d_mosaic(:,:,:)
 real, allocatable :: g_urb2d_mosaic(:,:,:)
 real, allocatable :: rn_urb2d_mosaic(:,:,:)
 real, allocatable :: ts_urb2d_mosaic(:,:,:)
 real, allocatable :: ts_rul_urb2d_mosaic(:,:,:)
 real, allocatable :: tslb_mosaic(:,:,:)
 real, allocatable :: smois_mosaic(:,:,:)
 real, allocatable :: sh2o_mosaic(:,:,:)
 real, allocatable :: trl_urb3d_mosaic(:,:,:)
 real, allocatable :: tbl_urb3d_mosaic(:,:,:)
 real, allocatable :: tgl_urb3d_mosaic(:,:,:)
 call oracle_root(root)
 do a=1,2
 nlcat=21
 if(a==2) nlcat=61
 do b=1,4
 select case(b)
 case(1)
 mc=1
 case(2)
 mc=3
 case(3)
 mc=5
 case(4)
 mc=8
 end select
 do frac=0,1
 allocate(ivgtyp(nx,ny),mosaic_cat_index(nx,nlcat,ny),landusef(nx,nlcat,ny),landusef2(nx,nlcat,ny))
 allocate(tsk_mosaic(nx,mc,ny))
 allocate(canwat_mosaic(nx,mc,ny))
 allocate(snow_mosaic(nx,mc,ny))
 allocate(snowh_mosaic(nx,mc,ny))
 allocate(snowc_mosaic(nx,mc,ny))
 allocate(albedo_mosaic(nx,mc,ny))
 allocate(albbck_mosaic(nx,mc,ny))
 allocate(emiss_mosaic(nx,mc,ny))
 allocate(embck_mosaic(nx,mc,ny))
 allocate(znt_mosaic(nx,mc,ny))
 allocate(z0_mosaic(nx,mc,ny))
 allocate(tr_urb2d_mosaic(nx,mc,ny))
 allocate(tb_urb2d_mosaic(nx,mc,ny))
 allocate(tg_urb2d_mosaic(nx,mc,ny))
 allocate(tc_urb2d_mosaic(nx,mc,ny))
 allocate(qc_urb2d_mosaic(nx,mc,ny))
 allocate(sh_urb2d_mosaic(nx,mc,ny))
 allocate(lh_urb2d_mosaic(nx,mc,ny))
 allocate(g_urb2d_mosaic(nx,mc,ny))
 allocate(rn_urb2d_mosaic(nx,mc,ny))
 allocate(ts_urb2d_mosaic(nx,mc,ny))
 allocate(ts_rul_urb2d_mosaic(nx,mc,ny))
 allocate(tslb_mosaic(nx,4*mc,ny))
 allocate(smois_mosaic(nx,4*mc,ny))
 allocate(sh2o_mosaic(nx,4*mc,ny))
 allocate(trl_urb3d_mosaic(nx,4*mc,ny))
 allocate(tbl_urb3d_mosaic(nx,4*mc,ny))
 allocate(tgl_urb3d_mosaic(nx,4*mc,ny))
 landusef=0.; landusef2=-99.; mosaic_cat_index=-99
 ivgtyp=7; xland=1.;xice=0.
 do j=1,ny
 do i=1,nx
 landusef(i,1,j)=.7;landusef(i,7,j)=.2;landusef(i,12,j)=.1
 select case(i)
 case(1)
 landusef(i,:,j)=0.;landusef(i,1:5,j)=[.4,.3,.15,.1,.05]
 case(2)
 landusef(i,:,j)=0.;landusef(i,1:4,j)=.25
 case(3)
 landusef(i,:,j)=0.;landusef(i,1:2,j)=.5
 case(4)
 landusef(i,:,j)=0.;landusef(i,1:3,j)=1./3.
 case(5)
 landusef(i,:,j)=0.;landusef(i,7,j)=1.
 case(6)
 landusef(i,:,j)=0.;landusef(i,21,j)=1.
 case(7)
 landusef(i,7,j)=.1;landusef(i,17,j)=.2
 case(8,9)
 landusef(i,1,j)=.2;landusef(i,17,j)=.7
 if(i==9) ivgtyp(i,j)=17
 case(10)
 xland(i,j)=2.;ivgtyp(i,j)=17
 case(11)
 xice(i,j)=.5;ivgtyp(i,j)=15;landusef(i,1,j)=.1;landusef(i,15,j)=.7
 case(12)
 xice(i,j)=.5;landusef(i,15,j)=.3
 case(13)
 landusef(i,:,j)=0.;landusef(i,1,j)=1e-7;landusef(i,7,j)=2e-7
 case(14)
 landusef(i,:,j)=0.;landusef(i,1,j)=.1;landusef(i,7,j)=1./3.;landusef(i,12,j)=.7
 case(15)
 xice(i,j)=.02;landusef(i,15,j)=.3
 case(16)
 xland(i,j)=2.;ivgtyp(i,j)=7
 case(17)
 if(nlcat==61) then
 landusef(i,:,j)=0.;landusef(i,51,j)=.5;landusef(i,61,j)=.3;landusef(i,13,j)=.2
 endif
 case(18)
 landusef(i,:,j)=0.
 end select
 tsk(i,j)=270.+i+j*.1; snow(i,j)=i*.1; snowc(i,j)=.1; snowh(i,j)=.2
 canwat(i,j)=.01;albedo(i,j)=.2;albbck(i,j)=.21;emiss(i,j)=.96;embck(i,j)=.95;znt(i,j)=.1
 do k=1,4
 tslb(i,k,j)=270.+i+k*.7; smois(i,k,j)=.2+k*.01;sh2o(i,k,j)=.1+k*.01
 enddo
 enddo
 enddo
 write(label,'(a,i0,a,i0,a,i0)') 'n',nlcat,'_m',mc,'_f',frac
 call oracle_open(trim(label))
 call oracle_put('nlcat',nlcat);call oracle_put('mosaic_cat',mc);call oracle_put('fractional_seaice',frac)
 call oracle_put('ivgtyp',ivgtyp); call oracle_put('landusef',landusef)
 call oracle_put('tsk',tsk)
 call oracle_put('snow',snow)
 call oracle_put('snowc',snowc)
 call oracle_put('snowh',snowh)
 call oracle_put('canwat',canwat)
 call oracle_put('albedo',albedo)
 call oracle_put('albbck',albbck)
 call oracle_put('emiss',emiss)
 call oracle_put('embck',embck)
 call oracle_put('znt',znt)
 call oracle_put('xland',xland)
 call oracle_put('xice',xice)
 call oracle_put('tslb',tslb)
 call oracle_put('smois',smois)
 call oracle_put('sh2o',sh2o)
 call lsm_mosaic_init(ivgtyp,17,13,15,xland,xice,frac, &
 tsk,tslb,smois,sh2o,snow,snowc,snowh,canwat, &
 1,nx+1,1,ny+1,1,2,1,nx,1,ny,1,2,1,nx,1,ny,1,2,.false., &
 landusef,landusef2,nlcat,4,1,mc,mosaic_cat_index, &
 tsk_mosaic,tslb_mosaic,smois_mosaic,sh2o_mosaic, &
 canwat_mosaic,snow_mosaic,snowh_mosaic,snowc_mosaic, &
 albedo,albbck,emiss,embck,znt, &
 albedo_mosaic,albbck_mosaic,emiss_mosaic,embck_mosaic,znt_mosaic,z0_mosaic, &
 tr_urb2d_mosaic,tb_urb2d_mosaic,tg_urb2d_mosaic,tc_urb2d_mosaic,qc_urb2d_mosaic, &
 trl_urb3d_mosaic,tbl_urb3d_mosaic,tgl_urb3d_mosaic, &
 sh_urb2d_mosaic,lh_urb2d_mosaic,g_urb2d_mosaic,rn_urb2d_mosaic, &
 ts_urb2d_mosaic,ts_rul_urb2d_mosaic)
 call oracle_put('landusef2_full',landusef2);call oracle_put('mosaic_cat_index_full',mosaic_cat_index)
 call oracle_put('tsk_mosaic',tsk_mosaic)
 call oracle_put('canwat_mosaic',canwat_mosaic)
 call oracle_put('snow_mosaic',snow_mosaic)
 call oracle_put('snowh_mosaic',snowh_mosaic)
 call oracle_put('snowc_mosaic',snowc_mosaic)
 call oracle_put('albedo_mosaic',albedo_mosaic)
 call oracle_put('albbck_mosaic',albbck_mosaic)
 call oracle_put('emiss_mosaic',emiss_mosaic)
 call oracle_put('embck_mosaic',embck_mosaic)
 call oracle_put('znt_mosaic',znt_mosaic)
 call oracle_put('z0_mosaic',z0_mosaic)
 call oracle_put('tr_urb2d_mosaic',tr_urb2d_mosaic)
 call oracle_put('tb_urb2d_mosaic',tb_urb2d_mosaic)
 call oracle_put('tg_urb2d_mosaic',tg_urb2d_mosaic)
 call oracle_put('tc_urb2d_mosaic',tc_urb2d_mosaic)
 call oracle_put('qc_urb2d_mosaic',qc_urb2d_mosaic)
 call oracle_put('sh_urb2d_mosaic',sh_urb2d_mosaic)
 call oracle_put('lh_urb2d_mosaic',lh_urb2d_mosaic)
 call oracle_put('g_urb2d_mosaic',g_urb2d_mosaic)
 call oracle_put('rn_urb2d_mosaic',rn_urb2d_mosaic)
 call oracle_put('ts_urb2d_mosaic',ts_urb2d_mosaic)
 call oracle_put('ts_rul_urb2d_mosaic',ts_rul_urb2d_mosaic)
 call oracle_put('tslb_mosaic',tslb_mosaic)
 call oracle_put('smois_mosaic',smois_mosaic)
 call oracle_put('sh2o_mosaic',sh2o_mosaic)
 call oracle_put('trl_urb3d_mosaic',trl_urb3d_mosaic)
 call oracle_put('tbl_urb3d_mosaic',tbl_urb3d_mosaic)
 call oracle_put('tgl_urb3d_mosaic',tgl_urb3d_mosaic)
 call oracle_close()
 deallocate(ivgtyp,mosaic_cat_index,landusef,landusef2)
 deallocate(tsk_mosaic)
 deallocate(canwat_mosaic)
 deallocate(snow_mosaic)
 deallocate(snowh_mosaic)
 deallocate(snowc_mosaic)
 deallocate(albedo_mosaic)
 deallocate(albbck_mosaic)
 deallocate(emiss_mosaic)
 deallocate(embck_mosaic)
 deallocate(znt_mosaic)
 deallocate(z0_mosaic)
 deallocate(tr_urb2d_mosaic)
 deallocate(tb_urb2d_mosaic)
 deallocate(tg_urb2d_mosaic)
 deallocate(tc_urb2d_mosaic)
 deallocate(qc_urb2d_mosaic)
 deallocate(sh_urb2d_mosaic)
 deallocate(lh_urb2d_mosaic)
 deallocate(g_urb2d_mosaic)
 deallocate(rn_urb2d_mosaic)
 deallocate(ts_urb2d_mosaic)
 deallocate(ts_rul_urb2d_mosaic)
 deallocate(tslb_mosaic)
 deallocate(smois_mosaic)
 deallocate(sh2o_mosaic)
 deallocate(trl_urb3d_mosaic)
 deallocate(tbl_urb3d_mosaic)
 deallocate(tgl_urb3d_mosaic)
 enddo
 enddo
 enddo
end program

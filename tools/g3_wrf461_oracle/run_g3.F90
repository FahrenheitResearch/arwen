! Column-domain oracle for WRF v4.6.1 Grell 3D (cu_physics = 5).
!
! Drives the byte-unmodified phys/module_cu_g3.F exactly as
! phys/module_cumulus_driver.F does for a single-tile, single-patch,
! non-periodic ARW run: G3DRV over the whole domain (CASE (G3SCHEME),
! :1218-1270), then conv_grell_spread3d (:1635-1650) on the same arrays.
! The forcing fold of :867-886 is NOT done here: the input file carries
! RTHFTEN/RQVFTEN already folded, i.e. what G3DRV receives.
!
! Memory is laid out the way WRF lays a domain-edge tile out: a 5-point
! halo outside ids..ide-1 on both axes, zero-filled (WRF allocates state
! zeroed and a non-periodic domain never fills the outer halo).  Every
! graded column sits in the interior band ids+4..ide-5, whose stencil never
! reaches the halo.
!
! usage: run_g3 <input.bin> <output.bin>
program run_g3
   use module_cu_g3, only: g3drv, conv_grell_spread3d
   implicit none

   real, parameter :: r_d = 287., cp = 7.*r_d/2., r_v = 461.6, g = 9.81, &
                      xlv = 2.5e6
   integer :: maxiens = 1, maxens = 3, maxens2 = 3, maxens3 = 16, ensdim = 144
   integer, parameter :: nh = 5
   character(len=32) :: arg

   character(len=1024) :: fin, fout
   integer :: nx, ny, nz, ishallow, avedx, ichoice, fqi, itimestep
   real :: dt, dx
   integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
   integer :: ips, ipe, jps, jpe, kps, kpe, its, ite, jts, jte, kts, kte
   integer :: i, j, k, n, ii, jj, ipx=0, ipy=0
   logical :: periodic_x, periodic_y
   logical :: f_qi

   real, allocatable, dimension(:,:,:) :: u, v, w, t, q, p, pi, rho, dz8w, &
        p8w, rthften, rqvften, rthblten, rqvblten, rthcuten, rqvcuten, &
        rqccuten, rqicuten, gdc, gdc2, cugd_tten, cugd_qvten, cugd_qcten, &
        cugd_ttens, cugd_qvtens, xf_ens, pr_ens
   real, allocatable, dimension(:,:) :: ht, xland, gsw, raincv, pratec, htop, &
        hbot, mass_flux, edt_out, xmb_shallow, apr_gr, apr_w, apr_mc, apr_st, &
        apr_as, apr_capma, apr_capme, apr_capmi
   integer, allocatable, dimension(:,:) :: kpbl, k22_shallow, kbcon_shallow, &
        ktop_shallow, ktop_deep
   logical, allocatable, dimension(:,:) :: cu_act_flag
   real, allocatable :: buf3(:,:,:), buf2(:,:)
   integer, allocatable :: ibuf2(:,:)

   call get_command_argument(1, fin)
   call get_command_argument(2, fout)
   call get_command_argument(3, arg); if (len_trim(arg)>0) read(arg,*) maxiens
   call get_command_argument(4, arg); if (len_trim(arg)>0) read(arg,*) maxens
   call get_command_argument(5, arg); if (len_trim(arg)>0) read(arg,*) maxens2
   call get_command_argument(6, arg); if (len_trim(arg)>0) read(arg,*) maxens3
   call get_command_argument(7, arg); if (len_trim(arg)>0) read(arg,*) ensdim
   call get_command_argument(8, arg); if (len_trim(arg)>0) read(arg,*) ipx
   call get_command_argument(9, arg); if (len_trim(arg)>0) read(arg,*) ipy
   periodic_x = ipx /= 0; periodic_y = ipy /= 0
   open(10, file=trim(fin), access='stream', form='unformatted', status='old')
   read(10) nx, ny, nz, ishallow, avedx, ichoice, fqi, itimestep
   read(10) dt, dx
   f_qi = fqi /= 0

   ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
   its = 1; ite = nx;     jts = 1; jte = ny;     kts = 1; kte = nz
   ips = 1; ipe = ide;    jps = 1; jpe = jde;    kps = 1; kpe = kde
   ims = ids - nh; ime = ide + nh; jms = jds - nh; jme = jde + nh
   kms = 1; kme = kde

#define A3(x) allocate(x(ims:ime,kms:kme,jms:jme)); x = 0.;
#define A2(x) allocate(x(ims:ime,jms:jme)); x = 0.;
   A3(u) A3(v) A3(w) A3(t) A3(q) A3(p) A3(pi) A3(rho) A3(dz8w) A3(p8w)
   A3(rthften) A3(rqvften) A3(rthblten) A3(rqvblten) A3(rthcuten)
   A3(rqvcuten) A3(rqccuten) A3(rqicuten) A3(gdc) A3(gdc2) A3(cugd_tten)
   A3(cugd_qvten) A3(cugd_qcten) A3(cugd_ttens) A3(cugd_qvtens)
   A2(ht) A2(xland) A2(gsw) A2(raincv) A2(pratec) A2(htop) A2(hbot)
   A2(mass_flux) A2(edt_out) A2(xmb_shallow) A2(apr_gr) A2(apr_w) A2(apr_mc)
   A2(apr_st) A2(apr_as) A2(apr_capma) A2(apr_capme) A2(apr_capmi)
   allocate(xf_ens(ims:ime,jms:jme,ensdim)); xf_ens = 0.
   allocate(pr_ens(ims:ime,jms:jme,ensdim)); pr_ens = 0.
   allocate(kpbl(ims:ime,jms:jme), k22_shallow(ims:ime,jms:jme), &
            kbcon_shallow(ims:ime,jms:jme), ktop_shallow(ims:ime,jms:jme), &
            ktop_deep(ims:ime,jms:jme), cu_act_flag(ims:ime,jms:jme))
   kpbl = 0; k22_shallow = 0; kbcon_shallow = 0; ktop_shallow = 0
   ktop_deep = 0; cu_act_flag = .false.

   allocate(buf3(nx, nz, ny), buf2(nx, ny), ibuf2(nx, ny))
#define RD3(x) read(10) buf3; do j=1,ny; do k=1,nz; do i=1,nx; x(i,k,j)=buf3(i,k,j); enddo; enddo; enddo;
   RD3(u) RD3(v) RD3(t) RD3(q) RD3(p) RD3(pi) RD3(rho) RD3(dz8w)
   RD3(rthften) RD3(rqvften) RD3(rthblten) RD3(rqvblten)
   deallocate(buf3); allocate(buf3(nx, nz+1, ny))
#define RD3W(x) read(10) buf3; do j=1,ny; do k=1,nz+1; do i=1,nx; x(i,k,j)=buf3(i,k,j); enddo; enddo; enddo;
   RD3W(w) RD3W(p8w)
#define RD2(x) read(10) buf2; x(1:nx,1:ny) = buf2;
   RD2(ht) RD2(xland) RD2(gsw)
   read(10) ibuf2; kpbl(1:nx,1:ny) = ibuf2
   RD2(apr_gr) RD2(apr_w) RD2(apr_mc) RD2(apr_st) RD2(apr_as)
   RD2(apr_capma) RD2(apr_capme) RD2(apr_capmi)
   close(10)

   ! Periodic tendency halo supplied by WRF's exchange before G3DRV.
   do j=jms,jme
      if ((j<1.or.j>ny).and..not.periodic_y) cycle
      jj=modulo(j-1,ny)+1
      do i=ims,ime
         if ((i<1.or.i>nx).and..not.periodic_x) cycle
         ii=modulo(i-1,nx)+1
         do k=1,nz
            rthften(i,k,j)=rthften(ii,k,jj)
            rqvften(i,k,j)=rqvften(ii,k,jj)
         enddo
      enddo
   enddo

   call g3drv(DT=dt, ITIMESTEP=itimestep, DX=dx, rho=rho, RAINCV=raincv,   &
        PRATEC=pratec, U=u, V=v, t=t, W=w, q=q, p=p, pi=pi, dz8w=dz8w,      &
        p8w=p8w, XLV=xlv, CP=cp, G=g, r_v=r_v, htop=htop, hbot=hbot,        &
        CU_ACT_FLAG=cu_act_flag, warm_rain=.false., APR_GR=apr_gr,          &
        APR_W=apr_w, APR_MC=apr_mc, APR_ST=apr_st, APR_AS=apr_as,           &
        APR_CAPMA=apr_capma, APR_CAPME=apr_capme, APR_CAPMI=apr_capmi,      &
        MASS_FLUX=mass_flux, XF_ENS=xf_ens, PR_ENS=pr_ens, HT=ht,           &
        XLAND=xland, gsw=gsw, edt_out=edt_out, GDC=gdc, GDC2=gdc2,          &
        kpbl=kpbl, k22_shallow=k22_shallow, kbcon_shallow=kbcon_shallow,    &
        ktop_shallow=ktop_shallow, xmb_shallow=xmb_shallow,                 &
        ktop_deep=ktop_deep, cugd_tten=cugd_tten, cugd_qvten=cugd_qvten,    &
        cugd_qcten=cugd_qcten, cugd_ttens=cugd_ttens,                       &
        cugd_qvtens=cugd_qvtens, cugd_avedx=avedx, imomentum=0,             &
        ensdim=ensdim, maxiens=maxiens, maxens=maxens, maxens2=maxens2,     &
        maxens3=maxens3, ichoice=ichoice, ishallow_g3=ishallow,             &
        ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,               &
        ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,               &
        ips=ips, ipe=ipe, jps=jps, jpe=jpe, kps=kps, kpe=kpe,               &
        its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte,               &
        periodic_x=periodic_x, periodic_y=periodic_y,                             &
        RQVCUTEN=rqvcuten, RQCCUTEN=rqccuten, RQICUTEN=rqicuten,            &
        RQVFTEN=rqvften, RTHFTEN=rthften, RTHCUTEN=rthcuten,                &
        rqvblten=rqvblten, rthblten=rthblten,                               &
        F_QV=.true., F_QC=.true., F_QR=.true., F_QI=f_qi, F_QS=f_qi)

   open(20, file=trim(fout), access='stream', form='unformatted', status='replace')
   ! ---- stage 1: what G3DRV wrote ------------------------------------------
#define WR3(x) write(20) ((( x(i,k,j), i=1,nx), k=1,nz), j=1,ny);
#define WR2(x) write(20) (( x(i,j), i=1,nx), j=1,ny);
   WR3(cugd_tten) WR3(cugd_qvten) WR3(cugd_qcten) WR3(cugd_ttens)
   WR3(cugd_qvtens) WR3(rqccuten) WR3(rqicuten) WR3(gdc) WR3(gdc2)
   WR2(raincv) WR2(pratec) WR2(htop) WR2(hbot) WR2(mass_flux) WR2(edt_out)
   WR2(xmb_shallow) WR2(apr_gr) WR2(apr_w) WR2(apr_mc) WR2(apr_st)
   WR2(apr_as) WR2(apr_capma) WR2(apr_capme) WR2(apr_capmi)
   WR2(k22_shallow) WR2(kbcon_shallow) WR2(ktop_shallow) WR2(ktop_deep)
   write(20) ((merge(1, 0, cu_act_flag(i,j)), i=1,nx), j=1,ny)
   write(20) ((( xf_ens(i,j,n), i=1,nx), j=1,ny), n=1,ensdim)
   write(20) ((( pr_ens(i,j,n), i=1,nx), j=1,ny), n=1,ensdim)

   call conv_grell_spread3d(rthcuten=rthcuten, rqvcuten=rqvcuten,         &
        rqccuten=rqccuten, raincv=raincv, cugd_avedx=avedx,                 &
        cugd_tten=cugd_tten, cugd_qvten=cugd_qvten, rqicuten=rqicuten,      &
        cugd_ttens=cugd_ttens, cugd_qvtens=cugd_qvtens,                     &
        cugd_qcten=cugd_qcten, pi_phy=pi, moist_qv=q, pratec=pratec, dt=dt, &
        num_tiles=1, imomentum=0, F_QV=.true., F_QC=.true., F_QR=.true.,    &
        F_QI=f_qi, F_QS=f_qi,                                               &
        ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,               &
        ips=ips, ipe=ipe, jps=jps, jpe=jpe, kps=kps, kpe=kpe,               &
        ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,               &
        its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

   ! ---- stage 2: what conv_grell_spread3d left -------------------------------
   WR3(rthcuten) WR3(rqvcuten) WR3(rqccuten) WR3(rqicuten)
   WR2(raincv) WR2(pratec)
   close(20)
end program run_g3

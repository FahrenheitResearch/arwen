! Oracle for the packaged Kain--Fritsch lookup table.
!
! This is WRF v4.6.1 `phys/module_cu_kfeta.F:3174-3301` copied verbatim, with
! only the module-level declarations it needs (`module_cu_kfeta.F:18-22`), the
! SVP constants it is called with (`share/module_model_constants.F:76-79`,
! via `kf_eta_init`), and a raw dump of the five results wrapped around it.
! WRF is public domain; see `gpuwm/data/kf_lutab/LICENSE.txt`.
!
! Parent file SHA-256:
!   b2ee225b2148d54afa464f941d967cb197a8a73e23d6fd3450086c6f3a705895
!
! Build and run (reference toolchain: gfortran 13.3.0, glibc 2.39, x86-64):
!
!     gfortran -O0 -ffp-contract=off -o kf_oracle tools/kf_lutab_oracle.F90
!     nm -u kf_oracle | grep -E 'expf|logf|powf'   # scalar glibc, no libmvec
!     ./kf_oracle                                  # writes the five .bin files
!
! Everything here is default `REAL`, i.e. binary32: WRF v4.6.1 ships
! NATIVE_RWORDSIZE = 4 and leaves `-fdefault-real-8` commented out in every
! arch/configure.defaults PROMOTION line, so this is the precision WRF runs.
!
! Output is little-endian binary32 stream, Fortran column-major:
!   ttab.bin    250*220   parcel temperature   -> `temperature` (transposed)
!   qstab.bin   250*220   saturation mixing r. -> `qsat`        (transposed)
!   the0k.bin   220       -> `thetae_base`
!   alu.bin     200       -> `log_ratio`
!   scalars.bin RDPR, RDTHK, PLUTOP
!
! `tools/generate_kf_lutab.py` reproduces all 110,420 table cells from this
! program bit-for-bit; `tests/test_kf.py` pins the digests recorded in
! `gpuwm/data/kf_lutab/PROVENANCE.md`.
MODULE kf_lutab_oracle_state
! module_cu_kfeta.F:18-22, verbatim
      INTEGER, PARAMETER, PRIVATE :: KFNT=250,KFNP=220
      REAL, DIMENSION(KFNT,KFNP),PRIVATE, SAVE :: TTAB,QSTAB
      REAL, DIMENSION(KFNP),PRIVATE, SAVE :: THE0K
      REAL, DIMENSION(200),PRIVATE, SAVE :: ALU
      REAL, PRIVATE, SAVE :: RDPR,RDTHK,PLUTOP
CONTAINS
! ---- BEGIN verbatim module_cu_kfeta.F:3174-3301 ----
      subroutine kf_lutab(SVP1,SVP2,SVP3,SVPT0)
!
!  This subroutine is a lookup table.
!  Given a series of series of saturation equivalent potential 
!  temperatures, the temperature is calculated.
!
!--------------------------------------------------------------------
   IMPLICIT NONE
!--------------------------------------------------------------------
! Lookup table variables
!     INTEGER, SAVE, PARAMETER :: KFNT=250,KFNP=220
!     REAL, SAVE, DIMENSION(1:KFNT,1:KFNP) :: TTAB,QSTAB
!     REAL, SAVE, DIMENSION(1:KFNP) :: THE0K
!     REAL, SAVE, DIMENSION(1:200) :: ALU
!     REAL, SAVE :: RDPR,RDTHK,PLUTOP
! End of Lookup table variables

     INTEGER :: KP,IT,ITCNT,I
     REAL :: DTH,TMIN,TOLER,PBOT,DPR,                               &
             TEMP,P,ES,QS,PI,THES,TGUES,THGUES,F0,T1,T0,THGS,F1,DT, &
             ASTRT,AINC,A1,THTGS
!    REAL    :: ALIQ,BLIQ,CLIQ,DLIQ,SVP1,SVP2,SVP3,SVPT0
     REAL    :: ALIQ,BLIQ,CLIQ,DLIQ
     REAL, INTENT(IN)    :: SVP1,SVP2,SVP3,SVPT0
!
! equivalent potential temperature increment
      data dth/1./
! minimum starting temp 
      data tmin/150./
! tolerance for accuracy of temperature 
      data toler/0.001/
! top pressure (pascals)
      plutop=5000.0
! bottom pressure (pascals)
      pbot=110000.0

      ALIQ = SVP1*1000.
      BLIQ = SVP2
      CLIQ = SVP2*SVPT0
      DLIQ = SVP3

!
! compute parameters
!
! 1._over_(sat. equiv. theta increment)
      rdthk=1./dth
! pressure increment
!
      DPR=(PBOT-PLUTOP)/REAL(KFNP-1)
!      dpr=(pbot-plutop)/REAL(kfnp-1)
! 1._over_(pressure increment)
      rdpr=1./dpr
! compute the spread of thes
!     thespd=dth*(kfnt-1)
!
! calculate the starting sat. equiv. theta
!
      temp=tmin 
      p=plutop-dpr
      do kp=1,kfnp
        p=p+dpr
        es=aliq*exp((bliq*temp-cliq)/(temp-dliq))
        qs=0.622*es/(p-es)
        pi=(1.e5/p)**(0.2854*(1.-0.28*qs))
        the0k(kp)=temp*pi*exp((3374.6525/temp-2.5403)*qs*        &
               (1.+0.81*qs))
      enddo   
!
! compute temperatures for each sat. equiv. potential temp.
!
      p=plutop-dpr
      do kp=1,kfnp
        thes=the0k(kp)-dth
        p=p+dpr
        do it=1,kfnt
! define sat. equiv. pot. temp.
          thes=thes+dth
! iterate to find temperature
! find initial guess
          if(it.eq.1) then
            tgues=tmin
          else
            tgues=ttab(it-1,kp)
          endif
          es=aliq*exp((bliq*tgues-cliq)/(tgues-dliq))
          qs=0.622*es/(p-es)
          pi=(1.e5/p)**(0.2854*(1.-0.28*qs))
          thgues=tgues*pi*exp((3374.6525/tgues-2.5403)*qs*      &
               (1.+0.81*qs))
          f0=thgues-thes
          t1=tgues-0.5*f0
          t0=tgues
          itcnt=0
! iteration loop
          do itcnt=1,11
            es=aliq*exp((bliq*t1-cliq)/(t1-dliq))
            qs=0.622*es/(p-es)
            pi=(1.e5/p)**(0.2854*(1.-0.28*qs))
            thtgs=t1*pi*exp((3374.6525/t1-2.5403)*qs*(1.+0.81*qs))
            f1=thtgs-thes
            if(abs(f1).lt.toler)then
              exit
            endif
!           itcnt=itcnt+1
            dt=f1*(t1-t0)/(f1-f0)
            t0=t1
            f0=f1
            t1=t1-dt
          enddo 
          ttab(it,kp)=t1 
          qstab(it,kp)=qs
        enddo
      enddo   
!
! lookup table for tlog(emix/aliq)
!
! set up intial values for lookup tables
!
       astrt=1.e-3
       ainc=0.075
!
       a1=astrt-ainc
       do i=1,200
         a1=a1+ainc
         alu(i)=alog(a1)
       enddo   
!
   END SUBROUTINE KF_LUTAB
! ---- END verbatim module_cu_kfeta.F:3174-3301 ----

      subroutine dump_tables()
      integer :: u
      open(newunit=u,file="ttab.bin",form="unformatted",access="stream")
      write(u) TTAB
      close(u)
      open(newunit=u,file="qstab.bin",form="unformatted",access="stream")
      write(u) QSTAB
      close(u)
      open(newunit=u,file="the0k.bin",form="unformatted",access="stream")
      write(u) THE0K
      close(u)
      open(newunit=u,file="alu.bin",form="unformatted",access="stream")
      write(u) ALU
      close(u)
      open(newunit=u,file="scalars.bin",form="unformatted",access="stream")
      write(u) RDPR,RDTHK,PLUTOP
      close(u)
      end subroutine dump_tables
END MODULE kf_lutab_oracle_state

PROGRAM kf_lutab_oracle
      USE kf_lutab_oracle_state
      IMPLICIT NONE
! share/module_model_constants.F:76-79, as passed by kf_eta_init
      REAL, PARAMETER :: SVP1=0.6112, SVP2=17.67, SVP3=29.65, SVPT0=273.15
      CALL kf_lutab(SVP1,SVP2,SVP3,SVPT0)
      CALL dump_tables()
END PROGRAM kf_lutab_oracle

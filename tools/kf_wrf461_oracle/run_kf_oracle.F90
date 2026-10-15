! Column driver for WRF v4.6.1 Kain-Fritsch (cu_physics=1), byte-unmodified
! phys/module_cu_kfeta.F.
!
! Each column is one call of KF_eta_CPS on a 1 x 1 horizontal tile, exactly
! the call module_cumulus_driver.F:1015-1050 makes, with the WRF constants
! from share/module_model_constants.F passed the way that call passes them.
! KF_eta_CPS then does what it does in WRF: advances W0AVG, applies the
! NCA >= 0.5*DT skip, copies the column to 1-D, calls KF_eta_PARA and maps
! DTDT/DQ*DT to the RTH/RQ*CUTEN arrays with the F_QR/F_QI/F_QS branches.
!
! Input (stream, little endian):  'KFCOL1  ', int32 ncol, then per column
!   int32  nz, warm_rain, f_qr, f_qi, f_qs, stepcu, adapt, trigger
!   real32 dt, dx, cudt(minutes), nca_in
!   real32 u, v, t, qv, p, dz8w, rho, pi, w0avg  (nz each), w (nz+1)
! Output: 'KFOUT1  ', int32 ncol, then per column
!   int32  nz, fatal, messages, unit98_bytes
!   real32 raincv, pratec, nca, cutop, cubot, shall
!   real32 rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten,
!          w0avg, cldfra_dp, cldfra_sh, qc_kf, qi_kf          (nz each)
! Tendency arrays a branch leaves unwritten (RQICUTEN without F_QI,
! RQSCUTEN without F_QS) keep the sentinel -999.
PROGRAM run_kf_oracle
  USE module_wrf_error, ONLY: kf_oracle_fatal, kf_oracle_messages
  USE module_cu_kfeta, ONLY: KF_eta_CPS, kf_lutab
  IMPLICIT NONE
  ! share/module_model_constants.F:17-81 (v4.6.1)
  REAL, PARAMETER :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  REAL, PARAMETER :: XLV0 = 3.15E6, XLV1 = 2370., XLS0 = 2.905E6, XLS1 = 259.532
  REAL, PARAMETER :: SVP1 = 0.6112, SVP2 = 17.67, SVP3 = 29.65, SVPT0 = 273.15
  REAL, PARAMETER :: EP_1 = R_v/R_d-1., EP_2 = R_d/R_v
  CHARACTER(LEN=8) :: magic
  CHARACTER(LEN=256) :: inpath, outpath
  INTEGER :: ncol, icol, nz, iwarm, ifqr, ifqi, ifqs, stepcu, iadapt, trig
  INTEGER :: ifatal, nmsg
  INTEGER(KIND=8) :: sz98_before, sz98_after
  REAL :: dt, dx, cudt, nca_in
  LOGICAL :: warm_rain, f_qr, f_qi, f_qs, adapt, ex98
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: u3, v3, t3, th3, qv3, p3, dz3, rho3, &
       pi3, w03, w3, rth, rqv, rqc, rqr, rqi, rqs, rqvften, cdp, csh, qck, qik, &
       udr, ddr, uer, der
  REAL, DIMENSION(1,1) :: raincv, pratec, nca, cutop, cubot, shall, timec_kf
  LOGICAL, DIMENSION(1,1) :: cu_act_flag
  REAL, ALLOCATABLE :: col(:)

  CALL get_command_argument(1, inpath)
  CALL get_command_argument(2, outpath)
  OPEN(10, FILE=TRIM(inpath), ACCESS='stream', FORM='unformatted', STATUS='old')
  OPEN(11, FILE=TRIM(outpath), ACCESS='stream', FORM='unformatted', STATUS='replace')
  READ(10) magic, ncol
  IF (magic /= 'KFCOL1  ') STOP 'bad input magic'
  WRITE(11) 'KFOUT1  ', ncol

  ! kf_eta_init's table build (module_cu_kfeta.F:3168), with the SVP
  ! constants module_physics_init passes it.
  CALL kf_lutab(SVP1, SVP2, SVP3, SVPT0)

  DO icol = 1, ncol
    READ(10) nz, iwarm, ifqr, ifqi, ifqs, stepcu, iadapt, trig
    READ(10) dt, dx, cudt, nca_in
    warm_rain = iwarm /= 0
    f_qr = ifqr /= 0
    f_qi = ifqi /= 0
    f_qs = ifqs /= 0
    adapt = iadapt /= 0
    ALLOCATE(u3(1,nz+1,1), v3(1,nz+1,1), t3(1,nz+1,1), th3(1,nz+1,1), &
             qv3(1,nz+1,1), p3(1,nz+1,1), dz3(1,nz+1,1), rho3(1,nz+1,1), &
             pi3(1,nz+1,1), w03(1,nz+1,1), w3(1,nz+1,1), rth(1,nz+1,1), &
             rqv(1,nz+1,1), rqc(1,nz+1,1), rqr(1,nz+1,1), rqi(1,nz+1,1), &
             rqs(1,nz+1,1), rqvften(1,nz+1,1), cdp(1,nz+1,1), csh(1,nz+1,1), &
             qck(1,nz+1,1), qik(1,nz+1,1), udr(1,nz+1,1), ddr(1,nz+1,1), &
             uer(1,nz+1,1), der(1,nz+1,1), col(nz+1))
    u3 = 0.; v3 = 0.; t3 = 0.; th3 = 0.; qv3 = 0.; p3 = 0.; dz3 = 0.
    rho3 = 0.; pi3 = 0.; w03 = 0.; w3 = 0.
    CALL rd(u3); CALL rd(v3); CALL rd(t3); CALL rd(qv3); CALL rd(p3)
    CALL rd(dz3); CALL rd(rho3); CALL rd(pi3); CALL rd(w03)
    READ(10) col(1:nz+1)
    w3(1,1:nz+1,1) = col(1:nz+1)
    th3 = t3
    rth = -999.; rqv = -999.; rqc = -999.; rqr = -999.; rqi = -999.; rqs = -999.
    rqvften = 0.; cdp = -999.; csh = -999.; qck = -999.; qik = -999.
    udr = 0.; ddr = 0.; uer = 0.; der = 0.
    raincv = -999.; pratec = -999.; nca = nca_in; cutop = -999.; cubot = -999.
    shall = -999.; timec_kf = 0.; cu_act_flag = .TRUE.
    kf_oracle_fatal = .FALSE.
    kf_oracle_messages = 0
    sz98_before = 0
    INQUIRE(FILE='fort.98', EXIST=ex98, SIZE=sz98_before)
    IF (.NOT. ex98) sz98_before = 0

    CALL KF_eta_CPS( &
         ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz+1, &
         ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz+1, &
         its=1, ite=1, jts=1, jte=1, kts=1, kte=nz, &
         trigger=trig, &
         DT=dt, KTAU=1, DX=dx, CUDT=cudt, ADAPT_STEP_FLAG=adapt, &
         rho=rho3, RAINCV=raincv, PRATEC=pratec, NCA=nca, &
         U=u3, V=v3, TH=th3, T=t3, W=w3, dz8w=dz3, Pcps=p3, pi=pi3, &
         W0AVG=w03, XLV0=XLV0, XLV1=XLV1, XLS0=XLS0, XLS1=XLS1, CP=cp, &
         R=r_d, G=g, EP1=EP_1, EP2=EP_2, SVP1=SVP1, SVP2=SVP2, SVP3=SVP3, &
         SVPT0=SVPT0, STEPCU=stepcu, CU_ACT_FLAG=cu_act_flag, &
         warm_rain=warm_rain, CUTOP=cutop, CUBOT=cubot, QV=qv3, &
         shall=shall, &
         F_QV=.TRUE., F_QC=.TRUE., F_QR=f_qr, F_QI=f_qi, F_QS=f_qs, &
         RTHCUTEN=rth, RQVCUTEN=rqv, RQCCUTEN=rqc, RQRCUTEN=rqr, &
         RQICUTEN=rqi, RQSCUTEN=rqs, RQVFTEN=rqvften, &
         cldfra_dp_KF=cdp, cldfra_sh_KF=csh, qc_KF=qck, qi_KF=qik, &
         UDR_KF=udr, DDR_KF=ddr, UER_KF=uer, DER_KF=der, &
         TIMEC_KF=timec_kf, KF_EDRATES=0)

    FLUSH(98, IOSTAT=nmsg)
    sz98_after = 0
    INQUIRE(FILE='fort.98', EXIST=ex98, SIZE=sz98_after)
    IF (.NOT. ex98) sz98_after = 0
    ifatal = 0
    IF (kf_oracle_fatal) ifatal = 1
    nmsg = kf_oracle_messages
    WRITE(11) nz, ifatal, nmsg, INT(sz98_after - sz98_before)
    WRITE(11) raincv(1,1), pratec(1,1), nca(1,1), cutop(1,1), cubot(1,1), shall(1,1)
    CALL wr(rth); CALL wr(rqv); CALL wr(rqc); CALL wr(rqr); CALL wr(rqi)
    CALL wr(rqs); CALL wr(w03); CALL wr(cdp); CALL wr(csh); CALL wr(qck)
    CALL wr(qik)
    DEALLOCATE(u3, v3, t3, th3, qv3, p3, dz3, rho3, pi3, w03, w3, rth, rqv, &
               rqc, rqr, rqi, rqs, rqvften, cdp, csh, qck, qik, udr, ddr, &
               uer, der, col)
  END DO
  CLOSE(10)
  CLOSE(11)
CONTAINS
  SUBROUTINE rd(a)
    REAL, INTENT(INOUT) :: a(:,:,:)
    READ(10) col(1:nz)
    a(1,1:nz,1) = col(1:nz)
  END SUBROUTINE rd
  SUBROUTINE wr(a)
    REAL, INTENT(IN) :: a(:,:,:)
    WRITE(11) a(1,1:nz,1)
  END SUBROUTINE wr
END PROGRAM run_kf_oracle

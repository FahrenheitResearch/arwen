! cu_ntiedtke_run's own body, cu_ntiedtke.F90:228-277 -- the conversion
! from cu_ntiedtke_pre_run's arrays into cumastrn's arguments.
!
! WHY THIS IS AN INCLUDE AND NOT TWO TRANSCRIPTIONS.  It was two: one in
! run_nt_cumastrn.F90, where the convergence proof against the real
! cu_ntiedtke_run covers it, and one in run_nt_cuinin.F90, where NOTHING
! covered it -- run_nt_cuinin has no comparison of any kind, so every CSV
! it records was recorded on a state graded by nothing.  A second copy is
! not merely duplication here; it is a copy outside the proof.
!
! cu_ntiedtke_run is public but this block is INSIDE it, so it cannot be
! called the way cu_ntiedtke_pre_run now is.  Transcription is genuinely
! forced.  One transcription, included twice, is the strongest available
! shape: the copy run_nt_cuinin uses is now the copy run_nt_cumastrn
! proves.
!
! It is the full :228-277 and not the subset run_nt_cuinin needs, because
! the scale_fac loop and the pgeoh(km1) assignment are the SAME do-loop in
! the source.  Splitting it would reorder statements, and statement order
! is the entire value of a transcription.  run_nt_cuinin therefore declares
! a few arrays it never reads; that is the cheaper of the two costs.
!
! Names assumed in scope: nc nz nz1 k n g_l vtmpc1_l tt zew zqs zcor and
! the arrays below.
dxref = 15000.
do n = 1, nc
  if (dx_hv(n) < dxref) then
    scale_fac(n) = (1.06133+log(dxref/dx_hv(n)))**3
    scale_fac2(n) = scale_fac(n)**0.5
  else
    scale_fac(n) = 1.+1.33e-5*dx_hv(n)
    scale_fac2(n) = 1.
  end if
  zrain(n)=0.0; locum(n)=0; prsfc(n)=0.0; pssfc(n)=0.0
  pqhfl(n)=qfx_hv(n); phhfl(n)=hfx_hv(n)
  pgeoh(n,nz1)=g_l*ghti(n,nz1)
end do
do k=1,nz
  do n=1,nc
    pcte(n,k)=0.0; pvom(n,k)=0.0; pvol(n,k)=0.0
    ztp1(n,k)=tf(n,k); zqp1(n,k)=qvf(n,k)/(1.0+qvf(n,k))
    pum1(n,k)=uf(n,k); pvm1(n,k)=vf(n,k)
    pverv(n,k)=omg(n,k); pgeo(n,k)=g_l*ghtl(n,k)
    pgeoh(n,k)=g_l*ghti(n,k)
    tt=ztp1(n,k)
    zew = nt_foeewm(tt)
    zqs = zew/prsl(n,k)
    zqs = min(0.5_kind_phys,zqs)
    zcor = 1./(1.-vtmpc1_l*zqs)
    zqsat(n,k)=zqs*zcor
    pqte(n,k)=qvftenz(n,k); zqq(n,k)=pqte(n,k)
    ptte(n,k)=thftenz(n,k); ztt(n,k)=ptte(n,k)
  end do
end do

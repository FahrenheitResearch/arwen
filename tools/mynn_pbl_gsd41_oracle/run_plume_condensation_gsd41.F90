! Exercise the source module's own condensation_edmf and phase blend.
program run_plume_condensation_gsd41
  use module_bl_mynn, only: condensation_edmf, qsat_blend
  use module_model_constants, only: rcp, p1000mb
  implicit none
  integer :: u, j, k, n
  real :: t, p, qt, thl, zagl, qc_in, qc, thv
  real, parameter :: temperatures(8) = [239., 245., 253., 254., 261.5, 267.15, 273.16, 280.]
  real, parameter :: factors(3) = [0.85, 1.05, 1.25]
  character(len=1024) :: output
  call get_command_argument(1, output)
  open(newunit=u, file=trim(output), status='replace', action='write')
  write(u,'(a)') 'case,qt,thl,p,zagl,qc_in,qc,thv'
  n = 0
  do j=1,8
    do k=1,3
      n = n+1
      t = temperatures(j)
      p = 80000. - real(j-1)*2000.
      qt = factors(k)*qsat_blend(t,p)
      thl = t/(p/p1000mb)**rcp
      zagl = 1000.
      if (j == 8 .and. k == 3) zagl = 50.
      qc_in = real(k-1)*0.0001
      qc = qc_in
      call condensation_edmf(qt,thl,p,zagl,thv,qc)
      write(u,'(i0,7(",",es24.16e3))') n,qt,thl,p,zagl,qc_in,qc,thv
    end do
  end do
  close(u)
end program run_plume_condensation_gsd41

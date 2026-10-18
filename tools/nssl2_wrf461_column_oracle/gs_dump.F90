! Diagnostic build only (instrument_gs.py): the stage dump of stage_dump.F90
! plus a per-gather-point dump of WRF's GS rates and source/sink sums.
!   $N2M_STAGE_DUMP  as in stage_dump.F90
!   $N2M_GS_DUMP     stream of records: int32 column, igs, kgs, n, then n float32
module n2m_gs_counter
  implicit none
  integer, save :: column = 0
end module n2m_gs_counter

subroutine n2m_stage_dump(stage, nx, nz, na, an, t0, t7, idx)
  use n2m_gs_counter, only: column
  implicit none
  integer, intent(in) :: stage, nx, nz, na, idx(17)
  real, intent(in) :: an(nx, 1, nz, na), t0(nx, 1, nz), t7(nx, 1, nz)
  integer, save :: u
  logical, save :: opened = .false.
  character(len=1024) :: path
  if (stage == 1) column = column + 1
  if (.not. opened) then
     call get_environment_variable('N2M_STAGE_DUMP', path)
     if (len_trim(path) == 0) return
     opened = .true.
     open(newunit=u, file=trim(path), access='stream', form='unformatted', &
          status='replace', action='write', convert='little_endian')
  end if
  write(u) stage, nx, nz, na, idx, an, t0, t7
end subroutine n2m_stage_dump

subroutine n2m_gs_dump(i, k, n, v)
  use n2m_gs_counter, only: column
  implicit none
  integer, intent(in) :: i, k, n
  real, intent(in) :: v(n)
  integer, save :: u
  logical, save :: opened = .false., off = .false.
  character(len=1024) :: path
  if (off) return
  if (.not. opened) then
     call get_environment_variable('N2M_GS_DUMP', path)
     if (len_trim(path) == 0) then
        off = .true.; return
     end if
     opened = .true.
     open(newunit=u, file=trim(path), access='stream', form='unformatted', &
          status='replace', action='write', convert='little_endian')
  end if
  write(u) column, i, k, n, v
end subroutine n2m_gs_dump

! WRF's preprocessed sources call wrf_error_fatal through the
! wrf_error_fatal3(file, line, message) form (inc/wrf_error_fatal.h).
subroutine wrf_error_fatal3(file_str, line, str)
  implicit none
  character(len=*), intent(in) :: file_str, str
  integer, intent(in) :: line
  print '(A,":",I0,": ",A)', trim(file_str), line, trim(str)
  error stop 1
end subroutine wrf_error_fatal3

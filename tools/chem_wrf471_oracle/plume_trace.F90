! Observer storage only. No value is read back by the pinned solver.
module plume_trace
  implicit none
  real :: trace_tops(2,4)=0.
  integer :: trace_steps(2,4)=0, trace_k1(4)=0,trace_k2(4)=0
  integer :: trace_group=1, trace_counter=0
contains
  subroutine trace_clear
    trace_tops=0.; trace_steps=0; trace_k1=0; trace_k2=0
    trace_group=1; trace_counter=0
  end subroutine
end module

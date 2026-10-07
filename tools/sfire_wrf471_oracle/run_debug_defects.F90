program run_debug_defects
use debug_reference,only:write_array_m3
implicit none
real,allocatable::edge(:,:,:)
integer::bound
character(len=32)::mode
call get_command_argument(1,mode)
bound=2147483646
if(trim(mode)=='bound-overflow')bound=2147483647
allocate(edge(16777217:16777217,bound:bound,-16777217:-16777217))
edge=transfer(int(z'3F123456'),0.0)
if(trim(mode)=='step-overflow')then
 call write_array_m3(16777217,16777217,bound,bound,-16777217,-16777217, &
  16777217,16777217,bound,bound,-16777217,-16777217,edge,'long_step',100000000)
else if(trim(mode)=='bound-overflow')then
 call write_array_m3(16777217,16777217,bound,bound,-16777217,-16777217, &
  16777217,16777217,bound,bound,-16777217,-16777217,edge,'bound_overflow',1)
else
 error stop 'Specify a debug defect control'
endif
end program run_debug_defects

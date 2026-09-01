
!======================================================================================!
! ORACLE ADDITION (not upstream): expose the private lookup-table module arrays so a
! driver can dump them.  Nothing above this point is modified.
 subroutine p3_oracle_dump(path)
   implicit none
   character(len=*), intent(in) :: path
   open(unit=71,file=trim(path)//"/itab.f32",form="unformatted",access="stream",status="replace")
   write(71) itab
   close(71)
   open(unit=71,file=trim(path)//"/itabcoll.f32",form="unformatted",access="stream",status="replace")
   write(71) itabcoll
   close(71)
   open(unit=71,file=trim(path)//"/vn_table.f32",form="unformatted",access="stream",status="replace")
   write(71) vn_table
   close(71)
   open(unit=71,file=trim(path)//"/vm_table.f32",form="unformatted",access="stream",status="replace")
   write(71) vm_table
   close(71)
   open(unit=71,file=trim(path)//"/revap_table.f32",form="unformatted",access="stream",status="replace")
   write(71) revap_table
   close(71)
   open(unit=71,file=trim(path)//"/mu_r_table.f32",form="unformatted",access="stream",status="replace")
   write(71) mu_r_table
   close(71)
   open(unit=71,file=trim(path)//"/itabcolli.f32",form="unformatted",access="stream",status="replace")
   write(71) itabcolli1
   write(71) itabcolli2
   close(71)
 end subroutine p3_oracle_dump


! Raw stream writer for chem/module_dep_simple.F column fixtures.
module oracle_io
 use iso_fortran_env,only:real32,int32
 implicit none
 character(len=512)::directory
 integer::manifest=0
 logical::has_manifest=.false.
 interface oracle_put
 module procedure put_r0
 module procedure put_r1
 module procedure put_r2
 module procedure put_r3
 module procedure put_i0
 module procedure put_i1
 module procedure put_i2
 module procedure put_i3
 end interface
contains
 subroutine oracle_open(case_name)
 character(len=*),intent(in)::case_name
 if(has_manifest) close(manifest)
 directory='fixtures/'//trim(case_name)
 call execute_command_line('mkdir -p '//trim(directory))
 open(newunit=manifest,file=trim(directory)//'/manifest.txt',status='replace')
 has_manifest=.true.
 end subroutine
 subroutine put_r0(name,array)
 character(len=*),intent(in)::name
 real(real32),intent(in)::array
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' float32'
 end subroutine
 subroutine put_r1(name,array)
 character(len=*),intent(in)::name
 real(real32),intent(in)::array(:)
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' float32',shape(array)
 end subroutine
 subroutine put_r2(name,array)
 character(len=*),intent(in)::name
 real(real32),intent(in)::array(:,:)
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' float32',shape(array)
 end subroutine
 subroutine put_r3(name,array)
 character(len=*),intent(in)::name
 real(real32),intent(in)::array(:,:,:)
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' float32',shape(array)
 end subroutine
 subroutine put_i0(name,array)
 character(len=*),intent(in)::name
 integer(int32),intent(in)::array
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' int32'
 end subroutine
 subroutine put_i1(name,array)
 character(len=*),intent(in)::name
 integer(int32),intent(in)::array(:)
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' int32',shape(array)
 end subroutine
 subroutine put_i2(name,array)
 character(len=*),intent(in)::name
 integer(int32),intent(in)::array(:,:)
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' int32',shape(array)
 end subroutine
 subroutine put_i3(name,array)
 character(len=*),intent(in)::name
 integer(int32),intent(in)::array(:,:,:)
 integer::unit
 open(newunit=unit,file=trim(directory)//'/'//trim(name)//'.bin', &
 access='stream',form='unformatted',convert='little_endian',status='replace')
 write(unit) array
 close(unit)
 write(manifest,*) trim(name),' int32',shape(array)
 end subroutine
end module

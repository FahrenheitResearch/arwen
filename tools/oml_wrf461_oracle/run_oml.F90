program run_oml
use module_sf_oml
use module_model_constants, only: stbolt, g
implicit none
integer, parameter :: n=112, nf=22, steps=16
real :: a(n,nf), tsk(n,1), tml(n,1),t0ml(n,1),hml(n,1),h0ml(n,1),huml(n,1),hvml(n,1),tmoml(n,1)
real :: hinit, depths(8)=[0.,1.e-8,0.01,0.5,5.,20.,50.,180.]
integer :: i,b,s,unit
character(1024) :: out,path
call get_command_argument(1,out)
do b=1,3
 a=0.
 do i=1,n
  tsk(i,1)=290.+real(mod(i,17))*0.7
  h0ml(i,1)=depths(mod(i-1,8)+1)
  huml(i,1)=0.01*real(mod(i,7)); hvml(i,1)=-0.02*real(mod(i,5))
  tmoml(i,1)=tsk(i,1)-5.
  if(mod(i,11)==0) tmoml(i,1)=200.5
  a(i,9)=50.+real(mod(i,5))*100. ! hfx
  a(i,10)=real(mod(i,7))*180. ! lh
  a(i,11)=real(mod(i,4))*400. ! gsw
  a(i,12)=300.+real(mod(i,6))*15. ! glw
  a(i,13)=real(mod(i,9))*10.; a(i,14)=-real(mod(i,6))*8.
  if(mod(i,8)==0) then
   a(i,13)=0.; a(i,14)=0.
  endif
  a(i,15)=0.2+real(mod(i,8))*0.6 ! ust
  a(i,16)=real(mod(i,3)-1)*1.e-4 ! f
  a(i,17)=0.98; a(i,18)=2. ! emiss,xland
  if(i>96) a(i,18)=1.
  a(i,19)=0.05+real(mod(i,4))*0.03 ! gamma
  a(i,20)=0.
  if(mod(i,2)==0) a(i,20)=86400.
  a(i,21)=30.+real(mod(i,4))*10. ! dt
  a(i,22)=stbolt ! stbolt
 enddo
 hinit=50.
 if(b==2) hinit=0.
 if(b==3) hinit=-1.
 a(:,1)=tsk(:,1); a(:,2)=tsk(:,1); a(:,3)=0.; a(:,4)=h0ml(:,1)
 a(:,5)=huml(:,1); a(:,6)=hvml(:,1); a(:,7)=tmoml(:,1); a(:,8)=tsk(:,1)
 write(path,'(A,"/branch",I1,"-input.bin")') trim(out),b
 open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='replace'); write(unit) a; close(unit)
 call omlinit(hinit,tsk,tml,t0ml,hml,h0ml,huml,hvml,tmoml,.true.,.true.,1,n+1,1,2,1,2,1,n,1,1,1,1,1,n,1,1,1,1)
 a(:,1)=tml(:,1); a(:,2)=t0ml(:,1); a(:,3)=hml(:,1); a(:,4)=h0ml(:,1)
 a(:,5)=huml(:,1); a(:,6)=hvml(:,1); a(:,7)=tmoml(:,1); a(:,8)=tsk(:,1)
 write(path,'(A,"/branch",I1,"-init.bin")') trim(out),b
 open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='replace'); write(unit) a; close(unit)
 ! Independent floor and near-zero depth cases after initialization.
 do i=1,n
  if(mod(i,13)==0) tml(i,1)=tmoml(i,1)-0.5
  if(mod(i,17)==0) tml(i,1)=tmoml(i,1)+1.e-4
 enddo
 a(:,1)=tml(:,1)
 write(path,'(A,"/branch",I1,"-start.bin")') trim(out),b
 open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='replace'); write(unit) a; close(unit)
 write(path,'(A,"/branch",I1,"-steps.bin")') trim(out),b
 open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='replace')
 do s=1,steps
  do i=1,n
   if(a(i,18)>1.5) call OML1D(i,1,tml(i,1),t0ml(i,1),hml(i,1),h0ml(i,1),huml(i,1),hvml(i,1),tsk(i,1), &
    a(i,9),a(i,10),a(i,11),a(i,12),tmoml(i,1),a(i,13),a(i,14),a(i,15),a(i,16),a(i,17),a(i,22),g,a(i,21),a(i,19),a(i,20), &
    1,n+1,1,2,1,2,1,n,1,1,1,1,1,n,1,1,1,1)
  enddo
  a(:,1)=tml(:,1); a(:,2)=t0ml(:,1); a(:,3)=hml(:,1); a(:,4)=h0ml(:,1)
  a(:,5)=huml(:,1); a(:,6)=hvml(:,1); a(:,7)=tmoml(:,1); a(:,8)=tsk(:,1)
  write(unit) a
 enddo
 close(unit)
enddo
end program
subroutine wrf_debug(level,message)
integer :: level
character(*) :: message
end subroutine

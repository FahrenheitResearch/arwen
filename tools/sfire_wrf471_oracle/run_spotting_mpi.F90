program run_spotting_mpi
use mpi
use module_domain
use module_dm
use module_firebrand_spotting_mpi
use oracle_io
implicit none
integer,parameter::n=18
type(domain)::grid
integer::rank,size,ierr,i,edge,source,dest
integer::pid(n),psrc(n),page(n),received(8),ids_list(8)
real::x(n),y(n),z(n),mass(n),diam(n),effd(n),temp(n),velocity(n),scalar
logical::mask(n)
real,allocatable::rx(:),ry(:),rz(:),rm(:),rd(:),re(:),rt(:),rv(:)
integer,allocatable::rid(:),rsrc(:),rage(:)
character(len=1024)::root
character(len=64)::name
call MPI_INIT(ierr)
call MPI_COMM_RANK(MPI_COMM_WORLD,rank,ierr)
call MPI_COMM_SIZE(MPI_COMM_WORLD,size,ierr)
if(size/=9)call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
mytask_x=mod(rank,3);mytask_y=rank/3
grid%ips=1+mytask_x*4;grid%ipe=grid%ips+3
grid%jps=1+mytask_y*4;grid%jpe=grid%jps+3
call fs_mpi_init(grid)
call oracle_root(root)
mask=.true.
do i=1,n
 edge=mod(i-1,9)
 x(i)=real(grid%ips)+1.375;y(i)=real(grid%jps)+1.625
 if(edge==0.or.edge==4.or.edge==6)x(i)=real(grid%ips)-.25
 if(edge==1.or.edge==5.or.edge==7)x(i)=real(grid%ipe)+1.25
 if(edge==2.or.edge==4.or.edge==5)y(i)=real(grid%jpe)+1.375
 if(edge==3.or.edge==6.or.edge==7)y(i)=real(grid%jps)-.375
 z(i)=17.5+.125*real(i)
 mass(i)=.1+.001*real(i+rank);diam(i)=10.+.015625*real(i)
 effd(i)=7.5+.0078125*real(i);temp(i)=850.+real(i+rank)
 velocity(i)=.125*real(i)
 pid(i)=rank*100+i;psrc(i)=rank*1000000+pid(i);page(i)=i+rank
enddo
mask(18)=.false.
write(name,'(a,i0)')'spotting_mpi/rank_',rank
call oracle_open(trim(name))
call oracle_put('rank',rank);call oracle_put('tile',[grid%ips,grid%ipe,grid%jps,grid%jpe])
call oracle_put('neighbors',task_id);call oracle_put('mask',merge(1,0,mask))
call oracle_put('x',x);call oracle_put('y',y);call oracle_put('z',z)
call oracle_put('mass',mass);call oracle_put('diam',diam);call oracle_put('effd',effd)
call oracle_put('temp',temp);call oracle_put('velocity',velocity)
call oracle_put('id',pid);call oracle_put('source',psrc);call oracle_put('life',page)
! Small native packets use eager MPI sends; all eight routing directions
! and edge/corner neighbor absences are exercised on the 3x3 topology.
call fs_mpi_send2neighbors(task_id,mask,x,y,z,pid,psrc,page,mass,diam,effd,temp,velocity)
received=fs_mpi_checkreceive(task_id,8)
call oracle_put('received_counts',received)
allocate(rx(sum(received)),ry(sum(received)),rz(sum(received)),rm(sum(received)),rd(sum(received)), &
 re(sum(received)),rt(sum(received)),rv(sum(received)),rid(sum(received)),rsrc(sum(received)),rage(sum(received)))
call fs_mpi_recv(received,task_id,rx,ry,rz,rm,rd,re,rt,rv,rid,rsrc,rage)
call oracle_put('received_x',rx);call oracle_put('received_y',ry);call oracle_put('received_z',rz)
call oracle_put('received_mass',rm);call oracle_put('received_diam',rd);call oracle_put('received_effd',re)
call oracle_put('received_temp',rt);call oracle_put('received_velocity',rv)
call oracle_put('received_id',rid);call oracle_put('received_source',rsrc);call oracle_put('received_life',rage)
! Native scalar message wrappers used by generation and history reduction.
source=mod(rank+8,9);dest=mod(rank+1,9)
call fs_mpi_sendbuff1_real(real(rank)+.375,dest)
scalar=fs_mpi_recvbuff1_real(source)
call oracle_put('received_scalar',scalar)
call fs_mpi_sendbuff1_int(rank+17,dest)
i=fs_mpi_recvbuff1_int(source)
call oracle_put('received_integer',i)
call oracle_close()
call MPI_FINALIZE(ierr)
end program

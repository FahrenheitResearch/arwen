program test_containment
 use native_geometry
 implicit none
 integer::q,i
 real::r,lat,lon,p(3),cell(3),a(3),b(3),v(3,3),mirror(3),out(5)
 logical::inside
 r=6371229.0
 do q=0,4095
   lat=-1.4+real(mod(q*31,2801))*0.001
   lon=real(mod(q*71,6281))*0.001
   call mpas_latlon_to_xyz(cell(1),cell(2),cell(3),r,lat,lon)
   call mpas_latlon_to_xyz(a(1),a(2),a(3),r,lat+0.01,lon-0.02)
   call mpas_latlon_to_xyz(b(1),b(2),b(3),r,lat-0.02,lon+0.01)
   call mpas_latlon_to_xyz(p(1),p(2),p(3),r,lat+real(mod(q,41)-20)*0.001,lon+0.007)
   v(:,1)=a
   v(:,2)=b
   call mpas_latlon_to_xyz(v(1,3),v(2,3),v(3,3),r,lat-0.015,lon-0.03)
   out(1)=mpas_arc_length(p(1),p(2),p(3),cell(1),cell(2),cell(3))
   out(2)=mpas_sphere_angle(a(1)/r,a(2)/r,a(3)/r,cell(1)/r,cell(2)/r,cell(3)/r,b(1)/r,b(2)/r,b(3)/r)
   call mpas_mirror_point(cell(1)/r,cell(2)/r,cell(3)/r,a(1)/r,a(2)/r,a(3)/r,b(1)/r,b(2)/r,b(3)/r,mirror(1),mirror(2),mirror(3))
   out(3:5)=mirror
   inside=mpas_in_cell(p(1),p(2),p(3),cell(1),cell(2),cell(3),3,[1,2,3],v(1,:),v(2,:),v(3,:))
   write(*,'(I5,1X,5(Z8.8,1X),I1)')q,(transfer(out(i),0),i=1,5),merge(1,0,inside)
 end do
end program

! Calls pristine MPAS helpers copied verbatim into native_geometry.
program support_words
 use native_geometry
 implicit none
 integer,parameter::nc=16,me=4,nv=64
 integer::q,c,e,v,owner,result,bits
 integer::ne(nc),neighbors(me,nc),vertices(me,nc)
 real::lat(nc),lon(nc),vl(nv),vo(nv),tl,to,radius,diameter
 do c=1,nc
  lat(c)=real(c-8)*0.03
  lon(c)=real(c)*0.31-3.0
  ne(c)=2+modulo(c,3)
  neighbors(:,c)=(/1+modulo(c,nc),1+modulo(c-2,nc),1+modulo(c+1,nc),nc+1/)
  do e=1,me
   v=(c-1)*me+e
   vertices(e,c)=v
   vl(v)=lat(c)+real(e-2)*0.007
   vo(v)=lon(c)+real(e-3)*0.013
  end do
 end do
 do q=0,2047
  tl=-0.25+real(modulo(q*31,701))*0.0007
  to=-3.0+real(modulo(q*17,1257))*0.005
  result=nearest_cell(tl,to,1+modulo(q,nc),nc,me,ne,neighbors,lat,lon)
  owner=modulo(q,17)
  radius=6371229.0*(1.0+real(modulo(q,11))*0.001)
  diameter=max_cell_diameter(owner,ne,vertices,lat,lon,vl,vo,radius)
  write(*,'(i0,1x,i0,1x,z8.8)') q,result,transfer(diameter,bits)
 end do
end program

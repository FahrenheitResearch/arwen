! MPAS routines called here are copied unchanged from the pinned sources.
program geometry_words
 use native_geometry
 implicit none
 type(mpas_geotile_mgr_type)::mgr
 type(mpas_geotile_type),target::t
 type(mpas_geotile_type),pointer::tile
 integer::q,s,i,j,k,bits
 real::lat,lon,x,y,z,d
 tile=>t
 mgr%pool=0
 do q=0,4095
  s=1+modulo(q,7)
  t%x=1+1200*modulo(q*17,36)
  t%y=1+1200*modulo(q*11,18)
  i=modulo(q*43,1200)*s+modulo(q*71,s)
  j=modulo(q*59,1200)*s+modulo(q*37,s)
  origin_lat=-89.995833
  origin_lon=-179.99583
  step_x=0.00833333
  step_y=0.00833333
  if(modulo(q,2)==0) then
   origin_lat=89.975
   origin_lon=-179.975
   step_x=0.05
   step_y=-0.05
   t%x=1+1200*modulo(q*17,6)
   t%y=1+1200*modulo(q*11,3)
  end if
  call mpas_geotile_mgr_tile_to_latlon(mgr,tile,j+s*border+1,i+s*border+1,lat,lon,s)
  call mpas_latlon_to_xyz(x,y,z,6371229.,lat,lon)
  d=sphere_distance(lat,lon,lat+0.01,lon+0.02,6371229.)
  write(*,'(i0,1x,6(z8.8,1x))') q,transfer(lat,bits),transfer(lon,bits),transfer(x,bits),transfer(y,bits),transfer(z,bits),transfer(d,bits)
 end do
end program

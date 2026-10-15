! Native NCEP/soil-temperature driver blocks are appended verbatim by regenerate.py.
program climatology
 use mpas_kind_types
 use mpas_log
 use iso_c_binding
 use init_atm_hinterp
 use init_atm_llxy, only:proj_info,map_set,PROJ_LATLON,latlon_to_ij
 implicit none
 integer,parameter::nCells=512
 real,parameter::DEG_PER_RAD=180.0/3.141592653589793
 real::latCell(nCells),lonCell(nCells),greenfrac(12,nCells),albedo12m(12,nCells)
 real::soiltemp(nCells),snoalb(nCells),shdmin(nCells),shdmax(nCells)
 integer::landmask(nCells),iCell,k,q,u
 real::lat,lon,x,y,scalefactor
 real(c_float),pointer,contiguous::rarray(:,:,:)
 real,allocatable::vegfra(:,:,:),maxsnowalb(:,:),soiltemp_1deg(:,:)
 type(proj_info)::proj
 type(c_ptr)::rarray_ptr
 integer(c_int)::nx,ny,nz,isigned,endian,wordsize,istatus
 integer::interp_list(5)
 character(512)::fname
 character(513)::geog_data_path
 character(c_char)::c_fname(513)
 interface
  subroutine read_geogrid(fname,rarray,nx,ny,nz,isigned,endian,wordsize,status)bind(C)
   import c_char,c_ptr,c_int
   character(c_char)::fname(*)
   type(c_ptr),value::rarray
   integer(c_int),value::nx,ny,nz,isigned,endian,wordsize
   integer(c_int)::status
  end subroutine
 end interface
 geog_data_path='./'
 interp_list=[FOUR_POINT,W_AVERAGE4,W_AVERAGE16,SEARCH,0]
 do q=0,nCells-1
  latCell(q+1)=(-90.0+real(mod(q*1739,180001))*0.001)/DEG_PER_RAD
  lonCell(q+1)=(-181.0+real(mod(q*2377,363001))*0.001)/DEG_PER_RAD
  if(q<16)then
   latCell(q+1)=real(q-8)*0.2/DEG_PER_RAD
   lonCell(q+1)=real(q-8)*0.1/DEG_PER_RAD
  endif
  landmask(q+1)=merge(0,1,mod(q,7)==0)
 enddo

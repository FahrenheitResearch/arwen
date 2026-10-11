//! Focused Rust extractor for source-column oracle inputs. Numeric decoding
//! is native NetCDF, with Rust owning bounds, windows, layout and output.
use std::{ffi::{CString,CStr},fs,io::Write,path::Path};
#[link(name="netcdf")]
unsafe extern "C" {
    fn nc_open(p:*const i8,mode:i32,id:*mut i32)->i32;
    fn nc_close(id:i32)->i32;
    fn nc_inq_varid(id:i32,n:*const i8,v:*mut i32)->i32;
    fn nc_inq_varndims(id:i32,v:i32,n:*mut i32)->i32;
    fn nc_inq_vardimid(id:i32,v:i32,d:*mut i32)->i32;
    fn nc_inq_dimlen(id:i32,d:i32,n:*mut usize)->i32;
    fn nc_get_vara_float(id:i32,v:i32,s:*const usize,c:*const usize,a:*mut f32)->i32;
    fn nc_strerror(code:i32)->*const i8;
}
fn check(code:i32) {if code!=0 {panic!("NetCDF: {}",unsafe{CStr::from_ptr(nc_strerror(code))}.to_string_lossy());}}
fn main() {
    let args:Vec<String>=std::env::args().collect();
    assert_eq!(args.len(),4,"column_reader SOURCE OUTPUT_DIRECTORY input|restart");
    let path=CString::new(args[1].as_str()).unwrap();
    let dest=Path::new(&args[2]);fs::create_dir_all(dest).unwrap();
    let restart=args[3]=="restart";
    let fields=["QVAPOR","QCLOUD","QICE","QRAIN","QSNOW","QGRAUP","QNICE","QNRAIN","P","PB","T",if restart {"W_2"} else {"W"},if restart {"PH_2"} else {"PH"},"PHB"];
    let mut id=0;check(unsafe{nc_open(path.as_ptr(),0,&mut id)});
    let mut metadata=String::from("[");
    for (f,name) in fields.iter().enumerate() {
        let key=CString::new(*name).unwrap();let mut v=0;
        check(unsafe{nc_inq_varid(id,key.as_ptr(),&mut v)});
        let mut rank=0;check(unsafe{nc_inq_varndims(id,v,&mut rank)});assert!(rank==3||rank==4);
        let mut dims=vec![0;rank as usize];check(unsafe{nc_inq_vardimid(id,v,dims.as_mut_ptr())});
        let mut shape=vec![0usize;rank as usize];
        for (d,len) in dims.iter().zip(shape.iter_mut()){check(unsafe{nc_inq_dimlen(id,*d,len)});}
        if rank==4 {assert_eq!(shape[0],1,"only one source time is admitted");}
        let n=shape.len();let (nz,ny,nx)=(shape[n-3],shape[n-2],shape[n-1]);
        let mut output=vec![0.0f32;256*nz];let mut window=0;
        for fy in [0.15,0.38,0.62,0.85] {for fx in [0.15,0.38,0.62,0.85] {
            let j=(ny as f64*fy) as usize;let i=(nx as f64*fx) as usize;
            assert!(j+4<=ny&&i+4<=nx);
            let mut start=vec![0usize;n];start[n-2]=j;start[n-1]=i;
            let mut count=vec![1usize;n];count[n-3]=nz;count[n-2]=4;count[n-1]=4;
            let mut buf=vec![0.0f32;nz*16];
            check(unsafe{nc_get_vara_float(id,v,start.as_ptr(),count.as_ptr(),buf.as_mut_ptr())});
            for k in 0..nz {for cell in 0..16 {output[(window*16+cell)*nz+k]=buf[k*16+cell];}}
            window+=1;
        }}
        let mut file=fs::File::create(dest.join(format!("{name}.f32"))).unwrap();
        for value in output {file.write_all(&value.to_le_bytes()).unwrap();}
        if f>0{metadata.push(',');}
        metadata.push_str(&format!("{{\"name\":\"{name}\",\"columns\":256,\"levels\":{nz},\"ny\":{ny},\"nx\":{nx}}}"));
        println!("read {name}: 256 x {nz}");
    }
    check(unsafe{nc_close(id)});metadata.push(']');
    fs::write(dest.join("metadata.json"),metadata).unwrap();
}

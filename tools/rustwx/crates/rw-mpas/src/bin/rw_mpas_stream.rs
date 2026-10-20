//! Serial MPAS native streams, with raw primitive-word handoff to the driver.
//!
//! MPAS v8.4.2 src/core_atmosphere/Registry.xml:498-1349 defines membership;
//! src/framework/mpas_io_streams.F:2430-3927 defines native file layout. Dimension order
//! is the reverse of Registry/Fortran order, including StrLen for text. This
//! bridge owns file decoding, encoding, permutation and connectivity conversion.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

// Built only with the `system-netcdf` feature (Cargo.toml required-features),
// which links the system netCDF C library; a default `cargo test -p rw-mpas`
// builds neither this binary nor the link.
#[cfg_attr(feature = "system-netcdf", link(name = "netcdf"))]
unsafe extern "C" {
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_create(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc__enddef(id: c_int, header_free: usize, variable_alignment: usize,
                 variable_free: usize, record_alignment: usize) -> c_int;
    fn nc_def_dim(id: c_int, name: *const c_char, len: usize, dim: *mut c_int) -> c_int;
    fn nc_def_var(id: c_int, name: *const c_char, kind: c_int, nd: c_int, dims: *const c_int, var: *mut c_int) -> c_int;
    fn nc_put_vara(id: c_int, var: c_int, start: *const usize, count: *const usize, data: *const c_void) -> c_int;
    fn nc_get_vara(id: c_int, var: c_int, start: *const usize, count: *const usize, data: *mut c_void) -> c_int;
    fn nc_put_att(id: c_int, var: c_int, name: *const c_char, kind: c_int, len: usize, data: *const c_void) -> c_int;
    fn nc_put_att_text(id: c_int, var: c_int, name: *const c_char, len: usize, data: *const c_char) -> c_int;
    fn nc_get_att(id: c_int, var: c_int, name: *const c_char, data: *mut c_void) -> c_int;
    fn nc_inq(id: c_int, nd: *mut c_int, nv: *mut c_int, na: *mut c_int, ud: *mut c_int) -> c_int;
    fn nc_inq_format(id: c_int, format: *mut c_int) -> c_int;
    fn nc_inq_unlimdims(id: c_int, n: *mut c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_inq_var(id: c_int, var: c_int, name: *mut c_char, kind: *mut c_int, nd: *mut c_int, dims: *mut c_int, na: *mut c_int) -> c_int;
    fn nc_inq_attname(id: c_int, var: c_int, index: c_int, name: *mut c_char) -> c_int;
    fn nc_inq_att(id: c_int, var: c_int, name: *const c_char, kind: *mut c_int, len: *mut usize) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
}
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn check(code: c_int) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy().into_owned().into()) }
}
fn c(text: &str) -> Result<CString> { Ok(CString::new(text)?) }
struct Nc(c_int);
impl Drop for Nc { fn drop(&mut self) { unsafe { nc_close(self.0); } } }
impl Nc {
    fn open(path: &Path, mode: c_int) -> Result<Self> {
        let mut id = 0; check(unsafe { nc_open(c(&path.to_string_lossy())?.as_ptr(), mode, &mut id) })?; Ok(Self(id))
    }
    fn create(path: &Path, format: &str) -> Result<Self> {
        let flag = match format { "netcdf4" => 0x1000, "pnetcdf,cdf5" => 0x20,
            "netcdf" | "pnetcdf" => 0x200, _ => return Err(format!("unknown file format {format}").into()) };
        let mut id = 0; check(unsafe { nc_create(c(&path.to_string_lossy())?.as_ptr(), flag | 4, &mut id) })?; Ok(Self(id))
    }
    fn close(self) -> Result<()> { let id = self.0; std::mem::forget(self); check(unsafe { nc_close(id) }) }
}
fn dtype(kind: i32) -> Result<(&'static str, usize)> {
    Ok(match kind { 1 => ("i1",1), 2 => ("S1",1), 3 => ("i2",2), 4 => ("i4",4),
        5 => ("f4",4), 6 => ("f8",8), 7 => ("u1",1), 8 => ("u2",2), 9 => ("u4",4),
        10 => ("i8",8), 11 => ("u8",8), _ => return Err(format!("unsupported NetCDF primitive {kind}").into()) })
}
fn kind(dtype: &str) -> Result<i32> {
    for k in 1..=11 { if self::dtype(k)?.0 == dtype { return Ok(k); } }
    Err(format!("unsupported handoff dtype {dtype}").into())
}
// Eight-byte alignment for libnetcdf's raw typed access, including doubles.
fn aligned(bytes: usize) -> Vec<u64> { vec![0; bytes.div_ceil(8)] }
fn bytes(buf: &[u64], len: usize) -> &[u8] { unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(),len) } }
fn bytes_mut(buf: &mut [u64], len: usize) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(),len) } }
fn product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |a,b| a.checked_mul(*b).ok_or_else(|| "array shape overflows usize".into()))
}
fn text_name(buf: &[c_char]) -> String { unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned() }

#[derive(Deserialize)]
struct Request {
    operation: String,
    path: PathBuf,
    #[serde(default = "default_format")] format: String,
    #[serde(default)] dimensions: BTreeMap<String,usize>,
    #[serde(default)] dimension_order: Vec<String>,
    #[serde(default)] global_attribute_order: Vec<String>,
    #[serde(default)] header_alignment: usize,
    #[serde(default)] variables: Vec<Variable>,
    #[serde(default)] global_attrs: BTreeMap<String,Value>,
    #[serde(default)] output_directory: PathBuf,
    #[serde(default)] select: Vec<String>,
    #[serde(default)] record: Option<usize>,
    #[serde(default)] record_index: Option<usize>,
    #[serde(default)] append: bool,
    #[serde(default)] layouts: BTreeMap<String,ReadLayout>,
}
#[derive(Deserialize)]
struct ReadLayout {
    dtype: String,
    dimensions: Vec<String>,
    expected_dimensions: Vec<String>,
    #[serde(default)] connectivity_offset: i32,
}
fn default_format() -> String { "netcdf".into() }
#[derive(Serialize,Deserialize,Clone)]
struct Variable {
    name: String,
    dtype: String,
    dimensions: Vec<String>,
    #[serde(default)] shape: Vec<usize>,
    #[serde(default)] input_dimensions: Vec<String>,
    #[serde(default)] filename: PathBuf,
    #[serde(default)] attrs: BTreeMap<String,Value>,
    #[serde(default)] attribute_order: Vec<String>,
    #[serde(default)] text: Vec<String>,
    #[serde(default)] connectivity_offset: i32,
    #[serde(default)] input_dtype: Option<String>,
}

// MPAS v8.4.2 src/framework/mpas_io_streams.F:4086-4244: native primitive/layout semantics.
fn encode_attr(value: &Value) -> Result<(i32, Vec<u8>)> {
    let (k, v) = if let Some(obj) = value.as_object() {
        (kind(obj["dtype"].as_str().ok_or("attribute dtype missing")?)?, obj.get("values").ok_or("attribute values missing")?)
    } else { (if value.is_i64() {4} else {6},value) };
    let values = if let Some(a) = v.as_array() { a.clone() } else { vec![v.clone()] };
    let mut out = Vec::new();
    for x in values {
        macro_rules! signed { ($t:ty) => { out.extend_from_slice(&<$t>::try_from(x.as_i64().ok_or("integer attribute required")?)?.to_ne_bytes()) } }
        macro_rules! unsigned { ($t:ty) => { out.extend_from_slice(&<$t>::try_from(x.as_u64().ok_or("unsigned attribute required")?)?.to_ne_bytes()) } }
        match k { 1 => signed!(i8), 3 => signed!(i16),4 => signed!(i32),10 => signed!(i64),
            7 => unsigned!(u8),8 => unsigned!(u16),9 => unsigned!(u32),11 => unsigned!(u64),
            5 => out.extend_from_slice(&(x.as_f64().ok_or("real attribute required")? as f32).to_ne_bytes()),
            6 => out.extend_from_slice(&x.as_f64().ok_or("real attribute required")?.to_ne_bytes()),
            _ => return Err("unsupported attribute primitive".into()) }
    }
    Ok((k,out))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:4315-4363: native primitive/layout semantics.
fn ordered_keys<'a,T>(values: &'a BTreeMap<String,T>, order: &'a [String]) -> Result<Vec<&'a String>> {
    if order.is_empty() {return Ok(values.keys().collect());}
    let keys:std::collections::BTreeSet<_>=order.iter().collect();
    if keys.len()!=order.len() || keys!=values.keys().collect() {
        return Err("explicit metadata order must list each key exactly once".into());
    }
    Ok(order.iter().collect())
}
fn put_attrs(nc: &Nc, var: i32, attrs: &BTreeMap<String,Value>, order: &[String], variable_kind: Option<i32>) -> Result<()> {
    for name in ordered_keys(attrs,order)? {
        let value=&attrs[name];
        if name == "_NCProperties" { continue; }
        let cname = c(name)?;
        if let Some(s) = value.as_str() {
            check(unsafe { nc_put_att_text(nc.0,var,cname.as_ptr(),s.len(),s.as_ptr().cast()) })?;
        } else {
            // _FillValue must have exactly the variable's primitive type.
            let v = if name == "_FillValue" && !value.is_object() {
                json!({"dtype":dtype(variable_kind.ok_or("global fill value")?)?.0,"values":value})
            } else {value.clone()};
            let (k,raw)=encode_attr(&v)?; let len=raw.len()/dtype(k)?.1;
            let mut buf=aligned(raw.len()); bytes_mut(&mut buf,raw.len()).copy_from_slice(&raw);
            check(unsafe { nc_put_att(nc.0,var,cname.as_ptr(),k,len,buf.as_ptr().cast()) })?;
        }
    }
    Ok(())
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:3930-4083,4315-4363: native primitive/layout semantics.
fn get_attrs(nc: &Nc, var: i32, n: i32) -> Result<BTreeMap<String,Value>> {
    let mut attrs=BTreeMap::new();
    for i in 0..n {
        let mut name=[0;257]; check(unsafe { nc_inq_attname(nc.0,var,i,name.as_mut_ptr()) })?;
        let mut k=0; let mut len=0; check(unsafe { nc_inq_att(nc.0,var,name.as_ptr(),&mut k,&mut len) })?;
        let size=dtype(k)?.1; let mut buf=aligned(len.checked_mul(size).ok_or("attribute size overflow")?);
        check(unsafe { nc_get_att(nc.0,var,name.as_ptr(),buf.as_mut_ptr().cast()) })?;
        let raw=bytes(&buf,len*size);
        let value=if k==2 { Value::String(String::from_utf8(raw.to_vec())?) } else {
            let mut values=Vec::new();
            for chunk in raw.chunks_exact(size) {
                macro_rules! decode { ($t:ty) => { json!(<$t>::from_ne_bytes(chunk.try_into()?)) } }
                values.push(match k {1=>decode!(i8),3=>decode!(i16),4=>decode!(i32),5=>decode!(f32),6=>decode!(f64),
                    7=>decode!(u8),8=>decode!(u16),9=>decode!(u32),10=>decode!(i64),11=>decode!(u64),_=>unreachable!()});
            }
            json!({"dtype":dtype(k)?.0,"values":values})
        };
        attrs.insert(text_name(&name),value);
    }
    Ok(attrs)
}

// Compare typed attribute words and order without float/JSON conversions.
fn attribute_layout(nc: &Nc, var: i32, n: i32) -> Result<(Vec<String>,BTreeMap<String,Value>)> {
    let mut order=Vec::new();let mut words=BTreeMap::new();
    for i in 0..n {
        let mut name=[0;257];check(unsafe {nc_inq_attname(nc.0,var,i,name.as_mut_ptr())})?;
        let mut k=0;let mut len=0;check(unsafe {nc_inq_att(nc.0,var,name.as_ptr(),&mut k,&mut len)})?;
        let size=len.checked_mul(dtype(k)?.1).ok_or("attribute size overflow")?;
        let mut data=aligned(size);check(unsafe {nc_get_att(nc.0,var,name.as_ptr(),data.as_mut_ptr().cast())})?;
        let name=text_name(&name);order.push(name.clone());
        words.insert(name,json!({"dtype":dtype(k)?.0,"length":len,
            "hex":bytes(&data,size).iter().map(|v|format!("{v:02x}")).collect::<String>()}));
    }
    Ok((order,words))
}

// MPAS v8.4.2 src/framework/mpas_io_streams.F:1659-1857,2430-3927: native primitive/layout semantics.
fn prepare(v: &Variable, dims: &BTreeMap<String,usize>) -> Result<Vec<u64>> {
    if v.dimensions.iter().collect::<std::collections::BTreeSet<_>>().len()!=v.dimensions.len() {
        return Err(format!("{}: duplicate output axis",v.name).into());
    }
    let out_shape: Vec<_>=v.dimensions.iter().map(|d| dims.get(d).copied().ok_or_else(|| format!("{}: missing dimension {d}",v.name))).collect::<std::result::Result<_,_>>()?;
    let count=product(&out_shape)?; let out_kind=kind(&v.dtype)?; let size=dtype(out_kind)?.1;
    let mut out=aligned(count.checked_mul(size).ok_or("variable size overflow")?);
    if out_kind==2 && !v.text.is_empty() {
        if v.dimensions.last().map(String::as_str)!=Some("StrLen") || out_shape.last()!=Some(&64) || count/64!=v.text.len() {
            return Err(format!("{}: text shape or record count mismatch",v.name).into());
        }
        for (target,text) in bytes_mut(&mut out,count).chunks_exact_mut(64).zip(&v.text) {
            if !text.is_ascii() || text.len()>64 { return Err("MPAS text must fit 64 ASCII characters".into()); }
            target.fill(b' '); target[..text.len()].copy_from_slice(text.as_bytes());
        }
        return Ok(out);
    }
    let input_kind=kind(v.input_dtype.as_deref().unwrap_or(&v.dtype))?;
    let input_size=dtype(input_kind)?.1;
    if v.input_dimensions.len()!=v.shape.len() { return Err("input dimensions/shape mismatch".into()); }
    let mut unique=std::collections::BTreeSet::new();
    for (d,n) in v.input_dimensions.iter().zip(&v.shape) {
        if !unique.insert(d) || !v.dimensions.contains(d) || dims.get(d)!=Some(n) { return Err(format!("{}: incompatible input axis {d}",v.name).into()); }
    }
    if v.dimensions.iter().any(|d| d!="Time" && !unique.contains(d)) { return Err(format!("{}: missing input axis",v.name).into()); }
    let raw=fs::read(&v.filename)?;
    if raw.len()!=product(&v.shape)?.checked_mul(input_size).ok_or("input size overflow")? { return Err(format!("{}: raw handoff byte count mismatch",v.name).into()); }
    let target=bytes_mut(&mut out,count*size);
    for flat in 0..count {
        let mut residual=flat; let mut coords=vec![0;out_shape.len()];
        for axis in (0..coords.len()).rev() { coords[axis]=residual%out_shape[axis]; residual/=out_shape[axis]; }
        let mut input_index=0;
        for (d,n) in v.input_dimensions.iter().zip(&v.shape) {
            input_index=input_index*n+coords[v.dimensions.iter().position(|a| a==d).ok_or("input axis absent")?];
        }
        let src=&raw[input_index*input_size..(input_index+1)*input_size]; let dest=&mut target[flat*size..(flat+1)*size];
        if v.connectivity_offset!=0 {
            if ![4,10].contains(&input_kind) || out_kind!=4 { return Err("connectivity requires signed int32/int64 input and int32 output".into()); }
            let value=if input_kind==4 {i32::from_ne_bytes(src.try_into()?) as i64} else {i64::from_ne_bytes(src.try_into()?)};
            let disk=i32::try_from(value.checked_add(v.connectivity_offset as i64).ok_or("connectivity overflow")?)?;
            dest.copy_from_slice(&disk.to_ne_bytes());
        } else if input_kind==out_kind { dest.copy_from_slice(src); }
        else if input_kind==10 && out_kind==4 {dest.copy_from_slice(&i32::try_from(i64::from_ne_bytes(src.try_into()?))?.to_ne_bytes());}
        else if input_kind==7 && out_kind==4 {dest.copy_from_slice(&(src[0] as i32).to_ne_bytes());}
        else if input_kind==5 && out_kind==6 { dest.copy_from_slice(&(f32::from_ne_bytes(src.try_into()?) as f64).to_ne_bytes()); }
        else if input_kind==6 && out_kind==5 { dest.copy_from_slice(&(f64::from_ne_bytes(src.try_into()?) as f32).to_ne_bytes()); }
        else { return Err(format!("{}: unsupported primitive conversion",v.name).into()); }
    }
    Ok(out)
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:3254-3927: native primitive/layout semantics.
fn write(req: &Request) -> Result<Value> {
    // Validate and prepare every variable before touching the destination.
    let mut names=std::collections::BTreeSet::new();
    let mut prepared=Vec::new();
    for v in &req.variables { if !names.insert(&v.name) { return Err("duplicate variable".into()); } prepared.push(prepare(v,&req.dimensions)?); }
    let nc=if req.append {Nc::open(&req.path,1)?} else {Nc::create(&req.path,&req.format)?};
    let mut dim_ids=BTreeMap::new();
    if !req.append {
        for name in ordered_keys(&req.dimensions,&req.dimension_order)? {
            let len=&req.dimensions[name];
            let mut id=0; check(unsafe {nc_def_dim(nc.0,c(name)?.as_ptr(),if name=="Time" {0} else {*len},&mut id)})?;
            dim_ids.insert(name.clone(),id);
        }
        put_attrs(&nc,-1,&req.global_attrs,&req.global_attribute_order,None)?;
    }
    let mut ids=Vec::new();
    for v in &req.variables {
        let mut id=0;
        if req.append {
            check(unsafe {nc_inq_varid(nc.0,c(&v.name)?.as_ptr(),&mut id)})?;
            let (k,dimensions,shape,_)=var_info(&nc,id)?;
            if k!=kind(&v.dtype)? || dimensions!=v.dimensions { return Err(format!("{}: append layout mismatch",v.name).into()); }
            for (axis,extent) in dimensions.iter().zip(shape) {
                if axis!="Time" && req.dimensions.get(axis)!=Some(&extent) {
                    return Err(format!("{}: append extent mismatch for {axis}",v.name).into());
                }
            }
        } else {
            let dims:Vec<_>=v.dimensions.iter().map(|d| dim_ids[d]).collect();
            check(unsafe {nc_def_var(nc.0,c(&v.name)?.as_ptr(),kind(&v.dtype)?,dims.len() as i32,dims.as_ptr(),&mut id)})?;
            put_attrs(&nc,id,&v.attrs,&v.attribute_order,Some(kind(&v.dtype)?))?;
        }
        ids.push(id);
    }
    if !req.append {
        if req.header_alignment > 0 {
            // SMIOL smiol.c:15,341-344 sets nc_header_align_size=131072.
            // Only the first data section is aligned. Subsequent fixed and
            // record variables retain their native four-byte packing.
            check(unsafe {nc__enddef(nc.0,0,req.header_alignment,0,4)})?;
        } else {check(unsafe {nc_enddef(nc.0)})?;}
    }
    let mut bytes_written=0;
    for ((v,id),data) in req.variables.iter().zip(ids).zip(prepared) {
        if req.append && !v.dimensions.iter().any(|d|d=="Time") {continue;}
        let mut start=vec![0;v.dimensions.len()];
        let count:Vec<_>=v.dimensions.iter().map(|d|req.dimensions[d]).collect();
        if let Some(axis)=v.dimensions.iter().position(|d|d=="Time") {start[axis]=req.record_index.unwrap_or(0);}
        check(unsafe {nc_put_vara(nc.0,id,start.as_ptr(),count.as_ptr(),data.as_ptr().cast())})?;
        bytes_written+=product(&count)?*dtype(kind(&v.dtype)?)?.1;
    }
    nc.close()?;
    Ok(json!({"schema":"mpas-native-stream/v1","variables":req.variables.len(),"bytes_written":bytes_written,"ordered_metadata":true}))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:2430-3251: native primitive/layout semantics.
fn var_info(nc: &Nc,id:i32) -> Result<(i32,Vec<String>,Vec<usize>,i32)> {
    let mut k=0; let mut nd=0; let mut na=0;
    check(unsafe {nc_inq_var(nc.0,id,std::ptr::null_mut(),&mut k,&mut nd,std::ptr::null_mut(),&mut na)})?;
    let mut dims=vec![0;nd as usize]; check(unsafe {nc_inq_var(nc.0,id,std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),dims.as_mut_ptr(),std::ptr::null_mut())})?;
    let mut dimensions=Vec::new(); let mut shape=Vec::new();
    for d in dims {let mut name=[0;257];let mut len=0;check(unsafe {nc_inq_dim(nc.0,d,name.as_mut_ptr(),&mut len)})?;dimensions.push(text_name(&name));shape.push(len);}
    Ok((k,dimensions,shape,na))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:4315-4363: typed native attributes.
// Header-only inspection keeps forecast verification independent of data dumps.
fn inspect(req: &Request) -> Result<Value> {
    let nc=Nc::open(&req.path,0)?;
    let mut format=0;check(unsafe {nc_inq_format(nc.0,&mut format)})?;
    let mut nd=0;let mut nv=0;let mut na=0;
    check(unsafe {nc_inq(nc.0,&mut nd,&mut nv,&mut na,std::ptr::null_mut())})?;
    let mut un=0;check(unsafe {nc_inq_unlimdims(nc.0,&mut un,std::ptr::null_mut())})?;
    let mut unlimited=vec![0;un as usize];
    check(unsafe {nc_inq_unlimdims(nc.0,&mut un,unlimited.as_mut_ptr())})?;
    let mut dimensions=BTreeMap::new();let mut unlimited_names=Vec::new();let mut dimension_order=Vec::new();
    for d in 0..nd {
        let mut name=[0;257];let mut len=0;
        check(unsafe {nc_inq_dim(nc.0,d,name.as_mut_ptr(),&mut len)})?;
        let name=text_name(&name);
        if unlimited.contains(&d) {unlimited_names.push(name.clone());}
        dimension_order.push(name.clone());
        dimensions.insert(name,len);
    }
    let mut variables=BTreeMap::new();let mut variable_order=Vec::new();
    for id in 0..nv {
        let mut name=[0;257];
        check(unsafe {nc_inq_var(nc.0,id,name.as_mut_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut())})?;
        let (k,dims,shape,attrs)=var_info(&nc,id)?;
        variable_order.push(text_name(&name));
        let (attribute_order,attribute_words)=attribute_layout(&nc,id,attrs)?;
        variables.insert(text_name(&name),json!({"dtype":dtype(k)?.0,
            "dimensions":dims,"shape":shape,"attrs":get_attrs(&nc,id,attrs)?,
            "attribute_order":attribute_order,"attribute_words":attribute_words}));
    }
    let attrs=get_attrs(&nc,-1,na)?;
    let (global_attribute_order,global_attribute_words)=attribute_layout(&nc,-1,na)?;
    nc.close()?;
    Ok(json!({"schema":"mpas-native-stream/v1","format":format,"dimensions":dimensions,
        "unlimited":unlimited_names,"global_attrs":attrs,"variables":variables,
        "dimension_order":dimension_order,"variable_order":variable_order,
        "global_attribute_order":global_attribute_order,"global_attribute_words":global_attribute_words}))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:2430-3251: native primitive/layout semantics.
// One variable of a read: its raw words (permuted by its layout, if any) in
// `output_directory/var-{id}.raw`, and its description.
fn read_one(nc: &Nc, req: &Request, id: i32, name: String) -> Result<Variable> {
    let (k,dims,mut shape,na)=var_info(nc,id)?; let mut start=vec![0;shape.len()];
    if let Some(record)=req.record {if let Some(axis)=dims.iter().position(|d|d=="Time") {if record>=shape[axis] {return Err("requested record outside file".into());} start[axis]=record;shape[axis]=1;}}
    let size=product(&shape)?.checked_mul(dtype(k)?.1).ok_or("read byte count overflow")?;
    let mut buf=aligned(size);check(unsafe {nc_get_vara(nc.0,id,start.as_ptr(),shape.as_ptr(),buf.as_mut_ptr().cast())})?;
    let filename=req.output_directory.join(format!("var-{id}.raw"));fs::write(&filename,bytes(&buf,size))?;
    let texts=if k==2 && dims.last().map(String::as_str)==Some("StrLen") {
        let width=*shape.last().ok_or("missing text width")?;
        if width==0 {return Err("zero-width native text".into());}
        bytes(&buf,size).chunks_exact(width).map(|s|String::from_utf8_lossy(s).trim_end_matches([' ','\0']).to_owned()).collect::<Vec<_>>()
    } else {Vec::new()};
    let attrs=get_attrs(nc,id,na)?;
    let (attribute_order,_)=attribute_layout(nc,id,na)?;
    if let Some(layout)=req.layouts.get(&name) {
        if dims!=layout.expected_dimensions {return Err(format!("{name}: native read dimension order mismatch").into());}
        let mut input_dimensions=dims.clone();let mut input_shape=shape.clone();
        if !layout.dimensions.iter().any(|d|d=="Time") {
            if let Some(axis)=input_dimensions.iter().position(|d|d=="Time") {
                if input_shape[axis]!=1 {return Err("authority read requires exactly one Time record".into());}
                input_dimensions.remove(axis);input_shape.remove(axis);
            }
        }
        let sizes:BTreeMap<_,_>=input_dimensions.iter().cloned().zip(input_shape.iter().copied()).collect();
        let v=Variable{name:name.clone(),dtype:layout.dtype.clone(),dimensions:layout.dimensions.clone(),
            shape:input_shape,input_dimensions,filename:filename.clone(),attrs:attrs.clone(),attribute_order,text:Vec::new(),
            connectivity_offset:layout.connectivity_offset,input_dtype:Some(dtype(k)?.0.into())};
        let data=prepare(&v,&sizes)?;
        let shape:Vec<_>=layout.dimensions.iter().map(|d|sizes[d]).collect();
        let count=product(&shape)?*dtype(kind(&layout.dtype)?)?.1;
        fs::write(&filename,bytes(&data,count))?;
        Ok(Variable{shape,input_dimensions:layout.dimensions.clone(),input_dtype:None,filename,..v})
    } else {
        Ok(Variable {name,dtype:dtype(k)?.0.into(),dimensions:dims.clone(),input_dimensions:dims,shape,filename,attrs,attribute_order,text:texts,connectivity_offset:0,input_dtype:None})
    }
}
// One worker's answer: each of its variables in order, stopping at the first
// that failed.
#[derive(Serialize,Deserialize)]
struct ShardItem { id: i32, variable: Option<Variable>, error: Option<String> }
// `rw_mpas_stream --read-shard REQUEST.json ID,ID,...`: read_one for those
// variables, in order, as JSON on stdout. The netCDF-C library is not
// thread-safe, so a read is shared out over processes, not threads.
fn read_shard(request: &Path, ids: &str) -> Result<()> {
    let req:Request=serde_json::from_slice(&fs::read(request)?)?;
    let nc=Nc::open(&req.path,0)?;
    let mut items=Vec::new();
    for id in ids.split(',').filter(|t|!t.is_empty()) {
        let id:i32=id.parse()?;
        let mut name=[0;257];
        check(unsafe {nc_inq_var(nc.0,id,name.as_mut_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut())})?;
        match read_one(&nc,&req,id,text_name(&name)) {
            Ok(v) => items.push(ShardItem{id,variable:Some(v),error:None}),
            Err(e) => {items.push(ShardItem{id,variable:None,error:Some(e.to_string())});break;}
        }
    }
    nc.close()?;
    println!("{}",serde_json::to_string(&items)?);Ok(())
}
// Every selected variable, in id order. Large reads go to worker processes,
// each opening the file itself; the variables are dealt out by size and put
// back in id order, and the first failure in id order is the one reported, so
// the response and the files are the serial reader's.
fn read_variables(nc: &Nc, req: &Request, request_path: &Path, wanted: Vec<(i32,String,usize)>) -> Result<Vec<Variable>> {
    let total: usize = wanted.iter().map(|w| w.2).sum();
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        .min(16).min(wanted.len());
    if workers < 2 || total < (64 << 20) {
        return wanted.into_iter().map(|(id,name,_)| read_one(nc,req,id,name)).collect();
    }
    let mut order: Vec<usize> = (0..wanted.len()).collect();
    order.sort_by(|a,b| wanted[*b].2.cmp(&wanted[*a].2).then(a.cmp(b)));
    let mut shards: Vec<(usize,Vec<usize>)> = vec![(0,Vec::new()); workers];
    for i in order {
        let lightest = (0..workers).min_by_key(|s| (shards[*s].0, *s)).expect("workers > 0");
        shards[lightest].0 += wanted[i].2; shards[lightest].1.push(i);
    }
    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for (_, mut members) in shards.into_iter().filter(|s| !s.1.is_empty()) {
        members.sort();
        let ids = members.iter().map(|i| wanted[*i].0.to_string()).collect::<Vec<_>>().join(",");
        children.push(std::process::Command::new(&exe).arg("--read-shard").arg(request_path).arg(ids)
            .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn()?);
    }
    let mut got: BTreeMap<i32,ShardItem> = BTreeMap::new();
    let mut crashed: Option<String> = None;
    for child in children {
        let out = child.wait_with_output()?;
        if !out.status.success() {
            let text = String::from_utf8_lossy(&out.stderr).trim().to_owned();
            crashed.get_or_insert(text.strip_prefix("rw_mpas_stream: ").unwrap_or(&text).to_owned());
            continue;
        }
        for item in serde_json::from_slice::<Vec<ShardItem>>(&out.stdout)? { got.insert(item.id,item); }
    }
    let mut variables = Vec::with_capacity(wanted.len());
    for (id,_,_) in &wanted {
        match got.remove(id) {
            Some(ShardItem{variable:Some(v),..}) => variables.push(v),
            Some(ShardItem{error:Some(e),..}) => return Err(e.into()),
            _ => return Err(crashed.unwrap_or_else(|| format!("variable {id} was not read")).into()),
        }
    }
    Ok(variables)
}
fn read(req: &Request, request_path: &Path) -> Result<Value> {
    let nc=Nc::open(&req.path,0)?;
    let mut nd=0;let mut nv=0;let mut na=0;
    check(unsafe {nc_inq(nc.0,&mut nd,&mut nv,&mut na,std::ptr::null_mut())})?;
    let mut un=0;check(unsafe {nc_inq_unlimdims(nc.0,&mut un,std::ptr::null_mut())})?;
    let mut unlimited=vec![0;un as usize];check(unsafe {nc_inq_unlimdims(nc.0,&mut un,unlimited.as_mut_ptr())})?;
    let mut dimensions=BTreeMap::new();let mut unlimited_names=Vec::new();let mut dimension_order=Vec::new();
    for d in 0..nd {let mut name=[0;257];let mut len=0;check(unsafe {nc_inq_dim(nc.0,d,name.as_mut_ptr(),&mut len)})?;let name=text_name(&name);if unlimited.contains(&d) {unlimited_names.push(name.clone());} dimension_order.push(name.clone());dimensions.insert(name,len);}
    let selected:std::collections::BTreeSet<_>=req.select.iter().cloned().collect();
    let mut found=std::collections::BTreeSet::new();let mut wanted=Vec::new();let mut variable_order=Vec::new();
    fs::create_dir_all(&req.output_directory)?;
    for id in 0..nv {
        let mut name=[0;257];check(unsafe {nc_inq_var(nc.0,id,name.as_mut_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut())})?;
        let name=text_name(&name);
        if !selected.is_empty() && !selected.contains(&name) {continue;}
        found.insert(name.clone());
        variable_order.push(name.clone());
        // The bytes this variable reads, to share the work out by size.
        let (k,_,shape,_)=var_info(&nc,id)?;
        let words=shape.iter().fold(1usize,|a,b| a.saturating_mul(*b));
        wanted.push((id,name,words.saturating_mul(dtype(k).map(|d|d.1).unwrap_or(1))));
    }
    let variables=read_variables(&nc,req,request_path,wanted)?;
    if !selected.is_subset(&found) {return Err(format!("missing requested fields: {:?}",selected.difference(&found).collect::<Vec<_>>()).into());}
    if req.layouts.keys().any(|n| !found.contains(n)) {return Err("missing requested authority layout field".into());}
    let attrs=get_attrs(&nc,-1,na)?;
    let (global_attribute_order,global_attribute_words)=attribute_layout(&nc,-1,na)?;
    nc.close()?;
    Ok(json!({"schema":"mpas-native-stream/v1","dimensions":dimensions,"unlimited":unlimited_names,"global_attrs":attrs,"variables":variables,
        "dimension_order":dimension_order,"variable_order":variable_order,
        "global_attribute_order":global_attribute_order,"global_attribute_words":global_attribute_words}))
}
fn main() {
    let _stamp = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let outcome=(|| -> Result<()> {
        let args:Vec<_>=std::env::args_os().collect();
        if args.len()==2 && args[1]=="--abi" {println!("mpas-native-stream/v1 raw-primitives native-netcdf read write permute connectivity");return Ok(());}
        if args.len()==4 && args[1]=="--read-shard" {return read_shard(Path::new(&args[2]),&args[3].to_string_lossy());}
        if args.len()!=2 {return Err("usage: rw_mpas_stream REQUEST.json | --abi".into());}
        let req:Request=serde_json::from_slice(&fs::read(&args[1])?)?;
        let response=match req.operation.as_str() {"write"=>write(&req)?,"read"=>read(&req,Path::new(&args[1]))?,"inspect"=>inspect(&req)?,_=>return Err("unknown operation".into())};
        println!("{}",serde_json::to_string(&response)?);Ok(())
    })();
    if let Err(error)=outcome {eprintln!("rw_mpas_stream: {error}");std::process::exit(1);}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_uses_rust_reader_and_preserves_words_and_header() -> Result<()> {
        let base=std::env::var_os("RW_MPAS_STREAM_TEST_DIRECTORY")
            .map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let root=base.join(format!("mpas-stream-roundtrip-{}-{nonce}",std::process::id()));
        fs::create_dir_all(&root)?;
        let words=[0x80000000u32,0x00000001,0x7fc12345,0x3f800000,0xff800000,0x7f7fffff];
        let raw:Vec<u8>=words.iter().flat_map(|v|v.to_ne_bytes()).collect();
        let handoff=root.join("theta.raw");fs::write(&handoff,&raw)?;
        let path=root.join("frame.nc");
        let request:Request=serde_json::from_value(json!({
            "operation":"write","path":path,"format":"pnetcdf","header_alignment":131072,
            "dimensions":{"nCells":3,"nVertLevels":2,"Time":1,"StrLen":64},
            "dimension_order":["nVertLevels","nCells","Time","StrLen"],
            "global_attrs":{"source":"MPAS","sphere_radius":{"dtype":"f4","values":6371220.0}},
            "global_attribute_order":["source","sphere_radius"],
            "variables":[{"name":"theta","dtype":"f4",
                "dimensions":["Time","nCells","nVertLevels"],
                "input_dimensions":["nVertLevels","nCells"],"shape":[2,3],"filename":handoff,
                "attrs":{"units":"K","long_name":"potential temperature"},
                "attribute_order":["units","long_name"]},
                {"name":"xtime","dtype":"S1","dimensions":["Time","StrLen"],
                 "text":["2023-03-31_12:00:00"]}]}))?;
        write(&request)?;
        let inspect_request:Request=serde_json::from_value(json!({"operation":"inspect","path":path}))?;
        let header=inspect(&inspect_request)?;
        assert_eq!(header["dimension_order"],json!(["nVertLevels","nCells","Time","StrLen"]));
        assert_eq!(header["variable_order"],json!(["theta","xtime"]));
        assert!(fs::metadata(&path)?.len()>=131072);
        assert_eq!(header["variables"]["theta"]["attribute_order"],json!(["units","long_name"]));
        assert_eq!(header["global_attribute_words"]["sphere_radius"]["dtype"],"f4");
        let read_request:Request=serde_json::from_value(json!({
            "operation":"read","path":path,"output_directory":root.join("decoded"),"record":0,
            "layouts":{"theta":{"dtype":"f4","dimensions":["nVertLevels","nCells"],
                "expected_dimensions":["Time","nCells","nVertLevels"]}}}))?;
        let decoded=read(&read_request,Path::new("unused-serial-test-request"))?;
        assert_eq!(fs::read(decoded["variables"][0]["filename"].as_str().unwrap())?,raw);
        assert_eq!(decoded["variables"][1]["text"],json!(["2023-03-31_12:00:00"]));
        assert_eq!(decoded["variables"][0]["attribute_order"],json!(["units","long_name"]));
        // Keep test processes self-contained. Runtime proofs supply their own
        // scratch root and retain hashes separately before disposing outputs.
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn explicit_order_rejects_missing_duplicate_or_unknown_metadata() {
        let map=BTreeMap::from([("one".to_owned(),1),("two".to_owned(),2)]);
        for order in [vec!["one"],vec!["one","one"],vec!["one","other"]] {
            assert!(ordered_keys(&map,&order.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
        }
    }
}

//! Native formatted inputs for the coupled fire initializer.
use std::collections::BTreeMap;
use std::path::{Path,PathBuf};
use crate::error::{Result,StaticError};
use crate::types::{Field,FieldSet,Grid2,Stack3};
use serde::{Deserialize,Serialize};
use sha2::{Digest,Sha256};

fn invalid(message:impl Into<String>)->StaticError {StaticError::Invalid(message.into())}

#[derive(Deserialize)]
pub struct InputFieldRequest {pub name:String,pub filename:String,pub ni:usize,pub nj:usize}
#[derive(Deserialize)]
pub struct LanduseRequest {pub path:PathBuf,pub section:String}
#[derive(Deserialize)]
pub struct IdealInputRequest {
    pub directory:PathBuf,
    #[serde(default)] pub fields:Vec<InputFieldRequest>,
    #[serde(default)] pub sounding:Option<PathBuf>,
    #[serde(default)] pub landuse:Option<LanduseRequest>,
}
#[derive(Serialize,Default)]
pub struct InputMetadata {
    pub schema:&'static str,
    pub shapes:BTreeMap<String,Vec<usize>>,
    pub field_names:Vec<String>,
    pub sounding_names:BTreeMap<String,String>,
    pub file_sha256:BTreeMap<String,String>,
    pub sounding_levels:Option<usize>,
    pub sounding_disposition:Option<&'static str>,
    pub landuse:Option<LanduseMetadata>,
}
#[derive(Serialize)]
pub struct LanduseMetadata {
    pub section:String,pub found:bool,pub lucats:usize,pub luseas:usize,
    pub columns:[&'static str;7],pub scfx_disposition:&'static str,
}

/// One formatted READ starts a new physical record. Excess values on its
/// final record are discarded; an incomplete list continues on the next.
struct Records<'a> {lines:Vec<&'a str>,next:usize}
impl<'a> Records<'a> {
    fn new(text:&'a str)->Self {Self {lines:text.lines().collect(),next:0}}
    fn read<T>(&mut self,count:usize,partial_eof:bool,mut parse:impl FnMut(usize,&str)->Result<T>)->Result<Option<Vec<T>>> {
        let mut values=Vec::with_capacity(count);
        while values.len()<count {
            let Some(line)=self.lines.get(self.next).copied() else {
                if values.is_empty() || partial_eof {return Ok(None);}
                return Err(invalid("Native formatted input ends inside a required record"));
            };
            self.next+=1;
            let mut cursor=0;
            while values.len()<count {
                let Some(token)=next_token(line,&mut cursor)? else {break;};
                if token=="/" || token.is_empty() {return Err(invalid("Null or slash input leaves required native REAL words undefined"));}
                let (repeat,value)=if let Some((n,v))=token.split_once('*') {
                    let n=n.parse::<usize>().map_err(|_|invalid("Invalid native list-directed repeat count"))?;
                    if n==0 || v.is_empty() {return Err(invalid("Native repeat must supply a positive count and an explicit value"));}
                    (n,v)
                } else {(1,token.as_str())};
                for _ in 0..repeat.min(count-values.len()) {values.push(parse(values.len(),value)?);}
            }
        }
        Ok(Some(values))
    }
    fn required<T>(&mut self,count:usize,parse:impl FnMut(usize,&str)->Result<T>)->Result<Vec<T>> {
        self.read(count,false,parse)?.ok_or_else(||invalid("Native formatted input lacks a required record"))
    }
    fn skip_record(&mut self)->Result<()> {
        if self.next==self.lines.len() {return Err(invalid("Native table ends inside its declared records"));}
        self.next+=1;Ok(())
    }
}

fn next_token(line:&str,cursor:&mut usize)->Result<Option<String>> {
    let bytes=line.as_bytes();
    let first=*cursor==0;
    while *cursor<bytes.len() && bytes[*cursor].is_ascii_whitespace() {*cursor+=1;}
    if *cursor<bytes.len() && bytes[*cursor]==b',' {
        *cursor+=1;
        if first {return Ok(Some(String::new()));}
        while *cursor<bytes.len() && bytes[*cursor].is_ascii_whitespace() {*cursor+=1;}
        if *cursor<bytes.len() && bytes[*cursor]==b',' {return Ok(Some(String::new()));}
    }
    if *cursor==bytes.len() || bytes[*cursor]==b'!' {return Ok(None);}
    if bytes[*cursor]==b'/' {*cursor+=1;return Ok(Some("/".into()));}
    if bytes[*cursor]==b'\'' || bytes[*cursor]==b'"' {
        let quote=bytes[*cursor];*cursor+=1;let mut value=String::new();
        while *cursor<bytes.len() {
            let ch=bytes[*cursor];*cursor+=1;
            if ch==quote {
                if *cursor<bytes.len() && bytes[*cursor]==quote {*cursor+=1;value.push(quote as char);}
                else {return Ok(Some(value));}
            } else {value.push(ch as char);}
        }
        return Err(invalid("Native quoted input has no closing quote"));
    }
    let start=*cursor;
    while *cursor<bytes.len() && !bytes[*cursor].is_ascii_whitespace() && ![b',',b'!',b'/'].contains(&bytes[*cursor]) {*cursor+=1;}
    Ok(Some(line[start..*cursor].to_string()))
}
fn real(token:&str)->Result<f32> {
    let mut normalized=token.replace(['d','D'],"e");
    if !normalized.contains(['e','E']) {
        if let Some(index)=normalized.char_indices().skip(1).find_map(|(i,c)|matches!(c,'+'|'-').then_some(i)) {
            normalized.insert(index,'e');
        }
    }
    let value=normalized.parse::<f32>().map_err(|_|invalid(format!("Invalid native REAL token {token:?}")))?;
    if value.is_infinite() && !matches!(normalized.to_ascii_lowercase().trim_start_matches(['+','-']),"inf"|"infinity") {
        return Err(invalid(format!("Native REAL token {token:?} exceeds default-REAL range")));
    }
    Ok(value)
}
fn positive_integer(token:&str)->Result<usize> {
    token.parse::<usize>().ok().filter(|v|*v>0).ok_or_else(||invalid(format!("Invalid positive native dimension {token:?}")))
}
fn read_source(path:&Path,metadata:&mut InputMetadata)->Result<String> {
    let bytes=std::fs::read(path)?;
    metadata.file_sha256.insert(path.to_string_lossy().into_owned(),format!("{:x}",Sha256::digest(&bytes)));
    String::from_utf8(bytes).map_err(|_|invalid(format!("{} is not formatted UTF-8 text",path.display())))
}
fn field(set:&mut FieldSet,metadata:&mut InputMetadata,name:&str,shape:Vec<usize>,values:Vec<f32>)->Result<()> {
    if set.fields.contains_key(name) {return Err(invalid(format!("Native input repeats field {name}")));}
    let data=values.into_iter().map(f64::from).collect();
    let value=match shape.as_slice() {
        []=>Field::Plane(Grid2 {ny:1,nx:1,data}),
        [n]=>Field::Plane(Grid2 {ny:1,nx:*n,data}),
        [ny,nx]=>Field::Plane(Grid2 {ny:*ny,nx:*nx,data}),
        [planes,ny,nx]=>Field::Stack(Stack3 {planes:*planes,ny:*ny,nx:*nx,data}),
        _=>return Err(invalid("Native ideal input rank exceeds the fieldset contract")),
    };
    metadata.shapes.insert(name.into(),shape);set.fields.insert(name.into(),value);Ok(())
}

pub fn read_inputs(request:&IdealInputRequest)->Result<(FieldSet,InputMetadata)> {
    let mut set=FieldSet::default();
    let mut metadata=InputMetadata {schema:"gpuwm-sfire-ideal-input-v1",..Default::default()};
    for entry in &request.fields {
        if entry.name.is_empty() || entry.filename.is_empty() || Path::new(&entry.filename).components().count()!=1 {
            return Err(invalid("Native ideal field requests require a name and one fixed filename"));
        }
        if entry.ni==0 || entry.nj==0 {return Err(invalid("Native ideal input geometry must be positive"));}
        let text=read_source(&request.directory.join(&entry.filename),&mut metadata)?;
        let mut records=Records::new(&text);
        let dimensions=records.required(2,|_,t|positive_integer(t))?;
        if dimensions!=[entry.ni,entry.nj] {return Err(invalid(format!("{} declares {dimensions:?}; native requested geometry is [{},{}]",entry.filename,entry.ni,entry.nj)));}
        let count=entry.ni.checked_mul(entry.nj).ok_or_else(||invalid("Native ideal geometry overflows the array length"))?;
        let mut values=Vec::new();values.try_reserve_exact(count).map_err(|e|invalid(format!("Native ideal array allocation failed: {e}")))?;values.resize(count,0.);
        for i in 0..entry.ni {
            let column=records.required(entry.nj,|_,t|real(t))?;
            for (j,v) in column.into_iter().enumerate() {values[j*entry.ni+i]=v;}
        }
        field(&mut set,&mut metadata,&entry.name,vec![entry.nj,entry.ni],values)?;
        metadata.field_names.push(entry.name.clone());
    }
    if let Some(path)=&request.sounding {
        let text=read_source(path,&mut metadata)?;let mut records=Records::new(&text);
        let surface=records.required(3,|_,t|real(t))?;
        for (name,value) in ["psurf","theta_surface","qv_surface"].into_iter().zip(surface) {
            let key=format!("_SOUNDING_{name}");field(&mut set,&mut metadata,&key,vec![],vec![value])?;
            metadata.sounding_names.insert(name.into(),key);
        }
        let mut profile=[Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new()];
        while let Some(row)=records.read(5,true,|_,t|real(t))? {
            if profile[0].len()==1000 {return Err(invalid("More than 1000 sounding levels would overflow the native initializer arrays"));}
            for (column,value) in profile.iter_mut().zip(row) {column.push(value);}
        }
        let levels=profile[0].len();
        if levels<2 {return Err(invalid("Native atmospheric interpolation needs at least two complete sounding levels"));}
        for (name,values) in ["height","theta","qv","u","v"].into_iter().zip(profile) {
            let key=format!("_SOUNDING_{name}");field(&mut set,&mut metadata,&key,vec![levels],values)?;
            metadata.sounding_names.insert(name.into(),key);
        }
        metadata.sounding_levels=Some(levels);
        metadata.sounding_disposition=Some("Native get_sounding ignores header qv_surface and uses first-profile qv for surface density; raw header preserved");
    }
    if let Some(request)=&request.landuse {
        let text=read_source(&request.path,&mut metadata)?;
        let mut records=Records::new(&text);let mut selected=None;
        while let Some(section)=records.read(1,false,|_,t|Ok(t.to_string()))? {
            let dimensions=records.required(2,|_,t|positive_integer(t))?;
            let (categories,seasons)=(dimensions[0],dimensions[1]);
            let found=section[0]==request.section;
            let mut values=Vec::new();
            for _ in 0..seasons {
                records.skip_record()?;
                for category in 1..=categories {
                    if found {
                        let row=records.required(8,|index,t| {
                            if index==0 {
                                let value=positive_integer(t)?;
                                if value!=category {return Err(invalid("Native LANDUSE table has a missing category record"));}
                                Ok(0.)
                            } else {real(t)}
                        })?;
                        values.extend_from_slice(&row[1..]);
                    } else {records.skip_record()?;}
                }
            }
            if found {selected=Some((categories,seasons,values));break;}
        }
        let (categories,seasons,values,found)=match selected {Some((c,s,v))=>(c,s,v,true),None=>(0,0,Vec::new(),false)};
        field(&mut set,&mut metadata,"_LANDUSE_TABLE",vec![seasons,categories,7],values)?;
        metadata.landuse=Some(LanduseMetadata {section:request.section.clone(),found,lucats:categories,luseas:seasons,
            columns:["ALBD","SLMO","SFEM","SFZ0","THERIN","SCFX","SFHC"],
            scfx_disposition:"Native landuse_init reuses the final season SCFX column for all seasons; raw columns preserved"});
    }
    Ok((set,metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn list_directed_records_split_repeat_and_discard_extras() {
        let mut records=Records::new("1D0 2e0 ignored extra\n3\n2*4.5\n-0.0 1.0-38\n");
        assert_eq!(records.required(2,|_,t|real(t)).unwrap(),vec![1.,2.]);
        assert_eq!(records.required(3,|_,t|real(t)).unwrap(),vec![3.,4.5,4.5]);
        let special=records.required(2,|_,t|real(t)).unwrap();
        assert_eq!(special[0].to_bits(),(-0.0_f32).to_bits());
        assert_eq!(special[1].to_bits(),1e-38_f32.to_bits());
    }
    #[test]
    fn required_array_eof_is_refused_but_native_partial_profile_eof_is_discarded() {
        assert!(Records::new("1 2").required(3,|_,t|real(t)).is_err());
        assert!(Records::new("1 2").read(3,true,|_,t|real(t)).unwrap().is_none());
        assert!(Records::new("1,,2").required(2,|_,t|real(t)).is_err());
        assert!(real("1e39").is_err());
    }
}

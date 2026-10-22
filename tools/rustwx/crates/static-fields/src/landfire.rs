//! LANDFIRE canopy contract, shared with acquisition and SFIRE field binding.
//! The native raster reader applies the source scale after masking NoData.
use crate::{error::{Result, StaticError}, types::FieldSet};
use std::collections::BTreeMap;

#[derive(serde::Deserialize)]
pub struct CanopyField { pub units:String, pub minimum:f64, pub maximum:f64 }
#[derive(serde::Deserialize)]
struct Contract { fields:BTreeMap<String,CanopyField> }
pub fn contract() -> BTreeMap<String,CanopyField> {
    serde_json::from_str::<Contract>(include_str!("../../../../../gpuwm/authorities/sfire-input-fields.v1.json"))
        .expect("packaged canopy field table").fields
}
pub fn validate_names<'a>(names:impl Iterator<Item=&'a str>) -> Result<()> {
    let actual:std::collections::BTreeSet<_>=names.collect();
    let table=contract();
    if !actual.is_empty() && actual!=table.keys().map(String::as_str).collect() {
        return Err(StaticError::Invalid("SFIRE canopy inputs require the complete cover, height, base height and bulk density set; partial canopies would misstate the fuel structure".into()));
    }
    Ok(())
}
pub fn validate_fields(fields:&FieldSet) -> Result<()> {
    for (name,row) in contract() {
        if let Some(field)=fields.fields.get(&name) {
            if let Some((p,v))=field.data().iter().enumerate().find(|(_,v)| !v.is_finite() || **v<row.minimum || **v>row.maximum) {
                return Err(StaticError::Invalid(format!("SFIRE {name} outside [{}, {}] {} at fine cell {p}: {v}",row.minimum,row.maximum,row.units)));
            }
        }
    }
    Ok(())
}

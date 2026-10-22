//! CF observations bound to source variables, physical units and parameter ids.
use crate::{cf_grid, verification_io::{GridData, RadarData, RadarSpec, source_identity}};
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Binding {
    quantity: String, variables: Vec<String>, source_variables: Vec<String>,
    parameters: Vec<String>, units: String, operation: String,
}
pub fn load(spec: &RadarSpec) -> Result<RadarData, String> {
    let rows: Vec<Binding> = serde_json::from_str(include_str!("../data/gridded-verification.json"))
        .map_err(|e|e.to_string())?;
    let row=rows.iter().find(|r|r.quantity==spec.quantity)
        .ok_or_else(||format!("no CF observation binding for {}",spec.quantity))?;
    let grid=cf_grid::CfGrid::open(&spec.path,None)?;
    let raw=netcrust::File::open(&spec.path).map_err(|e|e.to_string())?;
    let source=raw.attribute("source_store").and_then(|a|a.as_string().map(str::to_owned))
        .or_else(||raw.attribute("source").and_then(|a|a.as_string().map(str::to_owned)))
        .filter(|s|!s.trim().is_empty()).ok_or("CF observation source identity is missing")?;
    let products=cf_grid::product_rows(None)?;
    let mut planes=Vec::new();
    for (i,name) in row.variables.iter().enumerate() {
        let var=raw.variable(name).ok_or_else(||format!("missing reference variable {name}"))?;
        if var.attribute("source_variable").and_then(|a|a.as_string().map(str::to_owned)).as_deref()
            !=Some(row.source_variables[i].as_str()) {
            return Err(format!("{name}: source variable does not match the reference parameter binding"));
        }
        if var.attribute("units").and_then(|a|a.as_string().map(str::to_owned))
            .filter(|u|!u.is_empty()).is_none() { return Err(format!("{name}: reference units are missing")); }
        let (values, product)=grid.field(name,&products)?;
        if product.observation_product.as_deref()!=Some(row.parameters[i].as_str()) || product.display_units!=row.units {
            return Err("CF product and verification parameter/units bindings disagree".into());
        }
        planes.push(values);
    }
    let values:Vec<f64>=match row.operation.as_str() {
        "identity" if planes.len()==1=>planes[0].par_iter().map(|v|*v as f64).collect(),
        "magnitude" if planes.len()==2=>planes[0].par_iter().zip(&planes[1]).map(|(&u,&v)|(u as f64).hypot(v as f64)).collect(),
        _=>return Err("unsupported gridded observation operation or arity".into()),
    };
    let valid=values.par_iter().map(|v|v.is_finite() && v.abs()<1e20).collect();
    Ok(RadarData {
        grid:GridData{lat:grid.lat,lon:grid.lon,nx:grid.nx,ny:grid.ny,projection:None},
        values,valid,valid_time:grid.valid.to_rfc3339_opts(chrono::SecondsFormat::Secs,true),
        quantity:row.quantity.clone(),units:row.units.clone(),
        provenance:serde_json::json!({"source":source,"artifact":source_identity(&spec.path)?,
          "parameters":row.parameters,"source_variables":row.source_variables,"operation":row.operation,
          "vertical_support":"reference: surface to atmosphere top; forecast: surface to finite model lid",
          "is_stub":false}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    #[test]
    fn cf_flux_binding_checks_parameters_units_sign_time_and_vector_magnitude() {
        let root=std::env::temp_dir().join(format!("vapour-binding-{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        for (case,units,source) in [("valid","kg m-1 s-1","vertical_integral_of_eastward_water_vapour_flux"),
            ("wrong_units","m s-1","vertical_integral_of_eastward_water_vapour_flux"),
            ("wrong_quantity","kg m-1 s-1","eastward_wind")] {
            let path=root.join(format!("{case}.nc"));
            let mut s=Schema::new(NcFormat::Offset64);
            let t=s.def_dim("time",1,false).unwrap(); let y=s.def_dim("latitude",2,false).unwrap();
            let x=s.def_dim("longitude",2,false).unwrap();
            let tv=s.def_var("time",NcType::Double,&[t]).unwrap();
            let la=s.def_var("latitude",NcType::Float,&[y]).unwrap();
            let lo=s.def_var("longitude",NcType::Float,&[x]).unwrap();
            let u=s.def_var("viwve",NcType::Float,&[t,y,x]).unwrap();
            let v=s.def_var("viwvn",NcType::Float,&[t,y,x]).unwrap();
            s.put_global_attr("source_store",AttrValue::Text("observed reanalysis source".into())).unwrap();
            s.put_var_attr(tv,"units",AttrValue::Text("hours since 2023-01-09 00:00:00".into())).unwrap();
            for (id,unit,src) in [(u,units,source),(v,"kg m-1 s-1","vertical_integral_of_northward_water_vapour_flux")] {
                s.put_var_attr(id,"units",AttrValue::Text(unit.into())).unwrap();
                s.put_var_attr(id,"source_variable",AttrValue::Text(src.into())).unwrap();
            }
            let mut writer=NcWriter::create(&path,s).unwrap();
            writer.write_var(tv,VarData::F64(&[6.])).unwrap();
            writer.write_var(la,VarData::F32(&[30.,31.])).unwrap();
            writer.write_var(lo,VarData::F32(&[-125.,-124.])).unwrap();
            writer.write_var(u,VarData::F32(&[-300.;4])).unwrap();
            writer.write_var(v,VarData::F32(&[400.;4])).unwrap();writer.finish().unwrap();
            let spec=RadarSpec{quantity:"ivt_full_column".into(),path:path.clone(),grid_path:path.clone()};
            let output=load(&spec);
            if case=="valid" {
                let output=output.unwrap();assert_eq!(output.values,vec![500.;4]);
                assert_eq!(output.valid_time,"2023-01-09T06:00:00Z");
                assert_eq!(output.provenance["parameters"],serde_json::json!(["162071","162072"]));
                let signed=load(&RadarSpec{quantity:"ivt_eastward_full_column".into(),..spec.clone()}).unwrap();
                assert_eq!(signed.values,vec![-300.;4]);
                assert!(load(&RadarSpec{quantity:"ivt_1000_300".into(),..spec}).is_err());
            } else {assert!(output.is_err(),"{case} must be refused");}
            std::fs::remove_file(path).unwrap();
        }
        std::fs::remove_dir(root).unwrap();
    }
}

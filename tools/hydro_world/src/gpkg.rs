//! GeoPackage <-> contract structs. Reading accepts any NextGen v2.0.1-shaped
//! GeoPackage and records every place it deviates from the contract's declared
//! types (lossless coercions only; anything lossy is an error). Writing always
//! emits the contract's exact DDL, so every builder produces the same schema.
use crate::contract::{self, ColumnSpec, Contents, Fabric, Geometry, Srs, Storage, TableSpec};
use crate::sqlite::{ident, literal, Db, Value};
use crate::Result;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{Map, Number, Value as Json};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// What a read observed about the source container.
#[derive(Debug, Default, Serialize)]
pub struct ReadReport {
    /// Declared-type or storage-class differences from the contract, each
    /// coerced losslessly (e.g. "flowpaths.mainstem declared REAL").
    pub deviations: BTreeMap<String, usize>,
    /// Contract tables the source does not carry.
    pub missing_tables: Vec<String>,
    /// Source tables outside the contract (ignored, never copied).
    pub ignored_tables: Vec<String>,
}

const MAX_EXACT: f64 = 9_007_199_254_740_992.0; // 2^53

fn to_json(spec: &TableSpec, col: &ColumnSpec, v: &Value, report: &mut ReadReport) -> Result<Json> {
    let here = || format!("{}.{}", spec.name, col.name);
    Ok(match (col.storage, v) {
        (_, Value::Null) => Json::Null,
        (Storage::Integer, Value::Integer(i)) => Json::from(*i),
        (Storage::Integer, Value::Real(x)) if x.fract() == 0.0 && x.abs() < MAX_EXACT => {
            *report.deviations.entry(format!("{} stored REAL, contract INTEGER (lossless)", here())).or_default() += 1;
            Json::from(*x as i64)
        }
        (Storage::Real, Value::Real(x)) => Json::Number(Number::from_f64(*x).ok_or_else(|| format!("nonfinite {}", here()))?),
        (Storage::Real, Value::Integer(i)) if (*i as f64).abs() < MAX_EXACT => {
            *report.deviations.entry(format!("{} stored INTEGER, contract REAL (lossless)", here())).or_default() += 1;
            Json::Number(Number::from_f64(*i as f64).unwrap())
        }
        (Storage::Text, Value::Text(t)) => Json::String(t.clone()),
        (Storage::Boolean, Value::Integer(i)) if *i == 0 || *i == 1 => Json::Bool(*i == 1),
        (Storage::Geometry, Value::Blob(b)) => {
            validate_geometry(b).map_err(|e| format!("{}: {e}", here()))?;
            serde_json::to_value(Geometry(b.clone()))?
        }
        (s, v) => return Err(format!("{}: value {v:?} cannot be stored losslessly as {s:?}", here()).into()),
    })
}

/// GeoPackage binary header check (spec 2.1.3): magic "GP", version 0,
/// standard (not extended) geometry, valid envelope code.
pub fn validate_geometry(b: &[u8]) -> Result<()> {
    if b.len() < 8 || &b[..2] != b"GP" || b[2] != 0 {
        return Err("not a GeoPackage binary geometry".into());
    }
    if b[3] & 0x20 != 0 {
        return Err("extended GeoPackage geometry is outside the contract".into());
    }
    let envelope = match (b[3] >> 1) & 7 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => return Err("invalid envelope indicator".into()),
    };
    if b.len() < 8 + envelope + 5 {
        return Err("truncated geometry".into());
    }
    Ok(())
}

fn geometry_srs(b: &[u8]) -> i32 {
    let raw = [b[4], b[5], b[6], b[7]];
    if b[3] & 1 == 1 { i32::from_le_bytes(raw) } else { i32::from_be_bytes(raw) }
}

fn read_table<T: DeserializeOwned + Send>(db: &Db, spec: &TableSpec, report: &mut ReadReport) -> Result<Vec<T>> {
    let declared = db.query(&format!("PRAGMA table_info({})", ident(spec.name)))?;
    let mut have = BTreeMap::new();
    for r in &declared.rows {
        if let (Value::Text(name), Value::Text(decl)) = (&r[1], &r[2]) {
            have.insert(name.clone(), decl.clone());
        }
    }
    for col in spec.columns {
        let want = col.decl.split_whitespace().next().unwrap_or("");
        match have.get(col.name) {
            None => return Err(format!("{}.{} missing (contract column)", spec.name, col.name).into()),
            Some(d) if !d.eq_ignore_ascii_case(want) => *report.deviations.entry(format!("{}.{} declared {d}, contract {want}", spec.name, col.name)).or_default() += 1,
            _ => {}
        }
    }
    if let Some(extra) = have.keys().find(|k| !spec.columns.iter().any(|c| c.name == k.as_str())) {
        return Err(format!("{}.{extra} is not a contract column", spec.name).into());
    }
    // Physical scan order (rowid), which is the order the engine's reader sees.
    let cols = spec.columns.iter().map(|c| ident(c.name)).collect::<Vec<_>>().join(",");
    let rows = db.query(&format!("SELECT {cols} FROM {}", ident(spec.name)))?;
    // Value conversion on all threads; row order and deviation counts are kept.
    let converted: Vec<Result<(T, ReadReport)>> = rows
        .rows
        .par_iter()
        .map(|r| {
            let mut local = ReadReport::default();
            let mut m = Map::new();
            for (col, v) in spec.columns.iter().zip(r) {
                m.insert(col.name.to_string(), to_json(spec, col, v, &mut local)?);
            }
            Ok((serde_json::from_value(Json::Object(m)).map_err(|e| format!("{}: {e}", spec.name))?, local))
        })
        .collect();
    let mut out = Vec::with_capacity(converted.len());
    for c in converted {
        let (row, local) = c?;
        for (k, n) in local.deviations {
            *report.deviations.entry(k).or_default() += n;
        }
        out.push(row);
    }
    Ok(out)
}

fn joined<T>(x: std::thread::Result<Result<T>>) -> Result<T> {
    x.map_err(|_| crate::Error::from("table reader panicked"))?
}

/// Read one table on its own read-only connection (tables are read in parallel).
fn read_table_at<T: DeserializeOwned + Send>(path: &str, spec: &TableSpec) -> Result<(Vec<T>, ReadReport)> {
    let db = Db::open_read(path)?;
    let mut r = ReadReport::default();
    let rows = read_table(&db, spec, &mut r)?;
    Ok((rows, r))
}

fn text(v: &Value) -> Option<String> {
    match v {
        Value::Text(s) => Some(s.clone()),
        _ => None,
    }
}
fn real(v: &Value) -> Option<f64> {
    match v {
        Value::Real(x) => Some(*x),
        Value::Integer(i) => Some(*i as f64),
        _ => None,
    }
}

pub fn read(path: &str) -> Result<(Fabric, ReadReport)> {
    let db = Db::open_read(path)?;
    let mut report = ReadReport::default();
    let contents_rows = db.query("SELECT table_name,data_type,identifier,description,last_change,min_x,min_y,max_x,max_y,srs_id FROM gpkg_contents")?;
    let mut contents = BTreeMap::new();
    let mut present = Vec::new();
    for r in &contents_rows.rows {
        let name = text(&r[0]).ok_or("gpkg_contents.table_name")?;
        let Some(spec) = contract::table(&name) else {
            report.ignored_tables.push(name);
            continue;
        };
        if text(&r[1]).as_deref() != Some(spec.data_type) {
            *report.deviations.entry(format!("{name} data_type {:?}, contract {}", r[1], spec.data_type)).or_default() += 1;
        }
        let bbox = match (real(&r[5]), real(&r[6]), real(&r[7]), real(&r[8])) {
            (Some(a), Some(b), Some(c), Some(d)) => Some([a, b, c, d]),
            _ => None,
        };
        contents.insert(name.clone(), Contents { identifier: text(&r[2]), description: text(&r[3]), last_change: text(&r[4]).unwrap_or_default(), bbox });
        present.push(name);
    }
    let geoms = db.query("SELECT table_name,column_name,geometry_type_name,srs_id FROM gpkg_geometry_columns")?;
    let mut srs_ids = std::collections::BTreeSet::new();
    for r in &geoms.rows {
        let name = text(&r[0]).unwrap_or_default();
        if let Some(spec) = contract::table(&name) {
            if text(&r[1]).as_deref() != Some("geom") {
                return Err(format!("{name} geometry column must be geom").into());
            }
            let gt = text(&r[2]).unwrap_or_default();
            if Some(gt.as_str()) != spec.geometry_type {
                *report.deviations.entry(format!("{name}.geom geometry_type {gt}, contract {}", spec.geometry_type.unwrap_or("none"))).or_default() += 1;
            }
            if let Value::Integer(s) = r[3] {
                srs_ids.insert(s);
            }
        }
    }
    if srs_ids.len() != 1 {
        return Err(format!("contract requires one fabric CRS; found srs_ids {srs_ids:?}").into());
    }
    let srs_id = *srs_ids.iter().next().unwrap();
    let s = db.query(&format!("SELECT srs_name,srs_id,organization,organization_coordsys_id,definition,description FROM gpkg_spatial_ref_sys WHERE srs_id={srs_id}"))?;
    let r = s.rows.first().ok_or("fabric srs_id missing from gpkg_spatial_ref_sys")?;
    let srs = Srs {
        srs_name: text(&r[0]).unwrap_or_default(),
        srs_id,
        organization: text(&r[2]).unwrap_or_default(),
        organization_coordsys_id: match r[3] { Value::Integer(i) => i, _ => return Err("organization_coordsys_id".into()) },
        definition: text(&r[4]).unwrap_or_default(),
        description: text(&r[5]),
    };
    for t in contract::TABLES {
        if !present.iter().any(|p| p == t.name) {
            report.missing_tables.push(t.name.into());
        }
    }
    let need = |n: &str| -> Result<&'static TableSpec> {
        if present.iter().any(|p| p == n) {
            Ok(contract::table(n).unwrap())
        } else {
            Err(format!("source lacks required table {n}").into())
        }
    };
    let optional = |n: &str| present.iter().any(|p| p == n).then(|| contract::table(n).unwrap());
    let (dv, fp, nx, nw, fa, hl, lk, ma, md) = {
        let (sd, sf, sn, sw, sa, sh, sl) = (need("divides")?, need("flowpaths")?, need("nexus")?, need("network")?, need("flowpath_attributes")?, need("hydrolocations")?, need("lakes")?);
        let (om, od) = (optional("model_attributes"), optional("fabric_metadata"));
        std::thread::scope(|sc| {
            let a = sc.spawn(|| read_table_at::<contract::Divide>(path, sd));
            let b = sc.spawn(|| read_table_at::<contract::Flowpath>(path, sf));
            let c = sc.spawn(|| read_table_at::<contract::Nexus>(path, sn));
            let d = sc.spawn(|| read_table_at::<contract::NetworkRow>(path, sw));
            let e = sc.spawn(|| read_table_at::<contract::FlowpathAttributes>(path, sa));
            let f = sc.spawn(|| read_table_at::<contract::Hydrolocation>(path, sh));
            let g = sc.spawn(|| read_table_at::<contract::Lake>(path, sl));
            let h = sc.spawn(|| om.map(|t| read_table_at::<contract::ModelAttributes>(path, t)).transpose());
            let i = sc.spawn(|| od.map(|t| read_table_at::<contract::MetadataRow>(path, t)).transpose());
            Ok::<_, crate::Error>((joined(a.join())?, joined(b.join())?, joined(c.join())?, joined(d.join())?, joined(e.join())?, joined(f.join())?, joined(g.join())?, joined(h.join())?, joined(i.join())?))
        })?
    };
    let mut merge = |r: ReadReport| {
        for (k, n) in r.deviations {
            *report.deviations.entry(k).or_default() += n;
        }
    };
    let fabric = Fabric {
        contract: contract::CONTRACT.into(),
        srs,
        contents,
        divides: { merge(dv.1); dv.0 },
        flowpaths: { merge(fp.1); fp.0 },
        nexus: { merge(nx.1); nx.0 },
        network: { merge(nw.1); nw.0 },
        flowpath_attributes: { merge(fa.1); fa.0 },
        hydrolocations: { merge(hl.1); hl.0 },
        lakes: { merge(lk.1); lk.0 },
        model_attributes: ma.map(|m| { merge(m.1); m.0 }),
        fabric_metadata: md.map(|m| { merge(m.1); m.0 }),
    };
    // Every geometry header must name the fabric CRS.
    for (name, blobs) in [
        ("divides", fabric.divides.iter().filter_map(|r| r.geom.as_ref()).collect::<Vec<_>>()),
        ("flowpaths", fabric.flowpaths.iter().filter_map(|r| r.geom.as_ref()).collect()),
        ("nexus", fabric.nexus.iter().filter_map(|r| r.geom.as_ref()).collect()),
        ("hydrolocations", fabric.hydrolocations.iter().filter_map(|r| r.geom.as_ref()).collect()),
        ("lakes", fabric.lakes.iter().filter_map(|r| r.geom.as_ref()).collect()),
    ] {
        if let Some(g) = blobs.iter().find(|g| geometry_srs(&g.0) as i64 != srs_id) {
            return Err(format!("{name} geometry header srs {} differs from fabric srs {srs_id}", geometry_srs(&g.0)).into());
        }
    }
    Ok((fabric, report))
}

fn row_values<T: Serialize>(spec: &TableSpec, row: &T) -> Result<Vec<Value>> {
    let Json::Object(m) = serde_json::to_value(row)? else {
        return Err("row is not an object".into());
    };
    spec.columns
        .iter()
        .map(|col| {
            let v = m.get(col.name).ok_or_else(|| format!("{}.{} absent from struct", spec.name, col.name))?;
            Ok(match (col.storage, v) {
                (_, Json::Null) => Value::Null,
                (Storage::Integer, Json::Number(n)) => Value::Integer(n.as_i64().ok_or_else(|| format!("{}.{} not i64", spec.name, col.name))?),
                (Storage::Real, Json::Number(n)) => Value::Real(n.as_f64().unwrap()),
                (Storage::Text, Json::String(s)) => Value::Text(s.clone()),
                (Storage::Boolean, Json::Bool(b)) => Value::Integer(i64::from(*b)),
                (Storage::Geometry, g @ Json::String(_)) => Value::Blob(serde_json::from_value::<Geometry>(g.clone())?.0),
                (s, v) => return Err(format!("{}.{}: {v} is not {s:?}", spec.name, col.name).into()),
            })
        })
        .collect()
}

/// WGS 84 row the GeoPackage spec requires in every container (spec Table 2).
const WGS84_WKT: &str = r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]"#;

fn write_table<T: Serialize + Sync>(db: &Db, f: &Fabric, spec: &TableSpec, rows: &[T]) -> Result<()> {
    let cols = spec.columns.iter().map(|c| format!("{} {}", ident(c.name), c.decl)).collect::<Vec<_>>().join(", ");
    db.exec(&format!("CREATE TABLE {} ( {cols})", ident(spec.name)))?;
    let c = f.contents.get(spec.name);
    let bbox = c.and_then(|c| c.bbox);
    let num = |i: usize| bbox.map(|b| Value::Real(b[i])).unwrap_or(Value::Null);
    let srs = if spec.geometry_type.is_some() { f.srs.srs_id } else { 0 };
    // Bound values keep the bbox doubles bit-exact.
    db.insert_all(
        "INSERT INTO gpkg_contents (table_name,data_type,identifier,description,last_change,min_x,min_y,max_x,max_y,srs_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
        &[vec![
            Value::Text(spec.name.into()),
            Value::Text(spec.data_type.into()),
            // Present contents are written exactly; a builder without them gets
            // identifier = table name, empty description, epoch last_change.
            match c {
                Some(c) => c.identifier.clone().map(Value::Text).unwrap_or(Value::Null),
                None => Value::Text(spec.name.into()),
            },
            match c {
                Some(c) => c.description.clone().map(Value::Text).unwrap_or(Value::Null),
                None => Value::Text(String::new()),
            },
            Value::Text(c.map(|c| c.last_change.clone()).unwrap_or_else(|| "1970-01-01T00:00:00.000Z".into())),
            num(0),
            num(1),
            num(2),
            num(3),
            Value::Integer(srs),
        ]],
    )?;
    if let Some(gt) = spec.geometry_type {
        db.exec(&format!(
            "INSERT INTO gpkg_geometry_columns (table_name,column_name,geometry_type_name,srs_id,z,m) VALUES ({},'geom',{},{},0,0)",
            literal(spec.name), literal(gt), f.srs.srs_id
        ))?;
    }
    let values = rows.par_iter().map(|r| row_values(spec, r)).collect::<Result<Vec<_>>>()?;
    let marks = vec!["?"; spec.columns.len()].join(",");
    let names = spec.columns.iter().map(|c| ident(c.name)).collect::<Vec<_>>().join(",");
    db.insert_all(&format!("INSERT INTO {} ({names}) VALUES ({marks})", ident(spec.name)), &values)
}

/// Write a fabric with the contract's exact DDL. Refuses to overwrite.
/// The container carries no rtree index or GDAL triggers (they need GDAL's
/// ST_* SQL functions); the engine reader does not use them.
pub fn write(f: &Fabric, path: &str) -> Result<()> {
    if f.contract != contract::CONTRACT {
        return Err(format!("fabric contract {} is not {}", f.contract, contract::CONTRACT).into());
    }
    let db = Db::create(path)?;
    db.exec(&format!("PRAGMA application_id={}; PRAGMA user_version={};", contract::GPKG_APPLICATION_ID, contract::GPKG_USER_VERSION))?;
    db.exec("BEGIN")?;
    db.exec("CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL,srs_id INTEGER NOT NULL PRIMARY KEY,organization TEXT NOT NULL,organization_coordsys_id INTEGER NOT NULL,definition  TEXT NOT NULL,description TEXT)")?;
    db.exec("CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY,data_type TEXT NOT NULL,identifier TEXT UNIQUE,description TEXT DEFAULT '',last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),min_x DOUBLE, min_y DOUBLE,max_x DOUBLE, max_y DOUBLE,srs_id INTEGER,CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))")?;
    db.exec("CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL,column_name TEXT NOT NULL,geometry_type_name TEXT NOT NULL,srs_id INTEGER NOT NULL,z TINYINT NOT NULL,m TINYINT NOT NULL,CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),CONSTRAINT uk_gc_table_name UNIQUE (table_name),CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id))")?;
    let mut srs = vec![
        vec![Value::Text("Undefined Cartesian SRS".into()), Value::Integer(-1), Value::Text("NONE".into()), Value::Integer(-1), Value::Text("undefined".into()), Value::Text("undefined Cartesian coordinate reference system".into())],
        vec![Value::Text("Undefined geographic SRS".into()), Value::Integer(0), Value::Text("NONE".into()), Value::Integer(0), Value::Text("undefined".into()), Value::Text("undefined geographic coordinate reference system".into())],
        vec![Value::Text("WGS 84 geodetic".into()), Value::Integer(4326), Value::Text("EPSG".into()), Value::Integer(4326), Value::Text(WGS84_WKT.into()), Value::Text("longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid".into())],
    ];
    if ![-1, 0, 4326].contains(&f.srs.srs_id) {
        let s = &f.srs;
        srs.push(vec![
            Value::Text(s.srs_name.clone()),
            Value::Integer(s.srs_id),
            Value::Text(s.organization.clone()),
            Value::Integer(s.organization_coordsys_id),
            Value::Text(s.definition.clone()),
            s.description.clone().map(Value::Text).unwrap_or(Value::Null),
        ]);
    }
    db.insert_all("INSERT INTO gpkg_spatial_ref_sys (srs_name,srs_id,organization,organization_coordsys_id,definition,description) VALUES (?,?,?,?,?,?)", &srs)?;
    let t = |n: &str| contract::table(n).unwrap();
    write_table(&db, f, t("divides"), &f.divides)?;
    write_table(&db, f, t("flowpaths"), &f.flowpaths)?;
    write_table(&db, f, t("nexus"), &f.nexus)?;
    write_table(&db, f, t("network"), &f.network)?;
    write_table(&db, f, t("flowpath_attributes"), &f.flowpath_attributes)?;
    write_table(&db, f, t("hydrolocations"), &f.hydrolocations)?;
    write_table(&db, f, t("lakes"), &f.lakes)?;
    if let Some(m) = &f.model_attributes {
        write_table(&db, f, t("model_attributes"), m)?;
    }
    if let Some(m) = &f.fabric_metadata {
        write_table(&db, f, t("fabric_metadata"), m)?;
    }
    db.exec("COMMIT")
}

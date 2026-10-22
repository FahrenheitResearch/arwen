//! Static options from an MPAS init_atmosphere namelist. Rust owns decoding;
//! the command launcher only forwards the path. Other driver stages retain
//! ownership of their own namelist options.
use std::{collections::BTreeMap, path::{Path, PathBuf}};
use crate::{MpasError, MpasResult, static_builder::{StaticBuildConfig, select_static_source}};

#[derive(Debug, Clone, PartialEq)]
enum Token { Text(String), Equals, Separator }

/// MPAS: mpas_init_atm_static.F:1404-1410,1599-1609. Nonpositive native
/// factors execute no sampling iterations; zero represents that empty map.
pub fn sampling_factor(raw: &str) -> MpasResult<usize> {
    raw.parse::<i32>().map(|value| value.max(0) as usize)
        .map_err(|_| MpasError::Refusal(format!("sampling factor {raw:?} requires a native default integer")))
}

/// MPAS: core_init_atmosphere/Registry.xml:58-218,280-315 (scalar options).
/// Decode quoted Fortran strings and comments before extracting assignments.
fn tokens(text: &str) -> MpasResult<Vec<Token>> {
    let mut chars=text.chars().peekable();
    let mut out=Vec::new();
    while let Some(c)=chars.next() {
        match c {
            '!' => { for c in chars.by_ref() { if c=='\n' {break;} } },
            '\''|'"' => {
                let mut value=String::new(); let mut closed=false;
                while let Some(next)=chars.next() {
                    if next==c {
                        if chars.peek()==Some(&c) {chars.next();value.push(c);}
                        else {closed=true;break;}
                    } else {value.push(next);}
                }
                if !closed {return Err(MpasError::Refusal("unterminated namelist character value".into()));}
                out.push(Token::Text(value));
            },
            '=' => out.push(Token::Equals),
            ','|'/'|'&'|'$' => out.push(Token::Separator),
            c if c.is_whitespace() => {},
            _ => {
                let mut word=String::from(c);
                while chars.peek().is_some_and(|c| !c.is_whitespace() && !['!', '\'', '"', '=', ',', '/', '&', '$'].contains(c)) {
                    word.push(chars.next().unwrap());
                }
                out.push(Token::Text(word));
            },
        }
    }
    Ok(out)
}

/// MPAS: core_init_atmosphere/Registry.xml:58-218,280-315.
#[derive(Debug, Default)]
pub struct StaticNamelist { values: BTreeMap<String,String> }
impl StaticNamelist {
    /// MPAS: core_init_atmosphere/Registry.xml:58-218,280-315.
    pub fn parse(text: &str) -> MpasResult<Self> {
        let words=tokens(text)?;let mut values=BTreeMap::new();
        for triplet in words.windows(3) {
            if let [Token::Text(key),Token::Equals,Token::Text(value)]=triplet {
                values.insert(key.to_ascii_lowercase(),value.clone());
            }
        }
        Ok(Self{values})
    }

    /// MPAS: core_init_atmosphere/Registry.xml:145-151.
    pub fn geog_root(&self) -> Option<PathBuf> {
        self.values.get("config_geog_data_path").map(|s|PathBuf::from(s.trim_end()))
    }

    /// MPAS: core_init_atmosphere/Registry.xml:195-218,292-298.
    fn boolean(&self, name: &str, default: bool) -> MpasResult<bool> {
        match self.values.get(name).map(|s|s.to_ascii_lowercase()).as_deref() {
            None=>Ok(default),Some("true"|".true."|"t")=>Ok(true),Some("false"|".false."|"f")=>Ok(false),
            Some(other)=>Err(MpasError::Refusal(format!("{name} requires a Fortran logical, got {other}"))),
        }
    }

    /// MPAS: core_init_atmosphere/Registry.xml:195-208.
    fn factor(&self, name: &str, default: usize) -> MpasResult<usize> {
        match self.values.get(name) {
            None=>Ok(default),Some(raw)=>sampling_factor(raw),
        }
    }

    /// MPAS: mpas_init_atm_static.F:369-447,488-1250; Registry.xml:145-218,292-298.
    /// Native registry defaults apply only when the native namelist is selected.
    pub fn apply(&self, cfg: &mut StaticBuildConfig, root: &Path) -> MpasResult<()> {
        cfg.stages.native_init=true;
        cfg.supersample=self.factor("config_supersample_factor",3)?;
        cfg.supersample_landuse=self.factor("config_lu_supersample_factor",1)?;
        cfg.supersample_30s=self.factor("config_30s_supersample_factor",1)?;
        cfg.stages.noahmp=self.boolean("config_noahmp_static",true)?;
        cfg.stages.gwd=self.boolean("config_native_gwd_static",true)?;
        if let Some(stamp)=self.values.get("config_start_time") {cfg.valid_time=stamp.trim_end().into();}
        for (name,slot,default) in [
            ("config_topo_data","topo-data","GMTED2010"),
            ("config_landuse_data","landuse-data","MODIFIED_IGBP_MODIS_NOAH"),
            // MPAS-A's registry default is STATSGO (FAO-UNESCO outside CONUS,
            // not permissive); an unset key takes the permissive SoilGrids
            // tiles, and STATSGO is read only when the namelist names it.
            ("config_soilcat_data","soilcat-data","SOILGRIDS"),
            ("config_vegfrac_data","vegfrac-data","MODIS"),
            ("config_albedo_data","albedo-data","MODIS"),
            ("config_maxsnowalbedo_data","maxsnowalbedo-data","MODIS"),
        ] {
            let choice=self.values.get(name).map(|s|s.trim_end()).unwrap_or(default);
            select_static_source(cfg,root,slot,choice)?;
        }
        // MPAS: mpas_init_atm_gwd.F:102-156; Registry.xml:302.
        // Native statics and GWD share the terrain selector, but GWD uses USGS.
        cfg.native_gwd = if cfg.stages.gwd {
            let raw=self.values.get("config_gwd_cell_scaling").map(String::as_str).unwrap_or("1.0");
            let scaling=raw.replace(['d','D'],"e").parse::<f32>()
                .map_err(|_| MpasError::Refusal(format!("config_gwd_cell_scaling requires an RKIND real, got {raw}")))?;
            let topo=self.values.get("config_topo_data").map(|s|s.trim_end()).unwrap_or("GMTED2010");
            Some(crate::native_gwd::NativeGwdConfig::under(root,topo,scaling)?)
        } else {None};
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{static_builder::{StaticGeogPaths,StaticStageOptions,DEFAULT_VALID_TIME},staticfile::coordframe::CoordinateRepresentation};
    fn config() -> StaticBuildConfig {
        StaticBuildConfig {grid_path:"grid".into(),out_path:"out".into(),geog:StaticGeogPaths::under(Path::new("geog")),
            stages:StaticStageOptions::default(),native_gwd:None,supersample:1,supersample_landuse:1,supersample_30s:1,
            host_memory_limit_bytes:None,tile_workers:None,receipt_path:None,provenance:String::new(),
            valid_time:DEFAULT_VALID_TIME.into(),nominal_dx_m:None,clobber:false,
            coordinates:CoordinateRepresentation::Binary32EarthCentred}
    }
    #[test]
    fn static_namelist_matches_native_controls_with_comments_and_quotes() {
        let parsed=StaticNamelist::parse("&data_sources\n CONFIG_GEOG_DATA_PATH='geo!g/it''s', config_landuse_data='USGS' ! comment\n config_supersample_factor=7 config_vegfrac_data='NCEP'\n config_albedo_data='NCEP',config_maxsnowalbedo_data='NCEP' /\n &preproc_stages config_native_gwd_static=.FALSE. config_noahmp_static=F /\n &nhyd_model config_start_time='stamp' /").unwrap();
        assert_eq!(parsed.geog_root().unwrap(),PathBuf::from("geo!g/it's"));
        let mut cfg=config();parsed.apply(&mut cfg,Path::new("root")).unwrap();
        assert_eq!(cfg.supersample,7);assert_eq!(cfg.supersample_landuse,1);
        assert!(!cfg.stages.gwd && !cfg.stages.noahmp && cfg.stages.native_init);
        assert!(cfg.stages.vegetation_ncep && cfg.stages.albedo_ncep && cfg.stages.snow_ncep);
        assert_eq!(cfg.stages.landuse_label.as_deref(),Some("USGS"));assert_eq!(cfg.valid_time,"stamp");
        assert_eq!(cfg.geog.terrain,PathBuf::from("root/topo_gmted2010_30s"));
    }
    #[test]
    fn static_namelist_keeps_defaults_scoped_and_rejects_invalid_controls() {
        let mut cfg=config();StaticNamelist::parse("").unwrap().apply(&mut cfg,Path::new("root")).unwrap();
        assert_eq!(cfg.supersample,3);assert!(cfg.stages.gwd && cfg.stages.noahmp);
        assert_eq!(cfg.native_gwd.as_ref().unwrap().landuse,PathBuf::from("root/landuse_30s"));
        StaticNamelist::parse("config_topo_data='GTOPO30',config_gwd_cell_scaling=2.5D-1").unwrap()
            .apply(&mut cfg,Path::new("root")).unwrap();
        let gwd=cfg.native_gwd.as_ref().unwrap();
        assert_eq!(gwd.topo,PathBuf::from("root/topo_30s"));
        assert_eq!(gwd.cell_scaling.to_bits(),0.25f32.to_bits());
        StaticNamelist::parse("config_native_gwd_static=false").unwrap().apply(&mut cfg,Path::new("root")).unwrap();
        assert!(cfg.native_gwd.is_none());
        assert!(!StaticStageOptions::default().native_init);
        for input in ["config_supersample_factor=1.5","config_noahmp_static=yes","config_topo_data='unknown'"] {
            assert!(StaticNamelist::parse(input).unwrap().apply(&mut config(),Path::new("root")).is_err(),"{input}");
        }
        assert!(StaticNamelist::parse("config_topo_data='unterminated").is_err());
        let n=StaticNamelist::parse("config_supersample_factor=2, config_supersample_factor=5").unwrap();
        n.apply(&mut cfg,Path::new("root")).unwrap();assert_eq!(cfg.supersample,5);
        for input in ["config_supersample_factor=0","config_supersample_factor=-1"] {
            StaticNamelist::parse(input).unwrap().apply(&mut cfg,Path::new("root")).unwrap();
            assert_eq!(cfg.supersample,0);
        }
    }
}

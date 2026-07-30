use rustwx_render::ProductVisualMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DerivedRecipe {
    Sbcape,
    Sbcin,
    Sblcl,
    Mlcape,
    Mlcin,
    Mucape,
    Mucin,
    Dcape,
    Sbecape,
    Mlecape,
    Muecape,
    SbEcapeDerivedCapeRatio,
    MlEcapeDerivedCapeRatio,
    MuEcapeDerivedCapeRatio,
    SbEcapeNativeCapeRatio,
    MlEcapeNativeCapeRatio,
    MuEcapeNativeCapeRatio,
    Sbncape,
    Sbecin,
    Mlecin,
    EcapeScp,
    EcapeEhi01km,
    EcapeEhi03km,
    EcapeStp,
    ThetaE2m10mWinds,
    Vpd2m,
    DewpointDepression2m,
    Wetbulb2m,
    FireWeatherComposite,
    ApparentTemperature2m,
    HeatIndex2m,
    WindChill2m,
    LiftedIndex,
    LapseRate700500,
    LapseRate03km,
    BulkShear01km,
    BulkShear06km,
    Srh01km,
    Srh03km,
    Ehi01km,
    Ehi03km,
    StpFixed,
    ScpMu03km06kmProxy,
    TemperatureAdvection700mb,
    TemperatureAdvection850mb,
}

impl DerivedRecipe {
    pub(super) fn parse(slug: &str) -> Result<Self, String> {
        let normalized = normalize_slug(slug);
        match normalized.as_str() {
            "sbcape" => Ok(Self::Sbcape),
            "sbcin" => Ok(Self::Sbcin),
            "sblcl" => Ok(Self::Sblcl),
            "mlcape" => Ok(Self::Mlcape),
            "mlcin" => Ok(Self::Mlcin),
            "mucape" => Ok(Self::Mucape),
            "mucin" => Ok(Self::Mucin),
            "dcape" | "downdraft_cape" => Ok(Self::Dcape),
            "sbecape" => Ok(Self::Sbecape),
            "mlecape" => Ok(Self::Mlecape),
            "muecape" => Ok(Self::Muecape),
            "sb_ecape_derived_cape_ratio" | "sbecape_derived_cape_ratio" => {
                Ok(Self::SbEcapeDerivedCapeRatio)
            }
            "ml_ecape_derived_cape_ratio" | "mlecape_derived_cape_ratio" => {
                Ok(Self::MlEcapeDerivedCapeRatio)
            }
            "mu_ecape_derived_cape_ratio" | "muecape_derived_cape_ratio" => {
                Ok(Self::MuEcapeDerivedCapeRatio)
            }
            "sb_ecape_native_cape_ratio" | "sbecape_native_cape_ratio" => {
                Ok(Self::SbEcapeNativeCapeRatio)
            }
            "ml_ecape_native_cape_ratio" | "mlecape_native_cape_ratio" => {
                Ok(Self::MlEcapeNativeCapeRatio)
            }
            "mu_ecape_native_cape_ratio" | "muecape_native_cape_ratio" => {
                Ok(Self::MuEcapeNativeCapeRatio)
            }
            "sbncape" => Ok(Self::Sbncape),
            "sbecin" => Ok(Self::Sbecin),
            "mlecin" => Ok(Self::Mlecin),
            "ecape_scp" => Ok(Self::EcapeScp),
            "ecape_ehi" | "ecape_ehi_0_1km" | "ecape_ehi_01km" => Ok(Self::EcapeEhi01km),
            "ecape_ehi_0_3km" | "ecape_ehi_03km" => Ok(Self::EcapeEhi03km),
            "ecape_stp" => Ok(Self::EcapeStp),
            "theta_e_2m_10m_winds" | "2m_theta_e_10m_winds" => {
                Ok(Self::ThetaE2m10mWinds)
            }
            "vpd_2m" | "2m_vpd" | "vapor_pressure_deficit_2m" | "2m_vapor_pressure_deficit" => {
                Ok(Self::Vpd2m)
            }
            "dewpoint_depression_2m" | "2m_dewpoint_depression" => {
                Ok(Self::DewpointDepression2m)
            }
            "wetbulb_2m" | "wet_bulb_2m" | "2m_wetbulb" | "2m_wet_bulb" => {
                Ok(Self::Wetbulb2m)
            }
            "fire_weather_composite" | "fire_weather" | "fire_wx" => {
                Ok(Self::FireWeatherComposite)
            }
            "apparent_temperature_2m" | "2m_apparent_temperature" => {
                Ok(Self::ApparentTemperature2m)
            }
            "heat_index_2m" | "2m_heat_index" => Ok(Self::HeatIndex2m),
            "wind_chill_2m" | "2m_wind_chill" => Ok(Self::WindChill2m),
            "lifted_index" => Ok(Self::LiftedIndex),
            "lapse_rate_700_500" => Ok(Self::LapseRate700500),
            "lapse_rate_0_3km" => Ok(Self::LapseRate03km),
            "bulk_shear_0_1km" => Ok(Self::BulkShear01km),
            "bulk_shear_0_6km" => Ok(Self::BulkShear06km),
            "srh_0_1km" => Ok(Self::Srh01km),
            "srh_0_3km" => Ok(Self::Srh03km),
            "ehi_0_1km" | "ehi_sb_0_1km_proxy" => Ok(Self::Ehi01km),
            "ehi_0_3km" | "ehi_sb_0_3km_proxy" => Ok(Self::Ehi03km),
            "stp_fixed" => Ok(Self::StpFixed),
            "scp_mu_0_3km_0_6km_proxy" => Ok(Self::ScpMu03km06kmProxy),
            "temperature_advection_700mb" => Ok(Self::TemperatureAdvection700mb),
            "temperature_advection_850mb" => Ok(Self::TemperatureAdvection850mb),
            "stp_effective" => Err(
                "stp_effective requires mixed-layer CAPE/CIN/LCL plus effective SRH and effective bulk wind difference; rustwx-products does not yet derive effective SRH or EBWD from HRRR profiles".into(),
            ),
            "scp" | "scp_effective" => Err(
                "scp/scp_effective require effective SRH and effective bulk wind difference; rustwx-products does not yet derive those effective-layer kinematics from HRRR profiles".into(),
            ),
            other => Err(format!("unsupported derived recipe '{other}'")),
        }
    }

    pub(super) fn slug(self) -> &'static str {
        match self {
            Self::Sbcape => "sbcape",
            Self::Sbcin => "sbcin",
            Self::Sblcl => "sblcl",
            Self::Mlcape => "mlcape",
            Self::Mlcin => "mlcin",
            Self::Mucape => "mucape",
            Self::Mucin => "mucin",
            Self::Dcape => "dcape",
            Self::Sbecape => "sbecape",
            Self::Mlecape => "mlecape",
            Self::Muecape => "muecape",
            Self::SbEcapeDerivedCapeRatio => "sb_ecape_derived_cape_ratio",
            Self::MlEcapeDerivedCapeRatio => "ml_ecape_derived_cape_ratio",
            Self::MuEcapeDerivedCapeRatio => "mu_ecape_derived_cape_ratio",
            Self::SbEcapeNativeCapeRatio => "sb_ecape_native_cape_ratio",
            Self::MlEcapeNativeCapeRatio => "ml_ecape_native_cape_ratio",
            Self::MuEcapeNativeCapeRatio => "mu_ecape_native_cape_ratio",
            Self::Sbncape => "sbncape",
            Self::Sbecin => "sbecin",
            Self::Mlecin => "mlecin",
            Self::EcapeScp => "ecape_scp",
            Self::EcapeEhi01km => "ecape_ehi_0_1km",
            Self::EcapeEhi03km => "ecape_ehi_0_3km",
            Self::EcapeStp => "ecape_stp",
            Self::ThetaE2m10mWinds => "theta_e_2m_10m_winds",
            Self::Vpd2m => "vpd_2m",
            Self::DewpointDepression2m => "dewpoint_depression_2m",
            Self::Wetbulb2m => "wetbulb_2m",
            Self::FireWeatherComposite => "fire_weather_composite",
            Self::ApparentTemperature2m => "apparent_temperature_2m",
            Self::HeatIndex2m => "heat_index_2m",
            Self::WindChill2m => "wind_chill_2m",
            Self::LiftedIndex => "lifted_index",
            Self::LapseRate700500 => "lapse_rate_700_500",
            Self::LapseRate03km => "lapse_rate_0_3km",
            Self::BulkShear01km => "bulk_shear_0_1km",
            Self::BulkShear06km => "bulk_shear_0_6km",
            Self::Srh01km => "srh_0_1km",
            Self::Srh03km => "srh_0_3km",
            Self::Ehi01km => "ehi_0_1km",
            Self::Ehi03km => "ehi_0_3km",
            Self::StpFixed => "stp_fixed",
            Self::ScpMu03km06kmProxy => "scp_mu_0_3km_0_6km_proxy",
            Self::TemperatureAdvection700mb => "temperature_advection_700mb",
            Self::TemperatureAdvection850mb => "temperature_advection_850mb",
        }
    }

    pub(super) fn title(self) -> &'static str {
        match self {
            Self::Sbcape => "SBCAPE",
            Self::Sbcin => "SBCIN",
            Self::Sblcl => "SBLCL",
            Self::Mlcape => "MLCAPE",
            Self::Mlcin => "MLCIN",
            Self::Mucape => "MUCAPE",
            Self::Mucin => "MUCIN",
            Self::Dcape => "DCAPE",
            Self::Sbecape => "SBECAPE",
            Self::Mlecape => "MLECAPE",
            Self::Muecape => "MUECAPE",
            Self::SbEcapeDerivedCapeRatio => "SB ECAPE / Derived CAPE Ratio (EXP)",
            Self::MlEcapeDerivedCapeRatio => "ML ECAPE / Derived CAPE Ratio (EXP)",
            Self::MuEcapeDerivedCapeRatio => "MU ECAPE / Derived CAPE Ratio (EXP)",
            Self::SbEcapeNativeCapeRatio => "SB ECAPE / Native CAPE Ratio (EXP)",
            Self::MlEcapeNativeCapeRatio => "ML ECAPE / Native CAPE Ratio (EXP)",
            Self::MuEcapeNativeCapeRatio => "MU ECAPE / Native CAPE Ratio (EXP)",
            Self::Sbncape => "SBNCAPE",
            Self::Sbecin => "SBECIN",
            Self::Mlecin => "MLECIN",
            Self::EcapeScp => "ECAPE SCP (EXP)",
            Self::EcapeEhi01km => "ECAPE EHI 0-1 km (EXP)",
            Self::EcapeEhi03km => "ECAPE EHI 0-3 km (EXP)",
            Self::EcapeStp => "ECAPE STP (EXP)",
            Self::ThetaE2m10mWinds => "2 m Theta-e, 10 m Wind",
            Self::Vpd2m => "2 m Vapor Pressure Deficit",
            Self::DewpointDepression2m => "2 m Dewpoint Depression",
            Self::Wetbulb2m => "2 m Wet-Bulb Temperature",
            Self::FireWeatherComposite => "Fire Weather Composite",
            Self::ApparentTemperature2m => "2 m Apparent Temperature",
            Self::HeatIndex2m => "2 m Heat Index",
            Self::WindChill2m => "2 m Wind Chill",
            Self::LiftedIndex => "Surface-Based Lifted Index",
            Self::LapseRate700500 => "700-500 mb Virtual Temperature Lapse Rate",
            Self::LapseRate03km => "0-3 km Lapse Rate",
            Self::BulkShear01km => "0-1 km Bulk Shear",
            Self::BulkShear06km => "0-6 km Bulk Shear",
            Self::Srh01km => "0-1 km SRH",
            Self::Srh03km => "0-3 km SRH",
            Self::Ehi01km => "EHI 0-1 km",
            Self::Ehi03km => "EHI 0-3 km",
            Self::StpFixed => "STP (FIXED)",
            Self::ScpMu03km06kmProxy => "SCP (MU / 0-3 km / 0-6 km PROXY)",
            Self::TemperatureAdvection700mb => "700 mb Temperature Advection",
            Self::TemperatureAdvection850mb => "850 mb Temperature Advection",
        }
    }

    pub(super) fn visual_mode(self) -> ProductVisualMode {
        match self {
            Self::ThetaE2m10mWinds
            | Self::TemperatureAdvection700mb
            | Self::TemperatureAdvection850mb => ProductVisualMode::UpperAirAnalysis,
            Self::Vpd2m
            | Self::DewpointDepression2m
            | Self::Wetbulb2m
            | Self::ApparentTemperature2m
            | Self::HeatIndex2m
            | Self::WindChill2m => ProductVisualMode::FilledMeteorology,
            _ => ProductVisualMode::SevereDiagnostic,
        }
    }

    pub(super) fn is_heavy(self) -> bool {
        matches!(
            self,
            Self::Sbecape
                | Self::Mlecape
                | Self::Muecape
                | Self::SbEcapeDerivedCapeRatio
                | Self::MlEcapeDerivedCapeRatio
                | Self::MuEcapeDerivedCapeRatio
                | Self::SbEcapeNativeCapeRatio
                | Self::MlEcapeNativeCapeRatio
                | Self::MuEcapeNativeCapeRatio
                | Self::Sbncape
                | Self::Sbecin
                | Self::Mlecin
                | Self::EcapeScp
                | Self::EcapeEhi01km
                | Self::EcapeEhi03km
                | Self::EcapeStp
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DerivedRequirements {
    pub(super) sb: bool,
    pub(super) ml: bool,
    pub(super) mu: bool,
    pub(super) surface_thermo: bool,
    pub(super) surface_winds: bool,
    pub(super) dcape: bool,
    pub(super) lifted_index: bool,
    pub(super) lapse_rate_700_500: bool,
    pub(super) lapse_rate_0_3km: bool,
    pub(super) shear_01km: bool,
    pub(super) shear_06km: bool,
    pub(super) srh_01km: bool,
    pub(super) srh_03km: bool,
    pub(super) ehi_01km: bool,
    pub(super) ehi_03km: bool,
    pub(super) stp_fixed: bool,
    pub(super) scp_mu_03km_06km_proxy: bool,
    pub(super) temperature_advection_700mb: bool,
    pub(super) temperature_advection_850mb: bool,
}

impl DerivedRequirements {
    pub(super) fn from_recipes(recipes: &[DerivedRecipe]) -> Self {
        let mut requirements = Self::default();
        for &recipe in recipes {
            match recipe {
                DerivedRecipe::Sbcape | DerivedRecipe::Sbcin | DerivedRecipe::Sblcl => {
                    requirements.sb = true;
                }
                DerivedRecipe::Mlcape | DerivedRecipe::Mlcin => {
                    requirements.ml = true;
                }
                DerivedRecipe::Mucape | DerivedRecipe::Mucin => {
                    requirements.mu = true;
                }
                DerivedRecipe::Dcape => {
                    requirements.dcape = true;
                }
                DerivedRecipe::ThetaE2m10mWinds => {
                    requirements.surface_thermo = true;
                    requirements.surface_winds = true;
                }
                DerivedRecipe::Vpd2m
                | DerivedRecipe::DewpointDepression2m
                | DerivedRecipe::Wetbulb2m
                | DerivedRecipe::FireWeatherComposite
                | DerivedRecipe::ApparentTemperature2m
                | DerivedRecipe::HeatIndex2m
                | DerivedRecipe::WindChill2m => {
                    requirements.surface_thermo = true;
                }
                DerivedRecipe::LiftedIndex => {
                    requirements.lifted_index = true;
                }
                DerivedRecipe::LapseRate700500 => {
                    requirements.lapse_rate_700_500 = true;
                }
                DerivedRecipe::LapseRate03km => {
                    requirements.lapse_rate_0_3km = true;
                }
                DerivedRecipe::BulkShear01km => {
                    requirements.shear_01km = true;
                }
                DerivedRecipe::BulkShear06km => {
                    requirements.shear_06km = true;
                }
                DerivedRecipe::Srh01km => {
                    requirements.srh_01km = true;
                }
                DerivedRecipe::Srh03km => {
                    requirements.srh_03km = true;
                }
                DerivedRecipe::Ehi01km => {
                    requirements.ehi_01km = true;
                    requirements.sb = true;
                    requirements.srh_01km = true;
                }
                DerivedRecipe::Ehi03km => {
                    requirements.ehi_03km = true;
                    requirements.sb = true;
                    requirements.srh_03km = true;
                }
                DerivedRecipe::StpFixed => {
                    requirements.stp_fixed = true;
                    requirements.sb = true;
                    requirements.srh_01km = true;
                    requirements.shear_06km = true;
                }
                DerivedRecipe::ScpMu03km06kmProxy => {
                    requirements.scp_mu_03km_06km_proxy = true;
                    requirements.mu = true;
                    requirements.srh_03km = true;
                    requirements.shear_06km = true;
                }
                DerivedRecipe::TemperatureAdvection700mb => {
                    requirements.temperature_advection_700mb = true;
                }
                DerivedRecipe::TemperatureAdvection850mb => {
                    requirements.temperature_advection_850mb = true;
                }
                DerivedRecipe::Sbecape
                | DerivedRecipe::Mlecape
                | DerivedRecipe::Muecape
                | DerivedRecipe::SbEcapeDerivedCapeRatio
                | DerivedRecipe::MlEcapeDerivedCapeRatio
                | DerivedRecipe::MuEcapeDerivedCapeRatio
                | DerivedRecipe::SbEcapeNativeCapeRatio
                | DerivedRecipe::MlEcapeNativeCapeRatio
                | DerivedRecipe::MuEcapeNativeCapeRatio
                | DerivedRecipe::Sbncape
                | DerivedRecipe::Sbecin
                | DerivedRecipe::Mlecin
                | DerivedRecipe::EcapeScp
                | DerivedRecipe::EcapeEhi01km
                | DerivedRecipe::EcapeEhi03km
                | DerivedRecipe::EcapeStp => {}
            }
        }
        requirements
    }

    pub(super) fn needs_volume(self) -> bool {
        self.sb
            || self.ml
            || self.mu
            || self.dcape
            || self.lifted_index
            || self.lapse_rate_700_500
            || self.lapse_rate_0_3km
    }

    pub(super) fn needs_height_agl(self) -> bool {
        self.needs_volume() || self.shear_01km || self.shear_06km || self.srh_01km || self.srh_03km
    }

    pub(super) fn needs_grid_spacing(self) -> bool {
        self.temperature_advection_700mb || self.temperature_advection_850mb
    }

    pub(super) fn needs_pressure_fields(self) -> bool {
        self.needs_volume() || self.needs_height_agl() || self.needs_grid_spacing()
    }
}

pub(crate) fn derived_compute_recipes_need_pressure(recipes: &[DerivedRecipe]) -> bool {
    DerivedRequirements::from_recipes(recipes).needs_pressure_fields()
}

fn normalize_slug(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

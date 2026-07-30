/// Look up the human-readable name of a GRIB2 parameter.
/// Based on WMO GRIB2 Code Table 4.2 and NCEP local-use extensions.
/// Returns "Unknown" for unrecognized combinations.
pub fn parameter_name(discipline: u8, category: u8, number: u8) -> &'static str {
    match (discipline, category, number) {
        // =====================================================================
        // Discipline 0: Meteorological Products
        // =====================================================================

        // Category 0: Temperature
        (0, 0, 0) => "Temperature",
        (0, 0, 1) => "Virtual Temperature",
        (0, 0, 2) => "Potential Temperature",
        (0, 0, 3) => "Pseudo-Adiabatic Potential Temperature",
        (0, 0, 4) => "Maximum Temperature",
        (0, 0, 5) => "Minimum Temperature",
        (0, 0, 6) => "Dewpoint Temperature",
        (0, 0, 7) => "Dewpoint Depression",
        (0, 0, 8) => "Lapse Rate",
        (0, 0, 9) => "Temperature Anomaly",
        (0, 0, 10) => "Latent Heat Net Flux",
        (0, 0, 11) => "Sensible Heat Net Flux",
        (0, 0, 12) => "Heat Index",
        (0, 0, 13) => "Wind Chill Factor",
        (0, 0, 14) => "Minimum Dewpoint Depression",
        (0, 0, 15) => "Virtual Potential Temperature",
        (0, 0, 16) => "Snow Phase Change Heat Flux",
        (0, 0, 17) => "Skin Temperature",
        (0, 0, 18) => "Snow Temperature (top of snow)",
        (0, 0, 19) => "Turbulent Transfer Coefficient for Heat",
        (0, 0, 20) => "Turbulent Diffusion Coefficient for Heat",
        (0, 0, 21) => "Apparent Temperature",
        (0, 0, 22) => "Temperature Tendency due to Short-Wave Radiation",
        (0, 0, 23) => "Temperature Tendency due to Long-Wave Radiation",
        (0, 0, 24) => "Temperature Tendency due to Short-Wave Radiation, Clear Sky",
        (0, 0, 25) => "Temperature Tendency due to Long-Wave Radiation, Clear Sky",
        (0, 0, 26) => "Temperature Tendency due to Parameterizations",
        (0, 0, 27) => "Wet Bulb Temperature",
        (0, 0, 28) => "Unbalanced Component of Temperature",
        (0, 0, 29) => "Temperature Advection",
        (0, 0, 30) => "Latent Heat Net Flux Due to Evaporation",
        (0, 0, 31) => "Latent Heat Net Flux Due to Sublimation",
        (0, 0, 32) => "Wet-Bulb Potential Temperature",
        // NCEP Local Use
        (0, 0, 192) => "Snow Phase Change Heat Flux",
        (0, 0, 193) => "Temperature Tendency by All Radiation",
        (0, 0, 194) => "Relative Error Variance",
        (0, 0, 195) => "Large Scale Condensate Heating Rate",
        (0, 0, 196) => "Deep Convective Heating Rate",
        (0, 0, 197) => "Total Downward Heat Flux at Surface",
        (0, 0, 198) => "Temperature Tendency by All Physics",
        (0, 0, 199) => "Temperature Tendency by Non-radiation Physics",
        (0, 0, 200) => "Standard Dev. of IR Temp. over 1x1 deg. area",
        (0, 0, 201) => "Shallow Convective Heating Rate",
        (0, 0, 202) => "Vertical Diffusion Heating rate",
        (0, 0, 203) => "Potential Temperature at Top of Viscous Sublayer",
        (0, 0, 204) => "Tropical Cyclone Heat Potential",

        // Category 1: Moisture
        (0, 1, 0) => "Specific Humidity",
        (0, 1, 1) => "Relative Humidity",
        (0, 1, 2) => "Humidity Mixing Ratio",
        (0, 1, 3) => "Precipitable Water",
        (0, 1, 4) => "Vapor Pressure",
        (0, 1, 5) => "Saturation Deficit",
        (0, 1, 6) => "Evaporation",
        (0, 1, 7) => "Precipitation Rate",
        (0, 1, 8) => "Total Precipitation",
        (0, 1, 9) => "Large-Scale Precipitation (non-convective)",
        (0, 1, 10) => "Convective Precipitation",
        (0, 1, 11) => "Snow Depth",
        (0, 1, 12) => "Snowfall Rate Water Equivalent",
        (0, 1, 13) => "Water Equivalent of Accumulated Snow Depth",
        (0, 1, 14) => "Convective Snow",
        (0, 1, 15) => "Large-Scale Snow",
        (0, 1, 16) => "Snow Melt",
        (0, 1, 17) => "Snow Age",
        (0, 1, 18) => "Absolute Humidity",
        (0, 1, 19) => "Precipitation Type",
        (0, 1, 20) => "Integrated Liquid Water",
        (0, 1, 21) => "Condensate",
        (0, 1, 22) => "Cloud Mixing Ratio",
        (0, 1, 23) => "Ice Water Mixing Ratio",
        (0, 1, 24) => "Rain Mixing Ratio",
        (0, 1, 25) => "Snow Mixing Ratio",
        (0, 1, 26) => "Horizontal Moisture Convergence",
        (0, 1, 27) => "Maximum Relative Humidity",
        (0, 1, 28) => "Maximum Absolute Humidity",
        (0, 1, 29) => "Total Snowfall",
        (0, 1, 30) => "Precipitable Water Category",
        (0, 1, 31) => "Hail",
        (0, 1, 32) => "Graupel",
        (0, 1, 33) => "Categorical Rain",
        (0, 1, 34) => "Categorical Freezing Rain",
        (0, 1, 35) => "Categorical Ice Pellets",
        (0, 1, 36) => "Categorical Snow",
        (0, 1, 37) => "Convective Precipitation Rate",
        (0, 1, 38) => "Horizontal Moisture Divergence",
        (0, 1, 39) => "Percent Frozen Precipitation",
        (0, 1, 40) => "Potential Evaporation",
        (0, 1, 41) => "Potential Evaporation Rate",
        (0, 1, 42) => "Snow Cover",
        (0, 1, 43) => "Rain Fraction of Total Cloud Water",
        (0, 1, 44) => "Rime Factor",
        (0, 1, 45) => "Total Column Integrated Rain",
        (0, 1, 46) => "Total Column Integrated Snow",
        (0, 1, 47) => "Large Scale Water Precipitation (Non-Convective)",
        (0, 1, 48) => "Convective Water Precipitation",
        (0, 1, 49) => "Total Water Precipitation",
        (0, 1, 50) => "Total Snow Precipitation",
        (0, 1, 51) => "Total Column Water (Vertically integrated total water)",
        (0, 1, 52) => "Total Precipitation Rate",
        (0, 1, 53) => "Total Snowfall Rate Water Equivalent",
        (0, 1, 54) => "Large Scale Precipitation Rate",
        (0, 1, 55) => "Convective Snowfall Rate Water Equivalent",
        (0, 1, 56) => "Large Scale Snowfall Rate Water Equivalent",
        (0, 1, 57) => "Total Snowfall Rate",
        (0, 1, 58) => "Convective Snowfall Rate",
        (0, 1, 59) => "Large Scale Snowfall Rate",
        (0, 1, 60) => "Snow Depth Water Equivalent",
        (0, 1, 61) => "Snow Density",
        (0, 1, 62) => "Snow Evaporation",
        (0, 1, 64) => "Total Column Integrated Water Vapour",
        (0, 1, 65) => "Rain Precipitation Rate",
        (0, 1, 66) => "Snow Precipitation Rate",
        (0, 1, 67) => "Freezing Rain Precipitation Rate",
        (0, 1, 68) => "Ice Pellets Precipitation Rate",
        (0, 1, 69) => "Total Column Integrate Cloud Water",
        (0, 1, 70) => "Total Column Integrate Cloud Ice",
        (0, 1, 71) => "Hail Mixing Ratio",
        (0, 1, 72) => "Total Column Integrate Hail",
        (0, 1, 73) => "Hail Prepitation Rate",
        (0, 1, 74) => "Total Column Integrate Graupel",
        (0, 1, 75) => "Graupel Precipitation Rate",
        (0, 1, 76) => "Convective Rain Rate",
        (0, 1, 77) => "Large Scale Rain Rate",
        (0, 1, 78) => "Total Column Integrate Water (All components)",
        (0, 1, 79) => "Evaporation Rate",
        (0, 1, 80) => "Total Condensate",
        (0, 1, 81) => "Total Column-Integrate Condensate",
        (0, 1, 82) => "Cloud Ice Mixing Ratio",
        (0, 1, 83) => "Specific Cloud Liquid Water Content",
        (0, 1, 84) => "Specific Cloud Ice Water Content",
        (0, 1, 85) => "Specific Rain Water Content",
        (0, 1, 86) => "Specific Snow Water Content",
        (0, 1, 90) => "Total Kinematic Moisture Flux",
        (0, 1, 91) => "U-component of Moisture Flux",
        (0, 1, 92) => "V-component of Moisture Flux",
        (0, 1, 99) => "Liquid Precipitation Depth",
        (0, 1, 100) => "Liquid Precipitation Rate (Instantaneous)",
        // NCEP Local Use
        (0, 1, 192) => "Categorical Rain (yes=1; no=0)",
        (0, 1, 193) => "Categorical Freezing Rain (yes=1; no=0)",
        (0, 1, 194) => "Categorical Ice Pellets (yes=1; no=0)",
        (0, 1, 195) => "Categorical Snow (yes=1; no=0)",
        (0, 1, 196) => "Convective Precipitation Rate",
        (0, 1, 197) => "Horizontal Moisture Divergence",
        (0, 1, 198) => "Minimum Relative Humidity",
        (0, 1, 199) => "Potential Evaporation",
        (0, 1, 200) => "Potential Evaporation Rate",
        (0, 1, 201) => "Snow Cover",
        (0, 1, 202) => "Rain Fraction of Total Liquid Water",
        (0, 1, 203) => "Rime Factor",
        (0, 1, 204) => "Total Column Integrated Rain",
        (0, 1, 205) => "Total Column Integrated Snow",
        (0, 1, 206) => "Total Icing Potential (Diagnostic)",
        (0, 1, 207) => "Number of Freezing Levels",
        (0, 1, 208) => "Reflectivity",
        (0, 1, 209) => "Composite Reflectivity (Entire Atmosphere)",
        (0, 1, 210) => "Maximum/Composite Echo Top",
        (0, 1, 211) => "Supercooled Large Droplet (SLD) Icing",
        (0, 1, 212) => "Hourly Maximum of Column Vertically Integrated Graupel",
        (0, 1, 213) => "Frozen Rain",
        (0, 1, 214) => "Freezing Drizzle",
        (0, 1, 215) => "Freezing Light Rain",
        (0, 1, 216) => "Predominant Weather",
        (0, 1, 225) => "Total Precipitation (nearest grid point)",
        (0, 1, 227) => "Drag Coefficient",
        (0, 1, 241) => "Total Snow",
        (0, 1, 242) => "Relative Humidity with Respect to Precipitable Water",

        // Category 2: Momentum
        (0, 2, 0) => "Wind Direction (from which blowing)",
        (0, 2, 1) => "Wind Speed",
        (0, 2, 2) => "U-Component of Wind",
        (0, 2, 3) => "V-Component of Wind",
        (0, 2, 4) => "Stream Function",
        (0, 2, 5) => "Velocity Potential",
        (0, 2, 6) => "Montgomery Stream Function",
        (0, 2, 7) => "Sigma Coordinate Vertical Velocity",
        (0, 2, 8) => "Vertical Velocity (Pressure)",
        (0, 2, 9) => "Vertical Velocity (Geometric)",
        (0, 2, 10) => "Absolute Vorticity",
        (0, 2, 11) => "Absolute Divergence",
        (0, 2, 12) => "Relative Vorticity",
        (0, 2, 13) => "Relative Divergence",
        (0, 2, 14) => "Potential Vorticity",
        (0, 2, 15) => "Vertical U-Component Shear",
        (0, 2, 16) => "Vertical V-Component Shear",
        (0, 2, 17) => "Momentum Flux, U-Component",
        (0, 2, 18) => "Momentum Flux, V-Component",
        (0, 2, 19) => "Wind Mixing Energy",
        (0, 2, 20) => "Boundary Layer Dissipation",
        (0, 2, 21) => "Maximum Wind Speed",
        (0, 2, 22) => "Wind Speed (Gust)",
        (0, 2, 23) => "U-Component of Wind (Gust)",
        (0, 2, 24) => "V-Component of Wind (Gust)",
        (0, 2, 25) => "Vertical Speed Shear",
        (0, 2, 26) => "Horizontal Momentum Flux",
        (0, 2, 27) => "U-Component Storm Motion",
        (0, 2, 28) => "V-Component Storm Motion",
        (0, 2, 29) => "Drag Coefficient",
        (0, 2, 30) => "Frictional Velocity",
        (0, 2, 31) => "Turbulent Diffusion Coefficient for Momentum",
        (0, 2, 32) => "Eta Coordinate Vertical Velocity",
        (0, 2, 33) => "Wind Fetch",
        (0, 2, 34) => "Normal Wind Component",
        (0, 2, 35) => "Tangential Wind Component",
        (0, 2, 36) => "Areal Mean Wind Speed",
        (0, 2, 37) => "Stress Tensor U-Component",
        (0, 2, 38) => "Stress Tensor V-Component",
        (0, 2, 39) => "Unbalanced Component of Divergence",
        (0, 2, 40) => "Wind Speed Probability",
        // NCEP Local Use
        (0, 2, 192) => "Vertical Speed Shear",
        (0, 2, 193) => "Horizontal Momentum Flux",
        (0, 2, 194) => "U-Component of Friction Velocity",
        (0, 2, 195) => "V-Component of Friction Velocity",
        (0, 2, 196) => "Wind Gust Speed",
        (0, 2, 197) => "U-Component of Wind (at 10m)",
        (0, 2, 198) => "V-Component of Wind (at 10m)",
        (0, 2, 199) => "Ventilation Rate",
        (0, 2, 200) => "Transport Wind Speed",
        (0, 2, 201) => "Transport Wind Direction",
        (0, 2, 202) => "Estimated U-Component of Wind",
        (0, 2, 203) => "Estimated V-Component of Wind",
        (0, 2, 204) => "Mixing Coefficient",
        (0, 2, 220) => "Hourly Maximum of Upward Vertical Velocity",
        (0, 2, 221) => "Hourly Maximum of Downward Vertical Velocity",
        (0, 2, 222) => "U Component of Hourly Maximum of 10m Wind Speed",
        (0, 2, 223) => "V Component of Hourly Maximum of 10m Wind Speed",
        (0, 2, 224) => "Ventilation Rate",
        (0, 2, 225) => "Transport Wind Speed",
        (0, 2, 226) => "Transport Wind Direction",
        (0, 2, 227) => "Earliest Reasonable Arrival Time (10-knot)",
        (0, 2, 228) => "Most Likely Arrival Time (10-knot)",
        (0, 2, 229) => "Most Likely Departure Time (10-knot)",
        (0, 2, 230) => "Latest Reasonable Departure Time (10-knot)",
        (0, 2, 231) => "Tropical Wind Direction",
        (0, 2, 232) => "Tropical Wind Speed",

        // Category 3: Mass
        (0, 3, 0) => "Pressure",
        (0, 3, 1) => "Pressure Reduced to MSL",
        (0, 3, 2) => "Pressure Tendency",
        (0, 3, 3) => "ICAO Standard Atmosphere Reference Height",
        (0, 3, 4) => "Geopotential",
        (0, 3, 5) => "Geopotential Height",
        (0, 3, 6) => "Geometric Height",
        (0, 3, 7) => "Standard Deviation of Height",
        (0, 3, 8) => "Pressure Anomaly",
        (0, 3, 9) => "Geopotential Height Anomaly",
        (0, 3, 10) => "Density",
        (0, 3, 11) => "Altimeter Setting",
        (0, 3, 12) => "Thickness",
        (0, 3, 13) => "Pressure Altitude",
        (0, 3, 14) => "Density Altitude",
        (0, 3, 15) => "5-Wave Geopotential Height",
        (0, 3, 16) => "Zonal Flux of Gravity Wave Stress",
        (0, 3, 17) => "Meridional Flux of Gravity Wave Stress",
        (0, 3, 18) => "Planetary Boundary Layer Height",
        (0, 3, 19) => "5-Wave Geopotential Height Anomaly",
        (0, 3, 20) => "Standard Deviation of Sub-Grid Scale Orography",
        (0, 3, 21) => "Angle of Sub-Grid Scale Orography",
        (0, 3, 22) => "Slope of Sub-Grid Scale Orography",
        (0, 3, 23) => "Gravity Wave Dissipation",
        (0, 3, 24) => "Anisotropy of Sub-Grid Scale Orography",
        (0, 3, 25) => "Natural Logarithm of Pressure in Pa",
        (0, 3, 26) => "Exner Pressure",
        (0, 3, 27) => "Updraft Mass Flux",
        (0, 3, 28) => "Downdraft Mass Flux",
        // NCEP Local Use
        (0, 3, 192) => "MSLP (Eta Reduction)",
        (0, 3, 193) => "5-Wave Geopotential Height",
        (0, 3, 194) => "Zonal Flux of Gravity Wave Stress",
        (0, 3, 195) => "Meridional Flux of Gravity Wave Stress",
        (0, 3, 196) => "Planetary Boundary Layer Height",
        (0, 3, 197) => "5-Wave Geopotential Height Anomaly",
        (0, 3, 198) => "MSLP (MAPS System Reduction)",
        (0, 3, 199) => "3-hr Pressure Tendency (Std. Atmos. Reduction)",
        (0, 3, 200) => "Pressure of Level from which Parcel was Lifted",
        (0, 3, 201) => "X-Gradient of Log Pressure",
        (0, 3, 202) => "Y-Gradient of Log Pressure",
        (0, 3, 203) => "X-Gradient of Height",
        (0, 3, 204) => "Y-Gradient of Height",
        (0, 3, 205) => "Layer Thickness",
        (0, 3, 206) => "Natural Log of Surface Pressure",
        (0, 3, 207) => "Convective Updraft Mass Flux",
        (0, 3, 208) => "Convective Downdraft Mass Flux",
        (0, 3, 209) => "Convective Detrainment Mass Flux",
        (0, 3, 210) => "Mass Point Model Surface",
        (0, 3, 211) => "Geopotential Height (nearest grid point)",
        (0, 3, 212) => "Pressure (nearest grid point)",

        // Category 4: Short-wave Radiation
        (0, 4, 0) => "Net Short-Wave Radiation Flux (Surface)",
        (0, 4, 1) => "Net Short-Wave Radiation Flux (Top of Atmosphere)",
        (0, 4, 2) => "Short-Wave Radiation Flux",
        (0, 4, 3) => "Global Radiation Flux",
        (0, 4, 4) => "Brightness Temperature",
        (0, 4, 5) => "Radiance (with respect to wave number)",
        (0, 4, 6) => "Radiance (with respect to wavelength)",
        (0, 4, 7) => "Downward Short-Wave Radiation Flux",
        (0, 4, 8) => "Upward Short-Wave Radiation Flux",
        (0, 4, 9) => "Net Short-Wave Radiation Flux",
        (0, 4, 10) => "Photosynthetically Active Radiation",
        (0, 4, 11) => "Net Short-Wave Radiation Flux, Clear Sky",
        (0, 4, 12) => "Downward UV Radiation",
        (0, 4, 50) => "UV Index (Under Clear Sky)",
        (0, 4, 51) => "UV Index",
        (0, 4, 52) => "Downward Short-Wave Radiation Flux, Clear Sky",
        (0, 4, 53) => "Upward Short-Wave Radiation Flux, Clear Sky",
        // NCEP Local Use
        (0, 4, 192) => "Downward Short-Wave Radiation Flux",
        (0, 4, 193) => "Upward Short-Wave Radiation Flux",
        (0, 4, 194) => "UV-B Downward Solar Flux",
        (0, 4, 195) => "Clear Sky UV-B Downward Solar Flux",
        (0, 4, 196) => "Clear Sky Downward Solar Flux",
        (0, 4, 197) => "Solar Radiative Heating Rate",
        (0, 4, 198) => "Clear Sky Upward Solar Flux",
        (0, 4, 199) => "Cloud Forcing Net Solar Flux",
        (0, 4, 200) => "Visible Beam Downward Solar Flux",
        (0, 4, 201) => "Visible Diffuse Downward Solar Flux",
        (0, 4, 202) => "Near IR Beam Downward Solar Flux",
        (0, 4, 203) => "Near IR Diffuse Downward Solar Flux",
        (0, 4, 204) => "Downward Total Radiation Flux",
        (0, 4, 205) => "Upward Total Radiation Flux",

        // Category 5: Long-wave Radiation
        (0, 5, 0) => "Net Long-Wave Radiation Flux (Surface)",
        (0, 5, 1) => "Net Long-Wave Radiation Flux (Top of Atmosphere)",
        (0, 5, 2) => "Long-Wave Radiation Flux",
        (0, 5, 3) => "Downward Long-Wave Radiation Flux",
        (0, 5, 4) => "Upward Long-Wave Radiation Flux",
        (0, 5, 5) => "Net Long-Wave Radiation Flux",
        (0, 5, 6) => "Net Long-Wave Radiation Flux, Clear Sky",
        (0, 5, 7) => "Brightness Temperature",
        (0, 5, 8) => "Downward Long-Wave Radiation Flux, Clear Sky",
        // NCEP Local Use
        (0, 5, 192) => "Downward Long-Wave Radiation Flux",
        (0, 5, 193) => "Upward Long-Wave Radiation Flux",
        (0, 5, 194) => "Long-Wave Radiative Heating Rate",
        (0, 5, 195) => "Clear Sky Upward Long-Wave Flux",
        (0, 5, 196) => "Clear Sky Downward Long-Wave Flux",
        (0, 5, 197) => "Cloud Forcing Net Long-Wave Flux",

        // Category 6: Cloud
        (0, 6, 0) => "Cloud Ice",
        (0, 6, 1) => "Total Cloud Cover",
        (0, 6, 2) => "Convective Cloud Cover",
        (0, 6, 3) => "Low Cloud Cover",
        (0, 6, 4) => "Medium Cloud Cover",
        (0, 6, 5) => "High Cloud Cover",
        (0, 6, 6) => "Cloud Water",
        (0, 6, 7) => "Cloud Amount",
        (0, 6, 8) => "Cloud Type",
        (0, 6, 9) => "Thunderstorm Maximum Tops",
        (0, 6, 10) => "Thunderstorm Coverage",
        (0, 6, 11) => "Cloud Base",
        (0, 6, 12) => "Cloud Top",
        (0, 6, 13) => "Ceiling",
        (0, 6, 14) => "Non-Convective Cloud Cover",
        (0, 6, 15) => "Cloud Work Function",
        (0, 6, 16) => "Convective Cloud Efficiency",
        (0, 6, 17) => "Total Condensate",
        (0, 6, 18) => "Total Column-Integrated Cloud Water",
        (0, 6, 19) => "Total Column-Integrated Cloud Ice",
        (0, 6, 20) => "Total Column-Integrated Condensate",
        (0, 6, 21) => "Ice Fraction of Total Condensate",
        (0, 6, 22) => "Cloud Cover",
        (0, 6, 23) => "Cloud Ice Mixing Ratio",
        (0, 6, 24) => "Sunshine",
        (0, 6, 25) => "Horizontal Extent of Cumulonimbus (CB)",
        (0, 6, 26) => "Height of Convective Cloud Base",
        (0, 6, 27) => "Height of Convective Cloud Top",
        (0, 6, 28) => "Number Concentration of Cloud Droplets",
        (0, 6, 29) => "Number Concentration of Cloud Ice",
        (0, 6, 30) => "Number Density of Cloud Droplets",
        (0, 6, 31) => "Number Density of Cloud Ice",
        (0, 6, 32) => "Fraction of Cloud Cover",
        (0, 6, 33) => "Sunshine Duration",
        (0, 6, 34) => "Surface Downwelling Clear-Sky Shortwave Radiation",
        (0, 6, 35) => "Surface Downwelling Shortwave Radiation",
        (0, 6, 36) => "Sunshine Duration Fraction",
        // NCEP Local Use
        (0, 6, 192) => "Non-Convective Cloud Cover",
        (0, 6, 193) => "Cloud Work Function",
        (0, 6, 194) => "Convective Cloud Efficiency",
        (0, 6, 195) => "Total Condensate",
        (0, 6, 196) => "Total Column-Integrated Cloud Water",
        (0, 6, 197) => "Total Column-Integrated Cloud Ice",
        (0, 6, 198) => "Total Column-Integrated Condensate",
        (0, 6, 199) => "Ice Fraction of Total Condensate",
        (0, 6, 200) => "Convective Cloud Mass Flux",
        (0, 6, 201) => "Sunshine Duration",

        // Category 7: Thermodynamic Stability
        (0, 7, 0) => "K Index",
        (0, 7, 1) => "Total Totals Index",
        (0, 7, 2) => "Sweat Index",
        (0, 7, 3) => "Montgomery Stream Function",
        (0, 7, 4) => "Sigma Coordinate Vertical Velocity",
        (0, 7, 5) => "Planetary Boundary Layer Regime",
        (0, 7, 6) => "Convective Available Potential Energy",
        (0, 7, 7) => "Convective Inhibition",
        (0, 7, 8) => "Storm Relative Helicity",
        (0, 7, 9) => "Energy Helicity Index",
        (0, 7, 10) => "Surface Lifted Index",
        (0, 7, 11) => "Best (4-layer) Lifted Index",
        (0, 7, 12) => "Richardson Number",
        (0, 7, 13) => "Showalter Index",
        (0, 7, 14) => "Severe Weather Threat Index",
        (0, 7, 15) => "Updraft Helicity",
        (0, 7, 16) => "Bulk Richardson Number",
        (0, 7, 17) => "Gradient Richardson Number",
        (0, 7, 18) => "Flux Richardson Number",
        (0, 7, 19) => "Convective Available Potential Energy Shear",
        // NCEP Local Use
        (0, 7, 192) => "Surface Lifted Index",
        (0, 7, 193) => "Best (4-layer) Lifted Index",
        (0, 7, 194) => "Richardson Number",
        (0, 7, 195) => "Convective Weather Detection Index",
        (0, 7, 196) => "Ultra Violet Index",
        (0, 7, 197) => "Updraft Helicity",
        (0, 7, 198) => "Leaf Area Index",
        (0, 7, 199) => "Hourly Maximum of Updraft Helicity over Layer 2km to 5km AGL",

        // Category 13: Aerosols
        (0, 13, 0) => "Aerosol Type",
        (0, 13, 192) => "Particulate Matter (coarse)",
        (0, 13, 193) => "Percent Frozen Precipitation",
        (0, 13, 194) => "Particulate Matter (fine)",
        (0, 13, 195) => "Particulate Matter (fine)",

        // Category 14: Trace Gases
        (0, 14, 0) => "Total Ozone",
        (0, 14, 1) => "Ozone Mixing Ratio",
        (0, 14, 2) => "Total Column Integrated Ozone",
        // NCEP Local Use
        (0, 14, 192) => "Ozone Mixing Ratio",
        (0, 14, 193) => "Ozone Concentration",
        (0, 14, 194) => "Categorical Ozone Concentration",
        (0, 14, 195) => "Vorticity Advection",

        // Category 15: Radar
        (0, 15, 0) => "Base Spectrum Width",
        (0, 15, 1) => "Base Reflectivity",
        (0, 15, 2) => "Base Radial Velocity",
        (0, 15, 3) => "Vertically-Integrated Liquid",
        (0, 15, 4) => "Layer-Maximum Base Reflectivity",
        (0, 15, 5) => "Precipitation",
        (0, 15, 6) => "Radar Spectra (1)",
        (0, 15, 7) => "Radar Spectra (2)",
        (0, 15, 8) => "Radar Spectra (3)",
        (0, 15, 9) => "Reflectivity of Cloud Droplets",
        (0, 15, 10) => "Reflectivity of Cloud Ice",
        (0, 15, 11) => "Reflectivity of Snow",
        (0, 15, 12) => "Reflectivity of Rain",
        (0, 15, 13) => "Reflectivity of Graupel",
        (0, 15, 14) => "Reflectivity of Hail",

        // Category 16: Forecast Radar Imagery
        (0, 16, 0) => "Equivalent Radar Reflectivity Factor for Rain",
        (0, 16, 1) => "Equivalent Radar Reflectivity Factor for Snow",
        (0, 16, 2) => "Equivalent Radar Reflectivity Factor for Parameterized Convection",
        (0, 16, 3) => "Echo Top",
        (0, 16, 4) => "Reflectivity",
        (0, 16, 5) => "Composite Reflectivity",
        // NCEP Local Use
        (0, 16, 192) => "Equivalent Radar Reflectivity Factor for Rain",
        (0, 16, 193) => "Equivalent Radar Reflectivity Factor for Snow",
        (0, 16, 194) => "Equivalent Radar Reflectivity Factor for Parameterized Convection",
        (0, 16, 195) => "Reflectivity",
        (0, 16, 196) => "Composite Reflectivity",
        (0, 16, 197) => "Echo Top",
        (0, 16, 198) => "Hourly Maximum of Simulated Reflectivity at 1 km AGL",

        // Category 17: Electrodynamics
        (0, 17, 0) => "Lightning Strike Density",
        (0, 17, 1) => "Lightning Potential Index (LPI)",
        (0, 17, 192) => "Lightning",

        // Category 18: Nuclear/Radiology
        (0, 18, 0) => "Air Concentration of Caesium-137",
        (0, 18, 1) => "Air Concentration of Iodine-131",
        (0, 18, 2) => "Air Concentration of Radioactive Pollutant",
        (0, 18, 3) => "Ground Deposition of Caesium-137",
        (0, 18, 4) => "Ground Deposition of Iodine-131",
        (0, 18, 5) => "Ground Deposition of Radioactive Pollutant",
        (0, 18, 6) => "Time-Integrated Air Concentration of Cs Pollutant",
        (0, 18, 7) => "Time-Integrated Air Concentration of Iodine Pollutant",
        (0, 18, 8) => "Time-Integrated Air Concentration of Radioactive Pollutant",
        (0, 18, 10) => "Air Dose Rate",
        (0, 18, 11) => "Ground Dose Rate",
        (0, 18, 12) => "Thyroid Dose Rate",

        // Category 19: Physical Atmospheric Properties
        (0, 19, 0) => "Visibility",
        (0, 19, 1) => "Albedo",
        (0, 19, 2) => "Thunderstorm Probability",
        (0, 19, 3) => "Mixed Layer Depth",
        (0, 19, 4) => "Volcanic Ash",
        (0, 19, 5) => "Icing Top",
        (0, 19, 6) => "Icing Base",
        (0, 19, 7) => "Icing",
        (0, 19, 8) => "Turbulence Top",
        (0, 19, 9) => "Turbulence Base",
        (0, 19, 10) => "Turbulence",
        (0, 19, 11) => "Turbulent Kinetic Energy",
        (0, 19, 12) => "Planetary Boundary Layer Regime",
        (0, 19, 13) => "Contrail Intensity",
        (0, 19, 14) => "Contrail Engine Type",
        (0, 19, 15) => "Contrail Top",
        (0, 19, 16) => "Contrail Base",
        (0, 19, 17) => "Maximum Snow Albedo",
        (0, 19, 18) => "Snow-Free Albedo",
        (0, 19, 19) => "Snow Albedo",
        (0, 19, 20) => "Icing",
        (0, 19, 21) => "In-Cloud Turbulence",
        (0, 19, 22) => "Clear Air Turbulence (CAT)",
        (0, 19, 23) => "Supercooled Large Droplet Probability",
        (0, 19, 24) => "Convective Turbulent Kinetic Energy",
        (0, 19, 25) => "Weather Interpretation ww (WMO)",
        (0, 19, 26) => "Convective Precipitation Potential (CPC)",
        (0, 19, 27) => "Icing Scenario",
        // NCEP Local Use
        (0, 19, 192) => "Maximum Snow Albedo",
        (0, 19, 193) => "Snow-Free Albedo",
        (0, 19, 194) => "Slight Risk Convective Outlook",
        (0, 19, 195) => "Moderate Risk Convective Outlook",
        (0, 19, 196) => "High Risk Convective Outlook",
        (0, 19, 197) => "Tornado Probability",
        (0, 19, 198) => "Hail Probability",
        (0, 19, 199) => "Wind Probability",
        (0, 19, 200) => "Significant Tornado Probability",
        (0, 19, 201) => "Significant Hail Probability",
        (0, 19, 202) => "Significant Wind Probability",
        (0, 19, 203) => "Categorical Thunderstorm",
        (0, 19, 204) => "Number of Mixed Layers Next to Surface",
        (0, 19, 205) => "Flight Category",
        (0, 19, 206) => "Confidence - Ceiling",
        (0, 19, 207) => "Confidence - Visibility",
        (0, 19, 208) => "Confidence - Flight Category",
        (0, 19, 209) => "Low-Level Wind Shear Area",
        (0, 19, 210) => "Low-Level Wind Shear Height",
        (0, 19, 211) => "Icing Severity",
        (0, 19, 215) => "Total Probability of Severe Thunderstorms (Days 2,3)",
        (0, 19, 216) => "Total Probability of Extreme Severe Thunderstorms (Days 2,3)",
        (0, 19, 217) => "Surface Drag Coefficient",
        (0, 19, 220) => "Maximum of Icing Potential",
        (0, 19, 232) => "Derived Radar Reflectivity Backscatter from Rain",
        (0, 19, 233) => "Derived Radar Reflectivity Backscatter from Ice",
        (0, 19, 234) => "Composite Reflectivity (Maximum Hourly)",
        (0, 19, 235) => "Derived Radar Reflectivity Backscatter from Parameterized Convection",

        // Category 190: CCITT IA5 string (NCEP local)
        (0, 190, 0) => "Arbitrary Text String",

        // Category 191: Miscellaneous (NCEP local)
        (0, 191, 0) => "Seconds Prior to Initial Reference Time",
        (0, 191, 1) => "Geographical Latitude",
        (0, 191, 2) => "Geographical Longitude",
        (0, 191, 192) => "Latitude (-90 to 90)",
        (0, 191, 193) => "East Longitude (0 to 360)",
        (0, 191, 194) => "Seconds Prior to Initial Reference Time",

        // Category 192: Covariance (NCEP local)
        (0, 192, 1) => "Covariance between zonal and meridional components of the wind",
        (0, 192, 2) => "Covariance between izonal component of the wind and temperature",
        (0, 192, 3) => "Covariance between meridional component of the wind and temperature",
        (0, 192, 4) => "Covariance between temperature and vertical component of the wind",
        (0, 192, 5) => "Covariance between zonal and zonal components of the wind",
        (0, 192, 6) => "Covariance between meridional and meridional components of the wind",
        (0, 192, 7) => "Covariance between specific humidity and zonal component of the wind",
        (0, 192, 8) => "Covariance between specific humidity and meridional component of the wind",
        (0, 192, 9) => "Covariance between temperature and temperature",

        // =====================================================================
        // Discipline 1: Hydrological Products
        // =====================================================================

        // Category 0: Hydrology Basic
        (1, 0, 0) => "Flash Flood Guidance (Instantaneous)",
        (1, 0, 1) => "Flash Flood Runoff (Instantaneous)",
        (1, 0, 2) => "Remotely Sensed Snow Cover",
        (1, 0, 3) => "Elevation of Snow Covered Terrain",
        (1, 0, 4) => "Snow Water Equivalent Percent of Normal",
        (1, 0, 5) => "Baseflow-Groundwater Runoff",
        (1, 0, 6) => "Storm Surface Runoff",
        (1, 0, 7) => "Discharge from Rivers/Streams",
        // NCEP Local Use
        (1, 0, 192) => "Baseflow-Groundwater Runoff",
        (1, 0, 193) => "Storm Surface Runoff",

        // Category 1: Hydrology Probabilities
        (1, 1, 0) => "Conditional Percent Precipitation Amount Fractile for an Overall Period",
        (1, 1, 1) => "Percent Precipitation in a Sub-Period of an Overall Period",
        (1, 1, 2) => "Probability of 0.01 inch of Precipitation (POP)",
        // NCEP Local Use
        (1, 1, 192) => "Probability of Freezing Precipitation",
        (1, 1, 193) => "Probability of Frozen Precipitation",
        (1, 1, 194) => "Probability of Precipitation Exceeding Flash Flood Guidance Values",
        (1, 1, 195) => "Probability of Wetting Rain, exceeding in 0.10 in a given time period",

        // Category 2: Inland Water and Sediment Properties
        (1, 2, 0) => "Water Depth",
        (1, 2, 1) => "Water Temperature",
        (1, 2, 2) => "Water Fraction",
        (1, 2, 3) => "Sediment Thickness",
        (1, 2, 4) => "Sediment Temperature",
        (1, 2, 5) => "Ice Thickness",
        (1, 2, 6) => "Ice Cover",
        (1, 2, 7) => "Ice Temperature",

        // =====================================================================
        // Discipline 2: Land Surface Products
        // =====================================================================

        // Category 0: Vegetation/Biomass
        (2, 0, 0) => "Land Cover (0=sea, 1=land)",
        (2, 0, 1) => "Surface Roughness",
        (2, 0, 2) => "Soil Temperature",
        (2, 0, 3) => "Soil Moisture Content",
        (2, 0, 4) => "Vegetation",
        (2, 0, 5) => "Water Runoff",
        (2, 0, 6) => "Evapotranspiration",
        (2, 0, 7) => "Model Terrain Height",
        (2, 0, 8) => "Land Use",
        (2, 0, 9) => "Volumetric Soil Moisture Content",
        (2, 0, 10) => "Ground Heat Flux",
        (2, 0, 11) => "Moisture Availability",
        (2, 0, 12) => "Exchange Coefficient",
        (2, 0, 13) => "Plant Canopy Surface Water",
        (2, 0, 14) => "Blackadar's Mixing Length Scale",
        (2, 0, 15) => "Canopy Conductance",
        (2, 0, 16) => "Minimal Stomatal Resistance",
        (2, 0, 17) => "Wilting Point",
        (2, 0, 18) => "Solar parameter in canopy conductance",
        (2, 0, 19) => "Temperature parameter in canopy conductance",
        (2, 0, 20) => "Humidity parameter in canopy conductance",
        (2, 0, 21) => "Soil moisture parameter in canopy conductance",
        (2, 0, 22) => "Soil Moisture",
        (2, 0, 23) => "Column-Integrated Soil Water",
        (2, 0, 24) => "Heat Flux",
        (2, 0, 25) => "Volumetric Soil Moisture",
        (2, 0, 26) => "Wilting Point",
        (2, 0, 27) => "Volumetric Wilting Point",
        (2, 0, 28) => "Leaf Area Index",
        (2, 0, 29) => "Evergreen Forest Cover",
        (2, 0, 30) => "Deciduous Forest Cover",
        (2, 0, 31) => "Normalized Differential Vegetation Index (NDVI)",
        (2, 0, 32) => "Root Depth of Vegetation",
        (2, 0, 33) => "Water Runoff and Drainage",
        (2, 0, 34) => "Surface Water Runoff",
        (2, 0, 35) => "Tile Fraction",
        (2, 0, 36) => "Tile Class",
        // NCEP Local Use
        (2, 0, 192) => "Volumetric Soil Moisture Content",
        (2, 0, 193) => "Ground Heat Flux",
        (2, 0, 194) => "Moisture Availability",
        (2, 0, 195) => "Exchange Coefficient",
        (2, 0, 196) => "Plant Canopy Surface Water",
        (2, 0, 197) => "Blackadar's Mixing Length Scale",
        (2, 0, 198) => "Vegetation Type",
        (2, 0, 199) => "Canopy Conductance",
        (2, 0, 200) => "Minimal Stomatal Resistance",
        (2, 0, 201) => "Wilting Point",
        (2, 0, 202) => "Solar Parameter in Canopy Conductance",
        (2, 0, 203) => "Temperature Parameter in Canopy Conductance",
        (2, 0, 204) => "Humidity Parameter in Canopy Conductance",
        (2, 0, 205) => "Soil Moisture Parameter in Canopy Conductance",
        (2, 0, 206) => "Rate of Water Dropping from Canopy to Ground",
        (2, 0, 207) => "Ice-Free Water Surface",
        (2, 0, 208) => "Surface Exchange Coefficients for T and Q Divided by Delta z",
        (2, 0, 209) => "Surface Exchange Coefficients for Wind Divided by Delta z",
        (2, 0, 210) => "Vegetation Canopy Temperature",
        (2, 0, 211) => "Surface Water Storage",
        (2, 0, 212) => "Liquid Soil Moisture Content (non-frozen)",
        (2, 0, 213) => "Open Water Evaporation (standing water)",
        (2, 0, 214) => "Groundwater Recharge",
        (2, 0, 215) => "Flood Plain Recharge",
        (2, 0, 216) => "Roughness Length for Heat",
        (2, 0, 217) => "Normalized Difference Vegetation Index",
        (2, 0, 218) => "Land-Sea Coverage (nearest neighbor)",
        (2, 0, 219) => "Asymptotic Mixing Length Scale",
        (2, 0, 220) => "Water Vapor Added by Precip Assimilation",
        (2, 0, 221) => "Water Condensate Added by Precip Assimilation",
        (2, 0, 222) => "Water Vapor Flux Convergence (Vertical Integral)",
        (2, 0, 223) => "Water Condensate Flux Convergence (Vertical Integral)",
        (2, 0, 224) => "Water Vapor Zonal Flux (Vertical Integral)",
        (2, 0, 225) => "Water Vapor Meridional Flux (Vertical Integral)",
        (2, 0, 226) => "Water Condensate Zonal Flux (Vertical Integral)",
        (2, 0, 227) => "Water Condensate Meridional Flux (Vertical Integral)",
        (2, 0, 228) => "Aerodynamic Conductance",
        (2, 0, 229) => "Canopy Water Evaporation",
        (2, 0, 230) => "Transpiration",

        // Category 1: Agricultural/Aquacultural Special Products
        (2, 1, 192) => "Cold Advisory for Newborn Livestock",

        // Category 3: Soil Products
        (2, 3, 0) => "Soil Type",
        (2, 3, 1) => "Upper Layer Soil Temperature",
        (2, 3, 2) => "Upper Layer Soil Moisture",
        (2, 3, 3) => "Lower Layer Soil Moisture",
        (2, 3, 4) => "Bottom Layer Soil Temperature",
        (2, 3, 5) => "Soil Porosity",
        (2, 3, 6) => "Soil Liquid Volumetric Content (Field Capacity)",
        (2, 3, 7) => "Number of Soil Layers in Root Zone",
        (2, 3, 8) => "Transpiration Stress-onset (soil moisture)",
        (2, 3, 9) => "Direct Evaporation Cease (soil moisture)",
        (2, 3, 10) => "Soil Porosity",
        (2, 3, 11) => "Volumetric Saturation Of Soil Moisture",
        (2, 3, 12) => "Saturation Of Soil Moisture",
        (2, 3, 13) => "Soil Temperature",
        (2, 3, 14) => "Soil Moisture",
        (2, 3, 15) => "Column-Integrated Soil Moisture",
        (2, 3, 16) => "Soil Heat Flux",
        (2, 3, 17) => "Soil Depth",
        // NCEP Local Use
        (2, 3, 192) => "Liquid Volumetric Soil Moisture (non-frozen)",
        (2, 3, 193) => "Number of Soil Layers in Root Zone",
        (2, 3, 194) => "Surface Slope Type",
        (2, 3, 195) => "Transpiration Stress-onset (soil moisture)",
        (2, 3, 196) => "Direct Evaporation Cease (soil moisture)",
        (2, 3, 197) => "Soil Porosity",
        (2, 3, 198) => "Direct Evaporation from Bare Soil",
        (2, 3, 199) => "Land Surface Precipitation Accumulation",
        (2, 3, 200) => "Bare Soil Surface Skin Temperature",
        (2, 3, 201) => "Average Surface Skin Temperature",
        (2, 3, 202) => "Effective Radiative Skin Temperature",
        (2, 3, 203) => "Field Capacity",

        // Category 4: Fire Weather Products
        (2, 4, 0) => "Fire Outlook",
        (2, 4, 1) => "Fire Outlook Due to Dry Thunderstorm",
        (2, 4, 2) => "Haines Index",
        (2, 4, 3) => "Fire Burned Area",
        (2, 4, 4) => "Fosberg Index",
        (2, 4, 5) => "Fire Weather Index (FWI)",
        (2, 4, 6) => "Fine Fuel Moisture Code (FFMC)",
        (2, 4, 7) => "Duff Moisture Code (DMC)",
        (2, 4, 8) => "Drought Code (DC)",
        (2, 4, 9) => "Initial Spread Index (ISI)",
        (2, 4, 10) => "Fire Danger Index (Buildup)",
        (2, 4, 11) => "Evergreen Forest Fire Weather Index",
        (2, 4, 12) => "Deciduous Forest Fire Weather Index",
        // NCEP Local Use
        (2, 4, 192) => "Fire Burned Area",

        // Category 5: Glaciers and Inland Ice
        (2, 5, 0) => "Glacier Cover",
        (2, 5, 1) => "Glacier Temperature",
        (2, 5, 2) => "Ice Velocity in X Direction",
        (2, 5, 3) => "Ice Velocity in Y Direction",

        // =====================================================================
        // Discipline 3: Space Products
        // =====================================================================

        // Category 0: Image Format Products
        (3, 0, 0) => "Scaled Radiance",
        (3, 0, 1) => "Scaled Albedo",
        (3, 0, 2) => "Scaled Brightness Temperature",
        (3, 0, 3) => "Scaled Precipitable Water",
        (3, 0, 4) => "Scaled Lifted Index",
        (3, 0, 5) => "Scaled Cloud Top Pressure",
        (3, 0, 6) => "Scaled Skin Temperature",
        (3, 0, 7) => "Cloud Mask",
        (3, 0, 8) => "Pixel Scene Type",
        (3, 0, 9) => "Fire Detection Indicator",

        // Category 1: Quantitative Products
        (3, 1, 0) => "Estimated Precipitation",
        (3, 1, 1) => "Instantaneous Rain Rate",
        (3, 1, 2) => "Cloud Top Height",
        (3, 1, 3) => "Cloud Top Height Quality Indicator",
        (3, 1, 4) => "Estimated u-Component of Wind",
        (3, 1, 5) => "Estimated v-Component of Wind",
        (3, 1, 6) => "Number of Pixels Used",
        (3, 1, 7) => "Solar Zenith Angle",
        (3, 1, 8) => "Relative Azimuth Angle",
        (3, 1, 9) => "Reflectance in 0.6 Micron Channel",
        (3, 1, 10) => "Reflectance in 0.8 Micron Channel",
        (3, 1, 11) => "Reflectance in 1.6 Micron Channel",
        (3, 1, 12) => "Reflectance in 3.9 Micron Channel",
        (3, 1, 13) => "Atmospheric Divergence",
        // NCEP Local Use
        (3, 1, 192) => "Scatterometer Estimated U Wind Component",
        (3, 1, 193) => "Scatterometer Estimated V Wind Component",

        // Category 192: Forecast Satellite Imagery (NCEP local)
        (3, 192, 0) => "Simulated Brightness Temperature for GOES 12, Channel 2",
        (3, 192, 1) => "Simulated Brightness Temperature for GOES 12, Channel 3",
        (3, 192, 2) => "Simulated Brightness Temperature for GOES 12, Channel 4",
        (3, 192, 3) => "Simulated Brightness Temperature for GOES 12, Channel 6",
        (3, 192, 4) => "Simulated Brightness Counts for GOES 12, Channel 3",
        (3, 192, 5) => "Simulated Brightness Counts for GOES 12, Channel 4",
        (3, 192, 9) => "Simulated Brightness Temperature for GOES 11, Channel 2",
        (3, 192, 10) => "Simulated Brightness Temperature for GOES 11, Channel 3",
        (3, 192, 11) => "Simulated Brightness Temperature for GOES 11, Channel 4",
        (3, 192, 12) => "Simulated Brightness Temperature for GOES 11, Channel 5",

        // =====================================================================
        // Discipline 10: Oceanographic Products
        // =====================================================================

        // Category 0: Waves
        (10, 0, 0) => "Wave Spectra (1)",
        (10, 0, 1) => "Wave Spectra (2)",
        (10, 0, 2) => "Wave Spectra (3)",
        (10, 0, 3) => "Significant Height of Combined Wind Waves and Swell",
        (10, 0, 4) => "Direction of Wind Waves",
        (10, 0, 5) => "Significant Height of Wind Waves",
        (10, 0, 6) => "Mean Period of Wind Waves",
        (10, 0, 7) => "Direction of Swell Waves",
        (10, 0, 8) => "Significant Height of Swell Waves",
        (10, 0, 9) => "Mean Period of Swell Waves",
        (10, 0, 10) => "Primary Wave Direction",
        (10, 0, 11) => "Primary Wave Mean Period",
        (10, 0, 12) => "Secondary Wave Direction",
        (10, 0, 13) => "Secondary Wave Mean Period",
        (10, 0, 14) => "Direction of Combined Wind Waves and Swell",
        (10, 0, 15) => "Mean Period of Combined Wind Waves and Swell",
        (10, 0, 16) => "Coefficient of Drag with Waves",
        (10, 0, 17) => "Friction Velocity",
        (10, 0, 18) => "Wave Stress",
        (10, 0, 19) => "Normalised Waves Stress",
        (10, 0, 20) => "Mean Square Slope of Waves",
        (10, 0, 21) => "U-component Surface Stokes Drift",
        (10, 0, 22) => "V-component Surface Stokes Drift",
        (10, 0, 23) => "Period of Maximum Individual Wave Height",
        (10, 0, 24) => "Maximum Individual Wave Height",
        (10, 0, 25) => "Inverse Mean Wave Frequency",
        (10, 0, 26) => "Inverse Mean Frequency of Wind Waves",
        (10, 0, 27) => "Inverse Mean Frequency of Total Swell",
        (10, 0, 28) => "Mean Zero-Crossing Wave Period",
        (10, 0, 29) => "Mean Zero-Crossing Period of Wind Waves",
        (10, 0, 30) => "Mean Zero-Crossing Period of Total Swell",
        (10, 0, 31) => "Wave Directional Width",
        (10, 0, 32) => "Directional Width of Wind Waves",
        (10, 0, 33) => "Directional Width of Total Swell",
        (10, 0, 34) => "Peak Wave Period",
        (10, 0, 35) => "Peak Period of Wind Waves",
        (10, 0, 36) => "Peak Period of Total Swell",
        (10, 0, 37) => "Altimeter Wave Height",
        (10, 0, 38) => "Altimeter Corrected Wave Height",
        (10, 0, 39) => "Altimeter Range Relative Correction",
        (10, 0, 40) => "10 Metre Neutral Wind Speed Over Waves",
        (10, 0, 41) => "10 Metre Wind Direction Over Waves",
        (10, 0, 42) => "Wave Energy Spectrum",
        (10, 0, 43) => "Kurtosis of the Sea Surface Elevation Due to Waves",
        (10, 0, 44) => "Benjamin-Feir Index",
        (10, 0, 45) => "Spectral Peakedness Factor",

        // Category 1: Currents
        (10, 1, 0) => "Current Direction",
        (10, 1, 1) => "Current Speed",
        (10, 1, 2) => "U-Component of Current",
        (10, 1, 3) => "V-Component of Current",
        // NCEP Local Use
        (10, 1, 192) => "Ocean Mixed Layer U Velocity",
        (10, 1, 193) => "Ocean Mixed Layer V Velocity",
        (10, 1, 194) => "Barotropic U Velocity",
        (10, 1, 195) => "Barotropic V Velocity",

        // Category 2: Ice
        (10, 2, 0) => "Ice Cover",
        (10, 2, 1) => "Ice Thickness",
        (10, 2, 2) => "Direction of Ice Drift",
        (10, 2, 3) => "Speed of Ice Drift",
        (10, 2, 4) => "U-Component of Ice Drift",
        (10, 2, 5) => "V-Component of Ice Drift",
        (10, 2, 6) => "Ice Growth Rate",
        (10, 2, 7) => "Ice Divergence",
        (10, 2, 8) => "Ice Temperature",
        (10, 2, 9) => "Ice Internal Pressure",

        // Category 3: Surface Properties
        (10, 3, 0) => "Water Temperature",
        (10, 3, 1) => "Deviation of Sea Level from Mean",
        (10, 3, 2) => "Sea Surface Height Relative to Geoid",
        (10, 3, 192) => "Hurricane Storm Surge",
        (10, 3, 193) => "Extra Tropical Storm Surge",
        (10, 3, 194) => "Ocean Surface Elevation Relative to Geoid",
        (10, 3, 195) => "Sea Surface Temperature",
        (10, 3, 196) => "Sea Surface Temperature Anomaly",
        (10, 3, 197) => "Ocean Current Speed at Surface",
        (10, 3, 198) => "Ocean Current Direction at Surface",
        (10, 3, 242) => "20% Tropical Cyclone Storm Surge Exceedance",
        (10, 3, 243) => "30% Tropical Cyclone Storm Surge Exceedance",
        (10, 3, 244) => "40% Tropical Cyclone Storm Surge Exceedance",
        (10, 3, 245) => "50% Tropical Cyclone Storm Surge Exceedance",
        (10, 3, 246) => "Surge Plus Tide of Tropical Cyclone",

        // Category 4: Sub-Surface Properties
        (10, 4, 0) => "Main Thermocline Depth",
        (10, 4, 1) => "Main Thermocline Anomaly",
        (10, 4, 2) => "Transient Thermocline Depth",
        (10, 4, 3) => "Salinity",
        (10, 4, 4) => "Ocean Vertical Heat Diffusivity",
        (10, 4, 5) => "Ocean Vertical Salt Diffusivity",
        (10, 4, 6) => "Ocean Vertical Momentum Diffusivity",
        (10, 4, 7) => "Bathymetry",
        (10, 4, 11) => "Shape Factor With Respect To Salinity Profile",
        (10, 4, 12) => "Shape Factor With Respect To Temperature Profile In Thermocline",
        (10, 4, 13) => "Attenuation Coefficient Of Water With Respect To Solar Radiation",
        (10, 4, 14) => "Water Depth",
        (10, 4, 15) => "Water Temperature",
        // NCEP Local Use
        (10, 4, 192) => "3-D Temperature",
        (10, 4, 193) => "3-D Salinity",
        (10, 4, 194) => "Barotropic Kinetic Energy",
        (10, 4, 195) => "Geometric Depth Below Sea Surface",
        (10, 4, 196) => "Interface Depths",
        (10, 4, 197) => "Ocean Heat Content",

        _ => "Unknown",
    }
}

/// Look up the units of a GRIB2 parameter.
/// Based on WMO GRIB2 Code Table 4.2 and NCEP local-use extensions.
/// Returns "Unknown" for unrecognized combinations.
pub fn parameter_units(discipline: u8, category: u8, number: u8) -> &'static str {
    match (discipline, category, number) {
        // =====================================================================
        // Discipline 0: Meteorological Products
        // =====================================================================

        // Category 0: Temperature
        (0, 0, 0) | (0, 0, 1) | (0, 0, 2) | (0, 0, 3) => "K",
        (0, 0, 4) | (0, 0, 5) | (0, 0, 6) | (0, 0, 7) => "K",
        (0, 0, 8) => "K/m",
        (0, 0, 9) => "K",
        (0, 0, 10) | (0, 0, 11) => "W/m²",
        (0, 0, 12) | (0, 0, 13) => "K",
        (0, 0, 14) => "K",
        (0, 0, 15) => "K",
        (0, 0, 16) => "W/m²",
        (0, 0, 17) | (0, 0, 18) => "K",
        (0, 0, 19) => "Numeric",
        (0, 0, 20) => "m²/s",
        (0, 0, 21) => "K",
        (0, 0, 22) | (0, 0, 23) | (0, 0, 24) | (0, 0, 25) | (0, 0, 26) => "K/s",
        (0, 0, 27) => "K",
        (0, 0, 28) => "K",
        (0, 0, 29) => "K/s",
        (0, 0, 30) | (0, 0, 31) => "W/m²",
        (0, 0, 32) => "K",
        // NCEP Local
        (0, 0, 192) => "W/m²",
        (0, 0, 193) => "K/s",
        (0, 0, 194) => "K",
        (0, 0, 195) | (0, 0, 196) | (0, 0, 197) => "K/s",
        (0, 0, 198) | (0, 0, 199) => "K/s",
        (0, 0, 200) => "K",
        (0, 0, 201) | (0, 0, 202) => "K/s",
        (0, 0, 203) => "K",
        (0, 0, 204) => "J/m²",

        // Category 1: Moisture
        (0, 1, 0) => "kg/kg",
        (0, 1, 1) => "%",
        (0, 1, 2) => "kg/kg",
        (0, 1, 3) => "kg/m²",
        (0, 1, 4) => "Pa",
        (0, 1, 5) => "kg/kg",
        (0, 1, 6) => "kg/m²",
        (0, 1, 7) => "kg/m²/s",
        (0, 1, 8) | (0, 1, 9) | (0, 1, 10) => "kg/m²",
        (0, 1, 11) => "m",
        (0, 1, 12) => "kg/m²/s",
        (0, 1, 13) | (0, 1, 14) | (0, 1, 15) => "kg/m²",
        (0, 1, 16) => "kg/m²",
        (0, 1, 17) => "d",
        (0, 1, 18) => "kg/m³",
        (0, 1, 19) => "See Table 4.201",
        (0, 1, 20) => "kg/m²",
        (0, 1, 21) => "kg/kg",
        (0, 1, 22) | (0, 1, 23) | (0, 1, 24) | (0, 1, 25) => "kg/kg",
        (0, 1, 26) => "kg/kg/s",
        (0, 1, 27) => "%",
        (0, 1, 28) => "kg/m³",
        (0, 1, 29) => "m",
        (0, 1, 30) => "See Table 4.202",
        (0, 1, 31) => "m",
        (0, 1, 32) => "kg/kg",
        (0, 1, 33) | (0, 1, 34) | (0, 1, 35) | (0, 1, 36) => "Code table 4.222",
        (0, 1, 37) => "kg/m²/s",
        (0, 1, 38) => "kg/kg/s",
        (0, 1, 39) => "%",
        (0, 1, 40) => "kg/m²",
        (0, 1, 41) => "W/m²",
        (0, 1, 42) => "%",
        (0, 1, 43) => "Proportion",
        (0, 1, 44) => "Numeric",
        (0, 1, 45) | (0, 1, 46) => "kg/m²",
        (0, 1, 47) | (0, 1, 48) | (0, 1, 49) | (0, 1, 50) => "kg/m²",
        (0, 1, 51) => "kg/m²",
        (0, 1, 52) => "kg/m²/s",
        (0, 1, 53) => "kg/m²/s",
        (0, 1, 54) => "kg/m²/s",
        (0, 1, 55) => "kg/m²/s",
        (0, 1, 56) => "kg/m²/s",
        (0, 1, 57) => "m/s",
        (0, 1, 58) => "m/s",
        (0, 1, 59) => "m/s",
        (0, 1, 60) => "kg/m²",
        (0, 1, 61) => "kg/m³",
        (0, 1, 62) => "kg/m²",
        (0, 1, 64) => "kg/m²",
        (0, 1, 65) | (0, 1, 66) | (0, 1, 67) | (0, 1, 68) => "kg/m²/s",
        (0, 1, 69) | (0, 1, 70) => "kg/m²",
        (0, 1, 71) => "kg/kg",
        (0, 1, 72) => "kg/m²",
        (0, 1, 73) => "kg/m²/s",
        (0, 1, 74) => "kg/m²",
        (0, 1, 75) => "kg/m²/s",
        (0, 1, 76) | (0, 1, 77) => "kg/m²/s",
        (0, 1, 78) => "kg/m²",
        (0, 1, 79) => "kg/m²/s",
        (0, 1, 80) => "kg/kg",
        (0, 1, 81) => "kg/m²",
        (0, 1, 82) => "kg/kg",
        (0, 1, 83) | (0, 1, 84) | (0, 1, 85) | (0, 1, 86) => "kg/kg",
        (0, 1, 90) => "kg/kg m/s",
        (0, 1, 91) | (0, 1, 92) => "kg/kg m/s",
        (0, 1, 99) => "kg/m²",
        (0, 1, 100) => "kg/m²/s",
        // NCEP Local
        (0, 1, 192) | (0, 1, 193) | (0, 1, 194) | (0, 1, 195) => "Code table 4.222",
        (0, 1, 196) => "kg/m²/s",
        (0, 1, 197) => "kg/kg/s",
        (0, 1, 198) => "%",
        (0, 1, 199) => "kg/m²",
        (0, 1, 200) => "W/m²",
        (0, 1, 201) => "%",
        (0, 1, 202) => "Proportion",
        (0, 1, 203) => "Numeric",
        (0, 1, 204) | (0, 1, 205) => "kg/m²",
        (0, 1, 206) => "Numeric",
        (0, 1, 207) => "Numeric",
        (0, 1, 208) | (0, 1, 209) => "dBZ",
        (0, 1, 210) => "m",
        (0, 1, 211) => "See Table 4.207",
        (0, 1, 212) => "kg/m²",
        (0, 1, 213) | (0, 1, 214) | (0, 1, 215) => "Code table 4.222",
        (0, 1, 216) => "Numeric",
        (0, 1, 225) => "kg/m²",
        (0, 1, 227) => "Numeric",
        (0, 1, 241) => "kg/m²",
        (0, 1, 242) => "%",

        // Category 2: Momentum
        (0, 2, 0) => "degrees",
        (0, 2, 1) | (0, 2, 2) | (0, 2, 3) => "m/s",
        (0, 2, 4) | (0, 2, 5) => "m²/s",
        (0, 2, 6) => "m²/s²",
        (0, 2, 7) => "1/s",
        (0, 2, 8) => "Pa/s",
        (0, 2, 9) => "m/s",
        (0, 2, 10) | (0, 2, 11) | (0, 2, 12) | (0, 2, 13) => "1/s",
        (0, 2, 14) => "K m²/kg/s",
        (0, 2, 15) | (0, 2, 16) => "1/s",
        (0, 2, 17) | (0, 2, 18) => "N/m²",
        (0, 2, 19) => "J",
        (0, 2, 20) => "W/m²",
        (0, 2, 21) => "m/s",
        (0, 2, 22) => "m/s",
        (0, 2, 23) | (0, 2, 24) => "m/s",
        (0, 2, 25) => "1/s",
        (0, 2, 26) => "N/m²",
        (0, 2, 27) | (0, 2, 28) => "m/s",
        (0, 2, 29) => "Numeric",
        (0, 2, 30) => "m/s",
        (0, 2, 31) => "m²/s",
        (0, 2, 32) => "1/s",
        (0, 2, 33) => "m",
        (0, 2, 34) | (0, 2, 35) => "m/s",
        (0, 2, 36) => "m/s",
        (0, 2, 37) | (0, 2, 38) => "N/m²",
        (0, 2, 39) => "1/s",
        (0, 2, 40) => "%",
        // NCEP Local
        (0, 2, 192) => "1/s",
        (0, 2, 193) => "N/m²",
        (0, 2, 194) | (0, 2, 195) => "m/s",
        (0, 2, 196) | (0, 2, 197) | (0, 2, 198) => "m/s",
        (0, 2, 199) => "m²/s",
        (0, 2, 200) => "m/s",
        (0, 2, 201) => "degrees",
        (0, 2, 202) | (0, 2, 203) => "m/s",
        (0, 2, 204) => "m²/s",
        (0, 2, 220) | (0, 2, 221) => "m/s",
        (0, 2, 222) | (0, 2, 223) => "m/s",
        (0, 2, 224) => "m²/s",
        (0, 2, 225) => "m/s",
        (0, 2, 226) => "degrees",
        (0, 2, 227) | (0, 2, 228) | (0, 2, 229) | (0, 2, 230) => "s",
        (0, 2, 231) => "degrees",
        (0, 2, 232) => "m/s",

        // Category 3: Mass
        (0, 3, 0) | (0, 3, 1) => "Pa",
        (0, 3, 2) => "Pa/s",
        (0, 3, 3) => "m",
        (0, 3, 4) => "m²/s²",
        (0, 3, 5) => "gpm",
        (0, 3, 6) | (0, 3, 7) => "m",
        (0, 3, 8) => "Pa",
        (0, 3, 9) => "gpm",
        (0, 3, 10) => "kg/m³",
        (0, 3, 11) => "Pa",
        (0, 3, 12) => "m",
        (0, 3, 13) | (0, 3, 14) => "m",
        (0, 3, 15) => "gpm",
        (0, 3, 16) | (0, 3, 17) => "N/m²",
        (0, 3, 18) => "m",
        (0, 3, 19) => "gpm",
        (0, 3, 20) => "m",
        (0, 3, 21) => "rad",
        (0, 3, 22) => "Numeric",
        (0, 3, 23) => "W/m²",
        (0, 3, 24) => "Numeric",
        (0, 3, 25) => "Numeric",
        (0, 3, 26) => "Numeric",
        (0, 3, 27) | (0, 3, 28) => "kg/m²/s",
        // NCEP Local
        (0, 3, 192) | (0, 3, 198) => "Pa",
        (0, 3, 193) => "gpm",
        (0, 3, 194) | (0, 3, 195) => "N/m²",
        (0, 3, 196) => "m",
        (0, 3, 197) => "gpm",
        (0, 3, 199) => "Pa/s",
        (0, 3, 200) => "Pa",
        (0, 3, 201) | (0, 3, 202) => "1/m",
        (0, 3, 203) | (0, 3, 204) => "1/m",
        (0, 3, 205) => "m",
        (0, 3, 206) => "Numeric",
        (0, 3, 207) | (0, 3, 208) | (0, 3, 209) => "kg/m²/s",
        (0, 3, 210) => "Numeric",
        (0, 3, 211) => "gpm",
        (0, 3, 212) => "Pa",

        // Category 4: Short-wave Radiation
        (0, 4, 0) | (0, 4, 1) | (0, 4, 2) | (0, 4, 3) => "W/m²",
        (0, 4, 4) => "K",
        (0, 4, 5) => "W/m³/sr",
        (0, 4, 6) => "W/m²/sr/μm",
        (0, 4, 7) | (0, 4, 8) | (0, 4, 9) => "W/m²",
        (0, 4, 10) => "W/m²",
        (0, 4, 11) => "W/m²",
        (0, 4, 12) => "W/m²",
        (0, 4, 50) | (0, 4, 51) => "Numeric",
        (0, 4, 52) | (0, 4, 53) => "W/m²",
        // NCEP Local
        (0, 4, 192) | (0, 4, 193) => "W/m²",
        (0, 4, 194) | (0, 4, 195) | (0, 4, 196) => "W/m²",
        (0, 4, 197) => "K/s",
        (0, 4, 198) | (0, 4, 199) => "W/m²",
        (0, 4, 200) | (0, 4, 201) | (0, 4, 202) | (0, 4, 203) => "W/m²",
        (0, 4, 204) | (0, 4, 205) => "W/m²",

        // Category 5: Long-wave Radiation
        (0, 5, 0) | (0, 5, 1) | (0, 5, 2) => "W/m²",
        (0, 5, 3) | (0, 5, 4) | (0, 5, 5) | (0, 5, 6) => "W/m²",
        (0, 5, 7) => "K",
        (0, 5, 8) => "W/m²",
        // NCEP Local
        (0, 5, 192) | (0, 5, 193) => "W/m²",
        (0, 5, 194) => "K/s",
        (0, 5, 195) | (0, 5, 196) | (0, 5, 197) => "W/m²",

        // Category 6: Cloud
        (0, 6, 0) => "kg/m²",
        (0, 6, 1) | (0, 6, 2) | (0, 6, 3) | (0, 6, 4) | (0, 6, 5) => "%",
        (0, 6, 6) => "kg/m²",
        (0, 6, 7) => "%",
        (0, 6, 8) => "See Table 4.203",
        (0, 6, 9) => "m",
        (0, 6, 10) => "See Table 4.204",
        (0, 6, 11) | (0, 6, 12) | (0, 6, 13) => "m",
        (0, 6, 14) => "%",
        (0, 6, 15) => "J/kg",
        (0, 6, 16) => "Proportion",
        (0, 6, 17) => "kg/kg",
        (0, 6, 18) | (0, 6, 19) | (0, 6, 20) => "kg/m²",
        (0, 6, 21) => "Proportion",
        (0, 6, 22) => "%",
        (0, 6, 23) => "kg/kg",
        (0, 6, 24) => "Numeric",
        (0, 6, 25) => "%",
        (0, 6, 26) | (0, 6, 27) => "m",
        (0, 6, 28) | (0, 6, 29) => "1/kg",
        (0, 6, 30) | (0, 6, 31) => "1/m³",
        (0, 6, 32) => "%",
        (0, 6, 33) => "s",
        (0, 6, 34) | (0, 6, 35) => "W/m²",
        (0, 6, 36) => "Numeric",
        // NCEP Local
        (0, 6, 192) => "%",
        (0, 6, 193) => "J/kg",
        (0, 6, 194) => "Proportion",
        (0, 6, 195) => "kg/kg",
        (0, 6, 196) | (0, 6, 197) | (0, 6, 198) => "kg/m²",
        (0, 6, 199) => "Proportion",
        (0, 6, 200) => "Pa/s",
        (0, 6, 201) => "s",

        // Category 7: Thermodynamic Stability
        (0, 7, 0) | (0, 7, 1) | (0, 7, 2) => "K",
        (0, 7, 3) => "m²/s²",
        (0, 7, 4) => "1/s",
        (0, 7, 5) => "Numeric",
        (0, 7, 6) | (0, 7, 7) => "J/kg",
        (0, 7, 8) => "m²/s²",
        (0, 7, 9) => "Numeric",
        (0, 7, 10) | (0, 7, 11) | (0, 7, 13) => "K",
        (0, 7, 12) => "Numeric",
        (0, 7, 14) => "Numeric",
        (0, 7, 15) => "m²/s²",
        (0, 7, 16) | (0, 7, 17) | (0, 7, 18) => "Numeric",
        (0, 7, 19) => "m/s²",
        // NCEP Local
        (0, 7, 192) | (0, 7, 193) => "K",
        (0, 7, 194) => "Numeric",
        (0, 7, 195) | (0, 7, 196) => "Numeric",
        (0, 7, 197) => "m²/s²",
        (0, 7, 198) => "Numeric",
        (0, 7, 199) => "m²/s²",

        // Category 13: Aerosols
        (0, 13, 0) => "See Table 4.205",
        (0, 13, 192) => "µg/m³",
        (0, 13, 193) => "%",
        (0, 13, 194) | (0, 13, 195) => "µg/m³",

        // Category 14: Trace Gases
        (0, 14, 0) => "DU",
        (0, 14, 1) => "kg/kg",
        (0, 14, 2) => "DU",
        // NCEP Local
        (0, 14, 192) => "kg/kg",
        (0, 14, 193) => "µg/m³",
        (0, 14, 194) => "Numeric",
        (0, 14, 195) => "1/s²",

        // Category 15: Radar
        (0, 15, 0) => "m/s",
        (0, 15, 1) => "dBZ",
        (0, 15, 2) => "m/s",
        (0, 15, 3) => "kg/m²",
        (0, 15, 4) => "dBZ",
        (0, 15, 5) => "kg/m²",
        (0, 15, 6) | (0, 15, 7) | (0, 15, 8) => "-",
        (0, 15, 9) | (0, 15, 10) | (0, 15, 11) | (0, 15, 12) | (0, 15, 13) | (0, 15, 14) => "dBZ",

        // Category 16: Forecast Radar Imagery
        (0, 16, 0) | (0, 16, 1) | (0, 16, 2) => "mm^6/m^3",
        (0, 16, 3) => "m",
        (0, 16, 4) | (0, 16, 5) => "dBZ",
        // NCEP Local
        (0, 16, 192) | (0, 16, 193) | (0, 16, 194) => "mm^6/m^3",
        (0, 16, 195) | (0, 16, 196) => "dBZ",
        (0, 16, 197) => "m",
        (0, 16, 198) => "dBZ",

        // Category 17: Electrodynamics
        (0, 17, 0) => "m⁻²/s",
        (0, 17, 1) => "J/kg",
        (0, 17, 192) => "non-dim",

        // Category 18: Nuclear/Radiology
        (0, 18, 0) | (0, 18, 1) | (0, 18, 2) => "Bq/m³",
        (0, 18, 3) | (0, 18, 4) | (0, 18, 5) => "Bq/m²",
        (0, 18, 6) | (0, 18, 7) | (0, 18, 8) => "Bq s/m³",
        (0, 18, 10) | (0, 18, 11) => "Sv/s",
        (0, 18, 12) => "Sv/s",

        // Category 19: Physical Atmospheric Properties
        (0, 19, 0) => "m",
        (0, 19, 1) => "%",
        (0, 19, 2) => "%",
        (0, 19, 3) => "m",
        (0, 19, 4) => "See Table 4.206",
        (0, 19, 5) | (0, 19, 6) => "m",
        (0, 19, 7) => "See Table 4.207",
        (0, 19, 8) | (0, 19, 9) => "m",
        (0, 19, 10) => "See Table 4.208",
        (0, 19, 11) => "J/kg",
        (0, 19, 12) => "See Table 4.209",
        (0, 19, 13) | (0, 19, 14) => "See Table 4.210",
        (0, 19, 15) | (0, 19, 16) => "m",
        (0, 19, 17) | (0, 19, 18) | (0, 19, 19) => "%",
        (0, 19, 20) => "See Table 4.207",
        (0, 19, 21) | (0, 19, 22) => "See Table 4.208",
        (0, 19, 23) => "%",
        (0, 19, 24) => "J/kg",
        (0, 19, 25) => "Numeric",
        (0, 19, 26) => "%",
        (0, 19, 27) => "Numeric",
        // NCEP Local
        (0, 19, 192) | (0, 19, 193) => "%",
        (0, 19, 194) | (0, 19, 195) | (0, 19, 196) => "Categorical",
        (0, 19, 197) | (0, 19, 198) | (0, 19, 199) => "%",
        (0, 19, 200) | (0, 19, 201) | (0, 19, 202) => "%",
        (0, 19, 203) => "Categorical",
        (0, 19, 204) => "Numeric",
        (0, 19, 205) => "See Table 4.218",
        (0, 19, 206) | (0, 19, 207) | (0, 19, 208) => "%",
        (0, 19, 209) => "Numeric",
        (0, 19, 210) => "m",
        (0, 19, 211) => "non-dim",
        (0, 19, 215) | (0, 19, 216) => "%",
        (0, 19, 217) => "Numeric",
        (0, 19, 220) => "%",
        (0, 19, 232) | (0, 19, 233) | (0, 19, 234) | (0, 19, 235) => "dBZ",

        // Category 190: CCITT IA5 String
        (0, 190, 0) => "CCITT IA5",

        // Category 191: Miscellaneous
        (0, 191, 0) => "s",
        (0, 191, 1) => "°",
        (0, 191, 2) => "°",
        (0, 191, 192) => "°",
        (0, 191, 193) => "°",
        (0, 191, 194) => "s",

        // Category 192: Covariance
        (0, 192, 1) => "m²/s²",
        (0, 192, 2) | (0, 192, 3) => "K m/s",
        (0, 192, 4) => "K m/s",
        (0, 192, 5) | (0, 192, 6) => "m²/s²",
        (0, 192, 7) | (0, 192, 8) => "kg/kg m/s",
        (0, 192, 9) => "K²",

        // =====================================================================
        // Discipline 1: Hydrological Products
        // =====================================================================

        // Category 0: Hydrology Basic
        (1, 0, 0) | (1, 0, 1) => "kg/m²",
        (1, 0, 2) => "See Table 4.215",
        (1, 0, 3) => "m",
        (1, 0, 4) => "%",
        (1, 0, 5) | (1, 0, 6) => "kg/m²",
        (1, 0, 7) => "m³/s",
        // NCEP Local
        (1, 0, 192) | (1, 0, 193) => "kg/m²",

        // Category 1: Hydrology Probabilities
        (1, 1, 0) | (1, 1, 1) | (1, 1, 2) => "%",
        // NCEP Local
        (1, 1, 192) | (1, 1, 193) | (1, 1, 194) | (1, 1, 195) => "%",

        // Category 2: Inland Water and Sediment
        (1, 2, 0) => "m",
        (1, 2, 1) => "K",
        (1, 2, 2) => "Proportion",
        (1, 2, 3) => "m",
        (1, 2, 4) => "K",
        (1, 2, 5) => "m",
        (1, 2, 6) => "Proportion",
        (1, 2, 7) => "K",

        // =====================================================================
        // Discipline 2: Land Surface Products
        // =====================================================================

        // Category 0: Vegetation/Biomass
        (2, 0, 0) => "Proportion",
        (2, 0, 1) => "m",
        (2, 0, 2) => "K",
        (2, 0, 3) => "kg/m³",
        (2, 0, 4) => "%",
        (2, 0, 5) => "kg/m²",
        (2, 0, 6) | (2, 0, 7) => "kg/m²/s",
        (2, 0, 8) => "See Table 4.212",
        (2, 0, 9) => "Proportion",
        (2, 0, 10) => "W/m²",
        (2, 0, 11) => "%",
        (2, 0, 12) => "kg/m²/s",
        (2, 0, 13) => "kg/m²",
        (2, 0, 14) => "m",
        (2, 0, 15) => "m/s",
        (2, 0, 16) => "s/m",
        (2, 0, 17) => "kg/m³",
        (2, 0, 18) | (2, 0, 19) | (2, 0, 20) | (2, 0, 21) => "Proportion",
        (2, 0, 22) => "kg/m³",
        (2, 0, 23) => "kg/m²",
        (2, 0, 24) => "W/m²",
        (2, 0, 25) | (2, 0, 26) | (2, 0, 27) => "Proportion",
        (2, 0, 28) => "Numeric",
        (2, 0, 29) | (2, 0, 30) => "%",
        (2, 0, 31) => "Numeric",
        (2, 0, 32) => "m",
        (2, 0, 33) | (2, 0, 34) => "kg/m²",
        (2, 0, 35) => "Proportion",
        (2, 0, 36) => "Numeric",
        // NCEP Local
        (2, 0, 192) => "Proportion",
        (2, 0, 193) => "W/m²",
        (2, 0, 194) => "%",
        (2, 0, 195) => "kg/m²/s",
        (2, 0, 196) => "kg/m²",
        (2, 0, 197) => "m",
        (2, 0, 198) => "See Table 4.213",
        (2, 0, 199) => "m/s",
        (2, 0, 200) => "s/m",
        (2, 0, 201) => "kg/m³",
        (2, 0, 202) | (2, 0, 203) | (2, 0, 204) | (2, 0, 205) => "Proportion",
        (2, 0, 206) => "kg/m²/s",
        (2, 0, 207) => "Proportion",
        (2, 0, 208) | (2, 0, 209) => "1/s",
        (2, 0, 210) => "K",
        (2, 0, 211) => "kg/m²",
        (2, 0, 212) => "Proportion",
        (2, 0, 213) => "kg/m²/s",
        (2, 0, 214) | (2, 0, 215) => "kg/m²/s",
        (2, 0, 216) => "m",
        (2, 0, 217) => "Numeric",
        (2, 0, 218) => "Proportion",
        (2, 0, 219) => "m",
        (2, 0, 220) | (2, 0, 221) => "kg/m²",
        (2, 0, 222) | (2, 0, 223) => "kg/m²/s",
        (2, 0, 224) | (2, 0, 225) | (2, 0, 226) | (2, 0, 227) => "kg/m²/s",
        (2, 0, 228) => "m/s",
        (2, 0, 229) | (2, 0, 230) => "kg/m²/s",

        // Category 1: Agricultural
        (2, 1, 192) => "Numeric",

        // Category 3: Soil
        (2, 3, 0) => "See Table 4.213",
        (2, 3, 1) | (2, 3, 4) => "K",
        (2, 3, 2) | (2, 3, 3) => "kg/m³",
        (2, 3, 5) => "Proportion",
        (2, 3, 6) => "Proportion",
        (2, 3, 7) => "Numeric",
        (2, 3, 8) | (2, 3, 9) => "kg/m³",
        (2, 3, 10) => "Proportion",
        (2, 3, 11) | (2, 3, 12) => "kg/m³",
        (2, 3, 13) => "K",
        (2, 3, 14) => "kg/m³",
        (2, 3, 15) => "kg/m²",
        (2, 3, 16) => "W/m²",
        (2, 3, 17) => "m",
        // NCEP Local
        (2, 3, 192) => "Proportion",
        (2, 3, 193) => "Numeric",
        (2, 3, 194) => "Index",
        (2, 3, 195) | (2, 3, 196) => "kg/m³",
        (2, 3, 197) => "Proportion",
        (2, 3, 198) => "W/m²",
        (2, 3, 199) => "kg/m²",
        (2, 3, 200) | (2, 3, 201) | (2, 3, 202) => "K",
        (2, 3, 203) => "Proportion",

        // Category 4: Fire Weather
        (2, 4, 0) | (2, 4, 1) => "See Table 4.224",
        (2, 4, 2) => "Numeric",
        (2, 4, 3) => "m²",
        (2, 4, 4) => "Numeric",
        (2, 4, 5) | (2, 4, 6) | (2, 4, 7) | (2, 4, 8) | (2, 4, 9) | (2, 4, 10) => "Numeric",
        (2, 4, 11) | (2, 4, 12) => "Numeric",
        (2, 4, 192) => "m²",

        // Category 5: Glaciers and Inland Ice
        (2, 5, 0) => "Proportion",
        (2, 5, 1) => "K",
        (2, 5, 2) | (2, 5, 3) => "m/s",

        // =====================================================================
        // Discipline 3: Space Products
        // =====================================================================

        // Category 0: Image Format
        (3, 0, 0) => "Numeric",
        (3, 0, 1) => "%",
        (3, 0, 2) => "K",
        (3, 0, 3) | (3, 0, 4) | (3, 0, 5) => "Numeric",
        (3, 0, 6) => "K",
        (3, 0, 7) | (3, 0, 8) | (3, 0, 9) => "Numeric",

        // Category 1: Quantitative
        (3, 1, 0) => "kg/m²",
        (3, 1, 1) => "kg/m²/s",
        (3, 1, 2) => "m",
        (3, 1, 3) => "Code table",
        (3, 1, 4) | (3, 1, 5) => "m/s",
        (3, 1, 6) => "Numeric",
        (3, 1, 7) | (3, 1, 8) => "°",
        (3, 1, 9) | (3, 1, 10) | (3, 1, 11) | (3, 1, 12) => "%",
        (3, 1, 13) => "1/s",
        (3, 1, 192) | (3, 1, 193) => "m/s",

        // Category 192: Forecast Satellite Imagery (NCEP local)
        (3, 192, 0) | (3, 192, 1) | (3, 192, 2) | (3, 192, 3) => "K",
        (3, 192, 4) | (3, 192, 5) => "Numeric",
        (3, 192, 9) | (3, 192, 10) | (3, 192, 11) | (3, 192, 12) => "K",

        // =====================================================================
        // Discipline 10: Oceanographic Products
        // =====================================================================

        // Category 0: Waves
        (10, 0, 0) | (10, 0, 1) | (10, 0, 2) => "-",
        (10, 0, 3) => "m",
        (10, 0, 4) => "degrees",
        (10, 0, 5) => "m",
        (10, 0, 6) => "s",
        (10, 0, 7) => "degrees",
        (10, 0, 8) => "m",
        (10, 0, 9) => "s",
        (10, 0, 10) => "degrees",
        (10, 0, 11) => "s",
        (10, 0, 12) => "degrees",
        (10, 0, 13) => "s",
        (10, 0, 14) => "degrees",
        (10, 0, 15) => "s",
        (10, 0, 16) => "Numeric",
        (10, 0, 17) => "m/s",
        (10, 0, 18) => "N/m²",
        (10, 0, 19) => "Numeric",
        (10, 0, 20) => "Numeric",
        (10, 0, 21) | (10, 0, 22) => "m/s",
        (10, 0, 23) => "s",
        (10, 0, 24) => "m",
        (10, 0, 25) | (10, 0, 26) | (10, 0, 27) => "s",
        (10, 0, 28) | (10, 0, 29) | (10, 0, 30) => "s",
        (10, 0, 31) | (10, 0, 32) | (10, 0, 33) => "Numeric",
        (10, 0, 34) | (10, 0, 35) | (10, 0, 36) => "s",
        (10, 0, 37) | (10, 0, 38) => "m",
        (10, 0, 39) => "m",
        (10, 0, 40) => "m/s",
        (10, 0, 41) => "degrees",
        (10, 0, 42) => "m²/Hz/rad",
        (10, 0, 43) | (10, 0, 44) | (10, 0, 45) => "Numeric",

        // Category 1: Currents
        (10, 1, 0) => "degrees",
        (10, 1, 1) => "m/s",
        (10, 1, 2) | (10, 1, 3) => "m/s",
        // NCEP Local
        (10, 1, 192) | (10, 1, 193) | (10, 1, 194) | (10, 1, 195) => "m/s",

        // Category 2: Ice
        (10, 2, 0) => "Proportion",
        (10, 2, 1) => "m",
        (10, 2, 2) => "degrees",
        (10, 2, 3) => "m/s",
        (10, 2, 4) | (10, 2, 5) => "m/s",
        (10, 2, 6) => "m/s",
        (10, 2, 7) => "1/s",
        (10, 2, 8) => "K",
        (10, 2, 9) => "Pa",

        // Category 3: Surface Properties
        (10, 3, 0) => "K",
        (10, 3, 1) => "m",
        (10, 3, 2) => "m",
        // NCEP Local
        (10, 3, 192) | (10, 3, 193) | (10, 3, 194) => "m",
        (10, 3, 195) => "K",
        (10, 3, 196) => "K",
        (10, 3, 197) => "m/s",
        (10, 3, 198) => "degrees",
        (10, 3, 242) | (10, 3, 243) | (10, 3, 244) | (10, 3, 245) | (10, 3, 246) => "m",

        // Category 4: Sub-Surface Properties
        (10, 4, 0) => "m",
        (10, 4, 1) => "m",
        (10, 4, 2) => "m",
        (10, 4, 3) => "kg/kg",
        (10, 4, 4) | (10, 4, 5) | (10, 4, 6) => "m²/s",
        (10, 4, 7) => "m",
        (10, 4, 11) | (10, 4, 12) => "Numeric",
        (10, 4, 13) => "1/m",
        (10, 4, 14) => "m",
        (10, 4, 15) => "K",
        // NCEP Local
        (10, 4, 192) => "°C",
        (10, 4, 193) => "psu",
        (10, 4, 194) => "J/m²",
        (10, 4, 195) => "m",
        (10, 4, 196) => "m",
        (10, 4, 197) => "J/m²",

        _ => "?",
    }
}

/// Look up the human-readable name of a GRIB2 level type.
/// Based on WMO GRIB2 Code Table 4.5 and NCEP local-use extensions.
pub fn level_name(level_type: u8) -> &'static str {
    match level_type {
        1 => "Ground or Water Surface",
        2 => "Cloud Base Level",
        3 => "Cloud Top Level",
        4 => "Level of 0°C Isotherm",
        5 => "Level of Adiabatic Condensation Lifted from Surface",
        6 => "Maximum Wind Level",
        7 => "Tropopause",
        8 => "Nominal Top of Atmosphere",
        9 => "Sea Bottom",
        10 => "Entire Atmosphere",
        11 => "Cumulonimbus Base",
        12 => "Cumulonimbus Top",
        13 => "Lowest Level Where Vertically Integrated Cloud Cover Exceeds 71%",
        14 => "Level of Free Convection",
        15 => "Convection Condensation Level",
        16 => "Level of Neutral Buoyancy or Equilibrium",
        20 => "Isothermal Level",
        21 => "Lowest Level Where Mass Density Exceeds Value",
        22 => "Highest Level Where Mass Density Exceeds Value",
        23 => "Lowest Level Where ICAO Standard Atmosphere Pressure Exceeds Value",
        24 => "Highest Level Where ICAO Standard Atmosphere Pressure Exceeds Value",
        100 => "Isobaric Surface",
        101 => "Mean Sea Level",
        102 => "Specific Altitude Above Mean Sea Level",
        103 => "Specified Height Level Above Ground",
        104 => "Sigma Level",
        105 => "Hybrid Level",
        106 => "Depth Below Land Surface",
        107 => "Isentropic (Theta) Level",
        108 => "Level at Specified Pressure Difference from Ground to Level",
        109 => "Potential Vorticity Surface",
        110 => "Reserved",
        111 => "Eta Level",
        112 => "Reserved",
        113 => "Logarithmic Hybrid Level",
        114 => "Snow Level",
        117 => "Mixed Layer Depth (m)",
        118 => "Hybrid Height Level",
        119 => "Hybrid Pressure Level",
        150 => "Generalized Vertical Height Coordinate",
        160 => "Depth Below Sea Level",
        161 => "Depth Below Water Surface",
        162 => "Lake or River Bottom",
        163 => "Bottom of Sediment Layer",
        164 => "Bottom of Thermally Active Sediment Layer",
        165 => "Bottom of Sediment Layer Penetrated by Thermal Wave",
        166 => "Mixing Layer",
        167 => "Bottom of Root Zone",
        174 => "Top Surface of Ice",
        175 => "Top Surface of Ice under Supercooled Water",
        176 => "Bottom Surface of Ice",
        177 => "Deep Soil (of Tile Fraction)",
        200 => "Entire Atmosphere (as single layer)",
        201 => "Entire Ocean (as single layer)",
        204 => "Highest Tropospheric Freezing Level",
        206 => "Grid Scale Cloud Bottom Level",
        207 => "Grid Scale Cloud Top Level",
        209 => "Boundary Layer Cloud Bottom Level",
        210 => "Boundary Layer Cloud Top Level",
        211 => "Boundary Layer Cloud Layer",
        212 => "Low Cloud Bottom Level",
        213 => "Low Cloud Top Level",
        214 => "Low Cloud Layer",
        215 => "Cloud Ceiling",
        220 => "Planetary Boundary Layer",
        221 => "Layer Between Two Hybrid Levels",
        222 => "Middle Cloud Bottom Level",
        223 => "Middle Cloud Top Level",
        224 => "Middle Cloud Layer",
        232 => "High Cloud Bottom Level",
        233 => "High Cloud Top Level",
        234 => "High Cloud Layer",
        235 => "Ocean Isotherm Level (1/10 °C)",
        236 => "Layer Between Two Depths Below Ocean Surface",
        237 => "Bottom of Ocean Mixed Layer",
        238 => "Bottom of Ocean Isothermal Layer",
        239 => "Layer Ocean Surface and 26°C Ocean Isothermal Level",
        240 => "Ocean Mixed Layer",
        241 => "Ordered Sequence of Data",
        242 => "Convective Cloud Bottom Level",
        243 => "Convective Cloud Top Level",
        244 => "Convective Cloud Layer",
        245 => "Lowest Level of the Wet Bulb Zero",
        246 => "Maximum Equivalent Potential Temperature Level",
        247 => "Equilibrium Level",
        248 => "Shallow Convective Cloud Bottom Level",
        249 => "Shallow Convective Cloud Top Level",
        251 => "Deep Convective Cloud Bottom Level",
        252 => "Deep Convective Cloud Top Level",
        253 => "Lowest Bottom Level of Supercooled Liquid Water Layer",
        254 => "Highest Top Level of Supercooled Liquid Water Layer",
        _ => "Unknown Level Type",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parameter_name tests ----

    #[test]
    fn test_parameter_name_temperature() {
        assert_eq!(parameter_name(0, 0, 0), "Temperature");
    }

    #[test]
    fn test_parameter_name_dewpoint() {
        assert_eq!(parameter_name(0, 0, 6), "Dewpoint Temperature");
    }

    #[test]
    fn test_parameter_name_relative_humidity() {
        assert_eq!(parameter_name(0, 1, 1), "Relative Humidity");
    }

    #[test]
    fn test_parameter_name_u_wind() {
        assert_eq!(parameter_name(0, 2, 2), "U-Component of Wind");
    }

    #[test]
    fn test_parameter_name_v_wind() {
        assert_eq!(parameter_name(0, 2, 3), "V-Component of Wind");
    }

    #[test]
    fn test_parameter_name_total_precipitation() {
        assert_eq!(parameter_name(0, 1, 8), "Total Precipitation");
    }

    #[test]
    fn test_parameter_name_geopotential_height() {
        assert_eq!(parameter_name(0, 3, 5), "Geopotential Height");
    }

    #[test]
    fn test_parameter_name_specific_humidity() {
        assert_eq!(parameter_name(0, 1, 0), "Specific Humidity");
    }

    #[test]
    fn test_parameter_name_wind_speed() {
        assert_eq!(parameter_name(0, 2, 1), "Wind Speed");
    }

    #[test]
    fn test_parameter_name_ncep_local_categorical_rain() {
        assert_eq!(parameter_name(0, 1, 192), "Categorical Rain (yes=1; no=0)");
    }

    #[test]
    fn test_parameter_name_unknown() {
        assert_eq!(parameter_name(255, 255, 255), "Unknown");
    }

    #[test]
    fn test_parameter_name_unknown_within_known_category() {
        // Use a number that's between known entries and the catch-all
        assert_eq!(parameter_name(0, 0, 250), "Unknown");
    }

    // ---- parameter_units tests ----

    #[test]
    fn test_parameter_units_temperature_kelvin() {
        assert_eq!(parameter_units(0, 0, 0), "K");
    }

    #[test]
    fn test_parameter_units_relative_humidity_percent() {
        assert_eq!(parameter_units(0, 1, 1), "%");
    }

    #[test]
    fn test_parameter_units_wind_ms() {
        assert_eq!(parameter_units(0, 2, 1), "m/s");
    }

    #[test]
    fn test_parameter_units_precipitation_kgm2() {
        assert_eq!(parameter_units(0, 1, 8), "kg/m\u{b2}");
    }

    #[test]
    fn test_parameter_units_pressure_pa() {
        assert_eq!(parameter_units(0, 3, 0), "Pa");
    }

    #[test]
    fn test_parameter_units_geopotential_height() {
        assert_eq!(parameter_units(0, 3, 5), "gpm");
    }

    #[test]
    fn test_parameter_units_lapse_rate() {
        assert_eq!(parameter_units(0, 0, 8), "K/m");
    }

    #[test]
    fn test_parameter_units_heat_flux() {
        assert_eq!(parameter_units(0, 0, 10), "W/m\u{b2}");
    }

    #[test]
    fn test_parameter_units_unknown() {
        assert_eq!(parameter_units(255, 255, 255), "?");
    }

    // ---- level_name tests ----

    #[test]
    fn test_level_name_surface() {
        assert_eq!(level_name(1), "Ground or Water Surface");
    }

    #[test]
    fn test_level_name_isobaric() {
        assert_eq!(level_name(100), "Isobaric Surface");
    }

    #[test]
    fn test_level_name_msl() {
        assert_eq!(level_name(101), "Mean Sea Level");
    }

    #[test]
    fn test_level_name_height_above_ground() {
        assert_eq!(level_name(103), "Specified Height Level Above Ground");
    }

    #[test]
    fn test_level_name_hybrid() {
        assert_eq!(level_name(105), "Hybrid Level");
    }

    #[test]
    fn test_level_name_entire_atmosphere() {
        assert_eq!(level_name(200), "Entire Atmosphere (as single layer)");
    }

    #[test]
    fn test_level_name_tropopause() {
        assert_eq!(level_name(7), "Tropopause");
    }

    #[test]
    fn test_level_name_pbl() {
        assert_eq!(level_name(220), "Planetary Boundary Layer");
    }

    #[test]
    fn test_level_name_cloud_ceiling() {
        assert_eq!(level_name(215), "Cloud Ceiling");
    }

    #[test]
    fn test_level_name_unknown() {
        assert_eq!(level_name(199), "Unknown Level Type");
    }

    #[test]
    fn test_level_name_unknown_high_value() {
        assert_eq!(level_name(255), "Unknown Level Type");
    }
}

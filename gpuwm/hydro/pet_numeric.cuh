// SPDX-License-Identifier: Apache-2.0
// Derived from NOAA-OWP/evapotranspiration at
// 6c410f2a8bc843043ad25e13239928031b8c5e64, Apache-2.0.
// F. L. Ogden, NOAA National Weather Service, 2020.
// Modified for ArWen: CUDA device functions; host diagnostics and process exits
// removed (the kernel records invalid inputs in resident status instead).
// Numerical expressions, including upstream conditional defaults, retained.
// See PET-LICENSE and PET-TERMS.md distributed alongside this source.
#define TRUE  1
#define FALSE 0

#define CP  1.006e+03  //  specific heat of air at constant pressure, J/(kg K), a physical constant.
#define KV2 0.1681     //  von Karman's constant squared, equal to 0.41 squared, unitless
#define TK  273.15     //  temperature in Kelvin at zero degree Celcius
#define SB  5.67e-08   //  stefan_boltzmann_constant in units of W/m^2/K^4

// NOTE: SET YOUR EDIT WINDOW TO 120 CHARACTER WIDTH TO READ THIS CODE IN ITS ENTIRETY.

//#####################################
// evapotranspiration (ET) module,
// Version 1.0 by Fred L. Ogden, NOAA-NWS-OWP, May, 2020.
//
// includes five different methods to clculate ET, from Chow, Maidment & Mays Textbook, and UNFAO Penman-Monteith:
// 1. energy balance method
// 2. aerodynamic method
// 3. combination method, which combines 1 & 2.
// 4. Priestley-Taylor method, which assumes the ratio between 1 & 2, and only calculates 1.
// 5. Penman-Monteith method, which requires a value of canopy resistance term, and does not rely on 1 or 2.
//
// This subroutine requires a considerable amount of meteorological data as input.
// a. temperature and (relative-humidity or specific humidity) and the heights at which they are measured.
// b. near surface wind speed measurement and the height at which it was measured.
// c. the ambient atmospheric temperature lapse rate
// d. the fraction of the sky covered by clouds
// e. (optional) the height above ground to the cloud base. If not provided, then assumed.
// f. the day of the year (1-366) and time of day (UTC only!)
// g. the skin temperature of the earth's surface, TODO: should come from another module to calc. soil or veg. temp.
// h. the zero-plane roughness height of the atmospheric boundary layer assuming log-law behavior (from land cover)
// i. the average root zone soil temperature, or near-surface water temperature in the case of lake evaporation.
// j. the incoming solar (shortwave) radiation.  If not provided it is computed from d,e,f, using an
//    updated method similar to the one presented in Bras, R.L. Hydrology.  Requires value of the Linke atmospheric
//    turbidity factor, which varies from 2 for clear mountain air to 5 for smoggy air.  According to Hove & Manyumbu
//    2012, who calculated values over Zimbabwe that varied from 2.14 to 3.71.  Other values exist in the literature.
//    TODO: This turbidity factor could be calculated from satellite obs. or maybe NOAA already does this?
//
// All radiation calculations needed for 1, 3, 4, and 5 require net radiation calculations at the land surface.
// the net radiation is calculated using a, c, d, e, f, g, j, plus the Linke turbidity factor, which can be estimate
// from satellite observations.
//

// NOTE THE VALUE OF evapotranspiration_params.zero_plane_displacement_height COMES FROM LAND COVER DATA.
// Taken from:    https://websites.pmc.ucsc.edu/~jnoble/wind/extrap/
//
//Roughness Roughness  Landscape Type
// Class    Length (m)
//----------------------------------------------------------------------------------------------------------------
// 0         0.0002     Smooth water surface
// 0.2       0.0005     Inlet water
// 0.5       0.0024     Completely open terrain, smooth surface, e.g. concrete runways in airports, mowed grass, etc.
// 1         0.03       Open agricultural area without fences and hedgerows and very scattered buildings. Only softly
//                        rounded hills
// 1.5       0.055      Agricultural land with some houses and 8 metre tall sheltering hedgerows with a distance of
//                        approximately 1250 metres
// 2         0.1        Agricultural land with some houses and 8 metre tall sheltering hedgerows with a distance of
//                        approximately 500 metres
// 2.5       0.2        Agricultural land with many houses, shrubs and plants, or 8 metre tall sheltering hedgerows
//                        with a distance of approximately 250 metres
// 3         0.4        Villages, small towns, agricultural land with many or tall sheltering hedgerows, forests
//                        and very rough and uneven terrain
// 3.5       0.8        Larger cities with tall buildings
// 4         1.6        Very large cities with tall buildings and skyscrapers
// Roughness definitions according to the European Wind Atlas.
//
// According to the UN FAO Penman-Monteith example here:
//        http://www.fao.org/3/X0490E/x0490e06.htm#aerodynamic%20resistance%20(ra)
// the zero plane roughness length,"d" can be approximated as 2/3 of the vegetation height (H): d=2/3*H.
// the momentum roughness height "zom" can be estimated as 0.123*H.
// the heat transfer roughness height "zoh" can be approximated as 0.1 * zom.
//-----------------------------------------------------------------------------------------------------------------

struct pevapotranspiration_options  // these determine which method is applied to calculate ET.
{
  // element NAME                       DESCRIPTION
  //____________________________________________________________________________________________________________________

  // NOTE: these following options are exclusive.  Only one should be true for a particular catchment.
  int use_aerodynamic_method;       // set to TRUE if using just the aerodynamic method for calculating PET
  int use_combination_method;       // set to TRUE if using just the combination method for calculating PET
  int use_penman_monteith_method;   // set to TRUE if using just the Penman Monteith method for calculating PET
};

struct pevapotranspiration_params
{
  // element NAME                                    DESCRIPTION
  //____________________________________________________________________________________________________________________
  double wind_speed_measurement_height_m;          // set to 0.0 if unknown, will default =2.0 [m]
  double humidity_measurement_height_m;            // set to 0.0 if unknown, will default =2.0 [m]
  double vegetation_height_m;                      // TODO this should come from land cover data and a veg. height model
  double zero_plane_displacement_height_m;         // depends on surface roughness [m],
  double momentum_transfer_roughness_length_m;     // poorly defined.  If unknown, pass down 0.0 as a default
  double heat_transfer_roughness_length_m;         // poorly defined.  If unknown, pass down 0.0 as a default
};

struct pevapotranspiration_forcing
{
  // element NAME                          DESCRIPTION
  //___________________________________________________________________________________________________________________
  double net_radiation_W_per_sq_m;       // NOTE: ground heat flux is subtracted out in calculation subroutine
  double air_temperature_C;
  double relative_humidity_percent;      // this and specific_humidity_2m_kg_per_kg are redundant, so-
  double specific_humidity_2m_kg_per_kg; // specify the missing one using a negative number, the other will be used
  double air_pressure_Pa;
  double wind_speed_m_per_s;             // this is the value measured 2 m above the canopy
  double canopy_resistance_sec_per_m;    // depends on vegetation type and point in growing season
  double water_temperature_C;            // if >100, assume 15 C.  Used to calculate latent heat of vaporization
  double ground_heat_flux_W_per_sq_m;    // from a model or assumed =0.  typically small on average
};

struct surface_radiation_params
{
  // element NAME                          DESCRIPTION
  //____________________________________________________________________________________________________________________
  double surface_longwave_emissivity;  // dimensionless (0-1) < 1.0 for water and snow, all other surfaces = 1.0
  double surface_shortwave_albedo;     // dimensionless (0-1) from land cover.  Dynamic seasonally
};

struct surface_radiation_forcing
{
  // element NAME                          DESCRIPTION
  //____________________________________________________________________________________________________________________
  double incoming_shortwave_radiation_W_per_sq_m;  // TODO could be calculated if unavailable, but not now.
  double incoming_longwave_radiation_W_per_sq_m;  // set to a large negative number if unknown (e.g. -1.0e-5)
  double surface_skin_temperature_C;   // from a model or assumed...  smartly.  could be soil/rock, veg., snow, water
};

struct intermediate_vars
{
  // element NAME                       DESCRIPTION
  //____________________________________________________________________________________________________________________
  double liquid_water_density_kg_per_m3;       // rho_w
  double water_latent_heat_of_vaporization_J_per_kg;    // eqn 2.7.6 Chow etal., // aka 'lambda'
  double air_saturation_vapor_pressure_Pa;
  double air_actual_vapor_pressure_Pa;
  double vapor_pressure_deficit_Pa;            // VPD
  double moist_air_density_kg_per_m3;          // rho_a
  double slope_sat_vap_press_curve_Pa_s;       // delta
  //double water_latent_heat_of_vaporization_J_per_kg;
  double psychrometric_constant_Pa_per_C;      // gamma
};

struct pet_model {
 struct pevapotranspiration_options pet_options;
 struct pevapotranspiration_params pet_params;
 struct pevapotranspiration_forcing pet_forcing;
 struct intermediate_vars inter_vars;
 struct surface_radiation_params surf_rad_params;
 struct surface_radiation_forcing surf_rad_forcing;
};
typedef struct pet_model pet_model;

#ifndef PET_TOOLS_H
#define PET_TOOLS_H

__device__ double calculate_net_radiation_W_per_sq_m(pet_model *model);

__device__ double calculate_aerodynamic_resistance(pet_model *model);

__device__ void calculate_intermediate_variables(pet_model *model);

__device__ int is_fabs_less_than_eps(double a,double epsilon);  // returns TRUE iff fabs(a)<epsilon

__device__ double calc_air_saturation_vapor_pressure_Pa(double air_temperature_C);

__device__ double calc_slope_of_air_saturation_vapor_pressure_Pa_per_C(double air_temperature_C);

__device__ double calc_liquid_water_density_kg_per_m3(double water_temperature_C);



//############################################################*
// subroutine to calculate net radiation from all components  *
// of the radiation budget, using values provided by WRF      *
// Reference, Bras, R.L., HYDROLOGY an Introduction to        *
// Hydrologic Science, Addison Wesley, 1990.                  *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double calculate_net_radiation_W_per_sq_m(pet_model *model)
{
  // local variables
  double net_radiation_W_per_sq_m;
  double outgoing_longwave_radiation_W_per_sq_m;
  double stefan_boltzmann_constant = SB;         //W/m^2/K^4
  double surface_longwave_albedo;

  // CALCULATE OUTGOING LONGWAVE RADIATION FLUX FROM SURFACE
  // Native noncontracted radiation budget: preserve its multiply/subtract
  // boundaries even in fast mode. Fusing these changes cat-27 row1 PET 5 ULP.
  outgoing_longwave_radiation_W_per_sq_m=__dmul_rn(
      __dmul_rn(model->surf_rad_params.surface_longwave_emissivity,stefan_boltzmann_constant),
      pow(model->surf_rad_forcing.surface_skin_temperature_C+TK,4.0));
                                         // must convert C to K

  if(0.999 < model->surf_rad_params.surface_longwave_emissivity)
  {
    surface_longwave_albedo=0.0;    // soil, rock, concrete, asphalt, vegetation, snow
  }
  else
  {
    surface_longwave_albedo=0.03;   // water - actually not this simple, but close enough for now
  }

  net_radiation_W_per_sq_m=__dsub_rn(__dadd_rn(
      __dmul_rn(1.0-model->surf_rad_params.surface_shortwave_albedo,
                model->surf_rad_forcing.incoming_shortwave_radiation_W_per_sq_m),
      __dmul_rn(1.0-surface_longwave_albedo,
                model->surf_rad_forcing.incoming_longwave_radiation_W_per_sq_m)),
      outgoing_longwave_radiation_W_per_sq_m);

  return(net_radiation_W_per_sq_m);
}

//############################################################*
// subroutine to calculate aerodynamic resistance term needed *
// in both the Penman-Monteith FAO reference PET procedure.    *
// and the aerodynamic, combination, and Priestley-Taylor     *
// PET calculation methods.                                    *
// Reference: http://www.fao.org/3/X0490E/x0490e06.htm        *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double calculate_aerodynamic_resistance(pet_model *model)
{
  // define local variables to ease computations:
  double wind_speed_measurement_height_m = model->pet_params.wind_speed_measurement_height_m;
  double humidity_measurement_height_m = model->pet_params.humidity_measurement_height_m; // default =2.0 [m],
  double zero_plane_displacement_height_m = model->pet_params.zero_plane_displacement_height_m;// depends on surface roughness [m],
  double momentum_transfer_roughness_length_m = model->pet_params.momentum_transfer_roughness_length_m; // [m],
  double heat_transfer_roughness_length_m = model->pet_params.heat_transfer_roughness_length_m;     // [m],
  double wind_speed_m_per_s = model->pet_forcing.wind_speed_m_per_s;                    // [m s-1].

  double ra,zm,zh,d,zom,zoh,k,uz;
  double von_karman_constant_squared=KV2;  // this is dimensionless universal constant [-], K=0.41, squared.

  // input sanity checks.
  if(1.0e-06 >=wind_speed_measurement_height_m ) wind_speed_measurement_height_m=2.0;  // standard measurement height
  if(1.0e-06 >=humidity_measurement_height_m )     humidity_measurement_height_m=2.0;  // standard measurement height
  if(wind_speed_m_per_s <= 0.0 ) {
    wind_speed_m_per_s = 0.01; // small positive value to avoid divide by zero
  }
  // convert to smaller local variable names to keep equation readable
  zm=wind_speed_measurement_height_m;
  zh=humidity_measurement_height_m;
  d= zero_plane_displacement_height_m;
  zom=momentum_transfer_roughness_length_m;
  zoh=heat_transfer_roughness_length_m;
  uz=wind_speed_m_per_s;

  // here log is the natural logarithm.
  // add a necessary condition for ra calculation
  // TODO: check zero displacement height (parsing from config_file?)
  if (d >= zh) {
    d = 2.0/3.0 * zh;
  }

  ra=log((zm-d)/zom)*log((zh-d)/zoh)/(von_karman_constant_squared*uz);  // this is the equation for the aero. resist.
                                                                      // from the FAO reference PET document.

  return(ra);
}

//############################################################*
// function to calculate saturation vapor pressure of air     *
// based on the exponential relationship defined in Chow,     *
// Maidment, and Mays,textbook.  Input is air temp C          *
// F.L. Ogden, NOAA National Weather Service, 2020            *
// This function is guaranteed to return a value > 0.         *
//############################################################*
__device__ double calc_air_saturation_vapor_pressure_Pa(double air_temperature_C)
{
  double air_sat_vap_press_Pa= 611.0*exp(17.27*air_temperature_C/(237.3+air_temperature_C));  // it is 237.3

  return(air_sat_vap_press_Pa);
}


//#################################################################*
// function to calculate slope saturation vapor pressure curve     *
// based on the exponential relationship defined in Chow,          *
// Maidment, and Mays,textbook, Eqn. 3.2.10.  Input is air temp C  *
// calls function calc_air_saturation_vapor_pressure_Pa().         *
// F.L. Ogden, NOAA National Weather Service, 2020                 *
// This function is guaranteed to return a value > 0.              *
//#################################################################*
__device__ double calc_slope_of_air_saturation_vapor_pressure_Pa_per_C(double air_temperature_C)
{
  // calc_air_saturation_vapor_pressure_Pa is guaranteed to return >0
  // and will error/exit if air_temperature_C <= -237.3
  double slope_of_air_sat_vap_press_curve_Pa_per_C=
                         4098.0*calc_air_saturation_vapor_pressure_Pa(air_temperature_C)/
                         pow((237.3+air_temperature_C),2.0);  // it is 237.3
  return(slope_of_air_sat_vap_press_curve_Pa_per_C);
}

//############################################################*
// function to calculate density of liquid water by empirical *
// equation, as a function of water temperature in C          *
// fit of water density vs. temperature data by FLO.  Data    *
// from water properties table in Chow, Maidment, and Mays    *
// textbook.  Fit using tblcurve program.  r^2=0.9975         *
// this fit produces rho_w=1000.151 kg/m3 at T=0C, so I limit *
// it to 1000.0.  Doesn't accurately predict max. density at  *
// T=4C, so this equation is an approximation.  TODO maybe a  *
// better model exists.  But, this value is not required at   *
// super high precision to convert latent heat flux into a    *
// depth of water.                                            *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double calc_liquid_water_density_kg_per_m3(double water_temperature_C)
{
  double a=0.0009998492;  // this precision is necessary
  double b=4.9716595e-09; // ditto.

  double water_density_kg_per_m3=1.0/(a+b*water_temperature_C*water_temperature_C);

  if(1000<water_density_kg_per_m3) water_density_kg_per_m3=1000.0;   // this empirical function yield 1000.151 at 0C.

  return(water_density_kg_per_m3);
}


// Function to calculate hydrological variables needed for evapotranspiration calculation
__device__ void calculate_intermediate_variables(pet_model* model)
{
  // local variables
  double psychrometric_constant_Pa_per_C;
  double slope_sat_vap_press_curve_Pa_s;
  double air_saturation_vapor_pressure_Pa;
  double air_actual_vapor_pressure_Pa;
  double moist_air_density_kg_per_m3;
  double water_latent_heat_of_vaporization_J_per_kg;
  double moist_air_gas_constant_J_per_kg_K;
  double vapor_pressure_deficit_Pa;
  double liquid_water_density_kg_per_m3;

  // IF SOIL WATER TEMPERATURE NOT PROVIDED, USE A SANE VALUE
  if(100.0 > model->pet_forcing.water_temperature_C) model->pet_forcing.water_temperature_C=22.0; // growing season

  // CALCULATE VARS NEEDED FOR THE ALL METHODS:

  liquid_water_density_kg_per_m3 = calc_liquid_water_density_kg_per_m3(model->pet_forcing.water_temperature_C); // rho_w

  // all methods other than radiation balance method involve at least some of the aerodynamic method calculations

  // IF HEAT/MOMENTUM ROUGHNESS LENGTHS NOT GIVEN, USE DEFAULTS SO THAT THEIR RATIO IS EQUAL TO 1.
  if((1.0e-06> model->pet_params.heat_transfer_roughness_length_m) ||
   (1.0e-06> model->pet_params.momentum_transfer_roughness_length_m))   // zero should be passed down if these are unknown
  {
    model->pet_params.heat_transfer_roughness_length_m     =1.0;     // decent default values, and the ratio of these is 1.0
    model->pet_params.momentum_transfer_roughness_length_m =1.0;
  }

  // e_sat is needed for all aerodynamic and Penman-Monteith methods

  air_saturation_vapor_pressure_Pa=calc_air_saturation_vapor_pressure_Pa(model->pet_forcing.air_temperature_C);

  if( (0.0 < model->pet_forcing.relative_humidity_percent) && (100.0 >= model->pet_forcing.relative_humidity_percent))
  {
    // meaningful relative humidity value provided
    air_actual_vapor_pressure_Pa=model->pet_forcing.relative_humidity_percent/100.0 * air_saturation_vapor_pressure_Pa;

    // calculate specific humidity, q_v
    model->pet_forcing.specific_humidity_2m_kg_per_kg=0.622*air_actual_vapor_pressure_Pa/model->pet_forcing.air_pressure_Pa;
  }
  else
  {
    // if here, we must be using AORC forcing that provides specific humidity instead of relative humidity
    air_actual_vapor_pressure_Pa=model->pet_forcing.specific_humidity_2m_kg_per_kg*model->pet_forcing.air_pressure_Pa/0.622;
    if(air_actual_vapor_pressure_Pa > air_saturation_vapor_pressure_Pa)
    {
      // this is bad.   Actual vapor pressure of air should not be higher than saturated value.
      // reset to something meaningful
      air_actual_vapor_pressure_Pa=0.65*air_saturation_vapor_pressure_Pa;
    }
  }

  // VPD
  vapor_pressure_deficit_Pa = air_saturation_vapor_pressure_Pa - air_actual_vapor_pressure_Pa;

  moist_air_gas_constant_J_per_kg_K=287.0*(1.0+0.608*model->pet_forcing.specific_humidity_2m_kg_per_kg); //R_a

  moist_air_density_kg_per_m3=model->pet_forcing.air_pressure_Pa/(moist_air_gas_constant_J_per_kg_K*
                              (model->pet_forcing.air_temperature_C+TK)); // rho_a

  // DELTA
  slope_sat_vap_press_curve_Pa_s=calc_slope_of_air_saturation_vapor_pressure_Pa_per_C(model->pet_forcing.air_temperature_C);

  // gamma
  water_latent_heat_of_vaporization_J_per_kg=2.501e+06-2370.0*model->pet_forcing.water_temperature_C;  // eqn 2.7.6 Chow etal.
                                                                                              // aka 'lambda'
  psychrometric_constant_Pa_per_C=CP*model->pet_forcing.air_pressure_Pa/
                                  (0.622*water_latent_heat_of_vaporization_J_per_kg);

  model->inter_vars.liquid_water_density_kg_per_m3=liquid_water_density_kg_per_m3;
  model->inter_vars.water_latent_heat_of_vaporization_J_per_kg=water_latent_heat_of_vaporization_J_per_kg;
  model->inter_vars.air_saturation_vapor_pressure_Pa=air_saturation_vapor_pressure_Pa;
  model->inter_vars.air_actual_vapor_pressure_Pa=air_actual_vapor_pressure_Pa;
  model->inter_vars.vapor_pressure_deficit_Pa=vapor_pressure_deficit_Pa;
  model->inter_vars.moist_air_density_kg_per_m3=moist_air_density_kg_per_m3;
  model->inter_vars.slope_sat_vap_press_curve_Pa_s=slope_sat_vap_press_curve_Pa_s;
  model->inter_vars.psychrometric_constant_Pa_per_C=psychrometric_constant_Pa_per_C;
}

__device__ int is_fabs_less_than_eps(double a,double epsilon)  // returns true if fabs(a)<epsilon
{
  if(fabs(a)<epsilon) return(TRUE);
  else                return(FALSE);
}

#endif // PET_CALC_PPROPERTY_H

#ifndef PET_ENERGY_BALANCE_METHOD_H
#define PET_ENERGY_BALANCE_METHOD_H

// FUNCTION AND SUBROUTINE PROTOTYPES

__device__ double pevapotranspiration_energy_balance_method(pet_model *model);


//############################################################*
// subroutine to calculate evapotranspiration using           *
// Chow, Maidment, and Mays textbook                          *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double pevapotranspiration_energy_balance_method(pet_model *model)
{
  // local varibles
  double water_latent_heat_of_vaporization_J_per_kg;
  double liquid_water_density_kg_per_m3;
  double lambda_pet;
  double radiation_balance_pevapotranspiration_rate_m_per_s;

  // from FAO document: cp specific heat at constant pressure, 1.013 10-3 [MJ kg-1 �C-1],

  // IF SOIL WATER TEMPERATURE NOT PROVIDED, USE A SANE VALUE
  if(100.0 > model->pet_forcing.water_temperature_C) model->pet_forcing.water_temperature_C=22.0; // growing season

  // CALCULATE VARS NEEDED FOR THE ALL METHODS:

  liquid_water_density_kg_per_m3 = calc_liquid_water_density_kg_per_m3(model->pet_forcing.water_temperature_C); // rho_w

  water_latent_heat_of_vaporization_J_per_kg=2.501e+06-2370.0*model->pet_forcing.water_temperature_C;  // eqn 2.7.6 Chow etal.
                                                                                              // aka 'lambda'

  // We need this in all options except for aerodynamic or Penman-Monteith methods.
  // Radiation balance is the simplest method.  Involves only radiation calculations, no aerodynamic calculations.

  lambda_pet=0.0;
  if( (model->pet_options.use_aerodynamic_method == FALSE ) && (model->pet_options.use_penman_monteith_method==FALSE) )
  {
    // This is equation 3.5.9 from Chow, Maidment, and Mays textbook.
    lambda_pet=model->pet_forcing.net_radiation_W_per_sq_m;
    radiation_balance_pevapotranspiration_rate_m_per_s=lambda_pet/
                                  (liquid_water_density_kg_per_m3*water_latent_heat_of_vaporization_J_per_kg);
  }
  return(radiation_balance_pevapotranspiration_rate_m_per_s);
}

#endif // PET_ENERGY_BALANCE_METHOD_H

#ifndef PET_AERODYNAMIC_METHOD_H
#define PET_AERODYNAMIC_METHOD_H

// FUNCTION AND SUBROUTINE PROTOTYPES

__device__ double pevapotranspiration_aerodynamic_method(pet_model *model);


//############################################################*
// subroutine to calculate evapotranspiration using           *
// Chow, Maidment, and Mays textbook                          *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double pevapotranspiration_aerodynamic_method(pet_model *model)
{
  // local varibles
  double moist_air_density_kg_per_m3;
  double vapor_pressure_deficit_Pa;
  double liquid_water_density_kg_per_m3;
  double aerodynamic_method_pevapotranspiration_rate_m_per_s;
  double mass_flux;
  double von_karman_constant_squared=(double)KV2;  // a constant equal to 0.41 squared

  calculate_intermediate_variables(model);

  liquid_water_density_kg_per_m3 = model->inter_vars.liquid_water_density_kg_per_m3;
  vapor_pressure_deficit_Pa=model->inter_vars.vapor_pressure_deficit_Pa;
  moist_air_density_kg_per_m3=model->inter_vars.moist_air_density_kg_per_m3;

  if( model->pet_options.use_penman_monteith_method == FALSE)  // we don't use this term in Penman-Monteith method
  {
    // This is equation 3.5.16 from Chow, Maidment, and Mays textbook.
    mass_flux = 0.622*von_karman_constant_squared*moist_air_density_kg_per_m3*      // kg per sq. meter per sec.
                vapor_pressure_deficit_Pa*model->pet_forcing.wind_speed_m_per_s/
                (model->pet_forcing.air_pressure_Pa*
                pow(log(model->pet_params.wind_speed_measurement_height_m/model->pet_params.zero_plane_displacement_height_m),2.0));
    aerodynamic_method_pevapotranspiration_rate_m_per_s=mass_flux/liquid_water_density_kg_per_m3;
  }

  return(aerodynamic_method_pevapotranspiration_rate_m_per_s);
}

#endif // PET_AERODYNAMIC_METHOD_H

#ifndef PET_COMBINATION_METHOD_H
#define PET_COMBINATION_METHOD_H

// FUNCTION AND SUBROUTINE PROTOTYPES

__device__ double pevapotranspiration_combination_method(pet_model *model);

//############################################################*
// subroutine to calculate evapotranspiration using           *
// Chow, Maidment, and Mays textbook                          *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double pevapotranspiration_combination_method(pet_model *model)
{
  // local varibles
  double psychrometric_constant_Pa_per_C;
  double slope_sat_vap_press_curve_Pa_s;
  double moist_air_density_kg_per_m3;
  double water_latent_heat_of_vaporization_J_per_kg;
  double vapor_pressure_deficit_Pa;
  double liquid_water_density_kg_per_m3;
  double lambda_pet;
  double radiation_balance_pevapotranspiration_rate_m_per_s;
  double aerodynamic_method_pevapotranspiration_rate_m_per_s;
  double instantaneous_pet_rate_m_per_s;
  double mass_flux;
  double delta;
  double gamma;
  double von_karman_constant_squared=(double)KV2;  // a constant equal to 0.41 squared

  calculate_intermediate_variables(model);

  liquid_water_density_kg_per_m3 = model->inter_vars.liquid_water_density_kg_per_m3;
  vapor_pressure_deficit_Pa=model->inter_vars.vapor_pressure_deficit_Pa;
  moist_air_density_kg_per_m3=model->inter_vars.moist_air_density_kg_per_m3;
  slope_sat_vap_press_curve_Pa_s=model->inter_vars.slope_sat_vap_press_curve_Pa_s;
  water_latent_heat_of_vaporization_J_per_kg=model->inter_vars.water_latent_heat_of_vaporization_J_per_kg;
  psychrometric_constant_Pa_per_C=model->inter_vars.psychrometric_constant_Pa_per_C;

  delta=slope_sat_vap_press_curve_Pa_s;
  gamma=psychrometric_constant_Pa_per_C;

  if(model->pet_options.use_combination_method==TRUE)
    lambda_pet=0.0;
  if( (model->pet_options.use_aerodynamic_method == FALSE ) && (model->pet_options.use_penman_monteith_method==FALSE) )
  {
    // This is equation 3.5.9 from Chow, Maidment, and Mays textbook.
    lambda_pet=model->pet_forcing.net_radiation_W_per_sq_m;
    radiation_balance_pevapotranspiration_rate_m_per_s=lambda_pet/
                                        (liquid_water_density_kg_per_m3*water_latent_heat_of_vaporization_J_per_kg);
    mass_flux = 0.622*von_karman_constant_squared*moist_air_density_kg_per_m3*      // kg per sq. meter per sec.
                vapor_pressure_deficit_Pa*model->pet_forcing.wind_speed_m_per_s/
                (model->pet_forcing.air_pressure_Pa*
                pow(log(model->pet_params.wind_speed_measurement_height_m/model->pet_params.zero_plane_displacement_height_m),2.0));

    aerodynamic_method_pevapotranspiration_rate_m_per_s=mass_flux/liquid_water_density_kg_per_m3;
  }
  {
    // This is equation 3.5.26 from Chow, Maidment, and Mays textbook
    // Preserve the native rounding of the two weighted fluxes before adding.
    // Contracting the final multiply/add changes captured combination PET.
    instantaneous_pet_rate_m_per_s=__dadd_rn(
        __dmul_rn(delta/(delta+gamma),radiation_balance_pevapotranspiration_rate_m_per_s),
        __dmul_rn(gamma/(delta+gamma),aerodynamic_method_pevapotranspiration_rate_m_per_s));
  }
  return (instantaneous_pet_rate_m_per_s);
}

#endif // PET_COMBINATION_METHOD_H

#ifndef PET_PRIESTLEY_TAYLOR_METHOD_H
#define PET_PRIESTLEY_TAYLOR_METHOD_H

// FUNCTION AND SUBROUTINE PROTOTYPES

__device__ double pevapotranspiration_priestley_taylor_method(pet_model *model);

//############################################################*
// subroutine to calculate evapotranspiration using           *
// Chow, Maidment, and Mays textbook                          *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double pevapotranspiration_priestley_taylor_method(pet_model *model)
{
  // local varibles
  double psychrometric_constant_Pa_per_C;
  double slope_sat_vap_press_curve_Pa_s;
  double water_latent_heat_of_vaporization_J_per_kg;
  double liquid_water_density_kg_per_m3;
  double lambda_pet;
  double radiation_balance_pevapotranspiration_rate_m_per_s;
  double instantaneous_pet_rate_m_per_s;
  double delta;
  double gamma;

  calculate_intermediate_variables(model);

  liquid_water_density_kg_per_m3 = model->inter_vars.liquid_water_density_kg_per_m3;
  slope_sat_vap_press_curve_Pa_s=model->inter_vars.slope_sat_vap_press_curve_Pa_s;
  water_latent_heat_of_vaporization_J_per_kg=model->inter_vars.water_latent_heat_of_vaporization_J_per_kg;
  psychrometric_constant_Pa_per_C=model->inter_vars.psychrometric_constant_Pa_per_C;

  delta=slope_sat_vap_press_curve_Pa_s;
  gamma=psychrometric_constant_Pa_per_C;

  lambda_pet=0.0;
  if( (model->pet_options.use_aerodynamic_method == FALSE ) && (model->pet_options.use_penman_monteith_method==FALSE) )
  {
    // This is equation 3.5.9 from Chow, Maidment, and Mays textbook.
    lambda_pet=model->pet_forcing.net_radiation_W_per_sq_m;
    radiation_balance_pevapotranspiration_rate_m_per_s=lambda_pet/
                                        (liquid_water_density_kg_per_m3*water_latent_heat_of_vaporization_J_per_kg);
  }
  instantaneous_pet_rate_m_per_s=1.3*delta/(delta+gamma)*radiation_balance_pevapotranspiration_rate_m_per_s;
  return(instantaneous_pet_rate_m_per_s);
}

#endif // PET_PRIESTLEY_TAYLOR_METHOD_H

#ifndef PET_PENMAN_MONTEITH_METHOD_H
#define PET_PENMAN_MONTEITH_METHOD_H

// FUNCTION AND SUBROUTINE PROTOTYPES

__device__ double pevapotranspiration_penman_monteith_method(pet_model *model);

__device__ double penman_monteith_pet_calculation
(
  double delta,
  double gamma,
  double moist_air_density_kg_per_m3,
  double vapor_pressure_deficit_Pa,
  struct pet_model *model
);

//############################################################*
// subroutine to calculate evapotranspiration using Penman-   *
// Monteith FAO reference ET procedure.                       *
// Reference: http://www.fao.org/3/X0490E/x0490e06.htm        *
// F.L. Ogden, NOAA National Weather Service, 2020            *
//############################################################*
__device__ double pevapotranspiration_penman_monteith_method(pet_model *model)
{
  // local varibles
  double instantaneous_pet_rate_m_per_s;
  double psychrometric_constant_Pa_per_C;
  double slope_sat_vap_press_curve_Pa_s;
  double moist_air_density_kg_per_m3;
  double water_latent_heat_of_vaporization_J_per_kg;
  double vapor_pressure_deficit_Pa;
  double liquid_water_density_kg_per_m3;
  double lambda_pet;
  double delta;
  double gamma;

  calculate_intermediate_variables(model);

  liquid_water_density_kg_per_m3 = model->inter_vars.liquid_water_density_kg_per_m3;
  vapor_pressure_deficit_Pa=model->inter_vars.vapor_pressure_deficit_Pa;
  moist_air_density_kg_per_m3=model->inter_vars.moist_air_density_kg_per_m3;
  slope_sat_vap_press_curve_Pa_s=model->inter_vars.slope_sat_vap_press_curve_Pa_s;
  water_latent_heat_of_vaporization_J_per_kg=model->inter_vars.water_latent_heat_of_vaporization_J_per_kg;
  psychrometric_constant_Pa_per_C=model->inter_vars.psychrometric_constant_Pa_per_C;

  delta=slope_sat_vap_press_curve_Pa_s;
  gamma=psychrometric_constant_Pa_per_C;

  if(model->pet_options.use_penman_monteith_method==TRUE)
  {
    lambda_pet = penman_monteith_pet_calculation(delta,gamma,moist_air_density_kg_per_m3,vapor_pressure_deficit_Pa,model);
  }

  instantaneous_pet_rate_m_per_s= lambda_pet/(liquid_water_density_kg_per_m3*water_latent_heat_of_vaporization_J_per_kg);

  return(instantaneous_pet_rate_m_per_s);  // meters per second

}

//#####################################################################################*
// subroutine to calculate latent heat flux (lambda*pet)                                *
// using the Penman-Monteith equation as described by FAO                              *
//   see:   http://www.fao.org/3/X0490E/x0490e06.htm#aerodynamic%20resistance%20(ra)   *
// After Allen et al. ASCE Reference ET calculation method                             *
// F.L. Ogden, NOAA National Weather Service, 2020                                     *
//#####################################################################################*
__device__ double penman_monteith_pet_calculation
(
  double delta,
  double gamma,
  double moist_air_density_kg_per_m3,
  double vapor_pressure_deficit_Pa,
  struct pet_model *model
)

{
  // local varibles
  double pm_numerator;
  double pm_denominator;
  double aerodynamic_resistance_s_per_m;

  // this method requires more calculations
  if(is_fabs_less_than_eps(model->pet_params.vegetation_height_m,1.0e-06)==TRUE)
  {
    // the vegetation height was not specified.
    model->pet_params.vegetation_height_m=0.5;  // use a reasonable assumed value
  }

  // use approximations from UN FAO: http://www.fao.org/3/X0490E/x0490e06.htm#aerodynamic%20resistance%20(ra)
  //model->pet_params.zero_plane_displacement_height_m=2.0/3.0*model->pet_params.vegetation_height_m; //wwu
  //model->pet_params.momentum_transfer_roughness_length_m=0.123*model->pet_params.vegetation_height_m; //wwu
  model->pet_params.momentum_transfer_roughness_length_m=0.1845*model->pet_params.zero_plane_displacement_height_m; //wwu
  model->pet_params.heat_transfer_roughness_length_m=0.1*model->pet_params.momentum_transfer_roughness_length_m;

  aerodynamic_resistance_s_per_m = calculate_aerodynamic_resistance(model);

  // all the ingredients have been prepared.  Make Penman-Monteith soufle...
  // from: http://www.fao.org/3/X0490E/x0490e06.htm#aerodynamic%20resistance%20(ra)

  // The radiation and aerodynamic terms can nearly cancel. Keep their native
  // rounded boundaries in default fast mode as well as explicit strict proof.
  pm_numerator = __dadd_rn(
      __dmul_rn(delta,__dsub_rn(model->pet_forcing.net_radiation_W_per_sq_m,
                              model->pet_forcing.ground_heat_flux_W_per_sq_m)),
      __dmul_rn(__dmul_rn(moist_air_density_kg_per_m3,CP),
                vapor_pressure_deficit_Pa)/aerodynamic_resistance_s_per_m);

  pm_denominator = __dadd_rn(delta,__dmul_rn(gamma,__dadd_rn(1.0,
      model->pet_forcing.canopy_resistance_sec_per_m/aerodynamic_resistance_s_per_m)));

  return(pm_numerator/pm_denominator);  // Latent heat flux in Watts per sq. m., or J per (s m2)
}

#endif // PET_PENMAN_MONTEITH_METHOD_H

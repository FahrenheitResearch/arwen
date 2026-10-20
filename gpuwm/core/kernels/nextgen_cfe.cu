// SPDX-License-Identifier: Apache-2.0
// CUDA adaptation of NOAA-OWP/cfe f9182dfb3d81c407c66c5b782114173893c30cf2.
// Modified 2026-10-17: per-catchment GPU execution, explicit resident matrix
// boundary, checked ragged offsets, native static Nash scratch replaced with
// an identical ordered per-thread recurrence. Native warning output is omitted;
// fatal input conditions are recorded in resident status. See NOTICE-CFE.txt.
// Original CFE: Fred L. Ogden, May 22-24, 2020, in service of NOAA-NWS OWP,
// Tuscaloosa, Alabama; state-space adaptation July 2021; later contributors'
// attributions and scientific references remain in the adapted source below.
// NVRTC supplies device math builtins; no host math.h dependency.
#include "glibc_flt64.cuh"
#define TRUE 1
#define FALSE 0
#define Schaake 1
#define Xinanjiang 2
#define GIUH 1
#define NASH_CASCADE 2
template<typename T> __device__ inline T cfe_min(T a,T b){return a<b?a:b;}
template<typename T> __device__ inline T cfe_max(T a,T b){return a>b?a:b;}
struct NWM_soil_parameters {

// using same variable names as used in NWM.  <sorry>
double smcmax;  // effective porosity [V/V]
double wltsmc;  // wilting point soil moisture content [V/V]
double satdk;   // saturated hydraulic conductivity [m s-1]
double satpsi;	// saturated capillary head [m]
double bb;      // beta exponent on Clapp-Hornberger (1978) soil water relations [-]
double mult;    // the multiplier applied to satdk to route water rapidly downslope
double slop;   // this factor (0-1) modifies the gradient of the hydraulic head at the soil bottom.  0=no-flow.
double D;       // soil depth [m]
double wilting_point_m;
// LKC: Add this two parameters since they belong to soils. Makes the parameter specification consistent
double alpha_fc;
double refkdt;
double soil_storage;

};
struct conceptual_reservoir {

  // this data structure describes a nonlinear reservoir having two outlets, one primary with an activation
  // threshold that may be zero, and a secondary outlet with a threshold that may be zero
  // this will also simulate a linear reservoir by setting the exponent parameter to 1.0 iff is_exponential==FALSE
  // iff is_exponential==TRUE, then it uses the exponential discharge function from the NWM V2.0 forumulation
  // as the primary discharge with a zero threshold, and does not calculate a secondary discharge.
  //--------------------------------------------------------------------------------------------------
  int    is_exponential;                // set this true TRUE to use the exponential form of the discharge equation
  double gw_storage;                    // Initial Storage - LKC: added since I need to keep track of it when changing parameters
  double storage_max_m;                 // maximum storage in this reservoir
  double storage_m;                     // state variable.
  double storage_change_m;              // storage change in the current step
  double coeff_primary;                 // the primary outlet
  double exponent_primary;
  double storage_threshold_primary_m;
  double storage_threshold_secondary_m;
  double coeff_secondary;
  double exponent_secondary;
  double ice_fraction_schaake;
  double ice_fraction_xinanjiang;
  int    is_sft_coupled;                // boolean - true if SFT is ON otherwise OFF (default is OFF)

  //---Root zone adjusted AET development -rlm -ajk -------------
  double *smc_profile;                  //soil moisture content profile
  int    n_soil_layers;                 // number of soil layers
  double *soil_layer_depths_m;          // soil layer depths defined in the config file in units of [m]
  int    is_aet_rootzone;               // boolean - true if aet_root_zone is ON otherwise OFF (default is OFF)
  int    max_rootzone_layer;            // maximum root zone layer is used to identify the maximum layer to remove water from for AET
  double *delta_soil_layer_depth_m;     // used to calculate the total soil moisture in each layer
  double soil_water_content_field_capacity;  // water content [m/m] at field capacity.  Used in AET routine

  //---------------------------------------------------------------

};
struct evapotranspiration_structure {

    double potential_et_m_per_s;
    double potential_et_m_per_timestep;
    double reduced_potential_et_m_per_timestep;
    double actual_et_from_rain_m_per_timestep;
    double actual_et_from_retention_depth_m_per_timestep;
    double actual_et_from_soil_m_per_timestep;
    double actual_et_m_per_timestep;

};
struct massbal {

    double volstart            ;
    double vol_runoff          ;
    double vol_infilt          ;
    double vol_runon_infilt    ;
    double vol_out_surface     ;
    double vol_end_surface     ;
    double vol_to_gw           ;
    double vol_in_gw_start     ;
    double vol_in_gw_end       ;
    double vol_from_gw         ;
    double vol_in_nash         ;
    double vol_in_nash_end     ;  // note the nash cascade is empty at start of simulation.
    double vol_out_nash        ;
    double vol_soil_start      ;
    double vol_to_soil         ;
    double vol_soil_to_lat_flow;
    double vol_soil_to_gw      ;  // this should equal vol_to_gw
    double vol_soil_end        ;
    double vol_et_from_soil    ;
    double vol_et_from_rain    ;
    double vol_et_to_atm       ;
    double volin               ;
    double volout              ;
    double volend              ;
    double vol_et_from_retention_depth;
    double cumulative_vol; //volstart + volin
    double volume_in_domain; // total volume in domain at end of time step
    double leakage; // Accumulate any known leakage or other losses from the domain, e.g. deep groundwater

};
struct infiltration_excess_parameters_structure {

    int surface_water_partitioning_scheme;
    double Schaake_adjusted_magic_constant_by_soil_type;
    double a_Xinanjiang_inflection_point_parameter;
    double b_Xinanjiang_shape_parameter;
    double x_Xinanjiang_shape_parameter;
    double urban_decimal_fraction;
    double ice_content_threshold; // ice content above which soil is impermeable

};
struct nash_cascade_parameters {

  int    N_nash;             // Number of Nash cascade reservoirs; [-]
  double K_nash;             // Fraction of storage per hour that moves from one reservoir to the next (time constant); [1/hour]
  int    nsubsteps;          // the number of substeps that each dt is divided into
  double *nash_storage;      // storage array nash cascade reservoirs; [m]
  double retention_depth;    /* parameter that represents retention depth process that is equivalent [m]
				to that used in WRF-Hydro */
  double runon_infiltration; // infiltration losses from surface runoff water to soil (or riparian groundwater) [m/hr]
  int    is_riparian_gw;     // flag to turn on/off riparian groundwater (currently used in LASAM only)
  double K_infiltration;     // Fraction of storage per hour that moves from reservoirs to soil (time constant); [1/hour]

};// function prototypes
// --------------------------------
__device__ inline void Schaake_partitioning_scheme(double dt, double field_capacity_m, double magic_number, double deficit, double qinsur,
                                        double smcmax, double soil_depth, double *runsrf, double *pddum, double ice_fraction_schaake,
                                        double ice_content_threshold);

// xinanjiang_dev: XinJiang function written by Rachel adapted by Jmframe and FLO,
__device__ inline void Xinanjiang_partitioning_scheme(double water_input_depth_m, double field_capacity_m,
                                           double max_soil_moisture_storage_m, double column_total_soil_water_m,
                                           struct infiltration_excess_parameters_structure *parms, double *infiltration_excess_m,
                                           double *infiltration_depth_m, double ice_fraction_xinanjiang);

__device__ inline void et_from_rainfall(double *timestep_rainfall_input_m, struct evapotranspiration_structure *et_struct);

__device__ inline void et_from_retention_depth(struct nash_cascade_parameters *nash_surface_params,
				    struct evapotranspiration_structure *et_struct);

__device__ inline void et_from_soil(struct conceptual_reservoir *soil_res, struct evapotranspiration_structure *et_struct,
			 struct NWM_soil_parameters *soil_parms);

__device__ inline int is_fabs_less_than_epsilon(double a,double epsilon);

__device__ inline void cfe(
        double *soil_reservoir_storage_deficit_m_ptr,
        struct NWM_soil_parameters NWM_soil_params_struct,
        struct conceptual_reservoir *soil_reservoir_struct,
        double timestep_h,
        /* xinanjiang_dev: since we are doing the option for Schaake and XinJiang,
                           instead of passing in the constants
                           pass in a structure with the constants for both subroutines.
        //double Schaake_adjusted_magic_constant_by_soil_type,*/
        struct infiltration_excess_parameters_structure infiltration_excess_params_struct,
        double timestep_rainfall_input_m,
        double *infiltration_excess_m_ptr,
        double *infiltration_depth_m_ptr,
        double *flux_perc_m_ptr,
        double *flux_lat_m_ptr,
        double *gw_reservoir_storage_deficit_m_ptr,
        struct conceptual_reservoir *gw_reservoir_struct,
        double *flux_from_deep_gw_to_chan_m_ptr,
        double *giuh_runoff_m_ptr,
        int num_giuh_ordinates,
        double *giuh_ordinates_arr,
        double *runoff_queue_m_per_timestep_arr,
        double *nash_lateral_runoff_m_ptr,
        int num_lateral_flow_nash_reservoirs,
        double K_nash,
        double *nash_storage_arr,
        struct nash_cascade_parameters *nash_surface_params,
        struct evapotranspiration_structure *evap_struct,
        double *Qout_m_ptr,
        struct massbal *massbal_struct,
        double time_step_size,
        int surface_runoff_scheme
    );


__device__ inline void conceptual_reservoir_flux_calc(struct conceptual_reservoir*,double*,double*);
__device__ inline double giuh_convolution_integral(double,int,double*,double*);
__device__ inline double nash_cascade_subsurface(double,int,double,double*);
__device__ inline double nash_cascade_surface(double,double,struct nash_cascade_parameters*);

enum CFEField {
 F_SOIL_RESERVOIR_STORAGE_DEFICIT_M = 0,
 F_TIMESTEP_RAINFALL_INPUT_M = 1,
 F_INFILTRATION_EXCESS_M = 2,
 F_INFILTRATION_DEPTH_M = 3,
 F_FLUX_PERC_M = 4,
 F_FLUX_LAT_M = 5,
 F_GW_RESERVOIR_STORAGE_DEFICIT_M = 6,
 F_FLUX_FROM_DEEP_GW_TO_CHAN_M = 7,
 F_FLUX_DIRECT_RUNOFF_M = 8,
 F_FLUX_NASH_LATERAL_RUNOFF_M = 9,
 F_FLUX_QOUT_M = 10,
 F_TIMESTEP_H = 11,
 F_TIME_STEP_SIZE = 12,
 F_K_NASH_SUBSURFACE = 13,
 F_NWM_SOIL_PARAMS_SMCMAX = 14,
 F_NWM_SOIL_PARAMS_WLTSMC = 15,
 F_NWM_SOIL_PARAMS_SATDK = 16,
 F_NWM_SOIL_PARAMS_SATPSI = 17,
 F_NWM_SOIL_PARAMS_BB = 18,
 F_NWM_SOIL_PARAMS_MULT = 19,
 F_NWM_SOIL_PARAMS_SLOP = 20,
 F_NWM_SOIL_PARAMS_D = 21,
 F_NWM_SOIL_PARAMS_WILTING_POINT_M = 22,
 F_NWM_SOIL_PARAMS_ALPHA_FC = 23,
 F_NWM_SOIL_PARAMS_REFKDT = 24,
 F_NWM_SOIL_PARAMS_SOIL_STORAGE = 25,
 F_SOIL_RESERVOIR_GW_STORAGE = 26,
 F_SOIL_RESERVOIR_STORAGE_MAX_M = 27,
 F_SOIL_RESERVOIR_STORAGE_M = 28,
 F_SOIL_RESERVOIR_STORAGE_CHANGE_M = 29,
 F_SOIL_RESERVOIR_COEFF_PRIMARY = 30,
 F_SOIL_RESERVOIR_EXPONENT_PRIMARY = 31,
 F_SOIL_RESERVOIR_STORAGE_THRESHOLD_PRIMARY_M = 32,
 F_SOIL_RESERVOIR_STORAGE_THRESHOLD_SECONDARY_M = 33,
 F_SOIL_RESERVOIR_COEFF_SECONDARY = 34,
 F_SOIL_RESERVOIR_EXPONENT_SECONDARY = 35,
 F_SOIL_RESERVOIR_ICE_FRACTION_SCHAAKE = 36,
 F_SOIL_RESERVOIR_ICE_FRACTION_XINANJIANG = 37,
 F_SOIL_RESERVOIR_SOIL_WATER_CONTENT_FIELD_CAPACITY = 38,
 F_GW_RESERVOIR_GW_STORAGE = 39,
 F_GW_RESERVOIR_STORAGE_MAX_M = 40,
 F_GW_RESERVOIR_STORAGE_M = 41,
 F_GW_RESERVOIR_STORAGE_CHANGE_M = 42,
 F_GW_RESERVOIR_COEFF_PRIMARY = 43,
 F_GW_RESERVOIR_EXPONENT_PRIMARY = 44,
 F_GW_RESERVOIR_STORAGE_THRESHOLD_PRIMARY_M = 45,
 F_GW_RESERVOIR_STORAGE_THRESHOLD_SECONDARY_M = 46,
 F_GW_RESERVOIR_COEFF_SECONDARY = 47,
 F_GW_RESERVOIR_EXPONENT_SECONDARY = 48,
 F_GW_RESERVOIR_ICE_FRACTION_SCHAAKE = 49,
 F_GW_RESERVOIR_ICE_FRACTION_XINANJIANG = 50,
 F_GW_RESERVOIR_SOIL_WATER_CONTENT_FIELD_CAPACITY = 51,
 F_ET_STRUCT_POTENTIAL_ET_M_PER_S = 52,
 F_ET_STRUCT_POTENTIAL_ET_M_PER_TIMESTEP = 53,
 F_ET_STRUCT_REDUCED_POTENTIAL_ET_M_PER_TIMESTEP = 54,
 F_ET_STRUCT_ACTUAL_ET_FROM_RAIN_M_PER_TIMESTEP = 55,
 F_ET_STRUCT_ACTUAL_ET_FROM_RETENTION_DEPTH_M_PER_TIMESTEP = 56,
 F_ET_STRUCT_ACTUAL_ET_FROM_SOIL_M_PER_TIMESTEP = 57,
 F_ET_STRUCT_ACTUAL_ET_M_PER_TIMESTEP = 58,
 F_VOL_STRUCT_VOLSTART = 59,
 F_VOL_STRUCT_VOL_RUNOFF = 60,
 F_VOL_STRUCT_VOL_INFILT = 61,
 F_VOL_STRUCT_VOL_RUNON_INFILT = 62,
 F_VOL_STRUCT_VOL_OUT_SURFACE = 63,
 F_VOL_STRUCT_VOL_END_SURFACE = 64,
 F_VOL_STRUCT_VOL_TO_GW = 65,
 F_VOL_STRUCT_VOL_IN_GW_START = 66,
 F_VOL_STRUCT_VOL_IN_GW_END = 67,
 F_VOL_STRUCT_VOL_FROM_GW = 68,
 F_VOL_STRUCT_VOL_IN_NASH = 69,
 F_VOL_STRUCT_VOL_IN_NASH_END = 70,
 F_VOL_STRUCT_VOL_OUT_NASH = 71,
 F_VOL_STRUCT_VOL_SOIL_START = 72,
 F_VOL_STRUCT_VOL_TO_SOIL = 73,
 F_VOL_STRUCT_VOL_SOIL_TO_LAT_FLOW = 74,
 F_VOL_STRUCT_VOL_SOIL_TO_GW = 75,
 F_VOL_STRUCT_VOL_SOIL_END = 76,
 F_VOL_STRUCT_VOL_ET_FROM_SOIL = 77,
 F_VOL_STRUCT_VOL_ET_FROM_RAIN = 78,
 F_VOL_STRUCT_VOL_ET_TO_ATM = 79,
 F_VOL_STRUCT_VOLIN = 80,
 F_VOL_STRUCT_VOLOUT = 81,
 F_VOL_STRUCT_VOLEND = 82,
 F_VOL_STRUCT_VOL_ET_FROM_RETENTION_DEPTH = 83,
 F_VOL_STRUCT_CUMULATIVE_VOL = 84,
 F_VOL_STRUCT_VOLUME_IN_DOMAIN = 85,
 F_VOL_STRUCT_LEAKAGE = 86,
 F_INFILTRATION_EXCESS_PARAMS_STRUCT_SCHAAKE_ADJUSTED_MAGIC_CONSTANT_BY_SOIL_TYPE = 87,
 F_INFILTRATION_EXCESS_PARAMS_STRUCT_A_XINANJIANG_INFLECTION_POINT_PARAMETER = 88,
 F_INFILTRATION_EXCESS_PARAMS_STRUCT_B_XINANJIANG_SHAPE_PARAMETER = 89,
 F_INFILTRATION_EXCESS_PARAMS_STRUCT_X_XINANJIANG_SHAPE_PARAMETER = 90,
 F_INFILTRATION_EXCESS_PARAMS_STRUCT_URBAN_DECIMAL_FRACTION = 91,
 F_INFILTRATION_EXCESS_PARAMS_STRUCT_ICE_CONTENT_THRESHOLD = 92,
 F_NASH_SURFACE_PARAMS_K_NASH = 93,
 F_NASH_SURFACE_PARAMS_RETENTION_DEPTH = 94,
 F_NASH_SURFACE_PARAMS_RUNON_INFILTRATION = 95,
 F_NASH_SURFACE_PARAMS_K_INFILTRATION = 96
};
enum CFEOption {
 I_NUM_GIUH_ORDINATES = 0,
 I_N_NASH_SUBSURFACE = 1,
 I_SURFACE_RUNOFF_SCHEME = 2,
 I_SOIL_RESERVOIR_IS_EXPONENTIAL = 3,
 I_SOIL_RESERVOIR_IS_SFT_COUPLED = 4,
 I_SOIL_RESERVOIR_N_SOIL_LAYERS = 5,
 I_SOIL_RESERVOIR_IS_AET_ROOTZONE = 6,
 I_SOIL_RESERVOIR_MAX_ROOTZONE_LAYER = 7,
 I_GW_RESERVOIR_IS_EXPONENTIAL = 8,
 I_GW_RESERVOIR_IS_SFT_COUPLED = 9,
 I_GW_RESERVOIR_N_SOIL_LAYERS = 10,
 I_GW_RESERVOIR_IS_AET_ROOTZONE = 11,
 I_GW_RESERVOIR_MAX_ROOTZONE_LAYER = 12,
 I_INFILTRATION_EXCESS_PARAMS_STRUCT_SURFACE_WATER_PARTITIONING_SCHEME = 13,
 I_NASH_SURFACE_PARAMS_N_NASH = 14,
 I_NASH_SURFACE_PARAMS_NSUBSTEPS = 15,
 I_NASH_SURFACE_PARAMS_IS_RIPARIAN_GW = 16,
 I_NASH_SURFACE_PARAMS_HAS_STORAGE = 17
};



//##############################################################
//########## SINGLE OUTLET EXPONENTIAL RESERVOIR ###############
//##########                -or-                 ###############
//##########    TWO OUTLET NONLINEAR RESERVOIR   ###############
//################################################################
// This function calculates the flux from a linear, or nonlinear
// conceptual reservoir with one or two outlets, or from an
// exponential nonlinear conceptual reservoir with only one outlet.
// In the non-exponential instance, each outlet can have its own
// activation storage threshold.  Flow from the second outlet is
// turned off by setting the discharge coeff. to 0.0.
//################################################################
__device__ inline void conceptual_reservoir_flux_calc(struct conceptual_reservoir *da_reservoir,
                                           double *primary_flux_m,double *secondary_flux_m)
{
  //struct conceptual_reservoir  <<<<INCLUDED HERE FOR REFERENCE.>>>>
  //{
  // int    is_exponential;  // set this true TRUE to use the exponential form of the discharge equation
  // double storage_max_m;
  // double storage_m;
  // double coeff_primary;
  // double exponent_secondary;
  // double storage_threshold_primary_m;
  // double storage_threshold_secondary_m;
  // double coeff_secondary;
  // double exponent_secondary;
  // };
  // THIS FUNCTION CALCULATES THE FLUXES FROM A CONCEPTUAL NON-LINEAR (OR LINEAR) RESERVOIR WITH TWO OUTLETS
  // all fluxes calculated by this routine are instantaneous with units of the coefficient.

  //local variables
  double storage_above_threshold_m;

  // *****************************************************************************
  // ------------------ Conceptual Ground Water Reservoir -------------------------
  // single outlet reservoir like the NWM V1.2 exponential conceptual gw reservoir
  if (da_reservoir->is_exponential == TRUE) {
    // calculate the one flux and return.
    // NVRTC libdevice exp differs by one ULP on the captured wet-step argument.
    // Use ArWen's existing MIT Arm adaptation of the native libm arithmetic.
    double exp_term = glibc_exp(da_reservoir->exponent_primary * da_reservoir->storage_m / da_reservoir->storage_max_m);

    *primary_flux_m = da_reservoir->coeff_primary * (exp_term - 1.0);
    *secondary_flux_m = 0.0;

    return;
  }
  // *****************************************************************************

  // code goes past here iff it is not a single outlet exponential deep groundwater reservoir of the NWM variety
  // The vertical outlet is assumed to be primary and satisfied first.

  *primary_flux_m=0.0;
  storage_above_threshold_m = da_reservoir->storage_m - da_reservoir->storage_threshold_primary_m;

  if (storage_above_threshold_m > 0.0) {
    // flow is possible from the primary outlet
    *primary_flux_m = da_reservoir->coeff_primary *
      pow(storage_above_threshold_m / (da_reservoir->storage_max_m - da_reservoir->storage_threshold_primary_m),
	  da_reservoir->exponent_primary);

    if(*primary_flux_m > storage_above_threshold_m)
      *primary_flux_m = storage_above_threshold_m;  // limit to max. available
  }

  *secondary_flux_m=0.0;
  storage_above_threshold_m = da_reservoir->storage_m - da_reservoir->storage_threshold_secondary_m;

  if (storage_above_threshold_m > 0.0) {
    // flow is possible from the secondary outlet
    *secondary_flux_m = da_reservoir->coeff_secondary *
      pow(storage_above_threshold_m / (da_reservoir->storage_max_m - da_reservoir->storage_threshold_secondary_m),
	  da_reservoir->exponent_secondary);

    if (*secondary_flux_m > (storage_above_threshold_m-(*primary_flux_m)))
      *secondary_flux_m = storage_above_threshold_m-(*primary_flux_m);  // limit to max. available
  }
  return;
}






//##############################################################
//############### GIUH CONVOLUTION INTEGRAL   ##################
//##############################################################
__device__ inline double giuh_convolution_integral(double runoff_m,int num_giuh_ordinates,
					double *giuh_ordinates, double *runoff_queue_m_per_timestep)
{
  //##############################################################
  // This function solves the convolution integral involving N
  //  GIUH ordinates.
  //##############################################################
  double runoff_m_current_timestep;
  int N,i;

  N = num_giuh_ordinates;
  runoff_queue_m_per_timestep[N] = 0.0;

  for(i=0;i<N;i++)
    {
      runoff_queue_m_per_timestep[i] += giuh_ordinates[i]*runoff_m;
    }

  runoff_m_current_timestep = runoff_queue_m_per_timestep[0];

  for(i=1;i<=N;i++)  // shift all the entries in preperation for the next timestep
    {
      runoff_queue_m_per_timestep[i-1] = runoff_queue_m_per_timestep[i];
    }
  //runoff_queue_m_per_timestep[N-1]=0.0;

  return runoff_m_current_timestep;
}





//##############################################################
//###############   NASH CASCADE SUBSURFACE  ###################
//##############################################################
__device__ inline double nash_cascade_subsurface(double flux_lat_m,int N_nash,
			       double K_nash,double *nash_storage_arr)
{
  //##############################################################
  // Solve for the flow through the Nash cascade to delay the
  // arrival of the lateral flow into the channel
  //##############################################################
  // local vars
  double outflow_m;
  double previous_outflow = 0.0;

  //Loop through reservoirs
  for(int i = 0; i < N_nash; i++)
    {
      double current_outflow = K_nash*nash_storage_arr[i];
      nash_storage_arr[i]  -= current_outflow;

      if (i==0) nash_storage_arr[i] += flux_lat_m;
      else      nash_storage_arr[i] += previous_outflow;
      previous_outflow = current_outflow;
    }

  /*  Get Qout */
  outflow_m = previous_outflow;

  //Return the flow output
  return (outflow_m);

}


//##############################################################
//#################   NASH CASCADE SURFACE #### ################
//##############################################################
__device__ inline double nash_cascade_surface(double runoff_m, double soil_storage_deficit_m,
			    struct nash_cascade_parameters *nash_params)
{
  //##############################################################
  // Solve for the flow through the Nash cascade to delay the
  // arrival of the lateral flow into the channel
  //##############################################################

  int nsubsteps  = nash_params->nsubsteps;
  int N_nash     = nash_params->N_nash;
  double K_nash  = nash_params->K_nash;
  double K_infil = nash_params->K_infiltration;
  double retention_depth = nash_params->retention_depth;

  // local vars
  double dt_h  = 1.0;             // model timestep [hour]
  double subdt = dt_h/nsubsteps;
  double S     = 0.0;
  double dS    = 0.0;            // change in reservoir storage
  double dS_infil = 0.0;         // change in reservoir storage due to infiltration
  double Q_r;                    // discharge from reservoir
  double Q_out = 0.0;            // discharge at the outlet (the last reservoir) per subtimestep
  double Q_infil;                // discharge from reservoirs to soil
  double Q_to_channel_m = 0.0;   // total outflow to channel per timestep
  double Q_to_soil_m    = 0.0;   // runon infiltration (losses from surface runoff to soil)
  double soil_deficit_m = soil_storage_deficit_m; // local variable to track the soil storage deficit
  double infil_m = 0.0;

  nash_params->nash_storage[0] += runoff_m;

  // Loop through number of sub-timesteps
  for (int ts = 0; ts < nsubsteps; ts++) {

    //Loop through reservoirs
    for(int i = 0; i < N_nash; i++) {

      // if storage of ith reservoir is zero, move to the next reservoir
      if (nash_params->nash_storage[i] == 0.0)
	continue;

      // First: infiltration capacity should be satisfied before routing water through Nash reservoirs

      // compute runon infiltration
      Q_infil = K_infil * nash_params->nash_storage[i];
      infil_m = Q_infil * subdt;

      // case 1: soil_deficit greater than infil_m, all water can infiltrate, and soil_deficit decreases
      // case 2: soil_deficit less than infil_m, portion of water can infiltrate, soil_deficit gets zero
      // case 3: soil_deficit is zero, no infiltration

      // determine the right amount of infiltrated water based on soil deficit
      if (soil_deficit_m > 0.0 && soil_deficit_m > infil_m) {
	soil_deficit_m -= infil_m;           // update soil deficit for the next subtimstep iteration
      }
      else if (soil_deficit_m > 0.0) {
	infil_m = soil_deficit_m;
	soil_deficit_m = 0.0;
      }
      else {
	infil_m = 0.0; // if got here, then soil is fully saturated, so no infiltration
      }

      // update nash storage (subtract infiltrated water)
      if  (nash_params->nash_storage[i] >= infil_m)
        nash_params->nash_storage[i] = nash_params->nash_storage[i] - infil_m;
      else {
        infil_m = nash_params->nash_storage[i];
        nash_params->nash_storage[i] = 0.0;
      }


      Q_to_soil_m += infil_m;  // water volume that infiltrates per subtimestep from each reservoir

      /*=========================================================================================*/
      // Second: time to route water through Nash reservoirs
      S = nash_params->nash_storage[i];

      // determine the amount of surface water available for routing from the first reservoir
      if (i == 0 &&  S > retention_depth)
        S -= retention_depth;
      else if (i == 0)
        S = 0.0;

      Q_r = K_nash * S;                    // flow from reservoir i to i+1
      dS  = fmin(Q_r * subdt, S);          // storage change in reservoir i per subtimestep
      nash_params->nash_storage[i] -= dS;  // updated storage in reservoir i

      if(i < (N_nash-1))
        nash_params->nash_storage[i+1] += dS;
      else
        Q_out = Q_r;

    }

   Q_to_channel_m += Q_out * subdt;  // Q_r at the end of N_nash loop is the discharge at the outlet

  }

  nash_params->runon_infiltration = Q_to_soil_m;

  // Return the flow output
  return (Q_to_channel_m);

}




// CFE STATE SPACE FUNCTION // #######################################################################
// Adapted version of Conceptual Functional Equivalent model re-written in state-space form July, 2021
//####################################################################################################
__device__ inline void cfe(
        double *soil_reservoir_storage_deficit_m_ptr,
        struct NWM_soil_parameters NWM_soil_params_struct,
        struct conceptual_reservoir *soil_reservoir_struct,
        double timestep_h,
        struct infiltration_excess_parameters_structure infiltration_excess_params_struct,
        double timestep_rainfall_input_m,
        double *infiltration_excess_m_ptr,
        double *infiltration_depth_m_ptr,
        double *flux_perc_m_ptr,
        double *flux_lat_m_ptr,
        double *gw_reservoir_storage_deficit_m_ptr,
        struct conceptual_reservoir *gw_reservoir_struct,
        double *flux_from_deep_gw_to_chan_m_ptr,
        double *direct_runoff_m_ptr,
        int    num_giuh_ordinates,
        double *giuh_ordinates_arr,
        double *runoff_queue_m_per_timestep_arr,
        double *nash_lateral_runoff_m_ptr,
        int    N_nash_subsurface,
        double K_nash_subsurface,
        double *nash_storage_arr,
        struct nash_cascade_parameters *nash_surface_params,
        struct evapotranspiration_structure *evap_struct,
        double *Qout_m_ptr,
        struct massbal *massbal_struct,
        double time_step_size,
        int    surface_runoff_scheme
    ){                      // #######################################################################
// CFE STATE SPACE FUNCTION // #######################################################################

// ####    COPY THE MODEL FUNCTION STATE SPACE TO LOCAL VARIABLES    ####
// ####    Reason: so we don't have to re-write domain science code to de-reference a whole bunch of pointers
// ####        Note: all of thes variables are storages in [m] or fluxes in [m/timestep]
    double soil_reservoir_storage_deficit_m = *soil_reservoir_storage_deficit_m_ptr;   // storage [m]

    double infiltration_excess_m          = *infiltration_excess_m_ptr;                    // Schaake/Xinanjiang partitioned excess water this timestep [m]*/

    double infiltration_depth_m             = *infiltration_depth_m_ptr;               // Schaake partitioned infiltration this timestep [m]
    double flux_perc_m                      = *flux_perc_m_ptr;                        // water moved from soil reservoir to gw reservoir this timestep [m]
    double flux_lat_m                       = *flux_lat_m_ptr;                         // water moved from soil reservoir to lateral flow Nash cascad this timestep [m]
    double gw_reservoir_storage_deficit_m   = *gw_reservoir_storage_deficit_m_ptr;     // deficit in gw reservoir storage [m]
    double flux_from_deep_gw_to_chan_m      = *flux_from_deep_gw_to_chan_m_ptr;        // water moved from gw reservoir to catchment outlet nexus this timestep [m]
    double direct_runoff_m                 = *direct_runoff_m_ptr;                   // water leaving GIUH or NASH cascade reservoir to outlet this timestep [m]
    double nash_lateral_runoff_m            = *nash_lateral_runoff_m_ptr;              // water leaving lateral subsurface flow Nash cascade this timestep [m]
    double Qout_m                           = *Qout_m_ptr;                             // the total runoff this timestep (Surface+Nash+GW) [m]

    // LOCAL VARIABLES, the values of which are not important to describe the model state.  They are like notes on scrap paper.

    double diff=0.0;
    double primary_flux=0.0;      // pointers to these variables passed to conceptual nonlinear reservoir which has two outlets, primary & secondary
    double secondary_flux=0.0;    // pointers to these variables passed to conceptual nonlinear reservoir which has two outlets, primary & secondary
    double lateral_flux=0.0;      // flux from soil to lateral flow Nash cascade +to cascade  [m/timestep]

    double percolation_flux=0.0;  // flux from soil to gw nonlinear researvoir, +downward  [m/timestep]

    // store current soil storage_m in a temp. variable
    double storage_temp_m = soil_reservoir_struct->storage_m;

  //##################################################
  // partition rainfall using Schaake function
  //##################################################
  massbal_struct->volin += timestep_rainfall_input_m;
  massbal_struct->cumulative_vol += timestep_rainfall_input_m;

  evap_struct->potential_et_m_per_timestep = evap_struct->potential_et_m_per_s * time_step_size;
  evap_struct->reduced_potential_et_m_per_timestep = evap_struct->potential_et_m_per_s * time_step_size;

  evap_struct->actual_et_from_rain_m_per_timestep = 0;
  if (timestep_rainfall_input_m > 0) {
    et_from_rainfall(&timestep_rainfall_input_m,evap_struct);
  }

  massbal_struct->vol_et_from_rain = massbal_struct->vol_et_from_rain + evap_struct->actual_et_from_rain_m_per_timestep;
  massbal_struct->vol_et_to_atm    = massbal_struct->vol_et_to_atm + evap_struct->actual_et_from_rain_m_per_timestep;
  massbal_struct->volout           = massbal_struct->volout + evap_struct->actual_et_from_rain_m_per_timestep;

  // AET from surface retention depth
  evap_struct->actual_et_from_retention_depth_m_per_timestep = 0;
  // NJF Need some way to know if nash surface is used nash_storage
  // is properly allocated, otherwise this can segfault!
  if (nash_surface_params->nash_storage != NULL &&
      nash_surface_params->nash_storage[0] > 0.0 &&
      evap_struct->reduced_potential_et_m_per_timestep > 0.0) {
    et_from_retention_depth(nash_surface_params, evap_struct);
  }

  massbal_struct->vol_et_from_retention_depth += evap_struct->actual_et_from_retention_depth_m_per_timestep;
  massbal_struct->vol_et_to_atm               += evap_struct->actual_et_from_retention_depth_m_per_timestep;
  massbal_struct->volout                      += evap_struct->actual_et_from_retention_depth_m_per_timestep;
  massbal_struct->vol_out_surface             += evap_struct->actual_et_from_retention_depth_m_per_timestep;

  // LKC: Change this. Now evaporation happens before runoff calculation. This was creating issues since it modifies
  // storage_m and not storage_deficit
  evap_struct->actual_et_from_soil_m_per_timestep = 0;
  if(soil_reservoir_struct->storage_m > NWM_soil_params_struct.wilting_point_m) {
    et_from_soil(soil_reservoir_struct, evap_struct, &NWM_soil_params_struct);
  }

  massbal_struct->vol_et_from_soil = massbal_struct->vol_et_from_soil + evap_struct->actual_et_from_soil_m_per_timestep;
  massbal_struct->vol_et_to_atm    = massbal_struct->vol_et_to_atm + evap_struct->actual_et_from_soil_m_per_timestep;
  massbal_struct->volout           = massbal_struct->volout + evap_struct->actual_et_from_soil_m_per_timestep;

  evap_struct->actual_et_m_per_timestep = evap_struct->actual_et_from_rain_m_per_timestep +
                                          evap_struct->actual_et_from_retention_depth_m_per_timestep +
                                          evap_struct->actual_et_from_soil_m_per_timestep;

  // LKC: This needs to be calcualted here after et_from_soil since soil_reservoir_struct->storage_m changes
  soil_reservoir_storage_deficit_m=(NWM_soil_params_struct.smcmax*NWM_soil_params_struct.D-soil_reservoir_struct->storage_m);


  // NEW FLO
  if(0.0 < timestep_rainfall_input_m)
    {
    if (infiltration_excess_params_struct.surface_water_partitioning_scheme == Schaake)
      {
	// to ensure that ice_fraction_schaake is set to 0.0 for uncoupled SFT
	if(!soil_reservoir_struct->is_sft_coupled)
	  soil_reservoir_struct->ice_fraction_schaake = 0.0;

	Schaake_partitioning_scheme(timestep_h, soil_reservoir_struct->storage_threshold_primary_m,
				    infiltration_excess_params_struct.Schaake_adjusted_magic_constant_by_soil_type,
				    soil_reservoir_storage_deficit_m, timestep_rainfall_input_m, NWM_soil_params_struct.smcmax,
				    NWM_soil_params_struct.D, &infiltration_excess_m, &infiltration_depth_m,
				    soil_reservoir_struct->ice_fraction_schaake, infiltration_excess_params_struct.ice_content_threshold);
      }
    else if (infiltration_excess_params_struct.surface_water_partitioning_scheme == Xinanjiang)
      {

	// to ensure that ice_fraction_xinan is set to 0.0 for uncoupled SFT
	if(!soil_reservoir_struct->is_sft_coupled)
	  soil_reservoir_struct->ice_fraction_xinanjiang = 0.0;

	Xinanjiang_partitioning_scheme(timestep_rainfall_input_m, soil_reservoir_struct->storage_threshold_primary_m,
				       soil_reservoir_struct->storage_max_m, soil_reservoir_struct->storage_m,
				       &infiltration_excess_params_struct, &infiltration_excess_m, &infiltration_depth_m,
				       soil_reservoir_struct->ice_fraction_xinanjiang);
      }
    else
      {
      return; // Unreachable after resident option validation; native exits here.   // note -1 is arbitrary   #############BOMB################ NEW FLO
      }
    }
  else  // NEW FLO no need to call Schaake or Xinanjiang if it's not raining.
    {
    infiltration_excess_m = 0.0;
    infiltration_depth_m = 0.0;
    }


  // check to make sure that there is storage available in soil to hold the water that does not runoff
  //--------------------------------------------------------------------------------------------------
  if(soil_reservoir_storage_deficit_m<infiltration_depth_m)
    {
    infiltration_excess_m+=(infiltration_depth_m-soil_reservoir_storage_deficit_m);  // put infiltration that won't fit back into runoff
    infiltration_depth_m=soil_reservoir_storage_deficit_m;
    soil_reservoir_struct->storage_m=soil_reservoir_struct->storage_max_m;
    // LKC: should the deficit of soil be set to zero here - if the condition is reached, soil is full
    soil_reservoir_storage_deficit_m = 0;
    }





  massbal_struct->vol_runoff   += infiltration_excess_m;       // edit FLO
  massbal_struct->vol_infilt   += infiltration_depth_m;  // edit FLO

  // put infiltration flux into soil conceptual reservoir.  If not enough room
  // limit amount transferred to deficit
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  if(flux_perc_m>soil_reservoir_storage_deficit_m)
    {
    diff=flux_perc_m-soil_reservoir_storage_deficit_m;  // the amount that there is not capacity ffor
    infiltration_depth_m=soil_reservoir_storage_deficit_m;
    massbal_struct->vol_runoff+=diff;  // send excess water back to GIUH runoff  edit FLO
    massbal_struct->vol_infilt-=diff;  // correct overprediction of infilt.      edit FLO
    infiltration_excess_m+=diff; // bug found by Nels.  This was missing and fixes it.

    // LKC: again here
    soil_reservoir_storage_deficit_m = 0;
    }

  massbal_struct->vol_to_soil              += infiltration_depth_m;
  soil_reservoir_struct->storage_m += infiltration_depth_m;  // put the infiltrated water in the soil.


  // calculate fluxes from the soil storage into the deep groundwater (percolation) and to lateral subsurface flow
  //--------------------------------------------------------------------------------------------------------------

  conceptual_reservoir_flux_calc(soil_reservoir_struct,&percolation_flux,&lateral_flux);
  flux_perc_m=percolation_flux;  // m/h   <<<<<<<<<<<  flux of percolation from soil to g.w. reservoir >>>>>>>>>

  flux_lat_m=lateral_flux;  // m/h        <<<<<<<<<<<  flux into the lateral flow Nash cascade >>>>>>>>


  // calculate flux of base flow from deep groundwater reservoir to channel
  //--------------------------------------------------------------------------
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  gw_reservoir_storage_deficit_m= gw_reservoir_struct->storage_max_m-gw_reservoir_struct->storage_m;
  // limit amount transferred to deficit iff there is insuffienct avail. storage
  if(flux_perc_m>gw_reservoir_storage_deficit_m) {
    diff=flux_perc_m-gw_reservoir_storage_deficit_m;
    flux_perc_m=gw_reservoir_storage_deficit_m;
    massbal_struct->vol_runoff+=diff;  // send excess water back to GIUH runoff
    massbal_struct->vol_infilt-=diff;  // correct overprediction of infilt.
  }


  massbal_struct->vol_to_gw                +=flux_perc_m;
  massbal_struct->vol_soil_to_gw           +=flux_perc_m;

  gw_reservoir_struct->storage_m   += flux_perc_m;
  soil_reservoir_struct->storage_m -= flux_perc_m;
  soil_reservoir_struct->storage_m -= flux_lat_m;
  massbal_struct->vol_soil_to_lat_flow     += flux_lat_m;  //TODO add this to nash cascade as input
  massbal_struct->volout=massbal_struct->volout+flux_lat_m;

  // get change in storage (used in the soil moisture coupler)
  soil_reservoir_struct->storage_change_m = soil_reservoir_struct->storage_m - storage_temp_m ;

  conceptual_reservoir_flux_calc(gw_reservoir_struct,&primary_flux,&secondary_flux);


  flux_from_deep_gw_to_chan_m=primary_flux;  // m/h   <<<<<<<<<< BASE FLOW FLUX >>>>>>>>>
  if(flux_from_deep_gw_to_chan_m >  gw_reservoir_struct->storage_m)  {
    flux_from_deep_gw_to_chan_m=gw_reservoir_struct->storage_m;
    // TODO: set a flag when flux larger than storage
    // Native warning only; same water-limited flux is retained.
  }

  massbal_struct->vol_from_gw+=flux_from_deep_gw_to_chan_m;

  // in the instance of calling the gw reservoir the secondary flux should be zero- verify
  // Native diagnostic only; secondary flux remains unchanged.


  // adjust state of deep groundwater conceptual nonlinear reservoir
  //-----------------------------------------------------------------

  gw_reservoir_struct->storage_m -= flux_from_deep_gw_to_chan_m;

  // Solve the convolution integral ffor this time step

  if (surface_runoff_scheme == GIUH)
    direct_runoff_m = giuh_convolution_integral(infiltration_excess_m,num_giuh_ordinates,
                                                giuh_ordinates_arr,runoff_queue_m_per_timestep_arr);
  else if (surface_runoff_scheme == NASH_CASCADE)
  {
    direct_runoff_m =  nash_cascade_surface(infiltration_excess_m, soil_reservoir_storage_deficit_m,
					    nash_surface_params);
    soil_reservoir_struct->storage_m += nash_surface_params->runon_infiltration;  // put the runon infiltrated water in the soil.
    soil_reservoir_storage_deficit_m -= nash_surface_params->runon_infiltration; // update soil deficit

    massbal_struct->vol_runon_infilt += nash_surface_params->runon_infiltration;
  }

  massbal_struct->vol_out_surface  += direct_runoff_m;
  massbal_struct->volout           += direct_runoff_m;
  massbal_struct->volout           += flux_from_deep_gw_to_chan_m;

  // Route lateral flow through the Nash cascade.
  nash_lateral_runoff_m = nash_cascade_subsurface(flux_lat_m, N_nash_subsurface,
						  K_nash_subsurface, nash_storage_arr);
  massbal_struct->vol_in_nash   += flux_lat_m;
  massbal_struct->vol_out_nash  += nash_lateral_runoff_m;

  // Track the total volume held in the domain at the end of the time step
  int i = 0;
  //Reservoir storages
  massbal_struct->volume_in_domain = soil_reservoir_struct->storage_m+gw_reservoir_struct->storage_m;
  //Surface storage
  if (surface_runoff_scheme == GIUH) {
      for(i=0;i<num_giuh_ordinates;i++){
	      massbal_struct->volume_in_domain += runoff_queue_m_per_timestep_arr[i];
    }
  }
  if (surface_runoff_scheme == NASH_CASCADE){
    for(i=0;i<nash_surface_params->N_nash; i++){
      massbal_struct->volume_in_domain += nash_surface_params->nash_storage[i];
    }
  }
  //Subsurface storage
  for(i=0;i<N_nash_subsurface; i++){
    massbal_struct->volume_in_domain += nash_storage_arr[i];
  }



  Qout_m = direct_runoff_m + nash_lateral_runoff_m + flux_from_deep_gw_to_chan_m;

    // #### COPY BACK STATE VALUES BY POINTER REFERENCE SO VISIBLE TO FRAMEWORK    ####
    *soil_reservoir_storage_deficit_m_ptr = soil_reservoir_storage_deficit_m;
    *infiltration_excess_m_ptr            = infiltration_excess_m;
    *infiltration_depth_m_ptr             = infiltration_depth_m;
    *flux_perc_m_ptr                      = flux_perc_m;
    *flux_lat_m_ptr                       = flux_lat_m;
    *gw_reservoir_storage_deficit_m_ptr   = gw_reservoir_storage_deficit_m;
    *flux_from_deep_gw_to_chan_m_ptr      = flux_from_deep_gw_to_chan_m;
    *direct_runoff_m_ptr                  = direct_runoff_m;
    *nash_lateral_runoff_m_ptr            = nash_lateral_runoff_m;
    *Qout_m_ptr                           = Qout_m;



} // END CFE STATE SPACE FUNCTIONS
  //####################################################################################################
  //####################################################################################################


//##############################################################
//#########   SCHAAKE RUNOFF PARTITIONING SCHEME   #############
//##############################################################
__device__ inline void Schaake_partitioning_scheme(double timestep_h, double field_capacity_m, double Schaake_adjusted_magic_constant_by_soil_type,
                                 double column_total_soil_moisture_deficit_m, double water_input_depth_m, double smcmax,
                                 double soil_depth, double *infiltration_excess_m, double *infiltration_depth_m,
                                 double ice_fraction_schaake, double ice_content_threshold)
{


/*! ===============================================================================
  This subtroutine takes water_input_depth_m and partitions it into infiltration_excess_m and
  infiltration_depth_m using the scheme from Schaake et al. 1996.
! --------------------------------------------------------------------------------
! ! modified by FLO April 2020 to eliminate reference to ice processes,
! ! and to de-obfuscate and use descriptive and dimensionally consistent variable names.
! --------------------------------------------------------------------------------
    IMPLICIT NONE
! --------------------------------------------------------------------------------
! inputs
  double timestep_h
  double Schaake_adjusted_magic_constant_by_soil_type = C*Ks(soiltype)/Ks_ref, where C=3, and Ks_ref=2.0E-06 m/s
  double column_total_soil_moisture_deficit_m
  double water_input_depth_m  amount of water input to soil surface this time step [m]

! outputs
  double infiltration_excess_m      amount of water partitioned to surface water this time step [m]


--------------------------------------------------------------------------------*/

  // Validated by the resident kernel before calling CFE.

  double timestep_d,Schaake_parenthetical_term,Ic,Px;

  if(0.0 < water_input_depth_m) {
    if (0.0 > column_total_soil_moisture_deficit_m) {
      *infiltration_excess_m=water_input_depth_m;
      *infiltration_depth_m=0.0;
    }
    else {
      // partition time-step total applied water as per Schaake et al. 1996.
                                         // change from dt in [s] to dt1 in [d] because kdt has units of [d^(-1)]
      timestep_d = timestep_h /24.0;    // timestep_d is the time step in days.

      // calculate the parenthetical part of Eqn. 34 from Schaake et al. Note the magic constant has units of [d^(-1)]

      Schaake_parenthetical_term = (1.0 - exp ( - Schaake_adjusted_magic_constant_by_soil_type * timestep_d));

      // From Schaake et al. Eqn. 2., using the column total moisture deficit
      // BUT the way it is used here, it is the cumulative soil moisture deficit in the entire soil profile.
      // "Layer" info not used in this subroutine in noah-mp, except to sum up the total soil moisture storage.
      // NOTE: when column_total_soil_moisture_deficit_m becomes zero, which occurs when the soil column is saturated,
      // then Ic=0, where Ic in the Schaake paper is called the "spatially averaged infiltration capacity",
      // and is defined in Eqn. 12.

      Ic = column_total_soil_moisture_deficit_m * Schaake_parenthetical_term;

      Px=water_input_depth_m;   // Total water input to partitioning scheme this time step [m]

      // This is eqn 24 from Schaake et al.  NOTE: this is 0 in the case of a saturated soil column, when Ic=0.
      // Physically happens only if soil has no-flow lower b.c.

      *infiltration_depth_m = (Px * (Ic / (Px + Ic)));


      if( 0.0 < (water_input_depth_m-(*infiltration_depth_m)) ) {
        *infiltration_excess_m = water_input_depth_m - (*infiltration_depth_m);
      }
      else  *infiltration_excess_m=0.0;
      *infiltration_depth_m =  water_input_depth_m - (*infiltration_excess_m);
    }
  }
  else {
    *infiltration_excess_m = 0.0;
    *infiltration_depth_m = 0.0;
  }


  // impermeable fraction due to frozen soil
  double factor = 1.0;
  //field_capacity_m = SMCREF in NOAH_MP
  // ice_content_threshold = frzk in NOAH_MP
  //double frzk = 0.15; // Ice content above which soil is impermeable

  if (ice_fraction_schaake > 1.0E-2) {
    int cv_frz = 3; // should this be moved to config file as well, probably not.
    double field_capacity = field_capacity_m/soil_depth; // divide by the reservior depth to get SMCREF [m/m] (unitless)
    double frz_fact = smcmax/field_capacity * (0.412 / 0.468);
    double frzx = ice_content_threshold * frz_fact;

    double acrt = cv_frz * frzx / ice_fraction_schaake;
    double sum1 =1;

    for (int i1=1; i1 < cv_frz; i1++) {
      int k = 1;
      for (int i2=i1+1; i2<cv_frz; i2++)
        k *= i2;

      sum1 += pow(acrt,(cv_frz - i1)) / (double)k;
    }

   factor = 1. - exp(-acrt) * sum1;
 }

 *infiltration_depth_m = factor * (*infiltration_depth_m);

 *infiltration_excess_m = water_input_depth_m - (*infiltration_depth_m);

 return;
}


//##############################################################
//########   XINANJIANG RUNOFF PARTITIONING SCHEME   ###########
//##############################################################

__device__ inline void Xinanjiang_partitioning_scheme(double water_input_depth_m, double field_capacity_m,
                                    double max_soil_moisture_storage_m, double column_total_soil_water_m,
                                    struct infiltration_excess_parameters_structure *parms,
                                    double *infiltration_excess_m, double *infiltration_depth_m,
                                    double ice_fraction_xinanjiang)
{
  //------------------------------------------------------------------------
  //  This module takes the water_input_depth_m and separates it into infiltration_excess_m
  //  and infiltration_depth_m by calculating the saturated area and runoff based on a scheme developed
  //  for the Xinanjiang model by Jaywardena and Zhou (2000). According to Knoben et al.
  //  (2019) "the model uses a variable contributing area to simulate runoff.  [It] uses
  //  a double parabolic curve to simulate tension water capacities within the catchment,
  //  instead of the original single parabolic curve" which is also used as the standard
  //  VIC fomulation.  This runoff scheme was selected for implementation into NWM v3.0.
  //  REFERENCES:
  //  1. Jaywardena, A.W. and M.C. Zhou, 2000. A modified spatial soil moisture storage
  //     capacity distribution curve for the Xinanjiang model. Journal of Hydrology 227: 93-113
  //  2. Knoben, W.J.M. et al., 2019. Supplement of Modular Assessment of Rainfall-Runoff Models
  //     Toolbox (MARRMoT) v1.2: an open-source, extendable framework providing implementations
  //     of 46 conceptual hydrologic models as continuous state-space formulations. Supplement of
  //     Geosci. Model Dev. 12: 2463-2480.
  //-------------------------------------------------------------------------
  //  Written by RLM May 2021
  //  Adapted by JMFrame September 2021 for new version of CFE
  //  Reviewed by FLO Feb. 2025, compared against Jayawardena & Zhou (2000) and NWM 3.0 code.  Fixed bug and refactored.
  //        Compared against NWM 3.0 and found identical performance within approx. 1e-05 on calculated outputs
  //-------------------------------------------------------------------------
  // Inputs
  //   double  water_input_depth_m           amount of water input to soil surface this time step [m]
  //   double  field_capacity_m              amount of water stored in soil reservoir when at field capacity [m]
  //   double  max_soil_moisture_storage_m   total storage of the soil moisture reservoir (porosity*soil thickness) [m]
  //   double  column_total_soil_water_m     current storage of the soil moisture reservoir [m]
  //   double  a_inflection_point_parameter  a parameter
  //   double  b_shape_parameter             b parameter
  //   double  x_shape_parameter             x parameter
  //   double  urban_decimal_fraction        fraction of land cover in the modeled area that is classified as urban [unitless decimal]
  //   double  ice_fraction_xinanjiang       fraction of top soil layer that is frozen [unitless decimal]
  //
  // Outputs
  //   double  infiltration_excess_m         amount of water partitioned to surface water this time step [m]
  //   double  infiltration_depth_m          amount of water partitioned as infiltration (soil water input) this time step [m]
  //-------------------------------------------------------------------------

  // local variables
  double tension_water_m, free_water_m, max_tension_water_m, max_free_water_m, pervious_runoff_m,water_input_pervious_fraction_m;
  double impervious_fraction, impervious_runoff_m;
  double f_over_F;  // notation from Jayawardena and Zhou (2000) see Fig 2.

  // first thing, check whether we can just rreturn without calculating anything to save compute
  //----------------------                  NWM variable name
  *infiltration_excess_m   = 0.0;          // RUNSRF
  *infiltration_depth_m    = 0.0;          // PDDUM

  if(water_input_depth_m < 1.0e-08)   {  // zero or really close to zero so calculations not needed
      return;
  }

  // calculations required

  // initialize                          // NWM variable name
  tension_water_m               = 0.0;   // WM
  max_tension_water_m           = 0.0;   // WM_MAX
  free_water_m                  = 0.0;   // SM
  max_free_water_m              = 0.0;   // SM_MAX
  impervious_runoff_m           = 0.0;   // IRUNOFF


  // Partition the total soil water in the column between free water and tension water assuming that
  // total pore space in the soil Vtot, given by (porosity * soil_thickness) also equals max_tension_water_m + max_free_water_m.
  //
  // Water input to soil fills up the tension water storage first.  Once it is full, any additiona
  // input of water goes into free water storage.  This means that iff there is _any_ free water then
  // the tension water storage is full.

  if( column_total_soil_water_m - field_capacity_m  > 0.0)  { // soil moisture greater than field capacity
      free_water_m    = column_total_soil_water_m - field_capacity_m;
      tension_water_m = field_capacity_m;
  } else {
      tension_water_m = column_total_soil_water_m;
  }
  max_tension_water_m = field_capacity_m;
  max_free_water_m = max_soil_moisture_storage_m - field_capacity_m;

  if(tension_water_m >  max_tension_water_m) tension_water_m = max_tension_water_m; // as done in NWM - could cause massbal err.
  if(free_water_m    >  max_free_water_m)    free_water_m    = max_free_water_m;    // as done in NWM - could cause massbal err.

  // Calculate the impervious runoff (see eq. 309 from Knoben et al).
  // estimate the fraction of the modeled area that is impervious (impervious_fraction) based on
  // urban classification (hard coded 95% [0.95] impervious) and frozen soils (passed to cfe
  // from freeze-thaw model) using a weighted average.

  impervious_fraction = (parms->urban_decimal_fraction * 0.95) + ((1.0 - parms->urban_decimal_fraction) * ice_fraction_xinanjiang);
  impervious_runoff_m = impervious_fraction * water_input_depth_m;

  water_input_pervious_fraction_m = water_input_depth_m - impervious_runoff_m;  // from here on just deal with
                                                                                // water that enters the soil
  //edited by RLM; added logic block to handle what happens when porosity or field capacity = 0
  //FLO changed from 0.95 to 1.0 because what happens to the other 5% ?
  if (max_free_water_m <= 0.0 || max_tension_water_m <=0.0) {
      *infiltration_excess_m = water_input_pervious_fraction_m;

      *infiltration_excess_m += impervious_runoff_m;             // must add impervious runoff back in
      *infiltration_depth_m = 0.0;                               //added by FLO
      return;          // this is an unusual situation and should only happen when one or both parameters are zero
  }

  // if code gets to here, then field capacity and soil porosity were both nonzero, so there is some space in
  // the soil ffor more water, AND impervious runoff has already been abstracted from precipitation as impervious_runoff_m.

  // solve pervious surface runoff (m) based on Eq. 310
  // Use notation from Knoben et al. (2019) p. 71

  double a    = parms->a_Xinanjiang_inflection_point_parameter;   // contributing area curve inflection point
  double b    = parms->b_Xinanjiang_shape_parameter;              // contributing area curve shape parameter
  double Ex   = parms->x_Xinanjiang_shape_parameter;   // contributing area curve shape parameter ffor direct runoff
  double W    = tension_water_m;
  double Wmax = max_tension_water_m;

  if ( ( W / Wmax ) <= (0.5 - a)) {
  // THIS LINE IN ORIGINAL CFE XINANJIANG CODE IS A BUG
  //  f_over_F = (pow((0.5 - a),(1.0 - b)) * pow((1.0 - (tension_water_m/max_tension_water_m)) , b));

  // FLO correct comparing against Eqn. 2a in Jayawardena & Zhou (2000)
      f_over_F = (pow((0.5 - a),(1.0 - b)) * pow( W / Wmax , b));                     // Eqn. 2a in Jayawardena & Zhue

  } else {  // if ( (0.5-a) < W / Wmax )
      f_over_F = 1.0 - pow((0.5 + a), (1.0 - b)) * pow((1.0 - W / Wmax) , b); // Eqn. 2b in Jayawardena & Zhue
  }

  double R = water_input_pervious_fraction_m * f_over_F; // water moved from tension to free water storage

  double S = free_water_m;
  double Smax = max_free_water_m;

  *infiltration_excess_m = R * (1.0 - pow((1.0 - (S/Smax)), Ex)) + impervious_runoff_m;

  // Separate the infiltration from the total water input depth to the soil surface.

  *infiltration_depth_m = water_input_pervious_fraction_m - *infiltration_excess_m;


    return;
}



//##############################################################
//####################   ET FROM RAINFALL   ####################
//##############################################################
__device__ inline void et_from_rainfall(double *timestep_rainfall_input_m, struct evapotranspiration_structure *et_struct)
{
    /*
        iff it is raining, take PET from rainfall first.  Wet veg. is efficient evaporator.
    */

    if (*timestep_rainfall_input_m >0.0) {

        if (*timestep_rainfall_input_m > et_struct->potential_et_m_per_timestep) {

            et_struct->actual_et_from_rain_m_per_timestep = et_struct->potential_et_m_per_timestep;
            *timestep_rainfall_input_m -= et_struct->actual_et_from_rain_m_per_timestep;
        }

        else {
            // LKC: This was incorrectly set to potential instead of actual
	    et_struct->actual_et_from_rain_m_per_timestep = *timestep_rainfall_input_m;
            *timestep_rainfall_input_m = 0.0;

        }
    // Move this out of the loop since EVPT needs to be corrected wheter R > EVPT or not
    et_struct->reduced_potential_et_m_per_timestep = et_struct->potential_et_m_per_timestep-et_struct->actual_et_from_rain_m_per_timestep;
    }
}

//##############################################################
//###########   ET FROM SURFACE RETENTION DEPTH   ##############
//##############################################################
__device__ inline void et_from_retention_depth(struct nash_cascade_parameters *nash_surface_params, struct evapotranspiration_structure *et_struct)
{
  // NJF try not to use nash_storage if it hasn't been initialized...
  if(nash_surface_params->nash_storage == NULL) return;

  if (et_struct->reduced_potential_et_m_per_timestep >= nash_surface_params->nash_storage[0]) {
    et_struct->actual_et_from_retention_depth_m_per_timestep = nash_surface_params->nash_storage[0];
    nash_surface_params->nash_storage[0] = 0.0;
  }
  else {
    et_struct->actual_et_from_retention_depth_m_per_timestep = et_struct->reduced_potential_et_m_per_timestep;
    //et_struct->reduced_potential_et_m_per_timestep = 0.0;
    nash_surface_params->nash_storage[0] -= et_struct->actual_et_from_retention_depth_m_per_timestep;
  }

  et_struct->reduced_potential_et_m_per_timestep -= et_struct->actual_et_from_retention_depth_m_per_timestep;

}

//##############################################################
//####################   ET FROM SOIL   ########################
//##############################################################
__device__ inline void et_from_soil(struct conceptual_reservoir *soil_res, struct evapotranspiration_structure *et_struct, struct NWM_soil_parameters *soil_parms)
{
    /*
        take AET from soil moisture storage,
        using Budyko type function to limit PET if wilting<soilmoist<field_capacity
    */
    double Budyko_numerator;
    double Budyko_denominator;
    double Budyko;

/*-------------------- Root zone adjusted AET development -rlm --------------------------*/

    et_struct->actual_et_from_soil_m_per_timestep = 0;

    // if rootzone-based AET turned ON
    if (soil_res->is_aet_rootzone) {
      // Assuming the layer with the most moisture is the bottom layer of the root zone (max_rootzone_layer)
      // Convert volumetric soil moisture from the max root zone layer to moisture content [m] (layer_storage_m)
      double layer_storage_m = soil_res->smc_profile[soil_res->max_rootzone_layer] *
                               soil_res->delta_soil_layer_depth_m[soil_res->max_rootzone_layer];

      // If the moisture content from the layer with the most moisture is less than the
      // wilting point, actual et from this timestep is 0 and no water is removed.
      if ( soil_res->smc_profile[soil_res->max_rootzone_layer] <= soil_parms->wltsmc ) {
        et_struct->actual_et_from_soil_m_per_timestep = 0;
      }
      // calculate the amount of moisture removed by evapotranspiration for the bottom layer of the root zone
      else if (soil_res->smc_profile[soil_res->max_rootzone_layer] >= soil_res->soil_water_content_field_capacity) {
        et_struct->actual_et_from_soil_m_per_timestep = cfe_min(et_struct->reduced_potential_et_m_per_timestep, layer_storage_m);
      }
      else {
        Budyko_numerator = soil_res->smc_profile[soil_res->max_rootzone_layer] - soil_parms->wltsmc;
        Budyko_denominator = soil_res->soil_water_content_field_capacity - soil_parms->wltsmc;
        Budyko = Budyko_numerator / Budyko_denominator;

        et_struct->actual_et_from_soil_m_per_timestep = cfe_min(Budyko * et_struct->reduced_potential_et_m_per_timestep, layer_storage_m);
      }

      // Reduce remaining PET and remove moisture from soil profile equal to the calculated AET (actual_et_from_soil_m_per_timestep
      et_struct->reduced_potential_et_m_per_timestep -= et_struct->actual_et_from_soil_m_per_timestep;
      soil_res->smc_profile[soil_res->max_rootzone_layer] -= (et_struct->actual_et_from_soil_m_per_timestep /
							       soil_res->delta_soil_layer_depth_m[soil_res->max_rootzone_layer]);
      soil_res->storage_m -= et_struct->actual_et_from_soil_m_per_timestep;
    }

    else if (et_struct->reduced_potential_et_m_per_timestep > 0) {

        if (soil_res->storage_m >= soil_res->storage_threshold_primary_m) {
            et_struct->actual_et_from_soil_m_per_timestep = cfe_min(et_struct->reduced_potential_et_m_per_timestep, soil_res->storage_m);
        }
        else if (soil_res->storage_m > soil_parms->wilting_point_m && soil_res->storage_m < soil_res->storage_threshold_primary_m) {

            Budyko_numerator = soil_res->storage_m - soil_parms->wilting_point_m;
            Budyko_denominator = soil_res->storage_threshold_primary_m - soil_parms->wilting_point_m;
            Budyko = Budyko_numerator / Budyko_denominator;
            // LKC: Include check to guarantee EAT is not larger than soil storage
            et_struct->actual_et_from_soil_m_per_timestep = cfe_min(Budyko * et_struct->reduced_potential_et_m_per_timestep, soil_res->storage_m);

        }

        soil_res->storage_m -= et_struct->actual_et_from_soil_m_per_timestep;
        et_struct->reduced_potential_et_m_per_timestep = et_struct->reduced_potential_et_m_per_timestep - et_struct->actual_et_from_soil_m_per_timestep;
    }


}

__device__ inline int is_fabs_less_than_epsilon(double a,double epsilon)  // returns true if fabs(a)<epsilon
{
  if(fabs(a)<epsilon) return(TRUE);
  else                return(FALSE);
}

__device__ inline bool cfe_bounds(int offset,int count,int capacity){
 return offset>=0 && count>=0 && offset<=capacity && count<=capacity-offset;
}
extern "C" __global__ void nextgen_cfe_step(
 double *fields, const int *options, const int *offsets,
 double *giuh_ordinates, double *runoff_queue, double *nash_subsurface,
 double *nash_surface, double *smc_profile, double *soil_layer_depths,
 double *delta_soil_layer_depths, int *status, int n,
 int giuh_size, int queue_size, int lateral_size, int surface_size, int soil_size,
 const void *rainrate, const double *potential_et, int rain_dtype,
 int *current_step, const double *time_fraction, double *ponded_depth){
 const int p=blockIdx.x*blockDim.x+threadIdx.x;
 if(p>=n)return;
 #define F(name) fields[(long long)F_##name*n+p]
 #define I(name) options[(long long)I_##name*n+p]
 const int go=offsets[p], qo=offsets[n+p], lo=offsets[2*n+p];
 const int so=offsets[3*n+p], po=offsets[4*n+p];
 const int ng=I(NUM_GIUH_ORDINATES), nl=I(N_NASH_SUBSURFACE);
 const int ns=I(NASH_SURFACE_PARAMS_HAS_STORAGE) ? I(NASH_SURFACE_PARAMS_N_NASH) : 0, nz=I(SOIL_RESERVOIR_N_SOIL_LAYERS);
 if(ng<0 || nl<1 || ns<0 || nz<0 || ng==2147483647 ||
    !cfe_bounds(go,ng,giuh_size) || !cfe_bounds(qo,ng+1,queue_size) ||
    !cfe_bounds(lo,nl,lateral_size) || !cfe_bounds(so,ns,surface_size) ||
    !cfe_bounds(po,nz,soil_size)){
  if(status[p]==0)status[p]=1;return;
 }
 if((I(SURFACE_RUNOFF_SCHEME)!=GIUH && I(SURFACE_RUNOFF_SCHEME)!=NASH_CASCADE) ||
    (I(INFILTRATION_EXCESS_PARAMS_STRUCT_SURFACE_WATER_PARTITIONING_SCHEME)!=Schaake &&
     I(INFILTRATION_EXCESS_PARAMS_STRUCT_SURFACE_WATER_PARTITIONING_SCHEME)!=Xinanjiang) ||
    (I(SURFACE_RUNOFF_SCHEME)==GIUH && ng<1) ||
    (I(SURFACE_RUNOFF_SCHEME)==NASH_CASCADE && (ns<1 || I(NASH_SURFACE_PARAMS_NSUBSTEPS)<1)) ||
    (I(NASH_SURFACE_PARAMS_HAS_STORAGE)!=0 && I(NASH_SURFACE_PARAMS_HAS_STORAGE)!=1)){
  if(status[p]==0)status[p]=2;return;
 }
 if(I(SOIL_RESERVOIR_IS_AET_ROOTZONE) && (nz<1 ||
    I(SOIL_RESERVOIR_MAX_ROOTZONE_LAYER)<0 || I(SOIL_RESERVOIR_MAX_ROOTZONE_LAYER)>=nz)){
  if(status[p]==0)status[p]=3;return;
 }
 if(I(SOIL_RESERVOIR_IS_SFT_COUPLED) &&
    I(INFILTRATION_EXCESS_PARAMS_STRUCT_SURFACE_WATER_PARTITIONING_SCHEME)==Schaake &&
    !(F(SOIL_RESERVOIR_ICE_FRACTION_SCHAAKE)>=0.0)){
  if(status[p]==0)status[p]=4;return;
 }
 if(rain_dtype){
  const double rate=(rain_dtype&1) ? (double)((const float*)rainrate)[p]
                                 : ((const double*)rainrate)[p];
  // 1/2: canonical mm/s; 3/4: m/s -> native UDUNITS mm/h;
  // 5/6: already-native mm/h; 7/8: explicitly selected native mm/s -> mm/h.
  // Keep conversion before BMI /1000 and fraction.
  // Native UDUNITS 2.2.28 cv_convert_double uses this measured scale. The
  // exact mathematical 3600000.0 differs on all 15 nonzero framework links.
  const double precipitation_kg_per_m2=rain_dtype<=2 ? rate*F(TIME_STEP_SIZE)
       : rain_dtype<=4 ? rate*3600000.0000000005
       : rain_dtype<=6 ? rate : rate*3600.0000000000005;
  F(TIMESTEP_RAINFALL_INPUT_M)=(precipitation_kg_per_m2/1000.0)*time_fraction[p];
  F(ET_STRUCT_POTENTIAL_ET_M_PER_S)=potential_et[p];
 }
 struct NWM_soil_parameters NWM_soil_params = {};
 NWM_soil_params.smcmax = F(NWM_SOIL_PARAMS_SMCMAX);
 NWM_soil_params.wltsmc = F(NWM_SOIL_PARAMS_WLTSMC);
 NWM_soil_params.satdk = F(NWM_SOIL_PARAMS_SATDK);
 NWM_soil_params.satpsi = F(NWM_SOIL_PARAMS_SATPSI);
 NWM_soil_params.bb = F(NWM_SOIL_PARAMS_BB);
 NWM_soil_params.mult = F(NWM_SOIL_PARAMS_MULT);
 NWM_soil_params.slop = F(NWM_SOIL_PARAMS_SLOP);
 NWM_soil_params.D = F(NWM_SOIL_PARAMS_D);
 NWM_soil_params.wilting_point_m = F(NWM_SOIL_PARAMS_WILTING_POINT_M);
 NWM_soil_params.alpha_fc = F(NWM_SOIL_PARAMS_ALPHA_FC);
 NWM_soil_params.refkdt = F(NWM_SOIL_PARAMS_REFKDT);
 NWM_soil_params.soil_storage = F(NWM_SOIL_PARAMS_SOIL_STORAGE);
 struct conceptual_reservoir soil_reservoir = {};
 soil_reservoir.is_exponential = I(SOIL_RESERVOIR_IS_EXPONENTIAL);
 soil_reservoir.gw_storage = F(SOIL_RESERVOIR_GW_STORAGE);
 soil_reservoir.storage_max_m = F(SOIL_RESERVOIR_STORAGE_MAX_M);
 soil_reservoir.storage_m = F(SOIL_RESERVOIR_STORAGE_M);
 soil_reservoir.storage_change_m = F(SOIL_RESERVOIR_STORAGE_CHANGE_M);
 soil_reservoir.coeff_primary = F(SOIL_RESERVOIR_COEFF_PRIMARY);
 soil_reservoir.exponent_primary = F(SOIL_RESERVOIR_EXPONENT_PRIMARY);
 soil_reservoir.storage_threshold_primary_m = F(SOIL_RESERVOIR_STORAGE_THRESHOLD_PRIMARY_M);
 soil_reservoir.storage_threshold_secondary_m = F(SOIL_RESERVOIR_STORAGE_THRESHOLD_SECONDARY_M);
 soil_reservoir.coeff_secondary = F(SOIL_RESERVOIR_COEFF_SECONDARY);
 soil_reservoir.exponent_secondary = F(SOIL_RESERVOIR_EXPONENT_SECONDARY);
 soil_reservoir.ice_fraction_schaake = F(SOIL_RESERVOIR_ICE_FRACTION_SCHAAKE);
 soil_reservoir.ice_fraction_xinanjiang = F(SOIL_RESERVOIR_ICE_FRACTION_XINANJIANG);
 soil_reservoir.is_sft_coupled = I(SOIL_RESERVOIR_IS_SFT_COUPLED);
 soil_reservoir.n_soil_layers = I(SOIL_RESERVOIR_N_SOIL_LAYERS);
 soil_reservoir.is_aet_rootzone = I(SOIL_RESERVOIR_IS_AET_ROOTZONE);
 soil_reservoir.max_rootzone_layer = I(SOIL_RESERVOIR_MAX_ROOTZONE_LAYER);
 soil_reservoir.soil_water_content_field_capacity = F(SOIL_RESERVOIR_SOIL_WATER_CONTENT_FIELD_CAPACITY);
 struct conceptual_reservoir gw_reservoir = {};
 gw_reservoir.is_exponential = I(GW_RESERVOIR_IS_EXPONENTIAL);
 gw_reservoir.gw_storage = F(GW_RESERVOIR_GW_STORAGE);
 gw_reservoir.storage_max_m = F(GW_RESERVOIR_STORAGE_MAX_M);
 gw_reservoir.storage_m = F(GW_RESERVOIR_STORAGE_M);
 gw_reservoir.storage_change_m = F(GW_RESERVOIR_STORAGE_CHANGE_M);
 gw_reservoir.coeff_primary = F(GW_RESERVOIR_COEFF_PRIMARY);
 gw_reservoir.exponent_primary = F(GW_RESERVOIR_EXPONENT_PRIMARY);
 gw_reservoir.storage_threshold_primary_m = F(GW_RESERVOIR_STORAGE_THRESHOLD_PRIMARY_M);
 gw_reservoir.storage_threshold_secondary_m = F(GW_RESERVOIR_STORAGE_THRESHOLD_SECONDARY_M);
 gw_reservoir.coeff_secondary = F(GW_RESERVOIR_COEFF_SECONDARY);
 gw_reservoir.exponent_secondary = F(GW_RESERVOIR_EXPONENT_SECONDARY);
 gw_reservoir.ice_fraction_schaake = F(GW_RESERVOIR_ICE_FRACTION_SCHAAKE);
 gw_reservoir.ice_fraction_xinanjiang = F(GW_RESERVOIR_ICE_FRACTION_XINANJIANG);
 gw_reservoir.is_sft_coupled = I(GW_RESERVOIR_IS_SFT_COUPLED);
 gw_reservoir.n_soil_layers = I(GW_RESERVOIR_N_SOIL_LAYERS);
 gw_reservoir.is_aet_rootzone = I(GW_RESERVOIR_IS_AET_ROOTZONE);
 gw_reservoir.max_rootzone_layer = I(GW_RESERVOIR_MAX_ROOTZONE_LAYER);
 gw_reservoir.soil_water_content_field_capacity = F(GW_RESERVOIR_SOIL_WATER_CONTENT_FIELD_CAPACITY);
 struct evapotranspiration_structure et_struct = {};
 et_struct.potential_et_m_per_s = F(ET_STRUCT_POTENTIAL_ET_M_PER_S);
 et_struct.potential_et_m_per_timestep = F(ET_STRUCT_POTENTIAL_ET_M_PER_TIMESTEP);
 et_struct.reduced_potential_et_m_per_timestep = F(ET_STRUCT_REDUCED_POTENTIAL_ET_M_PER_TIMESTEP);
 et_struct.actual_et_from_rain_m_per_timestep = F(ET_STRUCT_ACTUAL_ET_FROM_RAIN_M_PER_TIMESTEP);
 et_struct.actual_et_from_retention_depth_m_per_timestep = F(ET_STRUCT_ACTUAL_ET_FROM_RETENTION_DEPTH_M_PER_TIMESTEP);
 et_struct.actual_et_from_soil_m_per_timestep = F(ET_STRUCT_ACTUAL_ET_FROM_SOIL_M_PER_TIMESTEP);
 et_struct.actual_et_m_per_timestep = F(ET_STRUCT_ACTUAL_ET_M_PER_TIMESTEP);
 struct massbal vol_struct = {};
 vol_struct.volstart = F(VOL_STRUCT_VOLSTART);
 vol_struct.vol_runoff = F(VOL_STRUCT_VOL_RUNOFF);
 vol_struct.vol_infilt = F(VOL_STRUCT_VOL_INFILT);
 vol_struct.vol_runon_infilt = F(VOL_STRUCT_VOL_RUNON_INFILT);
 vol_struct.vol_out_surface = F(VOL_STRUCT_VOL_OUT_SURFACE);
 vol_struct.vol_end_surface = F(VOL_STRUCT_VOL_END_SURFACE);
 vol_struct.vol_to_gw = F(VOL_STRUCT_VOL_TO_GW);
 vol_struct.vol_in_gw_start = F(VOL_STRUCT_VOL_IN_GW_START);
 vol_struct.vol_in_gw_end = F(VOL_STRUCT_VOL_IN_GW_END);
 vol_struct.vol_from_gw = F(VOL_STRUCT_VOL_FROM_GW);
 vol_struct.vol_in_nash = F(VOL_STRUCT_VOL_IN_NASH);
 vol_struct.vol_in_nash_end = F(VOL_STRUCT_VOL_IN_NASH_END);
 vol_struct.vol_out_nash = F(VOL_STRUCT_VOL_OUT_NASH);
 vol_struct.vol_soil_start = F(VOL_STRUCT_VOL_SOIL_START);
 vol_struct.vol_to_soil = F(VOL_STRUCT_VOL_TO_SOIL);
 vol_struct.vol_soil_to_lat_flow = F(VOL_STRUCT_VOL_SOIL_TO_LAT_FLOW);
 vol_struct.vol_soil_to_gw = F(VOL_STRUCT_VOL_SOIL_TO_GW);
 vol_struct.vol_soil_end = F(VOL_STRUCT_VOL_SOIL_END);
 vol_struct.vol_et_from_soil = F(VOL_STRUCT_VOL_ET_FROM_SOIL);
 vol_struct.vol_et_from_rain = F(VOL_STRUCT_VOL_ET_FROM_RAIN);
 vol_struct.vol_et_to_atm = F(VOL_STRUCT_VOL_ET_TO_ATM);
 vol_struct.volin = F(VOL_STRUCT_VOLIN);
 vol_struct.volout = F(VOL_STRUCT_VOLOUT);
 vol_struct.volend = F(VOL_STRUCT_VOLEND);
 vol_struct.vol_et_from_retention_depth = F(VOL_STRUCT_VOL_ET_FROM_RETENTION_DEPTH);
 vol_struct.cumulative_vol = F(VOL_STRUCT_CUMULATIVE_VOL);
 vol_struct.volume_in_domain = F(VOL_STRUCT_VOLUME_IN_DOMAIN);
 vol_struct.leakage = F(VOL_STRUCT_LEAKAGE);
 struct infiltration_excess_parameters_structure infiltration_excess_params_struct = {};
 infiltration_excess_params_struct.surface_water_partitioning_scheme = I(INFILTRATION_EXCESS_PARAMS_STRUCT_SURFACE_WATER_PARTITIONING_SCHEME);
 infiltration_excess_params_struct.Schaake_adjusted_magic_constant_by_soil_type = F(INFILTRATION_EXCESS_PARAMS_STRUCT_SCHAAKE_ADJUSTED_MAGIC_CONSTANT_BY_SOIL_TYPE);
 infiltration_excess_params_struct.a_Xinanjiang_inflection_point_parameter = F(INFILTRATION_EXCESS_PARAMS_STRUCT_A_XINANJIANG_INFLECTION_POINT_PARAMETER);
 infiltration_excess_params_struct.b_Xinanjiang_shape_parameter = F(INFILTRATION_EXCESS_PARAMS_STRUCT_B_XINANJIANG_SHAPE_PARAMETER);
 infiltration_excess_params_struct.x_Xinanjiang_shape_parameter = F(INFILTRATION_EXCESS_PARAMS_STRUCT_X_XINANJIANG_SHAPE_PARAMETER);
 infiltration_excess_params_struct.urban_decimal_fraction = F(INFILTRATION_EXCESS_PARAMS_STRUCT_URBAN_DECIMAL_FRACTION);
 infiltration_excess_params_struct.ice_content_threshold = F(INFILTRATION_EXCESS_PARAMS_STRUCT_ICE_CONTENT_THRESHOLD);
 struct nash_cascade_parameters nash_surface_params = {};
 nash_surface_params.N_nash = I(NASH_SURFACE_PARAMS_N_NASH);
 nash_surface_params.K_nash = F(NASH_SURFACE_PARAMS_K_NASH);
 nash_surface_params.nsubsteps = I(NASH_SURFACE_PARAMS_NSUBSTEPS);
 nash_surface_params.retention_depth = F(NASH_SURFACE_PARAMS_RETENTION_DEPTH);
 nash_surface_params.runon_infiltration = F(NASH_SURFACE_PARAMS_RUNON_INFILTRATION);
 nash_surface_params.is_riparian_gw = I(NASH_SURFACE_PARAMS_IS_RIPARIAN_GW);
 nash_surface_params.K_infiltration = F(NASH_SURFACE_PARAMS_K_INFILTRATION);
 soil_reservoir.smc_profile=smc_profile+po;
 soil_reservoir.soil_layer_depths_m=soil_layer_depths+po;
 soil_reservoir.delta_soil_layer_depth_m=delta_soil_layer_depths+po;
 nash_surface_params.nash_storage=ns ? nash_surface+so : nullptr;
 if(soil_reservoir.is_aet_rootzone &&
    !(soil_reservoir.delta_soil_layer_depth_m[soil_reservoir.max_rootzone_layer]>0.0)){
  if(status[p]==0)status[p]=5;return;
 }
 cfe(&F(SOIL_RESERVOIR_STORAGE_DEFICIT_M),NWM_soil_params,&soil_reservoir,
     F(TIMESTEP_H),infiltration_excess_params_struct,F(TIMESTEP_RAINFALL_INPUT_M),
     &F(INFILTRATION_EXCESS_M),&F(INFILTRATION_DEPTH_M),&F(FLUX_PERC_M),&F(FLUX_LAT_M),
     &F(GW_RESERVOIR_STORAGE_DEFICIT_M),&gw_reservoir,&F(FLUX_FROM_DEEP_GW_TO_CHAN_M),
     &F(FLUX_DIRECT_RUNOFF_M),ng,giuh_ordinates+go,runoff_queue+qo,
     &F(FLUX_NASH_LATERAL_RUNOFF_M),nl,F(K_NASH_SUBSURFACE),nash_subsurface+lo,
     &nash_surface_params,&et_struct,&F(FLUX_QOUT_M),&vol_struct,F(TIME_STEP_SIZE),
     I(SURFACE_RUNOFF_SCHEME));
 F(NWM_SOIL_PARAMS_SMCMAX) = NWM_soil_params.smcmax;
 F(NWM_SOIL_PARAMS_WLTSMC) = NWM_soil_params.wltsmc;
 F(NWM_SOIL_PARAMS_SATDK) = NWM_soil_params.satdk;
 F(NWM_SOIL_PARAMS_SATPSI) = NWM_soil_params.satpsi;
 F(NWM_SOIL_PARAMS_BB) = NWM_soil_params.bb;
 F(NWM_SOIL_PARAMS_MULT) = NWM_soil_params.mult;
 F(NWM_SOIL_PARAMS_SLOP) = NWM_soil_params.slop;
 F(NWM_SOIL_PARAMS_D) = NWM_soil_params.D;
 F(NWM_SOIL_PARAMS_WILTING_POINT_M) = NWM_soil_params.wilting_point_m;
 F(NWM_SOIL_PARAMS_ALPHA_FC) = NWM_soil_params.alpha_fc;
 F(NWM_SOIL_PARAMS_REFKDT) = NWM_soil_params.refkdt;
 F(NWM_SOIL_PARAMS_SOIL_STORAGE) = NWM_soil_params.soil_storage;
 F(SOIL_RESERVOIR_GW_STORAGE) = soil_reservoir.gw_storage;
 F(SOIL_RESERVOIR_STORAGE_MAX_M) = soil_reservoir.storage_max_m;
 F(SOIL_RESERVOIR_STORAGE_M) = soil_reservoir.storage_m;
 F(SOIL_RESERVOIR_STORAGE_CHANGE_M) = soil_reservoir.storage_change_m;
 F(SOIL_RESERVOIR_COEFF_PRIMARY) = soil_reservoir.coeff_primary;
 F(SOIL_RESERVOIR_EXPONENT_PRIMARY) = soil_reservoir.exponent_primary;
 F(SOIL_RESERVOIR_STORAGE_THRESHOLD_PRIMARY_M) = soil_reservoir.storage_threshold_primary_m;
 F(SOIL_RESERVOIR_STORAGE_THRESHOLD_SECONDARY_M) = soil_reservoir.storage_threshold_secondary_m;
 F(SOIL_RESERVOIR_COEFF_SECONDARY) = soil_reservoir.coeff_secondary;
 F(SOIL_RESERVOIR_EXPONENT_SECONDARY) = soil_reservoir.exponent_secondary;
 F(SOIL_RESERVOIR_ICE_FRACTION_SCHAAKE) = soil_reservoir.ice_fraction_schaake;
 F(SOIL_RESERVOIR_ICE_FRACTION_XINANJIANG) = soil_reservoir.ice_fraction_xinanjiang;
 F(SOIL_RESERVOIR_SOIL_WATER_CONTENT_FIELD_CAPACITY) = soil_reservoir.soil_water_content_field_capacity;
 F(GW_RESERVOIR_GW_STORAGE) = gw_reservoir.gw_storage;
 F(GW_RESERVOIR_STORAGE_MAX_M) = gw_reservoir.storage_max_m;
 F(GW_RESERVOIR_STORAGE_M) = gw_reservoir.storage_m;
 F(GW_RESERVOIR_STORAGE_CHANGE_M) = gw_reservoir.storage_change_m;
 F(GW_RESERVOIR_COEFF_PRIMARY) = gw_reservoir.coeff_primary;
 F(GW_RESERVOIR_EXPONENT_PRIMARY) = gw_reservoir.exponent_primary;
 F(GW_RESERVOIR_STORAGE_THRESHOLD_PRIMARY_M) = gw_reservoir.storage_threshold_primary_m;
 F(GW_RESERVOIR_STORAGE_THRESHOLD_SECONDARY_M) = gw_reservoir.storage_threshold_secondary_m;
 F(GW_RESERVOIR_COEFF_SECONDARY) = gw_reservoir.coeff_secondary;
 F(GW_RESERVOIR_EXPONENT_SECONDARY) = gw_reservoir.exponent_secondary;
 F(GW_RESERVOIR_ICE_FRACTION_SCHAAKE) = gw_reservoir.ice_fraction_schaake;
 F(GW_RESERVOIR_ICE_FRACTION_XINANJIANG) = gw_reservoir.ice_fraction_xinanjiang;
 F(GW_RESERVOIR_SOIL_WATER_CONTENT_FIELD_CAPACITY) = gw_reservoir.soil_water_content_field_capacity;
 F(ET_STRUCT_POTENTIAL_ET_M_PER_S) = et_struct.potential_et_m_per_s;
 F(ET_STRUCT_POTENTIAL_ET_M_PER_TIMESTEP) = et_struct.potential_et_m_per_timestep;
 F(ET_STRUCT_REDUCED_POTENTIAL_ET_M_PER_TIMESTEP) = et_struct.reduced_potential_et_m_per_timestep;
 F(ET_STRUCT_ACTUAL_ET_FROM_RAIN_M_PER_TIMESTEP) = et_struct.actual_et_from_rain_m_per_timestep;
 F(ET_STRUCT_ACTUAL_ET_FROM_RETENTION_DEPTH_M_PER_TIMESTEP) = et_struct.actual_et_from_retention_depth_m_per_timestep;
 F(ET_STRUCT_ACTUAL_ET_FROM_SOIL_M_PER_TIMESTEP) = et_struct.actual_et_from_soil_m_per_timestep;
 F(ET_STRUCT_ACTUAL_ET_M_PER_TIMESTEP) = et_struct.actual_et_m_per_timestep;
 F(VOL_STRUCT_VOLSTART) = vol_struct.volstart;
 F(VOL_STRUCT_VOL_RUNOFF) = vol_struct.vol_runoff;
 F(VOL_STRUCT_VOL_INFILT) = vol_struct.vol_infilt;
 F(VOL_STRUCT_VOL_RUNON_INFILT) = vol_struct.vol_runon_infilt;
 F(VOL_STRUCT_VOL_OUT_SURFACE) = vol_struct.vol_out_surface;
 F(VOL_STRUCT_VOL_END_SURFACE) = vol_struct.vol_end_surface;
 F(VOL_STRUCT_VOL_TO_GW) = vol_struct.vol_to_gw;
 F(VOL_STRUCT_VOL_IN_GW_START) = vol_struct.vol_in_gw_start;
 F(VOL_STRUCT_VOL_IN_GW_END) = vol_struct.vol_in_gw_end;
 F(VOL_STRUCT_VOL_FROM_GW) = vol_struct.vol_from_gw;
 F(VOL_STRUCT_VOL_IN_NASH) = vol_struct.vol_in_nash;
 F(VOL_STRUCT_VOL_IN_NASH_END) = vol_struct.vol_in_nash_end;
 F(VOL_STRUCT_VOL_OUT_NASH) = vol_struct.vol_out_nash;
 F(VOL_STRUCT_VOL_SOIL_START) = vol_struct.vol_soil_start;
 F(VOL_STRUCT_VOL_TO_SOIL) = vol_struct.vol_to_soil;
 F(VOL_STRUCT_VOL_SOIL_TO_LAT_FLOW) = vol_struct.vol_soil_to_lat_flow;
 F(VOL_STRUCT_VOL_SOIL_TO_GW) = vol_struct.vol_soil_to_gw;
 F(VOL_STRUCT_VOL_SOIL_END) = vol_struct.vol_soil_end;
 F(VOL_STRUCT_VOL_ET_FROM_SOIL) = vol_struct.vol_et_from_soil;
 F(VOL_STRUCT_VOL_ET_FROM_RAIN) = vol_struct.vol_et_from_rain;
 F(VOL_STRUCT_VOL_ET_TO_ATM) = vol_struct.vol_et_to_atm;
 F(VOL_STRUCT_VOLIN) = vol_struct.volin;
 F(VOL_STRUCT_VOLOUT) = vol_struct.volout;
 F(VOL_STRUCT_VOLEND) = vol_struct.volend;
 F(VOL_STRUCT_VOL_ET_FROM_RETENTION_DEPTH) = vol_struct.vol_et_from_retention_depth;
 F(VOL_STRUCT_CUMULATIVE_VOL) = vol_struct.cumulative_vol;
 F(VOL_STRUCT_VOLUME_IN_DOMAIN) = vol_struct.volume_in_domain;
 F(VOL_STRUCT_LEAKAGE) = vol_struct.leakage;
 F(INFILTRATION_EXCESS_PARAMS_STRUCT_SCHAAKE_ADJUSTED_MAGIC_CONSTANT_BY_SOIL_TYPE) = infiltration_excess_params_struct.Schaake_adjusted_magic_constant_by_soil_type;
 F(INFILTRATION_EXCESS_PARAMS_STRUCT_A_XINANJIANG_INFLECTION_POINT_PARAMETER) = infiltration_excess_params_struct.a_Xinanjiang_inflection_point_parameter;
 F(INFILTRATION_EXCESS_PARAMS_STRUCT_B_XINANJIANG_SHAPE_PARAMETER) = infiltration_excess_params_struct.b_Xinanjiang_shape_parameter;
 F(INFILTRATION_EXCESS_PARAMS_STRUCT_X_XINANJIANG_SHAPE_PARAMETER) = infiltration_excess_params_struct.x_Xinanjiang_shape_parameter;
 F(INFILTRATION_EXCESS_PARAMS_STRUCT_URBAN_DECIMAL_FRACTION) = infiltration_excess_params_struct.urban_decimal_fraction;
 F(INFILTRATION_EXCESS_PARAMS_STRUCT_ICE_CONTENT_THRESHOLD) = infiltration_excess_params_struct.ice_content_threshold;
 F(NASH_SURFACE_PARAMS_K_NASH) = nash_surface_params.K_nash;
 F(NASH_SURFACE_PARAMS_RETENTION_DEPTH) = nash_surface_params.retention_depth;
 F(NASH_SURFACE_PARAMS_RUNON_INFILTRATION) = nash_surface_params.runon_infiltration;
 F(NASH_SURFACE_PARAMS_K_INFILTRATION) = nash_surface_params.K_infiltration;
 if(rain_dtype){
  current_step[p]+=1;
  double depth=0.0;
  if(I(SURFACE_RUNOFF_SCHEME)==GIUH){
   for(int i=0;i<ng;++i)depth+=runoff_queue[qo+i];
  }else{
   for(int i=0;i<ns;++i)depth+=nash_surface[so+i];
  }
  ponded_depth[p]=depth;
 }
 #undef F
 #undef I
}

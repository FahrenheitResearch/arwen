/* SPDX-License-Identifier: Apache-2.0
 * Observation-only REFEREE TOOL. Compile against the pinned original PET repo;
 * never link this executable or the native BMI library into the product.
 * Input is the unchanged CSV shipped by pinned ngen. Every numerical update
 * goes through the original register_bmi_pet -> initialize/set_value/update.
 * Records: before parameters[12], canonical forcing[8], after parameters[12],
 * native PET[1], diagnostics[6], all native little-endian double on x86_64.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmi.h"
#include "bmi_pet.h"
#include "pet.h"

static void parameters(const pet_model *m, double *p) {
  p[0]=m->pet_params.wind_speed_measurement_height_m;
  p[1]=m->pet_params.humidity_measurement_height_m;
  p[2]=m->pet_params.vegetation_height_m;
  p[3]=m->pet_params.zero_plane_displacement_height_m;
  p[4]=m->pet_params.momentum_transfer_roughness_length_m;
  p[5]=m->pet_params.heat_transfer_roughness_length_m;
  p[6]=m->surf_rad_params.surface_longwave_emissivity;
  p[7]=m->surf_rad_params.surface_shortwave_albedo;
  p[8]=m->pet_forcing.canopy_resistance_sec_per_m;
  p[9]=m->pet_forcing.water_temperature_C;
  p[10]=m->pet_forcing.ground_heat_flux_W_per_sq_m;
  p[11]=m->surf_rad_forcing.surface_skin_temperature_C;
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr,"usage: capture_pet CONFIG CSV RECORDS.bin\n"); return 2; }
  Bmi bmi = {0};
  if (!register_bmi_pet(&bmi) || bmi.initialize(&bmi,argv[1])) return 3;
  FILE *in=fopen(argv[2],"r"), *out=fopen(argv[3],"wb");
  if (!in || !out) return 4;
  char *line=NULL; size_t cap=0; long records=0;
  if (getline(&line,&cap,in)<0) return 5;
  const char *names[]={
    "land_surface_radiation~incoming~longwave__energy_flux",
    "land_surface_air__pressure",
    "atmosphere_air_water~vapor__relative_saturation",
    "land_surface_radiation~incoming~shortwave__energy_flux",
    "land_surface_air__temperature",
    "land_surface_wind__x_component_of_velocity",
    "land_surface_wind__y_component_of_velocity"
  };
  const int csv_columns[]={1,3,4,2,5,6,7};
  while (getline(&line,&cap,in)>=0) {
    char *cursor=line;
    if (!strsep(&cursor,",")) return 6; /* native timestamp, unused by PET */
    double v[9];
    for(int k=0;k<9;k++) {
      char *word=strsep(&cursor,",");
      if(!word) return 7;
      char *end; v[k]=strtod(word,&end); if(end==word) return 8;
    }
    double record[39]={0};
    pet_model *m=bmi.data;
    parameters(m,record);
    double canonical[]={v[5],v[4],v[6],v[7],v[3],v[1],v[2],v[8]};
    memcpy(record+12,canonical,sizeof(canonical));
    for(int k=0;k<7;k++) if(bmi.set_value(&bmi,names[k],&v[csv_columns[k]])) return 9;
    if(bmi.update(&bmi)) return 10;
    parameters(m,record+20);
    if(bmi.get_value(&bmi,"water_potential_evaporation_flux",record+32)) return 11;
    record[33]=m->pet_forcing.net_radiation_W_per_sq_m;
    record[34]=m->pet_forcing.wind_speed_m_per_s;
    record[35]=m->inter_vars.air_saturation_vapor_pressure_Pa;
    record[36]=m->inter_vars.air_actual_vapor_pressure_Pa;
    record[37]=m->inter_vars.moist_air_density_kg_per_m3;
    record[38]=m->inter_vars.slope_sat_vap_press_curve_Pa_s;
    if(fwrite(record,sizeof(record),1,out)!=1) return 12;
    records++;
  }
  free(line); fclose(in); fclose(out); bmi.finalize(&bmi);
  printf("native BMI PET records=%ld width=39 dtype=float64\n",records);
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
// Observation-only REFEREE TOOL against original pinned ngen; no native code
// from this program is linked or installed on the product path.
#include "forcing/CsvPerFeatureForcingProvider.hpp"
#include <cstdio>

int main(int argc,char **argv) {
    if(argc!=3 && argc!=4) {std::fprintf(stderr,"usage: capture_forcing CSV OUTPUT.bin [rain-units]\n");return 2;}
    forcing_params config(argv[1],"CsvPerFeature","2015-12-01 00:00:00","2015-12-30 23:00:00");
    CsvPerFeatureForcingProvider provider(config);
    const char *names[]={CSDMS_STD_NAME_SURFACE_TEMP,NGEN_STD_NAME_SPECIFIC_HUMIDITY,
        CSDMS_STD_NAME_WIND_U_X,CSDMS_STD_NAME_WIND_V_Y,CSDMS_STD_NAME_SURFACE_AIR_PRESSURE,
        CSDMS_STD_NAME_SOLAR_LONGWAVE,CSDMS_STD_NAME_SOLAR_SHORTWAVE,CSDMS_STD_NAME_LIQUID_EQ_PRECIP_RATE};
    const char *units[]={"K","kg kg-1","m s-1","m s-1","Pa","W m-2","W m-2","mm s^-1"};
    if(argc==4) {
        init_unit_system();
        units[7]=argv[3];
        const double factor=UnitsHelper::get_converted_value("mm s^-1",1.0,units[7]);
        std::printf("native rain unit conversion factor=%a %.17g to %s\n",factor,factor,units[7]);
    }
    FILE *out=std::fopen(argv[2],"wb");if(!out)return 3;
    for(long hour=0;hour<720;hour++) {
        double row[8];
        for(int f=0;f<8;f++) row[f]=provider.get_value(
            CatchmentAggrDataSelector("cat-27",names[f],config.simulation_start_t+hour*3600,3600,units[f]),
            data_access::ReSampleMethod::MEAN);
        if(std::fwrite(row,sizeof(row),1,out)!=1)return 4;
    }
    std::fclose(out);std::puts("native ngen CsvPerFeature forcing records=720 width=8 float64, hourly");
}

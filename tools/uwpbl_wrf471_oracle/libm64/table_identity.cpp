#include "host_shim.h"
#include "glibc_flt64.cuh"
extern "C" const void *g64_table(int f){return f==0?(const void*)g64_exp::__exp_data.tab:f==1?(const void*)g64_log::__log_data.tab:(const void*)g64_pow::__pow_log_data.tab;}

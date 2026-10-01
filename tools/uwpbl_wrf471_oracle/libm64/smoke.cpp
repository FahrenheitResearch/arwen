#include "host_shim.h"
#include "glibc_flt64.cuh"
#include <stdio.h>
int main(){printf("%a %a %a %a %a\n",glibc_exp(0.3),glibc_log(0.3),glibc_pow(0.3,2./3.),uw_cos(0.3),uw_acos(0.3));}

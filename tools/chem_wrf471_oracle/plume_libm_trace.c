/* Observe scalar libm arguments; returned words are the original glibc words. */
#include <stdio.h>
#include <stdlib.h>
extern float __real_powf(float, float);
extern float __real_expf(float);
static FILE *stream;
static int initialized;
static void record(float x, float y, float value, float operation) {
    if (!initialized) {
        const char *path=getenv("PLUME_LIBM_TRACE");
        if(path) stream=fopen(path,"wb");
        initialized=1;
    }
    if(stream) { float row[4]={operation,x,y,value}; fwrite(row,sizeof(float),4,stream); }
}
float __wrap_powf(float x,float y) { float v=__real_powf(x,y); record(x,y,v,2.0f); return v; }
float __wrap_expf(float x) { float v=__real_expf(x); record(x,0.0f,v,1.0f); return v; }

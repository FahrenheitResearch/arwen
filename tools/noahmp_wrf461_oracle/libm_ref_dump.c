/* Dump live glibc logf/atanf/powf(x,0.25f) for a list of hex float32 inputs.
   Produces gpuwm/data/noahmp/oracle/noahmp-bareflux-libm.csv, which is what
   both the CPU and the CUDA transcriptions of those three functions are
   gated against.  Build:  gcc -O2 -o libm_ref_dump libm_ref_dump.c -lm
   Run:    ./libm_ref_dump < hex_inputs.txt
   The compiler flags do not matter here: the values come from libm.so.6,
   not from anything this file computes. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static float asf(uint32_t i){ float f; memcpy(&f,&i,4); return f; }
static uint32_t asu(float f){ uint32_t i; memcpy(&i,&f,4); return i; }

int main(int argc, char** argv){
    char line[64];
    while (fgets(line, sizeof line, stdin)) {
        uint32_t ix;
        if (sscanf(line, "%x", &ix) != 1) continue;
        float x = asf(ix);
        printf("%08X %08X %08X %08X\n", ix, asu(logf(x)), asu(atanf(x)), asu(powf(x, 0.25f)));
    }
    return 0;
}

// Widen subnormal words without the hardware DAZ conversion.
__device__ double cs_widen(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0u) {
        double a = __dmul_rn((double)(u & 0x7fffffu), 0x1p-149);
        return (u & 0x80000000u) ? -a : a;
    }
    return (double)x;
}
__device__ float cs_mul(float a, float b) { return gfk_d2f_rn(__dmul_rn(cs_widen(a),cs_widen(b))); }
__device__ float cs_div(float a, float b) { return gfk_d2f_rn(__ddiv_rn(cs_widen(a),cs_widen(b))); }
__device__ float cs_add(float a, float b) { return gfk_d2f_rn(__dadd_rn(cs_widen(a),cs_widen(b))); }
__device__ float cs_sub(float a, float b) { return gfk_d2f_rn(__dsub_rn(cs_widen(a),cs_widen(b))); }
extern "C" __global__ void cold_start_alt(const double* input, float* output, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) output[j] = gfk_d2f_rn(input[j]);
}
__device__ bool cs_nonpositive(float x) {
    unsigned int bits = __float_as_uint(x), mag = bits & 0x7fffffffu;
    return mag == 0u || ((bits & 0x80000000u) && mag <= 0x7f800000u);
}
extern "C" __global__ void cold_start_surface(
    const float* alt, const float* aerosol, unsigned char* surface, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        float nw = cs_mul(aerosol[j],cs_div(1.0f,alt[j]));
        surface[j] = isnan(nw) ? 2 : cs_nonpositive(nw);
    }
}
// Gathered cold-start cells. Operations follow the host mirror's widths.
extern "C" __global__ void cold_start_temperature(
    const double* theta, const double* pressure, const long long* indices,
    float* temperature, int n, double p0, double rcp)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    long long cell = indices[j];
    double ratio = __ddiv_rn(pressure[cell], p0);
    temperature[j] = gfk_d2f_rn(__dmul_rn(theta[cell],plm_pow(ratio,rcp)));
}
extern "C" __global__ void cold_start_numbers(
    const float* mass, const float* number, const float* alt,
    const float* temp, const float* aerosol, const float* xland,
    const float* retab, const float* gratio, const double* c,
    float* final_number, float* seed_volume, int n, int species)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float q = mass[j], old = number[j], a = alt[j];
    float rho = cs_div(1.0f, a), v = cs_mul(q, rho);
    float t = temp[j], seeded;
    if (species == 0) {
        double n0 = (double)(float)8.0e6;
        if (t <= (float)271.15) n0 = (double)(float)8.0e8;
        else if (t < (float)273.15)
            n0 = (double)cs_mul(8.0f, gfk_pow(10.0f, cs_sub((float)279.15,t)));
        double lam = __dsqrt_rn(__dsqrt_rn(__ddiv_rn(
            __dmul_rn(__dmul_rn(n0,c[0]),6.0),cs_widen(v))));
        double nv = __ddiv_rn(__dmul_rn(__dmul_rn(__dmul_rn(
            cs_widen(cs_div(v,(float)c[1])),lam),lam),lam),c[0]);
        seeded = gfk_d2f_rn(nv);
    } else if (species == 1) {
        long long raw_idx = (long long)cs_sub(t,179.0f);
        int idx = (int)max(1LL,min(raw_idx,94LL));
        float corr = cs_sub(t,truncf(t));
        float re = cs_add(cs_mul(retab[idx-1],cs_sub(1.0f,corr)),
                            cs_mul(retab[idx],corr));
        float de = cs_mul(cs_mul(2.0f,re),(float)1.0e-6);
        double lam = cs_widen(cs_div(3.0f,de));
        seeded = gfk_d2f_rn(__ddiv_rn(__dmul_rn(__dmul_rn(
            __dmul_rn(cs_widen(v),lam),lam),lam),c[2]));
    } else {
        float nw = cs_mul(aerosol[j],rho), dc;
        int nu;
        if (cs_nonpositive(nw)) {
            bool ocean = cs_sub(xland[j],1.5f) > 0.0f;
            dc = ocean ? (float)17.0e-6 : (float)11.0e-6;
            nu = ocean ? 12 : 4;
        } else {
            float bounded = fmaxf((float)99.0e6,fminf(nw,(float)5.0e10));
            double ratio = cs_widen(cs_div((float)2.5e10,bounded));
            nu = max(2,min(15,(int)floor(__dadd_rn(ratio,0.5))));
            float x = cs_sub(fmaxf(1.0f,fminf(cs_mul(bounded,(float)1.0e-9),10.0f)),1.0f);
            dc = cs_mul(cs_sub(30.0f,cs_div(cs_mul(x,20.0f),(float)c[3])),(float)1.0e-6);
        }
        double lam = __ddiv_rn(__dadd_rn(4.0,(double)nu),cs_widen(dc));
        double first = cs_widen(cs_div(v,gratio[nu-1]));
        seeded = gfk_d2f_rn(__ddiv_rn(__dmul_rn(__dmul_rn(__dmul_rn(first,lam),lam),lam),c[0]));
    }
    seed_volume[j] = seeded;
    float result = cs_div(seeded,rho);
    // Entry offenders use the DOUBLE reciprocal of the rounded specific volume.
    if (q > (float)c[4] && old <= 0.0f && species != 2) {
        double density = __ddiv_rn(1.0,cs_widen(a));
        float m = gfk_d2f_rn(__dmul_rn(cs_widen(q),density));
        float initial_num = gfk_d2f_rn(__dmul_rn(cs_widen(result),density));
        // NumPy maximum propagates NaN; CUDA fmax alone would hide a refusal.
        float num = isnan(initial_num) ? initial_num : fmaxf((float)c[5],initial_num);
        float exponent = (float)c[6];
        if (species == 0) {
            float pref = cs_mul((float)c[7],m);
            if (num <= (float)c[5]) num = gfk_d2f_rn(__ddiv_rn(__dmul_rn(cs_widen(pref),c[8]),c[9]));
            float arg = cs_div(cs_mul(cs_mul((float)c[9],6.0f),num),m);
            double lam = (double)gfk_pow(arg,exponent);
            float mvd = gfk_d2f_rn(__ddiv_rn(c[10],lam));
            if (mvd > (float)c[11]) num = gfk_d2f_rn(__ddiv_rn(__dmul_rn(cs_widen(pref),c[12]),c[9]));
            if (mvd < (float)c[13]) num = gfk_d2f_rn(__ddiv_rn(__dmul_rn(cs_widen(pref),c[14]),c[9]));
        } else {
            float pref = cs_div(cs_mul((float)c[7],m),(float)c[15]);
            double small = fmin(c[16],__dmul_rn(cs_widen(pref),c[17]));
            if (num <= (float)c[5]) num = gfk_d2f_rn(small);
            float arg = cs_div(cs_mul(cs_mul((float)c[15],6.0f),num),m);
            double lam = (double)gfk_pow(arg,exponent);
            float di = gfk_d2f_rn(__dmul_rn(4.0,__ddiv_rn(1.0,lam)));
            if (di < (float)c[18]) num = gfk_d2f_rn(small);
            else if (di > (float)c[19]) num = gfk_d2f_rn(__dmul_rn(cs_widen(pref),c[20]));
        }
        result = gfk_d2f_rn(__ddiv_rn(cs_widen(num),density));
    }
    final_number[j] = result;
}

// Independent FRAM, doi:10.1175/WAF-D-15-0118.1, NOAA/NWS.
// Saved-frame right-endpoint sampling, counters mm, wind m/s.
// Forecast state is const. Missing history remains missing.
extern "C" __global__ void saved_frame_ice(
    const float *rain, const float *previous_rain, const float *previous_liquid,
    const float *previous_horizontal, const float *previous_radial,
    const float *t2, const float *skin, const float *q2, const float *psfc,
    const float *u10, const float *v10, const float *qr, const float *qs, const float *qg,
    float *liquid, float *horizontal, float *radial, float *wetbulb, float *rate,
    unsigned long long cells, double hours, int have_skin) {
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=cells) return;
    double t=(double)t2[i]-273.15, pressure=psfc[i]/100.0;
    double q=q2[i], r=qr[i], s=qs[i], g=qg[i];
    double delta=(double)rain[i]-previous_rain[i];
    float missing=__int_as_float(0x7fc00000);
    if (!isfinite(t) || t2[i]<=0 || !isfinite(pressure) || pressure<=0 ||
        !isfinite(q) || q<0 || !isfinite(r) || !isfinite(s) || !isfinite(g) ||
        !isfinite(delta) || delta<0 || !isfinite(u10[i]) || !isfinite(v10[i]) ||
        (have_skin && (!isfinite(skin[i]) || skin[i]<=0)) ||
        !isfinite(previous_liquid[i]) || !isfinite(previous_horizontal[i]) || !isfinite(previous_radial[i])) {
        liquid[i]=horizontal[i]=radial[i]=wetbulb[i]=rate[i]=missing; return;
    }
    double vapor=pressure*q/(0.622+q);
    // Pressure-aware psychrometric wet bulb over water, monotone bisection.
    double lo=-100.0, hi=t;
    for (int iteration=0; iteration<40; ++iteration) {
        double tw=0.5*(lo+hi);
        double es=6.112*exp(17.67*tw/(tw+243.5));
        double e=es-0.00066*(1.0+0.00115*tw)*pressure*(t-tw);
        if (e>vapor) hi=tw; else lo=tw;
    }
    double tw=0.5*(lo+hi);
    wetbulb[i]=(float)tw;
    // Same lowest-level fall-flux classifier as rw_wrfbatch::push_precip_type.
    r=fmax(r,0.0)*5.0; s=fmax(s,0.0); g=fmax(g,0.0)*3.0;
    double total=r+s+g;
    double flux=psfc[i]/(287.04*t2[i])*total*3600.0;
    bool freezing_rain=total>0 && flux>=0.1 && (s+g)/total<=0.3 &&
        (t2[i]<273.15f || (have_skin && skin[i]<273.15f));
    double dz=(freezing_rain && delta>0 && hours>0) ? delta : 0.0, flat=0.0;
    if (freezing_rain && delta>0 && hours>0 && tw<1.0) {
        double p=(delta/hours)/25.4;
        double v=hypot((double)u10[i],(double)v10[i])/(1852.0/3600.0);
        double w=fmax(tw,-7.0);
        double ip=0.1395*pow(p,-0.541);
        double it=((-0.0071*w-0.1039)*w-0.3904)*w+0.5545;
        double iv=(0.0014*v+0.0027)*v+0.7574;
        double ilr=w> -0.35 ? 0.70*ip+0.29*it+0.01*iv :
            (v>12.0 ? 0.73*ip+0.01*it+0.26*iv : 0.79*ip+0.20*it+0.01*iv);
        flat=delta*fmax(ilr,0.0);
    }
    rate[i]=hours>0 ? (float)(dz/hours) : 0.0f;
    liquid[i]=(float)((double)previous_liquid[i]+dz);
    horizontal[i]=(float)((double)previous_horizontal[i]+flat);
    radial[i]=(float)((double)previous_radial[i]+flat*0.394);
}

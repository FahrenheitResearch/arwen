// SPDX-License-Identifier: Apache-2.0
// ArWen catchment area aggregation and prepared external time windows.

extern "C" __global__ void area_forcing(
    long long n, const unsigned long long *offsets,
    const unsigned int *indices, const double *weights,
    const float *t, const float *q, const float *u, const float *v,
    const float *p, const float *lw, const float *sw, const float *rain,
    const float *rain_a, const float *rain_b, double *previous_rain,
    int accumulated, int mixing_ratio, float *output, int *status, double dt) {
  const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const float *sources[8] = {t,q,u,v,p,lw,sw,rain};
  double sums[8] = {};
  int invalid = 0;
  for (unsigned long long j=offsets[c]; j<offsets[c+1]; ++j) {
    const unsigned int cell = indices[j];
    const double weight = weights[j];
    for (int f=0; f<7; ++f) {
      double value = (double)sources[f][cell];
      if (f==1 && mixing_ratio) {
        if (value<0.0 || !isfinite(value)) { invalid=1; value=nan(""); }
        else value=value/(1.0+value);
      }
      sums[f] += value*weight;
    }
    double rate;
    if (accumulated) {
      const double current = (double)rain_a[cell] + (double)rain_b[cell];
      const double delta = current - previous_rain[j];
      rate = delta / dt;
      if (delta < 0.0 || !isfinite(current)) { invalid=1; rate=nan(""); }
      previous_rain[j] = current;
    } else {
      rate=rain[cell];
      if (rate<0.0 || !isfinite(rate)) {invalid=1; rate=nan("");}
    }
    sums[7] += rate*weight;
  }
  for (int f=0; f<8; ++f) {
    output[f*n+c] = (float)sums[f];
    if (!isfinite(output[f*n+c])) invalid=1;
  }
  status[c] = invalid;
}

#ifndef SERIES_TYPE
#define SERIES_TYPE float
#endif
extern "C" __global__ void series_forcing(
    long long n, long long nt, double cadence,
    const SERIES_TYPE *frames, double *clock, SERIES_TYPE *output,
    int *status, double dt) {
  const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  const double start=clock[c], end=start+dt;
  if (!(dt>0.0) || start<0.0 || end>nt*cadence) {
    status[c]=1;
    for (int f=0;f<8;++f) output[f*n+c]=(SERIES_TYPE)nan("");
    return;
  }
  double sums[8] = {};
  long long frame=(long long)floor(start/cadence);
  double cursor=start;
  while (cursor<end) {
    const double next=fmin(end,(frame+1)*cadence);
    const double fraction=(next-cursor)/dt;
    for (int f=0;f<8;++f)
      sums[f] += (double)frames[(frame*8+f)*n+c]*fraction;
    cursor=next; ++frame;
  }
  int invalid=0;
  for (int f=0;f<8;++f) {
    output[f*n+c]=(SERIES_TYPE)sums[f];
    if (f==7 && output[f*n+c]<0.0) output[f*n+c]=(SERIES_TYPE)nan("");
    if (!isfinite(output[f*n+c])) invalid=1;
  }
  clock[c]=end;
  status[c]=invalid;
}

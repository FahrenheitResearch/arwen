// Actual Python DA call replay; all series were Rust-promoted from captured
// native buffers. Independent calls retain their captured carriers and clock.
// SPDX-License-Identifier: Apache-2.0
extern "C" __global__ void ngen_reservoir_da_replay(
 int n, int kind, const double *input, const long long *offsets,
 const double *series, const double *times, double *output) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 #define V(c) input[(c)*n+r]
 #define O(c) output[(c)*n+r]
 const double *values=series+offsets[r];
 if(kind==4) {
   NgenRFCState state={V(6),int(V(2))};
   auto answer=ngen_rfc(V(0)!=0.,values,int(V(3)),V(4),V(5),state,
      V(7),V(8),int(V(9)),V(10),V(11),V(12),V(13),V(14),V(15),int(V(16)));
   O(0)=answer.release;O(1)=answer.height;O(2)=state.update_time;
   O(3)=state.index;O(4)=answer.type;O(5)=answer.value;O(6)=answer.source;
 } else {
   NgenPersistenceState state={V(4),V(15),V(5),int(V(6))};
   auto answer=ngen_persistence(values,times+offsets[r],int(offsets[r+1]-offsets[r]),
      V(3),state,V(7),V(8),V(9),V(10),V(11),V(12),V(13),V(14),V(16),V(17));
   O(0)=answer.release;O(1)=state.persisted;O(2)=answer.height;O(3)=state.update_time;
   O(4)=state.index;O(5)=state.persistence_time;
 }
 #undef V
 #undef O
}

// Feed the exact native captured DA boundary into production resident arrays.
// This bypasses LP only because its native outputs are the module's inputs.
extern "C" __global__ void ngen_reservoir_da_feed(int n,int kind,
 const double* v,float* p,float* inflow,float* initial,float* lpq,float* lph,
 int* types,double* controls,double* clock,double* carriers,int* indices) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 #define V(c) v[(c)*n+r]
 types[r]=kind;clock[r]=V(kind==4?5:3);
 for(int c=0;c<9;++c)p[c*n+r]=0.f;
 for(int c=0;c<8;++c)controls[c*n+r]=0.;
 for(int c=0;c<3;++c)carriers[c*n+r]=0.;
 if(kind==4) {
   // Native r.area*1e6 arrives as a Python double; original area is f32.
   p[r]=float(V(14)/1.e6);p[2*n+r]=float(V(15));
   inflow[r]=float(V(10));initial[r]=float(V(11));lpq[r]=float(V(12));lph[r]=float(V(13));
   indices[r]=int(V(2));carriers[n+r]=V(6);
   controls[3*n+r]=V(0);controls[4*n+r]=V(7);controls[5*n+r]=V(8);
   controls[6*n+r]=V(3);controls[7*n+r]=V(16);
 } else {
   p[r]=float(V(10));p[2*n+r]=float(V(11));p[6*n+r]=float(V(12));
   inflow[r]=float(V(8));initial[r]=float(V(13));lpq[r]=float(V(7));lph[r]=0.f;
   carriers[r]=V(4);carriers[n+r]=V(15);carriers[2*n+r]=V(5);indices[r]=int(V(6));
   controls[r]=V(14);controls[n+r]=V(16);controls[2*n+r]=V(17);
 }
 #undef V
}
extern "C" __global__ void ngen_reservoir_da_record(int n,int kind,const double* release,
 const float* height,const double* carriers,const int* indices,const int* type,
 const double* assimilated,const int* source,double* output) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 output[r]=release[r];
 if(kind==4) {
   output[n+r]=height[r];output[2*n+r]=carriers[n+r];output[3*n+r]=indices[r];
   output[4*n+r]=type[r];output[5*n+r]=assimilated[r];output[6*n+r]=source[r];
 } else {
   output[n+r]=carriers[r];output[2*n+r]=height[r];output[3*n+r]=carriers[n+r];
   output[4*n+r]=indices[r];output[5*n+r]=carriers[2*n+r];
 }
}

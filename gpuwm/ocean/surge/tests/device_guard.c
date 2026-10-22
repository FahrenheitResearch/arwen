#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
static int guarded=0, copies=0;
static uintptr_t expected_stream;
void surge_test_stream(uintptr_t value){expected_stream=value;}
static void* symbol(const char* name){
 static void* handle; if(!handle)handle=dlopen("/usr/local/cuda/lib64/libcudart.so",RTLD_NOW|RTLD_LOCAL);
 void* result=dlsym(handle,name); if(!result){fprintf(stderr,"guard lookup failed: %s\n",name);abort();} return result;
}
void surge_test_guard(int on){guarded=on;}
int surge_test_d2d_count(void){return copies;}
static void fail(const char*name){fprintf(stderr,"FORBIDDEN CUDA ABI CALL: %s\n",name);abort();}
cudaError_t cudaMemcpy(void*d,const void*s,size_t n,enum cudaMemcpyKind k){
 if(guarded)fail("cudaMemcpy");
 cudaError_t(*real)(void*,const void*,size_t,enum cudaMemcpyKind)=symbol("cudaMemcpy");return real(d,s,n,k);
}
cudaError_t cudaMemcpyAsync(void*d,const void*s,size_t n,enum cudaMemcpyKind k,cudaStream_t stream){
 if(guarded&&k!=cudaMemcpyDeviceToDevice)fail("cudaMemcpyAsync host transfer");
 if(guarded){if((uintptr_t)stream!=expected_stream)fail("wrong CUDA stream");++copies;}
 cudaError_t(*real)(void*,const void*,size_t,enum cudaMemcpyKind,cudaStream_t)=symbol("cudaMemcpyAsync");return real(d,s,n,k,stream);
}
cudaError_t cudaDeviceSynchronize(void){
 if(guarded)fail("cudaDeviceSynchronize");cudaError_t(*real)(void)=symbol("cudaDeviceSynchronize");return real();
}
cudaError_t cudaStreamSynchronize(cudaStream_t s){
 if(guarded)fail("cudaStreamSynchronize");cudaError_t(*real)(cudaStream_t)=symbol("cudaStreamSynchronize");return real(s);
}

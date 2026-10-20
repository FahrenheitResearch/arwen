/* CPU-only shared fixture for the real Python/C adapter. */
#define main fixture_c_main
#include "test_native_host.c"
#undef main
int fixture_begin(Plan*p){
 if(recording || !p)return BAD_STATE;
 host_alloc=alloc_host;host_free=free_host;current_context=context;
 stream_sync=sync_cpu;stream_sync_ptsz=sync_cpu;real_cuMemcpyDtoHAsync_v2=copy_cpu;
 recording=p;p->context=(void*)1;return 0;
}
int fixture_read(int64_t*out,const int64_t*input){return wrap_cuMemcpyDtoHAsync_v2((DevicePtr)(uintptr_t)out,(DevicePtr)(uintptr_t)input,8,NULL);}
int fixture_end(Plan*p){Operation*o=append(KERNEL,"fake_downstream",(void*)kernel);if(!o)return NO_MEMORY;p->kernels++;return awstep_end(p);}
int fixture_consumer(const void*args,size_t bytes){return helper(args,bytes);}
unsigned fixture_kernels(void){return kernels;}

/* Only new readback-to-kernel argument cases; actual recorder, no CUDA. */
#define main previous_cpu_main
#include "test_native_host.c"
#undef main
static uint32_t received;
static int scalar_kernel(Handle f,unsigned a,unsigned b,unsigned c,unsigned d,unsigned e,unsigned g,unsigned h,Handle st,void**args,void**extra){
 (void)f;(void)a;(void)b;(void)c;(void)d;(void)e;(void)g;(void)h;(void)st;(void)extra;memcpy(&received,args[0],4);kernels++;return 0;
}
static size_t scalar(void){
 Operation*o=append(KERNEL,"scalar",(void*)scalar_kernel);assert(o);o->nargs=1;
 o->args=calloc(1,sizeof(Argument));o->params=calloc(1,sizeof(void*));assert(o->args && o->params);
 o->args[0].bytes=calloc(1,4);o->args[0].size=4;o->args[0].slot=SIZE_MAX;recording->kernels++;return recording->size-1;
}
static size_t readback(Plan*p,uint32_t*device,uint32_t*ordinary){
 size_t source;assert(awstep_host_begin(p,4,0)==0);
 assert(wrap_cuMemcpyDtoHAsync_v2((DevicePtr)(uintptr_t)ordinary,(DevicePtr)(uintptr_t)device,8,NULL)==0);
 assert(awstep_host_end(p,&source)==0);return source;
}
int main(void){
 unsigned cases=0;host_alloc=alloc_host;host_free=free_host;current_context=context;
 stream_sync=sync_cpu;stream_sync_ptsz=sync_cpu;real_cuMemcpyDtoHAsync_v2=copy_cpu;
 uint32_t device[2]={11,29},ordinary[2]={0,0};Plan*p=start();size_t source=readback(p,device,ordinary),op=scalar();
 assert(awstep_end(p)==0);
 assert(awstep_bind_host_argument(p,op,0,source,4)==0);cases++;
 ordinary[1]=999;device[1]=UINT32_C(0x80000000);
 assert(awstep_execute(p,NULL,0,0)==0 && received==device[1] && ordinary[1]==999);cases++;
 device[1]=UINT32_C(0x7fc00123);
 assert(awstep_execute(p,NULL,0,0)==0 && received==device[1]);cases++;
 assert(awstep_bind_host_argument(p,op,0,source,5)==BAD_ARGUMENT);cases++;
 assert(awstep_bind_host_argument(p,op,0,source,SIZE_MAX)==BAD_ARGUMENT);cases++;
 assert(awstep_bind_host_argument(p,op,1,source,0)==BAD_ARGUMENT);cases++;
 assert(awstep_bind_host_argument(p,op,0,op,0)==BAD_STATE);cases++;
 assert(awstep_bind_host_argument(p,source,0,source,0)==BAD_STATE);cases++;
 uint32_t frame=47;assert(awstep_bind(p,op,0,0,4)==0);
 assert(awstep_execute(p,&frame,4,0)==0 && received==47);cases++;
 assert(awstep_bind_host_argument(p,op,0,source,0)==0);
 assert(awstep_execute(p,NULL,0,0)==0 && received==11);cases++;drop(p);
 /* Producer predicates must cover the consumer. A flag mutation between
  * producer and consumer must not activate a consumer after skipped input. */
 p=start();assert(awstep_predicate(p,1,2)==0);source=readback(p,device,ordinary);
 assert(awstep_predicate(p,0,0)==0);op=scalar();assert(awstep_end(p)==0);
 assert(awstep_bind_host_argument(p,op,0,source,0)==BAD_ARGUMENT);cases++;drop(p);
 p=start();assert(awstep_predicate(p,1,2)==0);source=readback(p,device,ordinary);
 assert(awstep_predicate(p,5,10)==0);op=scalar();assert(awstep_end(p)==0);
 assert(awstep_bind_host_argument(p,op,0,source,0)==0);cases++;
 unsigned before=kernels;assert(awstep_execute(p,NULL,0,5)==0 && kernels==before+1 && received==11);cases++;
 before=kernels;assert(awstep_execute(p,NULL,0,7)==0 && kernels==before);cases++;drop(p);
 uint64_t count=3;p=start();assert(awstep_predicate(p,1,0)==0);source=readback(p,device,ordinary);
 assert(awstep_predicate(p,0,0)==0);assert(awstep_count_predicate(p,&count,0,0)==0);
 assert(awstep_predicate(p,1,0)==0);op=scalar();assert(awstep_end(p)==0);
 assert(awstep_bind_host_argument(p,op,0,source,0)==BAD_ARGUMENT);cases++;drop(p);
 p=start();assert(awstep_predicate(p,1,0)==0);source=readback(p,device,ordinary);
 assert(awstep_predicate(p,0,0)==0);assert(awstep_count_predicate(p,&count,0,2)==0);
 assert(awstep_predicate(p,1,0)==0);op=scalar();assert(awstep_end(p)==0);
 assert(awstep_bind_host_argument(p,op,0,source,0)==0);cases++;drop(p);
 p=start();Operation*fake=append(H2D,"not_readback",NULL);fake->host=ordinary;fake->count=8;
 op=scalar();assert(awstep_end(p)==0);assert(awstep_bind_host_argument(p,op,0,0,0)==BAD_ARGUMENT);cases++;drop(p);
 printf("{\"status\":\"PASS\",\"cpu_integration_cases\":%u,\"cuda_calls\":0,\"completed_previous_cases_repeated\":0}\n",cases);return 0;
}

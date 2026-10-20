/* New mutable-upload/count cases only, using the actual recorder. */
#define main previous_cpu_main
#include "test_native_host.c"
#undef main
int main(void){
 unsigned cases=0;host_alloc=alloc_host;host_free=free_host;current_context=context;
 stream_sync=sync_cpu;stream_sync_ptsz=sync_cpu;real_cuMemcpyHtoDAsync_v2=copy_cpu;
 Plan*p=start();int64_t input=5,original=0,device=0;Args args={&input,&original,7};size_t producer,upload;
 assert(awstep_native_host(p,(void*)helper,&args,sizeof(args),"fixture",&producer)==0);
 assert(awstep_host_begin(p,5,0)==0);
 assert(wrap_cuMemcpyHtoDAsync_v2((DevicePtr)(uintptr_t)&device,(DevicePtr)(uintptr_t)&original,8,NULL)==0);
 assert(awstep_host_end(p,&upload)==0 && device==12 && syncs==1);cases++;
 finish(p);unsigned oldcalls=calls;
 assert(awstep_execute(p,NULL,0,0)==BAD_ARGUMENT && calls==oldcalls);cases++;
 void*buffer;size_t bytes;assert(awstep_host_buffer(p,upload,&buffer,&bytes)==0 && bytes==8 && buffer!=&original);cases++;
 Args frame={&input,buffer,9};assert(awstep_bind(p,producer,0,0,sizeof(frame))==0);
 original=999;input=15;unsigned oldsyncs=syncs;
 assert(awstep_execute(p,&frame,sizeof(frame),0)==0 && device==24 && original==999 && syncs==oldsyncs+1);cases++;
 frame.delta=12;input=17;
 assert(awstep_execute(p,&frame,sizeof(frame),0)==0 && device==29 && original==999);cases++;
 input=-1;unsigned oldcopies=copies,oldkernels=kernels;
 assert(awstep_execute(p,&frame,sizeof(frame),0)==GUARD_FAILED && copies==oldcopies && kernels==oldkernels && device==29);cases++;drop(p);
 p=start();assert(awstep_host_begin(p,5,0)==0);
 assert(awstep_host_end_optional(p,&upload)==BAD_STATE && p->host_policy==5);cases++;
 assert(awstep_host_end(p,&upload)==BAD_STATE);cases++;drop(p);
 p=start();assert(awstep_host_begin(p,5,0)==0);
 for(unsigned j=0;j<2;j++)assert(wrap_cuMemcpyHtoDAsync_v2((DevicePtr)(uintptr_t)&device,(DevicePtr)(uintptr_t)&original,8,NULL)==0);
 assert(awstep_host_end(p,&upload)==BAD_STATE);cases++;drop(p);
 /* Current native output chooses one later kernel group. An unrelated bit
  * selects a second kernel and survives both set and clear operations. */
 uint64_t count=0;p=start();input=0;args=(Args){&input,(int64_t*)&count,0};
 assert(awstep_native_host(p,(void*)helper,&args,sizeof(args),"producer",&producer)==0);
 assert(awstep_count_predicate(p,&count,2,0)==0);
 assert(awstep_predicate(p,1,0)==0);append(KERNEL,"selected",(void*)kernel);p->kernels++;
 assert(awstep_predicate(p,8,0)==0);append(KERNEL,"unrelated",(void*)kernel);p->kernels++;
 assert(awstep_end(p)==0);oldkernels=kernels;
 assert(awstep_execute(p,NULL,0,9)==0 && count==0 && kernels==oldkernels+1);cases++;
 input=3;oldkernels=kernels;
 assert(awstep_execute(p,NULL,0,8)==0 && count==3 && kernels==oldkernels+2);cases++;
 input=2;oldkernels=kernels;
 assert(awstep_execute(p,NULL,0,9)==0 && count==2 && kernels==oldkernels+1);cases++;drop(p);
 p=start();assert(awstep_count_predicate(p,NULL,0,0)==BAD_ARGUMENT);cases++;
 assert(awstep_count_predicate(p,&count,0,64)==BAD_ARGUMENT);cases++;
 assert(awstep_host_begin(p,5,0)==0);
 assert(awstep_count_predicate(p,&count,0,0)==BAD_STATE);cases++;drop(p);
 /* Threshold uses all 64 bits, including the high bit. */
 p=start();count=UINT64_MAX;assert(awstep_count_predicate(p,&count,UINT64_MAX-1,63)==0);
 assert(awstep_predicate(p,UINT64_C(1)<<63,0)==0);finish(p);oldkernels=kernels;
 assert(awstep_execute(p,NULL,0,0)==0 && kernels==oldkernels+1);cases++;
 count=UINT64_MAX-1;oldkernels=kernels;
 assert(awstep_execute(p,NULL,0,UINT64_C(1)<<63)==0 && kernels==oldkernels);cases++;drop(p);
 printf("{\"status\":\"PASS\",\"cpu_integration_cases\":%u,\"cuda_calls\":0,\"completed_previous_cases_repeated\":0}\n",cases);return 0;
}

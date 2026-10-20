/* Test the actual recorder with CPU CUDA-call doubles, never a GPU. */
#include "../../gpuwm/native_step/recorder.c"
#include <assert.h>
typedef struct { const int64_t *input; int64_t *output; int64_t delta; } Args;
static unsigned copies,syncs,calls,kernels;
static int alloc_host(void **out,size_t n,unsigned flags){(void)flags;*out=malloc(n);return *out?0:NO_MEMORY;}
static int free_host(void *p){free(p);return 0;}
static int context(Handle *p){*p=(void*)1;return 0;}
static int copy_cpu(DevicePtr dst,DevicePtr src,size_t n,Handle st){(void)st;memcpy((void*)(uintptr_t)dst,(void*)(uintptr_t)src,n);copies++;return 0;}
static int sync_cpu(Handle st){(void)st;syncs++;return 0;}
static int helper(const void *data,size_t bytes){
 assert(bytes==sizeof(Args));const Args*a=data;calls++;
 if(*a->input<0)return GUARD_FAILED;
 *a->output=*a->input+a->delta;return 0;
}
static int kernel(Handle f,unsigned a,unsigned b,unsigned c,unsigned d,unsigned e,unsigned g,unsigned h,Handle st,void**args,void**extra){
 (void)f;(void)a;(void)b;(void)c;(void)d;(void)e;(void)g;(void)h;(void)st;(void)args;(void)extra;kernels++;return 0;
}
static Plan*start(void){Plan*p=awstep_create();assert(p);recording=p;p->context=(void*)1;return p;}
static void finish(Plan*p){Operation*o=append(KERNEL,"fake_downstream",(void*)kernel);assert(o);p->kernels++;assert(awstep_end(p)==0);}
static void drop(Plan*p){recording=NULL;awstep_destroy(p);}
int main(void){
 unsigned checks=0;host_alloc=alloc_host;host_free=free_host;current_context=context;
 stream_sync=sync_cpu;stream_sync_ptsz=sync_cpu;real_cuMemcpyDtoHAsync_v2=copy_cpu;
 int64_t device=5,original=0,result=0;size_t op,hostop;void*buffer;size_t bytes;
 Plan*p=start();assert(awstep_host_begin(p,4,0)==0);
 assert(wrap_cuMemcpyDtoHAsync_v2((DevicePtr)(uintptr_t)&original,(DevicePtr)(uintptr_t)&device,8,NULL)==0);
 assert(awstep_host_end(p,&hostop)==0 && copies==2 && syncs==1 && original==5);checks++;
 assert(awstep_host_buffer(p,hostop,&buffer,&bytes)==0 && bytes==8 && buffer!=&original && *(int64_t*)buffer==5);checks++;
 Args args={buffer,&result,7};assert(awstep_native_host(p,(void*)helper,&args,sizeof(args),"native_fixture",&op)==0 && result==12);checks++;
 args.delta=99;finish(p);device=11;original=991;unsigned before=calls;
 assert(awstep_execute(p,NULL,0,0)==0 && result==18 && original==991 && calls==before+1 && kernels==1);checks++;
 /* Full struct frame binding changes the live scalar without recapture. */
 assert(awstep_bind(p,op,0,0,sizeof(Args))==0);Args frame={buffer,&result,29};device=13;
 assert(awstep_execute(p,&frame,sizeof(frame),0)==0 && result==42 && kernels==2);checks++;
 before=calls;unsigned oldcopies=copies;
 assert(awstep_execute(p,&frame,sizeof(frame)-1,0)==BAD_ARGUMENT && calls==before && copies==oldcopies);checks++;
 assert(awstep_bind(p,op,0,0,sizeof(Args)-1)==BAD_ARGUMENT);checks++;
 /* Capture argument source is not read again; failure prevents downstream. */
 device=-1;unsigned oldkernels=kernels;
 assert(awstep_execute(p,&frame,sizeof(frame),0)==GUARD_FAILED && kernels==oldkernels && p->failed==op && result==42);checks++;
 assert(awstep_execute(p,&frame,sizeof(frame),0)==BAD_STATE);checks++;drop(p);
 /* Missing consumer acknowledgement refuses the entire replay in preflight. */
 p=start();device=2;assert(awstep_host_begin(p,4,0)==0);
 assert(wrap_cuMemcpyDtoHAsync_v2((DevicePtr)(uintptr_t)&original,(DevicePtr)(uintptr_t)&device,8,NULL)==0);
 assert(awstep_host_end(p,&hostop)==0);finish(p);oldcopies=copies;
 assert(awstep_execute(p,NULL,0,0)==BAD_ARGUMENT && copies==oldcopies);checks++;drop(p);
 /* Exactly one readback, no arbitrary host buffers, no open-scope callbacks. */
 p=start();assert(awstep_host_begin(p,4,0)==0);
 assert(awstep_native_host(p,(void*)helper,&args,sizeof(args),"fixture",&op)==BAD_STATE);checks++;
 assert(awstep_host_end(p,&hostop)==BAD_STATE);checks++;drop(p);
 p=start();Operation*o=append(H2D,"fixture",NULL);o->host=&original;o->count=8;
 assert(awstep_host_buffer(p,0,&buffer,&bytes)==BAD_ARGUMENT);checks++;
 assert(awstep_native_host(p,NULL,&args,sizeof(args),"fixture",&op)==BAD_ARGUMENT);checks++;
 assert(awstep_native_host(p,(void*)helper,NULL,sizeof(args),"fixture",&op)==BAD_ARGUMENT);checks++;
 assert(awstep_native_host(p,(void*)helper,&args,65537,"fixture",&op)==BAD_ARGUMENT);checks++;drop(p);
 p=start();assert(awstep_host_begin(p,4,0)==0);
 for(unsigned j=0;j<2;j++)assert(wrap_cuMemcpyDtoHAsync_v2((DevicePtr)(uintptr_t)&original,(DevicePtr)(uintptr_t)&device,8,NULL)==0);
 assert(awstep_host_end(p,&hostop)==BAD_STATE);checks++;drop(p);
 /* Capture-time failure reports the actual callback operation. */
 p=start();args.input=&device;device=-2;args.output=&result;
 assert(awstep_native_host(p,(void*)helper,&args,sizeof(args),"fixture",&op)==GUARD_FAILED && p->failed==op);checks++;drop(p);
 /* Operation predicates apply to compiled CPU work too. */
 p=start();device=1;args.delta=3;assert(awstep_predicate(p,4,8)==0);
 assert(awstep_native_host(p,(void*)helper,&args,sizeof(args),"fixture",&op)==0);
 assert(awstep_predicate(p,0,0)==0);finish(p);before=calls;
 assert(awstep_execute(p,NULL,0,0)==0 && calls==before);checks++;
 assert(awstep_execute(p,NULL,0,12)==0 && calls==before);checks++;
 assert(awstep_execute(p,NULL,0,4)==0 && calls==before+1);checks++;drop(p);
 printf("{\"status\":\"PASS\",\"cpu_integration_cases\":%u,\"cuda_calls\":0,\"scope\":\"actual recorder with CPU CUDA-call doubles\"}\n",checks);
 return 0;
}

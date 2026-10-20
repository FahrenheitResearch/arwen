/* SPDX-License-Identifier: Apache-2.0
 * CUDA launch recorder. Preload in one forecast child, never system-wide.
 * Replay submits ordinary cuLaunchKernel calls, with no CUDA graphs.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
typedef int Result;
typedef void *Handle;
typedef uint64_t DevicePtr;
typedef Result (*Launch)(Handle,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,Handle,void**,void**);
typedef Result (*Copy)(DevicePtr,DevicePtr,size_t,Handle);
typedef Result (*ParamInfo)(Handle,size_t,size_t*,size_t*);
typedef Result (*Context)(Handle*);
enum { KERNEL=1,D2D,H2D,D2H,FILL8,FILL16,FILL32,EVENT_RECORD,EVENT_WAIT,STREAM_SYNC,HOST_GUARD,HOST_OUTPUT,NATIVE_HOST,COUNT_PREDICATE };
enum { BAD_STATE=-1001,BAD_ARGUMENT=-1002,NO_MEMORY=-1003,UNSUPPORTED=-1004,WRONG_CONTEXT=-1005,BUSY=-1006,GUARD_FAILED=-1007 };
typedef struct { unsigned char *bytes; const unsigned char *host_source; size_t size,slot; } Argument;
typedef struct {
 unsigned type,dims[6],shared,value,flags;
 Handle function,stream,event,original_event; unsigned owns_event; void *call;
 DevicePtr dst,src; size_t count,nargs;
 void *host; unsigned guard_width,guard_mode,owns_host; uint64_t guard_mask,guard_expected;
 unsigned char *numeric_rules; size_t numeric_count; double numeric_threshold;
 Argument *args; void **params;
 uint64_t require,reject;
 char name[160],api[64];
} Operation;
typedef struct {
 Operation *ops; size_t size,capacity,kernels,failed;
 uint64_t require,reject; Handle context;
 int status,sealed,audit,host_policy; size_t scope_start,guard_index; uint64_t guard_value;
 unsigned guard_width,guard_mode; uint64_t guard_mask,guard_expected; char error[256]; atomic_flag running;
 unsigned char *numeric_rules; size_t numeric_count; double numeric_threshold;
 void (*error_hook)(int,size_t,const char*);
} Plan;
static _Thread_local Plan *recording;
static void *(*system_dlsym)(void*,const char*);
static pthread_once_t symbol_once=PTHREAD_ONCE_INIT;
static ParamInfo param_info,kernel_param_info;
static Context current_context;
static Result (*host_alloc)(void**,size_t,unsigned);
static Result (*host_free)(void*);
static Result (*event_create)(Handle*,unsigned),(*event_destroy)(Handle);
static Result (*stream_sync)(Handle),(*stream_sync_ptsz)(Handle);
static Result (*pointer_attribute)(void*,int,DevicePtr);
static Result (*function_name)(const char**,Handle);
static Result (*kernel_name)(const char**,Handle);
static void init_symbols(void) {
 system_dlsym=(void *(*)(void*,const char*))dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
 if(!system_dlsym)abort();
}
static void *symbol(void*h,const char*n) {
 pthread_once(&symbol_once,init_symbols); return system_dlsym(h,n);
}
static void fail(Plan*p,int code,const char*message) {
 if(p && !p->status){p->status=code;p->failed=p->size;snprintf(p->error,sizeof(p->error),"%s",message);
 if(p->error_hook)p->error_hook(code,p->failed,message);}
}
static Operation *append(unsigned type,const char*name,void*call) {
 Plan*p=recording; if(!p || p->status)return NULL;
 if(p->size==p->capacity){
  size_t n=p->capacity?p->capacity*2:256;
  Operation *v=realloc(p->ops,n*sizeof(*v));
  if(!v){fail(p,NO_MEMORY,"operation allocation failed");return NULL;}
  p->ops=v;p->capacity=n;
 }
 Operation*o=&p->ops[p->size++];memset(o,0,sizeof(*o));
 o->type=type;o->call=call;o->require=p->require;o->reject=p->reject;
 snprintf(o->name,sizeof(o->name),"%s",name);
 snprintf(o->api,sizeof(o->api),"%s",name);return o;
}
static void capture_launch(Launch call,const char*name,Handle f,
 unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,
 unsigned sh,Handle st,void**args,void**extra) {
 Plan*p=recording;if(!p)return;
 if(extra){fail(p,UNSUPPORTED,"packed extra launch arguments need an explicit ABI");return;}
 Operation*o=append(KERNEL,name,(void*)call);if(!o)return;
 o->function=f;o->stream=st;o->shared=sh;
 if(function_name){const char*label=NULL;Result nr=function_name(&label,f);
  if(nr==400 && kernel_name)nr=kernel_name(&label,f);
  if(!nr && label)snprintf(o->name,sizeof(o->name),"%s",label);}
 unsigned dims[]={gx,gy,gz,bx,by,bz};memcpy(o->dims,dims,sizeof(dims));
 if(!param_info){fail(p,UNSUPPORTED,"cuFuncGetParamInfo unavailable");return;}
 size_t off,size,n=0;
 ParamInfo layout=param_info;
 Result first=layout(f,0,&off,&size);
 if(first==400 && kernel_param_info)layout=kernel_param_info;
 for(;n<1024;n++){
  Result r=layout(f,n,&off,&size);
  if(r==1)break;
  if(r){fail(p,r,"cuFuncGetParamInfo failed");return;}
 }
 if(n==1024){fail(p,UNSUPPORTED,"too many kernel arguments");return;}
 o->nargs=n;o->args=calloc(n,sizeof(Argument));o->params=calloc(n,sizeof(void*));
 if(n && (!o->args || !o->params)){fail(p,NO_MEMORY,"parameter allocation failed");return;}
 for(size_t i=0;i<n;i++){
  Result r=layout(f,i,&off,&size);
  if(r || !size || size>65536 || !args || !args[i]){fail(p,UNSUPPORTED,"invalid parameter layout");return;}
  Argument*a=&o->args[i];a->size=size;a->slot=SIZE_MAX;a->bytes=malloc(size);
  if(!a->bytes){fail(p,NO_MEMORY,"argument allocation failed");return;}
  memcpy(a->bytes,args[i],size);o->params[i]=a->bytes;
 }
 p->kernels++;
}
#define LAUNCH(tag) \
static _Atomic(Launch) real_##tag; \
static Result wrap_##tag(Handle f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sh,Handle st,void**a,void**x){ \
 capture_launch(real_##tag,#tag,f,gx,gy,gz,bx,by,bz,sh,st,a,x); \
 return real_##tag(f,gx,gy,gz,bx,by,bz,sh,st,a,x);}
LAUNCH(cuLaunchKernel)
LAUNCH(cuLaunchKernel_ptsz)
static void capture_host(Operation*o,unsigned kind,DevicePtr source) {
 Plan*p=recording;
 if(!p || !o || (kind!=H2D && kind!=D2H))return;
 const char*message=kind==H2D?"host upload needs retained parameter source":"host readback needs native consumer";
 if((kind==H2D && p->host_policy!=1 && p->host_policy!=3 && p->host_policy!=5) || (kind==D2H && p->host_policy!=2 && p->host_policy!=4)){
  if(p->audit){if(p->error_hook)p->error_hook(UNSUPPORTED,p->size-1,message);}
  else fail(p,UNSUPPORTED,message);
  return;
 }
 if(!host_alloc || !host_free){fail(p,UNSUPPORTED,"pinned host allocation API unavailable");return;}
 Result r=host_alloc(&o->host,o->count?o->count:1,0);
 if(r){fail(p,r,"retained host allocation failed");return;}
 o->owns_host=1;
 if(kind==H2D && p->host_policy==5)o->flags=2;
 if(kind==H2D && o->count)memcpy(o->host,(void*)(uintptr_t)source,o->count);
}
#define COPY(tag,kind) \
static _Atomic(Copy) real_##tag; \
static Result wrap_##tag(DevicePtr d,DevicePtr s,size_t n,Handle st){ \
 Operation*o=append(kind,#tag,(void*)real_##tag);if(o){o->dst=d;o->src=s;o->count=n;o->stream=st;capture_host(o,kind,s);} \
 Result r=real_##tag(d,s,n,st); \
 /* The capture-only second read gives the native guard its own lifetime. */ \
 if(!r && o && kind==D2H && o->host){r=real_##tag((DevicePtr)(uintptr_t)o->host,s,n,st);if(r)fail(recording,r,"guard capture readback failed");} \
 return r;}
COPY(cuMemcpyDtoDAsync_v2,D2D)
COPY(cuMemcpyDtoDAsync_v2_ptsz,D2D)
COPY(cuMemcpyHtoDAsync_v2,H2D)
COPY(cuMemcpyHtoDAsync_v2_ptsz,H2D)
COPY(cuMemcpyDtoHAsync_v2,D2H)
COPY(cuMemcpyDtoHAsync_v2_ptsz,D2H)
/* cuMemcpyAsync has no _v2 ABI. Record it only for two device pointers. */
#define COPY_UNIFIED(tag) \
static _Atomic(Copy) real_##tag; \
static Result wrap_##tag(DevicePtr d,DevicePtr s,size_t n,Handle st){ \
 if(recording){unsigned dt=0,stt=0; \
 if(!pointer_attribute || pointer_attribute(&dt,2,d) || pointer_attribute(&stt,2,s) || dt!=2 || stt!=2) \
 fail(recording,UNSUPPORTED,#tag " requires device-only pointers"); \
 else {Operation*o=append(D2D,#tag,(void*)real_##tag);if(o){o->dst=d;o->src=s;o->count=n;o->stream=st;}}} \
 return real_##tag(d,s,n,st);}
COPY_UNIFIED(cuMemcpyAsync)
COPY_UNIFIED(cuMemcpyAsync_ptsz)

#define SET(tag,typ,kind) \
typedef Result(*fn_##tag)(DevicePtr,typ,size_t,Handle); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(DevicePtr d,typ v,size_t n,Handle st){ \
 Operation*o=append(kind,#tag,(void*)real_##tag);if(o){o->dst=d;o->value=v;o->count=n;o->stream=st;}return real_##tag(d,v,n,st);}
SET(cuMemsetD8Async,unsigned char,FILL8)
SET(cuMemsetD16Async,unsigned short,FILL16)
SET(cuMemsetD32Async,unsigned,FILL32)
SET(cuMemsetD8Async_ptsz,unsigned char,FILL8)
SET(cuMemsetD16Async_ptsz,unsigned short,FILL16)
SET(cuMemsetD32Async_ptsz,unsigned,FILL32)
/* Private upload events have no public observers. CuPy may destroy its
 * bookkeeping event after capture; replay therefore owns a separate event.
 * Other events retain the explicit caller-owned lifetime contract. */
static void capture_event(Operation*o,Handle e,Handle st) {
 if(!o)return;
 o->event=e;o->original_event=e;o->stream=st;
 if(recording->host_policy==3 || recording->host_policy==5){
  if(!event_create || !event_destroy){fail(recording,UNSUPPORTED,"private event API unavailable");return;}
  Handle owned=NULL;Result rc=event_create(&owned,2); /* CU_EVENT_DISABLE_TIMING */
  if(rc){fail(recording,rc,"private upload event allocation failed");return;}
  o->event=owned;o->owns_event=1;
 } else {
  for(size_t i=0;i+1<recording->size;i++)if(recording->ops[i].owns_event && recording->ops[i].original_event==e){
   fail(recording,UNSUPPORTED,"private upload event escaped its scope");return;
  }
 }
}
static void capture_wait(Operation*o,Handle st,Handle e,unsigned flags) {
 if(!o)return;
 o->stream=st;o->event=e;o->flags=flags;
 if(recording->host_policy==3 || recording->host_policy==5){fail(recording,UNSUPPORTED,"private upload scope cannot wait on events");return;}
 for(size_t i=0;i+1<recording->size;i++)if(recording->ops[i].owns_event && recording->ops[i].original_event==e){
  fail(recording,UNSUPPORTED,"private upload event has an external waiter");return;
 }
}
#define EVENT(tag) \
typedef Result(*fn_##tag)(Handle,Handle); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(Handle e,Handle st){Operation*o=append(EVENT_RECORD,#tag,(void*)real_##tag);capture_event(o,e,st);return real_##tag(e,st);}
EVENT(cuEventRecord)
EVENT(cuEventRecord_ptsz)
#define WAIT(tag) \
typedef Result(*fn_##tag)(Handle,Handle,unsigned); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(Handle st,Handle e,unsigned f){Operation*o=append(EVENT_WAIT,#tag,(void*)real_##tag);capture_wait(o,st,e,f);return real_##tag(st,e,f);}
WAIT(cuStreamWaitEvent)
WAIT(cuStreamWaitEvent_ptsz)
#define SYNC(tag) \
typedef Result(*fn_##tag)(Handle); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(Handle st){Operation*o=append(STREAM_SYNC,#tag,(void*)real_##tag);if(o)o->stream=st;return real_##tag(st);}
SYNC(cuStreamSynchronize)
SYNC(cuStreamSynchronize_ptsz)

/* Refuse additional submission APIs instead of silently omitting work. */
static void unsupported(const char*name){
 if(recording)fail(recording,UNSUPPORTED,name);
}
#define COPY_SYNC(tag) \
typedef Result(*fn_##tag)(DevicePtr,DevicePtr,size_t); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(DevicePtr d,DevicePtr s,size_t n){unsupported(#tag " needs explicit recording ABI");return real_##tag(d,s,n);}
COPY_SYNC(cuMemcpyDtoD_v2)
COPY_SYNC(cuMemcpyHtoD_v2)
COPY_SYNC(cuMemcpyDtoH_v2)
COPY_SYNC(cuMemcpy)
#define STRUCT_ASYNC(tag) \
typedef Result(*fn_##tag)(const void*,Handle); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(const void*d,Handle st){unsupported(#tag " needs explicit recording ABI");return real_##tag(d,st);}
STRUCT_ASYNC(cuMemcpy2DAsync_v2)
STRUCT_ASYNC(cuMemcpy2DAsync_v2_ptsz)
STRUCT_ASYNC(cuMemcpy3DAsync_v2)
STRUCT_ASYNC(cuMemcpy3DAsync_v2_ptsz)
#define LAUNCH_EX(tag) \
typedef Result(*fn_##tag)(const void*,Handle,void**,void**); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(const void*c,Handle f,void**p,void**x){unsupported(#tag " is outside ordinary launch ABI");return real_##tag(c,f,p,x);}
LAUNCH_EX(cuLaunchKernelEx)
LAUNCH_EX(cuLaunchKernelEx_ptsz)
#define COOPERATIVE(tag) \
typedef Result(*fn_##tag)(Handle,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,Handle,void**); \
static _Atomic(fn_##tag) real_##tag; \
static Result wrap_##tag(Handle f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sh,Handle st,void**a){unsupported(#tag " needs explicit recording ABI");return real_##tag(f,gx,gy,gz,bx,by,bz,sh,st,a);}
COOPERATIVE(cuLaunchCooperativeKernel)
COOPERATIVE(cuLaunchCooperativeKernel_ptsz)

static void *intercept(const char*,void*);
typedef Result(*GetProc)(const char*,void**,int,uint64_t);
typedef Result(*GetProcV2)(const char*,void**,int,uint64_t,void*);
static _Atomic(GetProc) real_getproc;
static _Atomic(GetProcV2) real_getproc_v2;
static void intercept_proc(const char*n,void**p,uint64_t flags){
 char name[256];const char*q=n;
 if((flags&2) && (strstr(n,"Async") || !strcmp(n,"cuLaunchKernel") ||
    !strcmp(n,"cuEventRecord") || !strcmp(n,"cuStreamWaitEvent") || !strcmp(n,"cuStreamSynchronize"))){
  snprintf(name,sizeof(name),"%s_ptsz",n);q=name;
 }
 if(!strcmp(n,"cuMemcpyDtoDAsync") || !strcmp(n,"cuMemcpyHtoDAsync") || !strcmp(n,"cuMemcpyDtoHAsync") ||
    !strcmp(n,"cuMemcpy2DAsync") || !strcmp(n,"cuMemcpy3DAsync") ||
    !strcmp(n,"cuMemcpyDtoD") || !strcmp(n,"cuMemcpyHtoD") || !strcmp(n,"cuMemcpyDtoH")){
  snprintf(name,sizeof(name),"%s_v2%s",n,(flags&2)?"_ptsz":"");q=name;
 }
 *p=intercept(q,*p);
}
static Result wrap_getproc(const char*n,void**p,int v,uint64_t flags){
 Result r=real_getproc(n,p,v,flags);if(!r && *p)intercept_proc(n,p,flags);return r;
}
static Result wrap_getproc_v2(const char*n,void**p,int v,uint64_t flags,void*status){
 Result r=real_getproc_v2(n,p,v,flags,status);if(!r && *p)intercept_proc(n,p,flags);return r;
}
#define MAP(tag) if(!strcmp(n,#tag)){if(p==(void*)wrap_##tag)return p;real_##tag=(__typeof__(real_##tag))p;return(void*)wrap_##tag;}
static void *intercept(const char*n,void*p){
 if(!p)return p;
 MAP(cuLaunchKernel) MAP(cuLaunchKernel_ptsz)
 MAP(cuMemcpyAsync) MAP(cuMemcpyAsync_ptsz) MAP(cuMemcpy)
 MAP(cuMemcpyDtoD_v2) MAP(cuMemcpyHtoD_v2) MAP(cuMemcpyDtoH_v2)
 MAP(cuMemcpy2DAsync_v2) MAP(cuMemcpy2DAsync_v2_ptsz)
 MAP(cuMemcpy3DAsync_v2) MAP(cuMemcpy3DAsync_v2_ptsz)
 MAP(cuLaunchKernelEx) MAP(cuLaunchKernelEx_ptsz)
 MAP(cuLaunchCooperativeKernel) MAP(cuLaunchCooperativeKernel_ptsz)
 MAP(cuMemcpyDtoDAsync_v2) MAP(cuMemcpyDtoDAsync_v2_ptsz)
 MAP(cuMemcpyHtoDAsync_v2) MAP(cuMemcpyHtoDAsync_v2_ptsz)
 MAP(cuMemcpyDtoHAsync_v2) MAP(cuMemcpyDtoHAsync_v2_ptsz)
 MAP(cuMemsetD8Async) MAP(cuMemsetD16Async) MAP(cuMemsetD32Async)
 MAP(cuMemsetD8Async_ptsz) MAP(cuMemsetD16Async_ptsz) MAP(cuMemsetD32Async_ptsz)
 MAP(cuEventRecord) MAP(cuEventRecord_ptsz) MAP(cuStreamWaitEvent) MAP(cuStreamWaitEvent_ptsz)
 MAP(cuStreamSynchronize) MAP(cuStreamSynchronize_ptsz)
 if(!strcmp(n,"cuGetProcAddress")){real_getproc=(GetProc)p;return(void*)wrap_getproc;}
 if(!strcmp(n,"cuGetProcAddress_v2")){real_getproc_v2=(GetProcV2)p;return(void*)wrap_getproc_v2;}
 return p;
}
void *dlsym(void*h,const char*n){return intercept(n,symbol(h,n));}
Plan *awstep_create(void){Plan*p=calloc(1,sizeof(*p));if(p)atomic_flag_clear(&p->running);return p;}
int awstep_audit(Plan*p){if(!p || p->sealed || recording==p)return BAD_STATE;p->audit=1;return 0;}
int awstep_error_hook(Plan*p,void(*hook)(int,size_t,const char*)){
 if(!p || p->sealed || recording==p)return BAD_STATE;
 p->error_hook=hook;return 0;
}
static pthread_once_t driver_once=PTHREAD_ONCE_INIT;
static void init_driver(void){
 void*d=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
 if(!d)return;
 host_alloc=(__typeof__(host_alloc))symbol(d,"cuMemHostAlloc");
 host_free=(__typeof__(host_free))symbol(d,"cuMemFreeHost");
 event_create=(__typeof__(event_create))symbol(d,"cuEventCreate");
 event_destroy=(__typeof__(event_destroy))symbol(d,"cuEventDestroy_v2");
 stream_sync=(__typeof__(stream_sync))symbol(d,"cuStreamSynchronize");
 stream_sync_ptsz=(__typeof__(stream_sync_ptsz))symbol(d,"cuStreamSynchronize_ptsz");
 function_name=(__typeof__(function_name))symbol(d,"cuFuncGetName");
 kernel_name=(__typeof__(kernel_name))symbol(d,"cuKernelGetName");
 kernel_param_info=(ParamInfo)symbol(d,"cuKernelGetParamInfo");
 pointer_attribute=(__typeof__(pointer_attribute))symbol(d,"cuPointerGetAttribute");
 param_info=(ParamInfo)symbol(d,"cuFuncGetParamInfo");current_context=(Context)symbol(d,"cuCtxGetCurrent");
}
int awstep_begin(Plan*p){
 if(!p || recording || p->size || p->sealed)return BAD_STATE;
 pthread_once(&driver_once,init_driver);
 if(!current_context || current_context(&p->context) || !p->context){fail(p,WRONG_CONTEXT,"recording needs current CUDA context");return p->status;}
 recording=p;return 0;
}
int awstep_end(Plan*p){
 if(recording!=p)return BAD_STATE;
 if(p->host_policy)fail(p,BAD_STATE,"host contract scope left open");
 recording=NULL;p->sealed=1;if(p->audit)fail(p,UNSUPPORTED,"diagnostic capture cannot replay");if(!p->kernels)fail(p,UNSUPPORTED,"no kernel launches intercepted");return p->status;
}
int awstep_predicate(Plan*p,uint64_t require,uint64_t reject){
 if(recording!=p || p->host_policy || (require&reject))return BAD_STATE;
 p->require=require;p->reject=reject;return 0;
}
int awstep_bind(Plan*p,size_t op,size_t arg,size_t offset,size_t size){
 if(!p || !p->sealed || p->status || op>=p->size)return BAD_STATE;
 Operation*o=&p->ops[op];if((o->type!=KERNEL && o->type!=NATIVE_HOST) || arg>=o->nargs || o->args[arg].size!=size || offset>SIZE_MAX-size)return BAD_ARGUMENT;
 o->args[arg].slot=offset;o->args[arg].host_source=NULL;return 0;
}
/* A current synchronized scalar readback can feed a later kernel argument
 * directly. Binding checks ownership, order, width and predicate coverage. */
int awstep_bind_host_argument(Plan*p,size_t op,size_t arg,size_t source,size_t offset){
 if(!p || !p->sealed || p->status || op>=p->size || source>=op)return BAD_STATE;
 Operation*o=&p->ops[op],*input=&p->ops[source];
 if(o->type!=KERNEL || arg>=o->nargs || input->type!=HOST_OUTPUT || !input->host ||
    offset>input->count || o->args[arg].size>input->count-offset ||
    (o->require&input->require)!=input->require || (o->reject&input->reject)!=input->reject)return BAD_ARGUMENT;
 for(size_t i=source+1;i<op;i++)if(p->ops[i].type==COUNT_PREDICATE &&
    (p->ops[i].guard_mask&(input->require|input->reject)))return BAD_ARGUMENT;
 o->args[arg].host_source=(const unsigned char*)input->host+offset;o->args[arg].slot=SIZE_MAX;
 input->flags=1;return 0;
}

/* Explicit compiled CPU work, using a fixed argument ABI. The caller retains
 * all pointed-to storage and the shared library. No Python trampoline, CUDA
 * calls or hidden stream dependency is permitted in the helper. */
int awstep_native_host(Plan*p,void*function,const void*arguments,size_t bytes,const char*name,size_t*index){
 if(!p || recording!=p || p->status || p->host_policy)return BAD_STATE;
 if(!function || !arguments || !bytes || bytes>65536 || !name || !index)return BAD_ARGUMENT;
 Operation*o=append(NATIVE_HOST,name,function);if(!o)return p->status;
 o->nargs=1;o->args=calloc(1,sizeof(Argument));
 if(!o->args){fail(p,NO_MEMORY,"native host argument allocation failed");return p->status;}
 o->args[0].size=bytes;o->args[0].slot=SIZE_MAX;o->args[0].bytes=malloc(bytes);
 if(!o->args[0].bytes){fail(p,NO_MEMORY,"native host argument allocation failed");return p->status;}
 memcpy(o->args[0].bytes,arguments,bytes);*index=p->size-1;
 int rc=((Result(*)(const void*,size_t))function)(arguments,bytes);
 if(rc){fail(p,rc,"native host function failed during capture");p->failed=*index;}
 return rc;
}

/* A native producer can select fixed-capacity groups from a current count.
 * The caller retains count storage and records this after its producer. */
int awstep_count_predicate(Plan*p,const uint64_t*count,uint64_t threshold,unsigned bit){
 if(!p || recording!=p || p->status || p->host_policy)return BAD_STATE;
 if(!count || bit>=64)return BAD_ARGUMENT;
 Operation*o=append(COUNT_PREDICATE,"host_count_predicate",NULL);if(!o)return p->status;
 o->host=(void*)count;o->guard_mask=UINT64_C(1)<<bit;o->guard_expected=threshold;return 0;
}

/* Rules consume the same binary64 values produced by deferred checks.
 * 1: numeric nonzero; 2: below threshold, excluding negative infinity.
 * IEEE comparisons intentionally preserve signed-zero and NaN semantics. */
static int numeric_bad(uint64_t bits,unsigned rule,double threshold){
 double value;memcpy(&value,&bits,sizeof(value));
 return rule==1?value!=0.0:(bits!=UINT64_C(0xfff0000000000000) && value<threshold);
}
static int check_guard(Plan*p,Operation*o,size_t index){
 const unsigned char*data=o->host;
 for(size_t j=0;j<o->count/o->guard_width;j++){
  uint64_t value=0;memcpy(&value,data+j*o->guard_width,o->guard_width);
  int bad=value!=0;
  if(o->guard_mode==4){
   bad=numeric_bad(value,o->numeric_rules[j],o->numeric_threshold);
  }else if(o->guard_mode){
   if(j>=64 || !(o->guard_mask & (UINT64_C(1)<<j)))continue;
   if(o->guard_mode==1){
    uint64_t exponent=o->guard_width==4?UINT64_C(0x7f800000):UINT64_C(0x7ff0000000000000);
    bad=(value&exponent)==exponent;
   }else if(o->guard_mode==3){bad=value!=o->guard_expected;
   }
  }
  if(bad){p->guard_index=j;p->guard_value=value;
   fail(p,GUARD_FAILED,"native status guard failed before downstream submission");p->failed=index;return GUARD_FAILED;}
 }
 return 0;
}
int awstep_host_begin(Plan*p,unsigned policy,unsigned width){
 if(recording!=p || p->status || p->host_policy || (policy!=1 && policy!=2 && policy!=3 && policy!=4 && policy!=5) ||
    (policy==2 && width!=1 && width!=4 && width!=8))return BAD_STATE;
 p->host_policy=policy;p->guard_width=width;p->guard_mode=0;p->guard_mask=0;p->scope_start=p->size;return 0;
}
int awstep_finite_begin(Plan*p,unsigned width,uint64_t mask){
 if((width!=4 && width!=8) || !mask)return BAD_ARGUMENT;
 int rc=awstep_host_begin(p,2,width);if(rc)return rc;
 p->guard_mode=1;p->guard_mask=mask;return 0;
}
int awstep_masked_zero_begin(Plan*p,unsigned width,uint64_t mask){
 if(!mask)return BAD_ARGUMENT;
 int rc=awstep_host_begin(p,2,width);if(rc)return rc;
 p->guard_mode=2;p->guard_mask=mask;return 0;
}
int awstep_value_begin(Plan*p,unsigned width,uint64_t mask,uint64_t expected){
 if((width!=1 && width!=4 && width!=8) || !mask || (width<8 && expected>>(width*8)))return BAD_ARGUMENT;
 int rc=awstep_host_begin(p,2,width);if(rc)return rc;
 p->guard_mode=3;p->guard_mask=mask;p->guard_expected=expected;return 0;
}
int awstep_numeric_begin(Plan*p,const unsigned char*rules,size_t count,double threshold){
 if(!rules || !count || count>SIZE_MAX/8)return BAD_ARGUMENT;
 uint64_t bits;memcpy(&bits,&threshold,sizeof(bits));
 if((bits&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000))return BAD_ARGUMENT;
 for(size_t j=0;j<count;j++)if(rules[j]!=1 && rules[j]!=2)return BAD_ARGUMENT;
 int rc=awstep_host_begin(p,2,8);if(rc)return rc;
 p->numeric_rules=malloc(count);
 if(!p->numeric_rules){p->host_policy=0;fail(p,NO_MEMORY,"numeric rule allocation failed");return p->status;}
 memcpy(p->numeric_rules,rules,count);p->numeric_count=count;p->numeric_threshold=threshold;p->guard_mode=4;
 return 0;
}
int awstep_host_end(Plan*p,size_t*guard);
/* A library may choose a device-only path with no descriptor upload.
 * This opt-in applies only to immutable-upload scopes, never readbacks. */
int awstep_host_end_optional(Plan*p,size_t*guard){
 if(recording!=p || !guard || (p->host_policy!=1 && p->host_policy!=3))return BAD_STATE;
 if(p->status)return awstep_host_end(p,guard);
 for(size_t i=p->scope_start;i<p->size;i++)if(p->ops[i].type==H2D)return awstep_host_end(p,guard);
 p->host_policy=0;*guard=SIZE_MAX;return 0;
}
int awstep_host_end(Plan*p,size_t*guard){
 if(recording!=p || !p->host_policy || !guard)return BAD_STATE;
 unsigned policy=p->host_policy;p->host_policy=0;*guard=SIZE_MAX;
 if(p->status)return p->status;
 size_t transfers=0,index=0;
 for(size_t i=p->scope_start;i<p->size;i++)if(p->ops[i].type==((policy==2 || policy==4)?D2H:H2D)){transfers++;index=i;}
 if(!transfers || ((policy==2 || policy==4 || policy==5) && transfers!=1)){fail(p,BAD_STATE,"host scope transfer count mismatch");return p->status;}
 if(policy!=2 && policy!=4 && policy!=5)return 0;
 Operation copy=p->ops[index];
 if(policy==2 && copy.count%p->guard_width){fail(p,BAD_ARGUMENT,"guard width does not divide readback size");return p->status;}
 if(p->guard_mode==4){
  if(copy.count/8!=p->numeric_count){fail(p,BAD_ARGUMENT,"numeric rule count differs from result words");return p->status;}
 }else if(p->guard_mode){
  size_t words=copy.count/p->guard_width;
  if(words>64 || (words<64 && (p->guard_mask>>words))){fail(p,BAD_ARGUMENT,"guard mask exceeds result words");return p->status;}
 }
 int synced=0;
 for(size_t i=index+1;i<p->size;i++)if(p->ops[i].type==STREAM_SYNC && p->ops[i].stream==copy.stream)synced=1;
 if(!synced){
  Result(*sync)(Handle)=strstr(copy.api,"_ptsz")?stream_sync_ptsz:stream_sync;
  if(!sync){fail(p,UNSUPPORTED,"guard stream synchronization unavailable");return p->status;}
  Result rc=sync(copy.stream);if(rc){fail(p,rc,"guard capture synchronization failed");return rc;}
  Operation*wait=append(STREAM_SYNC,"guard_stream_sync",(void*)sync);if(!wait)return p->status;wait->stream=copy.stream;
 }
 /* A native producer may overwrite this buffer on the next submission.
  * Complete its H2D before returning, including when no later GPU sync exists. */
 if(policy==5){*guard=index;return 0;}
 Operation*o=append(policy==4?HOST_OUTPUT:HOST_GUARD,policy==4?"host_output":p->guard_mode==4?"host_numeric_guard":p->guard_mode==1?"host_finite_guard":p->guard_mode==3?"host_value_guard":"host_zero_guard",NULL);if(!o)return p->status;
 o->host=copy.host;o->count=copy.count;o->guard_width=p->guard_width;o->guard_mode=p->guard_mode;o->guard_mask=p->guard_mask;o->guard_expected=p->guard_expected;
 if(p->guard_mode==4){
  o->numeric_rules=p->numeric_rules;o->numeric_count=p->numeric_count;o->numeric_threshold=p->numeric_threshold;
  p->numeric_rules=NULL;p->numeric_count=0;
 }
 /* Guard and readback must share a predicate, even if callers nest groups. */
 o->require=copy.require;o->reject=copy.reject;
 *guard=p->size-1;return policy==4?0:check_guard(p,o,*guard);
}
/* Borrow synchronized readback or explicit native-upload storage. Retain
 * the plan and preserve producer/consumer order and predicate compatibility. */
int awstep_host_buffer(Plan*p,size_t op,void**buffer,size_t*bytes){
 if(!p || p->status || op>=p->size || !buffer || !bytes)return BAD_ARGUMENT;
 Operation*o=&p->ops[op];
 if(!o->host || (o->type!=HOST_OUTPUT && !(o->type==H2D && (o->flags==2 || o->flags==3))))return BAD_ARGUMENT;
 o->flags=o->type==H2D?3:1;*buffer=o->host;*bytes=o->count;return 0;
}
size_t awstep_guard_index(Plan*p){return p?p->guard_index:SIZE_MAX;}
uint64_t awstep_guard_value(Plan*p){return p?p->guard_value:0;}
int awstep_execute(Plan*p,void*frame,size_t bytes,uint64_t flags){
 if(!p || !p->sealed || p->status)return BAD_STATE;
 if(atomic_flag_test_and_set(&p->running))return BUSY;
 Handle ctx=NULL;int rc=current_context(&ctx);
 if(rc || ctx!=p->context){atomic_flag_clear(&p->running);return WRONG_CONTEXT;}
 for(size_t i=0;i<p->size;i++)if((p->ops[i].type==HOST_OUTPUT && !p->ops[i].flags) ||
    (p->ops[i].type==H2D && p->ops[i].flags==2)){
  atomic_flag_clear(&p->running);return BAD_ARGUMENT;
 }
 for(size_t i=0;i<p->size;i++)for(size_t j=0;j<p->ops[i].nargs;j++){
  Argument*a=&p->ops[i].args[j];
  if(a->slot!=SIZE_MAX && (!frame || a->slot>bytes || a->size>bytes-a->slot)){atomic_flag_clear(&p->running);return BAD_ARGUMENT;}
 }
 for(size_t i=0;i<p->size;i++){
  Operation*o=&p->ops[i];if((flags&o->require)!=o->require || (flags&o->reject))continue;
  switch(o->type){
   case KERNEL:
    for(size_t j=0;j<o->nargs;j++)o->params[j]=o->args[j].host_source?(void*)o->args[j].host_source:(o->args[j].slot==SIZE_MAX?o->args[j].bytes:(unsigned char*)frame+o->args[j].slot);
    rc=((Launch)o->call)(o->function,o->dims[0],o->dims[1],o->dims[2],o->dims[3],o->dims[4],o->dims[5],o->shared,o->stream,o->params,NULL);break;
   case H2D:rc=((Copy)o->call)(o->dst,(DevicePtr)(uintptr_t)o->host,o->count,o->stream);break;
   case D2H:rc=((Copy)o->call)((DevicePtr)(uintptr_t)o->host,o->src,o->count,o->stream);break;
   case HOST_GUARD:rc=check_guard(p,o,i);break;
   case HOST_OUTPUT:rc=0;break;
   case COUNT_PREDICATE:{
    uint64_t count;memcpy(&count,o->host,sizeof(count));
    if(count>o->guard_expected)flags|=o->guard_mask;else flags&=~o->guard_mask;
    rc=0;break;
   }
   case NATIVE_HOST:
    rc=((Result(*)(const void*,size_t))o->call)(o->args[0].slot==SIZE_MAX?o->args[0].bytes:(unsigned char*)frame+o->args[0].slot,o->args[0].size);break;
   case D2D:rc=((Copy)o->call)(o->dst,o->src,o->count,o->stream);break;
   case FILL8:rc=((Result(*)(DevicePtr,unsigned char,size_t,Handle))o->call)(o->dst,o->value,o->count,o->stream);break;
   case FILL16:rc=((Result(*)(DevicePtr,unsigned short,size_t,Handle))o->call)(o->dst,o->value,o->count,o->stream);break;
   case FILL32:rc=((Result(*)(DevicePtr,unsigned,size_t,Handle))o->call)(o->dst,o->value,o->count,o->stream);break;
   case EVENT_RECORD:rc=((Result(*)(Handle,Handle))o->call)(o->event,o->stream);break;
   case EVENT_WAIT:rc=((Result(*)(Handle,Handle,unsigned))o->call)(o->stream,o->event,o->flags);break;
   case STREAM_SYNC:rc=((Result(*)(Handle))o->call)(o->stream);break;
   default:rc=UNSUPPORTED;
  }
  if(rc){fail(p,rc,"replay driver call failed; plan invalid after partial submission");p->failed=i;break;}
 }
 atomic_flag_clear(&p->running);return rc;
}
size_t awstep_count(Plan*p){return p?p->size:0;}
size_t awstep_kernel_count(Plan*p){return p?p->kernels:0;}
size_t awstep_failed(Plan*p){return p?p->failed:0;}
const char *awstep_error(Plan*p){return p?p->error:"null plan";}
int awstep_op(Plan*p,size_t i,unsigned*type,size_t*nargs,char*name,size_t n){
 if(!p || i>=p->size || !type || !nargs || !name || !n)return BAD_ARGUMENT;
 *type=p->ops[i].type;*nargs=p->ops[i].nargs;snprintf(name,n,"%s",p->ops[i].name);return 0;
}
int awstep_api(Plan*p,size_t i,char*out,size_t n){
 if(!p || i>=p->size || !out || !n)return BAD_ARGUMENT;
 snprintf(out,n,"%s",p->ops[i].api);return 0;
}
size_t awstep_transfer_bytes(Plan*p,size_t i){return p && i<p->size?p->ops[i].count:0;}
size_t awstep_arg(Plan*p,size_t i,size_t j,void*out,size_t cap){
 if(!p || i>=p->size || j>=p->ops[i].nargs)return 0;
 Argument*a=&p->ops[i].args[j];if(out && cap>=a->size)memcpy(out,a->bytes,a->size);return a->size;
}
int awstep_event_owned(Plan*p,size_t i){return p && i<p->size && p->ops[i].owns_event;}
void awstep_destroy(Plan*p){
 if(!p || recording==p)return;
 for(size_t i=0;i<p->size;i++){
  if(p->ops[i].args)for(size_t j=0;j<p->ops[i].nargs;j++)free(p->ops[i].args[j].bytes);
  free(p->ops[i].args);free(p->ops[i].params);free(p->ops[i].numeric_rules);
  if(p->ops[i].owns_host && host_free)host_free(p->ops[i].host);
  if(p->ops[i].owns_event && event_destroy)event_destroy(p->ops[i].event);
 }
 free(p->numeric_rules);free(p->ops);free(p);
}

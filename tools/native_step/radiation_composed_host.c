#include <stddef.h>
/* Exact existing compiled ozone ABI, with unchanged float32 weights. */
typedef int(*Ozone)(const float*,size_t,const float*,size_t,const float*,size_t,size_t,size_t,size_t,float,float,size_t,float*);
typedef struct {Ozone function;const float*values;size_t nvalues;const float*latitude;size_t nlatitude;const float*ozone;size_t levels,months,nm,np;float fact1,fact2;size_t workers;float*out;} OzoneArgs;
int host_ozone(const void*pointer,size_t bytes){
 if(bytes!=sizeof(OzoneArgs))return -1020;
 const OzoneArgs*a=pointer;
 if(!a->function || !a->values || !a->latitude || !a->ozone || !a->out || a->workers<1)return -1021;
 return a->function(a->values,a->nvalues,a->latitude,a->nlatitude,a->ozone,a->levels,a->months,a->nm,a->np,a->fact1,a->fact2,a->workers,a->out);
}

#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <omp.h>
typedef struct {const float*coszen;size_t ncols,capacity,workers;int64_t*day;int64_t*night;uint64_t*counts;size_t*work;} Daylight;
/* Stable compaction of the two original NumPy comparisons. No solar math. */
int host_daylight(const void*pointer,size_t bytes){
 if(bytes!=sizeof(Daylight))return -1020;
 const Daylight*a=pointer;
 if(!a->coszen || !a->day || !a->night || !a->counts || !a->work || !a->workers || a->workers>64 || a->capacity<a->ncols || a->capacity>INT64_MAX)return -1021;
 const size_t nt=a->workers,n=a->ncols;
 #pragma omp parallel for num_threads(nt) schedule(static)
 for(size_t t=0;t<nt;t++){
  size_t dc=0,nc=0,start=(n/nt)*t+(n%nt)*t/nt,end=(n/nt)*(t+1)+(n%nt)*(t+1)/nt;
  for(size_t i=start;i<end;i++){float x=a->coszen[i];dc+=x>0.0f;nc+=x<=0.0f;}
  a->work[2*t]=dc;a->work[2*t+1]=nc;
 }
 size_t day=0,night=0;
 for(size_t t=0;t<nt;t++){size_t dc=a->work[2*t],nc=a->work[2*t+1];a->work[2*t]=day;a->work[2*t+1]=night;day+=dc;night+=nc;}
 #pragma omp parallel for num_threads(nt) schedule(static)
 for(size_t t=0;t<nt;t++){
  size_t di=a->work[2*t],ni=a->work[2*t+1],start=(n/nt)*t+(n%nt)*t/nt,end=(n/nt)*(t+1)+(n%nt)*(t+1)/nt;
  for(size_t i=start;i<end;i++){float x=a->coszen[i];if(x>0.0f)a->day[di++]=(int64_t)i;if(x<=0.0f)a->night[ni++]=(int64_t)i;}
 }
 int64_t pad=day?a->day[0]:0;
 #pragma omp parallel for num_threads(nt) schedule(static)
 for(size_t i=day;i<a->capacity;i++)a->day[i]=pad;
 a->counts[0]=day;a->counts[1]=night;return 0;
}

#include <string.h>
typedef struct {OzoneArgs base;float*upload;} OzoneUpload;
int host_ozone_upload(const void*pointer,size_t bytes){
 if(bytes!=sizeof(OzoneUpload))return -1020;
 const OzoneUpload*a=pointer;
 if(!a->upload || !a->base.levels || a->base.nvalues>SIZE_MAX/a->base.levels/sizeof(float))return -1021;
 int rc=host_ozone(&a->base,sizeof(a->base));if(rc)return rc;
 if(a->upload!=a->base.out)memcpy(a->upload,a->base.out,a->base.nvalues*a->base.levels*sizeof(float));
 return 0;
}

#include <string.h>
typedef struct {Daylight base;int64_t*upload_day;uint64_t*upload_counts;} DaylightUpload;
/* Preserve private CPU compaction outputs and refresh both owned H2D payloads. */
int host_daylight_upload(const void*pointer,size_t bytes){
 if(bytes!=sizeof(DaylightUpload))return -1020;
 const DaylightUpload*a=pointer;
 if(!a->upload_day || !a->upload_counts || a->base.capacity>SIZE_MAX/sizeof(int64_t))return -1021;
 int rc=host_daylight(&a->base,sizeof(a->base));if(rc)return rc;
 if(a->upload_day!=a->base.day)memcpy(a->upload_day,a->base.day,a->base.capacity*sizeof(int64_t));
 if(a->upload_counts!=a->base.counts)memcpy(a->upload_counts,a->base.counts,2*sizeof(uint64_t));
 return 0;
}

#include <stddef.h>
#include <stdint.h>
#include <math.h>
typedef struct {size_t offset,count;uint32_t kind,reserved;uint64_t threshold;const uint64_t*active_count;} Rule;
typedef struct {const double*words;size_t nwords;const Rule*rules;size_t nrules;uint64_t*failed_rule;uint64_t*failed_word;} Guard;
/* kind 0 bool, 1 int32 status, 2 radii float32 max converted back to double. */
int host_numeric_guard(const void*pointer,size_t bytes){
 if(bytes!=sizeof(Guard))return -1020;
 const Guard*a=pointer;
 if(!a->words || !a->rules || !a->failed_rule || !a->failed_word)return -1021;
 *a->failed_rule=UINT64_MAX;*a->failed_word=UINT64_MAX;
 for(size_t i=0;i<a->nrules;i++){
  const Rule*r=&a->rules[i];
  if(r->offset>a->nwords || r->count>a->nwords-r->offset || r->kind>2 || (r->kind==2 && r->count!=1))return -1021;
 }
 for(size_t i=0;i<a->nrules;i++){
  const Rule*r=&a->rules[i];
  if(r->active_count && *r->active_count<=r->threshold)continue;
  for(size_t j=0;j<r->count;j++){
   double value=a->words[r->offset+j];int failed=0;
   if(r->kind==0)failed=value!=0.0;
   else if(r->kind==1)failed=!(value>-1.0 && value<1.0);
   else{double top=(double)(float)value;failed=top!=-INFINITY && top<1.0e-3;}
   if(failed){*a->failed_rule=i;*a->failed_word=j;return -1007;}
  }
 }
 return 0;
}

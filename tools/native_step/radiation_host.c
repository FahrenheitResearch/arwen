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

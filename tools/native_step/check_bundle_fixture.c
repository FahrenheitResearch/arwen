#include <stddef.h>
#include <stdint.h>
typedef struct {uint64_t size; uint64_t words[64];} Trace;
typedef struct {Trace *trace; uint64_t id;} Fixture;
int awstep_execute(void *opaque,void *frame,size_t bytes,uint64_t flags) {
 Fixture *p=opaque;
 if (flags & (UINT64_C(1)<<63)) return -77;
 if (!frame || bytes!=8 || p->trace->size>=64) return -78;
 p->trace->words[p->trace->size++]=p->id*1000+*(uint64_t*)frame+flags;
 return 0;
}

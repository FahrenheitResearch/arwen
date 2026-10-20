/* One host submission for an ordered set of already recorded phase plans. */
#include <stddef.h>
#include <stdint.h>
typedef int (*Execute)(void*, void*, size_t, uint64_t);
typedef struct {void *plan; void *frame; size_t bytes; uint64_t flags;} Stage;
int awstep_bundle(Execute execute, const Stage *stages, size_t count, size_t *completed) {
    if (!completed) return -2001;
    *completed = 0;
    if (!execute || !stages || !count || count > 32) return -2001;
    for (size_t i=0;i<count;i++)
        if (!stages[i].plan || (stages[i].bytes && !stages[i].frame)) return -2001;
    for (size_t i=0;i<count;i++) {
        int rc=execute(stages[i].plan, stages[i].frame, stages[i].bytes, stages[i].flags);
        if (rc) return rc;
        *completed=i+1;
    }
    return 0;
}

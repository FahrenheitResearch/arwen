/* Integration fixture only, no model arithmetic. */
#include <stdint.h>
#include <stddef.h>
typedef struct {const int64_t *input;int64_t *output;int64_t delta;} Args;
int fixture_consumer(const void*data,size_t bytes){
 if(!data || bytes!=sizeof(Args))return -1002;
 const Args*a=data;if(!a->input || !a->output)return -1002;
 if(*a->input<0)return -1007;
 *a->output=*a->input+a->delta;return 0;
}

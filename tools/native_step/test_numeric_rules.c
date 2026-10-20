/* CPU-only numerical predicate and owned-rule-vector contract. */
#include "../../gpuwm/native_step/recorder.c"
#include <assert.h>
#include <math.h>
static uint64_t word(double value){uint64_t bits;memcpy(&bits,&value,8);return bits;}
static Plan *fixture(const unsigned char*rules,size_t n,double*values,size_t count){
 Plan*p=awstep_create();assert(p);recording=p;
 assert(awstep_numeric_begin(p,rules,n,.001)==0);
 Operation*o=append(D2H,"fixture",NULL);o->host=values;o->count=count*8;
 append(STREAM_SYNC,"fixture",NULL);return p;
}
static void release(Plan*p){recording=NULL;awstep_destroy(p);}
int main(int argc,char**argv){
 if(argc==2 && strcmp(argv[1],"--stream")==0){
  unsigned rule;unsigned long long bits;
  while(scanf("%u %llx",&rule,&bits)==2){assert(rule==1 || rule==2);printf("%d\n",numeric_bad((uint64_t)bits,rule,.001));}
  return 0;
 }
 unsigned cases=0;
 double values[]={0.0,-0.0,1.0,-1.0,INFINITY,-INFINITY,NAN,0.001,0.0009999999999999998,0.0010000000000000002,0x1p-1074,-0x1p-1074};
 for(size_t j=0;j<sizeof(values)/sizeof(values[0]);j++){
  double x=values[j];assert(numeric_bad(word(x),1,.001)==(x!=0.0));
  assert(numeric_bad(word(x),2,.001)==(x!=-INFINITY && x<.001));cases+=2;
 }
 /* Existing guard modes retain their original selected-word semantics. */
 uint64_t words[]={0,7,0};Plan old={0};Operation op={0};
 op.host=words;op.count=sizeof(words);op.guard_width=8;
 assert(check_guard(&old,&op,9)==GUARD_FAILED && old.guard_index==1 && old.guard_value==7);cases++;
 memset(&old,0,sizeof(old));op.guard_mode=1;op.guard_mask=5;
 words[0]=word(-0.0);words[1]=word(NAN);words[2]=word(1.0);
 assert(check_guard(&old,&op,9)==0);cases++;
 words[2]=word(INFINITY);assert(check_guard(&old,&op,9)==GUARD_FAILED && old.guard_index==2);cases++;
 memset(&old,0,sizeof(old));op.guard_mode=2;op.guard_mask=2;words[1]=0;assert(check_guard(&old,&op,9)==0);cases++;
 memset(&old,0,sizeof(old));op.guard_mode=3;op.guard_expected=UINT64_MAX;words[0]=19;words[1]=UINT64_MAX;
 assert(check_guard(&old,&op,9)==0);cases++;
 words[1]=5;assert(check_guard(&old,&op,9)==GUARD_FAILED && old.guard_index==1 && old.guard_value==5);cases++;
 unsigned char rules[130];double data[130];memset(rules,1,sizeof(rules));memset(data,0,sizeof(data));
 rules[129]=2;data[129]=-INFINITY;
 Plan*p=fixture(rules,130,data,130);memset(rules,2,sizeof(rules));size_t guard;
 assert(awstep_host_end(p,&guard)==0);assert(p->ops[guard].numeric_rules[0]==1);cases++;
 data[128]=-7;assert(check_guard(p,&p->ops[guard],guard)==GUARD_FAILED);assert(p->guard_index==128 && p->guard_value==word(-7));cases++;release(p);
 rules[0]=1;p=fixture(rules,1,data,2);assert(awstep_host_end(p,&guard)==BAD_ARGUMENT);cases++;release(p);
 p=awstep_create();recording=p;rules[0]=0;assert(awstep_numeric_begin(p,rules,1,.001)==BAD_ARGUMENT);cases++;
 rules[0]=1;assert(awstep_numeric_begin(p,rules,1,NAN)==BAD_ARGUMENT);cases++;
 assert(awstep_numeric_begin(p,rules,1,INFINITY)==BAD_ARGUMENT);cases++;
 assert(awstep_numeric_begin(p,NULL,1,.001)==BAD_ARGUMENT);cases++;
 assert(awstep_numeric_begin(p,rules,0,.001)==BAD_ARGUMENT);cases++;
 release(p);
 p=awstep_create();recording=p;assert(awstep_host_begin(p,1,0)==0);
 assert(awstep_host_end_optional(p,&guard)==0 && p->host_policy==0);cases++;release(p);
 p=awstep_create();recording=p;assert(awstep_host_begin(p,3,0)==0);
 assert(awstep_host_end_optional(p,&guard)==0 && p->host_policy==0);cases++;release(p);
 p=awstep_create();recording=p;assert(awstep_host_begin(p,2,8)==0);
 assert(awstep_host_end_optional(p,&guard)==BAD_STATE && p->host_policy==2);cases++;release(p);
 p=awstep_create();recording=p;assert(awstep_host_begin(p,1,0)==0);
 assert(awstep_host_end(p,&guard)==BAD_STATE);cases++;release(p);
 printf("{\"status\":\"PASS\",\"cpu_assertion_cases\":%u,\"largest_vector_words\":130}\n",cases);return 0;
}

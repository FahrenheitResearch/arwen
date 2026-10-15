/* Oracle invokes pristine MPAS read_geogrid.c:45-144. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
int read_geogrid(char *, float *, int, int, int, int, int, int, int *);
int main(int argc,char **argv) {
 if(argc!=2) return 2;
 /* MPAS read_geogrid.c:67-84: missing and empty input both return status=1. */
 float probe=0; int status=-1;
 if(remove(argv[1])!=0) { /* The owned scratch file may not exist yet. */ }
 if(read_geogrid(argv[1],&probe,1,1,1,0,0,1,&status)!=1 || status!=1) return 5;
 FILE *empty=fopen(argv[1],"wb"); if(!empty) return 3; fclose(empty);
 status=-1;
 if(read_geogrid(argv[1],&probe,1,1,1,0,0,1,&status)!=1 || status!=1) return 6;
 for(int ws=1;ws<=4;ws++) {
  uint64_t half=(uint64_t)1<<(8*ws-1), mask=2*half-1;
  uint64_t words[16]={0,1,half-1,half,half+1,mask,mask-1,mask/3,mask/7,mask/11,16777215,16777216,16777217,2147483647,2147483648,4294967295};
  for(int sign=0;sign<2;sign++) for(int endian=0;endian<2;endian++) {
   FILE *f=fopen(argv[1],"wb"); if(!f) return 3;
   for(int i=0;i<16;i++) for(int b=0;b<ws;b++) {
    int shift=8*(endian==0 ? ws-1-b : b);
    fputc((words[i]&mask)>>shift &255,f);
   }
   fclose(f);
   float result[16]; int status=-1;
   read_geogrid(argv[1],result,16,1,1,sign,endian,ws,&status);
   if(status) return 4;
   for(int i=0;i<16;i++) printf("%d %d %d %d %08x %lld\n",ws,sign,endian,i,(uint32_t)(words[i]&mask),(long long)(int64_t)result[i]);
  }
 }
 return 0;
}

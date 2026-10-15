/* Synthetic native two-tile layouts; no geoindex files are needed. */
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
int main(void) {
    const char *names[]={"greenfrac","albedo_ncep","maxsnowalb","soiltemp_1deg"};
    for(int field=0;field<4;field++) {
        mkdir(names[field],0700);
        int tile=field<2?1250:180,months=field<2?12:1;
        for(int side=0;side<2;side++) {
            char name[256];
            snprintf(name,sizeof(name),"%s/%05d-%05d.00001-%05d",names[field],1+side*tile,(side+1)*tile,tile);
            FILE *f=fopen(name,"wb");if(!f)return 1;
            for(int month=0;month<months;month++)for(int j=0;j<tile+6;j++)for(int i=0;i<tile+6;i++) {
                int x=side*tile-2+i,y=-2+j;
                int value=abs(17*x+23*y+31*month)%151+1+side*5;
                if(field!=0 && x>=tile-8 && x<=tile+8 && y>=tile/2-8 && y<=tile/2+8)value=0;
                if(field==3 && value)value=25000+value*17;
                if(field==3)fputc((value>>8)&255,f);
                fputc(value&255,f);
            }
            if(fclose(f))return 2;
        }
    }
    return 0;
}

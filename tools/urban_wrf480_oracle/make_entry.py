"""Reuse the existing surface handover with the qualified 4.8 column solve."""
from pathlib import Path
import argparse,json


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    schema=json.loads((root/'gpuwm/data/urban/oracle/ucm480/schema.json').read_text())['arguments']
    text=(root/'gpuwm/core/kernels/urban_ucm.cu').read_text()
    aliases={'lh_kinematic':'lh_kin',**{n+'_urb':n for n in ('cmr','chr','cmc','chc','cmgr','chgr')}}
    extra={'tvg':43,'tt':44,'xxxvg':45,'tvgl':46,'smg':47,'cmcg':48,'flxhumvg':49,'flxhumt':50}
    out=['// WRF 4.8 column and existing Noah / Noah-MP handover.','struct Ucm480Col : UcmCol {',
         'float tvg,tt,xxxvg,tvgl[4],smg[4],cmcg,flxhumvg,flxhumt;',
         'float declin,cosz,xlat,ssgd,ssgq; bool lsolar;','};',
         '__device__ int ucm480_urban(const float* tab,const float* glob,const int* sw,Ucm480Col& c) {',
         'const U480Env env{tab,glob,sw};int status=0;const int layers=4;',
         'float zero=0.0f;float lf[4]={};']
    args=[]
    for v in schema:
        n=v['name']
        if n.startswith('num_'):value='layers'
        elif n in ('dzr','dzb','dzg'):value=f'glob+{dict(dzr=0,dzb=4,dzg=8)[n]}'
        elif n=='lf_urb':value='lf'
        elif n in ('mh_urb','stdh_urb','lp_urb','hgt_urb','frc_urb','lb_urb','zo_check','lf_urb_s','z0_urb','vegfrac_in'):value='zero'
        else:value='c.'+aliases.get(n,n)
        args.append(value)
    out+=['u480_urban('+','.join(args+['env','status'])+');','return status;','}']
    for action in ('load','store'):
        out.append(f'__device__ void ucm480_{action}_state(const unsigned long long* sp,size_t idx,size_t plane,Ucm480Col& c) {{')
        out.append(f'ucm_{action}_state(sp,idx,plane,c);')
        for n,index in extra.items():
            left=f'c.{n}';right=f'PLANE(sp,{index})[idx]'
            if n in ('tvgl','smg'):left+='[k]';right=f'PLANE(sp,{index})[k*plane+idx]'
            assignment=f'{left}={right};' if action=='load' else f'{right}={left};'
            out.append(('for(int k=0;k<4;++k) ' if n in ('tvgl','smg') else '')+assignment)
        out.append('}')
    start=text.index('extern "C" __global__ void ucm_noah_after_lsm(')
    end=text.index('// surface-driver overrides after SFCDIAGS.',start)
    handover=text[start:end]
    for n in ('urban','load_state','store_state','noah_after_lsm','noahmp_after_lsm'):
        handover=handover.replace('ucm_'+n,'ucm480_'+n)
    handover=handover.replace('UcmCol c;','Ucm480Col c{};')
    handover=handover.replace('const float* __restrict__ hrang,','const float* __restrict__ hrang,const float* __restrict__ cosz,const float* __restrict__ xlat,float declin,')
    handover=handover.replace('c.omg = hrang[idx];','c.omg = hrang[idx];c.cosz=cosz[idx];c.xlat=xlat[idx];c.declin=declin;c.lsolar=false;c.ssgd=FMUL(0.8f,c.ssg);c.ssgq=FSUB(c.ssg,c.ssgd);')
    out.append(handover)
    out+=['extern "C" __global__ void urban480_columns(const unsigned long long* args,const float* table,const float* globals,const int* switches,int* codes,int n) {',
          'int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;int status=0;U480Env env{table,globals,switches};']
    for i,v in enumerate(schema):
        kind=v['type'];name=v['name'];storage='int' if kind=='bool' else kind
        out.append(f'{kind} {name}[4];for(int k=0;k<4;++k){name}[k]=((const {storage}*)args[{i}])[row+k*n];' if v['layers'] else f'{kind} {name}=((const {storage}*)args[{i}])[row];')
    out.append('u480_urban('+','.join([v['name'] for v in schema]+['env','status'])+');')
    for i,v in enumerate(schema):
        kind=v['type'];name=v['name'];storage='int' if kind=='bool' else kind
        out.append(f'for(int k=0;k<4;++k)(({storage}*)args[{i}])[row+k*n]={name}[k];' if v['layers'] else f'(({storage}*)args[{i}])[row]={name};')
    out+=['codes[row]=status;','}']
    a.output.write_text('\n'.join(out)+'\n')


if __name__=='__main__':main()

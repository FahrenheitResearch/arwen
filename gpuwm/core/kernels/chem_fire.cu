// NOAA GSL ccpp-physics 3e6660c6, Apache-2.0.
// rrfs_smoke_wrapper.F90:855-1013; module_add_emiss_burn.F90:70-159.
// kpbl starts at nz-1 and is capped there. GSL leaves kpbl unwritten above
// the column and reads us3d(kpbl+1) out of bounds for kpbl=kte. All other
// column state is overwritten, except coef/fire_hist which carry across calls.
// The driver supplies glibc_flt32.cuh through the kernel loader.
// Input mappings: t/p/qv/u/v/z/z_at_w are chem_prep mass/interface fields;
// u10/v10/swdown/pblh are physics outputs. oro is model terrain in metres.
// t2m=physics.fields["t2"] (physics.py:339); dpt2m is explicit Kelvin
// dewpoint. Q2/PSFC alone is refused: surface_moisture_ledger.py:100-123
// uses a binary64 Bolton diagnostic, not a pinned float32 input to GSL.
// wetness is FV3 soil wetness,
// never volumetric smois. totprcp/totprcp_24hrs are metres; snow is SWE mm.
// LANDUSEF fractions are required for the dataset in the rule table;
// dominant ivgtyp is never substituted for fractional cover.
extern "C" {
__constant__ float fire_categories[64];
__constant__ float fire_groups[48];
__constant__ float fire_rules[160];
__constant__ float fire_curves[160];
__constant__ float fire_decay[64];
}

__device__ float fire_tv(int x,const float* t,const float* p,const float* q) {
    float theta=FMUL(t[x],gfk_pow(FDIV(1.e5f,p[x]),.286f));
    return FMUL(theta,FADD(1.0f,FMUL(.61f,q[x])));
}

extern "C" __global__ void chem_fire_prep(
    const float* t,const float* p,const float* q,const float* u,const float* v,
    const float* z,const float* zw,const float* pbl,const float* oro,
    const float* u10,const float* v10,const float* t2,const float* td2,
    const float* wet,const float* precip,const float* prevprecip,
    const float* sw,const float* snow,int* kp,int* ktv,float* avg,
    float* gust,float* hp,float* hwp,int nz,int nc,int hour,int method) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc) return;
    int kb=nz-1;
    for(int k=1;k<nz;++k) if(zw[k*nc+c]>pbl[c]) {
        kb=min(nz-1,max(2,k+1)); break;
    }
    kp[c]=kb;
    float tv0=fire_tv(c,t,p,q),boost=FADD(tv0,.5f);
    int kv=2;
    if(fire_tv(nc+c,t,p,q)<boost) for(int k=1;k<nz;++k) {
        kv=k+1;
        if(fire_tv(k*nc+c,t,p,q)>boost) break;
    }
    ktv[c]=kv;
    float sf=FSQRT(FADD(FMUL(u10[c],u10[c]),FMUL(v10[c],v10[c])));
    float a=sf,g=sf;
    for(int k=1;k<=kb;++k) {
        int x=k*nc+c;
        float wind=FSQRT(FADD(FMUL(u[x],u[x]),FMUL(v[x],v[x])));
        a=FADD(a,wind);
        float d=FSUB(z[x],oro[c]);
        float delta=FMUL(FSUB(wind,sf),FSUB(1.0f,fminf(.5f,FDIV(d,2000.0f))));
        g=fmaxf(g,FADD(sf,delta));
    }
    a=FDIV(a,(float)kb);
    float h=FSUB(zw[(kb-1)*nc+c],zw[c]);
    avg[c]=a;gust[c]=g;hp[c]=h;
    if(method==0) {hwp[c]=0.0f;return;}
    float pf=FADD(2.5f,FDIV(FMUL((float)hour,2.5f),24.0f));
    float deficit=FSUB(t2[c],td2[c]),answer=0.0f;
    if(method==1 || method==3) {
        float vent=method==1 ? fmaxf(sf,3.0f):a;
        answer=FDIV(FMUL(.022f,fmaxf(FSUB(pf,FMUL(FADD(precip[c],prevprecip[c]),1.e3f)),0.0f)),pf);
        answer=FMUL(answer,gfk_pow(FSUB(1.0f,wet[c]),.51f));
        answer=FMUL(answer,gfk_pow(FMUL(vent,h),.57f));
        answer=FMUL(answer,gfk_pow(fminf(25.0f,fmaxf(15.0f,deficit)),.74f));
        answer=FMUL(answer,gfk_pow(fminf(3.0f,FADD(1.0f,FDIV(sw[c],250.0f))),.18f));
    } else {
        float wg=method==2 ? fmaxf(g,3.0f):fmaxf(FMUL(1.69f,sf),3.0f);
        float se=fmaxf(FDIV(FSUB(25.0f,snow[c]),25.0f),0.0f);
        answer=FMUL(.177f,gfk_pow(wg,.97f));
        answer=FMUL(answer,gfk_pow(fmaxf(deficit,15.0f),1.03f));
        answer=FMUL(answer,gfk_pow(FSUB(1.0f,wet[c]),.4f));
        answer=FMUL(answer,se);
    }
    hwp[c]=answer;
}

// Generic rule interpreter. Category indices, thresholds and coordinates
// arrive from fire_type_rules.json, never from source-specific branches.
extern "C" __global__ void chem_fire_classify(
    const float* frac,const float* lat,const float* lon,const float* emission,
    int* firetype,int nc,int ng,int nr,float period,float minimum) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc) return;
    float sums[16];
    const float* cats=fire_categories;
    const float* groups=fire_groups;
    const float* rules=fire_rules;
    for(int g=0;g<ng;++g) {
        int offset=(int)groups[g*3],len=(int)groups[g*3+1],parent=(int)groups[g*3+2];
        float s=parent<0 ? 0.0f:sums[parent];
        for(int q=0;q<len;++q) s=FADD(s,frac[(int)cats[offset+q]*nc+c]);
        sums[g]=s;
    }
    if(emission[c]<minimum) {firetype[c]=0;return;}
    float east=fmodf(lon[c],period);
    if(east<0.0f) east=FADD(east,period);
    for(int r=0;r<nr;++r) {
        int g=(int)rules[r*5];
        bool match=g>=0 ? sums[g]>rules[r*5+1]:
            (g==-2 || (east>rules[r*5+1] && lat[c]>rules[r*5+2] && lat[c]<rules[r*5+3]));
        if(match) {firetype[c]=(int)rules[r*5+4];return;}
    }
}

extern "C" __global__ void chem_fire_cycle(
    float* coef,float* hist,const float* hwp,const float* prev,const float* sw,
    const float* ends,const int* typ,
    int nc,int nd,float time,float scale,int mode) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc || mode==1) return;
    const float* curves=fire_curves;
    const float* decay=fire_decay;
    int row=typ[c]*5,kind=(int)curves[row];
    float age=fmaxf(.01f,FADD(FDIV(time,3600.0f),FSUB(ends[c],2.0f)));
    if(kind==1) {
        float x=FSUB(gfk_log(age),curves[row+3]);
        coef[c]=FMUL(FDIV(curves[row+1],FMUL(curves[row+2],age)),
                    gfk_exp(FDIV(-FMUL(x,x),curves[row+4])));
    } else if(kind==2) {
        for(int d=0;d<nd;++d) if(sw[c]<.1f && age>decay[2*d] && hist[c]>decay[2*d+1])
            hist[c]=decay[2*d+1];
        float ratio=fminf(20.0f,fmaxf(0.0f,FDIV(hwp[c],fmaxf(10.0f,prev[c]))));
        coef[c]=FMUL(FMUL(scale,hist[c]),ratio);
    }
}

extern "C" __global__ void chem_fire_inject(
    float* chem,const float* ebu,const float* rho,const float* dryrho,
    const float* dz,const float* coef,float* emitted,int nc,int nz,
    int mode,float dt,float minimum,float weight,int surface,float state_factor) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc) return;
    float mass=0.0f;
    // GSL kfire_max = 51 (module_add_emiss_burn.F90:50) assumes RRFS's
    // 64-level column; a shorter column stops at its own top, where GSL
    // would read past it.
    int limit=surface ? 1:(nz<51 ? nz:51);
    for(int k=0;k<limit;++k) {
        int x=k*nc+c;
        float e=FMUL(ebu[x],weight);
        if(e<minimum) continue;
        float conv=mode==1 ? dt:FMUL(coef[c],dt);
        conv=FDIV(conv,FMUL(rho[x],dz[x]));
        float old=chem[x];
        chem[x]=fminf(fmaxf(FADD(old,FMUL(FMUL(conv,e),state_factor)),0.0f),5000.0f);
        mass=FADD(mass,FDIV(FMUL(FMUL(FSUB(chem[x],old),dryrho[x]),dz[x]),state_factor));
    }
    emitted[c]=FADD(emitted[c],mass);
}

// Source mass conversion uses dryrho for conservative ug/kg-dry state.
// GSL rho=p/(Rd*T) differs from chem_prep moist density. Injection accepts
// explicit denominator; the production driver supplies dryrho for mass identity.
extern "C" __global__ void chem_fire_units(
    const float* kg,const float* mx,const float* my,float* emission,float* area,
    float dx,float dy,int nc) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc) return;
    area[c]=FDIV(FMUL(dx,dy),FMUL(mx[c],my[c]));
    emission[c]=FDIV(FMUL(kg[c],1.e9f),FMUL(3600.0f,area[c]));
}

// rrfs_smoke_wrapper.F90:365-370,552-556. Store the scaled output, then
// divide by the previous coefficient's 1e-4 floor on a reuse step. The
// loss below that floor is source behavior, not an emission renormalization.
extern "C" __global__ void chem_fire_carry(
    float* ebu,const float* coef,int nc,int nz,int restore) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc) return;
    for(int k=0;k<nz;++k) {
        int x=k*nc+c;
        ebu[x]=restore ? FDIV(ebu[x],fmaxf(1.e-4f,coef[c])):FMUL(ebu[x],coef[c]);
    }
}

// rrfs_smoke_wrapper.F90:735,620. Preserve MW->W, coefficient, W->MW
// association instead of cancelling the unit factors before rounding.
extern "C" __global__ void chem_fire_frp_diag(
    const float* frp_mw,const float* coef,float* output,int nc) {
    int c=blockDim.x*blockIdx.x+threadIdx.x;
    if(c>=nc) return;
    output[c]=FMUL(FMUL(coef[c],FMUL(frp_mw[c],1.e6f)),1.e-6f);
}

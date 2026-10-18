// Apache-2.0. Pinned CCPP transcription. See tools/satmedmf_ccpp_oracle/source.json.
__device__ void sm_tridi1(int l, int n, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_r1, double* raw_au, double* raw_a1, SMWorkspace& ws);
__device__ void sm_tridi2(int l, int n, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_r1, double* raw_r2, double* raw_au, double* raw_a1, double* raw_a2, SMWorkspace& ws);
__device__ void sm_tridin(int l, int n, int nt, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_r1, double* raw_r2, double* raw_au, double* raw_a1, double* raw_a2, SMWorkspace& ws);
__device__ void sm_tridit(int l, int n, int nt, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_rt, double* raw_au, double* raw_at, SMWorkspace& ws);
__device__ void sm_mfpbltq(int im, int ix, int km, int kmpbl, int ntcw, int ntrac1, double delt, bool* raw_cnvflg, double* raw_zl, double* raw_zm, double* raw_q1, double* raw_t1, double* raw_u1, double* raw_v1, double* raw_plyr, double* raw_pix, double* raw_thlx, double* raw_thvx, double* raw_gdx, double* raw_hpbl, int* raw_kpbl, double* raw_vpert, double* raw_buo, double* raw_wush, double* raw_tkemean, double* raw_vez0fun, double* raw_xmf, double* raw_tcko, double* raw_qcko, double* raw_ucko, double* raw_vcko, double* raw_xlamueq, double a1, SMWorkspace& ws);
__device__ void sm_mfscuq(int im, int ix, int km, int kmscu, int ntcw, int ntrac1, double delt, bool* raw_cnvflg, double* raw_zl, double* raw_zm, double* raw_q1, double* raw_t1, double* raw_u1, double* raw_v1, double* raw_plyr, double* raw_pix, double* raw_thlx, double* raw_thvx, double* raw_thlvx, double* raw_gdx, double* raw_thetae, int* raw_krad, int* raw_mrad, double* raw_radmin, double* raw_buo, double* raw_wush, double* raw_tkemean, double* raw_vez0fun, double* raw_xmfd, double* raw_tcdo, double* raw_qcdo, double* raw_ucdo, double* raw_vcdo, double* raw_xlamdeq, double a1, SMWorkspace& ws);
__device__ void sm_satmedmfvdifq_run(int im, int km, int ntrac, int ntcw, int ntrw, int ntiw, int ntke, double grav, double rd, double cp, double rv, double hvap, double hfus, double fv, double eps, double epsm1, double* raw_dv, double* raw_du, double* raw_tdt, double* raw_rtg, double* raw_u1, double* raw_v1, double* raw_t1, double* raw_q1, double* raw_usfco, double* raw_vsfco, int icplocn2atm, double* raw_swh, double* raw_hlw, double* raw_xmu, double* raw_garea, double* raw_zvfun, double* raw_sigmaf, double* raw_psk, double* raw_rbsoil, double* raw_zorl, double* raw_u10m, double* raw_v10m, double* raw_fm, double* raw_fh, double* raw_tsea, double* raw_heat, double* raw_evap, double* raw_stress, double* raw_spd1, int* raw_kpbl, double* raw_prsi, double* raw_del, double* raw_prsl, double* raw_prslk, double* raw_phii, double* raw_phil, double delt, bool dspheat, double* raw_dusfc, double* raw_dvsfc, double* raw_dtsfc, double* raw_dqsfc, double* raw_hpbl, double* raw_dkt, double* raw_dku, int* raw_kinver, double xkzm_m, double xkzm_h, double xkzm_s, double dspfac, double bl_upfr, double bl_dnfr, double rlmx, double elmx, int sfc_rlm, int tc_pbl, int ntqv, double* raw_dtend, int* raw_dtidx, int index_of_temperature, int index_of_x_wind, int index_of_y_wind, int index_of_process_pbl, bool gen_tend, bool ldiag3d, char* errmsg, int& errflg, SMWorkspace& ws);
__device__ void sm_tridi1(int l, int n, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_r1, double* raw_au, double* raw_a1, SMWorkspace& ws) {
SMFrame frame(ws);
int one;
int k;
int i;
double fk;
SMView<double> cl(raw_cl, l, ((n)-(2)+1), 1, 1, 2, 1);
SMView<double> cm(raw_cm, l, n, 1, 1, 1, 1);
SMView<double> cu(raw_cu, l, n-1, 1, 1, 1, 1);
SMView<double> r1(raw_r1, l, n, 1, 1, 1, 1);
SMView<double> au(raw_au, l, n-1, 1, 1, 1, 1);
SMView<double> a1(raw_a1, l, n, 1, 1, 1, 1);
one = 0x1.0000000000000p+0; // tridi.f:14
// tridi.f:21
for(i=1; i<=l; i+=1) {
// tridi.f:22
fk = sm_div((double)(one),(double)(cm(i,1)));
// tridi.f:23
au(i,1) = sm_mul((double)(fk),(double)(cu(i,1)));
// tridi.f:24
a1(i,1) = sm_mul((double)(fk),(double)(r1(i,1)));
// tridi.f:25
}
// tridi.f:26
for(k=2; k<=(n - 1); k+=1) {
// tridi.f:27
for(i=1; i<=l; i+=1) {
// tridi.f:28
fk = sm_div((double)(one),(double)(sm_sub((double)(cm(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(au(i,(k - 1))))))));
// tridi.f:29
au(i,k) = sm_mul((double)(fk),(double)(cu(i,k)));
// tridi.f:30
a1(i,k) = sm_mul((double)(fk),(double)(sm_sub((double)(r1(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(a1(i,(k - 1))))))));
// tridi.f:31
}
// tridi.f:32
}
// tridi.f:33
for(i=1; i<=l; i+=1) {
// tridi.f:34
fk = sm_div((double)(one),(double)(sm_sub((double)(cm(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(au(i,(n - 1))))))));
// tridi.f:35
a1(i,n) = sm_mul((double)(fk),(double)(sm_sub((double)(r1(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(a1(i,(n - 1))))))));
// tridi.f:36
}
// tridi.f:37
for(k=(n - 1); k>=1; k+=(-1)) {
// tridi.f:38
for(i=1; i<=l; i+=1) {
// tridi.f:39
a1(i,k) = sm_sub((double)(a1(i,k)),(double)(sm_mul((double)(au(i,k)),(double)(a1(i,(k + 1))))));
// tridi.f:40
}
// tridi.f:41
}
// tridi.f:43
return;
}
__device__ void sm_tridi2(int l, int n, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_r1, double* raw_r2, double* raw_au, double* raw_a1, double* raw_a2, SMWorkspace& ws) {
SMFrame frame(ws);
int one;
int k;
int i;
double fk;
SMView<double> cl(raw_cl, l, ((n)-(2)+1), 1, 1, 2, 1);
SMView<double> cm(raw_cm, l, n, 1, 1, 1, 1);
SMView<double> cu(raw_cu, l, n-1, 1, 1, 1, 1);
SMView<double> r1(raw_r1, l, n, 1, 1, 1, 1);
SMView<double> r2(raw_r2, l, n, 1, 1, 1, 1);
SMView<double> au(raw_au, l, n-1, 1, 1, 1, 1);
SMView<double> a1(raw_a1, l, n, 1, 1, 1, 1);
SMView<double> a2(raw_a2, l, n, 1, 1, 1, 1);
one = 0x1.0000000000000p+0; // tridi.f:53
// tridi.f:60
for(i=1; i<=l; i+=1) {
// tridi.f:61
fk = sm_div((double)(one),(double)(cm(i,1)));
// tridi.f:62
au(i,1) = sm_mul((double)(fk),(double)(cu(i,1)));
// tridi.f:63
a1(i,1) = sm_mul((double)(fk),(double)(r1(i,1)));
// tridi.f:64
a2(i,1) = sm_mul((double)(fk),(double)(r2(i,1)));
// tridi.f:65
}
// tridi.f:66
for(k=2; k<=(n - 1); k+=1) {
// tridi.f:67
for(i=1; i<=l; i+=1) {
// tridi.f:68
fk = sm_div((double)(one),(double)(sm_sub((double)(cm(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(au(i,(k - 1))))))));
// tridi.f:69
au(i,k) = sm_mul((double)(fk),(double)(cu(i,k)));
// tridi.f:70
a1(i,k) = sm_mul((double)(fk),(double)(sm_sub((double)(r1(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(a1(i,(k - 1))))))));
// tridi.f:71
a2(i,k) = sm_mul((double)(fk),(double)(sm_sub((double)(r2(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(a2(i,(k - 1))))))));
// tridi.f:72
}
// tridi.f:73
}
// tridi.f:74
for(i=1; i<=l; i+=1) {
// tridi.f:75
fk = sm_div((double)(one),(double)(sm_sub((double)(cm(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(au(i,(n - 1))))))));
// tridi.f:76
a1(i,n) = sm_mul((double)(fk),(double)(sm_sub((double)(r1(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(a1(i,(n - 1))))))));
// tridi.f:77
a2(i,n) = sm_mul((double)(fk),(double)(sm_sub((double)(r2(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(a2(i,(n - 1))))))));
// tridi.f:78
}
// tridi.f:79
for(k=(n - 1); k>=1; k+=(-1)) {
// tridi.f:80
for(i=1; i<=l; i+=1) {
// tridi.f:81
a1(i,k) = sm_sub((double)(a1(i,k)),(double)(sm_mul((double)(au(i,k)),(double)(a1(i,(k + 1))))));
// tridi.f:82
a2(i,k) = sm_sub((double)(a2(i,k)),(double)(sm_mul((double)(au(i,k)),(double)(a2(i,(k + 1))))));
// tridi.f:83
}
// tridi.f:84
}
// tridi.f:86
return;
}
__device__ void sm_tridin(int l, int n, int nt, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_r1, double* raw_r2, double* raw_au, double* raw_a1, double* raw_a2, SMWorkspace& ws) {
SMFrame frame(ws);
int one;
int is;
int k;
int kk;
int i;
SMView<double> fk(ws.alloc<double>((l)*(1)*(1)), l, 1, 1, 1, 1, 1);
SMView<double> cl(raw_cl, l, ((n)-(2)+1), 1, 1, 2, 1);
SMView<double> cm(raw_cm, l, n, 1, 1, 1, 1);
SMView<double> cu(raw_cu, l, n-1, 1, 1, 1, 1);
SMView<double> r1(raw_r1, l, n, 1, 1, 1, 1);
SMView<double> r2(raw_r2, l, n*nt, 1, 1, 1, 1);
SMView<double> au(raw_au, l, n-1, 1, 1, 1, 1);
SMView<double> a1(raw_a1, l, n, 1, 1, 1, 1);
SMView<double> a2(raw_a2, l, n*nt, 1, 1, 1, 1);
SMView<double> fkk(ws.alloc<double>((l)*(((n-1)-(2)+1))*(1)), l, ((n-1)-(2)+1), 1, 1, 2, 1);
one = 0x1.0000000000000p+0; // tridi.f:98
// tridi.f:107
for(i=1; i<=l; i+=1) {
// tridi.f:108
fk(i) = sm_div((double)(one),(double)(cm(i,1)));
// tridi.f:109
au(i,1) = sm_mul((double)(fk(i)),(double)(cu(i,1)));
// tridi.f:110
a1(i,1) = sm_mul((double)(fk(i)),(double)(r1(i,1)));
// tridi.f:111
}
// tridi.f:112
for(k=1; k<=nt; k+=1) {
// tridi.f:113
is = ((k - 1) * n);
// tridi.f:114
for(i=1; i<=l; i+=1) {
// tridi.f:115
a2(i,(1 + is)) = sm_mul((double)(fk(i)),(double)(r2(i,(1 + is))));
// tridi.f:116
}
// tridi.f:117
}
// tridi.f:118
for(k=2; k<=(n - 1); k+=1) {
// tridi.f:119
for(i=1; i<=l; i+=1) {
// tridi.f:120
fkk(i,k) = sm_div((double)(one),(double)(sm_sub((double)(cm(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(au(i,(k - 1))))))));
// tridi.f:121
au(i,k) = sm_mul((double)(fkk(i,k)),(double)(cu(i,k)));
// tridi.f:122
a1(i,k) = sm_mul((double)(fkk(i,k)),(double)(sm_sub((double)(r1(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(a1(i,(k - 1))))))));
// tridi.f:123
}
// tridi.f:124
}
// tridi.f:125
for(kk=1; kk<=nt; kk+=1) {
// tridi.f:126
is = ((kk - 1) * n);
// tridi.f:127
for(k=2; k<=(n - 1); k+=1) {
// tridi.f:128
for(i=1; i<=l; i+=1) {
// tridi.f:129
a2(i,(k + is)) = sm_mul((double)(fkk(i,k)),(double)(sm_sub((double)(r2(i,(k + is))),(double)(sm_mul((double)(cl(i,k)),(double)(a2(i,((k + is) - 1))))))));
// tridi.f:130
}
// tridi.f:131
}
// tridi.f:132
}
// tridi.f:133
for(i=1; i<=l; i+=1) {
// tridi.f:134
fk(i) = sm_div((double)(one),(double)(sm_sub((double)(cm(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(au(i,(n - 1))))))));
// tridi.f:135
a1(i,n) = sm_mul((double)(fk(i)),(double)(sm_sub((double)(r1(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(a1(i,(n - 1))))))));
// tridi.f:136
}
// tridi.f:137
for(k=1; k<=nt; k+=1) {
// tridi.f:138
is = ((k - 1) * n);
// tridi.f:139
for(i=1; i<=l; i+=1) {
// tridi.f:140
a2(i,(n + is)) = sm_mul((double)(fk(i)),(double)(sm_sub((double)(r2(i,(n + is))),(double)(sm_mul((double)(cl(i,n)),(double)(a2(i,((n + is) - 1))))))));
// tridi.f:141
}
// tridi.f:142
}
// tridi.f:143
for(k=(n - 1); k>=1; k+=(-1)) {
// tridi.f:144
for(i=1; i<=l; i+=1) {
// tridi.f:145
a1(i,k) = sm_sub((double)(a1(i,k)),(double)(sm_mul((double)(au(i,k)),(double)(a1(i,(k + 1))))));
// tridi.f:146
}
// tridi.f:147
}
// tridi.f:148
for(kk=1; kk<=nt; kk+=1) {
// tridi.f:149
is = ((kk - 1) * n);
// tridi.f:150
for(k=(n - 1); k>=1; k+=(-1)) {
// tridi.f:151
for(i=1; i<=l; i+=1) {
// tridi.f:152
a2(i,(k + is)) = sm_sub((double)(a2(i,(k + is))),(double)(sm_mul((double)(au(i,k)),(double)(a2(i,((k + is) + 1))))));
// tridi.f:153
}
// tridi.f:154
}
// tridi.f:155
}
// tridi.f:157
return;
}
__device__ void sm_tridit(int l, int n, int nt, double* raw_cl, double* raw_cm, double* raw_cu, double* raw_rt, double* raw_au, double* raw_at, SMWorkspace& ws) {
SMFrame frame(ws);
int one;
int is;
int k;
int kk;
int i;
SMView<double> fk(ws.alloc<double>((l)*(1)*(1)), l, 1, 1, 1, 1, 1);
SMView<double> cl(raw_cl, l, ((n)-(2)+1), 1, 1, 2, 1);
SMView<double> cm(raw_cm, l, n, 1, 1, 1, 1);
SMView<double> cu(raw_cu, l, n-1, 1, 1, 1, 1);
SMView<double> rt(raw_rt, l, n*nt, 1, 1, 1, 1);
SMView<double> au(raw_au, l, n-1, 1, 1, 1, 1);
SMView<double> at(raw_at, l, n*nt, 1, 1, 1, 1);
SMView<double> fkk(ws.alloc<double>((l)*(((n-1)-(2)+1))*(1)), l, ((n-1)-(2)+1), 1, 1, 2, 1);
one = 0x1.0000000000000p+0; // tridi.f:168
// tridi.f:177
for(i=1; i<=l; i+=1) {
// tridi.f:178
fk(i) = sm_div((double)(one),(double)(cm(i,1)));
// tridi.f:179
au(i,1) = sm_mul((double)(fk(i)),(double)(cu(i,1)));
// tridi.f:180
}
// tridi.f:181
for(k=1; k<=nt; k+=1) {
// tridi.f:182
is = ((k - 1) * n);
// tridi.f:183
for(i=1; i<=l; i+=1) {
// tridi.f:184
at(i,(1 + is)) = sm_mul((double)(fk(i)),(double)(rt(i,(1 + is))));
// tridi.f:185
}
// tridi.f:186
}
// tridi.f:187
for(k=2; k<=(n - 1); k+=1) {
// tridi.f:188
for(i=1; i<=l; i+=1) {
// tridi.f:189
fkk(i,k) = sm_div((double)(one),(double)(sm_sub((double)(cm(i,k)),(double)(sm_mul((double)(cl(i,k)),(double)(au(i,(k - 1))))))));
// tridi.f:190
au(i,k) = sm_mul((double)(fkk(i,k)),(double)(cu(i,k)));
// tridi.f:191
}
// tridi.f:192
}
// tridi.f:193
for(kk=1; kk<=nt; kk+=1) {
// tridi.f:194
is = ((kk - 1) * n);
// tridi.f:195
for(k=2; k<=(n - 1); k+=1) {
// tridi.f:196
for(i=1; i<=l; i+=1) {
// tridi.f:197
at(i,(k + is)) = sm_mul((double)(fkk(i,k)),(double)(sm_sub((double)(rt(i,(k + is))),(double)(sm_mul((double)(cl(i,k)),(double)(at(i,((k + is) - 1))))))));
// tridi.f:198
}
// tridi.f:199
}
// tridi.f:200
}
// tridi.f:201
for(i=1; i<=l; i+=1) {
// tridi.f:202
fk(i) = sm_div((double)(one),(double)(sm_sub((double)(cm(i,n)),(double)(sm_mul((double)(cl(i,n)),(double)(au(i,(n - 1))))))));
// tridi.f:203
}
// tridi.f:204
for(k=1; k<=nt; k+=1) {
// tridi.f:205
is = ((k - 1) * n);
// tridi.f:206
for(i=1; i<=l; i+=1) {
// tridi.f:207
at(i,(n + is)) = sm_mul((double)(fk(i)),(double)(sm_sub((double)(rt(i,(n + is))),(double)(sm_mul((double)(cl(i,n)),(double)(at(i,((n + is) - 1))))))));
// tridi.f:208
}
// tridi.f:209
}
// tridi.f:210
for(kk=1; kk<=nt; kk+=1) {
// tridi.f:211
is = ((kk - 1) * n);
// tridi.f:212
for(k=(n - 1); k>=1; k+=(-1)) {
// tridi.f:213
for(i=1; i<=l; i+=1) {
// tridi.f:214
at(i,(k + is)) = sm_sub((double)(at(i,(k + is))),(double)(sm_mul((double)(au(i,k)),(double)(at(i,((k + is) + 1))))));
// tridi.f:215
}
// tridi.f:216
}
// tridi.f:217
}
// tridi.f:219
return;
}
__device__ void sm_mfpbltq(int im, int ix, int km, int kmpbl, int ntcw, int ntrac1, double delt, bool* raw_cnvflg, double* raw_zl, double* raw_zm, double* raw_q1, double* raw_t1, double* raw_u1, double* raw_v1, double* raw_plyr, double* raw_pix, double* raw_thlx, double* raw_thvx, double* raw_gdx, double* raw_hpbl, int* raw_kpbl, double* raw_vpert, double* raw_buo, double* raw_wush, double* raw_tkemean, double* raw_vez0fun, double* raw_xmf, double* raw_tcko, double* raw_qcko, double* raw_ucko, double* raw_vcko, double* raw_xlamueq, double a1, SMWorkspace& ws) {
SMFrame frame(ws);
const double grav=sm_g;
const double cp=sm_cp;
const double rv=sm_rv;
const double hvap=sm_hvap;
const double fv=sm_fvirt;
const double eps=sm_eps;
const double epsm1=sm_epsm1;
SMView<int> kpbl(raw_kpbl, im, 1, 1, 1, 1, 1);
SMView<bool> cnvflg(raw_cnvflg, im, 1, 1, 1, 1, 1);
SMView<double> q1(raw_q1, ix, km, ntrac1, 1, 1, 1);
SMView<double> t1(raw_t1, ix, km, 1, 1, 1, 1);
SMView<double> u1(raw_u1, ix, km, 1, 1, 1, 1);
SMView<double> v1(raw_v1, ix, km, 1, 1, 1, 1);
SMView<double> plyr(raw_plyr, im, km, 1, 1, 1, 1);
SMView<double> pix(raw_pix, im, km, 1, 1, 1, 1);
SMView<double> thlx(raw_thlx, im, km, 1, 1, 1, 1);
SMView<double> thvx(raw_thvx, im, km, 1, 1, 1, 1);
SMView<double> zl(raw_zl, im, km, 1, 1, 1, 1);
SMView<double> zm(raw_zm, im, km, 1, 1, 1, 1);
SMView<double> gdx(raw_gdx, im, 1, 1, 1, 1, 1);
SMView<double> hpbl(raw_hpbl, im, 1, 1, 1, 1, 1);
SMView<double> vpert(raw_vpert, im, 1, 1, 1, 1, 1);
SMView<double> buo(raw_buo, im, km, 1, 1, 1, 1);
SMView<double> wush(raw_wush, im, km, 1, 1, 1, 1);
SMView<double> tkemean(raw_tkemean, im, 1, 1, 1, 1, 1);
SMView<double> vez0fun(raw_vez0fun, im, 1, 1, 1, 1, 1);
SMView<double> xmf(raw_xmf, im, km, 1, 1, 1, 1);
SMView<double> tcko(raw_tcko, im, km, 1, 1, 1, 1);
SMView<double> qcko(raw_qcko, im, km, ntrac1, 1, 1, 1);
SMView<double> ucko(raw_ucko, im, km, 1, 1, 1, 1);
SMView<double> vcko(raw_vcko, im, km, 1, 1, 1, 1);
SMView<double> xlamueq(raw_xlamueq, im, km-1, 1, 1, 1, 1);
int i;
int j;
int k;
int n;
int ndc;
SMView<int> kpblx(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<int> kpbly(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
double dt2;
double dz;
double ce0;
double cm;
double cq;
double tkcrt;
double factor;
double gocp;
double cmxfac;
double g;
double b1;
double f1;
double bb1;
double bb2;
double alp;
double vpertmax;
double pgcon;
double qmin;
double qlmin;
double xmmx;
double rbint;
double tem;
double tem1;
double tem2;
double ptem;
double ptem1;
double ptem2;
double elocp;
double el2orc;
double qs;
double es;
double tlu;
double gamma;
double qlu;
double thup;
double thvu;
double dq;
SMView<double> rbdn(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> rbup(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> hpblx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> xlamue(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> xlamuem(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> delz(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> xlamax(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> ce0t(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> wu2(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> thlu(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qtx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qtu(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> xlamavg(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> sigma(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> scaldfunc(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> sumx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
bool totflg;
SMView<bool> flg(ws.alloc<bool>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
g = grav; // mfpbltq.f:74
gocp = sm_div((double)(g),(double)(cp)); // mfpbltq.f:75
elocp = sm_div((double)(hvap),(double)(cp)); // mfpbltq.f:76
el2orc = sm_div((double)(sm_mul((double)(hvap),(double)(hvap))),(double)(sm_mul((double)(rv),(double)(cp)))); // mfpbltq.f:76
ce0 = 0x1.99999a0000000p-2f; // mfpbltq.f:77
cm = 0x1.0000000000000p+0f; // mfpbltq.f:77
cq = 0x1.0000000000000p+0f; // mfpbltq.f:77
tkcrt = 0x1.0000000000000p+1f; // mfpbltq.f:77
cmxfac = 0x1.4000000000000p+2f; // mfpbltq.f:77
qmin = 0x1.5798ee0000000p-27f; // mfpbltq.f:78
qlmin = 0x1.1979980000000p-40f; // mfpbltq.f:78
alp = 0x1.8000000000000p+0f; // mfpbltq.f:79
vpertmax = 0x1.8000000000000p+1f; // mfpbltq.f:79
pgcon = 0x1.19999a0000000p-1f; // mfpbltq.f:79
b1 = 0x1.0000000000000p-1f; // mfpbltq.f:80
f1 = 0x1.3333340000000p-3f; // mfpbltq.f:80
// mfpbltq.f:84
totflg = true;
// mfpbltq.f:85
for(i=1; i<=im; i+=1) {
// mfpbltq.f:86
totflg = (totflg && (!cnvflg(i)));
// mfpbltq.f:87
}
// mfpbltq.f:88
if(totflg) return;
// mfpbltq.f:91
dt2 = delt;
// mfpbltq.f:93
for(k=1; k<=km; k+=1) {
// mfpbltq.f:94
for(i=1; i<=im; i+=1) {
// mfpbltq.f:95
if(cnvflg(i)) {
// mfpbltq.f:96
buo(i,k) = 0x0.0p+0f;
// mfpbltq.f:97
wu2(i,k) = 0x0.0p+0f;
// mfpbltq.f:98
qtx(i,k) = sm_add((double)(q1(i,k,1)),(double)(q1(i,k,ntcw)));
// mfpbltq.f:99
}
// mfpbltq.f:100
}
// mfpbltq.f:101
}
// mfpbltq.f:105
for(i=1; i<=im; i+=1) {
// mfpbltq.f:106
if(cnvflg(i)) {
// mfpbltq.f:107
ptem = sm_mul((double)(alp),(double)(vpert(i)));
// mfpbltq.f:108
ptem = sm_min<double>(ptem,vpertmax);
// mfpbltq.f:109
thlu(i,1) = sm_add((double)(thlx(i,1)),(double)(ptem));
// mfpbltq.f:110
qtu(i,1) = qtx(i,1);
// mfpbltq.f:111
buo(i,1) = sm_div((double)(sm_mul((double)(g),(double)(ptem))),(double)(thvx(i,1)));
// mfpbltq.f:112
}
// mfpbltq.f:113
}
// mfpbltq.f:119
for(i=1; i<=im; i+=1) {
// mfpbltq.f:120
if(cnvflg(i)) {
// mfpbltq.f:121
ce0t(i) = sm_mul((double)(ce0),(double)(vez0fun(i)));
// mfpbltq.f:122
if((tkemean(i) > tkcrt)) {
// mfpbltq.f:123
tem = sm_sqrt(sm_div((double)(tkemean(i)),(double)(tkcrt)));
// mfpbltq.f:124
tem1 = sm_min<double>(tem,cmxfac);
// mfpbltq.f:125
tem2 = sm_mul((double)(tem1),(double)(ce0));
// mfpbltq.f:126
ce0t(i) = sm_max<double>(ce0t(i),tem2);
// mfpbltq.f:127
}
// mfpbltq.f:128
}
// mfpbltq.f:129
}
// mfpbltq.f:131
for(i=1; i<=im; i+=1) {
// mfpbltq.f:132
if(cnvflg(i)) {
// mfpbltq.f:133
k = (kpbl(i) / 2);
// mfpbltq.f:134
k = sm_max<int>(k,1);
// mfpbltq.f:135
delz(i) = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfpbltq.f:136
xlamax(i) = sm_div((double)(ce0t(i)),(double)(delz(i)));
// mfpbltq.f:137
}
// mfpbltq.f:138
}
// mfpbltq.f:140
for(k=1; k<=kmpbl; k+=1) {
// mfpbltq.f:141
for(i=1; i<=im; i+=1) {
// mfpbltq.f:142
if(cnvflg(i)) {
// mfpbltq.f:143
if((k < kpbl(i))) {
// mfpbltq.f:144
ptem = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_add((double)(zm(i,k)),(double)(delz(i)))));
// mfpbltq.f:145
tem = sm_max<double>(sm_add((double)(sm_sub((double)(hpbl(i)),(double)(zm(i,k)))),(double)(delz(i))),delz(i));
// mfpbltq.f:146
ptem1 = sm_div((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:147
xlamue(i,k) = sm_mul((double)(ce0t(i)),(double)(sm_add((double)(ptem),(double)(ptem1))));
// mfpbltq.f:148
} else {
// mfpbltq.f:149
xlamue(i,k) = xlamax(i);
// mfpbltq.f:150
}
// mfpbltq.f:152
xlamueq(i,k) = sm_mul((double)(cq),(double)(xlamue(i,k)));
// mfpbltq.f:153
xlamuem(i,k) = sm_mul((double)(cm),(double)(xlamue(i,k)));
// mfpbltq.f:154
}
// mfpbltq.f:155
}
// mfpbltq.f:156
}
// mfpbltq.f:160
for(k=2; k<=kmpbl; k+=1) {
// mfpbltq.f:161
for(i=1; i<=im; i+=1) {
// mfpbltq.f:162
if(cnvflg(i)) {
// mfpbltq.f:163
dz = sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))));
// mfpbltq.f:164
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamue(i,(k - 1))))),(double)(dz));
// mfpbltq.f:165
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:167
thlu(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(thlu(i,(k - 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(thlx(i,(k - 1))),(double)(thlx(i,k)))))))),(double)(factor));
// mfpbltq.f:170
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamueq(i,(k - 1))))),(double)(dz));
// mfpbltq.f:171
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:172
qtu(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qtu(i,(k - 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(qtx(i,(k - 1))),(double)(qtx(i,k)))))))),(double)(factor));
// mfpbltq.f:175
tlu = sm_div((double)(thlu(i,k)),(double)(pix(i,k)));
// mfpbltq.f:176
es = sm_mul((double)(0x1.47ae140000000p-7f),(double)(sm_fpvs(tlu,ws.table)));
// mfpbltq.f:177
qs = sm_max<double>(qmin,sm_div((double)(sm_mul((double)(eps),(double)(es))),(double)(sm_add((double)(plyr(i,k)),(double)(sm_mul((double)(epsm1),(double)(es)))))));
// mfpbltq.f:178
dq = sm_sub((double)(qtu(i,k)),(double)(qs));
// mfpbltq.f:180
if((dq > 0x0.0p+0f)) {
// mfpbltq.f:181
gamma = sm_div((double)(sm_mul((double)(el2orc),(double)(qs))),(double)(sm_powi((double)(tlu),2)));
// mfpbltq.f:182
qlu = sm_div((double)(dq),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(gamma))));
// mfpbltq.f:183
qtu(i,k) = sm_add((double)(qs),(double)(qlu));
// mfpbltq.f:184
tem1 = sm_sub((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(qs))))),(double)(qlu));
// mfpbltq.f:185
thup = sm_add((double)(thlu(i,k)),(double)(sm_mul((double)(sm_mul((double)(pix(i,k)),(double)(elocp))),(double)(qlu))));
// mfpbltq.f:186
thvu = sm_mul((double)(thup),(double)(tem1));
// mfpbltq.f:187
} else {
// mfpbltq.f:188
tem1 = sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(qtu(i,k)))));
// mfpbltq.f:189
thvu = sm_mul((double)(thlu(i,k)),(double)(tem1));
// mfpbltq.f:190
}
// mfpbltq.f:191
buo(i,k) = sm_mul((double)(g),(double)(sm_sub((double)(sm_div((double)(thvu),(double)(thvx(i,k)))),(double)(0x1.0000000000000p+0f))));
// mfpbltq.f:193
}
// mfpbltq.f:194
}
// mfpbltq.f:195
}
// mfpbltq.f:212
bb1 = 0x1.0000000000000p+1f;
// mfpbltq.f:213
bb2 = 0x1.0000000000000p+2f;
// mfpbltq.f:215
for(i=1; i<=im; i+=1) {
// mfpbltq.f:216
if(cnvflg(i)) {
// mfpbltq.f:217
dz = zm(i,1);
// mfpbltq.f:218
tem = sm_mul((double)(sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(bb1))),(double)(xlamue(i,1)))),(double)(dz));
// mfpbltq.f:219
tem1 = sm_mul((double)(sm_mul((double)(bb2),(double)(buo(i,1)))),(double)(dz));
// mfpbltq.f:220
ptem1 = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:221
wu2(i,1) = sm_div((double)(tem1),(double)(ptem1));
// mfpbltq.f:222
}
// mfpbltq.f:223
}
// mfpbltq.f:224
for(k=2; k<=kmpbl; k+=1) {
// mfpbltq.f:225
for(i=1; i<=im; i+=1) {
// mfpbltq.f:226
if(cnvflg(i)) {
// mfpbltq.f:227
dz = sm_sub((double)(zm(i,k)),(double)(zm(i,(k - 1))));
// mfpbltq.f:228
tem = sm_mul((double)(sm_mul((double)(sm_mul((double)(0x1.0000000000000p-2f),(double)(bb1))),(double)(sm_add((double)(xlamue(i,(k - 1))),(double)(xlamue(i,k)))))),(double)(dz));
// mfpbltq.f:229
tem1 = sm_max<double>(wu2(i,(k - 1)),0x0.0p+0f);
// mfpbltq.f:230
tem1 = sm_sub((double)(sm_mul((double)(bb2),(double)(buo(i,k)))),(double)(sm_mul((double)(wush(i,k)),(double)(sm_sqrt(tem1)))));
// mfpbltq.f:231
tem2 = sm_mul((double)(tem1),(double)(dz));
// mfpbltq.f:232
ptem = sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(wu2(i,(k - 1))));
// mfpbltq.f:233
ptem1 = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:234
wu2(i,k) = sm_div((double)(sm_add((double)(ptem),(double)(tem2))),(double)(ptem1));
// mfpbltq.f:235
}
// mfpbltq.f:236
}
// mfpbltq.f:237
}
// mfpbltq.f:241
for(i=1; i<=im; i+=1) {
// mfpbltq.f:242
flg(i) = true;
// mfpbltq.f:243
kpblx(i) = 1;
// mfpbltq.f:244
kpbly(i) = kpbl(i);
// mfpbltq.f:245
if(cnvflg(i)) {
// mfpbltq.f:246
flg(i) = false;
// mfpbltq.f:247
rbup(i) = wu2(i,1);
// mfpbltq.f:248
}
// mfpbltq.f:249
}
// mfpbltq.f:250
for(k=2; k<=kmpbl; k+=1) {
// mfpbltq.f:251
for(i=1; i<=im; i+=1) {
// mfpbltq.f:252
if((!flg(i))) {
// mfpbltq.f:253
rbdn(i) = rbup(i);
// mfpbltq.f:254
rbup(i) = wu2(i,k);
// mfpbltq.f:255
kpblx(i) = k;
// mfpbltq.f:256
flg(i) = (rbup(i) <= 0x0.0p+0f);
// mfpbltq.f:257
}
// mfpbltq.f:258
}
// mfpbltq.f:259
}
// mfpbltq.f:260
for(i=1; i<=im; i+=1) {
// mfpbltq.f:261
if(cnvflg(i)) {
// mfpbltq.f:262
k = kpblx(i);
// mfpbltq.f:263
if((rbdn(i) <= 0x0.0p+0f)) {
// mfpbltq.f:264
rbint = 0x0.0p+0f;
// mfpbltq.f:265
} else if((rbup(i) >= 0x0.0p+0f)) {
// mfpbltq.f:266
rbint = 0x1.0000000000000p+0f;
// mfpbltq.f:267
} else {
// mfpbltq.f:268
rbint = sm_div((double)(rbdn(i)),(double)(sm_sub((double)(rbdn(i)),(double)(rbup(i)))));
// mfpbltq.f:269
}
// mfpbltq.f:270
hpblx(i) = sm_add((double)(zm(i,(k - 1))),(double)(sm_mul((double)(rbint),(double)(sm_sub((double)(zm(i,k)),(double)(zm(i,(k - 1))))))));
// mfpbltq.f:271
}
// mfpbltq.f:272
}
// mfpbltq.f:274
for(i=1; i<=im; i+=1) {
// mfpbltq.f:275
if(cnvflg(i)) {
// mfpbltq.f:276
if((kpblx(i) < kpbl(i))) {
// mfpbltq.f:277
kpbl(i) = kpblx(i);
// mfpbltq.f:278
hpbl(i) = hpblx(i);
// mfpbltq.f:279
}
// mfpbltq.f:280
if((kpbl(i) <= 1)) cnvflg(i) = false;
// mfpbltq.f:281
}
// mfpbltq.f:282
}
// mfpbltq.f:286
for(i=1; i<=im; i+=1) {
// mfpbltq.f:287
if(cnvflg(i)) {
// mfpbltq.f:288
k = (kpbl(i) / 2);
// mfpbltq.f:289
k = sm_max<int>(k,1);
// mfpbltq.f:290
delz(i) = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfpbltq.f:291
xlamax(i) = sm_div((double)(ce0t(i)),(double)(delz(i)));
// mfpbltq.f:292
}
// mfpbltq.f:293
}
// mfpbltq.f:295
for(k=1; k<=kmpbl; k+=1) {
// mfpbltq.f:296
for(i=1; i<=im; i+=1) {
// mfpbltq.f:297
if((cnvflg(i) && (kpblx(i) < kpbly(i)))) {
// mfpbltq.f:299
if((k < kpbl(i))) {
// mfpbltq.f:300
ptem = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_add((double)(zm(i,k)),(double)(delz(i)))));
// mfpbltq.f:301
tem = sm_max<double>(sm_add((double)(sm_sub((double)(hpbl(i)),(double)(zm(i,k)))),(double)(delz(i))),delz(i));
// mfpbltq.f:302
ptem1 = sm_div((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:303
xlamue(i,k) = sm_mul((double)(ce0t(i)),(double)(sm_add((double)(ptem),(double)(ptem1))));
// mfpbltq.f:304
} else {
// mfpbltq.f:305
xlamue(i,k) = xlamax(i);
// mfpbltq.f:306
}
// mfpbltq.f:308
xlamueq(i,k) = sm_mul((double)(cq),(double)(xlamue(i,k)));
// mfpbltq.f:309
xlamuem(i,k) = sm_mul((double)(cm),(double)(xlamue(i,k)));
// mfpbltq.f:310
}
// mfpbltq.f:311
}
// mfpbltq.f:312
}
// mfpbltq.f:316
for(i=1; i<=im; i+=1) {
// mfpbltq.f:317
xlamavg(i) = 0x0.0p+0f;
// mfpbltq.f:318
sumx(i) = 0x0.0p+0f;
// mfpbltq.f:319
}
// mfpbltq.f:320
for(k=1; k<=kmpbl; k+=1) {
// mfpbltq.f:321
for(i=1; i<=im; i+=1) {
// mfpbltq.f:322
if((cnvflg(i) && (k < kpbl(i)))) {
// mfpbltq.f:323
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfpbltq.f:324
xlamavg(i) = sm_add((double)(xlamavg(i)),(double)(sm_mul((double)(xlamue(i,k)),(double)(dz))));
// mfpbltq.f:325
sumx(i) = sm_add((double)(sumx(i)),(double)(dz));
// mfpbltq.f:326
}
// mfpbltq.f:327
}
// mfpbltq.f:328
}
// mfpbltq.f:329
for(i=1; i<=im; i+=1) {
// mfpbltq.f:330
if(cnvflg(i)) {
// mfpbltq.f:331
xlamavg(i) = sm_div((double)(xlamavg(i)),(double)(sumx(i)));
// mfpbltq.f:332
}
// mfpbltq.f:333
}
// mfpbltq.f:337
for(k=1; k<=kmpbl; k+=1) {
// mfpbltq.f:338
for(i=1; i<=im; i+=1) {
// mfpbltq.f:339
if((cnvflg(i) && (k < kpbl(i)))) {
// mfpbltq.f:340
xmf(i,k) = sm_mul((double)(a1),(double)(sm_sqrt(wu2(i,k))));
// mfpbltq.f:341
}
// mfpbltq.f:342
}
// mfpbltq.f:343
}
// mfpbltq.f:348
for(i=1; i<=im; i+=1) {
// mfpbltq.f:349
if(cnvflg(i)) {
// mfpbltq.f:350
tem = sm_div((double)(0x1.99999a0000000p-3f),(double)(xlamavg(i)));
// mfpbltq.f:351
tem1 = sm_mul((double)(sm_mul((double)(0x1.91eb860000000p+1f),(double)(tem))),(double)(tem));
// mfpbltq.f:352
sigma(i) = sm_div((double)(tem1),(double)(sm_mul((double)(gdx(i)),(double)(gdx(i)))));
// mfpbltq.f:353
sigma(i) = sm_max<double>(sigma(i),0x1.0624de0000000p-10f);
// mfpbltq.f:354
sigma(i) = sm_min<double>(sigma(i),0x1.ff7cee0000000p-1f);
// mfpbltq.f:355
}
// mfpbltq.f:356
}
// mfpbltq.f:361
for(i=1; i<=im; i+=1) {
// mfpbltq.f:362
if(cnvflg(i)) {
// mfpbltq.f:363
if((sigma(i) > a1)) {
// mfpbltq.f:364
scaldfunc(i) = sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sigma(i)))),(double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sigma(i)))));
// mfpbltq.f:365
scaldfunc(i) = sm_max<double>(sm_min<double>(scaldfunc(i),0x1.0000000000000p+0f),0x0.0p+0f);
// mfpbltq.f:366
} else {
// mfpbltq.f:367
scaldfunc(i) = 0x1.0000000000000p+0f;
// mfpbltq.f:368
}
// mfpbltq.f:369
}
// mfpbltq.f:370
}
// mfpbltq.f:374
for(k=1; k<=kmpbl; k+=1) {
// mfpbltq.f:375
for(i=1; i<=im; i+=1) {
// mfpbltq.f:376
if((cnvflg(i) && (k < kpbl(i)))) {
// mfpbltq.f:377
xmf(i,k) = sm_mul((double)(scaldfunc(i)),(double)(xmf(i,k)));
// mfpbltq.f:378
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfpbltq.f:379
xmmx = sm_div((double)(dz),(double)(dt2));
// mfpbltq.f:380
xmf(i,k) = sm_min<double>(xmf(i,k),xmmx);
// mfpbltq.f:381
}
// mfpbltq.f:382
}
// mfpbltq.f:383
}
// mfpbltq.f:388
for(i=1; i<=im; i+=1) {
// mfpbltq.f:389
if(cnvflg(i)) {
// mfpbltq.f:390
thlu(i,1) = thlx(i,1);
// mfpbltq.f:391
}
// mfpbltq.f:392
}
// mfpbltq.f:402
for(k=2; k<=kmpbl; k+=1) {
// mfpbltq.f:403
for(i=1; i<=im; i+=1) {
// mfpbltq.f:404
if((cnvflg(i) && (k <= kpbl(i)))) {
// mfpbltq.f:405
dz = sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))));
// mfpbltq.f:406
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamue(i,(k - 1))))),(double)(dz));
// mfpbltq.f:407
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:409
thlu(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(thlu(i,(k - 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(thlx(i,(k - 1))),(double)(thlx(i,k)))))))),(double)(factor));
// mfpbltq.f:412
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamueq(i,(k - 1))))),(double)(dz));
// mfpbltq.f:413
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:414
qtu(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qtu(i,(k - 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(qtx(i,(k - 1))),(double)(qtx(i,k)))))))),(double)(factor));
// mfpbltq.f:417
tlu = sm_div((double)(thlu(i,k)),(double)(pix(i,k)));
// mfpbltq.f:418
es = sm_mul((double)(0x1.47ae140000000p-7f),(double)(sm_fpvs(tlu,ws.table)));
// mfpbltq.f:419
qs = sm_max<double>(qmin,sm_div((double)(sm_mul((double)(eps),(double)(es))),(double)(sm_add((double)(plyr(i,k)),(double)(sm_mul((double)(epsm1),(double)(es)))))));
// mfpbltq.f:420
dq = sm_sub((double)(qtu(i,k)),(double)(qs));
// mfpbltq.f:422
if((dq > 0x0.0p+0f)) {
// mfpbltq.f:423
gamma = sm_div((double)(sm_mul((double)(el2orc),(double)(qs))),(double)(sm_powi((double)(tlu),2)));
// mfpbltq.f:424
qlu = sm_div((double)(dq),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(gamma))));
// mfpbltq.f:425
qtu(i,k) = sm_add((double)(qs),(double)(qlu));
// mfpbltq.f:426
qcko(i,k,1) = qs;
// mfpbltq.f:427
qcko(i,k,ntcw) = qlu;
// mfpbltq.f:428
tcko(i,k) = sm_add((double)(tlu),(double)(sm_mul((double)(elocp),(double)(qlu))));
// mfpbltq.f:429
} else {
// mfpbltq.f:430
qcko(i,k,1) = qtu(i,k);
// mfpbltq.f:431
qcko(i,k,ntcw) = 0x0.0p+0f;
// mfpbltq.f:432
tcko(i,k) = tlu;
// mfpbltq.f:433
}
// mfpbltq.f:435
}
// mfpbltq.f:436
}
// mfpbltq.f:437
}
// mfpbltq.f:439
for(k=2; k<=kmpbl; k+=1) {
// mfpbltq.f:440
for(i=1; i<=im; i+=1) {
// mfpbltq.f:441
if((cnvflg(i) && (k <= kpbl(i)))) {
// mfpbltq.f:442
dz = sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))));
// mfpbltq.f:443
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamuem(i,(k - 1))))),(double)(dz));
// mfpbltq.f:444
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:445
ptem = sm_add((double)(tem),(double)(pgcon));
// mfpbltq.f:446
ptem1 = sm_sub((double)(tem),(double)(pgcon));
// mfpbltq.f:447
ucko(i,k) = sm_div((double)(sm_add((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(ucko(i,(k - 1))))),(double)(sm_mul((double)(ptem),(double)(u1(i,k)))))),(double)(sm_mul((double)(ptem1),(double)(u1(i,(k - 1))))))),(double)(factor));
// mfpbltq.f:449
vcko(i,k) = sm_div((double)(sm_add((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(vcko(i,(k - 1))))),(double)(sm_mul((double)(ptem),(double)(v1(i,k)))))),(double)(sm_mul((double)(ptem1),(double)(v1(i,(k - 1))))))),(double)(factor));
// mfpbltq.f:451
}
// mfpbltq.f:452
}
// mfpbltq.f:453
}
// mfpbltq.f:455
if((ntcw > 2)) {
// mfpbltq.f:457
for(n=2; n<=(ntcw - 1); n+=1) {
// mfpbltq.f:458
for(k=2; k<=kmpbl; k+=1) {
// mfpbltq.f:459
for(i=1; i<=im; i+=1) {
// mfpbltq.f:460
if((cnvflg(i) && (k <= kpbl(i)))) {
// mfpbltq.f:461
dz = sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))));
// mfpbltq.f:462
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamueq(i,(k - 1))))),(double)(dz));
// mfpbltq.f:463
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:465
qcko(i,k,n) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qcko(i,(k - 1),n)))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(q1(i,k,n)),(double)(q1(i,(k - 1),n)))))))),(double)(factor));
// mfpbltq.f:467
}
// mfpbltq.f:468
}
// mfpbltq.f:469
}
// mfpbltq.f:470
}
// mfpbltq.f:472
}
// mfpbltq.f:474
ndc = (ntrac1 - ntcw);
// mfpbltq.f:476
if((ndc > 0)) {
// mfpbltq.f:478
for(n=(ntcw + 1); n<=ntrac1; n+=1) {
// mfpbltq.f:479
for(k=2; k<=kmpbl; k+=1) {
// mfpbltq.f:480
for(i=1; i<=im; i+=1) {
// mfpbltq.f:481
if((cnvflg(i) && (k <= kpbl(i)))) {
// mfpbltq.f:482
dz = sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))));
// mfpbltq.f:483
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamueq(i,(k - 1))))),(double)(dz));
// mfpbltq.f:484
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfpbltq.f:486
qcko(i,k,n) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qcko(i,(k - 1),n)))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(q1(i,k,n)),(double)(q1(i,(k - 1),n)))))))),(double)(factor));
// mfpbltq.f:488
}
// mfpbltq.f:489
}
// mfpbltq.f:490
}
// mfpbltq.f:491
}
// mfpbltq.f:493
}
// mfpbltq.f:495
return;
}
__device__ void sm_mfscuq(int im, int ix, int km, int kmscu, int ntcw, int ntrac1, double delt, bool* raw_cnvflg, double* raw_zl, double* raw_zm, double* raw_q1, double* raw_t1, double* raw_u1, double* raw_v1, double* raw_plyr, double* raw_pix, double* raw_thlx, double* raw_thvx, double* raw_thlvx, double* raw_gdx, double* raw_thetae, int* raw_krad, int* raw_mrad, double* raw_radmin, double* raw_buo, double* raw_wush, double* raw_tkemean, double* raw_vez0fun, double* raw_xmfd, double* raw_tcdo, double* raw_qcdo, double* raw_ucdo, double* raw_vcdo, double* raw_xlamdeq, double a1, SMWorkspace& ws) {
SMFrame frame(ws);
const double grav=sm_g;
const double cp=sm_cp;
const double rv=sm_rv;
const double hvap=sm_hvap;
const double fv=sm_fvirt;
const double eps=sm_eps;
const double epsm1=sm_epsm1;
SMView<int> krad(raw_krad, im, 1, 1, 1, 1, 1);
SMView<int> mrad(raw_mrad, im, 1, 1, 1, 1, 1);
SMView<bool> cnvflg(raw_cnvflg, im, 1, 1, 1, 1, 1);
SMView<double> q1(raw_q1, ix, km, ntrac1, 1, 1, 1);
SMView<double> t1(raw_t1, ix, km, 1, 1, 1, 1);
SMView<double> u1(raw_u1, ix, km, 1, 1, 1, 1);
SMView<double> v1(raw_v1, ix, km, 1, 1, 1, 1);
SMView<double> plyr(raw_plyr, im, km, 1, 1, 1, 1);
SMView<double> pix(raw_pix, im, km, 1, 1, 1, 1);
SMView<double> thlx(raw_thlx, im, km, 1, 1, 1, 1);
SMView<double> thvx(raw_thvx, im, km, 1, 1, 1, 1);
SMView<double> thlvx(raw_thlvx, im, km, 1, 1, 1, 1);
SMView<double> gdx(raw_gdx, im, 1, 1, 1, 1, 1);
SMView<double> zl(raw_zl, im, km, 1, 1, 1, 1);
SMView<double> zm(raw_zm, im, km, 1, 1, 1, 1);
SMView<double> thetae(raw_thetae, im, km, 1, 1, 1, 1);
SMView<double> radmin(raw_radmin, im, 1, 1, 1, 1, 1);
SMView<double> buo(raw_buo, im, km, 1, 1, 1, 1);
SMView<double> wush(raw_wush, im, km, 1, 1, 1, 1);
SMView<double> tkemean(raw_tkemean, im, 1, 1, 1, 1, 1);
SMView<double> vez0fun(raw_vez0fun, im, 1, 1, 1, 1, 1);
SMView<double> xmfd(raw_xmfd, im, km, 1, 1, 1, 1);
SMView<double> tcdo(raw_tcdo, im, km, 1, 1, 1, 1);
SMView<double> qcdo(raw_qcdo, im, km, ntrac1, 1, 1, 1);
SMView<double> ucdo(raw_ucdo, im, km, 1, 1, 1, 1);
SMView<double> vcdo(raw_vcdo, im, km, 1, 1, 1, 1);
SMView<double> xlamdeq(raw_xlamdeq, im, km-1, 1, 1, 1, 1);
int i;
int j;
int indx;
int k;
int n;
int kk;
int ndc;
SMView<int> krad1(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
double dt2;
double dz;
double ce0;
double cm;
double cq;
double tkcrt;
double cmxfac;
double gocp;
double factor;
double g;
double tau;
double b1;
double f1;
double bb1;
double bb2;
double a2;
double cteit;
double pgcon;
double qmin;
double qlmin;
double xmmx;
double tem;
double tem1;
double tem2;
double ptem;
double ptem1;
double ptem2;
double elocp;
double el2orc;
double qs;
double es;
double tld;
double gamma;
double qld;
double thdn;
double thvd;
double dq;
SMView<double> wd2(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> thld(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qtx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qtd(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> thlvd(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> hrad(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> xlamde(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> xlamdem(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> ra1(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> delz(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> xlamax(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> ce0t(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> xlamavg(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> sigma(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> scaldfunc(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> sumx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
bool totflg;
SMView<bool> flg(ws.alloc<bool>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
double actei;
double cldtime;
g = grav; // mfscuq.f:82
gocp = sm_div((double)(g),(double)(cp)); // mfscuq.f:83
elocp = sm_div((double)(hvap),(double)(cp)); // mfscuq.f:84
el2orc = sm_div((double)(sm_mul((double)(hvap),(double)(hvap))),(double)(sm_mul((double)(rv),(double)(cp)))); // mfscuq.f:84
ce0 = 0x1.99999a0000000p-2f; // mfscuq.f:85
cm = 0x1.0000000000000p+0f; // mfscuq.f:85
cq = 0x1.0000000000000p+0f; // mfscuq.f:85
pgcon = 0x1.19999a0000000p-1f; // mfscuq.f:85
tkcrt = 0x1.0000000000000p+1f; // mfscuq.f:86
cmxfac = 0x1.4000000000000p+2f; // mfscuq.f:86
qmin = 0x1.5798ee0000000p-27f; // mfscuq.f:87
qlmin = 0x1.1979980000000p-40f; // mfscuq.f:87
b1 = 0x1.cccccc0000000p-2f; // mfscuq.f:88
f1 = 0x1.3333340000000p-3f; // mfscuq.f:88
a2 = 0x1.0000000000000p-1f; // mfscuq.f:89
cldtime = 0x1.f400000000000p+8f; // mfscuq.f:90
actei = 0x1.6666660000000p-1f; // mfscuq.f:91
// mfscuq.f:96
totflg = true;
// mfscuq.f:97
for(i=1; i<=im; i+=1) {
// mfscuq.f:98
totflg = (totflg && (!cnvflg(i)));
// mfscuq.f:99
}
// mfscuq.f:100
if(totflg) return;
// mfscuq.f:102
dt2 = delt;
// mfscuq.f:104
for(k=1; k<=km; k+=1) {
// mfscuq.f:105
for(i=1; i<=im; i+=1) {
// mfscuq.f:106
if(cnvflg(i)) {
// mfscuq.f:107
buo(i,k) = 0x0.0p+0f;
// mfscuq.f:108
wd2(i,k) = 0x0.0p+0f;
// mfscuq.f:109
qtx(i,k) = sm_add((double)(q1(i,k,1)),(double)(q1(i,k,ntcw)));
// mfscuq.f:110
}
// mfscuq.f:111
}
// mfscuq.f:112
}
// mfscuq.f:114
for(i=1; i<=im; i+=1) {
// mfscuq.f:115
if(cnvflg(i)) {
// mfscuq.f:116
hrad(i) = zm(i,krad(i));
// mfscuq.f:117
krad1(i) = (krad(i) - 1);
// mfscuq.f:118
}
// mfscuq.f:119
}
// mfscuq.f:121
for(i=1; i<=im; i+=1) {
// mfscuq.f:122
if(cnvflg(i)) {
// mfscuq.f:123
k = krad(i);
// mfscuq.f:124
tem = sm_sub((double)(zm(i,(k + 1))),(double)(zm(i,k)));
// mfscuq.f:125
tem1 = sm_div((double)(sm_mul((double)(cldtime),(double)(radmin(i)))),(double)(tem));
// mfscuq.f:126
tem1 = sm_max<double>(tem1,(-0x1.8000000000000p+1f));
// mfscuq.f:127
thld(i,k) = sm_add((double)(thlx(i,k)),(double)(tem1));
// mfscuq.f:128
qtd(i,k) = qtx(i,k);
// mfscuq.f:129
thlvd(i) = sm_add((double)(thlvx(i,k)),(double)(tem1));
// mfscuq.f:130
buo(i,k) = sm_div((double)(sm_mul((double)((-g)),(double)(tem1))),(double)(thvx(i,k)));
// mfscuq.f:131
}
// mfscuq.f:132
}
// mfscuq.f:136
for(i=1; i<=im; i+=1) {
// mfscuq.f:137
if(cnvflg(i)) {
// mfscuq.f:138
ra1(i) = a1;
// mfscuq.f:139
}
// mfscuq.f:140
}
// mfscuq.f:145
for(i=1; i<=im; i+=1) {
// mfscuq.f:146
if(cnvflg(i)) {
// mfscuq.f:147
k = krad(i);
// mfscuq.f:148
tem = sm_sub((double)(thetae(i,k)),(double)(thetae(i,(k + 1))));
// mfscuq.f:149
tem1 = sm_sub((double)(qtx(i,k)),(double)(qtx(i,(k + 1))));
// mfscuq.f:150
if(((tem > 0x0.0p+0f) && (tem1 > 0x0.0p+0f))) {
// mfscuq.f:151
cteit = sm_div((double)(sm_mul((double)(cp),(double)(tem))),(double)(sm_mul((double)(hvap),(double)(tem1))));
// mfscuq.f:152
if((cteit > actei)) {
// mfscuq.f:153
ra1(i) = a2;
// mfscuq.f:154
}
// mfscuq.f:155
}
// mfscuq.f:156
}
// mfscuq.f:157
}
// mfscuq.f:161
for(i=1; i<=im; i+=1) {
// mfscuq.f:162
flg(i) = cnvflg(i);
// mfscuq.f:163
mrad(i) = krad(i);
// mfscuq.f:164
}
// mfscuq.f:165
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:166
for(i=1; i<=im; i+=1) {
// mfscuq.f:167
if((flg(i) && (k < krad(i)))) {
// mfscuq.f:168
if((thlvd(i) <= thlvx(i,k))) {
// mfscuq.f:169
mrad(i) = k;
// mfscuq.f:170
} else {
// mfscuq.f:171
flg(i) = false;
// mfscuq.f:172
}
// mfscuq.f:173
}
// mfscuq.f:174
}
// mfscuq.f:175
}
// mfscuq.f:176
for(i=1; i<=im; i+=1) {
// mfscuq.f:177
if(cnvflg(i)) {
// mfscuq.f:178
kk = (krad(i) - mrad(i));
// mfscuq.f:179
if((kk < 1)) cnvflg(i) = false;
// mfscuq.f:180
}
// mfscuq.f:181
}
// mfscuq.f:183
totflg = true;
// mfscuq.f:184
for(i=1; i<=im; i+=1) {
// mfscuq.f:185
totflg = (totflg && (!cnvflg(i)));
// mfscuq.f:186
}
// mfscuq.f:187
if(totflg) return;
// mfscuq.f:194
for(i=1; i<=im; i+=1) {
// mfscuq.f:195
if(cnvflg(i)) {
// mfscuq.f:196
ce0t(i) = sm_mul((double)(ce0),(double)(vez0fun(i)));
// mfscuq.f:197
if((tkemean(i) > tkcrt)) {
// mfscuq.f:198
tem = sm_sqrt(sm_div((double)(tkemean(i)),(double)(tkcrt)));
// mfscuq.f:199
tem1 = sm_min<double>(tem,cmxfac);
// mfscuq.f:200
tem2 = sm_mul((double)(tem1),(double)(ce0));
// mfscuq.f:201
ce0t(i) = sm_max<double>(ce0t(i),tem2);
// mfscuq.f:202
}
// mfscuq.f:203
}
// mfscuq.f:204
}
// mfscuq.f:206
for(i=1; i<=im; i+=1) {
// mfscuq.f:207
if(cnvflg(i)) {
// mfscuq.f:208
k = (mrad(i) + ((krad(i) - mrad(i)) / 2));
// mfscuq.f:209
k = sm_max<int>(k,mrad(i));
// mfscuq.f:210
delz(i) = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:211
xlamax(i) = sm_div((double)(ce0t(i)),(double)(delz(i)));
// mfscuq.f:212
}
// mfscuq.f:213
}
// mfscuq.f:215
for(k=1; k<=kmscu; k+=1) {
// mfscuq.f:216
for(i=1; i<=im; i+=1) {
// mfscuq.f:217
if(cnvflg(i)) {
// mfscuq.f:218
if(((k >= mrad(i)) && (k < krad(i)))) {
// mfscuq.f:219
if((mrad(i) == 1)) {
// mfscuq.f:220
ptem = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_add((double)(zm(i,k)),(double)(delz(i)))));
// mfscuq.f:221
} else {
// mfscuq.f:222
ptem = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_add((double)(sm_sub((double)(zm(i,k)),(double)(zm(i,(mrad(i) - 1))))),(double)(delz(i)))));
// mfscuq.f:223
}
// mfscuq.f:224
tem = sm_max<double>(sm_add((double)(sm_sub((double)(hrad(i)),(double)(zm(i,k)))),(double)(delz(i))),delz(i));
// mfscuq.f:225
ptem1 = sm_div((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:226
xlamde(i,k) = sm_mul((double)(ce0t(i)),(double)(sm_add((double)(ptem),(double)(ptem1))));
// mfscuq.f:227
} else {
// mfscuq.f:228
xlamde(i,k) = xlamax(i);
// mfscuq.f:229
}
// mfscuq.f:231
xlamdeq(i,k) = sm_mul((double)(cq),(double)(xlamde(i,k)));
// mfscuq.f:232
xlamdem(i,k) = sm_mul((double)(cm),(double)(xlamde(i,k)));
// mfscuq.f:233
}
// mfscuq.f:234
}
// mfscuq.f:235
}
// mfscuq.f:239
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:240
for(i=1; i<=im; i+=1) {
// mfscuq.f:241
if((cnvflg(i) && (k < krad(i)))) {
// mfscuq.f:242
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:243
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamde(i,k)))),(double)(dz));
// mfscuq.f:244
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:246
thld(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(thld(i,(k + 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(thlx(i,k)),(double)(thlx(i,(k + 1))))))))),(double)(factor));
// mfscuq.f:249
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamdeq(i,k)))),(double)(dz));
// mfscuq.f:250
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:251
qtd(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qtd(i,(k + 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(qtx(i,k)),(double)(qtx(i,(k + 1))))))))),(double)(factor));
// mfscuq.f:254
tld = sm_div((double)(thld(i,k)),(double)(pix(i,k)));
// mfscuq.f:255
es = sm_mul((double)(0x1.47ae140000000p-7f),(double)(sm_fpvs(tld,ws.table)));
// mfscuq.f:256
qs = sm_max<double>(qmin,sm_div((double)(sm_mul((double)(eps),(double)(es))),(double)(sm_add((double)(plyr(i,k)),(double)(sm_mul((double)(epsm1),(double)(es)))))));
// mfscuq.f:257
dq = sm_sub((double)(qtd(i,k)),(double)(qs));
// mfscuq.f:259
if((dq > 0x0.0p+0f)) {
// mfscuq.f:260
gamma = sm_div((double)(sm_mul((double)(el2orc),(double)(qs))),(double)(sm_powi((double)(tld),2)));
// mfscuq.f:261
qld = sm_div((double)(dq),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(gamma))));
// mfscuq.f:262
qtd(i,k) = sm_add((double)(qs),(double)(qld));
// mfscuq.f:263
tem1 = sm_sub((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(qs))))),(double)(qld));
// mfscuq.f:264
thdn = sm_add((double)(thld(i,k)),(double)(sm_mul((double)(sm_mul((double)(pix(i,k)),(double)(elocp))),(double)(qld))));
// mfscuq.f:265
thvd = sm_mul((double)(thdn),(double)(tem1));
// mfscuq.f:266
} else {
// mfscuq.f:267
tem1 = sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(qtd(i,k)))));
// mfscuq.f:268
thvd = sm_mul((double)(thld(i,k)),(double)(tem1));
// mfscuq.f:269
}
// mfscuq.f:270
buo(i,k) = sm_mul((double)(g),(double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sm_div((double)(thvd),(double)(thvx(i,k)))))));
// mfscuq.f:272
}
// mfscuq.f:273
}
// mfscuq.f:274
}
// mfscuq.f:290
bb1 = 0x1.0000000000000p+1f;
// mfscuq.f:291
bb2 = 0x1.0000000000000p+2f;
// mfscuq.f:293
for(i=1; i<=im; i+=1) {
// mfscuq.f:294
if(cnvflg(i)) {
// mfscuq.f:295
k = krad1(i);
// mfscuq.f:296
dz = sm_sub((double)(zm(i,(k + 1))),(double)(zm(i,k)));
// mfscuq.f:298
tem = sm_mul((double)(sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(bb1))),(double)(xlamde(i,k)))),(double)(dz));
// mfscuq.f:299
tem1 = sm_mul((double)(sm_mul((double)(bb2),(double)(buo(i,(k + 1))))),(double)(dz));
// mfscuq.f:300
ptem1 = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:301
wd2(i,k) = sm_div((double)(tem1),(double)(ptem1));
// mfscuq.f:302
}
// mfscuq.f:303
}
// mfscuq.f:304
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:305
for(i=1; i<=im; i+=1) {
// mfscuq.f:306
if((cnvflg(i) && (k < krad1(i)))) {
// mfscuq.f:307
dz = sm_sub((double)(zm(i,(k + 1))),(double)(zm(i,k)));
// mfscuq.f:308
tem = sm_mul((double)(sm_mul((double)(sm_mul((double)(0x1.0000000000000p-2f),(double)(bb1))),(double)(sm_add((double)(xlamde(i,k)),(double)(xlamde(i,(k + 1))))))),(double)(dz));
// mfscuq.f:309
tem1 = sm_max<double>(wd2(i,(k + 1)),0x0.0p+0f);
// mfscuq.f:310
tem1 = sm_sub((double)(sm_mul((double)(bb2),(double)(buo(i,(k + 1))))),(double)(sm_mul((double)(wush(i,(k + 1))),(double)(sm_sqrt(tem1)))));
// mfscuq.f:311
tem2 = sm_mul((double)(tem1),(double)(dz));
// mfscuq.f:312
ptem = sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(wd2(i,(k + 1))));
// mfscuq.f:313
ptem1 = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:314
wd2(i,k) = sm_div((double)(sm_add((double)(ptem),(double)(tem2))),(double)(ptem1));
// mfscuq.f:315
}
// mfscuq.f:316
}
// mfscuq.f:317
}
// mfscuq.f:319
for(i=1; i<=im; i+=1) {
// mfscuq.f:320
flg(i) = cnvflg(i);
// mfscuq.f:321
if(flg(i)) mrad(i) = krad(i);
// mfscuq.f:322
}
// mfscuq.f:323
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:324
for(i=1; i<=im; i+=1) {
// mfscuq.f:325
if((flg(i) && (k < krad(i)))) {
// mfscuq.f:326
if((wd2(i,k) > 0x0.0p+0f)) {
// mfscuq.f:327
mrad(i) = k;
// mfscuq.f:328
} else {
// mfscuq.f:329
flg(i) = false;
// mfscuq.f:330
}
// mfscuq.f:331
}
// mfscuq.f:332
}
// mfscuq.f:333
}
// mfscuq.f:335
for(i=1; i<=im; i+=1) {
// mfscuq.f:336
if(cnvflg(i)) {
// mfscuq.f:337
kk = (krad(i) - mrad(i));
// mfscuq.f:338
if((kk < 1)) cnvflg(i) = false;
// mfscuq.f:339
}
// mfscuq.f:340
}
// mfscuq.f:342
totflg = true;
// mfscuq.f:343
for(i=1; i<=im; i+=1) {
// mfscuq.f:344
totflg = (totflg && (!cnvflg(i)));
// mfscuq.f:345
}
// mfscuq.f:346
if(totflg) return;
// mfscuq.f:351
for(i=1; i<=im; i+=1) {
// mfscuq.f:352
if(cnvflg(i)) {
// mfscuq.f:353
k = (mrad(i) + ((krad(i) - mrad(i)) / 2));
// mfscuq.f:354
k = sm_max<int>(k,mrad(i));
// mfscuq.f:355
delz(i) = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:356
xlamax(i) = sm_div((double)(ce0t(i)),(double)(delz(i)));
// mfscuq.f:357
}
// mfscuq.f:358
}
// mfscuq.f:360
for(k=1; k<=kmscu; k+=1) {
// mfscuq.f:361
for(i=1; i<=im; i+=1) {
// mfscuq.f:362
if(cnvflg(i)) {
// mfscuq.f:363
if(((k >= mrad(i)) && (k < krad(i)))) {
// mfscuq.f:364
if((mrad(i) == 1)) {
// mfscuq.f:365
ptem = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_add((double)(zm(i,k)),(double)(delz(i)))));
// mfscuq.f:366
} else {
// mfscuq.f:367
ptem = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_add((double)(sm_sub((double)(zm(i,k)),(double)(zm(i,(mrad(i) - 1))))),(double)(delz(i)))));
// mfscuq.f:368
}
// mfscuq.f:369
tem = sm_max<double>(sm_add((double)(sm_sub((double)(hrad(i)),(double)(zm(i,k)))),(double)(delz(i))),delz(i));
// mfscuq.f:370
ptem1 = sm_div((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:371
xlamde(i,k) = sm_mul((double)(ce0t(i)),(double)(sm_add((double)(ptem),(double)(ptem1))));
// mfscuq.f:372
} else {
// mfscuq.f:373
xlamde(i,k) = xlamax(i);
// mfscuq.f:374
}
// mfscuq.f:376
xlamdeq(i,k) = sm_mul((double)(cq),(double)(xlamde(i,k)));
// mfscuq.f:377
xlamdem(i,k) = sm_mul((double)(cm),(double)(xlamde(i,k)));
// mfscuq.f:378
}
// mfscuq.f:379
}
// mfscuq.f:380
}
// mfscuq.f:384
for(i=1; i<=im; i+=1) {
// mfscuq.f:385
xlamavg(i) = 0x0.0p+0f;
// mfscuq.f:386
sumx(i) = 0x0.0p+0f;
// mfscuq.f:387
}
// mfscuq.f:388
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:389
for(i=1; i<=im; i+=1) {
// mfscuq.f:390
if((cnvflg(i) && ((k >= mrad(i)) && (k < krad(i))))) {
// mfscuq.f:392
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:393
xlamavg(i) = sm_add((double)(xlamavg(i)),(double)(sm_mul((double)(xlamde(i,k)),(double)(dz))));
// mfscuq.f:394
sumx(i) = sm_add((double)(sumx(i)),(double)(dz));
// mfscuq.f:395
}
// mfscuq.f:396
}
// mfscuq.f:397
}
// mfscuq.f:398
for(i=1; i<=im; i+=1) {
// mfscuq.f:399
if(cnvflg(i)) {
// mfscuq.f:400
xlamavg(i) = sm_div((double)(xlamavg(i)),(double)(sumx(i)));
// mfscuq.f:401
}
// mfscuq.f:402
}
// mfscuq.f:406
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:407
for(i=1; i<=im; i+=1) {
// mfscuq.f:408
if((cnvflg(i) && ((k >= mrad(i)) && (k < krad(i))))) {
// mfscuq.f:410
xmfd(i,k) = sm_mul((double)(ra1(i)),(double)(sm_sqrt(wd2(i,k))));
// mfscuq.f:411
}
// mfscuq.f:412
}
// mfscuq.f:413
}
// mfscuq.f:418
for(i=1; i<=im; i+=1) {
// mfscuq.f:419
if(cnvflg(i)) {
// mfscuq.f:420
tem = sm_div((double)(0x1.99999a0000000p-3f),(double)(xlamavg(i)));
// mfscuq.f:421
tem1 = sm_mul((double)(sm_mul((double)(0x1.91eb860000000p+1f),(double)(tem))),(double)(tem));
// mfscuq.f:422
sigma(i) = sm_div((double)(tem1),(double)(sm_mul((double)(gdx(i)),(double)(gdx(i)))));
// mfscuq.f:423
sigma(i) = sm_max<double>(sigma(i),0x1.0624de0000000p-10f);
// mfscuq.f:424
sigma(i) = sm_min<double>(sigma(i),0x1.ff7cee0000000p-1f);
// mfscuq.f:425
}
// mfscuq.f:426
}
// mfscuq.f:431
for(i=1; i<=im; i+=1) {
// mfscuq.f:432
if(cnvflg(i)) {
// mfscuq.f:433
if((sigma(i) > ra1(i))) {
// mfscuq.f:434
scaldfunc(i) = sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sigma(i)))),(double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sigma(i)))));
// mfscuq.f:435
scaldfunc(i) = sm_max<double>(sm_min<double>(scaldfunc(i),0x1.0000000000000p+0f),0x0.0p+0f);
// mfscuq.f:436
} else {
// mfscuq.f:437
scaldfunc(i) = 0x1.0000000000000p+0f;
// mfscuq.f:438
}
// mfscuq.f:439
}
// mfscuq.f:440
}
// mfscuq.f:444
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:445
for(i=1; i<=im; i+=1) {
// mfscuq.f:446
if((cnvflg(i) && ((k >= mrad(i)) && (k < krad(i))))) {
// mfscuq.f:448
xmfd(i,k) = sm_mul((double)(scaldfunc(i)),(double)(xmfd(i,k)));
// mfscuq.f:449
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:450
xmmx = sm_div((double)(dz),(double)(dt2));
// mfscuq.f:451
xmfd(i,k) = sm_min<double>(xmfd(i,k),xmmx);
// mfscuq.f:452
}
// mfscuq.f:453
}
// mfscuq.f:454
}
// mfscuq.f:459
for(i=1; i<=im; i+=1) {
// mfscuq.f:460
if(cnvflg(i)) {
// mfscuq.f:461
k = krad(i);
// mfscuq.f:462
thld(i,k) = thlx(i,k);
// mfscuq.f:463
}
// mfscuq.f:464
}
// mfscuq.f:477
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:478
for(i=1; i<=im; i+=1) {
// mfscuq.f:479
if((cnvflg(i) && ((k >= mrad(i)) && (k < krad(i))))) {
// mfscuq.f:481
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:482
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamde(i,k)))),(double)(dz));
// mfscuq.f:483
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:485
thld(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(thld(i,(k + 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(thlx(i,k)),(double)(thlx(i,(k + 1))))))))),(double)(factor));
// mfscuq.f:488
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamdeq(i,k)))),(double)(dz));
// mfscuq.f:489
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:490
qtd(i,k) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qtd(i,(k + 1))))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(qtx(i,k)),(double)(qtx(i,(k + 1))))))))),(double)(factor));
// mfscuq.f:493
tld = sm_div((double)(thld(i,k)),(double)(pix(i,k)));
// mfscuq.f:494
es = sm_mul((double)(0x1.47ae140000000p-7f),(double)(sm_fpvs(tld,ws.table)));
// mfscuq.f:495
qs = sm_max<double>(qmin,sm_div((double)(sm_mul((double)(eps),(double)(es))),(double)(sm_add((double)(plyr(i,k)),(double)(sm_mul((double)(epsm1),(double)(es)))))));
// mfscuq.f:496
dq = sm_sub((double)(qtd(i,k)),(double)(qs));
// mfscuq.f:498
if((dq > 0x0.0p+0f)) {
// mfscuq.f:499
gamma = sm_div((double)(sm_mul((double)(el2orc),(double)(qs))),(double)(sm_powi((double)(tld),2)));
// mfscuq.f:500
qld = sm_div((double)(dq),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(gamma))));
// mfscuq.f:501
qtd(i,k) = sm_add((double)(qs),(double)(qld));
// mfscuq.f:502
qcdo(i,k,1) = qs;
// mfscuq.f:503
qcdo(i,k,ntcw) = qld;
// mfscuq.f:504
tcdo(i,k) = sm_add((double)(tld),(double)(sm_mul((double)(elocp),(double)(qld))));
// mfscuq.f:505
} else {
// mfscuq.f:506
qcdo(i,k,1) = qtd(i,k);
// mfscuq.f:507
qcdo(i,k,ntcw) = 0x0.0p+0f;
// mfscuq.f:508
tcdo(i,k) = tld;
// mfscuq.f:509
}
// mfscuq.f:511
}
// mfscuq.f:512
}
// mfscuq.f:513
}
// mfscuq.f:515
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:516
for(i=1; i<=im; i+=1) {
// mfscuq.f:517
if((cnvflg(i) && (k < krad(i)))) {
// mfscuq.f:518
if((k >= mrad(i))) {
// mfscuq.f:519
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:520
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamdem(i,k)))),(double)(dz));
// mfscuq.f:521
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:522
ptem = sm_sub((double)(tem),(double)(pgcon));
// mfscuq.f:523
ptem1 = sm_add((double)(tem),(double)(pgcon));
// mfscuq.f:525
ucdo(i,k) = sm_div((double)(sm_add((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(ucdo(i,(k + 1))))),(double)(sm_mul((double)(ptem),(double)(u1(i,(k + 1))))))),(double)(sm_mul((double)(ptem1),(double)(u1(i,k)))))),(double)(factor));
// mfscuq.f:527
vcdo(i,k) = sm_div((double)(sm_add((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(vcdo(i,(k + 1))))),(double)(sm_mul((double)(ptem),(double)(v1(i,(k + 1))))))),(double)(sm_mul((double)(ptem1),(double)(v1(i,k)))))),(double)(factor));
// mfscuq.f:529
}
// mfscuq.f:530
}
// mfscuq.f:531
}
// mfscuq.f:532
}
// mfscuq.f:534
if((ntcw > 2)) {
// mfscuq.f:536
for(n=2; n<=(ntcw - 1); n+=1) {
// mfscuq.f:537
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:538
for(i=1; i<=im; i+=1) {
// mfscuq.f:539
if((cnvflg(i) && (k < krad(i)))) {
// mfscuq.f:540
if((k >= mrad(i))) {
// mfscuq.f:541
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:542
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamdeq(i,k)))),(double)(dz));
// mfscuq.f:543
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:545
qcdo(i,k,n) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qcdo(i,(k + 1),n)))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(q1(i,k,n)),(double)(q1(i,(k + 1),n)))))))),(double)(factor));
// mfscuq.f:547
}
// mfscuq.f:548
}
// mfscuq.f:549
}
// mfscuq.f:550
}
// mfscuq.f:551
}
// mfscuq.f:553
}
// mfscuq.f:555
ndc = (ntrac1 - ntcw);
// mfscuq.f:557
if((ndc > 0)) {
// mfscuq.f:559
for(n=(ntcw + 1); n<=ntrac1; n+=1) {
// mfscuq.f:560
for(k=kmscu; k>=1; k+=(-1)) {
// mfscuq.f:561
for(i=1; i<=im; i+=1) {
// mfscuq.f:562
if((cnvflg(i) && (k < krad(i)))) {
// mfscuq.f:563
if((k >= mrad(i))) {
// mfscuq.f:564
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// mfscuq.f:565
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamdeq(i,k)))),(double)(dz));
// mfscuq.f:566
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// mfscuq.f:568
qcdo(i,k,n) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qcdo(i,(k + 1),n)))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(q1(i,k,n)),(double)(q1(i,(k + 1),n)))))))),(double)(factor));
// mfscuq.f:570
}
// mfscuq.f:571
}
// mfscuq.f:572
}
// mfscuq.f:573
}
// mfscuq.f:574
}
// mfscuq.f:576
}
// mfscuq.f:578
return;
}
__device__ void sm_satmedmfvdifq_run(int im, int km, int ntrac, int ntcw, int ntrw, int ntiw, int ntke, double grav, double rd, double cp, double rv, double hvap, double hfus, double fv, double eps, double epsm1, double* raw_dv, double* raw_du, double* raw_tdt, double* raw_rtg, double* raw_u1, double* raw_v1, double* raw_t1, double* raw_q1, double* raw_usfco, double* raw_vsfco, int icplocn2atm, double* raw_swh, double* raw_hlw, double* raw_xmu, double* raw_garea, double* raw_zvfun, double* raw_sigmaf, double* raw_psk, double* raw_rbsoil, double* raw_zorl, double* raw_u10m, double* raw_v10m, double* raw_fm, double* raw_fh, double* raw_tsea, double* raw_heat, double* raw_evap, double* raw_stress, double* raw_spd1, int* raw_kpbl, double* raw_prsi, double* raw_del, double* raw_prsl, double* raw_prslk, double* raw_phii, double* raw_phil, double delt, bool dspheat, double* raw_dusfc, double* raw_dvsfc, double* raw_dtsfc, double* raw_dqsfc, double* raw_hpbl, double* raw_dkt, double* raw_dku, int* raw_kinver, double xkzm_m, double xkzm_h, double xkzm_s, double dspfac, double bl_upfr, double bl_dnfr, double rlmx, double elmx, int sfc_rlm, int tc_pbl, int ntqv, double* raw_dtend, int* raw_dtidx, int index_of_temperature, int index_of_x_wind, int index_of_y_wind, int index_of_process_pbl, bool gen_tend, bool ldiag3d, char* errmsg, int& errflg, SMWorkspace& ws) {
SMFrame frame(ws);
SMView<int> kinver(raw_kinver, im, 1, 1, 1, 1, 1);
SMView<int> kpbl(raw_kpbl, im, 1, 1, 1, 1, 1);
SMView<double> dv(raw_dv, im, km, 1, 1, 1, 1);
SMView<double> du(raw_du, im, km, 1, 1, 1, 1);
SMView<double> tdt(raw_tdt, im, km, 1, 1, 1, 1);
SMView<double> rtg(raw_rtg, im, km, ntrac, 1, 1, 1);
SMView<double> u1(raw_u1, im, km, 1, 1, 1, 1);
SMView<double> v1(raw_v1, im, km, 1, 1, 1, 1);
SMView<double> usfco(raw_usfco, im, 1, 1, 1, 1, 1);
SMView<double> vsfco(raw_vsfco, im, 1, 1, 1, 1, 1);
SMView<double> t1(raw_t1, im, km, 1, 1, 1, 1);
SMView<double> q1(raw_q1, im, km, ntrac, 1, 1, 1);
SMView<double> swh(raw_swh, im, km, 1, 1, 1, 1);
SMView<double> hlw(raw_hlw, im, km, 1, 1, 1, 1);
SMView<double> xmu(raw_xmu, im, 1, 1, 1, 1, 1);
SMView<double> garea(raw_garea, im, 1, 1, 1, 1, 1);
SMView<double> zvfun(raw_zvfun, im, 1, 1, 1, 1, 1);
SMView<double> sigmaf(raw_sigmaf, im, 1, 1, 1, 1, 1);
SMView<double> psk(raw_psk, im, 1, 1, 1, 1, 1);
SMView<double> rbsoil(raw_rbsoil, im, 1, 1, 1, 1, 1);
SMView<double> zorl(raw_zorl, im, 1, 1, 1, 1, 1);
SMView<double> tsea(raw_tsea, im, 1, 1, 1, 1, 1);
SMView<double> u10m(raw_u10m, im, 1, 1, 1, 1, 1);
SMView<double> v10m(raw_v10m, im, 1, 1, 1, 1, 1);
SMView<double> fm(raw_fm, im, 1, 1, 1, 1, 1);
SMView<double> fh(raw_fh, im, 1, 1, 1, 1, 1);
SMView<double> evap(raw_evap, im, 1, 1, 1, 1, 1);
SMView<double> heat(raw_heat, im, 1, 1, 1, 1, 1);
SMView<double> stress(raw_stress, im, 1, 1, 1, 1, 1);
SMView<double> spd1(raw_spd1, im, 1, 1, 1, 1, 1);
SMView<double> prsi(raw_prsi, im, km+1, 1, 1, 1, 1);
SMView<double> del(raw_del, im, km, 1, 1, 1, 1);
SMView<double> prsl(raw_prsl, im, km, 1, 1, 1, 1);
SMView<double> prslk(raw_prslk, im, km, 1, 1, 1, 1);
SMView<double> phii(raw_phii, im, km+1, 1, 1, 1, 1);
SMView<double> phil(raw_phil, im, km, 1, 1, 1, 1);
SMView<double> dtend(raw_dtend, im, km, ntrac, 1, 1, 1);
SMView<int> dtidx(raw_dtidx, 100+ntrac, 1, 1, 1, 1, 1);
SMView<double> dusfc(raw_dusfc, im, 1, 1, 1, 1, 1);
SMView<double> dvsfc(raw_dvsfc, im, 1, 1, 1, 1, 1);
SMView<double> dtsfc(raw_dtsfc, im, 1, 1, 1, 1, 1);
SMView<double> dqsfc(raw_dqsfc, im, 1, 1, 1, 1, 1);
SMView<double> hpbl(raw_hpbl, im, 1, 1, 1, 1, 1);
SMView<double> dkt(raw_dkt, im, km, 1, 1, 1, 1);
SMView<double> dku(raw_dku, im, km, 1, 1, 1, 1);
double spd1_m;
int i;
int is;
int k;
int n;
int ndt;
int km1;
int kmpbl;
int kmscu;
int ntrac1;
int idtend;
int kps;
int kbx;
int kmx;
SMView<int> lcld(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<int> kcld(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<int> krad(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<int> mrad(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<int> kx1(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<int> kb1(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<int> kpblx(ws.alloc<int>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> tke(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> tkeh(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> e2(ws.alloc<double>((im)*(((km)-(0)+1))*(1)), im, ((km)-(0)+1), 1, 1, 0, 1);
SMView<double> theta(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> thvx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> thlvx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qlx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> thetae(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> thlx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> slx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> svx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qtx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> tvx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> pix(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> radx(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> dkq(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> cku(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> ckt(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> plyr(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> rhly(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> cfly(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qstl(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> dtdz1(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> gdx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> phih(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> phim(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> phims(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> prn(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> rbdn(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> rbup(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> thermal(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> ustar(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> wstar(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> hpblx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> ust3(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> wst3(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> rho_a(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> z0(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> crb(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> tkemean(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> hgamt(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> hgamq(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> wscale(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> vpert(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> zol(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> sflux(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> sumx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> tx1(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> tx2(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> radmin(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> zi(ws.alloc<double>((im)*(km+1)*(1)), im, km+1, 1, 1, 1, 1);
SMView<double> zl(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> zm(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> xkzo(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> xkzmo(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> xkzm_hx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> xkzm_mx(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> tkmnz(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> rdzt(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> rlmnz(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> al(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> ad(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> au(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> f1(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> f2(ws.alloc<double>((im)*(km*(ntrac-1))*(1)), im, km*(ntrac-1), 1, 1, 1, 1);
SMView<double> elm(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> ele(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> ckz(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> chz(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> diss(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> prod(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> bf(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> shr2(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> wush(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> xlamue(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> xlamde(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> gotvx(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> rlam(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> tcko(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qcko(ws.alloc<double>((im)*(km)*(ntrac)), im, km, ntrac, 1, 1, 1);
SMView<double> ucko(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> vcko(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> buou(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> xmf(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> tcdo(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> qcdo(ws.alloc<double>((im)*(km)*(ntrac)), im, km, ntrac, 1, 1, 1);
SMView<double> ucdo(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> vcdo(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> buod(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> xmfd(ws.alloc<double>((im)*(km)*(1)), im, km, 1, 1, 1, 1);
SMView<double> e_half(ws.alloc<double>((im)*(km-1)*(1)), im, km-1, 1, 1, 1, 1);
SMView<double> e_diff(ws.alloc<double>((im)*(((km-1)-(0)+1))*(1)), im, ((km-1)-(0)+1), 1, 1, 0, 1);
SMView<double> q_half(ws.alloc<double>((im)*(km-1)*(ntrac-1)), im, km-1, ntrac-1, 1, 1, 1);
SMView<double> qh(ws.alloc<double>((im)*(km-1)*(ntrac-1)), im, km-1, ntrac-1, 1, 1, 1);
SMView<double> q_diff(ws.alloc<double>((im)*(((km-1)-(0)+1))*(ntrac-1)), im, ((km-1)-(0)+1), ntrac-1, 1, 0, 1);
double rrkp;
double phkp;
SMView<double> tsumn(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> tsump(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> rtnp(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> sfcpbl(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<double> vez0fun(ws.alloc<double>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<bool> pblflg(ws.alloc<bool>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<bool> sfcflg(ws.alloc<bool>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<bool> flg(ws.alloc<bool>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<bool> scuflg(ws.alloc<bool>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
SMView<bool> pcnvflg(ws.alloc<bool>((im)*(1)*(1)), im, 1, 1, 1, 1, 1);
bool mlenflg;
double aphi16;
double aphi5;
double wfac;
double cfac;
double gamcrt;
double gamcrq;
double sfcfrac;
double dsdz2;
double dsdzt;
double dkmax;
double dsig;
double dt2;
double dtodsd;
double dtodsu;
double g;
double factor;
double dz;
double gocp;
double gravi;
double zol1;
double zolcru;
double buop;
double shrp;
double dtn;
double prnum;
double prmax;
double prmin;
double prtke;
double prscu;
double pr0;
double ri;
double dw2;
double dw2min;
double zk;
double elmfac;
double elefac;
double dspmax;
double alp;
double clwt;
double cql;
double f0;
double robn;
double crbmin;
double crbmax;
double es;
double qs;
double value;
double onemrh;
double cfh;
double gamma;
double elocp;
double el2orc;
double epsi;
double beta;
double chx;
double cqx;
double rdt;
double rdz;
double qmin;
double qlmin;
double rimin;
double rbcr;
double rbint;
double tdzmin;
double rlmn;
double rlmn0;
double rlmn1;
double rlmn2;
double ttend;
double utend;
double vtend;
double qtend;
double zfac;
double zfmin;
double vk;
double spdk2;
double tkmin;
double tkbmx;
double xkgdx;
double xkinv1;
double xkinv2;
double zlup;
double zldn;
double cs0;
double csmf;
double tem;
double tem1;
double tem2;
double tem3;
double ptem;
double ptem0;
double ptem1;
double ptem2;
double slfac;
double vegflo;
double vegfup;
double z0lo;
double z0up;
double vc0;
double zc0;
double ck0;
double ck1;
double ch0;
double ch1;
double ce0;
double rchck;
double qlcr;
double zstblmax;
double hcrinv;
double h1;
double bfac;
double mffac;
bfac = 0x1.9000000000000p+6f; // satmedmfvdifq.F:263
wfac = 0x1.c000000000000p+2f; // satmedmfvdifq.F:264
cfac = 0x1.2000000000000p+2f; // satmedmfvdifq.F:264
gamcrt = 0x1.8000000000000p+1f; // satmedmfvdifq.F:265
gamcrq = 0x0.0p+0f; // satmedmfvdifq.F:265
sfcfrac = 0x1.99999a0000000p-4f; // satmedmfvdifq.F:265
vk = 0x1.99999a0000000p-2f; // satmedmfvdifq.F:266
rimin = (-0x1.9000000000000p+6f); // satmedmfvdifq.F:266
slfac = 0x1.99999a0000000p-4f; // satmedmfvdifq.F:266
rbcr = 0x1.0000000000000p-2f; // satmedmfvdifq.F:267
zolcru = (-0x1.47ae140000000p-6f); // satmedmfvdifq.F:267
tdzmin = 0x1.0624de0000000p-10f; // satmedmfvdifq.F:267
rlmn = 0x1.e000000000000p+4f; // satmedmfvdifq.F:268
rlmn0 = 0x1.4000000000000p+2f; // satmedmfvdifq.F:268
rlmn1 = 0x1.4000000000000p+2f; // satmedmfvdifq.F:268
rlmn2 = 0x1.4000000000000p+3f; // satmedmfvdifq.F:268
prmin = 0x1.0000000000000p-2f; // satmedmfvdifq.F:269
prmax = 0x1.0000000000000p+2f; // satmedmfvdifq.F:269
pr0 = 0x1.0000000000000p+0f; // satmedmfvdifq.F:270
prtke = 0x1.0000000000000p+0f; // satmedmfvdifq.F:270
prscu = 0x1.570a3e0000000p-1f; // satmedmfvdifq.F:270
f0 = 0x1.a36e2e0000000p-14f; // satmedmfvdifq.F:271
crbmin = 0x1.3333340000000p-3f; // satmedmfvdifq.F:271
crbmax = 0x1.6666660000000p-2f; // satmedmfvdifq.F:271
tkmin = 0x1.12e0be0000000p-30f; // satmedmfvdifq.F:272
tkbmx = 0x1.99999a0000000p-3f; // satmedmfvdifq.F:272
dspmax = 0x1.4000000000000p+3f; // satmedmfvdifq.F:272
qmin = 0x1.5798ee0000000p-27f; // satmedmfvdifq.F:273
qlmin = 0x1.1979980000000p-40f; // satmedmfvdifq.F:273
zfmin = 0x1.5798ee0000000p-27f; // satmedmfvdifq.F:273
aphi5 = 0x1.4000000000000p+2f; // satmedmfvdifq.F:274
aphi16 = 0x1.0000000000000p+4f; // satmedmfvdifq.F:274
elmfac = 0x1.0000000000000p+0f; // satmedmfvdifq.F:275
elefac = 0x1.0000000000000p+0f; // satmedmfvdifq.F:275
cql = 0x1.9000000000000p+6f; // satmedmfvdifq.F:275
dw2min = 0x1.a36e2e0000000p-14f; // satmedmfvdifq.F:276
dkmax = 0x1.f400000000000p+9f; // satmedmfvdifq.F:276
xkgdx = 0x1.f400000000000p+9f; // satmedmfvdifq.F:276
qlcr = 0x1.2599ee0000000p-15f; // satmedmfvdifq.F:277
zstblmax = 0x1.3880000000000p+11f; // satmedmfvdifq.F:277
xkinv1 = 0x1.3333340000000p-3f; // satmedmfvdifq.F:278
xkinv2 = 0x1.3333340000000p-2f; // satmedmfvdifq.F:278
h1 = 0x1.5555560000000p-2f; // satmedmfvdifq.F:279
hcrinv = 0x1.f400000000000p+7f; // satmedmfvdifq.F:279
vegflo = 0x1.99999a0000000p-4f; // satmedmfvdifq.F:280
vegfup = 0x1.0000000000000p+0f; // satmedmfvdifq.F:280
z0lo = 0x1.99999a0000000p-4f; // satmedmfvdifq.F:280
z0up = 0x1.0000000000000p+0f; // satmedmfvdifq.F:280
vc0 = 0x1.0000000000000p+0f; // satmedmfvdifq.F:281
zc0 = 0x1.0000000000000p+0f; // satmedmfvdifq.F:281
ck1 = 0x1.3333340000000p-3f; // satmedmfvdifq.F:282
ch1 = 0x1.3333340000000p-3f; // satmedmfvdifq.F:282
cs0 = 0x1.99999a0000000p-2f; // satmedmfvdifq.F:283
csmf = 0x1.0000000000000p-1f; // satmedmfvdifq.F:283
rchck = 0x1.8000000000000p+0f; // satmedmfvdifq.F:284
ndt = 20; // satmedmfvdifq.F:284
// satmedmfvdifq.F:286
if((tc_pbl == 0)) {
// satmedmfvdifq.F:287
ck0 = 0x1.99999a0000000p-2f;
// satmedmfvdifq.F:288
ch0 = 0x1.99999a0000000p-2f;
// satmedmfvdifq.F:289
ce0 = 0x1.99999a0000000p-2f;
// satmedmfvdifq.F:290
} else if((tc_pbl == 1)) {
// satmedmfvdifq.F:291
ck0 = 0x1.19999a0000000p-1f;
// satmedmfvdifq.F:292
ch0 = 0x1.19999a0000000p-1f;
// satmedmfvdifq.F:293
ce0 = 0x1.eb851e0000000p-4f;
// satmedmfvdifq.F:294
}
// satmedmfvdifq.F:295
gravi = sm_div((double)(0x1.0000000000000p+0f),(double)(grav));
// satmedmfvdifq.F:296
g = grav;
// satmedmfvdifq.F:297
gocp = sm_div((double)(g),(double)(cp));
// satmedmfvdifq.F:302
elocp = sm_div((double)(hvap),(double)(cp));
// satmedmfvdifq.F:303
el2orc = sm_div((double)(sm_mul((double)(hvap),(double)(hvap))),(double)(sm_mul((double)(rv),(double)(cp))));
// satmedmfvdifq.F:307
/* CCPP success message is empty. */
// satmedmfvdifq.F:308
errflg = 0;
// satmedmfvdifq.F:311
dt2 = delt;
// satmedmfvdifq.F:312
rdt = sm_div((double)(0x1.0000000000000p+0f),(double)(dt2));
// satmedmfvdifq.F:317
ntrac1 = (ntrac - 1);
// satmedmfvdifq.F:318
km1 = (km - 1);
// satmedmfvdifq.F:319
kmpbl = (km / 2);
// satmedmfvdifq.F:320
kmscu = (km / 2);
// satmedmfvdifq.F:323
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:324
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:325
zi(i,k) = sm_mul((double)(phii(i,k)),(double)(gravi));
// satmedmfvdifq.F:326
zl(i,k) = sm_mul((double)(phil(i,k)),(double)(gravi));
// satmedmfvdifq.F:327
xmf(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:328
xmfd(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:329
buou(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:330
buod(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:331
wush(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:332
ckz(i,k) = ck1;
// satmedmfvdifq.F:333
chz(i,k) = ch1;
// satmedmfvdifq.F:334
rlmnz(i,k) = rlmn0;
// satmedmfvdifq.F:335
}
// satmedmfvdifq.F:336
}
// satmedmfvdifq.F:337
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:338
zi(i,(km + 1)) = sm_mul((double)(phii(i,(km + 1))),(double)(gravi));
// satmedmfvdifq.F:339
}
// satmedmfvdifq.F:340
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:341
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:342
zm(i,k) = zi(i,(k + 1));
// satmedmfvdifq.F:343
}
// satmedmfvdifq.F:344
}
// satmedmfvdifq.F:346
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:347
gdx(i) = sm_sqrt(garea(i));
// satmedmfvdifq.F:348
}
// satmedmfvdifq.F:351
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:352
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:353
tke(i,k) = sm_max<double>(q1(i,k,ntke),tkmin);
// satmedmfvdifq.F:354
}
// satmedmfvdifq.F:355
}
// satmedmfvdifq.F:356
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:357
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:358
tkeh(i,k) = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(tke(i,k)),(double)(tke(i,(k + 1))))));
// satmedmfvdifq.F:359
}
// satmedmfvdifq.F:360
}
// satmedmfvdifq.F:362
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:363
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:364
rdzt(i,k) = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)))));
// satmedmfvdifq.F:365
prn(i,k) = pr0;
// satmedmfvdifq.F:366
}
// satmedmfvdifq.F:367
}
// satmedmfvdifq.F:380
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:381
kx1(i) = 1;
// satmedmfvdifq.F:382
tx1(i) = sm_div((double)(0x1.0000000000000p+0f),(double)(prsi(i,1)));
// satmedmfvdifq.F:383
tx2(i) = tx1(i);
// satmedmfvdifq.F:384
if((gdx(i) >= xkgdx)) {
// satmedmfvdifq.F:385
xkzm_hx(i) = xkzm_h;
// satmedmfvdifq.F:386
xkzm_mx(i) = xkzm_m;
// satmedmfvdifq.F:387
} else {
// satmedmfvdifq.F:388
tem = sm_div((double)(gdx(i)),(double)(xkgdx));
// satmedmfvdifq.F:389
xkzm_hx(i) = sm_mul((double)(xkzm_h),(double)(tem));
// satmedmfvdifq.F:390
xkzm_mx(i) = sm_mul((double)(xkzm_m),(double)(tem));
// satmedmfvdifq.F:391
}
// satmedmfvdifq.F:392
}
// satmedmfvdifq.F:393
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:394
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:395
xkzo(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:396
xkzmo(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:397
if((k < kinver(i))) {
// satmedmfvdifq.F:399
ptem = sm_mul((double)(prsi(i,(k + 1))),(double)(tx1(i)));
// satmedmfvdifq.F:400
tem1 = sm_sub((double)(0x1.0000000000000p+0f),(double)(ptem));
// satmedmfvdifq.F:401
tem2 = sm_mul((double)(sm_mul((double)(tem1),(double)(tem1))),(double)(0x1.4000000000000p+1f));
// satmedmfvdifq.F:402
tem2 = sm_min<double>(0x1.0000000000000p+0f,sm_exp((-tem2)));
// satmedmfvdifq.F:403
rlmnz(i,k) = sm_mul((double)(rlmn),(double)(tem2));
// satmedmfvdifq.F:404
rlmnz(i,k) = sm_max<double>(rlmnz(i,k),rlmn0);
// satmedmfvdifq.F:406
tem2 = sm_mul((double)(sm_mul((double)(tem1),(double)(tem1))),(double)(0x1.4000000000000p+3f));
// satmedmfvdifq.F:407
tem2 = sm_min<double>(0x1.0000000000000p+0f,sm_exp((-tem2)));
// satmedmfvdifq.F:408
xkzo(i,k) = sm_mul((double)(xkzm_hx(i)),(double)(tem2));
// satmedmfvdifq.F:411
if((ptem >= xkzm_s)) {
// satmedmfvdifq.F:412
xkzmo(i,k) = xkzm_mx(i);
// satmedmfvdifq.F:413
kx1(i) = (k + 1);
// satmedmfvdifq.F:414
} else {
// satmedmfvdifq.F:415
if(((k == kx1(i)) && (k > 1))) tx2(i) = sm_div((double)(0x1.0000000000000p+0f),(double)(prsi(i,k)));
// satmedmfvdifq.F:416
tem1 = sm_sub((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(prsi(i,(k + 1))),(double)(tx2(i)))));
// satmedmfvdifq.F:417
tem1 = sm_mul((double)(sm_mul((double)(tem1),(double)(tem1))),(double)(0x1.4000000000000p+2f));
// satmedmfvdifq.F:418
xkzmo(i,k) = sm_mul((double)(xkzm_mx(i)),(double)(sm_min<double>(0x1.0000000000000p+0f,sm_exp((-tem1)))));
// satmedmfvdifq.F:419
}
// satmedmfvdifq.F:420
}
// satmedmfvdifq.F:421
}
// satmedmfvdifq.F:422
}
// satmedmfvdifq.F:425
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:426
z0(i) = sm_mul((double)(0x1.47ae140000000p-7f),(double)(zorl(i)));
// satmedmfvdifq.F:427
rho_a(i) = sm_div((double)(prsl(i,1)),(double)(sm_mul((double)(sm_mul((double)(rd),(double)(t1(i,1)))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(sm_max<double>(q1(i,1,1),qmin)))))))));
// satmedmfvdifq.F:428
dusfc(i) = 0x0.0p+0f;
// satmedmfvdifq.F:429
dvsfc(i) = 0x0.0p+0f;
// satmedmfvdifq.F:430
dtsfc(i) = 0x0.0p+0f;
// satmedmfvdifq.F:431
dqsfc(i) = 0x0.0p+0f;
// satmedmfvdifq.F:432
kpbl(i) = 1;
// satmedmfvdifq.F:433
hpbl(i) = 0x0.0p+0f;
// satmedmfvdifq.F:434
kpblx(i) = 1;
// satmedmfvdifq.F:435
hpblx(i) = 0x0.0p+0f;
// satmedmfvdifq.F:436
pblflg(i) = true;
// satmedmfvdifq.F:437
sfcflg(i) = true;
// satmedmfvdifq.F:438
if((rbsoil(i) > 0x0.0p+0f)) sfcflg(i) = false;
// satmedmfvdifq.F:439
pcnvflg(i) = false;
// satmedmfvdifq.F:440
scuflg(i) = true;
// satmedmfvdifq.F:441
if(scuflg(i)) {
// satmedmfvdifq.F:442
radmin(i) = 0x0.0p+0f;
// satmedmfvdifq.F:443
mrad(i) = km1;
// satmedmfvdifq.F:444
krad(i) = 1;
// satmedmfvdifq.F:445
lcld(i) = km1;
// satmedmfvdifq.F:446
kcld(i) = km1;
// satmedmfvdifq.F:447
}
// satmedmfvdifq.F:448
}
// satmedmfvdifq.F:454
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:455
tem = sm_div((double)(sm_sub((double)(sigmaf(i)),(double)(vegflo))),(double)(sm_sub((double)(vegfup),(double)(vegflo))));
// satmedmfvdifq.F:456
tem = sm_min<double>(sm_max<double>(tem,0x0.0p+0f),0x1.0000000000000p+0f);
// satmedmfvdifq.F:457
tem1 = sm_sqrt(tem);
// satmedmfvdifq.F:458
ptem = sm_div((double)(sm_sub((double)(z0(i)),(double)(z0lo))),(double)(sm_sub((double)(z0up),(double)(z0lo))));
// satmedmfvdifq.F:459
ptem = sm_min<double>(sm_max<double>(ptem,0x0.0p+0f),0x1.0000000000000p+0f);
// satmedmfvdifq.F:460
vez0fun(i) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(vc0),(double)(tem1))))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(zc0),(double)(ptem))))));
// satmedmfvdifq.F:461
}
// satmedmfvdifq.F:465
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:466
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:467
pix(i,k) = sm_div((double)(psk(i)),(double)(prslk(i,k)));
// satmedmfvdifq.F:468
theta(i,k) = sm_mul((double)(t1(i,k)),(double)(pix(i,k)));
// satmedmfvdifq.F:469
if((ntiw > 0)) {
// satmedmfvdifq.F:470
tem = sm_max<double>(q1(i,k,ntcw),qlmin);
// satmedmfvdifq.F:471
tem1 = sm_max<double>(q1(i,k,ntiw),qlmin);
// satmedmfvdifq.F:472
qlx(i,k) = sm_add((double)(tem),(double)(tem1));
// satmedmfvdifq.F:473
ptem = sm_add((double)(sm_mul((double)(hvap),(double)(tem))),(double)(sm_mul((double)(sm_add((double)(hvap),(double)(hfus))),(double)(tem1))));
// satmedmfvdifq.F:474
slx(i,k) = sm_sub((double)(sm_add((double)(sm_mul((double)(cp),(double)(t1(i,k)))),(double)(phil(i,k)))),(double)(ptem));
// satmedmfvdifq.F:475
} else {
// satmedmfvdifq.F:476
qlx(i,k) = sm_max<double>(q1(i,k,ntcw),qlmin);
// satmedmfvdifq.F:477
slx(i,k) = sm_sub((double)(sm_add((double)(sm_mul((double)(cp),(double)(t1(i,k)))),(double)(phil(i,k)))),(double)(sm_mul((double)(hvap),(double)(qlx(i,k)))));
// satmedmfvdifq.F:478
}
// satmedmfvdifq.F:479
tem2 = sm_sub((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(sm_max<double>(q1(i,k,1),qmin)))))),(double)(qlx(i,k)));
// satmedmfvdifq.F:480
thvx(i,k) = sm_mul((double)(theta(i,k)),(double)(tem2));
// satmedmfvdifq.F:481
tvx(i,k) = sm_mul((double)(t1(i,k)),(double)(tem2));
// satmedmfvdifq.F:482
qtx(i,k) = sm_add((double)(sm_max<double>(q1(i,k,1),qmin)),(double)(qlx(i,k)));
// satmedmfvdifq.F:483
thlx(i,k) = sm_sub((double)(theta(i,k)),(double)(sm_mul((double)(sm_mul((double)(pix(i,k)),(double)(elocp))),(double)(qlx(i,k)))));
// satmedmfvdifq.F:484
thlvx(i,k) = sm_mul((double)(thlx(i,k)),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(qtx(i,k)))))));
// satmedmfvdifq.F:485
svx(i,k) = sm_mul((double)(cp),(double)(tvx(i,k)));
// satmedmfvdifq.F:486
ptem1 = sm_mul((double)(sm_mul((double)(elocp),(double)(pix(i,k)))),(double)(sm_max<double>(q1(i,k,1),qmin)));
// satmedmfvdifq.F:487
thetae(i,k) = sm_add((double)(theta(i,k)),(double)(ptem1));
// satmedmfvdifq.F:489
gotvx(i,k) = sm_div((double)(g),(double)(thvx(i,k)));
// satmedmfvdifq.F:490
}
// satmedmfvdifq.F:491
}
// satmedmfvdifq.F:495
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:496
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:497
plyr(i,k) = sm_mul((double)(0x1.47ae140000000p-7f),(double)(prsl(i,k)));
// satmedmfvdifq.F:499
es = sm_mul((double)(0x1.47ae140000000p-7f),(double)(sm_fpvs(t1(i,k),ws.table)));
// satmedmfvdifq.F:500
qs = sm_max<double>(qmin,sm_div((double)(sm_mul((double)(eps),(double)(es))),(double)(sm_add((double)(plyr(i,k)),(double)(sm_mul((double)(epsm1),(double)(es)))))));
// satmedmfvdifq.F:501
rhly(i,k) = sm_max<double>(0x0.0p+0f,sm_min<double>(0x1.0000000000000p+0f,sm_div((double)(sm_max<double>(qmin,q1(i,k,1))),(double)(qs))));
// satmedmfvdifq.F:502
qstl(i,k) = qs;
// satmedmfvdifq.F:503
}
// satmedmfvdifq.F:504
}
// satmedmfvdifq.F:506
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:507
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:508
cfly(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:509
clwt = sm_mul((double)(0x1.0c6f7a0000000p-20f),(double)(sm_mul((double)(plyr(i,k)),(double)(0x1.0624de0000000p-10f))));
// satmedmfvdifq.F:510
if((qlx(i,k) > clwt)) {
// satmedmfvdifq.F:511
onemrh = sm_max<double>(0x1.b7cdfe0000000p-34f,sm_sub((double)(0x1.0000000000000p+0f),(double)(rhly(i,k))));
// satmedmfvdifq.F:512
tem1 = sm_min<double>(sm_max<double>(sm_pow((double)(sm_mul((double)(onemrh),(double)(qstl(i,k)))),(double)(0x1.f5c2900000000p-2f)),0x1.a36e2e0000000p-14f),0x1.0000000000000p+0f);
// satmedmfvdifq.F:513
tem1 = sm_div((double)(cql),(double)(tem1));
// satmedmfvdifq.F:514
value = sm_max<double>(sm_min<double>(sm_mul((double)(tem1),(double)(qlx(i,k))),0x1.9000000000000p+5f),0x0.0p+0f);
// satmedmfvdifq.F:515
tem2 = sm_sqrt(sm_sqrt(rhly(i,k)));
// satmedmfvdifq.F:516
cfly(i,k) = sm_min<double>(sm_max<double>(sm_mul((double)(tem2),(double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sm_exp((-value)))))),0x0.0p+0f),0x1.0000000000000p+0f);
// satmedmfvdifq.F:517
}
// satmedmfvdifq.F:518
}
// satmedmfvdifq.F:519
}
// satmedmfvdifq.F:523
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:524
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:525
tem = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(svx(i,k)),(double)(svx(i,(k + 1))))));
// satmedmfvdifq.F:526
tem1 = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(t1(i,k)),(double)(t1(i,(k + 1))))));
// satmedmfvdifq.F:527
tem2 = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(qstl(i,k)),(double)(qstl(i,(k + 1))))));
// satmedmfvdifq.F:528
cfh = sm_min<double>(cfly(i,(k + 1)),sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(cfly(i,k)),(double)(cfly(i,(k + 1)))))));
// satmedmfvdifq.F:529
alp = sm_div((double)(g),(double)(tem));
// satmedmfvdifq.F:530
gamma = sm_div((double)(sm_mul((double)(el2orc),(double)(tem2))),(double)(sm_powi((double)(tem1),2)));
// satmedmfvdifq.F:531
epsi = sm_div((double)(tem1),(double)(elocp));
// satmedmfvdifq.F:532
beta = sm_div((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(sm_mul((double)(gamma),(double)(epsi))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(fv))))))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(gamma))));
// satmedmfvdifq.F:533
chx = sm_add((double)(sm_mul((double)(sm_mul((double)(cfh),(double)(alp))),(double)(beta))),(double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(cfh))),(double)(alp))));
// satmedmfvdifq.F:534
cqx = sm_mul((double)(sm_mul((double)(sm_mul((double)(cfh),(double)(alp))),(double)(hvap))),(double)(sm_sub((double)(beta),(double)(epsi))));
// satmedmfvdifq.F:535
cqx = sm_add((double)(cqx),(double)(sm_mul((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(cfh))),(double)(fv))),(double)(g))));
// satmedmfvdifq.F:536
ptem1 = sm_mul((double)(sm_sub((double)(slx(i,(k + 1))),(double)(slx(i,k)))),(double)(rdzt(i,k)));
// satmedmfvdifq.F:537
ptem2 = sm_mul((double)(sm_sub((double)(qtx(i,(k + 1))),(double)(qtx(i,k)))),(double)(rdzt(i,k)));
// satmedmfvdifq.F:538
bf(i,k) = sm_add((double)(sm_mul((double)(chx),(double)(ptem1))),(double)(sm_mul((double)(cqx),(double)(ptem2))));
// satmedmfvdifq.F:539
}
// satmedmfvdifq.F:540
}
// satmedmfvdifq.F:546
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:547
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:548
dku(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:549
dkt(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:550
}
// satmedmfvdifq.F:551
}
// satmedmfvdifq.F:552
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:553
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:554
dkq(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:555
cku(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:556
ckt(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:557
tem = sm_sub((double)(zi(i,(k + 1))),(double)(zi(i,k)));
// satmedmfvdifq.F:558
radx(i,k) = sm_mul((double)(tem),(double)(sm_add((double)(sm_mul((double)(swh(i,k)),(double)(xmu(i)))),(double)(hlw(i,k)))));
// satmedmfvdifq.F:559
}
// satmedmfvdifq.F:560
}
// satmedmfvdifq.F:564
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:565
sflux(i) = sm_add((double)(heat(i)),(double)(sm_mul((double)(sm_mul((double)(evap(i)),(double)(fv))),(double)(theta(i,1)))));
// satmedmfvdifq.F:566
if(((!sfcflg(i)) || (sflux(i) <= 0x0.0p+0f))) pblflg(i) = false;
// satmedmfvdifq.F:567
}
// satmedmfvdifq.F:586
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:587
if(pblflg(i)) {
// satmedmfvdifq.F:589
thermal(i) = thlvx(i,1);
// satmedmfvdifq.F:590
crb(i) = rbcr;
// satmedmfvdifq.F:591
} else {
// satmedmfvdifq.F:592
thermal(i) = sm_mul((double)(tsea(i)),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(sm_max<double>(q1(i,1,1),qmin)))))));
// satmedmfvdifq.F:593
tem = sm_sqrt(sm_add((double)(sm_powi((double)(u10m(i)),2)),(double)(sm_powi((double)(v10m(i)),2))));
// satmedmfvdifq.F:594
tem = sm_max<double>(tem,0x1.0000000000000p+0f);
// satmedmfvdifq.F:595
robn = sm_div((double)(tem),(double)(sm_mul((double)(f0),(double)(z0(i)))));
// satmedmfvdifq.F:596
tem1 = sm_mul((double)(0x1.ad7f2a0000000p-24f),(double)(robn));
// satmedmfvdifq.F:597
crb(i) = sm_mul((double)(0x1.47ae140000000p-3f),(double)(sm_pow((double)(tem1),(double)((-0x1.70a3d80000000p-3f)))));
// satmedmfvdifq.F:598
crb(i) = sm_max<double>(sm_min<double>(crb(i),crbmax),crbmin);
// satmedmfvdifq.F:599
}
// satmedmfvdifq.F:600
}
// satmedmfvdifq.F:602
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:603
dtdz1(i) = sm_div((double)(dt2),(double)(sm_sub((double)(zi(i,2)),(double)(zi(i,1)))));
// satmedmfvdifq.F:604
}
// satmedmfvdifq.F:606
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:607
ustar(i) = sm_sqrt(stress(i));
// satmedmfvdifq.F:608
}
// satmedmfvdifq.F:613
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:614
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:615
rdz = rdzt(i,k);
// satmedmfvdifq.F:617
dw2 = sm_add((double)(sm_powi((double)(sm_sub((double)(u1(i,k)),(double)(u1(i,(k + 1))))),2)),(double)(sm_powi((double)(sm_sub((double)(v1(i,k)),(double)(v1(i,(k + 1))))),2)));
// satmedmfvdifq.F:619
shr2(i,k) = sm_mul((double)(sm_mul((double)(sm_max<double>(dw2,dw2min)),(double)(rdz))),(double)(rdz));
// satmedmfvdifq.F:620
}
// satmedmfvdifq.F:621
}
// satmedmfvdifq.F:626
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:627
flg(i) = false;
// satmedmfvdifq.F:628
rbup(i) = rbsoil(i);
// satmedmfvdifq.F:629
}
// satmedmfvdifq.F:635
for(k=1; k<=kmpbl; k+=1) {
// satmedmfvdifq.F:636
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:637
if((!flg(i))) {
// satmedmfvdifq.F:638
rbdn(i) = rbup(i);
// satmedmfvdifq.F:639
if((tc_pbl == 0)) {
// satmedmfvdifq.F:640
spdk2 = sm_max<double>(sm_add((double)(sm_powi((double)(u1(i,k)),2)),(double)(sm_powi((double)(v1(i,k)),2))),0x1.0000000000000p+0f);
// satmedmfvdifq.F:643
rbup(i) = sm_div((double)(sm_mul((double)(sm_sub((double)(thlvx(i,k)),(double)(thermal(i)))),(double)(sm_div((double)(sm_mul((double)(g),(double)(zl(i,k)))),(double)(thlvx(i,1)))))),(double)(spdk2));
// satmedmfvdifq.F:645
} else if((tc_pbl == 1)) {
// satmedmfvdifq.F:646
spdk2 = sm_add((double)(sm_max<double>(sm_add((double)(sm_powi((double)(sm_sub((double)(u1(i,k)),(double)(u1(i,1)))),2)),(double)(sm_powi((double)(sm_sub((double)(v1(i,k)),(double)(v1(i,1)))),2))),0x1.0000000000000p+0f)),(double)(sm_mul((double)(bfac),(double)(sm_powi((double)(ustar(i)),2)))));
// satmedmfvdifq.F:648
rbup(i) = sm_div((double)(sm_mul((double)(sm_sub((double)(thlvx(i,k)),(double)(thermal(i)))),(double)(sm_div((double)(sm_mul((double)(g),(double)(sm_sub((double)(zl(i,k)),(double)(zl(i,1)))))),(double)(thlvx(i,1)))))),(double)(spdk2));
// satmedmfvdifq.F:650
}
// satmedmfvdifq.F:651
kpblx(i) = k;
// satmedmfvdifq.F:652
flg(i) = (rbup(i) > crb(i));
// satmedmfvdifq.F:653
}
// satmedmfvdifq.F:654
}
// satmedmfvdifq.F:655
}
// satmedmfvdifq.F:659
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:660
if((kpblx(i) > 1)) {
// satmedmfvdifq.F:661
k = kpblx(i);
// satmedmfvdifq.F:662
if((rbdn(i) >= crb(i))) {
// satmedmfvdifq.F:663
rbint = 0x0.0p+0f;
// satmedmfvdifq.F:664
} else if((rbup(i) <= crb(i))) {
// satmedmfvdifq.F:665
rbint = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:666
} else {
// satmedmfvdifq.F:667
rbint = sm_div((double)(sm_sub((double)(crb(i)),(double)(rbdn(i)))),(double)(sm_sub((double)(rbup(i)),(double)(rbdn(i)))));
// satmedmfvdifq.F:668
}
// satmedmfvdifq.F:669
hpblx(i) = sm_add((double)(zl(i,(k - 1))),(double)(sm_mul((double)(rbint),(double)(sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))))))));
// satmedmfvdifq.F:670
if((hpblx(i) < zi(i,kpblx(i)))) kpblx(i) = (kpblx(i) - 1);
// satmedmfvdifq.F:671
} else {
// satmedmfvdifq.F:672
hpblx(i) = zl(i,1);
// satmedmfvdifq.F:673
kpblx(i) = 1;
// satmedmfvdifq.F:674
}
// satmedmfvdifq.F:675
hpbl(i) = hpblx(i);
// satmedmfvdifq.F:676
kpbl(i) = kpblx(i);
// satmedmfvdifq.F:677
if((kpbl(i) <= 1)) pblflg(i) = false;
// satmedmfvdifq.F:678
}
// satmedmfvdifq.F:682
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:683
sfcpbl(i) = sm_mul((double)(slfac),(double)(hpbl(i)));
// satmedmfvdifq.F:684
kb1(i) = 1;
// satmedmfvdifq.F:685
flg(i) = false;
// satmedmfvdifq.F:686
if(pblflg(i)) {
// satmedmfvdifq.F:687
flg(i) = true;
// satmedmfvdifq.F:688
}
// satmedmfvdifq.F:689
}
// satmedmfvdifq.F:690
for(k=2; k<=kmpbl; k+=1) {
// satmedmfvdifq.F:691
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:692
if((flg(i) && (zl(i,k) <= sfcpbl(i)))) {
// satmedmfvdifq.F:693
kb1(i) = k;
// satmedmfvdifq.F:694
} else {
// satmedmfvdifq.F:695
flg(i) = false;
// satmedmfvdifq.F:696
}
// satmedmfvdifq.F:697
}
// satmedmfvdifq.F:698
}
// satmedmfvdifq.F:699
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:700
if(pblflg(i)) kb1(i) = sm_min<int>(kb1(i),kpbl(i));
// satmedmfvdifq.F:701
}
// satmedmfvdifq.F:705
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:706
flg(i) = true;
// satmedmfvdifq.F:707
if((pblflg(i) && (kb1(i) > 1))) {
// satmedmfvdifq.F:708
flg(i) = false;
// satmedmfvdifq.F:709
rbup(i) = rbsoil(i);
// satmedmfvdifq.F:711
thermal(i) = thlvx(i,kb1(i));
// satmedmfvdifq.F:712
kpblx(i) = kb1(i);
// satmedmfvdifq.F:713
hpblx(i) = zl(i,kb1(i));
// satmedmfvdifq.F:714
}
// satmedmfvdifq.F:715
}
// satmedmfvdifq.F:716
for(k=2; k<=kmpbl; k+=1) {
// satmedmfvdifq.F:717
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:718
if(((!flg(i)) && (k > kb1(i)))) {
// satmedmfvdifq.F:719
rbdn(i) = rbup(i);
// satmedmfvdifq.F:720
if((tc_pbl == 0)) {
// satmedmfvdifq.F:721
spdk2 = sm_max<double>(sm_add((double)(sm_powi((double)(u1(i,k)),2)),(double)(sm_powi((double)(v1(i,k)),2))),0x1.0000000000000p+0f);
// satmedmfvdifq.F:724
rbup(i) = sm_div((double)(sm_mul((double)(sm_sub((double)(thlvx(i,k)),(double)(thermal(i)))),(double)(sm_div((double)(sm_mul((double)(g),(double)(zl(i,k)))),(double)(thlvx(i,1)))))),(double)(spdk2));
// satmedmfvdifq.F:726
} else if((tc_pbl == 1)) {
// satmedmfvdifq.F:727
spdk2 = sm_add((double)(sm_max<double>(sm_add((double)(sm_powi((double)(sm_sub((double)(u1(i,k)),(double)(u1(i,1)))),2)),(double)(sm_powi((double)(sm_sub((double)(v1(i,k)),(double)(v1(i,1)))),2))),0x1.0000000000000p+0f)),(double)(sm_mul((double)(bfac),(double)(sm_powi((double)(ustar(i)),2)))));
// satmedmfvdifq.F:729
rbup(i) = sm_div((double)(sm_mul((double)(sm_sub((double)(thlvx(i,k)),(double)(thermal(i)))),(double)(sm_div((double)(sm_mul((double)(g),(double)(sm_sub((double)(zl(i,k)),(double)(zl(i,1)))))),(double)(thlvx(i,1)))))),(double)(spdk2));
// satmedmfvdifq.F:731
}
// satmedmfvdifq.F:732
kpblx(i) = k;
// satmedmfvdifq.F:733
flg(i) = (rbup(i) > crb(i));
// satmedmfvdifq.F:734
}
// satmedmfvdifq.F:735
}
// satmedmfvdifq.F:736
}
// satmedmfvdifq.F:737
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:738
if((pblflg(i) && (kb1(i) > 1))) {
// satmedmfvdifq.F:739
k = kpblx(i);
// satmedmfvdifq.F:740
if((rbdn(i) >= crb(i))) {
// satmedmfvdifq.F:741
rbint = 0x0.0p+0f;
// satmedmfvdifq.F:742
} else if((rbup(i) <= crb(i))) {
// satmedmfvdifq.F:743
rbint = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:744
} else {
// satmedmfvdifq.F:745
rbint = sm_div((double)(sm_sub((double)(crb(i)),(double)(rbdn(i)))),(double)(sm_sub((double)(rbup(i)),(double)(rbdn(i)))));
// satmedmfvdifq.F:746
}
// satmedmfvdifq.F:747
hpblx(i) = sm_add((double)(zl(i,(k - 1))),(double)(sm_mul((double)(rbint),(double)(sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))))))));
// satmedmfvdifq.F:748
if((hpblx(i) < zi(i,kpblx(i)))) kpblx(i) = (kpblx(i) - 1);
// satmedmfvdifq.F:749
hpbl(i) = hpblx(i);
// satmedmfvdifq.F:750
kpbl(i) = kpblx(i);
// satmedmfvdifq.F:751
}
// satmedmfvdifq.F:752
}
// satmedmfvdifq.F:756
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:757
sumx(i) = 0x0.0p+0f;
// satmedmfvdifq.F:758
tkemean(i) = 0x0.0p+0f;
// satmedmfvdifq.F:759
}
// satmedmfvdifq.F:760
for(k=1; k<=kmpbl; k+=1) {
// satmedmfvdifq.F:761
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:762
if((k < kpbl(i))) {
// satmedmfvdifq.F:763
dz = sm_sub((double)(zi(i,(k + 1))),(double)(zi(i,k)));
// satmedmfvdifq.F:764
tkemean(i) = sm_add((double)(tkemean(i)),(double)(sm_mul((double)(tke(i,k)),(double)(dz))));
// satmedmfvdifq.F:765
sumx(i) = sm_add((double)(sumx(i)),(double)(dz));
// satmedmfvdifq.F:766
}
// satmedmfvdifq.F:767
}
// satmedmfvdifq.F:768
}
// satmedmfvdifq.F:769
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:770
if(((tkemean(i) > 0x0.0p+0f) && (sumx(i) > 0x0.0p+0f))) {
// satmedmfvdifq.F:771
tkemean(i) = sm_div((double)(tkemean(i)),(double)(sumx(i)));
// satmedmfvdifq.F:772
}
// satmedmfvdifq.F:773
}
// satmedmfvdifq.F:778
kps = sm_max<int>(kmpbl,kmscu);
// satmedmfvdifq.F:779
for(k=2; k<=kps; k+=1) {
// satmedmfvdifq.F:780
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:781
dz = sm_sub((double)(zi(i,(k + 1))),(double)(zi(i,k)));
// satmedmfvdifq.F:782
tem = sm_powi((double)(sm_div((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_sub((double)(u1(i,(k - 1))),(double)(u1(i,(k + 1))))))),(double)(dz))),2);
// satmedmfvdifq.F:783
tem1 = sm_add((double)(tem),(double)(sm_powi((double)(sm_div((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_sub((double)(v1(i,(k - 1))),(double)(v1(i,(k + 1))))))),(double)(dz))),2)));
// satmedmfvdifq.F:784
wush(i,k) = sm_mul((double)(csmf),(double)(sm_sqrt(tem1)));
// satmedmfvdifq.F:785
}
// satmedmfvdifq.F:786
}
// satmedmfvdifq.F:797
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:798
zol(i) = sm_max<double>(sm_div((double)(sm_mul((double)(sm_mul((double)(rbsoil(i)),(double)(fm(i)))),(double)(fm(i)))),(double)(fh(i))),rimin);
// satmedmfvdifq.F:799
if(sfcflg(i)) {
// satmedmfvdifq.F:800
zol(i) = sm_min<double>(zol(i),(-zfmin));
// satmedmfvdifq.F:801
} else {
// satmedmfvdifq.F:802
zol(i) = sm_max<double>(zol(i),zfmin);
// satmedmfvdifq.F:803
}
// satmedmfvdifq.F:815
zol1 = sm_div((double)(sm_mul((double)(sm_mul((double)(zol(i)),(double)(sfcfrac))),(double)(hpbl(i)))),(double)(zl(i,1)));
// satmedmfvdifq.F:816
if(sfcflg(i)) {
// satmedmfvdifq.F:817
tem = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(aphi16),(double)(zol1))))));
// satmedmfvdifq.F:818
phih(i) = sm_sqrt(tem);
// satmedmfvdifq.F:819
phim(i) = sm_sqrt(phih(i));
// satmedmfvdifq.F:820
tem1 = sm_div((double)(0x1.0000000000000p+0f),(double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(aphi16),(double)(zol(i)))))));
// satmedmfvdifq.F:821
phims(i) = sm_sqrt(sm_sqrt(tem1));
// satmedmfvdifq.F:822
} else {
// satmedmfvdifq.F:823
phim(i) = sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(aphi5),(double)(zol1))));
// satmedmfvdifq.F:824
phih(i) = phim(i);
// satmedmfvdifq.F:825
phims(i) = sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(aphi5),(double)(zol(i)))));
// satmedmfvdifq.F:826
}
// satmedmfvdifq.F:827
}
// satmedmfvdifq.F:844
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:845
if(pblflg(i)) {
// satmedmfvdifq.F:846
if((zol(i) < zolcru)) {
// satmedmfvdifq.F:847
pcnvflg(i) = true;
// satmedmfvdifq.F:848
}
// satmedmfvdifq.F:849
wst3(i) = sm_mul((double)(sm_mul((double)(gotvx(i,1)),(double)(sflux(i)))),(double)(hpbl(i)));
// satmedmfvdifq.F:850
wstar(i) = sm_pow((double)(wst3(i)),(double)(h1));
// satmedmfvdifq.F:851
ust3(i) = sm_pow((double)(ustar(i)),(double)(0x1.8000000000000p+1f));
// satmedmfvdifq.F:852
wscale(i) = sm_pow((double)(sm_add((double)(ust3(i)),(double)(sm_mul((double)(sm_mul((double)(sm_mul((double)(wfac),(double)(vk))),(double)(wst3(i)))),(double)(sfcfrac))))),(double)(h1));
// satmedmfvdifq.F:853
ptem = sm_div((double)(ustar(i)),(double)(aphi5));
// satmedmfvdifq.F:854
wscale(i) = sm_max<double>(wscale(i),ptem);
// satmedmfvdifq.F:855
}
// satmedmfvdifq.F:856
}
// satmedmfvdifq.F:861
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:862
if(pcnvflg(i)) {
// satmedmfvdifq.F:863
hgamt(i) = sm_div((double)(heat(i)),(double)(wscale(i)));
// satmedmfvdifq.F:864
hgamq(i) = sm_div((double)(evap(i)),(double)(wscale(i)));
// satmedmfvdifq.F:865
vpert(i) = sm_add((double)(hgamt(i)),(double)(sm_mul((double)(sm_mul((double)(hgamq(i)),(double)(fv))),(double)(theta(i,1)))));
// satmedmfvdifq.F:866
vpert(i) = sm_max<double>(vpert(i),0x0.0p+0f);
// satmedmfvdifq.F:867
tem = sm_min<double>(sm_mul((double)(cfac),(double)(vpert(i))),gamcrt);
// satmedmfvdifq.F:868
thermal(i) = sm_add((double)(thermal(i)),(double)(tem));
// satmedmfvdifq.F:869
}
// satmedmfvdifq.F:870
}
// satmedmfvdifq.F:875
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:876
flg(i) = true;
// satmedmfvdifq.F:877
if(pcnvflg(i)) {
// satmedmfvdifq.F:878
flg(i) = false;
// satmedmfvdifq.F:879
rbup(i) = rbsoil(i);
// satmedmfvdifq.F:880
}
// satmedmfvdifq.F:881
}
// satmedmfvdifq.F:882
for(k=2; k<=kmpbl; k+=1) {
// satmedmfvdifq.F:883
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:884
if(((!flg(i)) && (k > kb1(i)))) {
// satmedmfvdifq.F:885
rbdn(i) = rbup(i);
// satmedmfvdifq.F:886
if((tc_pbl == 0)) {
// satmedmfvdifq.F:887
spdk2 = sm_max<double>(sm_add((double)(sm_powi((double)(u1(i,k)),2)),(double)(sm_powi((double)(v1(i,k)),2))),0x1.0000000000000p+0f);
// satmedmfvdifq.F:888
rbup(i) = sm_div((double)(sm_mul((double)(sm_sub((double)(thlvx(i,k)),(double)(thermal(i)))),(double)(sm_div((double)(sm_mul((double)(g),(double)(zl(i,k)))),(double)(thlvx(i,1)))))),(double)(spdk2));
// satmedmfvdifq.F:890
} else if((tc_pbl == 1)) {
// satmedmfvdifq.F:891
spdk2 = sm_add((double)(sm_max<double>(sm_add((double)(sm_powi((double)(sm_sub((double)(u1(i,k)),(double)(u1(i,1)))),2)),(double)(sm_powi((double)(sm_sub((double)(v1(i,k)),(double)(v1(i,1)))),2))),0x1.0000000000000p+0f)),(double)(sm_mul((double)(bfac),(double)(sm_powi((double)(ustar(i)),2)))));
// satmedmfvdifq.F:893
rbup(i) = sm_div((double)(sm_mul((double)(sm_sub((double)(thlvx(i,k)),(double)(thermal(i)))),(double)(sm_div((double)(sm_mul((double)(g),(double)(sm_sub((double)(zl(i,k)),(double)(zl(i,1)))))),(double)(thlvx(i,1)))))),(double)(spdk2));
// satmedmfvdifq.F:895
}
// satmedmfvdifq.F:896
kpbl(i) = k;
// satmedmfvdifq.F:897
flg(i) = (rbup(i) > crb(i));
// satmedmfvdifq.F:898
}
// satmedmfvdifq.F:899
}
// satmedmfvdifq.F:900
}
// satmedmfvdifq.F:901
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:902
if(pcnvflg(i)) {
// satmedmfvdifq.F:903
k = kpbl(i);
// satmedmfvdifq.F:904
if((rbdn(i) >= crb(i))) {
// satmedmfvdifq.F:905
rbint = 0x0.0p+0f;
// satmedmfvdifq.F:906
} else if((rbup(i) <= crb(i))) {
// satmedmfvdifq.F:907
rbint = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:908
} else {
// satmedmfvdifq.F:909
rbint = sm_div((double)(sm_sub((double)(crb(i)),(double)(rbdn(i)))),(double)(sm_sub((double)(rbup(i)),(double)(rbdn(i)))));
// satmedmfvdifq.F:910
}
// satmedmfvdifq.F:911
hpbl(i) = sm_add((double)(zl(i,(k - 1))),(double)(sm_mul((double)(rbint),(double)(sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))))))));
// satmedmfvdifq.F:912
if((hpbl(i) < zi(i,kpbl(i)))) {
// satmedmfvdifq.F:913
kpbl(i) = (kpbl(i) - 1);
// satmedmfvdifq.F:914
}
// satmedmfvdifq.F:915
if((kpbl(i) <= 1)) {
// satmedmfvdifq.F:916
pcnvflg(i) = false;
// satmedmfvdifq.F:917
pblflg(i) = false;
// satmedmfvdifq.F:918
}
// satmedmfvdifq.F:919
}
// satmedmfvdifq.F:920
}
// satmedmfvdifq.F:928
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:929
flg(i) = scuflg(i);
// satmedmfvdifq.F:930
}
// satmedmfvdifq.F:931
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:932
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:933
if((flg(i) && (zl(i,k) >= zstblmax))) {
// satmedmfvdifq.F:934
lcld(i) = k;
// satmedmfvdifq.F:935
flg(i) = false;
// satmedmfvdifq.F:936
}
// satmedmfvdifq.F:937
}
// satmedmfvdifq.F:938
}
// satmedmfvdifq.F:939
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:940
flg(i) = scuflg(i);
// satmedmfvdifq.F:941
}
// satmedmfvdifq.F:942
for(k=kmscu; k>=1; k+=(-1)) {
// satmedmfvdifq.F:943
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:944
if((flg(i) && (k <= lcld(i)))) {
// satmedmfvdifq.F:945
if((qlx(i,k) >= qlcr)) {
// satmedmfvdifq.F:946
kcld(i) = k;
// satmedmfvdifq.F:947
flg(i) = false;
// satmedmfvdifq.F:948
}
// satmedmfvdifq.F:949
}
// satmedmfvdifq.F:950
}
// satmedmfvdifq.F:951
}
// satmedmfvdifq.F:952
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:953
if((scuflg(i) && (kcld(i) == km1))) scuflg(i) = false;
// satmedmfvdifq.F:954
}
// satmedmfvdifq.F:960
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:961
flg(i) = scuflg(i);
// satmedmfvdifq.F:962
}
// satmedmfvdifq.F:963
for(k=kmscu; k>=1; k+=(-1)) {
// satmedmfvdifq.F:964
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:965
if((flg(i) && (k <= kcld(i)))) {
// satmedmfvdifq.F:966
if((qlx(i,k) >= qlcr)) {
// satmedmfvdifq.F:967
if((radx(i,k) < radmin(i))) {
// satmedmfvdifq.F:968
radmin(i) = radx(i,k);
// satmedmfvdifq.F:969
krad(i) = k;
// satmedmfvdifq.F:970
}
// satmedmfvdifq.F:971
} else {
// satmedmfvdifq.F:972
flg(i) = false;
// satmedmfvdifq.F:973
}
// satmedmfvdifq.F:974
}
// satmedmfvdifq.F:975
}
// satmedmfvdifq.F:976
}
// satmedmfvdifq.F:977
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:978
if((scuflg(i) && (krad(i) <= 1))) scuflg(i) = false;
// satmedmfvdifq.F:979
if((scuflg(i) && (radmin(i) >= 0x0.0p+0f))) scuflg(i) = false;
// satmedmfvdifq.F:980
}
// satmedmfvdifq.F:987
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:988
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:989
if(pcnvflg(i)) {
// satmedmfvdifq.F:990
tcko(i,k) = t1(i,k);
// satmedmfvdifq.F:991
ucko(i,k) = u1(i,k);
// satmedmfvdifq.F:992
vcko(i,k) = v1(i,k);
// satmedmfvdifq.F:993
}
// satmedmfvdifq.F:994
if(scuflg(i)) {
// satmedmfvdifq.F:995
tcdo(i,k) = t1(i,k);
// satmedmfvdifq.F:996
ucdo(i,k) = u1(i,k);
// satmedmfvdifq.F:997
vcdo(i,k) = v1(i,k);
// satmedmfvdifq.F:998
}
// satmedmfvdifq.F:999
}
// satmedmfvdifq.F:1000
}
// satmedmfvdifq.F:1001
for(n=1; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:1002
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1003
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1004
if(pcnvflg(i)) {
// satmedmfvdifq.F:1005
qcko(i,k,n) = q1(i,k,n);
// satmedmfvdifq.F:1006
}
// satmedmfvdifq.F:1007
if(scuflg(i)) {
// satmedmfvdifq.F:1008
qcdo(i,k,n) = q1(i,k,n);
// satmedmfvdifq.F:1009
}
// satmedmfvdifq.F:1010
}
// satmedmfvdifq.F:1011
}
// satmedmfvdifq.F:1012
}
// satmedmfvdifq.F:1015
sm_mfpbltq(im, im, km, kmpbl, ntcw, ntrac1, dt2, pcnvflg.p, zl.p, zm.p, q1.p, t1.p, u1.p, v1.p, plyr.p, pix.p, thlx.p, thvx.p, gdx.p, hpbl.p, kpbl.p, vpert.p, buou.p, wush.p, tkemean.p, vez0fun.p, xmf.p, tcko.p, qcko.p, ucko.p, vcko.p, xlamue.p, bl_upfr, ws);
// satmedmfvdifq.F:1021
sm_mfscuq(im, im, km, kmscu, ntcw, ntrac1, dt2, scuflg.p, zl.p, zm.p, q1.p, t1.p, u1.p, v1.p, plyr.p, pix.p, thlx.p, thvx.p, thlvx.p, gdx.p, thetae.p, krad.p, mrad.p, radmin.p, buod.p, wush.p, tkemean.p, vez0fun.p, xmfd.p, tcdo.p, qcdo.p, ucdo.p, vcdo.p, xlamde.p, bl_dnfr, ws);
// satmedmfvdifq.F:1027
if((tc_pbl == 1)) {
// satmedmfvdifq.F:1029
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1030
if((zol(i) > (-0x1.0000000000000p-1f))) {
// satmedmfvdifq.F:1031
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1032
xmf(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:1033
}
// satmedmfvdifq.F:1034
}
// satmedmfvdifq.F:1035
}
// satmedmfvdifq.F:1037
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1038
tem = sm_sqrt(sm_add((double)(sm_powi((double)(u10m(i)),2)),(double)(sm_powi((double)(v10m(i)),2))));
// satmedmfvdifq.F:1039
mffac = sm_sub((double)(0x1.0000000000000p+0f),(double)(sm_div((double)(sm_min<double>(sm_max<double>(sm_sub((double)(tem),(double)(0x1.4000000000000p+4f)),0x0.0p+0f),0x1.4000000000000p+3f)),(double)(0x1.4000000000000p+3f))));
// satmedmfvdifq.F:1040
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1041
xmf(i,k) = sm_mul((double)(xmf(i,k)),(double)(mffac));
// satmedmfvdifq.F:1042
xmfd(i,k) = sm_mul((double)(xmfd(i,k)),(double)(mffac));
// satmedmfvdifq.F:1043
}
// satmedmfvdifq.F:1044
}
// satmedmfvdifq.F:1045
}
// satmedmfvdifq.F:1050
for(k=1; k<=kmpbl; k+=1) {
// satmedmfvdifq.F:1051
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1052
if((k < kpbl(i))) {
// satmedmfvdifq.F:1053
tem = sm_div((double)(phih(i)),(double)(phim(i)));
// satmedmfvdifq.F:1054
ptem = sm_mul((double)(sfcfrac),(double)(hpbl(i)));
// satmedmfvdifq.F:1055
tem1 = sm_max<double>(sm_sub((double)(zi(i,(k + 1))),(double)(ptem)),0x0.0p+0f);
// satmedmfvdifq.F:1056
tem2 = sm_div((double)(tem1),(double)(sm_sub((double)(hpbl(i)),(double)(ptem))));
// satmedmfvdifq.F:1057
if(pcnvflg(i)) {
// satmedmfvdifq.F:1058
tem = sm_min<double>(tem,pr0);
// satmedmfvdifq.F:1059
prn(i,k) = sm_add((double)(tem),(double)(sm_mul((double)(sm_sub((double)(pr0),(double)(tem))),(double)(tem2))));
// satmedmfvdifq.F:1060
} else {
// satmedmfvdifq.F:1061
tem = sm_max<double>(tem,pr0);
// satmedmfvdifq.F:1062
prn(i,k) = tem;
// satmedmfvdifq.F:1063
}
// satmedmfvdifq.F:1064
prn(i,k) = sm_min<double>(prn(i,k),prmax);
// satmedmfvdifq.F:1065
prn(i,k) = sm_max<double>(prn(i,k),prmin);
// satmedmfvdifq.F:1067
ckz(i,k) = sm_add((double)(ck0),(double)(sm_mul((double)(sm_sub((double)(ck1),(double)(ck0))),(double)(tem2))));
// satmedmfvdifq.F:1068
ckz(i,k) = sm_max<double>(sm_min<double>(ckz(i,k),ck0),ck1);
// satmedmfvdifq.F:1069
chz(i,k) = sm_add((double)(ch0),(double)(sm_mul((double)(sm_sub((double)(ch1),(double)(ch0))),(double)(tem2))));
// satmedmfvdifq.F:1070
chz(i,k) = sm_max<double>(sm_min<double>(chz(i,k),ch0),ch1);
// satmedmfvdifq.F:1072
}
// satmedmfvdifq.F:1073
}
// satmedmfvdifq.F:1074
}
// satmedmfvdifq.F:1086
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1087
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1088
if((zi(i,(k + 1)) > hcrinv)) {
// satmedmfvdifq.F:1089
tem1 = sm_sub((double)(tvx(i,(k + 1))),(double)(tvx(i,k)));
// satmedmfvdifq.F:1090
if((tem1 >= 0x0.0p+0f)) {
// satmedmfvdifq.F:1091
xkzo(i,k) = sm_min<double>(xkzo(i,k),xkinv1);
// satmedmfvdifq.F:1092
xkzmo(i,k) = sm_min<double>(xkzmo(i,k),xkinv1);
// satmedmfvdifq.F:1093
rlmnz(i,k) = sm_min<double>(rlmnz(i,k),rlmn1);
// satmedmfvdifq.F:1094
}
// satmedmfvdifq.F:1095
} else {
// satmedmfvdifq.F:1096
tem1 = sm_sub((double)(tvx(i,(k + 1))),(double)(tvx(i,k)));
// satmedmfvdifq.F:1097
if((tem1 > 0x0.0p+0f)) {
// satmedmfvdifq.F:1098
ptem = sm_mul((double)(xkzo(i,k)),(double)(zvfun(i)));
// satmedmfvdifq.F:1099
xkzo(i,k) = sm_min<double>(sm_max<double>(ptem,xkinv2),xkzo(i,k));
// satmedmfvdifq.F:1100
ptem = sm_mul((double)(xkzmo(i,k)),(double)(zvfun(i)));
// satmedmfvdifq.F:1101
xkzmo(i,k) = sm_min<double>(sm_max<double>(ptem,xkinv2),xkzmo(i,k));
// satmedmfvdifq.F:1102
ptem = sm_mul((double)(rlmnz(i,k)),(double)(zvfun(i)));
// satmedmfvdifq.F:1103
rlmnz(i,k) = sm_min<double>(sm_max<double>(ptem,rlmn2),rlmnz(i,k));
// satmedmfvdifq.F:1104
}
// satmedmfvdifq.F:1105
}
// satmedmfvdifq.F:1106
}
// satmedmfvdifq.F:1107
}
// satmedmfvdifq.F:1108
for(k=2; k<=km1; k+=1) {
// satmedmfvdifq.F:1109
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1110
rlmnz(i,k) = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(rlmnz(i,(k - 1))),(double)(rlmnz(i,k)))));
// satmedmfvdifq.F:1111
}
// satmedmfvdifq.F:1112
}
// satmedmfvdifq.F:1117
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1118
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1119
zlup = 0x0.0p+0f;
// satmedmfvdifq.F:1120
mlenflg = true;
// satmedmfvdifq.F:1121
e2(i,k) = sm_max<double>(sm_mul((double)(0x1.0000000000000p+1f),(double)(tke(i,k))),0x1.0624de0000000p-10f);
// satmedmfvdifq.F:1122
for(n=k; n<=km1; n+=1) {
// satmedmfvdifq.F:1123
if(mlenflg) {
// satmedmfvdifq.F:1124
dz = sm_sub((double)(zl(i,(n + 1))),(double)(zl(i,n)));
// satmedmfvdifq.F:1125
tem1 = sm_mul((double)(sm_mul((double)(0x1.0000000000000p+1f),(double)(gotvx(i,(n + 1))))),(double)(sm_sub((double)(thvx(i,k)),(double)(thvx(i,(n + 1))))));
// satmedmfvdifq.F:1126
tem2 = sm_mul((double)(sm_mul((double)(cs0),(double)(sm_sqrt(e2(i,n))))),(double)(sm_sqrt(shr2(i,n))));
// satmedmfvdifq.F:1127
e2(i,(n + 1)) = sm_add((double)(e2(i,n)),(double)(sm_mul((double)(sm_sub((double)(tem1),(double)(tem2))),(double)(dz))));
// satmedmfvdifq.F:1128
zlup = sm_add((double)(zlup),(double)(dz));
// satmedmfvdifq.F:1129
if((e2(i,(n + 1)) < 0x0.0p+0f)) {
// satmedmfvdifq.F:1130
ptem = sm_div((double)(e2(i,(n + 1))),(double)(sm_sub((double)(e2(i,(n + 1))),(double)(e2(i,n)))));
// satmedmfvdifq.F:1131
zlup = sm_sub((double)(zlup),(double)(sm_mul((double)(ptem),(double)(dz))));
// satmedmfvdifq.F:1132
zlup = sm_max<double>(zlup,0x0.0p+0f);
// satmedmfvdifq.F:1133
mlenflg = false;
// satmedmfvdifq.F:1134
}
// satmedmfvdifq.F:1135
}
// satmedmfvdifq.F:1136
}
// satmedmfvdifq.F:1137
zldn = 0x0.0p+0f;
// satmedmfvdifq.F:1138
mlenflg = true;
// satmedmfvdifq.F:1139
for(n=k; n>=1; n+=(-1)) {
// satmedmfvdifq.F:1140
if(mlenflg) {
// satmedmfvdifq.F:1141
if((n == 1)) {
// satmedmfvdifq.F:1142
dz = zl(i,1);
// satmedmfvdifq.F:1143
tem = sm_mul((double)(tsea(i)),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(fv),(double)(sm_max<double>(q1(i,1,1),qmin)))))));
// satmedmfvdifq.F:1144
tem1 = sm_mul((double)(sm_mul((double)(0x1.0000000000000p+1f),(double)(gotvx(i,n)))),(double)(sm_sub((double)(tem),(double)(thvx(i,k)))));
// satmedmfvdifq.F:1145
tem2 = sm_div((double)(sm_mul((double)(ustar(i)),(double)(phims(i)))),(double)(sm_mul((double)(vk),(double)(dz))));
// satmedmfvdifq.F:1146
tem2 = sm_mul((double)(sm_mul((double)(cs0),(double)(sm_sqrt(e2(i,n))))),(double)(tem2));
// satmedmfvdifq.F:1147
e2(i,(n - 1)) = sm_add((double)(e2(i,n)),(double)(sm_mul((double)(sm_sub((double)(tem1),(double)(tem2))),(double)(dz))));
// satmedmfvdifq.F:1148
} else {
// satmedmfvdifq.F:1149
dz = sm_sub((double)(zl(i,n)),(double)(zl(i,(n - 1))));
// satmedmfvdifq.F:1150
tem1 = sm_mul((double)(sm_mul((double)(0x1.0000000000000p+1f),(double)(gotvx(i,(n - 1))))),(double)(sm_sub((double)(thvx(i,(n - 1))),(double)(thvx(i,k)))));
// satmedmfvdifq.F:1151
tem2 = sm_mul((double)(sm_mul((double)(cs0),(double)(sm_sqrt(e2(i,n))))),(double)(sm_sqrt(shr2(i,(n - 1)))));
// satmedmfvdifq.F:1152
e2(i,(n - 1)) = sm_add((double)(e2(i,n)),(double)(sm_mul((double)(sm_sub((double)(tem1),(double)(tem2))),(double)(dz))));
// satmedmfvdifq.F:1153
}
// satmedmfvdifq.F:1154
zldn = sm_add((double)(zldn),(double)(dz));
// satmedmfvdifq.F:1155
if((e2(i,(n - 1)) < 0x0.0p+0f)) {
// satmedmfvdifq.F:1156
ptem = sm_div((double)(e2(i,(n - 1))),(double)(sm_sub((double)(e2(i,(n - 1))),(double)(e2(i,n)))));
// satmedmfvdifq.F:1157
zldn = sm_sub((double)(zldn),(double)(sm_mul((double)(ptem),(double)(dz))));
// satmedmfvdifq.F:1158
zldn = sm_max<double>(zldn,0x0.0p+0f);
// satmedmfvdifq.F:1159
mlenflg = false;
// satmedmfvdifq.F:1160
}
// satmedmfvdifq.F:1161
}
// satmedmfvdifq.F:1162
}
// satmedmfvdifq.F:1164
tem = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_sub((double)(zi(i,(k + 1))),(double)(zi(i,k)))));
// satmedmfvdifq.F:1165
tem1 = sm_min<double>(tem,rlmnz(i,k));
// satmedmfvdifq.F:1182
ptem2 = sm_min<double>(zlup,zldn);
// satmedmfvdifq.F:1183
rlam(i,k) = sm_mul((double)(elmfac),(double)(ptem2));
// satmedmfvdifq.F:1184
rlam(i,k) = sm_max<double>(rlam(i,k),tem1);
// satmedmfvdifq.F:1185
rlam(i,k) = sm_min<double>(rlam(i,k),rlmx);
// satmedmfvdifq.F:1187
ptem2 = sm_sqrt(sm_mul((double)(zlup),(double)(zldn)));
// satmedmfvdifq.F:1188
ele(i,k) = sm_mul((double)(elefac),(double)(ptem2));
// satmedmfvdifq.F:1189
ele(i,k) = sm_max<double>(ele(i,k),tem1);
// satmedmfvdifq.F:1190
ele(i,k) = sm_min<double>(ele(i,k),elmx);
// satmedmfvdifq.F:1192
}
// satmedmfvdifq.F:1193
}
// satmedmfvdifq.F:1196
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1197
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1198
tem = sm_mul((double)(vk),(double)(zl(i,k)));
// satmedmfvdifq.F:1199
if((zol(i) < 0x0.0p+0f)) {
// satmedmfvdifq.F:1200
ptem = sm_sub((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(0x1.9000000000000p+6f),(double)(zol(i)))));
// satmedmfvdifq.F:1201
ptem1 = sm_pow((double)(ptem),(double)(0x1.99999a0000000p-3f));
// satmedmfvdifq.F:1202
zk = sm_mul((double)(tem),(double)(ptem1));
// satmedmfvdifq.F:1203
} else if((zol(i) >= 0x1.0000000000000p+0f)) {
// satmedmfvdifq.F:1204
zk = sm_div((double)(tem),(double)(0x1.d9999a0000000p+1f));
// satmedmfvdifq.F:1205
} else {
// satmedmfvdifq.F:1206
ptem = sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(0x1.59999a0000000p+1f),(double)(zol(i)))));
// satmedmfvdifq.F:1207
zk = sm_div((double)(tem),(double)(ptem));
// satmedmfvdifq.F:1208
}
// satmedmfvdifq.F:1210
if((tc_pbl == 0)) {
// satmedmfvdifq.F:1211
elm(i,k) = sm_div((double)(sm_mul((double)(zk),(double)(rlam(i,k)))),(double)(sm_add((double)(rlam(i,k)),(double)(zk))));
// satmedmfvdifq.F:1213
if((sfc_rlm == 1)) {
// satmedmfvdifq.F:1214
if((sfcflg(i) && (zl(i,k) < sm_min<double>(0x1.9000000000000p+6f,sm_mul((double)(hpbl(i)),(double)(0x1.99999a0000000p-5f)))))) elm(i,k) = zk;
// satmedmfvdifq.F:1216
}
// satmedmfvdifq.F:1217
} else if((tc_pbl == 1)) {
// satmedmfvdifq.F:1219
elm(i,k) = sm_sqrt(sm_div((double)(0x1.0000000000000p+0f),(double)(sm_add((double)(sm_div((double)(0x1.0000000000000p+0f),(double)(sm_powi((double)(zk),2)))),(double)(sm_div((double)(0x1.0000000000000p+0f),(double)(sm_powi((double)(rlam(i,k)),2))))))));
// satmedmfvdifq.F:1220
}
// satmedmfvdifq.F:1223
if((k == 1)) elm(i,k) = zk;
// satmedmfvdifq.F:1225
dz = sm_sub((double)(zi(i,(k + 1))),(double)(zi(i,k)));
// satmedmfvdifq.F:1226
tem = sm_max<double>(gdx(i),dz);
// satmedmfvdifq.F:1227
elm(i,k) = sm_min<double>(elm(i,k),tem);
// satmedmfvdifq.F:1229
if((tc_pbl == 0)) {
// satmedmfvdifq.F:1230
ele(i,k) = sm_min<double>(ele(i,k),tem);
// satmedmfvdifq.F:1231
} else if((tc_pbl == 1)) {
// satmedmfvdifq.F:1232
ele(i,k) = elm(i,k);
// satmedmfvdifq.F:1233
}
// satmedmfvdifq.F:1235
}
// satmedmfvdifq.F:1236
}
// satmedmfvdifq.F:1237
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1238
elm(i,km) = elm(i,km1);
// satmedmfvdifq.F:1239
ele(i,km) = ele(i,km1);
// satmedmfvdifq.F:1240
}
// satmedmfvdifq.F:1246
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1247
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1248
tem = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(elm(i,k)),(double)(elm(i,(k + 1))))));
// satmedmfvdifq.F:1249
tem = sm_mul((double)(tem),(double)(sm_sqrt(tkeh(i,k))));
// satmedmfvdifq.F:1250
ri = sm_max<double>(sm_div((double)(bf(i,k)),(double)(shr2(i,k))),rimin);
// satmedmfvdifq.F:1251
if((k < kpbl(i))) {
// satmedmfvdifq.F:1252
if(pcnvflg(i)) {
// satmedmfvdifq.F:1253
dku(i,k) = sm_mul((double)(ckz(i,k)),(double)(tem));
// satmedmfvdifq.F:1254
dkt(i,k) = sm_div((double)(dku(i,k)),(double)(prn(i,k)));
// satmedmfvdifq.F:1255
} else {
// satmedmfvdifq.F:1256
if((ri < 0x0.0p+0f)) {
// satmedmfvdifq.F:1257
dku(i,k) = sm_mul((double)(ckz(i,k)),(double)(tem));
// satmedmfvdifq.F:1258
dkt(i,k) = sm_div((double)(dku(i,k)),(double)(prn(i,k)));
// satmedmfvdifq.F:1259
} else {
// satmedmfvdifq.F:1260
dkt(i,k) = sm_mul((double)(chz(i,k)),(double)(tem));
// satmedmfvdifq.F:1261
dku(i,k) = sm_mul((double)(dkt(i,k)),(double)(prn(i,k)));
// satmedmfvdifq.F:1262
}
// satmedmfvdifq.F:1263
}
// satmedmfvdifq.F:1264
} else {
// satmedmfvdifq.F:1265
if((ri < 0x0.0p+0f)) {
// satmedmfvdifq.F:1266
dku(i,k) = sm_mul((double)(ck1),(double)(tem));
// satmedmfvdifq.F:1267
dkt(i,k) = sm_mul((double)(rchck),(double)(dku(i,k)));
// satmedmfvdifq.F:1268
} else {
// satmedmfvdifq.F:1269
dkt(i,k) = sm_mul((double)(ch1),(double)(tem));
// satmedmfvdifq.F:1270
prnum = sm_add((double)(0x1.0000000000000p+0f),(double)(sm_mul((double)(0x1.0ccccc0000000p+1f),(double)(ri))));
// satmedmfvdifq.F:1271
prnum = sm_min<double>(prnum,prmax);
// satmedmfvdifq.F:1272
dku(i,k) = sm_mul((double)(dkt(i,k)),(double)(prnum));
// satmedmfvdifq.F:1273
}
// satmedmfvdifq.F:1274
}
// satmedmfvdifq.F:1276
if(scuflg(i)) {
// satmedmfvdifq.F:1277
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:1278
tem1 = sm_mul((double)(ckz(i,k)),(double)(tem));
// satmedmfvdifq.F:1279
ptem1 = sm_div((double)(tem1),(double)(prscu));
// satmedmfvdifq.F:1280
dku(i,k) = sm_max<double>(dku(i,k),tem1);
// satmedmfvdifq.F:1281
dkt(i,k) = sm_max<double>(dkt(i,k),ptem1);
// satmedmfvdifq.F:1282
}
// satmedmfvdifq.F:1283
}
// satmedmfvdifq.F:1285
dkq(i,k) = sm_mul((double)(prtke),(double)(dkt(i,k)));
// satmedmfvdifq.F:1287
dkt(i,k) = sm_min<double>(dkt(i,k),dkmax);
// satmedmfvdifq.F:1288
dkt(i,k) = sm_max<double>(dkt(i,k),xkzo(i,k));
// satmedmfvdifq.F:1289
dkq(i,k) = sm_min<double>(dkq(i,k),dkmax);
// satmedmfvdifq.F:1290
dkq(i,k) = sm_max<double>(dkq(i,k),xkzo(i,k));
// satmedmfvdifq.F:1291
dku(i,k) = sm_min<double>(dku(i,k),dkmax);
// satmedmfvdifq.F:1292
dku(i,k) = sm_max<double>(dku(i,k),xkzmo(i,k));
// satmedmfvdifq.F:1294
}
// satmedmfvdifq.F:1295
}
// satmedmfvdifq.F:1299
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1300
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1301
if((k == 1)) {
// satmedmfvdifq.F:1302
tem = ckz(i,1);
// satmedmfvdifq.F:1303
tem1 = sm_mul((double)(0x1.0000000000000p-1f),(double)(xkzmo(i,1)));
// satmedmfvdifq.F:1304
} else {
// satmedmfvdifq.F:1305
tem = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(ckz(i,(k - 1))),(double)(ckz(i,k)))));
// satmedmfvdifq.F:1306
tem1 = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(xkzmo(i,(k - 1))),(double)(xkzmo(i,k)))));
// satmedmfvdifq.F:1307
}
// satmedmfvdifq.F:1308
ptem = sm_div((double)(tem1),(double)(sm_mul((double)(tem),(double)(elm(i,k)))));
// satmedmfvdifq.F:1309
tkmnz(i,k) = sm_mul((double)(ptem),(double)(ptem));
// satmedmfvdifq.F:1310
tkmnz(i,k) = sm_min<double>(tkmnz(i,k),tkbmx);
// satmedmfvdifq.F:1311
tkmnz(i,k) = sm_max<double>(tkmnz(i,k),tkmin);
// satmedmfvdifq.F:1312
}
// satmedmfvdifq.F:1313
}
// satmedmfvdifq.F:1319
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1320
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1321
if((k == 1)) {
// satmedmfvdifq.F:1322
tem = sm_mul((double)((-dkt(i,1))),(double)(bf(i,1)));
// satmedmfvdifq.F:1326
ptem1 = 0x0.0p+0f;
// satmedmfvdifq.F:1328
if((scuflg(i) && (mrad(i) == 1))) {
// satmedmfvdifq.F:1329
ptem2 = sm_mul((double)(xmfd(i,1)),(double)(buod(i,1)));
// satmedmfvdifq.F:1330
} else {
// satmedmfvdifq.F:1331
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1332
}
// satmedmfvdifq.F:1333
tem = sm_add((double)(sm_add((double)(tem),(double)(ptem1))),(double)(ptem2));
// satmedmfvdifq.F:1334
buop = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(sm_mul((double)(gotvx(i,1)),(double)(sflux(i)))),(double)(tem))));
// satmedmfvdifq.F:1336
tem1 = sm_mul((double)(dku(i,1)),(double)(shr2(i,1)));
// satmedmfvdifq.F:1338
tem = sm_mul((double)(sm_sub((double)(u1(i,2)),(double)(u1(i,1)))),(double)(rdzt(i,1)));
// satmedmfvdifq.F:1343
ptem1 = 0x0.0p+0f;
// satmedmfvdifq.F:1345
if((scuflg(i) && (mrad(i) == 1))) {
// satmedmfvdifq.F:1346
ptem = sm_sub((double)(sm_sub((double)(sm_add((double)(ucdo(i,1)),(double)(ucdo(i,2)))),(double)(u1(i,1)))),(double)(u1(i,2)));
// satmedmfvdifq.F:1347
ptem = sm_mul((double)(sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem))),(double)(xmfd(i,1)))),(double)(ptem));
// satmedmfvdifq.F:1348
} else {
// satmedmfvdifq.F:1349
ptem = 0x0.0p+0f;
// satmedmfvdifq.F:1350
}
// satmedmfvdifq.F:1351
ptem1 = sm_add((double)(ptem1),(double)(ptem));
// satmedmfvdifq.F:1353
tem = sm_mul((double)(sm_sub((double)(v1(i,2)),(double)(v1(i,1)))),(double)(rdzt(i,1)));
// satmedmfvdifq.F:1358
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1360
if((scuflg(i) && (mrad(i) == 1))) {
// satmedmfvdifq.F:1361
ptem = sm_sub((double)(sm_sub((double)(sm_add((double)(vcdo(i,1)),(double)(vcdo(i,2)))),(double)(v1(i,1)))),(double)(v1(i,2)));
// satmedmfvdifq.F:1362
ptem = sm_mul((double)(sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem))),(double)(xmfd(i,1)))),(double)(ptem));
// satmedmfvdifq.F:1363
} else {
// satmedmfvdifq.F:1364
ptem = 0x0.0p+0f;
// satmedmfvdifq.F:1365
}
// satmedmfvdifq.F:1366
ptem2 = sm_add((double)(ptem2),(double)(ptem));
// satmedmfvdifq.F:1368
tem2 = sm_div((double)(sm_mul((double)(sm_mul((double)(stress(i)),(double)(ustar(i)))),(double)(phims(i)))),(double)(sm_mul((double)(vk),(double)(zl(i,1)))));
// satmedmfvdifq.F:1369
shrp = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(sm_add((double)(sm_add((double)(tem1),(double)(ptem1))),(double)(ptem2))),(double)(tem2))));
// satmedmfvdifq.F:1370
} else {
// satmedmfvdifq.F:1371
tem1 = sm_mul((double)((-dkt(i,(k - 1)))),(double)(bf(i,(k - 1))));
// satmedmfvdifq.F:1372
tem2 = sm_mul((double)((-dkt(i,k))),(double)(bf(i,k)));
// satmedmfvdifq.F:1373
tem = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(tem1),(double)(tem2))));
// satmedmfvdifq.F:1374
if((pcnvflg(i) && (k <= kpbl(i)))) {
// satmedmfvdifq.F:1375
ptem = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(xmf(i,(k - 1))),(double)(xmf(i,k)))));
// satmedmfvdifq.F:1376
ptem1 = sm_mul((double)(ptem),(double)(buou(i,k)));
// satmedmfvdifq.F:1377
} else {
// satmedmfvdifq.F:1378
ptem1 = 0x0.0p+0f;
// satmedmfvdifq.F:1379
}
// satmedmfvdifq.F:1380
if(scuflg(i)) {
// satmedmfvdifq.F:1381
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:1382
ptem0 = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(xmfd(i,(k - 1))),(double)(xmfd(i,k)))));
// satmedmfvdifq.F:1383
ptem2 = sm_mul((double)(ptem0),(double)(buod(i,k)));
// satmedmfvdifq.F:1384
} else {
// satmedmfvdifq.F:1385
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1386
}
// satmedmfvdifq.F:1387
} else {
// satmedmfvdifq.F:1388
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1389
}
// satmedmfvdifq.F:1390
buop = sm_add((double)(sm_add((double)(tem),(double)(ptem1))),(double)(ptem2));
// satmedmfvdifq.F:1392
tem1 = sm_mul((double)(dku(i,(k - 1))),(double)(shr2(i,(k - 1))));
// satmedmfvdifq.F:1393
tem2 = sm_mul((double)(dku(i,k)),(double)(shr2(i,k)));
// satmedmfvdifq.F:1394
tem = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(tem1),(double)(tem2))));
// satmedmfvdifq.F:1395
tem1 = sm_mul((double)(sm_sub((double)(u1(i,(k + 1))),(double)(u1(i,k)))),(double)(rdzt(i,k)));
// satmedmfvdifq.F:1396
tem2 = sm_mul((double)(sm_sub((double)(u1(i,k)),(double)(u1(i,(k - 1))))),(double)(rdzt(i,(k - 1))));
// satmedmfvdifq.F:1397
if((pcnvflg(i) && (k <= kpbl(i)))) {
// satmedmfvdifq.F:1398
ptem = sm_add((double)(sm_mul((double)(xmf(i,k)),(double)(tem1))),(double)(sm_mul((double)(xmf(i,(k - 1))),(double)(tem2))));
// satmedmfvdifq.F:1399
ptem1 = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(ptem))),(double)(sm_sub((double)(u1(i,k)),(double)(ucko(i,k)))));
// satmedmfvdifq.F:1400
} else {
// satmedmfvdifq.F:1401
ptem1 = 0x0.0p+0f;
// satmedmfvdifq.F:1402
}
// satmedmfvdifq.F:1403
if(scuflg(i)) {
// satmedmfvdifq.F:1404
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:1405
ptem0 = sm_add((double)(sm_mul((double)(xmfd(i,k)),(double)(tem1))),(double)(sm_mul((double)(xmfd(i,(k - 1))),(double)(tem2))));
// satmedmfvdifq.F:1406
ptem2 = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(ptem0))),(double)(sm_sub((double)(ucdo(i,k)),(double)(u1(i,k)))));
// satmedmfvdifq.F:1407
} else {
// satmedmfvdifq.F:1408
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1409
}
// satmedmfvdifq.F:1410
} else {
// satmedmfvdifq.F:1411
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1412
}
// satmedmfvdifq.F:1413
shrp = sm_add((double)(sm_add((double)(tem),(double)(ptem1))),(double)(ptem2));
// satmedmfvdifq.F:1414
tem1 = sm_mul((double)(sm_sub((double)(v1(i,(k + 1))),(double)(v1(i,k)))),(double)(rdzt(i,k)));
// satmedmfvdifq.F:1415
tem2 = sm_mul((double)(sm_sub((double)(v1(i,k)),(double)(v1(i,(k - 1))))),(double)(rdzt(i,(k - 1))));
// satmedmfvdifq.F:1416
if((pcnvflg(i) && (k <= kpbl(i)))) {
// satmedmfvdifq.F:1417
ptem = sm_add((double)(sm_mul((double)(xmf(i,k)),(double)(tem1))),(double)(sm_mul((double)(xmf(i,(k - 1))),(double)(tem2))));
// satmedmfvdifq.F:1418
ptem1 = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(ptem))),(double)(sm_sub((double)(v1(i,k)),(double)(vcko(i,k)))));
// satmedmfvdifq.F:1419
} else {
// satmedmfvdifq.F:1420
ptem1 = 0x0.0p+0f;
// satmedmfvdifq.F:1421
}
// satmedmfvdifq.F:1422
if(scuflg(i)) {
// satmedmfvdifq.F:1423
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:1424
ptem0 = sm_add((double)(sm_mul((double)(xmfd(i,k)),(double)(tem1))),(double)(sm_mul((double)(xmfd(i,(k - 1))),(double)(tem2))));
// satmedmfvdifq.F:1425
ptem2 = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(ptem0))),(double)(sm_sub((double)(vcdo(i,k)),(double)(v1(i,k)))));
// satmedmfvdifq.F:1426
} else {
// satmedmfvdifq.F:1427
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1428
}
// satmedmfvdifq.F:1429
} else {
// satmedmfvdifq.F:1430
ptem2 = 0x0.0p+0f;
// satmedmfvdifq.F:1431
}
// satmedmfvdifq.F:1432
shrp = sm_add((double)(sm_add((double)(shrp),(double)(ptem1))),(double)(ptem2));
// satmedmfvdifq.F:1433
}
// satmedmfvdifq.F:1434
prod(i,k) = sm_add((double)(buop),(double)(shrp));
// satmedmfvdifq.F:1435
}
// satmedmfvdifq.F:1436
}
// satmedmfvdifq.F:1441
dtn = sm_div((double)(dt2),(double)(((float)(ndt))));
// satmedmfvdifq.F:1442
for(n=1; n<=ndt; n+=1) {
// satmedmfvdifq.F:1443
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1444
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1445
tem = sm_sqrt(tke(i,k));
// satmedmfvdifq.F:1446
ptem = sm_div((double)(ce0),(double)(ele(i,k)));
// satmedmfvdifq.F:1447
diss(i,k) = sm_mul((double)(sm_mul((double)(ptem),(double)(tke(i,k)))),(double)(tem));
// satmedmfvdifq.F:1448
tem1 = sm_add((double)(prod(i,k)),(double)(sm_div((double)(tke(i,k)),(double)(dtn))));
// satmedmfvdifq.F:1449
diss(i,k) = sm_max<double>(sm_min<double>(diss(i,k),tem1),0x0.0p+0f);
// satmedmfvdifq.F:1450
tke(i,k) = sm_add((double)(tke(i,k)),(double)(sm_mul((double)(dtn),(double)(sm_sub((double)(prod(i,k)),(double)(diss(i,k)))))));
// satmedmfvdifq.F:1452
tke(i,k) = sm_max<double>(tke(i,k),tkmnz(i,k));
// satmedmfvdifq.F:1453
}
// satmedmfvdifq.F:1454
}
// satmedmfvdifq.F:1455
}
// satmedmfvdifq.F:1459
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1460
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1461
if(pcnvflg(i)) {
// satmedmfvdifq.F:1462
qcko(i,k,ntke) = tke(i,k);
// satmedmfvdifq.F:1463
}
// satmedmfvdifq.F:1464
if(scuflg(i)) {
// satmedmfvdifq.F:1465
qcdo(i,k,ntke) = tke(i,k);
// satmedmfvdifq.F:1466
}
// satmedmfvdifq.F:1467
}
// satmedmfvdifq.F:1468
}
// satmedmfvdifq.F:1469
for(k=2; k<=kmpbl; k+=1) {
// satmedmfvdifq.F:1470
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1471
if((pcnvflg(i) && (k <= kpbl(i)))) {
// satmedmfvdifq.F:1472
dz = sm_sub((double)(zl(i,k)),(double)(zl(i,(k - 1))));
// satmedmfvdifq.F:1473
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamue(i,(k - 1))))),(double)(dz));
// satmedmfvdifq.F:1474
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// satmedmfvdifq.F:1475
qcko(i,k,ntke) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qcko(i,(k - 1),ntke)))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(tke(i,k)),(double)(tke(i,(k - 1))))))))),(double)(factor));
// satmedmfvdifq.F:1477
}
// satmedmfvdifq.F:1478
}
// satmedmfvdifq.F:1479
}
// satmedmfvdifq.F:1480
for(k=kmscu; k>=1; k+=(-1)) {
// satmedmfvdifq.F:1481
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1482
if((scuflg(i) && (k < krad(i)))) {
// satmedmfvdifq.F:1483
if((k >= mrad(i))) {
// satmedmfvdifq.F:1484
dz = sm_sub((double)(zl(i,(k + 1))),(double)(zl(i,k)));
// satmedmfvdifq.F:1485
tem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(xlamde(i,k)))),(double)(dz));
// satmedmfvdifq.F:1486
factor = sm_add((double)(0x1.0000000000000p+0f),(double)(tem));
// satmedmfvdifq.F:1487
qcdo(i,k,ntke) = sm_div((double)(sm_add((double)(sm_mul((double)(sm_sub((double)(0x1.0000000000000p+0f),(double)(tem))),(double)(qcdo(i,(k + 1),ntke)))),(double)(sm_mul((double)(tem),(double)(sm_add((double)(tke(i,k)),(double)(tke(i,(k + 1))))))))),(double)(factor));
// satmedmfvdifq.F:1489
}
// satmedmfvdifq.F:1490
}
// satmedmfvdifq.F:1491
}
// satmedmfvdifq.F:1492
}
// satmedmfvdifq.F:1498
kps = sm_max<int>(kmpbl,kmscu);
// satmedmfvdifq.F:1502
for(n=1; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:1503
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1504
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1505
qh(i,k,n) = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(q1(i,k,n)),(double)(q1(i,(k + 1),n)))));
// satmedmfvdifq.F:1506
}
// satmedmfvdifq.F:1507
}
// satmedmfvdifq.F:1508
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1509
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1510
q_diff(i,k,n) = sm_sub((double)(q1(i,k,n)),(double)(q1(i,(k + 1),n)));
// satmedmfvdifq.F:1511
}
// satmedmfvdifq.F:1512
}
// satmedmfvdifq.F:1513
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1514
if((q1(i,1,n) >= 0x0.0p+0f)) {
// satmedmfvdifq.F:1515
q_diff(i,0,n) = sm_sub((double)(sm_max<double>(0x0.0p+0f,sm_sub((double)(sm_mul((double)(0x1.0000000000000p+1f),(double)(q1(i,1,n)))),(double)(q1(i,2,n))))),(double)(q1(i,1,n)));
// satmedmfvdifq.F:1517
} else {
// satmedmfvdifq.F:1518
q_diff(i,0,n) = sm_sub((double)(sm_min<double>(0x0.0p+0f,sm_sub((double)(sm_mul((double)(0x1.0000000000000p+1f),(double)(q1(i,1,n)))),(double)(q1(i,2,n))))),(double)(q1(i,1,n)));
// satmedmfvdifq.F:1520
}
// satmedmfvdifq.F:1521
}
// satmedmfvdifq.F:1522
}
// satmedmfvdifq.F:1524
for(n=1; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:1526
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1527
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1528
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1529
q_half(i,k,n) = qh(i,k,n);
// satmedmfvdifq.F:1530
if(((pcnvflg(i) || scuflg(i)) && (k < kmx))) {
// satmedmfvdifq.F:1531
tem = 0x0.0p+0f;
// satmedmfvdifq.F:1532
if((pcnvflg(i) && (k < kpbl(i)))) {
// satmedmfvdifq.F:1533
tem = xmf(i,k);
// satmedmfvdifq.F:1534
}
// satmedmfvdifq.F:1535
if((scuflg(i) && ((k >= mrad(i)) && (k < krad(i))))) {
// satmedmfvdifq.F:1537
tem = sm_sub((double)(tem),(double)(xmfd(i,k)));
// satmedmfvdifq.F:1538
}
// satmedmfvdifq.F:1539
if((tem > 0x0.0p+0f)) {
// satmedmfvdifq.F:1540
rrkp = 0x0.0p+0f;
// satmedmfvdifq.F:1541
if((sm_abs(q_diff(i,k,n)) > 0x1.e392020000000p-74f)) rrkp = sm_div((double)(q_diff(i,(k + 1),n)),(double)(q_diff(i,k,n)));
// satmedmfvdifq.F:1543
phkp = sm_div((double)(sm_add((double)(rrkp),(double)(sm_abs(rrkp)))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_abs(rrkp)))));
// satmedmfvdifq.F:1544
q_half(i,k,n) = sm_add((double)(q1(i,(k + 1),n)),(double)(sm_mul((double)(phkp),(double)(sm_sub((double)(qh(i,k,n)),(double)(q1(i,(k + 1),n)))))));
// satmedmfvdifq.F:1546
} else if((tem < 0x0.0p+0f)) {
// satmedmfvdifq.F:1547
rrkp = 0x0.0p+0f;
// satmedmfvdifq.F:1548
if((sm_abs(q_diff(i,k,n)) > 0x1.e392020000000p-74f)) rrkp = sm_div((double)(q_diff(i,(k - 1),n)),(double)(q_diff(i,k,n)));
// satmedmfvdifq.F:1550
phkp = sm_div((double)(sm_add((double)(rrkp),(double)(sm_abs(rrkp)))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_abs(rrkp)))));
// satmedmfvdifq.F:1551
q_half(i,k,n) = sm_add((double)(q1(i,k,n)),(double)(sm_mul((double)(phkp),(double)(sm_sub((double)(qh(i,k,n)),(double)(q1(i,k,n)))))));
// satmedmfvdifq.F:1553
}
// satmedmfvdifq.F:1554
}
// satmedmfvdifq.F:1555
}
// satmedmfvdifq.F:1556
}
// satmedmfvdifq.F:1558
}
// satmedmfvdifq.F:1562
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1563
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1564
tkeh(i,k) = sm_mul((double)(0x1.0000000000000p-1f),(double)(sm_add((double)(tke(i,k)),(double)(tke(i,(k + 1))))));
// satmedmfvdifq.F:1565
}
// satmedmfvdifq.F:1566
}
// satmedmfvdifq.F:1567
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1568
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1569
e_diff(i,k) = sm_sub((double)(tke(i,k)),(double)(tke(i,(k + 1))));
// satmedmfvdifq.F:1570
}
// satmedmfvdifq.F:1571
}
// satmedmfvdifq.F:1572
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1573
if((tke(i,1) >= 0x0.0p+0f)) {
// satmedmfvdifq.F:1574
e_diff(i,0) = sm_sub((double)(sm_max<double>(0x0.0p+0f,sm_sub((double)(sm_mul((double)(0x1.0000000000000p+1f),(double)(tke(i,1)))),(double)(tke(i,2))))),(double)(tke(i,1)));
// satmedmfvdifq.F:1576
} else {
// satmedmfvdifq.F:1577
e_diff(i,0) = sm_sub((double)(sm_min<double>(0x0.0p+0f,sm_sub((double)(sm_mul((double)(0x1.0000000000000p+1f),(double)(tke(i,1)))),(double)(tke(i,2))))),(double)(tke(i,1)));
// satmedmfvdifq.F:1579
}
// satmedmfvdifq.F:1580
}
// satmedmfvdifq.F:1582
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1583
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1584
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1585
e_half(i,k) = tkeh(i,k);
// satmedmfvdifq.F:1586
if(((pcnvflg(i) || scuflg(i)) && (k < kmx))) {
// satmedmfvdifq.F:1587
tem = 0x0.0p+0f;
// satmedmfvdifq.F:1588
if((pcnvflg(i) && (k < kpbl(i)))) {
// satmedmfvdifq.F:1589
tem = xmf(i,k);
// satmedmfvdifq.F:1590
}
// satmedmfvdifq.F:1591
if((scuflg(i) && ((k >= mrad(i)) && (k < krad(i))))) {
// satmedmfvdifq.F:1593
tem = sm_sub((double)(tem),(double)(xmfd(i,k)));
// satmedmfvdifq.F:1594
}
// satmedmfvdifq.F:1595
if((tem > 0x0.0p+0f)) {
// satmedmfvdifq.F:1596
rrkp = 0x0.0p+0f;
// satmedmfvdifq.F:1597
if((sm_abs(e_diff(i,k)) > 0x1.e392020000000p-74f)) rrkp = sm_div((double)(e_diff(i,(k + 1))),(double)(e_diff(i,k)));
// satmedmfvdifq.F:1599
phkp = sm_div((double)(sm_add((double)(rrkp),(double)(sm_abs(rrkp)))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_abs(rrkp)))));
// satmedmfvdifq.F:1600
e_half(i,k) = sm_add((double)(tke(i,(k + 1))),(double)(sm_mul((double)(phkp),(double)(sm_sub((double)(tkeh(i,k)),(double)(tke(i,(k + 1))))))));
// satmedmfvdifq.F:1602
} else if((tem < 0x0.0p+0f)) {
// satmedmfvdifq.F:1603
rrkp = 0x0.0p+0f;
// satmedmfvdifq.F:1604
if((sm_abs(e_diff(i,k)) > 0x1.e392020000000p-74f)) rrkp = sm_div((double)(e_diff(i,(k - 1))),(double)(e_diff(i,k)));
// satmedmfvdifq.F:1606
phkp = sm_div((double)(sm_add((double)(rrkp),(double)(sm_abs(rrkp)))),(double)(sm_add((double)(0x1.0000000000000p+0f),(double)(sm_abs(rrkp)))));
// satmedmfvdifq.F:1607
e_half(i,k) = sm_add((double)(tke(i,k)),(double)(sm_mul((double)(phkp),(double)(sm_sub((double)(tkeh(i,k)),(double)(tke(i,k)))))));
// satmedmfvdifq.F:1609
}
// satmedmfvdifq.F:1610
}
// satmedmfvdifq.F:1611
}
// satmedmfvdifq.F:1612
}
// satmedmfvdifq.F:1617
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1618
ad(i,1) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:1619
f1(i,1) = tke(i,1);
// satmedmfvdifq.F:1620
}
// satmedmfvdifq.F:1622
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1623
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1624
dtodsd = sm_div((double)(dt2),(double)(del(i,k)));
// satmedmfvdifq.F:1625
dtodsu = sm_div((double)(dt2),(double)(del(i,(k + 1))));
// satmedmfvdifq.F:1626
dsig = sm_sub((double)(prsl(i,k)),(double)(prsl(i,(k + 1))));
// satmedmfvdifq.F:1627
rdz = rdzt(i,k);
// satmedmfvdifq.F:1628
tem1 = sm_mul((double)(sm_mul((double)(dsig),(double)(dkq(i,k)))),(double)(rdz));
// satmedmfvdifq.F:1629
dsdz2 = sm_mul((double)(tem1),(double)(rdz));
// satmedmfvdifq.F:1630
au(i,k) = sm_mul((double)((-dtodsd)),(double)(dsdz2));
// satmedmfvdifq.F:1631
al(i,k) = sm_mul((double)((-dtodsu)),(double)(dsdz2));
// satmedmfvdifq.F:1632
ad(i,k) = sm_sub((double)(ad(i,k)),(double)(au(i,k)));
// satmedmfvdifq.F:1633
ad(i,(k + 1)) = sm_sub((double)(0x1.0000000000000p+0f),(double)(al(i,k)));
// satmedmfvdifq.F:1634
tem2 = sm_mul((double)(dsig),(double)(rdz));
// satmedmfvdifq.F:1636
if((pcnvflg(i) && (k < kpbl(i)))) {
// satmedmfvdifq.F:1637
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmf(i,k)));
// satmedmfvdifq.F:1638
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1639
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1640
ptem = sm_add((double)(qcko(i,k,ntke)),(double)(qcko(i,(k + 1),ntke)));
// satmedmfvdifq.F:1641
f1(i,k) = sm_sub((double)(f1(i,k)),(double)(sm_mul((double)(ptem),(double)(ptem1))));
// satmedmfvdifq.F:1642
f1(i,(k + 1)) = sm_add((double)(tke(i,(k + 1))),(double)(sm_mul((double)(ptem),(double)(ptem2))));
// satmedmfvdifq.F:1643
} else {
// satmedmfvdifq.F:1644
f1(i,(k + 1)) = tke(i,(k + 1));
// satmedmfvdifq.F:1645
}
// satmedmfvdifq.F:1647
if(scuflg(i)) {
// satmedmfvdifq.F:1648
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:1649
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmfd(i,k)));
// satmedmfvdifq.F:1650
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1651
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1652
ptem = sm_add((double)(qcdo(i,k,ntke)),(double)(qcdo(i,(k + 1),ntke)));
// satmedmfvdifq.F:1653
f1(i,k) = sm_add((double)(f1(i,k)),(double)(sm_mul((double)(ptem),(double)(ptem1))));
// satmedmfvdifq.F:1654
f1(i,(k + 1)) = sm_sub((double)(f1(i,(k + 1))),(double)(sm_mul((double)(ptem),(double)(ptem2))));
// satmedmfvdifq.F:1655
}
// satmedmfvdifq.F:1656
}
// satmedmfvdifq.F:1658
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1659
if(((pcnvflg(i) || scuflg(i)) && (k < kmx))) {
// satmedmfvdifq.F:1660
ptem = sm_mul((double)(tem2),(double)(sm_sub((double)(xmf(i,k)),(double)(xmfd(i,k)))));
// satmedmfvdifq.F:1661
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1662
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1663
f1(i,k) = sm_add((double)(f1(i,k)),(double)(sm_mul((double)(e_half(i,k)),(double)(ptem1))));
// satmedmfvdifq.F:1664
f1(i,(k + 1)) = sm_sub((double)(f1(i,(k + 1))),(double)(sm_mul((double)(e_half(i,k)),(double)(ptem2))));
// satmedmfvdifq.F:1665
}
// satmedmfvdifq.F:1667
}
// satmedmfvdifq.F:1668
}
// satmedmfvdifq.F:1672
sm_tridit(im, km, 1, al.p, ad.p, au.p, f1.p, au.p, f1.p, ws);
// satmedmfvdifq.F:1677
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1678
tsumn(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1679
tsump(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1680
rtnp(i) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:1681
}
// satmedmfvdifq.F:1682
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1683
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1684
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:1685
kbx = 1;
// satmedmfvdifq.F:1686
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1687
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:1688
kbx = 1;
// satmedmfvdifq.F:1689
kmx = kpbl(i);
// satmedmfvdifq.F:1690
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:1691
kbx = mrad(i);
// satmedmfvdifq.F:1692
kmx = krad(i);
// satmedmfvdifq.F:1693
}
// satmedmfvdifq.F:1694
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:1696
tem = sm_mul((double)(sm_mul((double)(f1(i,k)),(double)(del(i,k)))),(double)(gravi));
// satmedmfvdifq.F:1697
if((f1(i,k) < 0x0.0p+0f)) tsumn(i) = sm_add((double)(tsumn(i)),(double)(tem));
// satmedmfvdifq.F:1698
if((f1(i,k) > 0x0.0p+0f)) tsump(i) = sm_add((double)(tsump(i)),(double)(tem));
// satmedmfvdifq.F:1699
}
// satmedmfvdifq.F:1700
}
// satmedmfvdifq.F:1701
}
// satmedmfvdifq.F:1702
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1703
if((pcnvflg(i) || scuflg(i))) {
// satmedmfvdifq.F:1704
if(((tsump(i) > 0x0.0p+0f) && (tsumn(i) < 0x0.0p+0f))) {
// satmedmfvdifq.F:1705
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:1706
rtnp(i) = sm_div((double)(tsumn(i)),(double)(tsump(i)));
// satmedmfvdifq.F:1707
} else {
// satmedmfvdifq.F:1708
rtnp(i) = sm_div((double)(tsump(i)),(double)(tsumn(i)));
// satmedmfvdifq.F:1709
}
// satmedmfvdifq.F:1710
}
// satmedmfvdifq.F:1711
}
// satmedmfvdifq.F:1712
}
// satmedmfvdifq.F:1713
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1714
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1715
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:1716
kbx = 1;
// satmedmfvdifq.F:1717
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1718
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:1719
kbx = 1;
// satmedmfvdifq.F:1720
kmx = kpbl(i);
// satmedmfvdifq.F:1721
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:1722
kbx = mrad(i);
// satmedmfvdifq.F:1723
kmx = krad(i);
// satmedmfvdifq.F:1724
}
// satmedmfvdifq.F:1725
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:1727
if((rtnp(i) < 0x0.0p+0f)) {
// satmedmfvdifq.F:1728
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:1729
if((f1(i,k) < 0x0.0p+0f)) f1(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:1730
if((f1(i,k) > 0x0.0p+0f)) f1(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f1(i,k)));
// satmedmfvdifq.F:1731
} else {
// satmedmfvdifq.F:1732
if((f1(i,k) < 0x0.0p+0f)) f1(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f1(i,k)));
// satmedmfvdifq.F:1733
if((f1(i,k) > 0x0.0p+0f)) f1(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:1734
}
// satmedmfvdifq.F:1735
}
// satmedmfvdifq.F:1736
}
// satmedmfvdifq.F:1737
}
// satmedmfvdifq.F:1738
}
// satmedmfvdifq.F:1744
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1745
tsumn(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1746
tsump(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1747
rtnp(i) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:1748
}
// satmedmfvdifq.F:1749
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1750
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1751
tem = sm_mul((double)(sm_mul((double)(f1(i,k)),(double)(del(i,k)))),(double)(gravi));
// satmedmfvdifq.F:1752
if((f1(i,k) < 0x0.0p+0f)) tsumn(i) = sm_add((double)(tsumn(i)),(double)(tem));
// satmedmfvdifq.F:1753
if((f1(i,k) > 0x0.0p+0f)) tsump(i) = sm_add((double)(tsump(i)),(double)(tem));
// satmedmfvdifq.F:1754
}
// satmedmfvdifq.F:1755
}
// satmedmfvdifq.F:1756
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1757
if(((tsump(i) > 0x0.0p+0f) && (tsumn(i) < 0x0.0p+0f))) {
// satmedmfvdifq.F:1758
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:1759
rtnp(i) = sm_div((double)(tsumn(i)),(double)(tsump(i)));
// satmedmfvdifq.F:1760
} else {
// satmedmfvdifq.F:1761
rtnp(i) = sm_div((double)(tsump(i)),(double)(tsumn(i)));
// satmedmfvdifq.F:1762
}
// satmedmfvdifq.F:1763
}
// satmedmfvdifq.F:1764
}
// satmedmfvdifq.F:1765
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1766
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1767
if((rtnp(i) < 0x0.0p+0f)) {
// satmedmfvdifq.F:1768
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:1769
if((f1(i,k) < 0x0.0p+0f)) f1(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:1770
if((f1(i,k) > 0x0.0p+0f)) f1(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f1(i,k)));
// satmedmfvdifq.F:1771
} else {
// satmedmfvdifq.F:1772
if((f1(i,k) < 0x0.0p+0f)) f1(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f1(i,k)));
// satmedmfvdifq.F:1773
if((f1(i,k) > 0x0.0p+0f)) f1(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:1774
}
// satmedmfvdifq.F:1775
}
// satmedmfvdifq.F:1776
}
// satmedmfvdifq.F:1777
}
// satmedmfvdifq.F:1781
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1782
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1784
qtend = sm_mul((double)(sm_sub((double)(f1(i,k)),(double)(q1(i,k,ntke)))),(double)(rdt));
// satmedmfvdifq.F:1785
rtg(i,k,ntke) = sm_add((double)(rtg(i,k,ntke)),(double)(qtend));
// satmedmfvdifq.F:1786
}
// satmedmfvdifq.F:1787
}
// satmedmfvdifq.F:1788
if(ldiag3d) {
// satmedmfvdifq.F:1789
idtend = dtidx((ntke + 100),index_of_process_pbl);
// satmedmfvdifq.F:1790
if((idtend > 0)) {
// satmedmfvdifq.F:1791
for(int sm_k=1;sm_k<=km;++sm_k) for(int sm_i=1;sm_i<=im;++sm_i) dtend(sm_i,sm_k,idtend) = sm_add((double)(dtend(sm_i,sm_k,idtend)),(double)(sm_mul((double)(sm_sub((double)(f1(sm_i,sm_k)),(double)(q1(sm_i,sm_k,ntke)))),(double)(rdt))));
// satmedmfvdifq.F:1793
}
// satmedmfvdifq.F:1794
}
// satmedmfvdifq.F:1798
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1799
ad(i,1) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:1800
f1(i,1) = sm_add((double)(t1(i,1)),(double)(sm_mul((double)(dtdz1(i)),(double)(heat(i)))));
// satmedmfvdifq.F:1801
f2(i,1) = sm_add((double)(q1(i,1,1)),(double)(sm_mul((double)(dtdz1(i)),(double)(evap(i)))));
// satmedmfvdifq.F:1802
}
// satmedmfvdifq.F:1803
if((ntrac1 >= 2)) {
// satmedmfvdifq.F:1804
for(n=2; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:1805
is = ((n - 1) * km);
// satmedmfvdifq.F:1806
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1807
f2(i,(1 + is)) = q1(i,1,n);
// satmedmfvdifq.F:1808
}
// satmedmfvdifq.F:1809
}
// satmedmfvdifq.F:1810
}
// satmedmfvdifq.F:1812
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1813
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1814
dtodsd = sm_div((double)(dt2),(double)(del(i,k)));
// satmedmfvdifq.F:1815
dtodsu = sm_div((double)(dt2),(double)(del(i,(k + 1))));
// satmedmfvdifq.F:1816
dsig = sm_sub((double)(prsl(i,k)),(double)(prsl(i,(k + 1))));
// satmedmfvdifq.F:1817
rdz = rdzt(i,k);
// satmedmfvdifq.F:1818
tem1 = sm_mul((double)(sm_mul((double)(dsig),(double)(dkt(i,k)))),(double)(rdz));
// satmedmfvdifq.F:1819
dsdzt = sm_mul((double)(tem1),(double)(gocp));
// satmedmfvdifq.F:1820
dsdz2 = sm_mul((double)(tem1),(double)(rdz));
// satmedmfvdifq.F:1821
au(i,k) = sm_mul((double)((-dtodsd)),(double)(dsdz2));
// satmedmfvdifq.F:1822
al(i,k) = sm_mul((double)((-dtodsu)),(double)(dsdz2));
// satmedmfvdifq.F:1823
ad(i,k) = sm_sub((double)(ad(i,k)),(double)(au(i,k)));
// satmedmfvdifq.F:1824
ad(i,(k + 1)) = sm_sub((double)(0x1.0000000000000p+0f),(double)(al(i,k)));
// satmedmfvdifq.F:1825
tem2 = sm_mul((double)(dsig),(double)(rdz));
// satmedmfvdifq.F:1827
if((pcnvflg(i) && (k < kpbl(i)))) {
// satmedmfvdifq.F:1828
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmf(i,k)));
// satmedmfvdifq.F:1829
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1830
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1831
tem = sm_add((double)(t1(i,k)),(double)(t1(i,(k + 1))));
// satmedmfvdifq.F:1832
ptem = sm_add((double)(tcko(i,k)),(double)(tcko(i,(k + 1))));
// satmedmfvdifq.F:1833
f1(i,k) = sm_sub((double)(sm_add((double)(f1(i,k)),(double)(sm_mul((double)(dtodsd),(double)(dsdzt))))),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem1))));
// satmedmfvdifq.F:1834
f1(i,(k + 1)) = sm_add((double)(sm_sub((double)(t1(i,(k + 1))),(double)(sm_mul((double)(dtodsu),(double)(dsdzt))))),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem2))));
// satmedmfvdifq.F:1835
ptem = sm_add((double)(qcko(i,k,1)),(double)(qcko(i,(k + 1),1)));
// satmedmfvdifq.F:1836
f2(i,k) = sm_sub((double)(f2(i,k)),(double)(sm_mul((double)(ptem),(double)(ptem1))));
// satmedmfvdifq.F:1837
f2(i,(k + 1)) = sm_add((double)(q1(i,(k + 1),1)),(double)(sm_mul((double)(ptem),(double)(ptem2))));
// satmedmfvdifq.F:1838
} else {
// satmedmfvdifq.F:1839
f1(i,k) = sm_add((double)(f1(i,k)),(double)(sm_mul((double)(dtodsd),(double)(dsdzt))));
// satmedmfvdifq.F:1840
f1(i,(k + 1)) = sm_sub((double)(t1(i,(k + 1))),(double)(sm_mul((double)(dtodsu),(double)(dsdzt))));
// satmedmfvdifq.F:1841
f2(i,(k + 1)) = q1(i,(k + 1),1);
// satmedmfvdifq.F:1842
}
// satmedmfvdifq.F:1844
if(scuflg(i)) {
// satmedmfvdifq.F:1845
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:1846
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmfd(i,k)));
// satmedmfvdifq.F:1847
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1848
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1849
ptem = sm_add((double)(tcdo(i,k)),(double)(tcdo(i,(k + 1))));
// satmedmfvdifq.F:1850
tem = sm_add((double)(t1(i,k)),(double)(t1(i,(k + 1))));
// satmedmfvdifq.F:1851
f1(i,k) = sm_add((double)(f1(i,k)),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem1))));
// satmedmfvdifq.F:1852
f1(i,(k + 1)) = sm_sub((double)(f1(i,(k + 1))),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem2))));
// satmedmfvdifq.F:1853
ptem = sm_add((double)(qcdo(i,k,1)),(double)(qcdo(i,(k + 1),1)));
// satmedmfvdifq.F:1854
f2(i,k) = sm_add((double)(f2(i,k)),(double)(sm_mul((double)(ptem),(double)(ptem1))));
// satmedmfvdifq.F:1855
f2(i,(k + 1)) = sm_sub((double)(f2(i,(k + 1))),(double)(sm_mul((double)(ptem),(double)(ptem2))));
// satmedmfvdifq.F:1856
}
// satmedmfvdifq.F:1857
}
// satmedmfvdifq.F:1859
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1860
if(((pcnvflg(i) || scuflg(i)) && (k < kmx))) {
// satmedmfvdifq.F:1861
ptem = sm_mul((double)(tem2),(double)(sm_sub((double)(xmf(i,k)),(double)(xmfd(i,k)))));
// satmedmfvdifq.F:1862
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1863
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1864
f2(i,k) = sm_add((double)(f2(i,k)),(double)(sm_mul((double)(q_half(i,k,1)),(double)(ptem1))));
// satmedmfvdifq.F:1865
f2(i,(k + 1)) = sm_sub((double)(f2(i,(k + 1))),(double)(sm_mul((double)(q_half(i,k,1)),(double)(ptem2))));
// satmedmfvdifq.F:1866
}
// satmedmfvdifq.F:1868
}
// satmedmfvdifq.F:1869
}
// satmedmfvdifq.F:1871
if((ntrac1 >= 2)) {
// satmedmfvdifq.F:1872
for(n=2; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:1873
is = ((n - 1) * km);
// satmedmfvdifq.F:1874
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:1875
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1876
dtodsd = sm_div((double)(dt2),(double)(del(i,k)));
// satmedmfvdifq.F:1877
dtodsu = sm_div((double)(dt2),(double)(del(i,(k + 1))));
// satmedmfvdifq.F:1878
dsig = sm_sub((double)(prsl(i,k)),(double)(prsl(i,(k + 1))));
// satmedmfvdifq.F:1879
tem2 = sm_mul((double)(dsig),(double)(rdzt(i,k)));
// satmedmfvdifq.F:1881
if((pcnvflg(i) && (k < kpbl(i)))) {
// satmedmfvdifq.F:1882
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmf(i,k)));
// satmedmfvdifq.F:1883
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1884
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1885
ptem = sm_add((double)(qcko(i,k,n)),(double)(qcko(i,(k + 1),n)));
// satmedmfvdifq.F:1886
f2(i,(k + is)) = sm_sub((double)(f2(i,(k + is))),(double)(sm_mul((double)(ptem),(double)(ptem1))));
// satmedmfvdifq.F:1887
f2(i,((k + 1) + is)) = sm_add((double)(q1(i,(k + 1),n)),(double)(sm_mul((double)(ptem),(double)(ptem2))));
// satmedmfvdifq.F:1888
} else {
// satmedmfvdifq.F:1889
f2(i,((k + 1) + is)) = q1(i,(k + 1),n);
// satmedmfvdifq.F:1890
}
// satmedmfvdifq.F:1892
if(scuflg(i)) {
// satmedmfvdifq.F:1893
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:1894
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmfd(i,k)));
// satmedmfvdifq.F:1895
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1896
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1897
ptem = sm_add((double)(qcdo(i,k,n)),(double)(qcdo(i,(k + 1),n)));
// satmedmfvdifq.F:1898
f2(i,(k + is)) = sm_add((double)(f2(i,(k + is))),(double)(sm_mul((double)(ptem),(double)(ptem1))));
// satmedmfvdifq.F:1899
f2(i,((k + 1) + is)) = sm_sub((double)(f2(i,((k + 1) + is))),(double)(sm_mul((double)(ptem),(double)(ptem2))));
// satmedmfvdifq.F:1900
}
// satmedmfvdifq.F:1901
}
// satmedmfvdifq.F:1903
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1904
if(((pcnvflg(i) || scuflg(i)) && (k < kmx))) {
// satmedmfvdifq.F:1905
ptem = sm_mul((double)(tem2),(double)(sm_sub((double)(xmf(i,k)),(double)(xmfd(i,k)))));
// satmedmfvdifq.F:1906
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:1907
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:1908
f2(i,(k + is)) = sm_add((double)(f2(i,(k + is))),(double)(sm_mul((double)(q_half(i,k,n)),(double)(ptem1))));
// satmedmfvdifq.F:1909
f2(i,((k + 1) + is)) = sm_sub((double)(f2(i,((k + 1) + is))),(double)(sm_mul((double)(q_half(i,k,n)),(double)(ptem2))));
// satmedmfvdifq.F:1910
}
// satmedmfvdifq.F:1912
}
// satmedmfvdifq.F:1913
}
// satmedmfvdifq.F:1914
}
// satmedmfvdifq.F:1915
}
// satmedmfvdifq.F:1919
sm_tridin(im, km, ntrac1, al.p, ad.p, au.p, f1.p, f2.p, au.p, f1.p, f2.p, ws);
// satmedmfvdifq.F:1924
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1925
tsumn(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1926
tsump(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1927
rtnp(i) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:1928
}
// satmedmfvdifq.F:1929
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1930
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1931
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:1932
kbx = 1;
// satmedmfvdifq.F:1933
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1934
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:1935
kbx = 1;
// satmedmfvdifq.F:1936
kmx = kpbl(i);
// satmedmfvdifq.F:1937
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:1938
kbx = mrad(i);
// satmedmfvdifq.F:1939
kmx = krad(i);
// satmedmfvdifq.F:1940
}
// satmedmfvdifq.F:1941
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:1943
tem = sm_mul((double)(sm_mul((double)(f2(i,k)),(double)(del(i,k)))),(double)(gravi));
// satmedmfvdifq.F:1944
if((f2(i,k) < 0x0.0p+0f)) tsumn(i) = sm_add((double)(tsumn(i)),(double)(tem));
// satmedmfvdifq.F:1945
if((f2(i,k) > 0x0.0p+0f)) tsump(i) = sm_add((double)(tsump(i)),(double)(tem));
// satmedmfvdifq.F:1946
}
// satmedmfvdifq.F:1947
}
// satmedmfvdifq.F:1948
}
// satmedmfvdifq.F:1949
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1950
if((pcnvflg(i) || scuflg(i))) {
// satmedmfvdifq.F:1951
if(((tsump(i) > 0x0.0p+0f) && (tsumn(i) < 0x0.0p+0f))) {
// satmedmfvdifq.F:1952
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:1953
rtnp(i) = sm_div((double)(tsumn(i)),(double)(tsump(i)));
// satmedmfvdifq.F:1954
} else {
// satmedmfvdifq.F:1955
rtnp(i) = sm_div((double)(tsump(i)),(double)(tsumn(i)));
// satmedmfvdifq.F:1956
}
// satmedmfvdifq.F:1957
}
// satmedmfvdifq.F:1958
}
// satmedmfvdifq.F:1959
}
// satmedmfvdifq.F:1960
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:1961
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1962
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:1963
kbx = 1;
// satmedmfvdifq.F:1964
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:1965
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:1966
kbx = 1;
// satmedmfvdifq.F:1967
kmx = kpbl(i);
// satmedmfvdifq.F:1968
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:1969
kbx = mrad(i);
// satmedmfvdifq.F:1970
kmx = krad(i);
// satmedmfvdifq.F:1971
}
// satmedmfvdifq.F:1972
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:1974
if((rtnp(i) < 0x0.0p+0f)) {
// satmedmfvdifq.F:1975
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:1976
if((f2(i,k) < 0x0.0p+0f)) f2(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:1977
if((f2(i,k) > 0x0.0p+0f)) f2(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,k)));
// satmedmfvdifq.F:1978
} else {
// satmedmfvdifq.F:1979
if((f2(i,k) < 0x0.0p+0f)) f2(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,k)));
// satmedmfvdifq.F:1980
if((f2(i,k) > 0x0.0p+0f)) f2(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:1981
}
// satmedmfvdifq.F:1982
}
// satmedmfvdifq.F:1983
}
// satmedmfvdifq.F:1984
}
// satmedmfvdifq.F:1985
}
// satmedmfvdifq.F:1992
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1993
tsumn(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1994
tsump(i) = 0x0.0p+0f;
// satmedmfvdifq.F:1995
rtnp(i) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:1996
}
// satmedmfvdifq.F:1997
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:1998
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:1999
tem = sm_mul((double)(sm_mul((double)(f2(i,k)),(double)(del(i,k)))),(double)(gravi));
// satmedmfvdifq.F:2000
if((f2(i,k) < 0x0.0p+0f)) tsumn(i) = sm_add((double)(tsumn(i)),(double)(tem));
// satmedmfvdifq.F:2001
if((f2(i,k) > 0x0.0p+0f)) tsump(i) = sm_add((double)(tsump(i)),(double)(tem));
// satmedmfvdifq.F:2002
}
// satmedmfvdifq.F:2003
}
// satmedmfvdifq.F:2004
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2005
if(((tsump(i) > 0x0.0p+0f) && (tsumn(i) < 0x0.0p+0f))) {
// satmedmfvdifq.F:2006
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:2007
rtnp(i) = sm_div((double)(tsumn(i)),(double)(tsump(i)));
// satmedmfvdifq.F:2008
} else {
// satmedmfvdifq.F:2009
rtnp(i) = sm_div((double)(tsump(i)),(double)(tsumn(i)));
// satmedmfvdifq.F:2010
}
// satmedmfvdifq.F:2011
}
// satmedmfvdifq.F:2012
}
// satmedmfvdifq.F:2013
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2014
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2015
if((rtnp(i) < 0x0.0p+0f)) {
// satmedmfvdifq.F:2016
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:2017
if((f2(i,k) < 0x0.0p+0f)) f2(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:2018
if((f2(i,k) > 0x0.0p+0f)) f2(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,k)));
// satmedmfvdifq.F:2019
} else {
// satmedmfvdifq.F:2020
if((f2(i,k) < 0x0.0p+0f)) f2(i,k) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,k)));
// satmedmfvdifq.F:2021
if((f2(i,k) > 0x0.0p+0f)) f2(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:2022
}
// satmedmfvdifq.F:2023
}
// satmedmfvdifq.F:2024
}
// satmedmfvdifq.F:2025
}
// satmedmfvdifq.F:2035
if((ntrac1 >= 2)) {
// satmedmfvdifq.F:2036
is = ((ntcw - 1) * km);
// satmedmfvdifq.F:2037
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:2038
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2039
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:2040
kbx = 1;
// satmedmfvdifq.F:2041
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:2042
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:2043
kbx = 1;
// satmedmfvdifq.F:2044
kmx = kpbl(i);
// satmedmfvdifq.F:2045
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:2046
kbx = mrad(i);
// satmedmfvdifq.F:2047
kmx = krad(i);
// satmedmfvdifq.F:2048
}
// satmedmfvdifq.F:2049
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:2051
if((f2(i,(k + is)) < 0x0.0p+0f)) {
// satmedmfvdifq.F:2052
tem = sm_add((double)(f2(i,k)),(double)(f2(i,(k + is))));
// satmedmfvdifq.F:2053
if((tem >= 0x0.0p+0f)) {
// satmedmfvdifq.F:2054
f2(i,k) = tem;
// satmedmfvdifq.F:2055
f1(i,k) = sm_sub((double)(f1(i,k)),(double)(sm_mul((double)(elocp),(double)(f2(i,(k + is))))));
// satmedmfvdifq.F:2056
f2(i,(k + is)) = 0x0.0p+0f;
// satmedmfvdifq.F:2057
} else if((f2(i,k) > 0x0.0p+0f)) {
// satmedmfvdifq.F:2058
f2(i,(k + is)) = tem;
// satmedmfvdifq.F:2059
f1(i,k) = sm_add((double)(f1(i,k)),(double)(sm_mul((double)(elocp),(double)(f2(i,k)))));
// satmedmfvdifq.F:2060
f2(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:2061
}
// satmedmfvdifq.F:2062
}
// satmedmfvdifq.F:2063
}
// satmedmfvdifq.F:2064
}
// satmedmfvdifq.F:2065
}
// satmedmfvdifq.F:2066
}
// satmedmfvdifq.F:2072
if(((ntrac1 >= 2) && (ntrw > 0))) {
// satmedmfvdifq.F:2073
is = ((ntrw - 1) * km);
// satmedmfvdifq.F:2074
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:2075
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2076
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:2077
kbx = 1;
// satmedmfvdifq.F:2078
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:2079
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:2080
kbx = 1;
// satmedmfvdifq.F:2081
kmx = kpbl(i);
// satmedmfvdifq.F:2082
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:2083
kbx = mrad(i);
// satmedmfvdifq.F:2084
kmx = krad(i);
// satmedmfvdifq.F:2085
}
// satmedmfvdifq.F:2086
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:2088
if((f2(i,(k + is)) < 0x0.0p+0f)) {
// satmedmfvdifq.F:2089
tem = sm_add((double)(f2(i,k)),(double)(f2(i,(k + is))));
// satmedmfvdifq.F:2090
if((tem >= 0x0.0p+0f)) {
// satmedmfvdifq.F:2091
f2(i,k) = tem;
// satmedmfvdifq.F:2092
f1(i,k) = sm_sub((double)(f1(i,k)),(double)(sm_mul((double)(elocp),(double)(f2(i,(k + is))))));
// satmedmfvdifq.F:2093
f2(i,(k + is)) = 0x0.0p+0f;
// satmedmfvdifq.F:2094
} else if((f2(i,k) > 0x0.0p+0f)) {
// satmedmfvdifq.F:2095
f2(i,(k + is)) = tem;
// satmedmfvdifq.F:2096
f1(i,k) = sm_add((double)(f1(i,k)),(double)(sm_mul((double)(elocp),(double)(f2(i,k)))));
// satmedmfvdifq.F:2097
f2(i,k) = 0x0.0p+0f;
// satmedmfvdifq.F:2098
}
// satmedmfvdifq.F:2099
}
// satmedmfvdifq.F:2100
}
// satmedmfvdifq.F:2101
}
// satmedmfvdifq.F:2102
}
// satmedmfvdifq.F:2103
}
// satmedmfvdifq.F:2105
if((ntrac1 >= 2)) {
// satmedmfvdifq.F:2106
for(n=2; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:2107
is = ((n - 1) * km);
// satmedmfvdifq.F:2109
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2110
tsumn(i) = 0x0.0p+0f;
// satmedmfvdifq.F:2111
tsump(i) = 0x0.0p+0f;
// satmedmfvdifq.F:2112
rtnp(i) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:2113
}
// satmedmfvdifq.F:2114
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:2115
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2116
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:2117
kbx = 1;
// satmedmfvdifq.F:2118
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:2119
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:2120
kbx = 1;
// satmedmfvdifq.F:2121
kmx = kpbl(i);
// satmedmfvdifq.F:2122
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:2123
kbx = mrad(i);
// satmedmfvdifq.F:2124
kmx = krad(i);
// satmedmfvdifq.F:2125
}
// satmedmfvdifq.F:2126
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:2128
tem = sm_mul((double)(sm_mul((double)(f2(i,(k + is))),(double)(del(i,k)))),(double)(gravi));
// satmedmfvdifq.F:2129
if((f2(i,(k + is)) < 0x0.0p+0f)) tsumn(i) = sm_add((double)(tsumn(i)),(double)(tem));
// satmedmfvdifq.F:2130
if((f2(i,(k + is)) > 0x0.0p+0f)) tsump(i) = sm_add((double)(tsump(i)),(double)(tem));
// satmedmfvdifq.F:2131
}
// satmedmfvdifq.F:2132
}
// satmedmfvdifq.F:2133
}
// satmedmfvdifq.F:2134
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2135
if((pcnvflg(i) || scuflg(i))) {
// satmedmfvdifq.F:2136
if(((tsump(i) > 0x0.0p+0f) && (tsumn(i) < 0x0.0p+0f))) {
// satmedmfvdifq.F:2137
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:2138
rtnp(i) = sm_div((double)(tsumn(i)),(double)(tsump(i)));
// satmedmfvdifq.F:2139
} else {
// satmedmfvdifq.F:2140
rtnp(i) = sm_div((double)(tsump(i)),(double)(tsumn(i)));
// satmedmfvdifq.F:2141
}
// satmedmfvdifq.F:2142
}
// satmedmfvdifq.F:2143
}
// satmedmfvdifq.F:2144
}
// satmedmfvdifq.F:2145
for(k=1; k<=kps; k+=1) {
// satmedmfvdifq.F:2146
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2147
if((pcnvflg(i) && scuflg(i))) {
// satmedmfvdifq.F:2148
kbx = 1;
// satmedmfvdifq.F:2149
kmx = sm_max<int>(kpbl(i),krad(i));
// satmedmfvdifq.F:2150
} else if((pcnvflg(i) && (!scuflg(i)))) {
// satmedmfvdifq.F:2151
kbx = 1;
// satmedmfvdifq.F:2152
kmx = kpbl(i);
// satmedmfvdifq.F:2153
} else if(((!pcnvflg(i)) && scuflg(i))) {
// satmedmfvdifq.F:2154
kbx = mrad(i);
// satmedmfvdifq.F:2155
kmx = krad(i);
// satmedmfvdifq.F:2156
}
// satmedmfvdifq.F:2157
if(((pcnvflg(i) || scuflg(i)) && ((k >= kbx) && (k <= kmx)))) {
// satmedmfvdifq.F:2159
if((rtnp(i) < 0x0.0p+0f)) {
// satmedmfvdifq.F:2160
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:2161
if((f2(i,(k + is)) < 0x0.0p+0f)) f2(i,(k + is)) = 0x0.0p+0f;
// satmedmfvdifq.F:2162
if((f2(i,(k + is)) > 0x0.0p+0f)) f2(i,(k + is)) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,(k + is))));
// satmedmfvdifq.F:2163
} else {
// satmedmfvdifq.F:2164
if((f2(i,(k + is)) < 0x0.0p+0f)) f2(i,(k + is)) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,(k + is))));
// satmedmfvdifq.F:2165
if((f2(i,(k + is)) > 0x0.0p+0f)) f2(i,(k + is)) = 0x0.0p+0f;
// satmedmfvdifq.F:2166
}
// satmedmfvdifq.F:2167
}
// satmedmfvdifq.F:2168
}
// satmedmfvdifq.F:2169
}
// satmedmfvdifq.F:2170
}
// satmedmfvdifq.F:2177
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2178
tsumn(i) = 0x0.0p+0f;
// satmedmfvdifq.F:2179
tsump(i) = 0x0.0p+0f;
// satmedmfvdifq.F:2180
rtnp(i) = 0x1.0000000000000p+0f;
// satmedmfvdifq.F:2181
}
// satmedmfvdifq.F:2182
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2183
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2184
tem = sm_mul((double)(sm_mul((double)(f2(i,(k + is))),(double)(del(i,k)))),(double)(gravi));
// satmedmfvdifq.F:2185
if((f2(i,(k + is)) < 0x0.0p+0f)) tsumn(i) = sm_add((double)(tsumn(i)),(double)(tem));
// satmedmfvdifq.F:2186
if((f2(i,(k + is)) > 0x0.0p+0f)) tsump(i) = sm_add((double)(tsump(i)),(double)(tem));
// satmedmfvdifq.F:2187
}
// satmedmfvdifq.F:2188
}
// satmedmfvdifq.F:2189
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2190
if(((tsump(i) > 0x0.0p+0f) && (tsumn(i) < 0x0.0p+0f))) {
// satmedmfvdifq.F:2191
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:2192
rtnp(i) = sm_div((double)(tsumn(i)),(double)(tsump(i)));
// satmedmfvdifq.F:2193
} else {
// satmedmfvdifq.F:2194
rtnp(i) = sm_div((double)(tsump(i)),(double)(tsumn(i)));
// satmedmfvdifq.F:2195
}
// satmedmfvdifq.F:2196
}
// satmedmfvdifq.F:2197
}
// satmedmfvdifq.F:2198
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2199
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2200
if((rtnp(i) < 0x0.0p+0f)) {
// satmedmfvdifq.F:2201
if((tsump(i) > sm_abs(tsumn(i)))) {
// satmedmfvdifq.F:2202
if((f2(i,(k + is)) < 0x0.0p+0f)) f2(i,(k + is)) = 0x0.0p+0f;
// satmedmfvdifq.F:2203
if((f2(i,(k + is)) > 0x0.0p+0f)) f2(i,(k + is)) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,(k + is))));
// satmedmfvdifq.F:2204
} else {
// satmedmfvdifq.F:2205
if((f2(i,(k + is)) < 0x0.0p+0f)) f2(i,(k + is)) = sm_mul((double)(sm_add((double)(0x1.0000000000000p+0f),(double)(rtnp(i)))),(double)(f2(i,(k + is))));
// satmedmfvdifq.F:2206
if((f2(i,(k + is)) > 0x0.0p+0f)) f2(i,(k + is)) = 0x0.0p+0f;
// satmedmfvdifq.F:2207
}
// satmedmfvdifq.F:2208
}
// satmedmfvdifq.F:2209
}
// satmedmfvdifq.F:2210
}
// satmedmfvdifq.F:2212
}
// satmedmfvdifq.F:2213
}
// satmedmfvdifq.F:2217
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2218
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2219
ttend = sm_mul((double)(sm_sub((double)(f1(i,k)),(double)(t1(i,k)))),(double)(rdt));
// satmedmfvdifq.F:2220
qtend = sm_mul((double)(sm_sub((double)(f2(i,k)),(double)(q1(i,k,1)))),(double)(rdt));
// satmedmfvdifq.F:2221
tdt(i,k) = sm_add((double)(tdt(i,k)),(double)(ttend));
// satmedmfvdifq.F:2222
rtg(i,k,1) = sm_add((double)(rtg(i,k,1)),(double)(qtend));
// satmedmfvdifq.F:2225
}
// satmedmfvdifq.F:2226
}
// satmedmfvdifq.F:2228
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2229
dtsfc(i) = sm_mul((double)(sm_mul((double)(rho_a(i)),(double)(cp))),(double)(heat(i)));
// satmedmfvdifq.F:2230
dqsfc(i) = sm_mul((double)(sm_mul((double)(rho_a(i)),(double)(hvap))),(double)(evap(i)));
// satmedmfvdifq.F:2231
}
// satmedmfvdifq.F:2233
if((ldiag3d && (!gen_tend))) {
// satmedmfvdifq.F:2234
idtend = dtidx(index_of_temperature,index_of_process_pbl);
// satmedmfvdifq.F:2235
if((idtend >= 1)) {
// satmedmfvdifq.F:2236
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2237
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2238
ttend = sm_mul((double)(sm_sub((double)(f1(i,k)),(double)(t1(i,k)))),(double)(rdt));
// satmedmfvdifq.F:2239
dtend(i,k,idtend) = sm_add((double)(dtend(i,k,idtend)),(double)(sm_mul((double)(ttend),(double)(delt))));
// satmedmfvdifq.F:2240
}
// satmedmfvdifq.F:2241
}
// satmedmfvdifq.F:2242
}
// satmedmfvdifq.F:2244
idtend = dtidx((100 + ntqv),index_of_process_pbl);
// satmedmfvdifq.F:2245
if((idtend >= 1)) {
// satmedmfvdifq.F:2246
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2247
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2248
qtend = sm_mul((double)(sm_sub((double)(f2(i,k)),(double)(q1(i,k,1)))),(double)(rdt));
// satmedmfvdifq.F:2249
dtend(i,k,idtend) = sm_add((double)(dtend(i,k,idtend)),(double)(sm_mul((double)(qtend),(double)(delt))));
// satmedmfvdifq.F:2250
}
// satmedmfvdifq.F:2251
}
// satmedmfvdifq.F:2252
}
// satmedmfvdifq.F:2253
}
// satmedmfvdifq.F:2255
if((ntrac1 >= 2)) {
// satmedmfvdifq.F:2256
for(n=2; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:2257
is = ((n - 1) * km);
// satmedmfvdifq.F:2258
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2259
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2260
qtend = sm_mul((double)(sm_sub((double)(f2(i,(k + is))),(double)(q1(i,k,n)))),(double)(rdt));
// satmedmfvdifq.F:2261
rtg(i,k,n) = sm_add((double)(rtg(i,k,n)),(double)(qtend));
// satmedmfvdifq.F:2262
}
// satmedmfvdifq.F:2263
}
// satmedmfvdifq.F:2264
}
// satmedmfvdifq.F:2265
if((ldiag3d && (!gen_tend))) {
// satmedmfvdifq.F:2267
for(n=2; n<=ntrac1; n+=1) {
// satmedmfvdifq.F:2268
is = ((n - 1) * km);
// satmedmfvdifq.F:2269
idtend = dtidx((n + 100),index_of_process_pbl);
// satmedmfvdifq.F:2270
if((idtend >= 1)) {
// satmedmfvdifq.F:2271
if((n != ntke)) {
// satmedmfvdifq.F:2272
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2273
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2274
qtend = sm_mul((double)(sm_sub((double)(f2(i,(k + is))),(double)(q1(i,k,n)))),(double)(rdt));
// satmedmfvdifq.F:2275
dtend(i,k,idtend) = sm_add((double)(dtend(i,k,idtend)),(double)(sm_mul((double)(qtend),(double)(delt))));
// satmedmfvdifq.F:2276
}
// satmedmfvdifq.F:2277
}
// satmedmfvdifq.F:2278
}
// satmedmfvdifq.F:2279
}
// satmedmfvdifq.F:2280
}
// satmedmfvdifq.F:2281
}
// satmedmfvdifq.F:2282
}
// satmedmfvdifq.F:2286
if(dspheat) {
// satmedmfvdifq.F:2287
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:2288
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2291
ttend = sm_div((double)(diss(i,k)),(double)(cp));
// satmedmfvdifq.F:2292
tdt(i,k) = sm_add((double)(tdt(i,k)),(double)(sm_mul((double)(dspfac),(double)(ttend))));
// satmedmfvdifq.F:2293
}
// satmedmfvdifq.F:2294
}
// satmedmfvdifq.F:2295
if((ldiag3d && (!gen_tend))) {
// satmedmfvdifq.F:2296
idtend = dtidx(index_of_temperature,index_of_process_pbl);
// satmedmfvdifq.F:2297
if((idtend >= 1)) {
// satmedmfvdifq.F:2298
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:2299
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2300
ttend = sm_div((double)(diss(i,k)),(double)(cp));
// satmedmfvdifq.F:2301
dtend(i,k,idtend) = sm_add((double)(dtend(i,k,idtend)),(double)(sm_mul((double)(sm_mul((double)(dspfac),(double)(ttend))),(double)(delt))));
// satmedmfvdifq.F:2302
}
// satmedmfvdifq.F:2303
}
// satmedmfvdifq.F:2304
}
// satmedmfvdifq.F:2305
}
// satmedmfvdifq.F:2306
}
// satmedmfvdifq.F:2310
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2311
ad(i,1) = sm_add((double)(0x1.0000000000000p+0f),(double)(sm_div((double)(sm_mul((double)(dtdz1(i)),(double)(stress(i)))),(double)(spd1(i)))));
// satmedmfvdifq.F:2312
f1(i,1) = u1(i,1);
// satmedmfvdifq.F:2313
f2(i,1) = v1(i,1);
// satmedmfvdifq.F:2314
}
// satmedmfvdifq.F:2316
for(k=1; k<=km1; k+=1) {
// satmedmfvdifq.F:2317
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2318
dtodsd = sm_div((double)(dt2),(double)(del(i,k)));
// satmedmfvdifq.F:2319
dtodsu = sm_div((double)(dt2),(double)(del(i,(k + 1))));
// satmedmfvdifq.F:2320
dsig = sm_sub((double)(prsl(i,k)),(double)(prsl(i,(k + 1))));
// satmedmfvdifq.F:2321
rdz = rdzt(i,k);
// satmedmfvdifq.F:2322
tem1 = sm_mul((double)(sm_mul((double)(dsig),(double)(dku(i,k)))),(double)(rdz));
// satmedmfvdifq.F:2323
dsdz2 = sm_mul((double)(tem1),(double)(rdz));
// satmedmfvdifq.F:2324
au(i,k) = sm_mul((double)((-dtodsd)),(double)(dsdz2));
// satmedmfvdifq.F:2325
al(i,k) = sm_mul((double)((-dtodsu)),(double)(dsdz2));
// satmedmfvdifq.F:2326
ad(i,k) = sm_sub((double)(ad(i,k)),(double)(au(i,k)));
// satmedmfvdifq.F:2327
ad(i,(k + 1)) = sm_sub((double)(0x1.0000000000000p+0f),(double)(al(i,k)));
// satmedmfvdifq.F:2328
tem2 = sm_mul((double)(dsig),(double)(rdz));
// satmedmfvdifq.F:2330
if((pcnvflg(i) && (k < kpbl(i)))) {
// satmedmfvdifq.F:2331
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmf(i,k)));
// satmedmfvdifq.F:2332
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:2333
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:2334
tem = sm_add((double)(u1(i,k)),(double)(u1(i,(k + 1))));
// satmedmfvdifq.F:2335
ptem = sm_add((double)(ucko(i,k)),(double)(ucko(i,(k + 1))));
// satmedmfvdifq.F:2336
f1(i,k) = sm_sub((double)(f1(i,k)),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem1))));
// satmedmfvdifq.F:2337
f1(i,(k + 1)) = sm_add((double)(u1(i,(k + 1))),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem2))));
// satmedmfvdifq.F:2338
tem = sm_add((double)(v1(i,k)),(double)(v1(i,(k + 1))));
// satmedmfvdifq.F:2339
ptem = sm_add((double)(vcko(i,k)),(double)(vcko(i,(k + 1))));
// satmedmfvdifq.F:2340
f2(i,k) = sm_sub((double)(f2(i,k)),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem1))));
// satmedmfvdifq.F:2341
f2(i,(k + 1)) = sm_add((double)(v1(i,(k + 1))),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem2))));
// satmedmfvdifq.F:2342
} else {
// satmedmfvdifq.F:2343
f1(i,(k + 1)) = u1(i,(k + 1));
// satmedmfvdifq.F:2344
f2(i,(k + 1)) = v1(i,(k + 1));
// satmedmfvdifq.F:2345
}
// satmedmfvdifq.F:2347
if(scuflg(i)) {
// satmedmfvdifq.F:2348
if(((k >= mrad(i)) && (k < krad(i)))) {
// satmedmfvdifq.F:2349
ptem = sm_mul((double)(sm_mul((double)(0x1.0000000000000p-1f),(double)(tem2))),(double)(xmfd(i,k)));
// satmedmfvdifq.F:2350
ptem1 = sm_mul((double)(dtodsd),(double)(ptem));
// satmedmfvdifq.F:2351
ptem2 = sm_mul((double)(dtodsu),(double)(ptem));
// satmedmfvdifq.F:2352
tem = sm_add((double)(u1(i,k)),(double)(u1(i,(k + 1))));
// satmedmfvdifq.F:2353
ptem = sm_add((double)(ucdo(i,k)),(double)(ucdo(i,(k + 1))));
// satmedmfvdifq.F:2354
f1(i,k) = sm_add((double)(f1(i,k)),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem1))));
// satmedmfvdifq.F:2355
f1(i,(k + 1)) = sm_sub((double)(f1(i,(k + 1))),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem2))));
// satmedmfvdifq.F:2356
tem = sm_add((double)(v1(i,k)),(double)(v1(i,(k + 1))));
// satmedmfvdifq.F:2357
ptem = sm_add((double)(vcdo(i,k)),(double)(vcdo(i,(k + 1))));
// satmedmfvdifq.F:2358
f2(i,k) = sm_add((double)(f2(i,k)),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem1))));
// satmedmfvdifq.F:2359
f2(i,(k + 1)) = sm_sub((double)(f2(i,(k + 1))),(double)(sm_mul((double)(sm_sub((double)(ptem),(double)(tem))),(double)(ptem2))));
// satmedmfvdifq.F:2360
}
// satmedmfvdifq.F:2361
}
// satmedmfvdifq.F:2363
}
// satmedmfvdifq.F:2364
}
// satmedmfvdifq.F:2368
sm_tridi2(im, km, al.p, ad.p, au.p, f1.p, f2.p, au.p, f1.p, f2.p, ws);
// satmedmfvdifq.F:2372
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2373
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2374
utend = sm_mul((double)(sm_sub((double)(f1(i,k)),(double)(u1(i,k)))),(double)(rdt));
// satmedmfvdifq.F:2375
vtend = sm_mul((double)(sm_sub((double)(f2(i,k)),(double)(v1(i,k)))),(double)(rdt));
// satmedmfvdifq.F:2376
du(i,k) = sm_add((double)(du(i,k)),(double)(utend));
// satmedmfvdifq.F:2377
dv(i,k) = sm_add((double)(dv(i,k)),(double)(vtend));
// satmedmfvdifq.F:2380
}
// satmedmfvdifq.F:2381
}
// satmedmfvdifq.F:2382
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2383
if((icplocn2atm == 0)) {
// satmedmfvdifq.F:2384
dusfc(i) = sm_div((double)(sm_mul((double)(sm_mul((double)(sm_mul((double)((-0x1.0000000000000p+0f)),(double)(rho_a(i)))),(double)(stress(i)))),(double)(u1(i,1)))),(double)(spd1(i)));
// satmedmfvdifq.F:2385
dvsfc(i) = sm_div((double)(sm_mul((double)(sm_mul((double)(sm_mul((double)((-0x1.0000000000000p+0f)),(double)(rho_a(i)))),(double)(stress(i)))),(double)(v1(i,1)))),(double)(spd1(i)));
// satmedmfvdifq.F:2386
} else if((icplocn2atm == 1)) {
// satmedmfvdifq.F:2387
spd1_m = sm_sqrt(sm_add((double)(sm_powi((double)(sm_sub((double)(u1(i,1)),(double)(usfco(i)))),2)),(double)(sm_powi((double)(sm_sub((double)(v1(i,1)),(double)(vsfco(i)))),2))));
// satmedmfvdifq.F:2388
dusfc(i) = sm_div((double)(sm_mul((double)(sm_mul((double)(sm_mul((double)((-0x1.0000000000000p+0f)),(double)(rho_a(i)))),(double)(stress(i)))),(double)(sm_sub((double)(u1(i,1)),(double)(usfco(i)))))),(double)(spd1_m));
// satmedmfvdifq.F:2389
dvsfc(i) = sm_div((double)(sm_mul((double)(sm_mul((double)(sm_mul((double)((-0x1.0000000000000p+0f)),(double)(rho_a(i)))),(double)(stress(i)))),(double)(sm_sub((double)(v1(i,1)),(double)(vsfco(i)))))),(double)(spd1_m));
// satmedmfvdifq.F:2390
}
// satmedmfvdifq.F:2391
}
// satmedmfvdifq.F:2393
if((ldiag3d && (!gen_tend))) {
// satmedmfvdifq.F:2394
idtend = dtidx(index_of_x_wind,index_of_process_pbl);
// satmedmfvdifq.F:2395
if((idtend >= 1)) {
// satmedmfvdifq.F:2396
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2397
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2398
utend = sm_mul((double)(sm_sub((double)(f1(i,k)),(double)(u1(i,k)))),(double)(rdt));
// satmedmfvdifq.F:2399
dtend(i,k,idtend) = sm_add((double)(dtend(i,k,idtend)),(double)(sm_mul((double)(utend),(double)(delt))));
// satmedmfvdifq.F:2400
}
// satmedmfvdifq.F:2401
}
// satmedmfvdifq.F:2402
}
// satmedmfvdifq.F:2404
idtend = dtidx(index_of_y_wind,index_of_process_pbl);
// satmedmfvdifq.F:2405
if((idtend >= 1)) {
// satmedmfvdifq.F:2406
for(k=1; k<=km; k+=1) {
// satmedmfvdifq.F:2407
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2408
vtend = sm_mul((double)(sm_sub((double)(f2(i,k)),(double)(v1(i,k)))),(double)(rdt));
// satmedmfvdifq.F:2409
dtend(i,k,idtend) = sm_add((double)(dtend(i,k,idtend)),(double)(sm_mul((double)(vtend),(double)(delt))));
// satmedmfvdifq.F:2410
}
// satmedmfvdifq.F:2411
}
// satmedmfvdifq.F:2412
}
// satmedmfvdifq.F:2413
}
// satmedmfvdifq.F:2418
for(i=1; i<=im; i+=1) {
// satmedmfvdifq.F:2419
hpbl(i) = hpblx(i);
// satmedmfvdifq.F:2420
kpbl(i) = kpblx(i);
// satmedmfvdifq.F:2421
}
// satmedmfvdifq.F:2424
return;
}

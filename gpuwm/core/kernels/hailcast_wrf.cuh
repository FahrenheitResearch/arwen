// Generated from public-domain WRF v4.6.1 HAILCAST.
// Notice: tools/hail_wrf461/upstream/LICENSE.txt
// SHA256: 8fe1d76319ddcb63473752165485ca3cd2e30e62545620d16a2ee0c9f2721f14
template<int NZ> struct HailColumn {
__device__ void hailstone_driver(const HailArray<float>& tca,const HailArray<float>& h1d,float ht,const HailArray<float>& pa,const HailArray<float>& rho1d,const HailArray<float>& ra,const HailArray<float>& qi1d,const HailArray<float>& qc1d,const HailArray<float>& qr1d,const HailArray<float>& qs1d,const HailArray<float>& qg1d,const HailArray<float>& vuu,float wdur,int nz,float& dhail1,float& dhail2,float& dhail3,float& dhail4,float& dhail5) {
float zbas = 0;
float tbas = 0;
float wbasp = 0;
float rbas = 0;
float cwitot = 0;
int kbas = 0;
float tk_embryo = 0;
float zfzl = 0;
float tfzl = 0;
float wfzlp = 0;
float rfzl = 0;
float vufzl = 0;
float densafzl = 0;
int kfzl = 0;
int nofroze = 0;
int cloudon = 0;
float rtime = 0;
float tau = 0;
float tau_1 = 0;
float tau_2 = 0;
float deltau = 0;
float g = 0;
float r_d = 0;
double dd = 0;
double d = 0;
double d_ice = 0;
float vt = 0;
float v = 0;
float ts = 0;
float tsm1 = 0;
float tsm2 = 0;
float fw = 0;
float water = 0;
float crit = 0;
float dense = 0;
int itype = 0;
HailStorage<float,NZ> ria;
HailStorage<float,NZ> rwa;
HailStorage<float,NZ> vuu_pert;
HailStorage<float,NZ> rwa_adiabat;
HailStorage<float,NZ> rwa_new;
HailStorage<float,NZ> esva;
HailStorage<float,NZ> rsa;
float p = 0;
float rs = 0;
float ri = 0;
float rw = 0;
float xi = 0;
float xw = 0;
float pc = 0;
float tc = 0;
float vu = 0;
float vumax = 0;
float vucore = 0;
float densa = 0;
float z = 0;
float delrw = 0;
HailStorage<float,5> dhails;
float tlayer = 0;
float rlayer = 0;
float player = 0;
float tsum = 0;
float rsum = 0;
float psum = 0;
float ldepth = 0;
float gm = 0;
float gm1 = 0;
float gmw = 0;
float gmi = 0;
float dgm = 0;
float dgmw = 0;
float dgmi = 0;
float dgmv = 0;
float di = 0;
float anu = 0;
float re = 0;
float ae = 0;
float dum = 0;
float icefactor = 0;
float adiabatic_factor = 0;
float sec = 0;
float secdel = 0;
int i = 0;
int j = 0;
int k = 0;
int ifout = 0;
HailStorage<int,1> ind;
// WRF:567
secdel = 5.0f;
// WRF:568
g = 9.81f;
// WRF:569
r_d = 287.f;
// WRF:575
tau = 7200.f;
// WRF:579
for(i=1; i<=nz; i+=1) {
// WRF:580
vuu_pert(i) = (vuu(i)*1.f);
// WRF:581
}
// WRF:585
for(i=1; i<=5; i+=1) {
// WRF:586
dhails(i) = 0.f;
// WRF:587
}
// WRF:590
if((wdur>2000)) {
// WRF:591
rtime = 2000.f;
// WRF:592
} else {
// WRF:593
rtime = wdur;
// WRF:594
}
// WRF:601
kbas = nz;
// WRF:603
for(k=1; k<=nz; k+=1) {
// WRF:604
cwitot = (qi1d(k)+qc1d(k));
// WRF:607
ria(k) = (qi1d(k)+qs1d(k));
// WRF:609
rwa(k) = qc1d(k);
// WRF:614
if((((cwitot>=1.e-12f))&&((k<kbas)))) {
// WRF:615
kbas = k;
// WRF:616
}
// WRF:617
}
// WRF:624
zbas = h1d(kbas);
// WRF:625
tbas = tca(kbas);
// WRF:626
wbasp = pa(kbas);
// WRF:627
rbas = ra(kbas);
// WRF:634
for(k=1; k<=nz; k+=1) {
// WRF:635
rwa_new(k) = rwa(k);
// WRF:636
if((k<kbas)) {
// WRF:637
rwa_adiabat(k) = 0.f;
// WRF:638
continue;
// WRF:639
}
// WRF:642
esva(k) = (611.2f*lake_exp(lake_div((17.67f*((tca(k)-273.155f))),((tca(k)-29.655f)))));
// WRF:644
rsa(k) = lake_div((0.62197f*esva(k)),((pa(k)-esva(k))));
// WRF:647
if((tca(k)>242.155f)) {
// WRF:648
icefactor = 1.f;
// WRF:649
} else if((((tca(k)<=242.155f))&&((tca(k)>235.155f)))) {
// WRF:650
icefactor = ((1-lake_div(((242.155f-tca(k))),5.f)));
// WRF:651
} else {
// WRF:652
icefactor = 0.f;
// WRF:653
}
// WRF:655
if((rbas>rsa(k))) {
// WRF:656
rwa_adiabat(k) = (((rbas-rsa(k)))*icefactor);
// WRF:657
} else {
// WRF:658
rwa_adiabat(k) = rwa(k);
// WRF:659
}
// WRF:662
if((k==kbas)) {
// WRF:663
rwa_new(k) = rwa_adiabat(k);
// WRF:664
} else if((((k>=(kbas+1)))&&((rwa_adiabat(k)>=1.e-12f)))) {
// WRF:665
rwa_new(k) = ((rwa_adiabat(k)*((h1d(k)-h1d((k-1)))))-rwa_new((k-1)));
// WRF:666
if((rwa_new(k)<0)) { rwa_new(k) = 0.f; }
// WRF:667
}
// WRF:668
}
// WRF:670
for(k=(kbas+1); k<=nz; k+=1) {
// WRF:671
rwa_new(k) = lake_div(rwa_new(k),((h1d(k)-h1d((k-1)))));
// WRF:672
}
// WRF:683
for(i=1; i<=5; i+=1) {
// WRF:684
switch(i) {
// WRF:685
case 1:
// WRF:687
dd = 5.e-3f;
// WRF:688
tk_embryo = 265.155f;
// WRF:689
break; case 2:
// WRF:690
dd = 7.5e-3f;
// WRF:691
tk_embryo = 265.155f;
// WRF:692
break; case 3:
// WRF:693
dd = 5.e-3f;
// WRF:694
tk_embryo = 260.155f;
// WRF:695
break; case 4:
// WRF:696
dd = 7.5e-3f;
// WRF:697
tk_embryo = 260.155f;
// WRF:698
break; case 5:
// WRF:699
tk_embryo = 260.155f;
// WRF:700
dd = 1.e-2f;
// WRF:701
break; }
// WRF:704
rtime = 2000.f;
// WRF:705
if((wdur<rtime)) { rtime = wdur; }
sec=0; d=0;
// WRF:707
tfzl = tk_embryo;
// WRF:708
interpp(pa,wfzlp,tca,tk_embryo,ifout,nz);
if(ifout) { dhails(i)=0; continue; }
// WRF:709
interp(h1d,zfzl,wfzlp,ifout,pa,nz);
// WRF:710
interp(ra,rfzl,wfzlp,ifout,pa,nz);
// WRF:711
interp(vuu_pert,vufzl,wfzlp,ifout,pa,nz);
// WRF:712
interp(rho1d,densafzl,wfzlp,ifout,pa,nz);
// WRF:716
if((zfzl<(zbas-1000))) { goto hail_label_400; }
// WRF:720
sec = 0.f;
// WRF:727
p = wfzlp;
// WRF:728
rs = rfzl;
// WRF:729
tc = tfzl;
// WRF:730
vu = vufzl;
// WRF:731
z = (zfzl-ht);
// WRF:732
ldepth = z;
// WRF:733
densa = densafzl;
// WRF:736
nofroze = 1;
// WRF:737
ts = tc;
// WRF:738
tsm1 = ts;
// WRF:739
tsm2 = ts;
// WRF:740
d = dd;
// WRF:741
fw = 0.0f;
// WRF:742
dense = 500.f;
// WRF:743
itype = 1.f;
// WRF:744
cloudon = 1;
// WRF:747
while((sec<tau)) {
// WRF:748
sec = (sec+secdel);
// WRF:754
interp(vuu_pert,vumax,p,ifout,pa,nz);
// WRF:758
if((ifout==1)) { goto hail_label_100; }
// WRF:761
if(((sec>0.0f)&&(sec<rtime))) {
// WRF:762
vucore = ((vumax*(((0.5f*lake_sin(lake_div(((3.14159f*sec)),(rtime))))+0.5f)))*1.2f);
// WRF:763
vu = vucore;
// WRF:764
} else if((sec>=rtime)) {
// WRF:765
vu = 0.0f;
// WRF:766
cloudon = 0;
// WRF:767
}
// WRF:771
terminl(densa,dense,d,vt,tc);
// WRF:774
v = (vu-vt);
// WRF:777
p = (p-(((densa*g)*v)*secdel));
// WRF:778
z = (z+(v*secdel));
// WRF:781
interp(tca,tc,p,ifout,pa,nz);
// WRF:782
interp(ra,rs,p,ifout,pa,nz);
// WRF:785
densa = lake_div(p,(((r_d*((1.f+lake_div((0.609f*rs),((1.f+rs))))))*tc)));
// WRF:788
interp(ria,ri,p,ifout,pa,nz);
// WRF:789
interp(rwa_new,rw,p,ifout,pa,nz);
// WRF:790
xi = ((ri*densa)*cloudon);
// WRF:791
xw = ((rw*densa)*cloudon);
// WRF:792
if((((xw+xi))>0)) {
// WRF:793
pc = lake_div(xi,((xw+xi)));
// WRF:794
} else {
// WRF:795
pc = 1.f;
// WRF:796
}
// WRF:800
vaporclose(delrw,pc,ts,tc,itype);
// WRF:804
massagr(d,gm,gm1,gmw,gmi,dgm,dgmw,dgmi,dgmv,di,anu,re,ae,tc,ts,p,dense,densa,fw,vt,xw,xi,secdel,itype,delrw);
// WRF:809
heatbud(ts,tsm1,tsm2,fw,tc,vt,delrw,d,densa,gm1,gm,dgm,dgmw,dgmv,dgmi,gmw,gmi,di,anu,re,ae,secdel,itype,p);
// WRF:816
water = (fw*gm);
// WRF:818
crit = 2.0e-4f;
// WRF:819
if((water>crit)) {
// WRF:820
breakup(dense,d,gm,fw,crit);
// WRF:821
}
// WRF:825
if((z<=zbas)) { goto hail_label_200; }
// WRF:828
d_ice = lake_pow((lake_div((((6*gm)*((1.f-fw)))),((3.141592654f*dense)))),0.33333333f);
// WRF:831
if((((d_ice<1.e-8f))&&((tc>273.155f)))) { goto hail_label_300; }
// WRF:834
tsm1 = ts;
// WRF:835
tsm2 = tsm1;
// WRF:837
}
// WRF:839
hail_label_100:;
// WRF:840
hail_label_200:;
// WRF:841
hail_label_300:;
// WRF:846
if((p<pa(nz))) {
// WRF:847
d = 0.0f;
// WRF:849
} else if((lake_abs((fw-1.0f))<0.001f)) {
// WRF:850
d = 0.0f;
// WRF:851
} else if((z>0)) {
// WRF:856
tsum = 0.f;
// WRF:857
rsum = 0.f;
// WRF:858
psum = 0.f;
// WRF:859
for(k=1; k<=kbas; k+=1) {
// WRF:860
tsum = (tsum+tca(k));
// WRF:861
psum = (psum+pa(k));
// WRF:862
rsum = (rsum+ra(k));
// WRF:863
}
// WRF:864
tlayer = lake_div(tsum,kbas);
// WRF:865
player = lake_div(psum,kbas);
// WRF:866
rlayer = lake_div(rsum,kbas);
// WRF:871
d_ice = lake_pow((lake_div((((6*gm)*((1.f-fw)))),((3.141592654f*dense)))),0.33333333f);
// WRF:872
d = d_ice;
// WRF:873
melt(d,tlayer,player,rlayer,ldepth,vt,ts,dense);
// WRF:875
}
// WRF:878
hail_label_400:;
// WRF:881
if((sec<60)) { d = 0.f; }
// WRF:884
if((d>0.254f)) { d = 0.f; }
// WRF:887
dhails(i) = (d*1000);
// WRF:889
}
// WRF:892
dhail1 = dhails(1);
// WRF:893
dhail2 = dhails(2);
// WRF:894
dhail3 = dhails(3);
// WRF:895
dhail4 = dhails(4);
// WRF:896
dhail5 = dhails(5);
}
__device__ void interpp(const HailArray<float>& pa,float& pval,const HailArray<float>& ta,float& tval,int& ifout,int& itel) {
int i = 0;
float fract = 0;
// WRF:924
ifout = 1;
// WRF:926
for(i=1; i<=(itel-1); i+=1) {
// WRF:927
if(((((tval<ta(i))&&(tval>=ta((i+1)))))||(((tval>ta(i))&&(tval<=ta((i+1))))))) {
// WRF:930
fract = lake_div(((ta(i)-tval)),((ta(i)-ta((i+1)))));
// WRF:932
pval = (((((1.0f-fract))*pa(i)))+((fract*pa((i+1)))));
// WRF:935
ifout = 0;
// WRF:936
break;
// WRF:937
}
// WRF:938
}
}
__device__ void interp(const HailArray<float>& aa,float& a,float& p,int& ifout,const HailArray<float>& pa,int& itel) {
int i = 0;
float pdiff = 0;
float vdiff = 0;
float rdiff = 0;
float verh = 0;
float adiff = 0;
// WRF:967
ifout = 1;
// WRF:969
for(i=1; i<=(itel-1); i+=1) {
// WRF:970
if(((p<=pa(i))&&(p>pa((i+1))))) {
// WRF:972
pdiff = (pa(i)-pa((i+1)));
// WRF:973
vdiff = (pa(i)-p);
// WRF:974
verh = lake_div(vdiff,pdiff);
// WRF:977
rdiff = (aa((i+1))-aa(i));
// WRF:980
a = (aa(i)+(rdiff*verh));
// WRF:983
ifout = 0;
// WRF:984
break;
// WRF:985
}
// WRF:986
}
}
__device__ void terminl(float& densa,float& dense,double& d,float& vt,float& tc) {
float gmass = 0;
float gx = 0;
float re = 0;
float w = 0;
float y = 0;
const float pi = 3.141592654f;
const float g = 9.78956f;
float anu = 0;
// WRF:1012
gmass = lake_div((((dense*pi)*(lake_pow(d,3.f)))),6.f);
// WRF:1015
anu = (lake_div(((0.00001718f)*((273.155f+120.f))),((tc+120.f)))*lake_pow((lake_div(tc,273.155f)),(1.5f)));
// WRF:1018
gx = lake_div(((((8.0f*gmass)*g)*densa)),((pi*((anu*anu)))));
// WRF:1019
re = lake_pow((lake_div(gx,0.6f)),0.5f);
// WRF:1023
if((gx<550)) {
// WRF:1024
w = lake_log10(gx);
// WRF:1025
y = (((-1.7095f)+(1.33438f*w))-(0.11591f*(lake_pow(w,2.0f))));
// WRF:1026
re = lake_pow(10,y);
// WRF:1027
vt = lake_div((anu*re),((d*densa)));
// WRF:1028
} else if(((gx>=550)&&(gx<1800))) {
// WRF:1029
w = lake_log10(gx);
// WRF:1030
y = ((((-1.81391f)+(1.34671f*w))-(0.12427f*(lake_pow(w,2.0f))))+(0.0063f*(lake_pow(w,3.0f))));
// WRF:1031
re = lake_pow(10,y);
// WRF:1032
vt = lake_div((anu*re),((d*densa)));
// WRF:1033
} else if(((gx>=1800)&&(gx<3.45e08f))) {
// WRF:1034
re = (0.4487f*(lake_pow(gx,0.5536f)));
// WRF:1035
vt = lake_div((anu*re),((d*densa)));
// WRF:1036
} else {
// WRF:1037
re = lake_pow((lake_div(gx,0.6f)),0.5f);
// WRF:1038
vt = lake_div((anu*re),((d*densa)));
// WRF:1039
}
}
__device__ void vaporclose(float& delrw,float& pc,float& ts,float& tc,int& itype) {
float rv = 0;
float alv = 0;
float als = 0;
float ratio = 0;
float esat = 0;
float rhokor = 0;
float esatw = 0;
float rhoomgw = 0;
float esati = 0;
float rhoomgi = 0;
float rhoomg = 0;
// WRF:1066
rv = 461.48f; alv = 2500000.f; als = 2836050.f;
// WRF:1070
ratio = lake_div(1.f,273.155f);
// WRF:1071
if((itype==2)) {
// WRF:1072
esat = (611.f*lake_exp((lake_div(alv,rv)*((ratio-lake_div(1.f,ts))))));
// WRF:1073
} else {
// WRF:1074
esat = (611.f*lake_exp((lake_div(als,rv)*((ratio-lake_div(1.f,ts))))));
// WRF:1075
}
// WRF:1076
rhokor = lake_div(esat,((rv*ts)));
// WRF:1079
esatw = (611.f*lake_exp((lake_div(alv,rv)*((ratio-lake_div(1.f,tc))))));
// WRF:1080
rhoomgw = lake_div(esatw,((rv*tc)));
// WRF:1081
esati = (611.f*lake_exp((lake_div(als,rv)*((ratio-lake_div(1.f,tc))))));
// WRF:1082
rhoomgi = lake_div(esati,((rv*tc)));
// WRF:1084
rhoomg = rhoomgi;
// WRF:1088
delrw = ((rhokor-rhoomg));
}
__device__ void massagr(double& d,float& gm,float& gm1,float& gmw,float& gmi,float& dgm,float& dgmw,float& dgmi,float& dgmv,float& di,float& anu,float& re,float& ae,float& tc,float& ts,float& p,float& dense,float& densa,float& fw,float& vt,float& xw,float& xi,float& sekdel,int& itype,float& delrw) {
float pi = 0;
float d0 = 0;
float gmw2 = 0;
float gmi2 = 0;
float ew = 0;
float ei = 0;
float densel = 0;
float denseli = 0;
float denselw = 0;
float dc = 0;
float voll = 0;
float volt = 0;
float vol1 = 0;
float dgmw_nosoak = 0;
float soak = 0;
float soakm = 0;
float densac = 0;
float e = 0;
float afactor = 0;
float ns = 0;
float tscelsius = 0;
float vimp = 0;
float w = 0;
// WRF:1112
pi = 3.141592654f;
// WRF:1115
d0 = (0.226f*1.e-4f);
// WRF:1116
di = ((d0*lake_pow((lake_div(tc,273.155f)),1.81f))*(lake_div(100000.f,p)));
// WRF:1129
ew = 1.0f;
// WRF:1131
if((tc>=273.155f)) {
// WRF:1132
ei = 1.00f;
// WRF:1133
} else if((tc>=233.155f)) {
// WRF:1134
ei = (1.0f-(lake_div(((273.155f-ts)),40.f)));
// WRF:1135
} else {
// WRF:1136
ei = 0.0f;
// WRF:1137
}
// WRF:1142
densac = ((densa*(1.e3f))*(1.e-6f));
// WRF:1144
anu = ((1.717e-5f*(lake_div(393.0f,((tc+120.0f)))))*lake_pow((lake_div(tc,273.155f)),1.5f));
// WRF:1146
re = lake_div(((d*vt)*densac),anu);
// WRF:1147
e = (lake_pow((0.60f),(0.333333333f))*(lake_pow(re,0.50f)));
// WRF:1149
if((re<6000.0f)) {
// WRF:1150
ae = (0.78f+(0.308f*e));
// WRF:1151
} else if(((re>=6000.0f)&&(re<20000.0f))) {
// WRF:1153
ae = ((0.5f*0.76f)*e);
// WRF:1154
} else if((re>=20000.0f)) {
// WRF:1155
ae = ((0.5f*((0.57f+(9.0e-6f*re))))*e);
// WRF:1156
}
// WRF:1161
gm = ((lake_div(pi,6.f)*(lake_pow(d,3.f)))*dense);
// WRF:1162
gmw = (fw*gm);
// WRF:1163
gmi = (gm-gmw);
// WRF:1166
gm1 = gm;
// WRF:1173
gmw2 = (gmw+(sekdel*(((((lake_div(pi,4.f)*lake_pow(d,2.f))*vt)*xw)*ew))));
// WRF:1174
dgmw = (gmw2-gmw);
// WRF:1175
gmw = gmw2;
// WRF:1178
gmi2 = (gmi+(sekdel*(((((lake_div(pi,4.f)*lake_pow(d,2.f))*vt)*xi)*ei))));
// WRF:1179
dgmi = (gmi2-gmi);
// WRF:1180
gmi = gmi2;
// WRF:1184
dgmv = ((((((sekdel*2)*pi)*d)*ae)*di)*delrw);
// WRF:1185
if((dgmv<0)) { dgmv = 0; }
// WRF:1188
dgm = ((dgmw+dgmi)+dgmv);
// WRF:1190
denselw = 900.f;
// WRF:1191
denseli = 700.f;
// WRF:1193
if((itype==1)) {
// WRF:1196
if((((dgmw>0))||((dgmv>0)))) {
// WRF:1198
dc = (lake_pow((lake_div((0.74f*xw),(((3.14159f*1000.f)*3.e8f)))),0.33333333f)*1.e6f);
// WRF:1200
ns = lake_div((((2*vt)*100.f)*lake_pow(((dc*1.e-4f)),2.f)),((((9*anu)*d)*50)));
// WRF:1203
if((ns<0.1f)) {
// WRF:1204
w = (-1.f);
// WRF:1205
} else {
// WRF:1206
w = lake_log10(ns);
// WRF:1207
}
// WRF:1208
if((re>200)) {
// WRF:1209
if((ns<0.1f)) {
// WRF:1210
vimp = 0.0f;
// WRF:1211
} else if((((ns>=0.1f))&&((ns<=10)))) {
// WRF:1212
vimp = ((((((0.356f+(0.4738f*w))-(0.1233f*lake_pow(w,2.f)))-(0.1618f*lake_pow(w,3.f)))+(0.0807f*lake_pow(w,4.f))))*vt);
// WRF:1214
} else if((ns>10)) {
// WRF:1215
vimp = (0.63f*vt);
// WRF:1216
}
// WRF:1217
} else if((((re>65))&&((re<=200)))) {
// WRF:1218
if((ns<0.1f)) {
// WRF:1219
vimp = 0.0f;
// WRF:1220
} else if((((ns>=0.1f))&&((ns<=10)))) {
// WRF:1221
vimp = ((((((0.3272f+(0.4907f*w))-(0.09452f*lake_pow(w,2.f)))-(0.1906f*lake_pow(w,3.f)))+(0.07105f*lake_pow(w,4.f))))*vt);
// WRF:1223
} else if((ns>10)) {
// WRF:1224
vimp = (0.61f*vt);
// WRF:1225
}
// WRF:1226
} else if((((re>20))&&((re<=65)))) {
// WRF:1227
if((ns<0.1f)) {
// WRF:1228
vimp = 0.0f;
// WRF:1229
} else if((((ns>=0.1f))&&((ns<=10)))) {
// WRF:1230
vimp = ((((((0.2927f+(0.5085f*w))-(0.03453f*lake_pow(w,2.f)))-(0.2184f*lake_pow(w,3.f)))+(0.03595f*lake_pow(w,4.f))))*vt);
// WRF:1232
} else if((ns>10)) {
// WRF:1233
vimp = (0.59f*vt);
// WRF:1234
}
// WRF:1235
} else if((re<=20)) {
// WRF:1236
if((ns<0.4f)) {
// WRF:1237
vimp = 0.0f;
// WRF:1238
} else if((((ns>=0.4f))&&((ns<=10)))) {
// WRF:1239
vimp = ((((((0.1701f+(0.7246f*w))+(0.2257f*lake_pow(w,2.f)))-(1.13f*lake_pow(w,3.f)))+(0.5756f*lake_pow(w,4.f))))*vt);
// WRF:1241
} else if((ns>10)) {
// WRF:1242
vimp = (0.57f*vt);
// WRF:1243
}
// WRF:1244
}
// Physical domain of the empirical impact-speed fit.
vimp = lake_max(vimp,0.f);
// WRF:1247
tscelsius = (ts-273.16f);
// WRF:1248
afactor = lake_div(((-dc)*vimp),tscelsius);
// WRF:1249
if((((tscelsius<=(-5.f)))||((afactor>=(-1.60f))))) {
// WRF:1250
denselw = (0.30f*lake_pow((afactor),0.44f));
// WRF:1251
} else {
// WRF:1252
denselw = lake_exp(((((-0.03115f)-(1.7030f*afactor))+(0.9116f*lake_pow(afactor,2.f)))-(0.1224f*lake_pow(afactor,3.f))));
// WRF:1254
}
// WRF:1256
denselw = (denselw*1000.f);
// WRF:1258
if((denselw<100)) { denselw = 100; }
// WRF:1259
if((denselw>900)) { denselw = 900; }
// WRF:1262
}
// WRF:1263
if((dgmi>0)) {
// WRF:1265
denseli = 700.f;
// WRF:1266
}
// WRF:1269
dgmw_nosoak = dgmw;
// WRF:1272
} else {
// WRF:1276
vol1 = lake_div(gm,dense);
// WRF:1279
soak = ((900*vol1)-gm);
// WRF:1281
soakm = dgmw;
// WRF:1283
if((soakm>soak)) { soakm = soak; }
// WRF:1284
gm = (gm+soakm);
// WRF:1286
dense = lake_div(gm,vol1);
// WRF:1289
dgmw_nosoak = (dgmw-soakm);
// WRF:1292
denselw = 900.f;
// WRF:1293
denseli = 900.f;
// WRF:1295
}
// WRF:1301
if((dgmi<=0)) {
// WRF:1302
voll = lake_div(((dgmw_nosoak+dgmv)),denselw);
// WRF:1303
} else if((dgmw<=0)) {
// WRF:1304
voll = lake_div((dgmi),denseli);
// WRF:1305
} else {
// WRF:1306
voll = (lake_div((dgmi),denseli)+lake_div(((dgmw_nosoak+dgmv)),denselw));
// WRF:1307
}
// WRF:1310
volt = (voll+lake_div(gm,dense));
// WRF:1312
dense = lake_div(((((gm+dgmi)+dgmv)+dgmw_nosoak)),volt);
// WRF:1314
gm = (((gm+dgmi)+dgmw_nosoak)+dgmv);
// WRF:1315
d = lake_pow((lake_div(((6*gm)),((pi*dense)))),0.33333333f);
}
__device__ void heatbud(float& ts,float& tsm1,float& tsm2,float& fw,float& tc,float& vt,float& delrw,double& d,float& densa,float& gm1,float& gm,float& dgm,float& dgmw,float& dgmv,float& dgmi,float& gmw,float& gmi,float& di,float& anu,float& re,float& ae,float& sekdel,int& itype,float& p) {
float rv = 0;
float rd = 0;
float g = 0;
float pi = 0;
float alf = 0;
float alv = 0;
float als = 0;
float ci = 0;
float cw = 0;
float ak = 0;
float h = 0;
float ah = 0;
float tcc = 0;
float tsc = 0;
float delrwc = 0;
float densac = 0;
float tdiff = 0;
float dcm = 0;
float gmc = 0;
float dgmwc = 0;
float dgmic = 0;
float gm1c = 0;
float dgmc = 0;
float dmlt = 0;
float tscm1 = 0;
float tscm2 = 0;
// WRF:1350
rv = 461.48f; rd = 287.04f; g = 9.78956f;
// WRF:1351
pi = 3.141592654f; alf = 79.7f; alv = 597.3f;
// WRF:1352
als = 677.0f; ci = 0.5f; cw = 1.f;
// WRF:1355
tsc = (ts-273.155f);
// WRF:1356
tscm1 = (tsm1-273.155f);
// WRF:1357
tscm2 = (tsm2-273.155f);
// WRF:1358
tcc = (tc-273.155f);
// WRF:1359
delrwc = ((delrw*(1.e3f))*(1.e-6f));
// WRF:1360
densac = ((densa*(1.e3f))*(1.e-6f));
// WRF:1362
dcm = (d*100.f);
// WRF:1363
gmc = (gm*1000.f);
// WRF:1364
gm1c = (gm1*1000.f);
// WRF:1365
dgmwc = (dgmw*1000.f);
// WRF:1366
dgmic = (dgmi*1000.f);
// WRF:1367
dgmc = (dgm*1000.f);
// WRF:1371
ak = (((5.8f+(0.0184f*tcc)))*1.e-5f);
// WRF:1378
h = (lake_pow((0.71f),(0.333333333f))*(lake_pow(re,0.50f)));
// WRF:1382
if((re<6000.0f)) {
// WRF:1383
ah = (0.78f+(0.308f*h));
// WRF:1385
} else if(((re>=6000.0f)&&(re<20000.0f))) {
// WRF:1387
ah = ((0.5f*0.76f)*h);
// WRF:1388
} else if((re>=20000.0f)) {
// WRF:1390
ah = ((0.5f*((0.57f+(9.0e-6f*re))))*h);
// WRF:1392
}
// WRF:1398
if((itype==1)) {
// WRF:1405
tsc = (((0.6f*(((tsc-lake_div((tsc*dgmc),gm1c))+(lake_div(sekdel,((gm1c*ci)))*((((((2.f*pi)*dcm)*((((ah*ak)*((tcc-tsc)))-(((ae*als)*di)*delrwc))))+(lake_div(dgmwc,sekdel)*((alf+(cw*tcc)))))+((lake_div(dgmic,sekdel)*ci)*tcc)))))))+(0.2f*tscm1))+(0.2f*tscm2));
// WRF:1410
ts = (tsc+273.155f);
// WRF:1411
if((ts>=273.155f)) {
// WRF:1412
ts = 273.155f;
// WRF:1413
}
// WRF:1414
tdiff = lake_abs((ts-273.155f));
// WRF:1415
if((tdiff<=1.e-6f)) { itype = 2; }
// WRF:1417
} else if((itype==2)) {
// WRF:1420
if((tcc<0.f)) {
// WRF:1426
fw = ((fw-lake_div((fw*dgmc),gmc))+(lake_div(sekdel,((gmc*alf)))*((((((2.f*pi)*dcm)*((((ah*ak)*tcc)-(((ae*alv)*di)*delrwc))))+(lake_div(dgmwc,sekdel)*((alf+(cw*tcc)))))+((lake_div(dgmic,sekdel)*ci)*tcc)))));
// WRF:1429
} else {
// WRF:1434
dmlt = (lake_div(sekdel,alf)*((((((((2.f*pi)*dcm)*ah)*ak)*tcc)+((((((2.f*pi)*dcm)*ae)*alv)*di)*delrwc))+((lake_div(dgmwc,sekdel)*cw)*tcc))));
// WRF:1436
fw = lake_div(((((fw*gm1c)+dmlt)+dgmwc)),gmc);
// WRF:1437
}
// WRF:1439
if((fw>1.f)) { fw = 1.f; }
// WRF:1440
if((fw<0.f)) { fw = 0.f; }
// WRF:1443
if((fw<=1.e-6f)) {
// WRF:1444
itype = 1;
// WRF:1445
}
// WRF:1447
}
}
__device__ void breakup(float& dense,double& d,float& gm,float& fw,float& crit) {
float water = 0;
float gmi = 0;
float wat = 0;
float pi = 0;
// WRF:1464
pi = 3.141592654f;
// WRF:1466
water = (fw*gm);
// WRF:1477
wat = (water-crit);
// WRF:1478
gm = (gm-wat);
// WRF:1479
fw = lake_div((crit),gm);
// WRF:1481
if((fw>1.0f)) { fw = 1.0f; }
// WRF:1482
if((fw<0.0f)) { fw = 0.0f; }
// WRF:1486
d = lake_pow((lake_div((6.f*gm),((pi*dense)))),(0.333333333f));
}
__device__ void melt(double& d,float& tlayer,float& player,float& rlayer,float& ldepth,float& vt,float& ts,float& dense) {
float eenv = 0;
float delta = 0;
float ewet = 0;
float de = 0;
float der = 0;
float wetold = 0;
float wetbulb = 0;
float wetbulbk = 0;
float tdclayer = 0;
float tclayer = 0;
float eps = 0;
float b = 0;
float hplayer = 0;
double a = 0;
float sd = 0;
float lt = 0;
float ka = 0;
float lf = 0;
float lv = 0;
float t0 = 0;
float dv = 0;
float pi = 0;
float rv = 0;
float rhoice = 0;
float tres = 0;
float re = 0;
float delt = 0;
float esenv = 0;
float rhosenv = 0;
float essfc = 0;
float rhosfc = 0;
float dsig = 0;
float dmdt = 0;
float mass = 0;
float massorg = 0;
float newmass = 0;
float gamma = 0;
float r = 0;
float rho = 0;
int wcnt = 0;
// WRF:1523
eps = 0.622f;
// WRF:1524
tclayer = (tlayer-273.155f);
// WRF:1525
a = 2.53e11f;
// WRF:1526
b = 5.42e3f;
// WRF:1527
tdclayer = lake_div(b,lake_log(lake_div((a*eps),((rlayer*player)))));
// WRF:1528
hplayer = lake_div(player,100.f);
// WRF:1531
eenv = lake_div(((player*rlayer)),((rlayer+eps)));
// WRF:1532
eenv = lake_div(eenv,100.f);
// WRF:1535
gamma = (6.6e-4f*player);
// WRF:1536
delta = lake_div(((4098.0f*eenv)),((((tdclayer+237.7f))*((tdclayer+237.7f)))));
// WRF:1537
wetbulb = lake_div(((((gamma*tclayer))+((delta*tdclayer)))),((gamma+delta)));
// WRF:1540
wcnt = 0;
// WRF:1541
while((wcnt<11)) {
// WRF:1542
ewet = (6.108f*(lake_exp(lake_div(((17.27f*wetbulb)),((237.3f+wetbulb))))));
// WRF:1543
de = ((((0.0006355f*hplayer)*((tclayer-wetbulb))))-((ewet-eenv)));
// WRF:1544
der = (((ewet*((.0091379024f-(lake_div(6106.396f,lake_pow(((273.155f+wetbulb)),2)))))))-((0.0006355f*hplayer)));
// WRF:1546
wetold = wetbulb;
// WRF:1547
wetbulb = (wetbulb-lake_div(de,der));
// WRF:1548
wcnt = (wcnt+1);
// WRF:1549
if(((lake_div(lake_abs((wetbulb-wetold)),wetbulb)>0.0001f))) {
// WRF:1550
break;
// WRF:1551
}
// WRF:1552
}
// WRF:1554
wetbulbk = (wetbulb+273.155f);
// WRF:1555
ka = .02f;
// WRF:1556
lf = 3.34e5f;
// WRF:1557
lv = 2.5e6f;
// WRF:1558
t0 = 273.155f;
// WRF:1559
dv = 0.25e-4f;
// WRF:1560
pi = 3.1415927f;
// WRF:1561
rv = (1004.f-287.f);
// WRF:1563
rhoice = dense;
// WRF:1564
r = lake_div(d,2.f);
// WRF:1567
tres = lake_div(ldepth,vt);
// WRF:1573
rho = lake_div(player,((287.f*tlayer)));
// WRF:1575
re = lake_div((((rho*r)*2)*vt),1.7e-5f);
// WRF:1579
delt = (wetbulb-((ts-273.155f)));
// WRF:1583
esenv = (610.8f*(lake_exp(lake_div(((17.27f*wetbulb)),((237.3f+wetbulb))))));
// WRF:1585
rhosenv = lake_div(esenv,((rv*wetbulbk)));
// WRF:1586
essfc = (610.8f*(lake_exp(lake_div(((17.27f*((t0-273.155f)))),((237.3f+((t0-273.155f))))))));
// WRF:1588
rhosfc = lake_div(essfc,((rv*t0)));
// WRF:1589
dsig = (rhosenv-rhosfc);
// WRF:1592
dmdt = ((lake_div(((((-1.7f)*pi)*r)*(lake_pow(re,0.5f))),lf))*((((ka*delt))+(((((lv-lf))*dv)*dsig)))));
// WRF:1593
if((dmdt>0.f)) { dmdt = 0; }
// WRF:1594
mass = (dmdt*tres);
// WRF:1597
massorg = (((((1.33333333f*pi)*r)*r)*r)*rhoice);
// WRF:1598
newmass = (massorg+mass);
// WRF:1599
if((newmass<0.0f)) { newmass = 0.0f; }
// WRF:1600
d = (2.f*lake_pow((lake_div((0.75f*newmass),((pi*rhoice)))),0.333333333f));
}
};

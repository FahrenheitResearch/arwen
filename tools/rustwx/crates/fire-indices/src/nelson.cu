// Independent CUDA discretization of Nelson (2000), matching nfdrs.rs.
#define N 21
#define PI 3.14159265358979323846
__device__ double clip(double x,double a,double b){return fmin(fmax(x,a),b);}
__device__ double sat(double t){return 610.78*exp(17.2694*t/(t+237.3));}
__device__ double diffusion(double moisture,double t){
 double m=clip(moisture,0.0001,0.3),tk=t+273.2;
 double fa=0.4*m/(1.+0.4*m),fw=(400./1530.+0.4*m)/(1.+0.4*m);
 double c=1./(2.-fa),f=1./(fw*(1.-sqrt(1.-fw)));
 double qv=56720.-42.8*tk,qw=21000.*exp(-14.*m),cv=30.22+0.00994*tk+1.12e-6*tk*tk;
 double db=0.035*c*f*exp(-(qv+qw-cv*tk)/(1.2*8.32*tk));
 double u=0.1617-0.001419*t,z=0.4657-0.003578*t,q=pow(m/u,1./z),dh=exp(-q)*q/(z*m);
 double da=0.0792*pow(tk/273.2,1.75),dv=0.016*da*0.018*(1.-fw)*sat(t)*dh/(400.*8.32*tk);
 double s=clip((fmax(moisture,0.3)-0.3)/1.55,0.,1.),cap=0.;
 if(s>0.286/4. && s<0.286){double nu=0.00158+6.37e-7*pow(fmax(338.76-tk,0.),2.1237);cap=2e-17*9.44e5/(nu*1.84e-5*0.286*400.*1.55)*pow(0.286/s,1.5);}
 return db+dv+cap;
}
extern "C" __global__ void nelson_hour(double *data,int sticks){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=sticks)return;
 int k=j%4; double *packet=data+j*29;
 double p[N];for(int i=0;i<N;i++)p[i]=packet[i];
 double ta=packet[21],rh=packet[22],solar=packet[23],rain=packet[24],prev_rain=packet[25],prev_rh=packet[26],wet=packet[27];
 double radii[4]={0.002,0.0064,0.02,0.0381},hcs[4]={2.5,0.38,0.3,0.12},ads[4]={0.462252733,0.079548303,0.06,0.06};
 double radius=radii[k],hc=hcs[k]*41840.,sky=solar>0.?6.:3.;
 double hr=4.*0.000204*0.85*pow(sky+273.2,3.)/PI;
 double hv=rain>0.?0.:0.622*hc*pow(0.7/0.58,0.667)*(56720.-42.8*(ta+273.2))*0.000772/(1005.*0.018);
 double ts=clip((solar*3600.*0.4/PI+hr*sky+(hc+hv)*ta)/(hr+hc+hv),-80.,65.);
 double qv=56720.-42.8*(ts+273.2),qw=21000.*exp(-14.*fmin(p[N-1],0.3)),pd=rh*sat(ta);
 double ps=clip(pd+101325.*0.000772*(ta-ts)*qv/(qv+qw),0.,0.99*sat(ts)),hs=clip(ps/sat(ts),0.00001,0.99);
 double me=(0.1617-0.001419*ts)*pow(-log1p(-hs),0.4657-0.003578*ts),dx=radius/(N-1),dt=1./60.;
 for(int step=0;step<60;step++){
  double lo[N],diag[N],hi[N],rhs[N],dd[N];for(int i=0;i<N;i++){lo[i]=0.;diag[i]=1.;hi[i]=0.;rhs[i]=p[i];dd[i]=diffusion(p[i],ts);}
  bool boundary=false;double ms=0.;
  if(rain>0.){double captured=rain/PI,uptake=prev_rain<=0.?(rh>=prev_rh?0.8:0.12):0.007*wet/PI;ms=fmin(p[N-1]+dt*uptake*(prev_rain<=0.?1.-exp(-10.*captured):1.),0.35);boundary=true;}
  else if(ts<=237.3*log(pd/610.78)/(17.2694-log(pd/610.78)) || p[N-1]>0.3){
   double ep=0.622*hc*pow(0.7/0.58,0.667)*(ps-pd)/(400.*1005.*101325.*radius);
   double nu=0.00158+6.37e-7*pow(fmax(338.76-(ts+273.2),0.),2.1237),ec=ep>0.?PI*9.44e5*pow(9.2e-6,2.)/(16.*nu*0.5*400.*radius*radius):0.;
   ms=clip(p[N-1]-dt*(2.*ep+ec),fmin(me,0.3),0.35);boundary=true;
  }
  for(int i=0;i<N-1;i++){double r=i*dx,left=fmax(r-dx/2.,0.),right=r+dx/2.,vol=(right*right-left*left)/2.;double a=i>0?dt*left*0.5*(dd[i]+dd[i-1])/(vol*dx):0.,b=dt*right*0.5*(dd[i]+dd[i+1])/(vol*dx);lo[i]=-a;hi[i]=-b;diag[i]=1.+a+b;}
  if(boundary)rhs[N-1]=ms;else{double hm=me>p[N-1]?ads[k]/100.:0.06/100.,bi=hm*dx/dd[N-1];lo[N-1]=-1.;diag[N-1]=1.+bi;rhs[N-1]=bi*me;}
  for(int i=1;i<N;i++){double q=lo[i]/diag[i-1];diag[i]-=q*hi[i-1];rhs[i]-=q*rhs[i-1];}
  p[N-1]=rhs[N-1]/diag[N-1];for(int i=N-2;i>=0;i--)p[i]=(rhs[i]-hi[i]*p[i+1])/diag[i];
  for(int i=0;i<N;i++)p[i]=clip(p[i],0.00001,0.35);
 }
 for(int i=0;i<N;i++)packet[i]=p[i];packet[28]=ts;
}

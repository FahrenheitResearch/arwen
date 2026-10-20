// SPDX-License-Identifier: Apache-2.0

typedef int integer;
typedef int logical;
typedef float real;
typedef double doublereal;
#define TRUE_ 1
#define FALSE_ 0
#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) abs(x)
#define dmin(a,b) ((a) <= (b) ? (a) : (b))
#define dmax(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) dmin(a,b)
#define max(a,b) dmax(a,b)
__device__ __forceinline__ float pow_dd(float*a,float*b){return OWP_POW(*a,*b);}
__device__ __forceinline__ float pow_ri(float*a,int*b){
  int n=*b; float x=*a,r=1.f;
  if(n<0){x=1.f/x; n=-n;}
  while(n){if(n&1)r*=x;n>>=1;if(n)x*=x;}return r;
}
__device__ __forceinline__ float r_mod(float*a,float*b){return fmodf(*a,*b);}
__device__ __forceinline__ float r_sign(float*a,float*b){return *b>=0.f?fabsf(*a):-fabsf(*a);}
__device__ __forceinline__ float r_lg10(float*a){return OWP_LOG10(*a);}
#define exp(x) OWP_EXP((float)(x))
#define log(x) OWP_LOG((float)(x))
#define sqrt(x) sqrtf((float)(x))
#define sin(x) OWP_SIN((float)(x))
#define cos(x) OWP_COS((float)(x))
#define tan(x) tanf((float)(x))
#define asin(x) OWP_ASIN((float)(x))
#define atan(x) OWP_ATAN((float)(x))
#define tanh(x) OWP_TANH((float)(x))
// Generated from Apache-2.0 Noah-OWP; see NOTICE and schema.json.



/* Common Block Declarations */

struct Model {
    doublereal f0011, f0012, f0013, f0016;
    integer f0000, f0001, f0002, f0003, f0004;
    real f0005;
    integer f0006, f0007, f0008, f0009, f0010, f0014, f0015;
    real f0017, f0018, f0019, f0020, f0021;
    integer f0022, f0023, f0024, f0025;
    real f0026[4], f0027[7], f0028[7];
    integer f0029, f0030, f0031, f0032, f0033, f0034, f0035, f0036, f0037,
	    f0038, f0039, f0040, f0041, f0042, f0043, f0044, f0045;
    real f0046[4], f0047[4], f0048[4], f0049[4], f0050[4], f0051[4], f0052[4],
	     f0053, f0054, f0055, f0056, f0057, f0058, f0059, f0060, f0061,
	    f0062, f0063, f0064, f0065, f0066, f0067, f0068, f0069, f0070,
	    f0071;
    logical f0072;
    real f0073[12], f0074[12], f0075, f0076, f0077;
    integer f0078;
    real f0079, f0080, f0081, f0082, f0083, f0084, f0085, f0086, f0087, f0088,
	     f0089, f0090, f0091, f0092, f0093, f0094, f0095, f0096, f0097,
	    f0098, f0099, f0100, f0101, f0102, f0103, f0104, f0105, f0106;
    logical f0107;
    real f0108, f0109[2], f0110[2], f0111[2], f0112[2];
    integer f0113, f0114, f0115, f0116, f0117, f0118, f0119, f0120, f0121,
	    f0122;
    real f0123, f0124, f0125, f0126, f0127, f0128, f0129, f0130, f0131, f0132,
	     f0133, f0134, f0135, f0136, f0137, f0138, f0139, f0140, f0141,
	    f0142, f0143, f0144, f0145, f0146, f0147, f0148, f0149, f0150,
	    f0151, f0152, f0153, f0154, f0155, f0156, f0157, f0158, f0159[2],
	    f0160[2], f0161[2], f0162[2], f0163[2], f0164, f0165, f0166[2],
	    f0167, f0168, f0169;
    integer f0170;
    real f0171, f0172, f0173, f0174, f0175, f0176, f0177, f0178, f0179, f0180,
	     f0181, f0182, f0183, f0184, f0185, f0186, f0187, f0188, f0189,
	    f0190, f0191, f0192, f0193, f0194, f0195, f0196, f0197, f0198,
	    f0199, f0200, f0201, f0202, f0203, f0204, f0205;
    integer f0206;
    real f0207[2], f0208[2], f0209, f0210, f0211, f0212, f0213;
    logical f0214, f0215;
    integer f0216[7];
    real f0217[7], f0218[7], f0219[7], f0220[7], f0221, f0222, f0223, f0224,
	    f0225, f0226, f0227, f0228, f0229, f0230[2], f0231[2], f0232[2],
	    f0233[2], f0234[2], f0235[2], f0236[2], f0237[2], f0238[2], f0239[
	    2], f0240[2], f0241[2], f0242[2], f0243[2], f0244[2], f0245[2],
	    f0246[2], f0247[2], f0248, f0249, f0250, f0251, f0252, f0253,
	    f0254, f0255, f0256, f0257, f0258, f0259, f0260, f0261, f0262,
	    f0263, f0264, f0265, f0266, f0267, f0268, f0269, f0270, f0271,
	    f0272, f0273, f0274, f0275, f0276, f0277, f0278, f0279, f0280,
	    f0281, f0282, f0283, f0284, f0285, f0286, f0287, f0288, f0289,
	    f0290, f0291, f0292, f0293, f0294, f0295, f0296, f0297, f0298,
	    f0299, f0300, f0301, f0302, f0303, f0304, f0305, f0306, f0307,
	    f0308, f0309, f0310, f0311, f0312, f0313, f0314, f0315, f0316,
	    f0317, f0318, f0319, f0320, f0321, f0322, f0323, f0324, f0325,
	    f0326, f0327, f0328, f0329, f0330, f0331, f0332, f0333, f0334,
	    f0335, f0336, f0337;
    integer f0338;
    real f0339, f0340, f0341, f0342, f0343, f0344, f0345, f0346, f0347, f0348,
	     f0349, f0350, f0351, f0352, f0353, f0354, f0355, f0356, f0357,
	    f0358, f0359, f0360, f0361, f0362, f0363, f0364, f0365, f0366,
	    f0367, f0368, f0369, f0370, f0371, f0372, f0373, f0374, f0375,
	    f0376, f0377, f0378, f0379, f0380, f0381, f0382, f0383, f0384,
	    f0385, f0386, f0387, f0388, f0389;
    integer f0390;
    real f0391[4], f0392[4], f0393[4], f0394[4], f0395[4], f0396[4], f0397[4],
	     f0398[4], f0399[3], f0400[3], f0401[3], f0402[3], f0403[3],
	    f0404[3], f0405[3], f0406, f0407;
    integer status;
};




__device__ __forceinline__ /* Subroutine */ int r0000_(Model &m, integer *itime);
__device__ __forceinline__ integer r0001_(Model &m, integer *year);
__device__ __forceinline__ integer r0002_(Model &m, integer *yr);
__device__ __forceinline__ integer r0003_(Model &m, integer *yr, integer *mo, integer *dy);
__device__ __forceinline__ /* Subroutine */ int r0004_(Model &m, integer *days, integer *yr, integer *mo, integer *
	dy);
__device__ __forceinline__ /* Subroutine */ int r0005_(Model &m, integer *yr, integer *mo, integer *dy, integer *
	hr, integer *mi, integer *dminutes, integer *yr2, integer *mo2,
	integer *dy2, integer *hr2, integer *mi2);
__device__ __forceinline__ integer r0006_(Model &m, integer *yr, integer *mo, integer *dy);
__device__ __forceinline__ /* Subroutine */ int r0007_(Model &m, integer *iyear, integer *iday, integer *ihour,
	integer *iminute, integer *isecond, real *latitude, real *longitude,
	real *slope, real *azimuth, real *cosz, real *cosz_horiz__, integer *
	yearlen, real *julian);
__device__ __forceinline__ /* Subroutine */ int r0008_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0009_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0010_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0011_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0012_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0013_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0014_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0015_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0016_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0017_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0018_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0019_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0020_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0021_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0022_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0023_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0024_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0025_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0026_(Model &m, real *gdir, real *ext, integer *ib, integer *ic);
__device__ __forceinline__ /* Subroutine */ int r0027_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0028_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0029_(Model &m, integer *iter, real *vai, real *rhoair, real *hg,
	real *tah, real *zpd, real *z0mg, real *z0hg, real *hcan, real *uc,
	real *z0h, real *fv, integer *vegtyp, real *tv, real *mozg, real *fhg,
	 real *ramg, real *rahg, real *rawg, real *rb);
__device__ __forceinline__ /* Subroutine */ int r0030_(Model &m, integer *vegtyp, real *apar, real *foln, real *tv,
	 real *ei, real *ea, real *sfctmp, real *sfcprs, real *o2, real *co2,
	real *igs, real *btran, real *rb, real *rs, real *psn);
__device__ __forceinline__ /* Subroutine */ int r0031_(Model &m, real *par, real *tv, real *btran, real *eah, real
	*sfcprs, real *rs, real *psn);
__device__ __forceinline__ /* Subroutine */ int r0032_(Model &m, real *tv, real *sfcprs, real *q2sat, real *dqsdt2);
__device__ __forceinline__ /* Subroutine */ int r0033_(Model &m, integer *iter, real *sfctmp, real *rhoair, real *
	h__, real *qair, real *zlvl, real *zpd, real *z0m, real *z0h, real *
	ur, real *moz, integer *mozsgn, real *fm, real *fh, real *fm2, real *
	fh2, real *cm, real *ch, real *fv, real *ch2);
__device__ __forceinline__ /* Subroutine */ int r0034_(Model &m, integer *iter, real *z0, real *thz0, real *thlm,
	real *sfcspd, real *zlm, real *akms, real *akhs, real *rlmo, real *
	wstar2, real *ustar);
__device__ __forceinline__ /* Subroutine */ int r0035_(Model &m, real *t, real *esw, real *esi, real *desw, real *
	desi);
__device__ __forceinline__ real r0036_(Model &m, real *t, real *tfrz);
__device__ __forceinline__ real r0037_(Model &m, real *ab, real *bc);
__device__ __forceinline__ real r0038_(Model &m, real *ab);
__device__ __forceinline__ real r0039_(Model &m, real *zz);
__device__ __forceinline__ real r0040_(Model &m, real *zz, real *rric);
__device__ __forceinline__ real r0041_(Model &m, real *zz);
__device__ __forceinline__ real r0042_(Model &m, real *zz, real *rfac);
__device__ __forceinline__ real r0043_(Model &m, real *xx, real *pihf);
__device__ __forceinline__ real r0044_(Model &m, real *yy);
__device__ __forceinline__ real r0045_(Model &m, real *xx);
__device__ __forceinline__ real r0046_(Model &m, real *yy);
__device__ __forceinline__ /* Subroutine */ int r0047_(Model &m, integer *ice, integer *isnow, real *ssoil, real *
	df, real *hcpct, real *sag, real *snowh, real *tg, real *stc);
__device__ __forceinline__ /* Subroutine */ int r0048_(Model &m, integer *nsnow, integer *nsoil, integer *isnow,
	real *stc, real *zbot, real *df, real *hcpct, real *ssoil, real *phi,
	real *ai, real *bi, real *ci, real *rhsts, real *botflx);
__device__ __forceinline__ /* Subroutine */ int r0049_(Model &m, integer *nsnow, integer *nsoil, integer *isnow,
	real *dt, real *ai, real *bi, real *ci, real *rhsts, real *stc);
__device__ __forceinline__ /* Subroutine */ int r0050_(Model &m, real *a, real *b, real *d__, integer *ntop,
	integer *nsoil, integer *nsnow, real *c__, real *p, real *delta);
__device__ __forceinline__ /* Subroutine */ int r0051_(Model &m, integer *nsnow, integer *nsoil);
__device__ __forceinline__ /* Subroutine */ int r0052_(Model &m, integer *isoil, real *tkelv, real *smc, real *
	sh2o, real *free);
__device__ __forceinline__ /* Subroutine */ int r0053_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0054_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0055_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0056_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0057_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0058_(Model &m, real *dz, real *wliq, real *wice, real *t, real *
	dz2, real *wliq2, real *wice2, real *t2);
__device__ __forceinline__ /* Subroutine */ int r0059_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0060_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0061_(Model &m, real *dt, real *rhstt, real *ai, real *bi, real *
	ci);
__device__ __forceinline__ /* Subroutine */ int r0062_(Model &m, real *dt, real *ai, real *bi, real *ci, real *
	rhstt, real *wplus);
__device__ __forceinline__ /* Subroutine */ int r0063_(Model &m, real *p, real *a, real *b, real *c__, real *d__,
	real *delta, integer *ntop, integer *nnsnow);
__device__ __forceinline__ /* Subroutine */ int r0064_(Model &m, real *wdf, real *wcnd, real *smc, real *fcr,
	integer *isoil);
__device__ __forceinline__ /* Subroutine */ int r0065_(Model &m, real *wdf, real *wcnd, real *smc, real *sice,
	integer *isoil);
__device__ __forceinline__ /* Subroutine */ int r0066_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0067_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0068_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0069_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0070_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0071_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0072_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0073_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0074_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0075_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0076_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0077_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0078_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0079_(Model &m, real *i_0__, real *i_max__, real *yd, real *r1);
__device__ __forceinline__ /* Subroutine */ int r0080_(Model &m, real *yd, real *y0, real *r1, real *fmax, real *
	fsur, real *dt, real *dp, real *bb, real *r2);
__device__ __forceinline__ /* Subroutine */ int r0081_(Model &m, real *fsur, integer *inflmax);
__device__ __forceinline__ /* Subroutine */ int r0082_(Model &m, real *fsur, integer *inflmax);
__device__ __forceinline__ /* Subroutine */ int r0083_(Model &m, real *dt, real *fsur, integer *inflmax);
__device__ __forceinline__ /* Subroutine */ int r0084_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0085_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0086_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0087_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0088_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0089_(Model &m);
__device__ __forceinline__ /* Subroutine */ int r0090_(Model &m);
/* Table of constant values */

static __device__ integer c__0 = 0;
static __device__ integer c__1 = 1;

__device__ __forceinline__ /* Subroutine */ int r0000_(Model &m, integer *itime) {
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer now_year__, now_hour__, now_month__, now_minute__;

    integer idt;

    integer now_day__;

    idt = *itime * (m.f0005 / 60);
    r0005_(m, &m.f0006, &m.f0007, &m.f0008, &m.f0009, &
	    m.f0010, &idt, &now_year__, &now_month__, &now_day__, &
	    now_hour__, &now_minute__);
    i__1 = r0006_(m, &now_year__, &now_month__, &now_day__);
    r0007_(m, &now_year__, &i__1, &now_hour__, &now_minute__, &c__0, &
	    m.f0017, &m.f0018, &m.f0020, &m.f0021, &
	    m.f0248, &m.f0249, &m.f0206, &m.f0205);
    return 0;
} /* r0000_ */

__device__ __forceinline__ integer r0001_(Model &m, integer *year) {
    /* System generated locals */
    integer ret_val;

    ret_val = 28;
    if (*year % 4 == 0) {
	ret_val = 29;
	if (*year % 100 == 0) {
	    ret_val = 28;
	    if (*year % 400 == 0) {
		ret_val = 29;
		if (*year % 3600 == 0) {
		    ret_val = 28;
		}
	    }
	}
    }
    return ret_val;
} /* r0001_ */

__device__ __forceinline__ integer r0002_(Model &m, integer *yr) {
    /* System generated locals */
    integer ret_val;

    ret_val = (*yr - 1) * 365 + (*yr - 1) / 4 - (*yr - 1) / 100 + (*yr - 1) /
	    400 - (*yr - 1) / 3600;
    return ret_val;
} /* r0002_ */

__device__ __forceinline__ integer r0003_(Model &m, integer *yr, integer *mo, integer *dy) {
    /* System generated locals */
    integer ret_val;

    /* Local variables */


    ret_val = r0002_(m, yr) + r0006_(m, yr, mo, dy);
    return ret_val;
} /* r0003_ */

__device__ __forceinline__ /* Subroutine */ int r0004_(Model &m, integer *days, integer *yr, integer *mo, integer *
	dy) {
    /* Initialized data */

    integer mday[12] = { 31,28,31,30,31,30,31,31,30,31,30,31 };

    /* System generated locals */
    integer i__1;

    /* Local variables */

    integer doy, mlen;

    *yr = (integer) (((doublereal) (*days) + .5) / 365.2425) + 1;
L9003:
    i__1 = *yr + 1;
    if (! (r0002_(m, &i__1) <= *days)) {
	goto L9004;
    }
    ++(*yr);
    goto L9003;
L9004:
L9006:
    if (! (r0002_(m, yr) > *days)) {
	goto L9007;
    }
    --(*yr);
    goto L9006;
L9007:
    doy = *days - r0002_(m, yr);
    *mo = 1;
L9009:
    mlen = mday[*mo - 1];
    if (*mo == 2) {
	mlen = r0001_(m, yr);
    }
    if (doy < mlen) {
	goto L9010;
    }
    doy -= mlen;
    ++(*mo);
    goto L9009;
L9010:
    *dy = doy + 1;
    return 0;
} /* r0004_ */

__device__ __forceinline__ /* Subroutine */ int r0005_(Model &m, integer *yr, integer *mo, integer *dy, integer *
	hr, integer *mi, integer *dminutes, integer *yr2, integer *mo2,
	integer *dy2, integer *hr2, integer *mi2) {
    /* System generated locals */
    integer i__1;

    /* Local variables */

    integer total, minute_of_day__, dday;


    total = *hr * 60 + *mi + *dminutes;
    minute_of_day__ = (total % 1440 + 1440) % 1440;
    dday = (total - minute_of_day__) / 1440;
    i__1 = r0003_(m, yr, mo, dy) + dday;
    r0004_(m, &i__1, yr2, mo2, dy2);
    *hr2 = minute_of_day__ / 60;
    *mi2 = minute_of_day__ % 60;
    return 0;
} /* r0005_ */

__device__ __forceinline__ integer r0006_(Model &m, integer *yr, integer *mo, integer *dy) {
    /* Initialized data */

    integer cum[12] = { 0,31,59,90,120,151,181,212,243,273,304,334 };

    /* System generated locals */
    integer ret_val;

    /* Local variables */


    ret_val = cum[(0 + (0 + (*mo - 1 << 2))) / 4] + *dy - 1;
    if (*mo > 2) {
	ret_val += r0001_(m, yr) - 28;
    }
    return ret_val;
} /* r0006_ */

__device__ __forceinline__ /* Subroutine */ int r0007_(Model &m, integer *iyear, integer *iday, integer *ihour,
	integer *iminute, integer *isecond, real *latitude, real *longitude,
	real *slope, real *azimuth, real *cosz, real *cosz_horiz__, integer *
	yearlen, real *julian) {
    /* Initialized data */

    real rc0000 = 3.14159265f;
    real rc0001 = 180.f;
    real rc0002 = 360.f;
    real rc0003 = 365.f;
    real rc0004 = 24.f;
    real rc0005 = 0.f;
    real rc0006 = 23.5f;
    real rc0007 = 80.f;
    real rc0008 = 285.f;
    real rc0009 = 60.f;
    real rc0010 = 3600.f;
    real rc0011 = 15.f;
    real rc0012 = 24.f;
    real rc0013 = 15.f;
    real rc0014 = 12.f;
    real rc0015 = 0.f;
    real rc0016 = 1.f;

    /* System generated locals */
    real r__1;

    /* Builtin functions */


    /* Local variables */
    real degrad, dpd, hrang, declin, obecl, sinob, sxlong, arg, nvx, nvy, nvz,
	     svx, svy, svz, tloctim;

    degrad = rc0000 / rc0001;
    dpd = rc0002 / rc0003;
    *yearlen = 365;
    if (*iyear % 4 == 0) {
	*yearlen = 366;
	if (*iyear % 100 == 0) {
	    *yearlen = 365;
	    if (*iyear % 400 == 0) {
		*yearlen = 366;
		if (*iyear % 3600 == 0) {
		    *yearlen = 365;
		}
	    }
	}
    }
    *julian = (real) (*iday) + (real) (*ihour) / rc0004;
    declin = rc0005;
    obecl = rc0006 * degrad;
    sinob = (real)sin(obecl);
    if (*julian >= rc0007) {
	sxlong = dpd * (*julian - rc0007) * degrad;
    }
    if (*julian < rc0007) {
	sxlong = dpd * (*julian + rc0008) * degrad;
    }
    arg = sinob * (real)sin(sxlong);
    declin = (real)asin(arg);
    tloctim = (real) (*ihour) + (real) (*iminute) / rc0009 + (real) (*isecond)
	     / rc0010 + *longitude / rc0011;
    r__1 = tloctim + rc0012;
    tloctim = (real)r_mod(&r__1, &rc0012);
    hrang = rc0013 * (tloctim - rc0014) * degrad;
    nvx = (real)sin((real)(*azimuth * degrad)) * (real)sin((real)(*slope *
	    degrad));
    nvy = -(real)cos((real)(*azimuth * degrad)) * (real)sin((real)(*slope *
	    degrad));
    nvz = (real)cos((real)(*slope * degrad));
    svx = -(real)sin(hrang) * (real)cos(declin);
    svy = (real)sin((real)(*latitude * degrad)) * (real)cos(hrang) * (real)
	    cos(declin) - (real)cos((real)(*latitude * degrad)) * (real)sin(
	    declin);
    svz = (real)cos((real)(*latitude * degrad)) * (real)cos(hrang) * (real)
	    cos(declin) + (real)sin((real)(*latitude * degrad)) * (real)sin(
	    declin);
    *cosz = nvx * svx + nvy * svy + nvz * svz;
    nvx = rc0015;
    nvy = rc0015;
    nvz = rc0016;
    *cosz_horiz__ = nvx * svx + nvy * svy + nvz * svz;
    return 0;
} /* r0007_ */

__device__ __forceinline__ /* Subroutine */ int r0008_(Model &m) {
    /* Initialized data */

    real rc0000 = 500.f;
    real rc0001 = 917.f;
    real rc0002 = 1360.f;
    real rc0003 = .622f;
    real rc0004 = .378f;
    real rc0005 = 0.f;
    real rc0006 = .7f;
    real rc0007 = .5f;
    real rc0008 = .3f;
    real rc0009 = .1f;
    real rc0010 = .9f;
    real rc0011 = 0.f;
    real rc0012 = 10.f;
    real rc0013 = .263f;
    real rc0014 = 17.67f;
    real rc0015 = 273.15f;
    real rc0016 = 29.65f;
    real rc0017 = 100.f;
    real rc0018 = 2.5f;
    real rc0019 = 1.f;
    real rc0020 = 2.f;
    real rc0021 = 1.f;
    real rc0022 = 54.632f;
    real rc0023 = .2f;
    real rc0024 = .6f;
    real rc0025 = .151977f;
    real rc0026 = 8.313659f;
    real rc0027 = 1.676331f;
    real rc0028 = .00391838f;
    real rc0029 = 1.5f;
    real rc0030 = .023101f;
    real rc0031 = 4.86035f;
    real rc0032 = 10.04f;
    real rc0033 = 1.41f;
    real rc0034 = .09f;
    real rc0035 = 120.f;
    real rc0036 = 67.92f;
    real rc0037 = 51.25f;
    real rc0038 = 2.59f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2, d__3, d__4;

    /* Builtin functions */


    /* Local variables */
    real rho_hail__, rho_grpl__, snow_prob__, rh, pair, prcp_frozen__, qprecc,
	     qprecl, temp, tair_c__, twet_c__, solar_const__, dir_irr__,
	    qv_curr__;

    rho_grpl__ = rc0000;
    rho_hail__ = rc0001;
    solar_const__ = rc0002;
    pair = m.f0180;
    d__1 = (doublereal) (m.f0180 / pair);
    d__2 = (doublereal) (m.f0135 / m.f0131);
    m.f0199 = m.f0181 * (real)pow_dd(&d__1, &d__2);
    m.f0200 = m.f0182;
    m.f0201 = m.f0200 * m.f0180 / (rc0003 + rc0004 *
	    m.f0200);
    m.f0202 = (m.f0180 - rc0004 * m.f0201) / (m.f0135
	    * m.f0181);
    m.f0193 = m.f0173 * pair;
    m.f0194 = m.f0174 * pair;
    m.f0264 = m.f0181;
    qv_curr__ = m.f0182 / (1 - m.f0182);
    m.f0265 = m.f0180 * qv_curr__ / (rc0003 + qv_curr__);
    if (m.f0248 <= rc0005 || m.f0249 <= rc0005) {
	m.f0204 = rc0005;
    } else {
	dir_irr__ = m.f0190 / m.f0249;
	if (dir_irr__ > solar_const__) {
	    m.f0204 = solar_const__ * m.f0248;
	} else {
	    m.f0204 = m.f0190 * (m.f0248 / m.f0249);
	}
    }
    m.f0207[0] = m.f0204 * rc0006 * rc0007;
    m.f0207[1] = m.f0204 * rc0006 * rc0007;
    m.f0208[0] = m.f0204 * rc0008 * rc0007;
    m.f0208[1] = m.f0204 * rc0008 * rc0007;
    m.f0183 = m.f0185;
    if (m.f0029 == 4) {
	qprecc = m.f0184 + m.f0186;
	qprecl = m.f0185;
    } else {
	qprecc = rc0009 * m.f0183;
	qprecl = rc0010 * m.f0183;
    }
    m.f0357 = rc0005;
    if (qprecc + qprecl > rc0011) {
	m.f0357 = (qprecc + qprecl) / (rc0012 * qprecc + qprecl);
    }
    if (m.f0029 == 6 || m.f0029 == 7) {
	rh = rc0013 * m.f0180 * m.f0182 * (1 / (real)exp((real)(
		rc0014 * (m.f0181 - rc0015) / (m.f0181 - rc0016)))
		);
	rh = min(rh,rc0017);
    }
    if (m.f0029 == 1) {
	if (m.f0181 > m.f0125 + rc0018) {
	    m.f0203 = rc0011;
	} else {
	    if (m.f0181 <= m.f0125 + rc0007) {
		m.f0203 = rc0019;
	    } else if (m.f0181 <= m.f0125 + rc0020) {
		m.f0203 = rc0021 - (-rc0022 + rc0023 * m.f0181);
	    } else {
		m.f0203 = rc0024;
	    }
	}
    } else if (m.f0029 == 2 || m.f0029 == 3 || m.f0029 == 5
	    || m.f0029 == 6) {
	if (m.f0029 == 6) {
	    tair_c__ = m.f0181 - m.f0125;
	    d__1 = (doublereal) (rh + rc0026);
	    d__2 = (doublereal) rc0007;
	    d__3 = (doublereal) rh;
	    d__4 = (doublereal) rc0029;
	    twet_c__ = tair_c__ * (real)atan((real)(rc0025 * (real)pow_dd(&
		    d__1, &d__2))) + (real)atan((real)(tair_c__ + rh)) - (
		    real)atan((real)(rh - rc0027)) + rc0028 * (real)pow_dd(&
		    d__3, &d__4) * (real)atan((real)(rc0030 * rh)) - rc0031;
	    temp = twet_c__ + m.f0125;
	} else {
	    temp = m.f0181;
	}
	if (temp >= m.f0178) {
	    m.f0203 = rc0011;
	} else {
	    m.f0203 = rc0019;
	}
    } else if (m.f0029 == 4) {
	prcp_frozen__ = m.f0187 + m.f0188 + m.f0189;
	if (m.f0185 > rc0011 && prcp_frozen__ > rc0011) {
/* Computing MIN */
	    r__1 = rc0019, r__2 = prcp_frozen__ / m.f0185;
	    m.f0203 = min(r__1,r__2);
	    m.f0203 = max(rc0005,m.f0203);
	} else {
	    m.f0203 = rc0005;
	}
    } else if (m.f0029 == 7) {
	tair_c__ = m.f0181 - m.f0125;
	snow_prob__ = 1 / ((real)exp((real)(-rc0032 + rc0033 * tair_c__ +
		rc0034 * rh)) + 1);
	if (snow_prob__ >= rc0007) {
	    m.f0203 = rc0019;
	} else {
	    m.f0203 = rc0005;
	}
    } else {
    }
    m.f0354 = m.f0183 * (rc0021 - m.f0203);
    m.f0355 = m.f0183 * m.f0203;
/* Computing MIN */
    r__1 = rc0035, r__2 = rc0036 + rc0037 * (real)exp((real)((m.f0181 -
	    m.f0125) / rc0038));
    m.f0356 = min(r__1,r__2);
    if (m.f0029 == 4) {
	m.f0356 = m.f0356 * (m.f0187 / prcp_frozen__) +
		rho_grpl__ * (m.f0188 / prcp_frozen__) + rho_hail__ * (
		m.f0189 / prcp_frozen__);
    }
/* Computing MAX */
    d__1 = (doublereal) m.f0195;
    d__2 = (doublereal) rc0020;
    d__3 = (doublereal) m.f0196;
    d__4 = (doublereal) rc0020;
    r__1 = (real)sqrt((real)((real)pow_dd(&d__1, &d__2) + (real)pow_dd(&d__3,
	    &d__4)));
    m.f0198 = max(r__1,rc0021);
    return 0;
} /* r0008_ */

__device__ __forceinline__ /* Subroutine */ int r0009_(Model &m) {


    r0008_(m);
    return 0;
} /* r0009_ */

__device__ __forceinline__ /* Subroutine */ int r0010_(Model &m) {


    r0011_(m);
    r0012_(m);
    return 0;
} /* r0010_ */

__device__ __forceinline__ /* Subroutine */ int r0011_(Model &m) {
    /* Initialized data */

    real rc0001 = .5f;
    real rc0002 = 12.f;
    real rc0003 = 1.f;
    real rc0004 = .05f;
    real rc0005 = .1f;
    real rc0006 = 0.f;
    real rc0007 = 1e-6f;
    real rc0008 = 1.f;
    real rc0009 = .2f;
    real rc0010 = .52f;
    real rc0000 = 0.f;

    /* System generated locals */
    real r__1, r__2, r__3;

    /* Builtin functions */


    /* Local variables */
    real t, db, fb;
    integer it1, it2;
    real wt1, wt2, snowhc, day;

    if (m.f0023 == 0) {
	if (m.f0034 == 1 || m.f0034 == 3 || m.f0034 == 4) {
	    if (m.f0017 >= rc0000) {
		day = m.f0205;
	    } else {
		r__1 = m.f0205 + rc0001 * m.f0206;
		r__2 = (real) m.f0206;
		day = (real)r_mod(&r__1, &r__2);
	    }
	    t = rc0002 * day / (real) m.f0206;
	    it1 = t + rc0001;
	    it2 = it1 + 1;
	    wt1 = it1 + rc0001 - t;
	    wt2 = rc0003 - wt1;
	    if (it1 < 1) {
		it1 = 12;
	    }
	    if (it2 > 12) {
		it2 = 1;
	    }
	    m.f0075 = wt1 * m.f0073[it1 - 1] + wt2 *
		    m.f0073[it2 - 1];
	    m.f0076 = wt1 * m.f0074[it1 - 1] + wt2 *
		    m.f0074[it2 - 1];
	}
	if (m.f0034 == 7 || m.f0034 == 8 || m.f0034 == 9) {
/* Computing MAX */
	    r__1 = rc0004, r__2 = rc0005 * m.f0075;
	    m.f0076 = max(r__1,r__2);
	    if (m.f0075 < rc0004) {
		m.f0076 = rc0006;
	    }
	}
	if (m.f0076 < rc0004) {
	    m.f0076 = rc0006;
	}
	if (m.f0075 < rc0004 || m.f0076 == rc0006) {
	    m.f0075 = rc0006;
	}
	if (m.f0022 == m.f0114 || m.f0022 == m.f0115
		|| m.f0022 == m.f0116 || m.f0072) {
	    m.f0075 = rc0000;
	    m.f0076 = rc0000;
	}
    }
/* Computing MIN */
/* Computing MAX */
    r__3 = m.f0375 - m.f0080;
    r__1 = max(r__3,rc0000), r__2 = m.f0079 - m.f0080;
    db = min(r__1,r__2);
/* Computing MAX */
    r__1 = rc0007, r__2 = m.f0079 - m.f0080;
    fb = db / max(r__1,r__2);
    if (m.f0079 > rc0000 && m.f0079 <= rc0008) {
	snowhc = m.f0079 * (real)exp((real)(-m.f0375 / rc0009));
	fb = min(m.f0375,snowhc) / snowhc;
    }
    m.f0104 = m.f0075 * (rc0003 - fb);
    m.f0105 = m.f0076 * (rc0003 - fb);
    if (m.f0105 < rc0004 && m.f0023 == 0) {
	m.f0105 = rc0006;
    }
    if ((m.f0104 < rc0004 || m.f0105 == rc0006) && m.f0023
	    == 0) {
	m.f0104 = rc0006;
    }
    if (m.f0209 > m.f0081 && m.f0023 == 0) {
	m.f0213 = rc0003;
    } else {
	m.f0213 = rc0000;
    }
    if (m.f0034 == 1 || m.f0034 == 6 || m.f0034 == 7) {
	m.f0108 = m.f0082;
	if (m.f0108 <= rc0004) {
	    m.f0108 = rc0004;
	}
    } else if (m.f0034 == 2 || m.f0034 == 3 || m.f0034 == 8)
	     {
	m.f0108 = rc0003 - (real)exp((real)(-rc0010 * (m.f0075 +
		m.f0076)));
	if (m.f0108 <= rc0004) {
	    m.f0108 = rc0004;
	}
    } else if (m.f0034 == 4 || m.f0034 == 5 || m.f0034 == 9)
	     {
	m.f0108 = m.f0083;
	if (m.f0108 <= rc0004) {
	    m.f0108 = rc0004;
	}
    }
    if (m.f0023 > 0) {
	m.f0108 = m.f0083;
	if (m.f0108 <= rc0004) {
	    m.f0108 = rc0004;
	}
    }
    if (m.f0072 || m.f0022 == m.f0115) {
	m.f0108 = rc0006;
    }
    if (m.f0104 + m.f0105 == rc0006) {
	m.f0108 = rc0006;
    }
    return 0;
} /* r0011_ */

__device__ __forceinline__ /* Subroutine */ int r0012_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 0.f;
    real rc0002 = 1.f;
    real rc0003 = 6.6f;
    real rc0004 = .27f;
    real rc0005 = 46.f;
    real rc0006 = 270.15f;
    real rc0007 = 1.87e5f;
    real rc0008 = 1.56e5f;
    real rc0009 = 1.f;
    real rc0010 = 1e-6f;
    real rc0011 = .667f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real ft, fv, maxsno, maxliq, icedrip;

    m.f0362 = rc0000;
    m.f0363 = rc0000;
    m.f0364 = rc0000;
    m.f0362 = rc0000;
    m.f0365 = rc0000;
    m.f0366 = rc0000;
    m.f0367 = rc0000;
    m.f0368 = rc0001;
    m.f0369 = rc0001;
    m.f0370 = rc0001;
    icedrip = rc0001;
    maxliq = m.f0077 * (m.f0104 + m.f0105);
    if (m.f0104 + m.f0105 > rc0000) {
	m.f0362 = m.f0108 * m.f0354 * m.f0357;
/* Computing MIN */
	r__1 = m.f0362, r__2 = (maxliq - m.f0358) / m.f0005
		* (rc0002 - (real)exp((real)(-m.f0354 * m.f0005 /
		maxliq)));
	m.f0362 = min(r__1,r__2);
	m.f0362 = max(m.f0362,rc0000);
	m.f0363 = m.f0108 * m.f0354 - m.f0362;
	m.f0364 = (rc0002 - m.f0108) * m.f0354;
/* Computing MAX */
	r__1 = rc0000, r__2 = m.f0358 + m.f0362 * m.f0005;
	m.f0358 = max(r__1,r__2);
    } else {
	m.f0362 = rc0000;
	m.f0363 = rc0000;
	m.f0364 = m.f0354;
	if (m.f0358 > rc0000) {
	    m.f0363 += m.f0358 / m.f0005;
	    m.f0358 = rc0001;
	}
    }
    maxsno = rc0003 * (rc0004 + rc0005 / m.f0356) * (m.f0104 +
	    m.f0105);
    if (m.f0104 + m.f0105 > 0.f) {
	m.f0365 = m.f0108 * m.f0355 * m.f0357;
/* Computing MIN */
	r__1 = m.f0365, r__2 = (maxsno - m.f0359) / m.f0005
		* (rc0002 - (real)exp((real)(-m.f0355 * m.f0005 /
		maxsno)));
	m.f0365 = min(r__1,r__2);
	m.f0365 = max(m.f0365,rc0000);
/* Computing MAX */
	r__1 = rc0001, r__2 = (m.f0209 - rc0006) / rc0007;
	ft = max(r__1,r__2);
	fv = (real)sqrt((real)(m.f0195 * m.f0195 + m.f0196 *
		 m.f0196)) / rc0008;
	icedrip = max(rc0000,m.f0359) * (fv + ft);
	m.f0366 = m.f0108 * m.f0355 - m.f0365 +
		icedrip;
	m.f0367 = (rc0009 - m.f0108) * m.f0355;
/* Computing MAX */
	r__1 = rc0000, r__2 = m.f0359 + (m.f0365 - icedrip) *
		m.f0005;
	m.f0359 = max(r__1,r__2);
    } else {
	m.f0365 = rc0000;
	m.f0366 = rc0000;
	m.f0367 = m.f0355;
	if (m.f0359 > rc0000) {
	    m.f0366 += m.f0359 / m.f0005;
	    m.f0359 = rc0001;
	}
    }
    if (m.f0359 > 0.f) {
	m.f0360 = max(rc0000,m.f0359) / max(maxsno,rc0010);
    } else {
	m.f0360 = max(rc0000,m.f0358) / max(maxliq,rc0010);
    }
    d__1 = (doublereal) min(m.f0360,rc0002);
    d__2 = (doublereal) rc0011;
    m.f0360 = (real)pow_dd(&d__1, &d__2);
    m.f0361 = m.f0358 + m.f0359;
    m.f0368 = m.f0363 + m.f0364;
    m.f0369 = m.f0366 + m.f0367;
    m.f0370 = m.f0369 / m.f0356;
    if (m.f0025 == 2 && m.f0210 > m.f0125) {
	m.f0369 = rc0000;
	m.f0370 = rc0000;
    }
    return 0;
} /* r0012_ */

__device__ __forceinline__ /* Subroutine */ int r0013_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 0.f;
    real rc0002 = 1e-6f;
    real rc0003 = 6.6f;
    real rc0004 = .27f;
    real rc0005 = 46.f;
    real rc0006 = 1.f;
    real rc0007 = .667f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real maxsno, maxliq, qevac, qdewc, qfroc, qsubc, qmeltc, qfrzc;

    m.f0371 = rc0000;
    maxliq = m.f0077 * (m.f0104 + m.f0105);
    if (! m.f0214) {
/* Computing MAX */
	r__1 = m.f0212 / m.f0127;
	m.f0372 = max(r__1,rc0001);
/* Computing MAX */
	r__1 = m.f0211 / m.f0127;
	qevac = max(r__1,rc0001);
/* Computing MIN */
	r__2 = m.f0211 / m.f0127;
	qdewc = (r__1 = min(r__2,rc0001), abs(r__1));
	qsubc = rc0001;
	qfroc = rc0001;
    } else {
/* Computing MAX */
	r__1 = m.f0212 / m.f0126;
	m.f0372 = max(r__1,rc0001);
	qevac = rc0001;
	qdewc = rc0001;
/* Computing MAX */
	r__1 = m.f0211 / m.f0126;
	qsubc = max(r__1,rc0001);
/* Computing MIN */
	r__2 = m.f0211 / m.f0126;
	qfroc = (r__1 = min(r__2,rc0001), abs(r__1));
    }
/* Computing MIN */
    r__1 = m.f0358 / m.f0005;
    qevac = min(r__1,qevac);
/* Computing MAX */
    r__1 = rc0001, r__2 = m.f0358 + (qdewc - qevac) * m.f0005;
    m.f0358 = max(r__1,r__2);
    if (m.f0358 <= rc0002) {
	m.f0358 = rc0000;
    }
    maxsno = rc0003 * (rc0004 + rc0005 / m.f0356) * (m.f0104 +
	    m.f0105);
/* Computing MIN */
    r__1 = m.f0359 / m.f0005;
    qsubc = min(r__1,qsubc);
/* Computing MAX */
    r__1 = rc0001, r__2 = m.f0359 + (qfroc - qsubc) * m.f0005;
    m.f0359 = max(r__1,r__2);
    if (m.f0359 <= 1e-6f) {
	m.f0359 = rc0001;
    }
    if (m.f0359 > 0.f) {
	m.f0360 = max(rc0001,m.f0359) / max(maxsno,rc0002);
    } else {
	m.f0360 = max(rc0001,m.f0358) / max(maxliq,rc0002);
    }
    d__1 = (doublereal) min(m.f0360,rc0006);
    d__2 = (doublereal) rc0007;
    m.f0360 = (real)pow_dd(&d__1, &d__2);
    qmeltc = rc0001;
    qfrzc = rc0001;
    if (m.f0359 > 1e-6f && m.f0209 > m.f0125) {
/* Computing MIN */
	r__1 = m.f0359 / m.f0005, r__2 = (m.f0209 -
		m.f0125) * m.f0130 * m.f0359 /
		m.f0138 / (m.f0005 * m.f0128);
	qmeltc = min(r__1,r__2);
/* Computing MAX */
	r__1 = rc0001, r__2 = m.f0359 - qmeltc * m.f0005;
	m.f0359 = max(r__1,r__2);
/* Computing MAX */
	r__1 = rc0001, r__2 = m.f0358 + qmeltc * m.f0005;
	m.f0358 = max(r__1,r__2);
	m.f0209 = m.f0360 * m.f0125 + (rc0006 -
		m.f0360) * m.f0209;
    }
    if (m.f0358 > 1e-6f && m.f0209 < m.f0125) {
/* Computing MIN */
	r__1 = m.f0358 / m.f0005, r__2 = (m.f0125 -
		m.f0209) * m.f0129 * m.f0358 /
		m.f0137 / (m.f0005 * m.f0128);
	qfrzc = min(r__1,r__2);
/* Computing MAX */
	r__1 = rc0001, r__2 = m.f0358 - qfrzc * m.f0005;
	m.f0358 = max(r__1,r__2);
/* Computing MAX */
	r__1 = rc0001, r__2 = m.f0359 + qfrzc * m.f0005;
	m.f0359 = max(r__1,r__2);
	m.f0209 = m.f0360 * m.f0125 + (rc0006 -
		m.f0360) * m.f0209;
    }
    m.f0361 = m.f0358 + m.f0359;
    m.f0371 = qevac + qsubc - qdewc - qfroc;
    return 0;
} /* r0013_ */

__device__ __forceinline__ /* Subroutine */ int r0014_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 0.f;
    real rc0002 = 100.f;
    real rc0003 = 2.5f;
    real rc0004 = .01f;
    real rc0005 = 1.f;
    real rc0006 = .65f;
    real rc0007 = 1.f;
    real rc0008 = .98f;
    real rc0009 = 5.8f;
    real rc0010 = 2.71828f;
    real rc0011 = 2.2e-5f;
    real rc0012 = 2.f;
    real rc0013 = 3.f;
    real rc0014 = 8.25f;
    real rc0015 = 4.225f;
    real rc0016 = 6.f;
    real rc0017 = .001f;
    real rc0018 = 1e6f;
    real rc0019 = .622f;
    real rc0020 = .378f;
    real rc0021 = .25f;
    real rc0022 = .05f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3, r__4;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real gx;
    integer iz, ixarr;
    real fmelt, zpdg, psi, bevap, fire;

    real d_rsurf__;

    real l_rsurf__;


    m.f0303 = rc0000;
    m.f0304 = rc0000;
    m.f0281 = rc0000;
    m.f0283 = rc0000;
    m.f0282 = rc0000;
    m.f0284 = rc0000;
    m.f0286 = rc0000;
    m.f0280 = rc0000;
    m.f0288 = rc0000;
    m.f0291 = rc0000;
    m.f0332 = rc0000;
    m.f0333 = rc0000;
    m.f0292 = rc0000;
    m.f0297 = rc0000;
    m.f0314 = rc0000;
    m.f0293 = rc0000;
    m.f0294 = rc0000;
    m.f0295 = rc0000;
    m.f0317 = rc0000;
    m.f0106 = m.f0104 + m.f0105;
    if (m.f0106 > rc0001) {
	m.f0107 = TRUE_;
    } else {
	m.f0107 = FALSE_;
    }
    m.f0406 = rc0001;
    if (m.f0375 > rc0001) {
	m.f0378 = m.f0376 / m.f0375;
	d__1 = (doublereal) (m.f0378 / rc0002);
	d__2 = (doublereal) m.f0143;
	fmelt = (real)pow_dd(&d__1, &d__2);
	m.f0406 = m.f0179 * (real)tanh((real)(m.f0375 / (
		rc0003 * m.f0064 * fmelt)));
    }
    if (m.f0025 == 2) {
	if (m.f0210 <= m.f0125) {
	    m.f0267 = rc0004 * (rc0005 - m.f0406) + m.f0406
		    * m.f0144;
	} else {
	    m.f0267 = rc0004;
	}
    } else {
	m.f0267 = m.f0064 * (rc0005 - m.f0406) +
		m.f0406 * m.f0144;
    }
    zpdg = m.f0375;
    if (m.f0107) {
	m.f0268 = m.f0084;
	m.f0266 = rc0006 * m.f0079;
	if (m.f0375 > m.f0266) {
	    m.f0266 = m.f0375;
	}
    } else {
	m.f0268 = m.f0267;
	m.f0266 = zpdg;
    }
    if (m.f0072) {
	m.f0267 = m.f0084;
	zpdg = rc0006 * m.f0079;
	m.f0268 = m.f0267;
	m.f0266 = zpdg;
    }
    m.f0269 = max(m.f0266,m.f0079) + m.f0019;
    if (zpdg >= m.f0269) {
	m.f0269 = zpdg + m.f0019;
    }
    r0015_(m);
    r0018_(m);
    r0019_(m);
    m.f0276 = rc0007 - (real)exp((real)(-(m.f0104 + m.f0105)
	     / rc0005));
    if (m.f0338 == 1) {
	m.f0277 = rc0008 * (rc0007 - m.f0406) + rc0005 *
		m.f0406;
    } else {
	m.f0277 = m.f0166[m.f0025 - 1] * (rc0007 -
		m.f0406) + rc0005 * m.f0406;
    }
    m.f0407 = rc0000;
    if (m.f0025 == 1) {
	i__1 = m.f0078;
	for (iz = 1; iz <= i__1; ++iz) {
	    if (m.f0043 == 1) {
		gx = (m.f0394[iz - 1] - m.f0048[iz - 1]) / (
			m.f0049[iz - 1] - m.f0048[iz - 1]);
	    }
	    if (m.f0043 == 2) {
		if (m.f0045 == 1) {
/* Computing MAX */
/* Computing MAX */
		    r__3 = rc0004, r__4 = m.f0394[iz - 1];
		    d__1 = (doublereal) (max(r__3,r__4) / m.f0047[iz -
			    1]);
		    d__2 = (doublereal) (-m.f0046[iz - 1]);
		    r__1 = m.f0175, r__2 = -m.f0052[iz - 1] * (
			    real)pow_dd(&d__1, &d__2);
		    psi = max(r__1,r__2);
		}
		if (m.f0045 == 2) {
		    psi = m.f0345 - (m.f0026[iz - 1] +
			    m.f0027[iz + 2] / 2);
		}
		gx = (rc0007 - psi / m.f0175) / (rc0007 + m.f0052[
			iz - 1] / m.f0175);
	    }
	    if (m.f0043 == 3) {
		if (m.f0045 == 1) {
/* Computing MAX */
/* Computing MAX */
		    r__3 = rc0004, r__4 = m.f0394[iz - 1];
		    d__1 = (doublereal) (max(r__3,r__4) / m.f0047[iz -
			    1]);
		    d__2 = (doublereal) (-m.f0046[iz - 1]);
		    r__1 = m.f0175, r__2 = -m.f0052[iz - 1] * (
			    real)pow_dd(&d__1, &d__2);
		    psi = max(r__1,r__2);
		}
		if (m.f0045 == 2) {
		    psi = m.f0345 - (m.f0026[iz - 1] +
			    m.f0027[iz + 2] / 2);
		}
		gx = rc0007 - (real)exp((real)(-rc0009 * (real)log((real)(
			m.f0175 / psi))));
	    }
	    if (m.f0043 == 4) {
		gx = rc0005;
	    }
/* Computing MIN */
	    r__1 = rc0007, r__2 = max(rc0000,gx);
	    gx = min(r__1,r__2);
/* Computing MAX */
	    r__1 = m.f0171, r__2 = m.f0027[iz + 2] / (
		    -m.f0026[m.f0078 - 1]) * gx;
	    m.f0396[iz - 1] = max(r__1,r__2);
	    m.f0407 += m.f0396[iz - 1];
/* L9004: */
	}
/* L9005: */
	m.f0407 = max(m.f0171,m.f0407);
	i__1 = m.f0078;
	for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	    m.f0396[ixarr - 1] /= m.f0407;
	}
    }
/* Computing MAX */
    r__1 = rc0001, r__2 = m.f0394[0] / m.f0047[0];
    bevap = max(r__1,r__2);
    if (m.f0025 == 2) {
	m.f0301 = rc0005;
	m.f0302 = rc0005;
    } else {
	if (m.f0044 == 1 || m.f0044 == 4 || m.f0044 == 5) {
/* Computing MIN */
	    r__1 = rc0005, r__2 = m.f0394[0] / m.f0047[0];
	    d__1 = (doublereal) (rc0005 - min(r__1,r__2));
	    d__2 = (doublereal) m.f0158;
	    l_rsurf__ = -m.f0026[0] * ((real)exp((real)((real)pow_dd(&
		    d__1, &d__2))) - rc0005) / (rc0010 - rc0005);
	    d__1 = (doublereal) (rc0005 - m.f0048[0] / m.f0047[0])
		    ;
	    d__2 = (doublereal) (rc0012 + rc0013 / m.f0046[0]);
	    d_rsurf__ = rc0011 * m.f0047[0] * m.f0047[0] * (real)
		    pow_dd(&d__1, &d__2);
	    m.f0301 = l_rsurf__ / d_rsurf__;
	} else if (m.f0044 == 2) {
	    m.f0301 = m.f0406 * rc0007 + (rc0007 - m.f0406)
		    * (real)exp((real)(rc0014 - rc0015 * bevap));
	} else if (m.f0044 == 3) {
	    m.f0301 = m.f0406 * rc0007 + (rc0007 - m.f0406)
		    * (real)exp((real)(rc0014 - rc0016 * bevap));
	}
	if (m.f0044 == 4) {
	    m.f0301 = rc0007 / (m.f0406 * (rc0007 / m.f0157)
		     + (rc0007 - m.f0406) * (rc0007 / max(m.f0301,
		    rc0017)));
	}
	if (m.f0044 == 5) {
	    m.f0301 = rc0007 / (m.f0406 * (rc0007 / m.f0157)
		     + (rc0007 - m.f0406) * (rc0007 / rc0017));
	}
	if (m.f0394[0] < rc0004 && m.f0375 == rc0000) {
	    m.f0301 = rc0018;
	}
	d__1 = (doublereal) (max(rc0004,m.f0394[0]) / m.f0047[0]);
	d__2 = (doublereal) (-m.f0046[0]);
	psi = -m.f0052[0] * (real)pow_dd(&d__1, &d__2);
	m.f0302 = m.f0406 + (rc0007 - m.f0406) * (real)exp((
		real)(psi * m.f0177 / (m.f0136 * m.f0210)));
    }
    if (m.f0072 && m.f0375 == rc0000) {
	m.f0301 = rc0018;
    }
    if (m.f0209 > m.f0125) {
	m.f0298 = m.f0127;
	m.f0214 = FALSE_;
    } else {
	m.f0298 = m.f0126;
	m.f0214 = TRUE_;
    }
    m.f0278 = m.f0131 * m.f0180 / (rc0019 * m.f0298);
    if (m.f0210 > m.f0125) {
	m.f0299 = m.f0127;
	m.f0215 = FALSE_;
    } else {
	m.f0299 = m.f0126;
	m.f0215 = TRUE_;
    }
    m.f0279 = m.f0131 * m.f0180 / (rc0019 * m.f0299);
    if (m.f0107 && m.f0108 > 0.f) {
	m.f0313 = m.f0210;
	m.f0270 = m.f0272;
	m.f0314 = m.f0273;
	r0027_(m);
    }
    m.f0274 = m.f0210;
    m.f0271 = m.f0272;
    m.f0326 = m.f0273;
    r0028_(m);
    if (m.f0107 && m.f0108 > 0.f) {
	m.f0307 = m.f0108 * m.f0303 + (rc0005 -
		m.f0108) * m.f0305;
	m.f0308 = m.f0108 * m.f0304 + (rc0005 -
		m.f0108) * m.f0306;
	m.f0318 = m.f0108 * m.f0282 + (rc0005 -
		m.f0108) * m.f0322 + m.f0281;
	m.f0319 = m.f0108 * m.f0284 + (rc0005 -
		m.f0108) * m.f0285 + m.f0283;
	m.f0320 = m.f0108 * m.f0286 + (rc0005 -
		m.f0108) * m.f0287;
	m.f0323 = m.f0108 * m.f0291 + (rc0005 -
		m.f0108) * m.f0290;
	m.f0289 = m.f0108 * m.f0291 + (rc0005 -
		m.f0108) * m.f0290;
	m.f0211 = m.f0280;
	m.f0212 = m.f0288;
	m.f0225 = m.f0108 * m.f0223 + (rc0005 -
		m.f0108) * m.f0224 + m.f0222;
	m.f0210 = m.f0108 * m.f0313 + (rc0005 -
		m.f0108) * m.f0274;
	m.f0324 = m.f0108 * m.f0292 + (rc0005 -
		m.f0108) * m.f0311;
	m.f0325 = m.f0108 * m.f0209 + (rc0005 -
		m.f0108) * m.f0274;
	m.f0272 = m.f0108 * m.f0270 + (rc0005 -
		m.f0108) * m.f0271;
	m.f0273 = m.f0108 * m.f0314 + (rc0005 -
		m.f0108) * m.f0326;
	m.f0327 = m.f0108 * (m.f0265 * rc0019 / (
		m.f0180 - rc0020 * m.f0265)) + (rc0005 -
		m.f0108) * m.f0275;
	m.f0328 = m.f0108 * m.f0297 + (rc0005 -
		m.f0108) * m.f0312;
	m.f0329 = m.f0268;
    } else {
	m.f0307 = m.f0305;
	m.f0308 = m.f0306;
	m.f0318 = m.f0322;
	m.f0319 = m.f0285;
	m.f0320 = m.f0287;
	m.f0323 = m.f0290;
	m.f0210 = m.f0274;
	m.f0324 = m.f0311;
	m.f0211 = rc0000;
	m.f0212 = rc0000;
	m.f0225 = m.f0224;
	m.f0325 = m.f0210;
	m.f0272 = m.f0271;
	m.f0273 = m.f0326;
	m.f0327 = m.f0275;
	m.f0328 = m.f0312;
	m.f0315 = rc0001;
	m.f0316 = rc0001;
	m.f0313 = m.f0274;
	m.f0314 = m.f0326;
	m.f0329 = m.f0267;
    }
    fire = m.f0191 + m.f0318;
    if (fire <= 0.f) {
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 2;
	return 0;
    }
    m.f0330 = m.f0108 * (m.f0277 * (1 - m.f0276) +
	    m.f0276 + m.f0276 * (1 - m.f0276) * (1 -
	    m.f0277)) + (1 - m.f0108) * m.f0277;
    d__1 = (doublereal) ((fire - (1 - m.f0330) * m.f0191) / (
	    m.f0330 * m.f0123));
    d__2 = (doublereal) rc0021;
    m.f0321 = (real)pow_dd(&d__1, &d__2);
    m.f0334 = m.f0256 * m.f0254 + m.f0257 *
	    m.f0255;
    m.f0331 = m.f0332 * m.f0254 + m.f0333 *
	    m.f0255;
    r0047_(m, &m.f0338, &m.f0390, &m.f0323, m.f0218,
	    m.f0219, &m.f0259, &m.f0375, &m.f0210,
	    m.f0217);
    if (m.f0040 == 2) {
	if (m.f0375 > rc0022 && m.f0210 > m.f0125) {
	    m.f0313 = m.f0125;
	    m.f0274 = m.f0125;
	    if (m.f0107 && m.f0108 > 0.f) {
		m.f0210 = m.f0108 * m.f0313 + (rc0005 -
			m.f0108) * m.f0274;
		m.f0325 = m.f0108 * m.f0209 + (rc0005 -
			m.f0108) * m.f0274;
	    } else {
		m.f0210 = m.f0274;
		m.f0325 = m.f0274;
	    }
	}
    }
    r0051_(m, &m.f0001, &m.f0000);
    m.f0336 = m.f0211 + m.f0320 + m.f0212;
    if (m.f0375 > rc0022) {
	m.f0337 = m.f0217[2];
    } else {
	m.f0337 = m.f0210;
    }
    return 0;
} /* r0014_ */

__device__ __forceinline__ /* Subroutine */ int r0015_(Model &m) {
    /* Initialized data */

    real rc0000 = 3.24f;
    real rc0001 = .35f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer iz;


    i__1 = m.f0000;
    for (iz = m.f0390 + 1; iz <= i__1; ++iz) {
	if (iz == m.f0390 + 1) {
	    m.f0027[iz + 2] = -m.f0028[iz + 2];
	} else {
	    m.f0027[iz + 2] = m.f0028[iz + 1] - m.f0028[iz
		    + 2];
	}
/* L9004: */
    }
/* L9005: */
    r0016_(m);
    r0017_(m);
    if (m.f0072) {
	i__1 = m.f0000;
	for (iz = 1; iz <= i__1; ++iz) {
	    m.f0218[iz + 2] = rc0000;
/* L9007: */
	}
/* L9008: */
    }
    if (m.f0025 == 2) {
	i__1 = m.f0000;
	for (iz = 1; iz <= i__1; ++iz) {
	    if (m.f0217[iz + 2] > m.f0125) {
		m.f0219[iz + 2] = m.f0129;
		m.f0218[iz + 2] = m.f0132;
	    } else {
		m.f0219[iz + 2] = m.f0130;
		m.f0218[iz + 2] = m.f0133;
	    }
/* L9010: */
	}
/* L9011: */
    }
    i__1 = m.f0000;
    for (iz = m.f0390 + 1; iz <= i__1; ++iz) {
	m.f0220[iz + 2] = m.f0005 / (m.f0219[iz + 2] *
		m.f0027[iz + 2]);
/* L9013: */
    }
/* L9014: */
    if (m.f0390 == 0) {
	m.f0218[3] = (m.f0218[3] * m.f0027[3] + rc0001 *
		m.f0375) / (m.f0375 + m.f0027[3]);
    } else {
	m.f0218[3] = (m.f0218[3] * m.f0027[3] +
		m.f0218[2] * m.f0027[2]) / (m.f0027[2] +
		m.f0027[3]);
    }
    return 0;
} /* r0015_ */

__device__ __forceinline__ /* Subroutine */ int r0016_(Model &m) {
    /* Initialized data */

    real rc0000 = 1.f;
    real rc0001 = 3.2217e-6f;
    real rc0002 = 2.f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    integer iz;
    real bdsnoi[3], cvsno[3], tksno[3];

    for (iz = m.f0390 + 1; iz <= 0; ++iz) {
/* Computing MIN */
	r__1 = rc0000, r__2 = m.f0400[iz + 2] / (m.f0027[iz + 2] *
		 m.f0138);
	m.f0402[iz + 2] = min(r__1,r__2);
	m.f0405[iz + 2] = rc0000 - m.f0402[iz + 2];
/* Computing MIN */
	r__1 = m.f0405[iz + 2], r__2 = m.f0401[iz + 2] / (
		m.f0027[iz + 2] * m.f0137);
	m.f0403[iz + 2] = min(r__1,r__2);
	bdsnoi[iz + 2] = (m.f0400[iz + 2] + m.f0401[iz + 2]) /
		m.f0027[iz + 2];
	cvsno[iz + 2] = m.f0130 * m.f0402[iz + 2] + m.f0129
		* m.f0403[iz + 2];
	d__1 = (doublereal) bdsnoi[iz + 2];
	d__2 = (doublereal) rc0002;
	tksno[iz + 2] = rc0001 * (real)pow_dd(&d__1, &d__2);
	m.f0218[iz + 2] = tksno[iz + 2];
	m.f0219[iz + 2] = cvsno[iz + 2];
/* L9004: */
    }
/* L9005: */
    return 0;
} /* r0016_ */

__device__ __forceinline__ /* Subroutine */ int r0017_(Model &m) {
    /* Initialized data */

    real rc0000 = 1.f;
    real rc0001 = 1.f;
    real rc0002 = 0.f;
    real rc0003 = 2700.f;
    real rc0004 = .135f;
    real rc0005 = 64.7f;
    real rc0006 = .947f;
    real rc0007 = 5e-4f;
    real rc0008 = .1f;

    /* System generated locals */
    integer i__1;
    real d__1, d__2, d__3, d__4, d__5, d__6;

    /* Builtin functions */


    /* Local variables */
    real satratio;
    integer iz;
    real xu, ake, gammd, thkdry, thksat, thks, xunfroz;

    i__1 = m.f0000;
    for (iz = 1; iz <= i__1; ++iz) {
	m.f0393[iz - 1] = m.f0391[iz - 1] - m.f0394[iz - 1];
	m.f0219[iz + 2] = m.f0394[iz - 1] * m.f0129 + (
		rc0000 - m.f0047[iz - 1]) * m.f0063 + (
		m.f0047[iz - 1] - m.f0391[iz - 1]) *
		m.f0131 + m.f0393[iz - 1] * m.f0130;
	satratio = m.f0391[iz - 1] / m.f0047[iz - 1];
	d__1 = (doublereal) m.f0141;
	d__2 = (doublereal) m.f0059;
	d__3 = (doublereal) m.f0140;
	d__4 = (doublereal) (rc0001 - m.f0059);
	thks = (real)pow_dd(&d__1, &d__2) * (real)pow_dd(&d__3, &d__4);
	xunfroz = rc0000;
	if (m.f0391[iz - 1] > rc0002) {
	    xunfroz = m.f0394[iz - 1] / m.f0391[iz - 1];
	}
	xu = xunfroz * m.f0047[iz - 1];
	d__1 = (doublereal) thks;
	d__2 = (doublereal) (rc0001 - m.f0047[iz - 1]);
	d__3 = (doublereal) m.f0133;
	d__4 = (doublereal) (m.f0047[iz - 1] - xu);
	d__5 = (doublereal) m.f0139;
	d__6 = (doublereal) xu;
	thksat = (real)pow_dd(&d__1, &d__2) * (real)pow_dd(&d__3, &d__4) * (
		real)pow_dd(&d__5, &d__6);
	gammd = (rc0001 - m.f0047[iz - 1]) * rc0003;
	thkdry = (rc0004 * gammd + rc0005) / (rc0003 - rc0006 * gammd);
	if (m.f0394[iz - 1] + rc0007 < m.f0391[iz - 1]) {
	    ake = satratio;
	} else {
	    if (satratio > rc0008) {
		ake = (real)r_lg10(&satratio) + rc0000;
	    } else {
		ake = rc0002;
	    }
	}
	m.f0218[iz + 2] = ake * (thksat - thkdry) + thkdry;
/* L9004: */
    }
/* L9005: */
    return 0;
} /* r0017_ */

__device__ __forceinline__ /* Subroutine */ int r0018_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1e3f;
    real rc0002 = 0.f;
    real rc0003 = 1.f;
    real rc0004 = 20.f;

    /* System generated locals */
    real r__1, r__2;

    /* Local variables */
    real pah_ac__, pah_cg__, pah_ag__;

    pah_ac__ = rc0000;
    pah_cg__ = rc0000;
    pah_ag__ = rc0000;
    m.f0222 = rc0000;
    m.f0223 = rc0000;
    m.f0224 = rc0000;
    m.f0225 = rc0000;
    pah_ac__ = m.f0108 * m.f0354 * (m.f0129 / rc0001) * (
	    m.f0181 - m.f0209);
    pah_cg__ = m.f0363 * (m.f0129 / rc0001) * (m.f0209 -
	    m.f0210);
    pah_ag__ = m.f0364 * (m.f0129 / rc0001) * (m.f0181 -
	    m.f0210);
    pah_ac__ += m.f0108 * m.f0355 * (m.f0130 / rc0001) * (
	    m.f0181 - m.f0209);
    pah_cg__ += m.f0366 * (m.f0130 / rc0001) * (m.f0209 -
	    m.f0210);
    pah_ag__ += m.f0367 * (m.f0130 / rc0001) * (m.f0181 -
	    m.f0210);
    m.f0222 = pah_ac__ - pah_cg__;
    m.f0223 = pah_cg__;
    m.f0224 = pah_ag__;
    if (m.f0108 > rc0002 && m.f0108 < rc0003) {
	m.f0223 /= m.f0108;
	m.f0224 /= rc0003 - m.f0108;
    } else if (m.f0108 <= rc0002) {
	m.f0224 = m.f0223 + m.f0224;
	m.f0223 = rc0002;
	m.f0222 = rc0002;
    } else if (m.f0108 >= rc0003) {
	m.f0224 = rc0002;
    }
/* Computing MAX */
    r__1 = m.f0222, r__2 = -rc0004;
    m.f0222 = max(r__1,r__2);
    m.f0222 = min(m.f0222,rc0004);
/* Computing MAX */
    r__1 = m.f0223, r__2 = -rc0004;
    m.f0223 = max(r__1,r__2);
    m.f0223 = min(m.f0223,rc0004);
/* Computing MAX */
    r__1 = m.f0224, r__2 = -rc0004;
    m.f0224 = max(r__1,r__2);
    m.f0224 = min(m.f0224,rc0004);
    return 0;
} /* r0018_ */

__device__ __forceinline__ /* Subroutine */ int r0019_(Model &m) {
    /* Initialized data */

    real rc0000 = 1.f;



    r0021_(m);
    m.f0253 = rc0000 - m.f0252;
    m.f0254 = m.f0104 * m.f0252;
    m.f0255 = m.f0104 * m.f0253;
    r0020_(m);
    return 0;
} /* r0019_ */

__device__ __forceinline__ /* Subroutine */ int r0020_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;
    real rc0002 = 0.f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer ib;
    real abs__, rnir, rvis, laifra, trd, tri, cad[2], cai[2];

    m.f0259 = rc0000;
    m.f0258 = rc0000;
    m.f0260 = rc0000;
    i__1 = m.f0170;
    for (ib = 1; ib <= i__1; ++ib) {
	cad[ib - 1] = m.f0207[ib - 1] * m.f0236[ib - 1];
	cai[ib - 1] = m.f0208[ib - 1] * m.f0237[ib - 1];
	m.f0258 = m.f0258 + cad[ib - 1] + cai[ib - 1];
	m.f0260 = m.f0260 + cad[ib - 1] + cai[ib - 1];
	trd = m.f0207[ib - 1] * m.f0238[ib - 1];
	tri = m.f0207[ib - 1] * m.f0240[ib - 1] + m.f0208[
		ib - 1] * m.f0241[ib - 1];
	abs__ = trd * (rc0001 - m.f0232[ib - 1]) + tri * (rc0001 -
		m.f0233[ib - 1]);
	m.f0259 += abs__;
	m.f0260 += abs__;
/* L9004: */
    }
/* L9005: */
    laifra = m.f0104 / max(m.f0106,m.f0171);
    if (m.f0252 > rc0002) {
	m.f0256 = (cad[0] + m.f0252 * cai[0]) * laifra / max(
		m.f0254,m.f0171);
	m.f0257 = m.f0253 * cai[0] * laifra / max(m.f0255,
		m.f0171);
    } else {
	m.f0256 = rc0000;
	m.f0257 = (cad[0] + cai[0]) * laifra / max(m.f0255,
		m.f0171);
    }
    rvis = m.f0230[0] * m.f0207[0] + m.f0231[0] *
	    m.f0208[0];
    rnir = m.f0230[1] * m.f0207[1] + m.f0231[1] *
	    m.f0208[1];
    m.f0261 = rvis + rnir;
    m.f0262 = m.f0242[0] * m.f0207[0] + m.f0243[0] *
	    m.f0208[0] + m.f0242[1] * m.f0207[1] +
	    m.f0243[1] * m.f0208[1];
    m.f0263 = m.f0244[0] * m.f0207[0] + m.f0245[0] *
	    m.f0208[0] + m.f0244[1] * m.f0207[1] +
	    m.f0245[1] * m.f0208[1];
    return 0;
} /* r0020_ */

__device__ __forceinline__ /* Subroutine */ int r0021_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;
    real rc0002 = .01f;
    real rc0003 = 0.f;

    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */


    /* Local variables */
    integer ib, ic;
    real wl, ws, gdir, ext;


    m.f0250 = rc0000;
    m.f0251 = rc0000;
    i__1 = m.f0170;
    for (ib = 1; ib <= i__1; ++ib) {
	m.f0230[ib - 1] = rc0000;
	m.f0231[ib - 1] = rc0000;
	m.f0232[ib - 1] = rc0000;
	m.f0233[ib - 1] = rc0000;
	m.f0234[ib - 1] = rc0000;
	m.f0235[ib - 1] = rc0000;
	m.f0236[ib - 1] = rc0000;
	m.f0237[ib - 1] = rc0000;
	m.f0238[ib - 1] = rc0000;
	m.f0240[ib - 1] = rc0000;
	m.f0241[ib - 1] = rc0000;
	if (ib == 1) {
	    m.f0252 = rc0000;
	}
/* L9004: */
    }
/* L9005: */
    if (m.f0248 <= 0.f) {
	goto L100;
    }
    i__1 = m.f0170;
    for (ib = 1; ib <= i__1; ++ib) {
	wl = m.f0104 / max(m.f0106,m.f0171);
	ws = m.f0105 / max(m.f0106,m.f0171);
/* Computing MAX */
	r__1 = m.f0109[ib - 1] * wl + m.f0110[ib - 1] * ws;
	m.f0246[ib - 1] = max(r__1,m.f0171);
/* Computing MAX */
	r__1 = m.f0111[ib - 1] * wl + m.f0112[ib - 1] * ws;
	m.f0247[ib - 1] = max(r__1,m.f0171);
/* L9007: */
    }
/* L9008: */
    r0022_(m);
    if (m.f0035 == 1) {
	r0023_(m);
    }
    if (m.f0035 == 2) {
	r0024_(m);
	m.f0229 = m.f0228;
    }
    r0025_(m);
    i__1 = m.f0170;
    for (ib = 1; ib <= i__1; ++ib) {
	ic = 0;
	r0026_(m, &gdir, &ext, &ib, &ic);
	ic = 1;
	r0026_(m, &gdir, &ext, &ib, &ic);
/* L9010: */
    }
/* L9011: */
    ext = gdir / m.f0248 * (real)sqrt((real)(rc0001 - m.f0246[0]
	    - m.f0247[0]));
/* Computing MAX */
    r__1 = ext * m.f0106;
    m.f0252 = (rc0001 - (real)exp((real)(-ext * m.f0106))) / max(
	    r__1,m.f0171);
    ext = m.f0252;
    if (ext < rc0002) {
	wl = rc0003;
    } else {
	wl = ext;
    }
    m.f0252 = wl;
L100:
    return 0;
} /* r0021_ */

__device__ __forceinline__ /* Subroutine */ int r0022_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;
    real rc0002 = 0.f;
    real rc0003 = 1.f;

    /* System generated locals */
    real r__1, r__2;

    /* Builtin functions */


    /* Local variables */
    real tage, age1, age2, age3, dela, sge, dels, dela0, arg;

    if (m.f0376 <= rc0000) {
	m.f0226 = rc0000;
    } else {
	dela0 = m.f0005 / m.f0146;
	arg = m.f0147 * (rc0001 / m.f0125 - rc0001 /
		m.f0210);
	age1 = (real)exp(arg);
/* Computing MIN */
	r__1 = rc0002, r__2 = m.f0148 * arg;
	age2 = (real)exp((real)(min(r__1,r__2)));
	age3 = m.f0149;
	tage = age1 + age2 + age3;
	dela = dela0 * tage;
/* Computing MAX */
	r__1 = rc0000, r__2 = m.f0376 - m.f0377;
	dels = max(r__1,r__2) / m.f0145;
	sge = (m.f0226 + dela) * (rc0003 - dels);
	m.f0226 = max(rc0002,sge);
    }
    m.f0227 = m.f0226 / (m.f0226 + rc0003);
    return 0;
} /* r0022_ */

__device__ __forceinline__ /* Subroutine */ int r0023_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;
    real rc0002 = 2.f;
    real rc0003 = 0.f;
    real rc0004 = 1.f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    real cf1, sl1, sl2;
    integer ixarr;
    real fzen;

    i__1 = m.f0170 - 1 + 1;
    for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	m.f0234[ixarr - 1] = rc0000;
    }
    i__1 = m.f0170;
    for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	m.f0235[ixarr - 1] = rc0000;
    }
    sl1 = rc0001 / m.f0150;
    sl2 = rc0002 * m.f0150;
    cf1 = (rc0001 + sl1) / (rc0001 + sl2 * m.f0248) - sl1;
    fzen = max(cf1,rc0003);
    m.f0235[0] = m.f0151 * (rc0004 - m.f0153 *
	    m.f0227);
    m.f0235[1] = m.f0152 * (rc0004 - m.f0154 *
	    m.f0227);
    m.f0234[0] = m.f0235[0] + m.f0155 * fzen * (rc0004 -
	    m.f0235[0]);
    m.f0234[1] = m.f0235[1] + m.f0156 * fzen * (rc0004 -
	    m.f0235[1]);
    return 0;
} /* r0023_ */

__device__ __forceinline__ /* Subroutine */ int r0024_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = .55f;
    real rc0002 = .01f;
    real rc0003 = 3600.f;
    real rc0004 = .84f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */


    /* Local variables */
    integer ixarr;

    i__1 = m.f0170 - 1 + 1;
    for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	m.f0234[ixarr - 1] = rc0000;
    }
    i__1 = m.f0170;
    for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	m.f0235[ixarr - 1] = rc0000;
    }
    m.f0228 = rc0001 + (m.f0229 - rc0001) * (real)exp((real)(
	    -rc0002 * m.f0005 / rc0003));
    if (m.f0369 > rc0000) {
/* Computing MIN */
	r__1 = m.f0369, r__2 = m.f0145 / m.f0005;
	m.f0228 += min(r__1,r__2) * (rc0004 - m.f0228) / (
		m.f0145 / m.f0005);
    }
    m.f0235[0] = m.f0228;
    m.f0235[1] = m.f0228;
    m.f0234[0] = m.f0228;
    m.f0234[1] = m.f0228;
    return 0;
} /* r0024_ */

__device__ __forceinline__ /* Subroutine */ int r0025_(Model &m) {
    /* Initialized data */

    real rc0000 = .11f;
    real rc0001 = .4f;
    real rc0002 = 0.f;
    real rc0003 = .06f;
    real rc0004 = .01f;
    real rc0005 = 1.7f;
    real rc0006 = .15f;
    real rc0007 = 1.f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    integer ib;
    real inc, albsod, albsoi;

    i__1 = m.f0170;
    for (ib = 1; ib <= i__1; ++ib) {
/* Computing MAX */
	r__1 = rc0000 - rc0001 * m.f0391[0];
	inc = max(r__1,rc0002);
	if (m.f0025 == 1) {
/* Computing MIN */
	    r__1 = m.f0159[ib - 1] + inc, r__2 = m.f0160[ib - 1];
	    albsod = min(r__1,r__2);
	    albsoi = albsod;
	} else if (m.f0210 > m.f0125) {
	    d__1 = (doublereal) max(rc0004,m.f0248);
	    d__2 = (doublereal) rc0005;
	    albsod = rc0003 / ((real)pow_dd(&d__1, &d__2) + rc0006);
	    albsoi = rc0003;
	} else {
	    albsod = m.f0162[ib - 1];
	    albsoi = albsod;
	}
	m.f0232[ib - 1] = albsod * (rc0007 - m.f0406) +
		m.f0234[ib - 1] * m.f0406;
	m.f0233[ib - 1] = albsoi * (rc0007 - m.f0406) +
		m.f0235[ib - 1] * m.f0406;
/* L9004: */
    }
/* L9005: */
    return 0;
} /* r0025_ */

__device__ __forceinline__ /* Subroutine */ int r0026_(Model &m, real *gdir, real *ext, integer *ib, integer *ic) {
    /* Initialized data */

    real rc0000 = 3.14159265f;
    real rc0001 = 0.f;
    real rc0002 = 1.f;
    real rc0003 = .01f;
    real rc0004 = .5f;
    real rc0005 = 1.33f;
    real rc0006 = 3.f;
    real rc0007 = .05f;
    real rc0008 = .001f;
    real rc0009 = .4f;
    real rc0010 = .6f;
    real rc0011 = .633f;
    real rc0012 = .33f;
    real rc0013 = .877f;
    real rc0014 = 1.f;
    real rc0015 = 2.f;
    real rc0016 = 1e-6f;
    real rc0017 = 0.f;

    /* System generated locals */
    real r__1, r__2, r__3;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real b, c__, d__, f, h__, d1, d2, h1, h2, h3, h4, h5, h6, h7, h8, p1, p2,
	    p3, p4, s1, s2, u1, u2, u3, h9, bb, fa, h10, hd, vai_spread__,
	    omega, omegal, betai, betail, betad, betadl, avmu, coszi, asu,
	    chil, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9,
	    phi1, phi2, sigma, ftds, ftis, fres, denfveg, freveg, frebar, pai,
	     thetap, newvai, kopen, gap;

    pai = rc0000;
    vai_spread__ = m.f0106;
    if (m.f0106 == rc0001) {
	gap = rc0002;
	kopen = rc0002;
    } else {
	if (m.f0036 == 1) {
/* Computing MAX */
	    r__1 = rc0002 - m.f0108;
/* Computing 2nd power */
	    r__2 = m.f0085;
	    denfveg = -(real)log((real)(max(r__1,rc0003))) / (pai * (r__2 *
		    r__2));
	    hd = m.f0079 - m.f0080;
	    bb = rc0004 * hd;
	    thetap = (real)atan((real)(bb / m.f0085 * (real)tan((real)((
		    real)acos((real)(max(rc0003,m.f0248)))))));
/* Computing 2nd power */
	    r__1 = m.f0085;
	    m.f0250 = (real)exp((real)(-denfveg * pai * (r__1 * r__1) /
		    (real)cos(thetap)));
	    d__1 = (doublereal) m.f0085;
	    d__2 = (doublereal) rc0006;
	    fa = m.f0106 / (rc0005 * pai * (real)pow_dd(&d__1, &d__2) *
		    (bb / m.f0085) * denfveg);
	    newvai = hd * fa;
	    m.f0251 = (rc0002 - m.f0250) * (real)exp((real)(
		    -rc0004 * newvai / m.f0248));
/* Computing MIN */
	    r__1 = rc0002 - m.f0108, r__2 = m.f0250 +
		    m.f0251;
	    gap = min(r__1,r__2);
	    kopen = rc0007;
	}
	if (m.f0036 == 2) {
	    gap = rc0001;
	    kopen = rc0001;
	}
	if (m.f0036 == 3) {
	    gap = rc0002 - m.f0108;
	    kopen = rc0002 - m.f0108;
	}
    }
    coszi = max(rc0008,m.f0248);
/* Computing MIN */
/* Computing MAX */
    r__2 = m.f0086, r__3 = -rc0009;
    r__1 = max(r__2,r__3);
    chil = min(r__1,rc0010);
    if (abs(chil) <= rc0003) {
	chil = rc0003;
    }
    phi1 = rc0004 - rc0011 * chil - rc0012 * chil * chil;
    phi2 = rc0013 * (rc0014 - rc0015 * phi1);
    *gdir = phi1 + phi2 * coszi;
    *ext = *gdir / coszi;
    avmu = (rc0014 - phi1 / phi2 * (real)log((real)((phi1 + phi2) / phi1))) /
	    phi2;
    omegal = m.f0246[*ib - 1] + m.f0247[*ib - 1];
    tmp0 = *gdir + phi2 * coszi;
    tmp1 = phi1 * coszi;
    asu = rc0004 * omegal * *gdir / tmp0 * (rc0014 - tmp1 / tmp0 * (real)log((
	    real)((tmp1 + tmp0) / tmp1)));
    betadl = (rc0014 + avmu * *ext) / (omegal * avmu * *ext) * asu;
/* Computing 2nd power */
    r__1 = (rc0014 + chil) / rc0015;
    betail = rc0004 * (m.f0246[*ib - 1] + m.f0247[*ib - 1] + (
	    m.f0246[*ib - 1] - m.f0247[*ib - 1]) * (r__1 * r__1))
	    / omegal;
    if (m.f0209 > m.f0125) {
	tmp0 = omegal;
	tmp1 = betadl;
	tmp2 = betail;
    } else {
	tmp0 = (rc0014 - m.f0360) * omegal + m.f0360 *
		m.f0163[*ib - 1];
	tmp1 = ((rc0014 - m.f0360) * omegal * betadl + m.f0360 *
		m.f0163[*ib - 1] * m.f0164) / tmp0;
	tmp2 = ((rc0014 - m.f0360) * omegal * betail + m.f0360 *
		m.f0163[*ib - 1] * m.f0165) / tmp0;
    }
    omega = tmp0;
    betad = tmp1;
    betai = tmp2;
    b = rc0014 - omega + omega * betai;
    c__ = omega * betai;
    tmp0 = avmu * *ext;
    d__ = tmp0 * omega * betad;
    f = tmp0 * omega * (rc0014 - betad);
    tmp1 = b * b - c__ * c__;
    h__ = (real)sqrt(tmp1) / avmu;
    sigma = tmp0 * tmp0 - tmp1;
    if (abs(sigma) < rc0016) {
	sigma = (real)r_sign(&rc0016, &sigma);
    }
    p1 = b + avmu * h__;
    p2 = b - avmu * h__;
    p3 = b + tmp0;
    p4 = b - tmp0;
    s1 = (real)exp((real)(-h__ * m.f0106));
    s2 = (real)exp((real)(-(*ext) * m.f0106));
    if (*ic == 0) {
	u1 = b - c__ / m.f0232[*ib - 1];
	u2 = b - c__ * m.f0232[*ib - 1];
	u3 = f + c__ * m.f0232[*ib - 1];
    } else {
	u1 = b - c__ / m.f0233[*ib - 1];
	u2 = b - c__ * m.f0233[*ib - 1];
	u3 = f + c__ * m.f0233[*ib - 1];
    }
    tmp2 = u1 - avmu * h__;
    tmp3 = u1 + avmu * h__;
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1;
    tmp4 = u2 + avmu * h__;
    tmp5 = u2 - avmu * h__;
    d2 = tmp4 / s1 - tmp5 * s1;
    h1 = -d__ * p4 - c__ * f;
    tmp6 = d__ - h1 * p3 / sigma;
    tmp7 = (d__ - c__ - h1 / sigma * (u1 + tmp0)) * s2;
    h2 = (tmp6 * tmp2 / s1 - p2 * tmp7) / d1;
    h3 = -(tmp6 * tmp3 * s1 - p1 * tmp7) / d1;
    h4 = -f * p3 - c__ * d__;
    tmp8 = h4 / sigma;
    tmp9 = (u3 - tmp8 * (u2 - tmp0)) * s2;
    h5 = -(tmp8 * tmp4 / s1 + tmp9) / d2;
    h6 = (tmp8 * tmp5 * s1 + tmp9) / d2;
    h7 = c__ * tmp2 / (d1 * s1);
    h8 = -c__ * tmp3 * s1 / d1;
    h9 = tmp4 / (d2 * s1);
    h10 = -tmp5 * s1 / d2;
    if (*ic == 0) {
	ftds = s2 * (rc0002 - gap) + gap;
	ftis = (h4 * s2 / sigma + h5 * s1 + h6 / s1) * (rc0002 - gap);
	m.f0238[*ib - 1] = ftds;
	m.f0240[*ib - 1] = ftis;
    } else {
	ftds = rc0017;
	ftis = (h9 * s1 + h10 / s1) * (rc0002 - kopen) + kopen;
	m.f0239[*ib - 1] = ftds;
	m.f0241[*ib - 1] = ftis;
    }
    if (*ic == 0) {
	fres = (h1 / sigma + h2 + h3) * (rc0002 - gap) + m.f0232[*ib -
		1] * gap;
	freveg = (h1 / sigma + h2 + h3) * (rc0002 - gap);
	frebar = m.f0232[*ib - 1] * gap;
	m.f0230[*ib - 1] = fres;
	m.f0242[*ib - 1] = freveg;
	m.f0244[*ib - 1] = frebar;
	m.f0236[*ib - 1] = rc0014 - m.f0230[*ib - 1] - (rc0014 -
		m.f0232[*ib - 1]) * m.f0238[*ib - 1] - (rc0014 -
		m.f0233[*ib - 1]) * m.f0240[*ib - 1];
    } else {
	fres = (h7 + h8) * (rc0002 - kopen) + m.f0233[*ib - 1] * kopen;
	freveg = (h7 + h8) * (rc0002 - kopen) + m.f0233[*ib - 1] *
		kopen;
	frebar = 0.f;
	m.f0231[*ib - 1] = fres;
	m.f0243[*ib - 1] = freveg;
	m.f0245[*ib - 1] = frebar;
	m.f0237[*ib - 1] = rc0014 - m.f0231[*ib - 1] - (rc0014 -
		m.f0232[*ib - 1]) * m.f0239[*ib - 1] - (rc0014 -
		m.f0233[*ib - 1]) * m.f0241[*ib - 1];
    }
    return 0;
} /* r0026_ */

__device__ __forceinline__ /* Subroutine */ int r0027_(Model &m) {
    /* Initialized data */

    real rc0000 = .1f;
    real rc0001 = 0.f;
    real rc0002 = 6.f;
    real rc0003 = .622f;
    real rc0004 = .378f;
    real rc0005 = 1.f;
    real rc0006 = 2.f;
    real rc0007 = 4.f;
    real rc0008 = .01f;
    real rc0009 = .05f;
    real rc0010 = 1e-5f;

    /* System generated locals */
    real r__1, r__2;

    /* Builtin functions */


    /* Local variables */

    real a, b, h__, t, fh, hg, fm, rb, uc, fv, ch2, fh2, fm2, z0h;

    real wstar, z0hg, ramc, rahc, rawc, ramg, rahg, rawg, dtv, dtg, air, cir,
	    csh, cev, cgh, ctr, ata, bta, aea, bea, estv, estg, destv, destg,
	    esatw, esati, dsatw, dsati, fhg, hcan, cvh, caw, ctw, cew, cgw,
	    cond, moz, mozg, mozold, cq2v, qfx, vaie, laisune, laishae;
    integer iter;
    real cah;
    integer mozsgn, liter;
    real glaie, frsu;

    liter = 0;
    fv = rc0000;
    dtv = rc0001;
    dtg = rc0001;
    moz = rc0001;
    mozsgn = 0;
    mozold = rc0001;
    fh2 = rc0001;
    hg = rc0001;
    h__ = rc0001;
    qfx = rc0001;
    vaie = min(rc0002,m.f0106);
    laisune = min(rc0002,m.f0254);
    laishae = min(rc0002,m.f0255);
    t = (real)r0036_(m, &m.f0210, &m.f0125);
    r0035_(m, &t, &esatw, &esati, &dsatw, &dsati);
    if (t > rc0001) {
	estg = esatw;
    } else {
	estg = esati;
    }
    m.f0275 = rc0003 * m.f0201 / (m.f0180 - rc0004 *
	    m.f0201);
    hcan = m.f0079;
    uc = m.f0198 * (real)log((real)(hcan / m.f0268)) / (real)log((
	    real)(m.f0269 / m.f0268));
    uc = m.f0198 * (real)log((real)((hcan - m.f0266 +
	    m.f0268) / m.f0268)) / (real)log((real)(m.f0269
	    / m.f0268));
    if (hcan - m.f0266 <= rc0001) {
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
	m.status |= 1;
    }
/* Computing 4th power */
    r__1 = m.f0210, r__1 *= r__1;
    air = -m.f0276 * (rc0005 + (rc0005 - m.f0276) * (rc0005 -
	    m.f0277)) * m.f0191 - m.f0276 * m.f0277 *
	    m.f0123 * (r__1 * r__1);
    cir = (rc0006 - m.f0276 * (rc0005 - m.f0277)) * m.f0276
	    * m.f0123;
    for (iter = 1; iter <= 20; ++iter) {
	if (iter == 1) {
	    z0h = m.f0268;
	    z0hg = m.f0267;
	} else {
	    z0h = m.f0268;
	    z0hg = m.f0267;
	}
	if (m.f0037 == 1) {
	    r0033_(m, &iter, &m.f0181, &m.f0202, &h__, &
		    m.f0200, &m.f0269, &m.f0266, &
		    m.f0268, &z0h, &m.f0198, &moz, &mozsgn, &fm, &
		    fh, &fm2, &fh2, &m.f0272, &m.f0273, &fv, &ch2)
		    ;
	}
	if (m.f0037 == 2) {
	    r0034_(m, &iter, &m.f0268, &m.f0264, &m.f0199, &
		    m.f0198, &m.f0269, &m.f0272, &
		    m.f0273, &moz, &wstar, &fv);
	    m.f0273 /= m.f0198;
	    m.f0272 /= m.f0198;
	}
/* Computing MAX */
	r__1 = rc0005, r__2 = rc0005 / (m.f0272 * m.f0198);
	ramc = max(r__1,r__2);
/* Computing MAX */
	r__1 = rc0005, r__2 = rc0005 / (m.f0273 * m.f0198);
	rahc = max(r__1,r__2);
	rawc = rahc;
	r0029_(m, &iter, &vaie, &m.f0202, &hg, &m.f0264, &
		m.f0266, &m.f0267, &z0hg, &hcan, &uc, &z0h, &fv, &
		m.f0022, &m.f0209, &mozg, &fhg, &ramg, &rahg, &
		rawg, &rb);
	t = (real)r0036_(m, &m.f0209, &m.f0125);
	r0035_(m, &t, &esatw, &esati, &dsatw, &dsati);
	if (t > rc0001) {
	    estv = esatw;
	    destv = dsatw;
	} else {
	    estv = esati;
	    destv = dsati;
	}
	if (iter == 1) {
	    if (m.f0038 == 1) {
		r0030_(m, &m.f0022, &m.f0256, &m.f0192, &
			m.f0209, &estv, &m.f0265, &m.f0181,
			&m.f0180, &m.f0193, &m.f0194, &
			m.f0213, &m.f0407, &rb, &m.f0315, &
			m.f0332);
		r0030_(m, &m.f0022, &m.f0257, &m.f0192, &
			m.f0209, &estv, &m.f0265, &m.f0181,
			&m.f0180, &m.f0193, &m.f0194, &
			m.f0213, &m.f0407, &rb, &m.f0316, &
			m.f0333);
	    }
	    if (m.f0038 == 2) {
		r0031_(m, &m.f0256, &m.f0209, &m.f0407, &
			m.f0265, &m.f0180, &m.f0315, &
			m.f0332);
		r0031_(m, &m.f0257, &m.f0209, &m.f0407, &
			m.f0265, &m.f0180, &m.f0316, &
			m.f0333);
	    }
	    if (m.f0043 == 4) {
		m.f0315 = m.f0093;
		m.f0316 = m.f0093;
	    }
	}
	cah = rc0005 / rahc;
	cvh = rc0006 * vaie / rb;
	cgh = rc0005 / rahg;
	cond = cah + cvh + cgh;
	ata = (m.f0181 * cah + m.f0210 * cgh) / cond;
	bta = cvh / cond;
	csh = (rc0005 - bta) * m.f0202 * m.f0131 * cvh;
	caw = rc0005 / rawc;
	cew = m.f0360 * vaie / rb;
	if (m.f0039 != 2) {
	    ctw = (rc0005 - m.f0360) * (laisune / (rb + m.f0315)
		    + laishae / (rb + m.f0316));
	} else {
	    ctw = (rc0005 - m.f0360) * (rc0005 / (rb / (frsu * glaie) +
		    m.f0315) + rc0005 / (rb / ((rc0005 - frsu) * glaie)
		    + m.f0316));
	}
	cgw = rc0005 / (rawg + m.f0301);
	cond = caw + cew + ctw + cgw;
	aea = (m.f0201 * caw + estg * cgw) / cond;
	bea = (cew + ctw) / cond;
	cev = (rc0005 - bea) * cew * m.f0202 * m.f0131 /
		m.f0278;
	ctr = (rc0005 - bea) * ctw * m.f0202 * m.f0131 /
		m.f0278;
	m.f0264 = ata + bta * m.f0209;
	m.f0265 = aea + bea * estv;
/* Computing 4th power */
	r__1 = m.f0209, r__1 *= r__1;
	m.f0281 = m.f0108 * (air + cir * (r__1 * r__1));
	m.f0283 = m.f0108 * m.f0202 * m.f0131 * cvh *
		(m.f0209 - m.f0264);
	m.f0280 = m.f0108 * m.f0202 * m.f0131 * cew *
		(estv - m.f0265) / m.f0278;
	m.f0288 = m.f0108 * m.f0202 * m.f0131 * ctw *
		(estv - m.f0265) / m.f0278;
	if (m.f0209 > m.f0125) {
/* Computing MIN */
	    r__1 = m.f0358 * m.f0298 / m.f0005;
	    m.f0280 = min(r__1,m.f0280);
	} else {
/* Computing MIN */
	    r__1 = m.f0359 * m.f0298 / m.f0005;
	    m.f0280 = min(r__1,m.f0280);
	}
	b = m.f0258 - m.f0281 - m.f0283 - m.f0280 -
		m.f0288 + m.f0222;
/* Computing 3rd power */
	r__1 = m.f0209;
	a = m.f0108 * (rc0007 * cir * (r__1 * (r__1 * r__1)) + csh + (
		cev + ctr) * destv);
	dtv = b / a;
/* Computing 3rd power */
	r__1 = m.f0209;
	m.f0281 += m.f0108 * rc0007 * cir * (r__1 * (r__1 * r__1))
		 * dtv;
	m.f0283 += m.f0108 * csh * dtv;
	m.f0280 += m.f0108 * cev * destv * dtv;
	m.f0288 += m.f0108 * ctr * destv * dtv;
	m.f0209 += dtv;
	h__ = m.f0202 * m.f0131 * (m.f0264 - m.f0181)
		/ rahc;
	hg = m.f0202 * m.f0131 * (m.f0210 - m.f0264) /
		 rahg;
	m.f0275 = rc0003 * m.f0265 / (m.f0180 - rc0004 *
		m.f0265);
	if (liter == 1) {
	    goto L9005;
	}
	if (iter >= 5 && abs(dtv) <= rc0008 && liter == 0) {
	    liter = 1;
	}
/* L9004: */
    }
L9005:
/* Computing 4th power */
    r__1 = m.f0209, r__1 *= r__1;
    air = -m.f0277 * (rc0005 - m.f0276) * m.f0191 -
	    m.f0277 * m.f0276 * m.f0123 * (r__1 * r__1);
    cir = m.f0277 * m.f0123;
    csh = m.f0202 * m.f0131 / rahg;
    cev = m.f0202 * m.f0131 / (m.f0279 * (rawg +
	    m.f0301));
    cgh = rc0006 * m.f0218[m.f0390 + 3] / m.f0027[
	    m.f0390 + 3];
    for (iter = 1; iter <= 5; ++iter) {
	t = (real)r0036_(m, &m.f0210, &m.f0125);
	r0035_(m, &t, &esatw, &esati, &dsatw, &dsati);
	if (t > rc0001) {
	    estg = esatw;
	    destg = dsatw;
	} else {
	    estg = esati;
	    destg = dsati;
	}
/* Computing 4th power */
	r__1 = m.f0210, r__1 *= r__1;
	m.f0282 = cir * (r__1 * r__1) + air;
	m.f0284 = csh * (m.f0210 - m.f0264);
	m.f0286 = cev * (estg * m.f0302 - m.f0265);
	m.f0291 = cgh * (m.f0210 - m.f0217[m.f0390 +
		3]);
	b = m.f0259 - m.f0282 - m.f0284 - m.f0286 -
		m.f0291 + m.f0223;
/* Computing 3rd power */
	r__1 = m.f0210;
	a = rc0007 * cir * (r__1 * (r__1 * r__1)) + csh + cev * destg + cgh;
	dtg = b / a;
/* Computing 3rd power */
	r__1 = m.f0210;
	m.f0282 += rc0007 * cir * (r__1 * (r__1 * r__1)) * dtg;
	m.f0284 += csh * dtg;
	m.f0286 += cev * destg * dtg;
	m.f0291 += cgh * dtg;
	m.f0210 += dtg;
/* L9007: */
    }
/* L9008: */
    if (m.f0040 == 1 || m.f0040 == 3) {
	if (m.f0375 > rc0009 && m.f0210 > m.f0125) {
	    if (m.f0040 == 1) {
		m.f0210 = m.f0125;
	    }
	    if (m.f0040 == 3) {
		m.f0210 = (rc0005 - m.f0406) * m.f0210 +
			m.f0406 * m.f0125;
	    }
/* Computing 4th power */
	    r__1 = m.f0210, r__1 *= r__1;
/* Computing 4th power */
	    r__2 = m.f0209, r__2 *= r__2;
	    m.f0282 = cir * (r__1 * r__1) - m.f0277 * (rc0005 -
		    m.f0276) * m.f0191 - m.f0277 *
		    m.f0276 * m.f0123 * (r__2 * r__2);
	    m.f0284 = csh * (m.f0210 - m.f0264);
	    m.f0286 = cev * (estg * m.f0302 - m.f0265);
	    m.f0291 = m.f0259 + m.f0223 - (m.f0282 +
		    m.f0284 + m.f0286);
	}
    }
    m.f0303 = -m.f0202 * m.f0272 * m.f0198 *
	    m.f0195;
    m.f0304 = -m.f0202 * m.f0272 * m.f0198 *
	    m.f0196;
    if (m.f0037 == 1 || m.f0037 == 2) {
	m.f0309 = fv * m.f0124 / (real)log((real)((rc0006 + z0h) /
		 z0h));
	m.f0309 = fv * m.f0124 / ((real)log((real)((rc0006 + z0h)
		/ z0h)) - fh2);
	cq2v = m.f0309;
	if (m.f0309 < rc0010) {
	    m.f0292 = m.f0264;
	    m.f0297 = m.f0275;
	} else {
	    m.f0292 = m.f0264 - (m.f0284 + m.f0283 /
		    m.f0108) / (m.f0202 * m.f0131) * rc0005
		    / m.f0309;
	    m.f0297 = m.f0275 - ((m.f0280 + m.f0288) /
		     m.f0108 + m.f0286) / (m.f0298 *
		    m.f0202) * rc0005 / cq2v;
	}
    }
    m.f0273 = cah;
    m.f0293 = cvh;
    m.f0294 = rc0005 / rahg;
    return 0;
} /* r0027_ */

__device__ __forceinline__ /* Subroutine */ int r0028_(Model &m) {
    /* Initialized data */

    integer niterb = 5;
    real rc0000 = 0.f;
    real rc0001 = .1f;
    real rc0002 = 2.f;
    real rc0003 = .01f;
    real rc0004 = 1.f;
    real rc0005 = 4.f;
    real rc0006 = .622f;
    real rc0007 = .378f;
    real rc0008 = .05f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */


    /* Local variables */
    real a, b, h__, t, fh, fm, fv, ch2, fh2, fm2, z0h;

    real ehb, wstar, ramb, rahb, rawb, dtg, cir, csh, cev, cgh, cq2b, emb,
	    qfx, estg, destg, esatw, esati, dsatw, dsati, moz, mozold;
    integer mozsgn, iter;


    dtg = rc0000;
    moz = rc0000;
    mozsgn = 0;
    mozold = rc0000;
    fh2 = rc0000;
    h__ = rc0000;
    qfx = rc0000;
    fv = rc0001;
    cir = m.f0277 * m.f0123;
    cgh = rc0002 * m.f0218[m.f0390 + 3] / m.f0027[
	    m.f0390 + 3];
    i__1 = niterb;
    for (iter = 1; iter <= i__1; ++iter) {
	if (iter == 1) {
	    z0h = m.f0268;
	} else {
	    z0h = m.f0268;
	}
	if (m.f0037 == 1) {
	    r0033_(m, &iter, &m.f0181, &m.f0202, &h__, &
		    m.f0200, &m.f0269, &m.f0266, &
		    m.f0268, &z0h, &m.f0198, &moz, &mozsgn, &fm, &
		    fh, &fm2, &fh2, &m.f0272, &m.f0273, &fv, &ch2)
		    ;
	}
	if (m.f0037 == 2) {
	    r0034_(m, &iter, &m.f0268, &m.f0274, &m.f0199, &
		    m.f0198, &m.f0269, &m.f0272, &
		    m.f0273, &moz, &wstar, &fv);
	    m.f0273 /= m.f0198;
	    m.f0272 /= m.f0198;
	    if (m.f0375 > rc0000) {
		m.f0272 = min(rc0003,m.f0272);
		m.f0273 = min(rc0003,m.f0273);
	    }
	}
/* Computing MAX */
	r__1 = rc0004, r__2 = rc0004 / (m.f0272 * m.f0198);
	ramb = max(r__1,r__2);
/* Computing MAX */
	r__1 = rc0004, r__2 = rc0004 / (m.f0273 * m.f0198);
	rahb = max(r__1,r__2);
	rawb = rahb;
	emb = rc0004 / ramb;
	ehb = rc0004 / rahb;
	t = (real)r0036_(m, &m.f0274, &m.f0125);
	r0035_(m, &t, &esatw, &esati, &dsatw, &dsati);
	if (t > rc0000) {
	    estg = esatw;
	    destg = dsatw;
	} else {
	    estg = esati;
	    destg = dsati;
	}
	csh = m.f0202 * m.f0131 / rahb;
	cev = m.f0202 * m.f0131 / m.f0279 / (m.f0301
		+ rawb);
/* Computing 4th power */
	r__1 = m.f0274, r__1 *= r__1;
	m.f0322 = cir * (r__1 * r__1) - m.f0277 * m.f0191;
	m.f0285 = csh * (m.f0274 - m.f0181);
	m.f0287 = cev * (estg * m.f0302 - m.f0201);
	m.f0290 = cgh * (m.f0274 - m.f0217[m.f0390 +
		3]);
	b = m.f0259 - m.f0322 - m.f0285 - m.f0287 -
		m.f0290 + m.f0224;
/* Computing 3rd power */
	r__1 = m.f0274;
	a = rc0005 * cir * (r__1 * (r__1 * r__1)) + csh + cev * destg + cgh;
	dtg = b / a;
/* Computing 3rd power */
	r__1 = m.f0274;
	m.f0322 += rc0005 * cir * (r__1 * (r__1 * r__1)) * dtg;
	m.f0285 += csh * dtg;
	m.f0287 += cev * destg * dtg;
	m.f0290 += cgh * dtg;
	m.f0274 += dtg;
	h__ = csh * (m.f0274 - m.f0181);
	t = (real)r0036_(m, &m.f0274, &m.f0125);
	r0035_(m, &t, &esatw, &esati, &dsatw, &dsati);
	if (t > rc0000) {
	    estg = esatw;
	} else {
	    estg = esati;
	}
	m.f0275 = rc0006 * (estg * m.f0302) / (m.f0180 -
		rc0007 * (estg * m.f0302));
	qfx = (m.f0275 - m.f0200) * cev * m.f0279 /
		m.f0131;
/* L9004: */
    }
/* L9005: */
    if (m.f0040 == 1 || m.f0040 == 3) {
	if (m.f0375 > rc0008 && m.f0274 > m.f0125) {
	    if (m.f0040 == 1) {
		m.f0274 = m.f0125;
	    }
	    if (m.f0040 == 3) {
		m.f0274 = (rc0004 - m.f0406) * m.f0274 +
			m.f0406 * m.f0125;
	    }
/* Computing 4th power */
	    r__1 = m.f0274, r__1 *= r__1;
	    m.f0322 = cir * (r__1 * r__1) - m.f0277 *
		    m.f0191;
	    m.f0285 = csh * (m.f0274 - m.f0181);
	    m.f0287 = cev * (estg * m.f0302 - m.f0201);
	    m.f0290 = m.f0259 + m.f0224 - (m.f0322 +
		    m.f0285 + m.f0287);
	}
    }
    m.f0305 = -m.f0202 * m.f0272 * m.f0198 *
	    m.f0195;
    m.f0306 = -m.f0202 * m.f0272 * m.f0198 *
	    m.f0196;
    if (m.f0037 == 1 || m.f0037 == 2) {
	m.f0310 = fv * m.f0124 / (real)log((real)((rc0002 + z0h) /
		 z0h));
	m.f0310 = fv * m.f0124 / ((real)log((real)((rc0002 + z0h)
		/ z0h)) - fh2);
	cq2b = m.f0310;
	if (m.f0310 < 1e-5f) {
	    m.f0311 = m.f0274;
	    m.f0312 = m.f0275;
	} else {
	    m.f0311 = m.f0274 - m.f0285 / (m.f0202 *
		    m.f0131) * rc0004 / m.f0310;
	    m.f0312 = m.f0275 - m.f0287 / (m.f0299 *
		    m.f0202) * (rc0004 / cq2b + m.f0301);
	}
	if (m.f0072) {
	    m.f0312 = m.f0275;
	}
    }
    m.f0273 = ehb;
    return 0;
} /* r0028_ */

__device__ __forceinline__ /* Subroutine */ int r0029_(Model &m, integer *iter, real *vai, real *rhoair, real *hg,
	real *tah, real *zpd, real *z0mg, real *z0hg, real *hcan, real *uc,
	real *z0h, real *fv, integer *vegtyp, real *tv, real *mozg, real *fhg,
	 real *ramg, real *rahg, real *rawg, real *rb) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;
    real rc0002 = 15.f;
    real rc0003 = .25f;
    real rc0004 = 4.7f;
    real rc0005 = .5f;
    real rc0006 = 50.f;
    real rc0007 = 2.f;
    real rc0008 = 100.f;

    /* System generated locals */
    real r__1;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real kh, tmp1, tmp2, tmprah2, tmprb, molg, fhgnew, cwpc;

    *mozg = rc0000;
    molg = rc0000;
    if (*iter > 1) {
	tmp1 = m.f0124 * (m.f0177 / *tah) * *hg / (*rhoair *
		m.f0131);
	if (abs(tmp1) <= m.f0171) {
	    tmp1 = m.f0171;
	}
/* Computing 3rd power */
	r__1 = *fv;
	molg = -rc0001 * (r__1 * (r__1 * r__1)) / tmp1;
/* Computing MIN */
	r__1 = (*zpd - *z0mg) / molg;
	*mozg = min(r__1,rc0001);
    }
    if (*mozg < rc0000) {
	d__1 = (doublereal) (rc0001 - rc0002 * *mozg);
	d__2 = (doublereal) (-rc0003);
	fhgnew = (real)pow_dd(&d__1, &d__2);
    } else {
	fhgnew = rc0001 + rc0004 * *mozg;
    }
    if (*iter == 1) {
	*fhg = fhgnew;
    } else {
	*fhg = rc0005 * (*fhg + fhgnew);
    }
    d__1 = (doublereal) (m.f0099 * *vai * *hcan * *fhg);
    d__2 = (doublereal) rc0005;
    cwpc = (real)pow_dd(&d__1, &d__2);
    tmp1 = (real)exp((real)(-cwpc * *z0hg / *hcan));
    tmp2 = (real)exp((real)(-cwpc * (*z0h + *zpd) / *hcan));
    tmprah2 = *hcan * (real)exp(cwpc) / cwpc * (tmp1 - tmp2);
/* Computing MAX */
    r__1 = m.f0124 * *fv * (*hcan - *zpd);
    kh = max(r__1,m.f0171);
    *ramg = rc0000;
    *rahg = tmprah2 / kh;
    *rawg = *rahg;
    tmprb = cwpc * rc0006 / (rc0001 - (real)exp((real)(-cwpc / rc0007)));
    *rb = tmprb * (real)sqrt((real)(m.f0101 / *uc));
    *rb = max(*rb,rc0008);
    return 0;
} /* r0029_ */

__device__ __forceinline__ /* Subroutine */ int r0030_(Model &m, integer *vegtyp, real *apar, real *foln, real *tv,
	 real *ei, real *ea, real *sfctmp, real *sfcprs, real *o2, real *co2,
	real *igs, real *btran, real *rb, real *rs, real *psn) {
    /* Initialized data */

    integer niter = 3;
    real rc0000 = 8.314f;
    real rc0001 = 1e6f;
    real rc0002 = 1.f;
    real rc0003 = 0.f;
    real rc0004 = 1.f;
    real rc0005 = 4.6f;
    real rc0006 = .5f;
    real rc0007 = .21f;
    real rc0008 = .7f;
    real rc0009 = .4f;
    real rc0010 = .25f;
    real rc0011 = .4f;
    real rc0012 = 2.f;
    real rc0013 = 4e3f;
    real rc0014 = 1.37f;
    real rc0015 = 4.f;
    real rc0016 = 1.65f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */


    /* Local variables */
    real a, b, c__, j, q, r1, r2, cf, ci, kc, cp, cs, tc, ko, wc, we, wj;

    integer iter;
    real rlb, fnf, ppf, awc, vcmx, cea;

    cf = *sfcprs / (rc0000 * *sfctmp) * rc0001;
    *rs = rc0002 / m.f0087 * cf;
    *psn = rc0003;
    if (*apar <= rc0003) {
	return 0;
    }
/* Computing MIN */
    r__1 = *foln / max(m.f0171,m.f0088);
    fnf = min(r__1,rc0004);
    tc = *tv - m.f0125;
    ppf = rc0005 * *apar;
    j = ppf * m.f0089;
    kc = m.f0102 * (real)r0037_(m, &m.f0095, &tc);
    ko = m.f0103 * (real)r0037_(m, &m.f0096, &tc);
    awc = kc * (rc0002 + *o2 / ko);
    cp = rc0006 * kc / ko * *o2 * rc0007;
    vcmx = m.f0090 / (real)r0038_(m, &tc) * fnf * *btran * (real)r0037_(m, &
	    m.f0097, &tc);
    ci = rc0008 * *co2 * m.f0100 + rc0009 * *co2 * (rc0002 -
	    m.f0100);
    rlb = *rb / cf;
/* Computing MAX */
    r__1 = rc0010 * *ei * m.f0100 + rc0011 * *ei * (rc0002 -
	    m.f0100), r__2 = min(*ea,*ei);
    cea = max(r__1,r__2);
    i__1 = niter;
    for (iter = 1; iter <= i__1; ++iter) {
/* Computing MAX */
	r__1 = ci - cp;
	wj = max(r__1,rc0003) * j / (ci + rc0012 * cp) * m.f0100 + j * (
		rc0002 - m.f0100);
/* Computing MAX */
	r__1 = ci - cp;
	wc = max(r__1,rc0003) * vcmx / (ci + awc) * m.f0100 + vcmx * (
		rc0002 - m.f0100);
	we = rc0006 * vcmx * m.f0100 + rc0013 * vcmx * ci / *sfcprs * (
		rc0002 - m.f0100);
/* Computing MIN */
	r__1 = min(wj,wc);
	*psn = min(r__1,we) * *igs;
/* Computing MAX */
	r__1 = *co2 - rc0014 * rlb * *sfcprs * *psn;
	cs = max(r__1,m.f0171);
	a = m.f0091 * *psn * *sfcprs * cea / (cs * *ei) + m.f0087;
	b = (m.f0091 * *psn * *sfcprs / cs + m.f0087) * rlb -
		rc0002;
	c__ = -rlb;
	if (b >= rc0003) {
	    q = -rc0006 * (b + (real)sqrt((real)(b * b - rc0015 * a * c__)));
	} else {
	    q = -rc0006 * (b - (real)sqrt((real)(b * b - rc0015 * a * c__)));
	}
	r1 = q / a;
	r2 = c__ / q;
	*rs = max(r1,r2);
/* Computing MAX */
	r__1 = cs - *psn * *sfcprs * rc0016 * *rs;
	ci = max(r__1,rc0003);
/* L9004: */
    }
/* L9005: */
    *rs *= cf;
    return 0;
} /* r0030_ */

__device__ __forceinline__ /* Subroutine */ int r0031_(Model &m, real *par, real *tv, real *btran, real *eah, real
	*sfcprs, real *rs, real *psn) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = .622f;
    real rc0002 = .378f;
    real rc0003 = 1.f;
    real rc0004 = 2.f;
    real rc0005 = 1e-4f;
    real rc0006 = .0016f;
    real rc0007 = 0.f;
    real rc0008 = .01f;
    real rc0009 = 999.99f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real q2, ff, rcq, rcs, rct, q2sat, dqsdt2;


    *rs = rc0000;
    rcs = rc0000;
    rct = rc0000;
    rcq = rc0000;
    q2 = rc0001 * *eah / (*sfcprs - rc0002 * *eah);
    q2 /= rc0003 + q2;
    r0032_(m, tv, sfcprs, &q2sat, &dqsdt2);
    ff = rc0004 * *par / m.f0092;
    rcs = (ff + m.f0093 / m.f0098) / (rc0003 + ff);
    rcs = max(rcs,rc0005);
    d__1 = (doublereal) (m.f0172 - *tv);
    d__2 = (doublereal) rc0004;
    rct = rc0003 - rc0006 * (real)pow_dd(&d__1, &d__2);
    rct = max(rct,rc0005);
/* Computing MAX */
    r__1 = rc0007, r__2 = q2sat - q2;
    rcq = rc0003 / (rc0003 + m.f0094 * max(r__1,r__2));
    rcq = max(rcq,rc0008);
    *rs = m.f0093 / (rcs * rct * rcq * *btran);
    *psn = -rc0009;
    return 0;
} /* r0031_ */

__device__ __forceinline__ /* Subroutine */ int r0032_(Model &m, real *tv, real *sfcprs, real *q2sat, real *dqsdt2) {
    /* Initialized data */

    real rc0000 = 17.67f;
    real rc0001 = 273.15f;
    real rc0002 = 29.65f;
    real rc0003 = 2.501e6f;
    real rc0004 = .611f;
    real rc0005 = 461.f;
    real rc0006 = .622f;
    real rc0007 = 1.f;
    real rc0008 = .001f;
    real rc0009 = 1e3f;

    /* System generated locals */
    real r__1;

    /* Builtin functions */


    /* Local variables */
    real a2, a3, a4, e0, es, rv, elwv, a23m4, epsilon, sfcprsx;

    a2 = rc0000;
    a3 = rc0001;
    a4 = rc0002;
    elwv = rc0003;
    a23m4 = a2 * (a3 - a4);
    e0 = rc0004;
    rv = rc0005;
    epsilon = rc0006;
    es = e0 * (real)exp((real)(elwv / rv * (rc0007 / a3 - rc0007 / *tv)));
    sfcprsx = *sfcprs * rc0008;
    *q2sat = epsilon * es / (sfcprsx - es);
/* Computing 2nd power */
    r__1 = *tv - a4;
    *dqsdt2 = *q2sat * rc0009 / (*q2sat * rc0009 + 1) * a23m4 / (r__1 * r__1);
    return 0;
} /* r0032_ */

__device__ __forceinline__ /* Subroutine */ int r0033_(Model &m, integer *iter, real *sfctmp, real *rhoair, real *
	h__, real *qair, real *zlvl, real *zpd, real *z0m, real *z0h, real *
	ur, real *moz, integer *mozsgn, real *fm, real *fh, real *fm2, real *
	fh2, real *cm, real *ch, real *fv, real *ch2) {
    /* Initialized data */

    real rc0000 = 2.f;
    real rc0001 = 0.f;
    real rc0002 = 1.f;
    real rc0003 = .61f;
    real rc0004 = 0.f;
    real rc0005 = 16.f;
    real rc0006 = .25f;
    real rc0007 = 2.f;
    real rc0008 = 1.5707963f;
    real rc0009 = 5.f;
    real rc0010 = .5f;
    real rc0011 = .9f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real mol, tmpcm, tmpch, fmnew, fhnew, mozold, tmp1, tmp2, tmp3, tvir,
	    moz2, tmpcm2, tmpch2, fm2new, fh2new, tmp12, tmp22, tmp32, cmfm,
	    chfh, cm2fm2, ch2fh2;

    mozold = *moz;
    if (*zlvl <= *zpd) {
	m.status |= 1;
    }
    tmpcm = (real)log((real)((*zlvl - *zpd) / *z0m));
    tmpch = (real)log((real)((*zlvl - *zpd) / *z0h));
    tmpcm2 = (real)log((real)((rc0000 + *z0m) / *z0m));
    tmpch2 = (real)log((real)((rc0000 + *z0h) / *z0h));
    if (*iter == 1) {
	*fv = rc0001;
	*moz = rc0001;
	mol = rc0001;
	moz2 = rc0001;
    } else {
	tvir = (rc0002 + rc0003 * *qair) * *sfctmp;
	tmp1 = m.f0124 * (m.f0177 / tvir) * *h__ / (*rhoair *
		m.f0131);
	if (abs(tmp1) <= m.f0171) {
	    tmp1 = m.f0171;
	}
/* Computing 3rd power */
	r__1 = *fv;
	mol = -rc0002 * (r__1 * (r__1 * r__1)) / tmp1;
/* Computing MIN */
	r__1 = (*zlvl - *zpd) / mol;
	*moz = min(r__1,rc0002);
/* Computing MIN */
	r__1 = (rc0000 + *z0h) / mol;
	moz2 = min(r__1,rc0002);
    }
    if (mozold * *moz < rc0004) {
	++(*mozsgn);
    }
    if (*mozsgn >= 2) {
	*moz = rc0004;
	*fm = rc0004;
	*fh = rc0004;
	moz2 = rc0004;
	*fm2 = rc0004;
	*fh2 = rc0004;
    }
    if (*moz < rc0004) {
	d__1 = (doublereal) (rc0002 - rc0005 * *moz);
	d__2 = (doublereal) rc0006;
	tmp1 = (real)pow_dd(&d__1, &d__2);
	tmp2 = (real)log((real)((rc0002 + tmp1 * tmp1) / rc0007));
	tmp3 = (real)log((real)((rc0002 + tmp1) / rc0007));
	fmnew = rc0007 * tmp3 + tmp2 - rc0007 * (real)atan(tmp1) + rc0008;
	fhnew = tmp2 * 2;
	d__1 = (doublereal) (rc0002 - rc0005 * moz2);
	d__2 = (doublereal) rc0006;
	tmp12 = (real)pow_dd(&d__1, &d__2);
	tmp22 = (real)log((real)((rc0002 + tmp12 * tmp12) / rc0007));
	tmp32 = (real)log((real)((rc0002 + tmp12) / rc0007));
	fm2new = rc0007 * tmp32 + tmp22 - rc0007 * (real)atan(tmp12) + rc0008;
	fh2new = tmp22 * 2;
    } else {
	fmnew = -rc0009 * *moz;
	fhnew = fmnew;
	fm2new = -rc0009 * moz2;
	fh2new = fm2new;
    }
    if (*iter == 1) {
	*fm = fmnew;
	*fh = fhnew;
	*fm2 = fm2new;
	*fh2 = fh2new;
    } else {
	*fm = rc0010 * (*fm + fmnew);
	*fh = rc0010 * (*fh + fhnew);
	*fm2 = rc0010 * (*fm2 + fm2new);
	*fh2 = rc0010 * (*fh2 + fh2new);
    }
/* Computing MIN */
    r__1 = *fh, r__2 = rc0011 * tmpch;
    *fh = min(r__1,r__2);
/* Computing MIN */
    r__1 = *fm, r__2 = rc0011 * tmpcm;
    *fm = min(r__1,r__2);
/* Computing MIN */
    r__1 = *fh2, r__2 = rc0011 * tmpch2;
    *fh2 = min(r__1,r__2);
/* Computing MIN */
    r__1 = *fm2, r__2 = rc0011 * tmpcm2;
    *fm2 = min(r__1,r__2);
    cmfm = tmpcm - *fm;
    chfh = tmpch - *fh;
    cm2fm2 = tmpcm2 - *fm2;
    ch2fh2 = tmpch2 - *fh2;
    if (abs(cmfm) <= m.f0171) {
	cmfm = m.f0171;
    }
    if (abs(chfh) <= m.f0171) {
	chfh = m.f0171;
    }
    if (abs(cm2fm2) <= m.f0171) {
	cm2fm2 = m.f0171;
    }
    if (abs(ch2fh2) <= m.f0171) {
	ch2fh2 = m.f0171;
    }
    *cm = m.f0124 * m.f0124 / (cmfm * cmfm);
    *ch = m.f0124 * m.f0124 / (cmfm * chfh);
    *ch2 = m.f0124 * m.f0124 / (cm2fm2 * ch2fh2);
    *fv = *ur * (real)sqrt(*cm);
    *ch2 = m.f0124 * *fv / ch2fh2;
    return 0;
} /* r0033_ */

__device__ __forceinline__ /* Subroutine */ int r0034_(Model &m, integer *iter, real *z0, real *thz0, real *thlm,
	real *sfcspd, real *zlm, real *akms, real *akhs, real *rlmo, real *
	wstar2, real *ustar) {
    /* Initialized data */

    real rc0000 = 1.2f;
    real rc0001 = .4f;
    real rc0002 = .001f;
    real rc0003 = 1.f;
    real rc0004 = 270.f;
    real rc0005 = .15f;
    real rc0006 = 3.14159265f;
    real rc0007 = 2.f;
    real rc0008 = 1e-4f;
    real rc0009 = .07f;
    real rc0010 = 1e-8f;
    real rc0011 = 5.f;
    real rc0012 = 1e3f;
    real rc0013 = 258.2f;
    real rc0014 = .183f;
    real rc0015 = .8f;
    real rc0016 = .191f;
    real rc0017 = 1.f;
    real rc0018 = 0.f;
    real rc0019 = 3.f;
    real rc0020 = 1e-6f;
    real rc0021 = 0.f;
    real rc0022 = 16.f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real xt, xu, zt, zu, du2, xt4, xu4;

    real zilfc, rdz, cxch, dthv, btgh, zslu, zslt, rlogu, rlogt, zetalt,
	    zetalu, zetau, zetat, xlu4, xlt4, xlu, xlt, psmz, simm, pshz,
	    simh, ustark, rlmn, rlma, btg, elfc;
    integer ilech;
    real wwst, wwst2, vkrm, excm, beta, wold, wnew, pihf, epsu2, epsust,
	    epsit, epsa, ztmin, ztmax, hpbl, sqvisc, ric, rric, fhneu, rfc,
	    rfac;

    wwst = rc0000;
    wwst2 = wwst * wwst;
    vkrm = rc0001;
    excm = rc0002;
    beta = rc0003 / rc0004;
    wold = rc0005;
    wnew = rc0003 - wold;
    pihf = rc0006 / rc0007;
    epsu2 = rc0008;
    epsust = rc0009;
    epsit = rc0008;
    epsa = rc0010;
    ztmin = -rc0011;
    ztmax = rc0003;
    hpbl = rc0012;
    sqvisc = rc0013;
    ric = rc0014;
    rric = rc0003 / ric;
    fhneu = rc0015;
    rfc = rc0016;
    rfac = ric / (fhneu * rfc * rfc);
    btg = beta * m.f0177;
    elfc = vkrm * btg;
    ilech = 0;
    zilfc = -m.f0065 * vkrm * sqvisc;
    zu = *z0;
    rdz = rc0017 / *zlm;
    cxch = excm * rdz;
    dthv = *thlm - *thz0;
/* Computing MAX */
    r__1 = *sfcspd * *sfcspd;
    du2 = max(r__1,epsu2);
    btgh = btg * hpbl;
    if (*iter == 1) {
	if (btgh * *akhs * dthv != rc0018) {
	    d__1 = (r__1 = btgh * *akhs * dthv, (doublereal) abs(r__1));
	    d__2 = (doublereal) (rc0007 / rc0019);
	    *wstar2 = wwst2 * (real)pow_dd(&d__1, &d__2);
	} else {
	    *wstar2 = rc0018;
	}
/* Computing MAX */
	r__1 = (real)sqrt((real)(*akms * (real)sqrt((real)(du2 + *wstar2))));
	*ustar = max(r__1,epsust);
/* Computing 3rd power */
	r__1 = *ustar;
	*rlmo = elfc * *akhs * dthv / (r__1 * (r__1 * r__1));
    }
/* Computing MAX */
    r__1 = rc0020, r__2 = (real)exp((real)(zilfc * (real)sqrt((real)(*ustar *
	    *z0)))) * *z0;
    zt = max(r__1,r__2);
    zslu = *zlm + zu;
    zslt = *zlm + zt;
    rlogu = (real)log((real)(zslu / zu));
    rlogt = (real)log((real)(zslt / zt));
/* Computing MAX */
    r__1 = zslt * *rlmo;
    zetalt = max(r__1,ztmin);
    *rlmo = zetalt / zslt;
    zetalu = zslu * *rlmo;
    zetau = zu * *rlmo;
    zetat = zt * *rlmo;
    if (ilech == 0) {
	if (*rlmo < rc0021) {
	    xlu4 = rc0017 - rc0022 * zetalu;
	    xlt4 = rc0017 - rc0022 * zetalt;
	    xu4 = rc0017 - rc0022 * zetau;
	    xt4 = rc0017 - rc0022 * zetat;
	    xlu = (real)sqrt((real)((real)sqrt(xlu4)));
	    xlt = (real)sqrt((real)((real)sqrt(xlt4)));
	    xu = (real)sqrt((real)((real)sqrt(xu4)));
	    xt = (real)sqrt((real)((real)sqrt(xt4)));
	    psmz = (real)r0043_(m, &xu, &pihf);
	    simm = (real)r0043_(m, &xlu, &pihf) - psmz + rlogu;
	    pshz = (real)r0045_(m, &xt);
	    simh = (real)r0045_(m, &xlt) - pshz + rlogt;
	} else {
	    zetalu = min(zetalu,ztmax);
	    zetalt = min(zetalt,ztmax);
/* Computing MIN */
	    r__1 = zetau, r__2 = ztmax / (zslu / zu);
	    zetau = min(r__1,r__2);
/* Computing MIN */
	    r__1 = zetat, r__2 = ztmax / (zslt / zt);
	    zetat = min(r__1,r__2);
	    psmz = (real)r0044_(m, &zetau);
	    simm = (real)r0044_(m, &zetalu) - psmz + rlogu;
	    pshz = (real)r0046_(m, &zetat);
	    simh = (real)r0046_(m, &zetalt) - pshz + rlogt;
	}
    } else {
	if (*rlmo < rc0021) {
	    psmz = (real)r0039_(m, &zetau);
	    simm = (real)r0039_(m, &zetalu) - psmz + rlogu;
	    pshz = (real)r0041_(m, &zetat);
	    simh = (real)r0041_(m, &zetalt) - pshz + rlogt;
	} else {
	    zetalu = min(zetalu,ztmax);
	    zetalt = min(zetalt,ztmax);
	    psmz = (real)r0040_(m, &zetau, &rric);
	    simm = (real)r0040_(m, &zetalu, &rric) - psmz + rlogu;
	    pshz = (real)r0042_(m, &zetat, &rfac);
	    simh = (real)r0042_(m, &zetalt, &rfac) - pshz + rlogt;
	}
    }
/* Computing MAX */
    r__1 = (real)sqrt((real)(*akms * (real)sqrt((real)(du2 + *wstar2))));
    *ustar = max(r__1,epsust);
/* Computing MAX */
    r__1 = rc0020, r__2 = (real)exp((real)(zilfc * (real)sqrt((real)(*ustar *
	    *z0)))) * *z0;
    zt = max(r__1,r__2);
    zslt = *zlm + zt;
    rlogt = (real)log((real)(zslt / zt));
    ustark = *ustar * vkrm;
    if (simm < rc0020) {
	simm = rc0020;
    }
/* Computing MAX */
    r__1 = ustark / simm;
    *akms = max(r__1,cxch);
    if (simh < rc0020) {
	simh = rc0020;
    }
/* Computing MAX */
    r__1 = ustark / simh;
    *akhs = max(r__1,cxch);
    if (btgh * *akhs * dthv != rc0018) {
	d__1 = (r__1 = btgh * *akhs * dthv, (doublereal) abs(r__1));
	d__2 = (doublereal) (rc0007 / rc0019);
	*wstar2 = wwst2 * (real)pow_dd(&d__1, &d__2);
    } else {
	*wstar2 = rc0018;
    }
/* Computing 3rd power */
    r__1 = *ustar;
    rlmn = elfc * *akhs * dthv / (r__1 * (r__1 * r__1));
    rlma = *rlmo * wold + rlmn * wnew;
    *rlmo = rlma;
    return 0;
} /* r0034_ */

__device__ __forceinline__ /* Subroutine */ int r0035_(Model &m, real *t, real *esw, real *esi, real *desw, real *
	desi) {
    /* Initialized data */

    real rc0000 = 100.f;

    *esw = rc0000 * (6.107799961f + *t * (.4436518521f + *t * (.01428945805f
	    + *t * (2.650648471e-4f + *t * (3.031240396e-6f + *t * (
	    2.034080948e-8f + *t * 6.136820929e-11f))))));
    *esi = rc0000 * (*t * (*t * (*t * (*t * (*t * (*t * 1.838826904e-10f +
	    4.838803174e-8f) + 5.82472028e-6f) + 4.176223716e-4f) +
	    .01886013408f) + .503469897f) + 6.109177956f);
    *desw = rc0000 * (*t * (*t * (*t * (*t * (*t * (*t * -7.090244804e-13f +
	    3.53242181e-10f) + 1.036561403e-7f) + 1.215215065e-5f) +
	    7.93805404e-4f) + .02857002636f) + .4438099984f);
    *desi = rc0000 * (*t * (*t * (*t * (*t * (*t * (*t * 7.131097725e-12f +
	    2.158542548e-9f) + 3.005693132e-7f) + 2.477563108e-5f) +
	    .001267995369f) + .0377325502f) + .5030305237f);
    return 0;
} /* r0035_ */

__device__ __forceinline__ real r0036_(Model &m, real *t, real *tfrz) {
    /* Initialized data */

    real rc0000 = 50.f;

    /* System generated locals */
    real ret_val, r__1, r__2, r__3, r__4;

/* Computing MIN */
/* Computing MAX */
    r__3 = -rc0000, r__4 = *t - *tfrz;
    r__1 = rc0000, r__2 = max(r__3,r__4);
    ret_val = min(r__1,r__2);
    return ret_val;
} /* r0036_ */

__device__ __forceinline__ real r0037_(Model &m, real *ab, real *bc) {
    /* Initialized data */

    real rc0000 = 25.f;
    real rc0001 = 10.f;

    /* System generated locals */
    real ret_val;
    real d__1, d__2;

    /* Builtin functions */


    d__1 = (doublereal) (*ab);
    d__2 = (doublereal) ((*bc - rc0000) / rc0001);
    ret_val = (real)pow_dd(&d__1, &d__2);
    return ret_val;
} /* r0037_ */

__device__ __forceinline__ real r0038_(Model &m, real *ab) {
    /* Initialized data */

    real rc0000 = 1.f;
    real rc0001 = 2.2e5f;
    real rc0002 = 710.f;
    real rc0003 = 273.16f;
    real rc0004 = 8.314f;

    /* System generated locals */
    real ret_val;

    /* Builtin functions */


    ret_val = rc0000 + (real)exp((real)((-rc0001 + rc0002 * (*ab + rc0003)) /
	    (rc0004 * (*ab + rc0003))));
    return ret_val;
} /* r0038_ */

__device__ __forceinline__ real r0039_(Model &m, real *zz) {
    /* Initialized data */

    real rc0000 = .96f;
    real rc0001 = 1.f;
    real rc0002 = 4.5f;

    /* System generated locals */
    real ret_val;

    /* Builtin functions */


    ret_val = -rc0000 * (real)log((real)(rc0001 - rc0002 * *zz));
    return ret_val;
} /* r0039_ */

__device__ __forceinline__ real r0040_(Model &m, real *zz, real *rric) {
    /* Initialized data */

    real rc0000 = 2.076f;
    real rc0001 = 1.f;

    /* System generated locals */
    real ret_val;

    ret_val = *zz * *rric - rc0000 * (rc0001 - rc0001 / (*zz + rc0001));
    return ret_val;
} /* r0040_ */

__device__ __forceinline__ real r0041_(Model &m, real *zz) {
    /* Initialized data */

    real rc0000 = .96f;
    real rc0001 = 1.f;
    real rc0002 = 4.5f;

    /* System generated locals */
    real ret_val;

    /* Builtin functions */


    ret_val = -rc0000 * (real)log((real)(rc0001 - rc0002 * *zz));
    return ret_val;
} /* r0041_ */

__device__ __forceinline__ real r0042_(Model &m, real *zz, real *rfac) {
    /* Initialized data */

    real rc0000 = 2.076f;
    real rc0001 = 1.f;

    /* System generated locals */
    real ret_val;

    ret_val = *zz * *rfac - rc0000 * (rc0001 - rc0001 / (*zz + rc0001));
    return ret_val;
} /* r0042_ */

__device__ __forceinline__ real r0043_(Model &m, real *xx, real *pihf) {
    /* Initialized data */

    real rc0000 = 2.f;
    real rc0001 = 1.f;
    real rc0002 = .5f;

    /* System generated locals */
    real ret_val;

    /* Builtin functions */


    ret_val = -rc0000 * (real)log((real)((*xx + rc0001) * rc0002)) - (real)
	    log((real)((*xx * *xx + rc0001) * rc0002)) + rc0000 * (real)atan(*
	    xx) - *pihf;
    return ret_val;
} /* r0043_ */

__device__ __forceinline__ real r0044_(Model &m, real *yy) {
    /* Initialized data */

    real rc0000 = 5.f;

    /* System generated locals */
    real ret_val;

    ret_val = rc0000 * *yy;
    return ret_val;
} /* r0044_ */

__device__ __forceinline__ real r0045_(Model &m, real *xx) {
    /* Initialized data */

    real rc0000 = 2.f;
    real rc0001 = 1.f;
    real rc0002 = .5f;

    /* System generated locals */
    real ret_val;

    /* Builtin functions */


    ret_val = -rc0000 * (real)log((real)((*xx * *xx + rc0001) * rc0002));
    return ret_val;
} /* r0045_ */

__device__ __forceinline__ real r0046_(Model &m, real *yy) {
    /* Initialized data */

    real rc0000 = 5.f;

    /* System generated locals */
    real ret_val;

    ret_val = rc0000 * *yy;
    return ret_val;
} /* r0046_ */

__device__ __forceinline__ /* Subroutine */ int r0047_(Model &m, integer *ice, integer *isnow, real *ssoil, real *
	df, real *hcpct, real *sag, real *snowh, real *tg, real *stc) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = .5f;
    real rc0002 = 0.f;
    real rc0003 = 1.f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    real ai[7], bi[7], ci[7];
    integer iz, ixarr;
    real rhsts[7], eflxb, phi[7], tbeg[7], ssoil2, eflxb2;

    real err_est__, zbotsno;

    /* Parameter adjustments */
    stc -= -2;
    hcpct -= -2;
    df -= -2;

    /* Function Body */
    i__1 = m.f0000 - (*isnow + 1) + 1;
    for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	phi[*isnow + 1 + ixarr + 1] = rc0000;
    }
    zbotsno = m.f0066 - *snowh;
    i__1 = m.f0000;
    for (iz = *isnow + 1; iz <= i__1; ++iz) {
	tbeg[iz + 2] = stc[iz];
/* L9004: */
    }
/* L9005: */
    r0048_(m, &m.f0001, &m.f0000, isnow, &stc[-2], &zbotsno, &df[-2],
	     &hcpct[-2], ssoil, phi, ai, bi, ci, rhsts, &eflxb);
    r0049_(m, &m.f0001, &m.f0000, isnow, &m.f0005, ai, bi, ci,
	    rhsts, &stc[-2]);
    if (m.f0041 == 1) {
	eflxb2 = rc0000;
    } else if (m.f0041 == 2) {
	eflxb2 = df[m.f0000] * (m.f0197 - stc[m.f0000]) / (
		rc0001 * (m.f0028[m.f0000 + 1] + m.f0028[
		m.f0000 + 2]) - zbotsno);
    }
    return 0;
    err_est__ = rc0002;
    i__1 = m.f0000;
    for (iz = *isnow + 1; iz <= i__1; ++iz) {
	err_est__ += (stc[iz] - tbeg[iz + 2]) * m.f0027[iz + 2] * hcpct[
		iz] / m.f0005;
/* L9007: */
    }
/* L9008: */
    if (m.f0040 == 1 || m.f0040 == 3) {
	err_est__ -= *ssoil + eflxb;
    } else {
	ssoil2 = df[*isnow + 1] * (*tg - stc[*isnow + 1]) / (rc0001 *
		m.f0027[*isnow + 3]);
	err_est__ -= ssoil2 + eflxb2;
    }
    if (abs(err_est__) > rc0003) {
	m.status |= 1;
	m.status |= 1;
    }
    return 0;
} /* r0047_ */

__device__ __forceinline__ /* Subroutine */ int r0048_(Model &m, integer *nsnow, integer *nsoil, integer *isnow,
	real *stc, real *zbot, real *df, real *hcpct, real *ssoil, real *phi,
	real *ai, real *bi, real *ci, real *rhsts, real *botflx) {
    /* Initialized data */

    real rc0000 = 2.f;
    real rc0001 = 0.f;
    real rc0002 = .5f;
    real rc0003 = 0.f;

    /* System generated locals */
    integer stc_offset, df_offset, hcpct_offset, phi_offset, rhsts_offset,
	    ai_offset, bi_offset, ci_offset, i__1;

    /* Local variables */
    integer k;
    real ddz[7], denom[7], dtsdz[7], eflux[7], temp1;

    /* Parameter adjustments */
    rhsts_offset = -(*nsnow) + 1;
    rhsts -= rhsts_offset;
    ci_offset = -(*nsnow) + 1;
    ci -= ci_offset;
    bi_offset = -(*nsnow) + 1;
    bi -= bi_offset;
    ai_offset = -(*nsnow) + 1;
    ai -= ai_offset;
    phi_offset = -(*nsnow) + 1;
    phi -= phi_offset;
    hcpct_offset = -(*nsnow) + 1;
    hcpct -= hcpct_offset;
    df_offset = -(*nsnow) + 1;
    df -= df_offset;
    stc_offset = -(*nsnow) + 1;
    stc -= stc_offset;

    /* Function Body */
    i__1 = *nsoil;
    for (k = *isnow + 1; k <= i__1; ++k) {
	if (k == *isnow + 1) {
	    denom[k + 2] = -m.f0028[k + 2] * hcpct[k];
	    temp1 = -m.f0028[k + 3];
	    ddz[k + 2] = rc0000 / temp1;
	    dtsdz[k + 2] = rc0000 * (stc[k] - stc[k + 1]) / temp1;
	    eflux[k + 2] = df[k] * dtsdz[k + 2] - *ssoil - phi[k];
	} else if (k < *nsoil) {
	    denom[k + 2] = (m.f0028[k + 1] - m.f0028[k + 2]) *
		    hcpct[k];
	    temp1 = m.f0028[k + 1] - m.f0028[k + 3];
	    ddz[k + 2] = rc0000 / temp1;
	    dtsdz[k + 2] = rc0000 * (stc[k] - stc[k + 1]) / temp1;
	    eflux[k + 2] = df[k] * dtsdz[k + 2] - df[k - 1] * dtsdz[k + 1] -
		    phi[k];
	} else if (k == *nsoil) {
	    denom[k + 2] = (m.f0028[k + 1] - m.f0028[k + 2]) *
		    hcpct[k];
	    temp1 = m.f0028[k + 1] - m.f0028[k + 2];
	    if (m.f0041 == 1) {
		*botflx = rc0001;
	    }
	    if (m.f0041 == 2) {
		dtsdz[k + 2] = (stc[k] - m.f0176) / (rc0002 * (
			m.f0028[k + 1] + m.f0028[k + 2]) - *zbot);
		*botflx = -df[k] * dtsdz[k + 2];
	    }
	    eflux[k + 2] = -(*botflx) - df[k - 1] * dtsdz[k + 1] - phi[k];
	}
/* L9004: */
    }
/* L9005: */
    i__1 = *nsoil;
    for (k = *isnow + 1; k <= i__1; ++k) {
	if (k == *isnow + 1) {
	    ai[k] = rc0003;
	    ci[k] = -df[k] * ddz[k + 2] / denom[k + 2];
	    if (m.f0040 == 1 || m.f0040 == 3) {
		bi[k] = -ci[k];
	    }
	    if (m.f0040 == 2) {
		bi[k] = -ci[k] + df[k] / (rc0002 * m.f0028[k + 2] *
			m.f0028[k + 2] * hcpct[k]);
	    }
	} else if (k < *nsoil) {
	    ai[k] = -df[k - 1] * ddz[k + 1] / denom[k + 2];
	    ci[k] = -df[k] * ddz[k + 2] / denom[k + 2];
	    bi[k] = -(ai[k] + ci[k]);
	} else if (k == *nsoil) {
	    ai[k] = -df[k - 1] * ddz[k + 1] / denom[k + 2];
	    ci[k] = rc0003;
	    bi[k] = -(ai[k] + ci[k]);
	}
	rhsts[k] = eflux[k + 2] / (-denom[k + 2]);
/* L9007: */
    }
/* L9008: */
    return 0;
} /* r0048_ */

__device__ __forceinline__ /* Subroutine */ int r0049_(Model &m, integer *nsnow, integer *nsoil, integer *isnow,
	real *dt, real *ai, real *bi, real *ci, real *rhsts, real *stc) {
    /* Initialized data */

    real rc0000 = 1.f;

    /* System generated locals */
    integer rhsts_offset, ai_offset, bi_offset, ci_offset, stc_offset, i__1;

    /* Local variables */
    integer k;
    real ciin[7];

    real rhstsin[7];

    /* Parameter adjustments */
    stc_offset = -(*nsnow) + 1;
    stc -= stc_offset;
    rhsts_offset = -(*nsnow) + 1;
    rhsts -= rhsts_offset;
    ci_offset = -(*nsnow) + 1;
    ci -= ci_offset;
    bi_offset = -(*nsnow) + 1;
    bi -= bi_offset;
    ai_offset = -(*nsnow) + 1;
    ai -= ai_offset;

    /* Function Body */
    i__1 = *nsoil;
    for (k = *isnow + 1; k <= i__1; ++k) {
	rhsts[k] *= *dt;
	ai[k] *= *dt;
	bi[k] = rc0000 + bi[k] * *dt;
	ci[k] *= *dt;
/* L9004: */
    }
/* L9005: */
    i__1 = *nsoil;
    for (k = *isnow + 1; k <= i__1; ++k) {
	rhstsin[k + 2] = rhsts[k];
	ciin[k + 2] = ci[k];
/* L9007: */
    }
/* L9008: */
    i__1 = *isnow + 1;
    r0050_(m, &ai[ai_offset], &bi[bi_offset], rhstsin, &i__1, nsoil, nsnow, ciin,
	     &ci[ci_offset], &rhsts[rhsts_offset]);
    i__1 = *nsoil;
    for (k = *isnow + 1; k <= i__1; ++k) {
	stc[k] += ci[k];
/* L9010: */
    }
/* L9011: */
    return 0;
} /* r0049_ */

__device__ __forceinline__ /* Subroutine */ int r0050_(Model &m, real *a, real *b, real *d__, integer *ntop,
	integer *nsoil, integer *nsnow, real *c__, real *p, real *delta) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;

    /* System generated locals */
    integer a_offset, b_offset, d_offset, c_offset, p_offset, delta_offset,
	    i__1;

    /* Local variables */
    integer k, kk;

    /* Parameter adjustments */
    delta_offset = -(*nsnow) + 1;
    delta -= delta_offset;
    p_offset = -(*nsnow) + 1;
    p -= p_offset;
    c_offset = -(*nsnow) + 1;
    c__ -= c_offset;
    d_offset = -(*nsnow) + 1;
    d__ -= d_offset;
    b_offset = -(*nsnow) + 1;
    b -= b_offset;
    a_offset = -(*nsnow) + 1;
    a -= a_offset;

    /* Function Body */
    c__[(0 + (0 + (*nsoil << 2))) / 4] = rc0000;
    p[*ntop] = -c__[*ntop] / b[*ntop];
    delta[*ntop] = d__[*ntop] / b[*ntop];
    i__1 = *nsoil;
    for (k = *ntop + 1; k <= i__1; ++k) {
	p[k] = -c__[k] * (rc0001 / (b[k] + a[k] * p[k - 1]));
	delta[k] = (d__[k] - a[k] * delta[k - 1]) * (rc0001 / (b[k] + a[k] *
		p[k - 1]));
/* L9004: */
    }
/* L9005: */
    p[*nsoil] = delta[*nsoil];
    i__1 = *nsoil;
    for (k = *ntop + 1; k <= i__1; ++k) {
	kk = *nsoil - k + (*ntop - 1) + 1;
	p[kk] = p[kk] * p[kk + 1] + delta[kk];
/* L9007: */
    }
/* L9008: */
    return 0;
} /* r0050_ */

__device__ __forceinline__ /* Subroutine */ int r0051_(Model &m, integer *nsnow, integer *nsoil) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 0.f;
    real rc0002 = 1e3f;
    real rc0003 = 1.f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    integer j;
    real supercool[7], hm[7], xm[7], wmass0[7], wice0[7], wliq0[7], mice[7],
	    mliq[7], heatr, temp1, propor, smp, xmf;


    m.f0335 = rc0000;
    m.f0380 = rc0000;
    xmf = rc0000;
    i__1 = *nsoil;
    for (j = -(*nsnow) + 1; j <= i__1; ++j) {
	supercool[j + 2] = rc0001;
/* L9004: */
    }
/* L9005: */
    for (j = m.f0390 + 1; j <= 0; ++j) {
	mice[j + 2] = m.f0400[j + 2];
	mliq[j + 2] = m.f0401[j + 2];
/* L9007: */
    }
/* L9008: */
    i__1 = *nsoil;
    for (j = 1; j <= i__1; ++j) {
	mliq[j + 2] = m.f0394[j - 1] * m.f0027[j + 2] * rc0002;
	mice[j + 2] = (m.f0391[j - 1] - m.f0394[j - 1]) *
		m.f0027[j + 2] * rc0002;
/* L9010: */
    }
/* L9011: */
    i__1 = *nsoil;
    for (j = m.f0390 + 1; j <= i__1; ++j) {
	m.f0216[j + 2] = 0;
	hm[j + 2] = rc0000;
	xm[j + 2] = rc0000;
	wice0[j + 2] = mice[j + 2];
	wliq0[j + 2] = mliq[j + 2];
	wmass0[j + 2] = mice[j + 2] + mliq[j + 2];
/* L9013: */
    }
/* L9014: */
    if (m.f0025 == 1) {
	i__1 = *nsoil;
	for (j = 1; j <= i__1; ++j) {
	    if (m.f0042 == 1) {
		if (m.f0217[j + 2] < m.f0125) {
		    smp = m.f0128 * (m.f0125 - m.f0217[j +
			    2]) / (m.f0177 * m.f0217[j + 2]);
		    d__1 = (doublereal) (smp / m.f0052[j - 1]);
		    d__2 = (doublereal) (-rc0003 / m.f0046[j - 1]);
		    supercool[j + 2] = m.f0047[j - 1] * (real)pow_dd(&
			    d__1, &d__2);
		    supercool[j + 2] = supercool[j + 2] * m.f0027[j + 2]
			     * rc0002;
		}
	    }
	    if (m.f0042 == 2) {
		r0052_(m, &j, &m.f0217[j + 2], &m.f0391[j - 1], &
			m.f0394[j - 1], &supercool[j + 2]);
		supercool[j + 2] = supercool[j + 2] * m.f0027[j + 2] *
			rc0002;
	    }
/* L9016: */
	}
/* L9017: */
    }
    i__1 = *nsoil;
    for (j = m.f0390 + 1; j <= i__1; ++j) {
	if (mice[j + 2] > rc0000 && m.f0217[j + 2] >= m.f0125) {
	    m.f0216[j + 2] = 1;
	}
	if (mliq[j + 2] > supercool[j + 2] && m.f0217[j + 2] <
		m.f0125) {
	    m.f0216[j + 2] = 2;
	}
	if (m.f0390 == 0 && m.f0376 > rc0000 && j == 1) {
	    if (m.f0217[j + 2] >= m.f0125) {
		m.f0216[j + 2] = 1;
	    }
	}
/* L9019: */
    }
/* L9020: */
    i__1 = *nsoil;
    for (j = m.f0390 + 1; j <= i__1; ++j) {
	if (m.f0216[j + 2] > 0) {
	    hm[j + 2] = (m.f0217[j + 2] - m.f0125) /
		    m.f0220[j + 2];
	    m.f0217[j + 2] = m.f0125;
	}
	if (m.f0216[j + 2] == 1 && hm[j + 2] < rc0000) {
	    hm[j + 2] = rc0000;
	    m.f0216[j + 2] = 0;
	}
	if (m.f0216[j + 2] == 2 && hm[j + 2] > rc0000) {
	    hm[j + 2] = rc0000;
	    m.f0216[j + 2] = 0;
	}
	xm[j + 2] = hm[j + 2] * m.f0005 / m.f0128;
/* L9022: */
    }
/* L9023: */
    if (m.f0390 == 0 && m.f0376 > rc0000 && xm[3] > rc0000) {
	temp1 = m.f0376;
/* Computing MAX */
	r__1 = rc0000, r__2 = temp1 - xm[3];
	m.f0376 = max(r__1,r__2);
	propor = m.f0376 / temp1;
/* Computing MAX */
	r__1 = rc0000, r__2 = propor * m.f0375;
	m.f0375 = max(r__1,r__2);
	heatr = hm[3] - m.f0128 * (temp1 - m.f0376) /
		m.f0005;
	if (heatr > rc0000) {
	    xm[3] = heatr * m.f0005 / m.f0128;
	    hm[3] = heatr;
	} else {
	    xm[3] = rc0000;
	    hm[3] = rc0000;
	}
/* Computing MAX */
	r__1 = rc0000, r__2 = temp1 - m.f0376;
	m.f0335 = max(r__1,r__2) / m.f0005;
	xmf = m.f0128 * m.f0335;
	m.f0380 = temp1 - m.f0376;
    }
    i__1 = *nsoil;
    for (j = m.f0390 + 1; j <= i__1; ++j) {
	if (m.f0216[j + 2] > 0 && (r__1 = hm[j + 2], abs(r__1)) >
		rc0000) {
	    heatr = rc0000;
	    if (xm[j + 2] > rc0000) {
/* Computing MAX */
		r__1 = rc0000, r__2 = wice0[j + 2] - xm[j + 2];
		mice[j + 2] = max(r__1,r__2);
		heatr = hm[j + 2] - m.f0128 * (wice0[j + 2] - mice[j +
			2]) / m.f0005;
	    } else if (xm[j + 2] < rc0000) {
		if (j <= 0) {
/* Computing MIN */
		    r__1 = wmass0[j + 2], r__2 = wice0[j + 2] - xm[j + 2];
		    mice[j + 2] = min(r__1,r__2);
		} else {
		    if (wmass0[j + 2] < supercool[j + 2]) {
			mice[j + 2] = rc0000;
		    } else {
/* Computing MIN */
			r__1 = wmass0[j + 2] - supercool[j + 2], r__2 = wice0[
				j + 2] - xm[j + 2];
			mice[j + 2] = min(r__1,r__2);
/* Computing MAX */
			r__1 = mice[j + 2];
			mice[j + 2] = max(r__1,rc0001);
		    }
		}
		heatr = hm[j + 2] - m.f0128 * (wice0[j + 2] - mice[j +
			2]) / m.f0005;
	    }
/* Computing MAX */
	    r__1 = rc0000, r__2 = wmass0[j + 2] - mice[j + 2];
	    mliq[j + 2] = max(r__1,r__2);
	    if (abs(heatr) > rc0000) {
		m.f0217[j + 2] += m.f0220[j + 2] * heatr;
		if (j <= 0) {
		    if (mliq[j + 2] * mice[j + 2] > rc0000) {
			m.f0217[j + 2] = m.f0125;
		    }
		    if (mice[j + 2] == rc0000) {
			m.f0217[j + 2] = m.f0125;
			hm[j + 3] += heatr;
			xm[j + 3] = hm[j + 3] * m.f0005 / m.f0128;
		    }
		}
	    }
	    xmf += m.f0128 * (wice0[j + 2] - mice[j + 2]) /
		    m.f0005;
	    if (j < 1) {
/* Computing MAX */
		r__1 = rc0000, r__2 = wice0[j + 2] - mice[j + 2];
		m.f0335 += max(r__1,r__2) / m.f0005;
	    }
	}
/* L9025: */
    }
/* L9026: */
    for (j = m.f0390 + 1; j <= 0; ++j) {
	m.f0401[j + 2] = mliq[j + 2];
	m.f0400[j + 2] = mice[j + 2];
/* L9028: */
    }
/* L9029: */
    i__1 = *nsoil;
    for (j = 1; j <= i__1; ++j) {
	m.f0394[j - 1] = mliq[j + 2] / (rc0002 * m.f0027[j + 2]);
	m.f0391[j - 1] = (mliq[j + 2] + mice[j + 2]) / (rc0002 *
		m.f0027[j + 2]);
/* L9031: */
    }
/* L9032: */
    return 0;
} /* r0051_ */

__device__ __forceinline__ /* Subroutine */ int r0052_(Model &m, integer *isoil, real *tkelv, real *smc, real *
	sh2o, real *free) {
    /* Initialized data */

    real rc0000 = 8.f;
    real rc0001 = 5.5f;
    real rc0002 = .005f;
    real rc0003 = 920.f;
    real rc0004 = .001f;
    real rc0005 = 0.f;
    real rc0006 = .02f;
    real rc0007 = 0.f;
    real rc0008 = 1.f;
    real rc0009 = 2.f;

    /* System generated locals */
    real r__1;
    real d__1, d__2, d__3, d__4;

    /* Builtin functions */


    /* Local variables */
    real df, ck, fk, bx, denom, dswl, swl, swlk;
    integer nlog, kcount;
    real blim, error, dice;

    ck = rc0000;
    blim = rc0001;
    error = rc0002;
    dice = rc0003;
    bx = m.f0046[*isoil - 1];
    if (m.f0046[*isoil - 1] > blim) {
	bx = blim;
    }
    nlog = 0;
    kcount = 0;
    if (*tkelv > m.f0125 - rc0004) {
	*free = *smc;
    } else {
	if (ck != rc0005) {
	    swl = *smc - *sh2o;
	    if (swl > *smc - rc0006) {
		swl = *smc - rc0006;
	    }
	    if (swl < rc0007) {
		swl = rc0007;
	    }
L1001:
	    if (! (nlog < 10 && kcount == 0)) {
		goto L1002;
	    }
	    ++nlog;
	    d__1 = (doublereal) (rc0008 + ck * swl);
	    d__2 = (doublereal) rc0009;
	    d__3 = (doublereal) (m.f0047[*isoil - 1] / (*smc - swl));
	    d__4 = (doublereal) bx;
	    df = (real)log((real)(m.f0052[*isoil - 1] * m.f0177 /
		    m.f0128 * (real)pow_dd(&d__1, &d__2) * (real)pow_dd(
		    &d__3, &d__4))) - (real)log((real)(-(*tkelv -
		    m.f0125) / *tkelv));
	    denom = rc0009 * ck / (rc0008 + ck * swl) + bx / (*smc - swl);
	    swlk = swl - df / denom;
	    if (swlk > *smc - rc0006) {
		swlk = *smc - rc0006;
	    }
	    if (swlk < rc0007) {
		swlk = rc0007;
	    }
	    dswl = (r__1 = swlk - swl, abs(r__1));
	    swl = swlk;
	    if (dswl <= error) {
		++kcount;
	    }
	    goto L1001;
L1002:
	    *free = *smc - swl;
	}
	if (kcount == 0) {
	    m.status |= 1;
	    d__1 = (doublereal) (m.f0128 / (m.f0177 * (
		    -m.f0052[*isoil - 1])) * ((*tkelv - m.f0125) /
		     *tkelv));
	    d__2 = (doublereal) (-1 / bx);
	    fk = (real)pow_dd(&d__1, &d__2) * m.f0047[*isoil - 1];
	    if (fk < rc0006) {
		fk = rc0006;
	    }
	    *free = min(fk,*smc);
	}
    }
    return 0;
} /* r0052_ */

__device__ __forceinline__ /* Subroutine */ int r0053_(Model &m) {
    /* Initialized data */

    real rc0000 = 999999.f;
    real rc0001 = 0.f;
    real rc0002 = 0.f;
    real rc0003 = 5e3f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer iz, ixarr;
    real red001, red002, bdsnow, realmissing;


    realmissing = -rc0000;
    m.f0349 = rc0001;
    m.f0381 = rc0001;
    m.f0382 = rc0001;
    r0054_(m);
    if (m.f0390 < 0) {
	r0059_(m);
    }
    if (m.f0390 < 0) {
	r0056_(m);
    }
    if (m.f0390 < 0) {
	r0057_(m);
    }
    r0055_(m);
    i__1 = m.f0390;
    for (iz = -m.f0001 + 1; iz <= i__1; ++iz) {
	m.f0400[iz + 2] = rc0002;
	m.f0401[iz + 2] = rc0002;
	m.f0217[iz + 2] = rc0002;
	m.f0027[iz + 2] = rc0002;
	m.f0028[iz + 2] = rc0002;
/* L9004: */
    }
/* L9005: */
    if (m.f0376 > rc0003) {
	bdsnow = m.f0400[2] / m.f0027[2];
	m.f0349 = m.f0376 - rc0003;
	m.f0400[2] -= m.f0349;
	m.f0027[2] -= m.f0349 / bdsnow;
	m.f0349 /= m.f0005;
    }
    if (m.f0390 < 0) {
	m.f0376 = rc0002;
	for (iz = m.f0390 + 1; iz <= 0; ++iz) {
	    m.f0376 = m.f0376 + m.f0400[iz + 2] +
		    m.f0401[iz + 2];
/* L9007: */
	}
/* L9008: */
    }
    for (iz = m.f0390 + 1; iz <= 0; ++iz) {
	m.f0027[iz + 2] = -m.f0027[iz + 2];
/* L9010: */
    }
/* L9011: */
    m.f0027[3] = m.f0026[0];
    i__1 = m.f0000;
    for (iz = 2; iz <= i__1; ++iz) {
	m.f0027[iz + 2] = m.f0026[iz - 1] - m.f0026[iz - 2];
/* L9013: */
    }
/* L9014: */
    m.f0028[m.f0390 + 3] = m.f0027[m.f0390 + 3];
    i__1 = m.f0000;
    for (iz = m.f0390 + 2; iz <= i__1; ++iz) {
	m.f0028[iz + 2] = m.f0028[iz + 1] + m.f0027[iz + 2];
/* L9016: */
    }
/* L9017: */
    i__1 = m.f0000;
    for (iz = m.f0390 + 1; iz <= i__1; ++iz) {
	m.f0027[iz + 2] = -m.f0027[iz + 2];
/* L9019: */
    }
/* L9020: */
    red001 = rc0001;
    i__1 = -(-m.f0001 + 1) + 1;
    for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	red001 = red001 + m.f0400[-m.f0001 + 1 + ixarr + 1] +
		m.f0401[-m.f0001 + 1 + ixarr + 1];
    }
    if (red001 > 0.f) {
	red001 = rc0001;
	i__1 = -(-m.f0001 + 1) + 1;
	for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	    red001 += m.f0217[-m.f0001 + 1 + ixarr + 1] * (
		    m.f0400[-m.f0001 + 1 + ixarr + 1] +
		    m.f0401[-m.f0001 + 1 + ixarr + 1]);
	}
	red002 = rc0001;
	i__1 = -(-m.f0001 + 1) + 1;
	for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	    red002 = red002 + m.f0400[-m.f0001 + 1 + ixarr + 1] +
		    m.f0401[-m.f0001 + 1 + ixarr + 1];
	}
	m.f0221 = red001 / red002;
    } else {
	m.f0221 = realmissing;
    }
    return 0;
} /* r0053_ */

__device__ __forceinline__ /* Subroutine */ int r0054_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = .025f;
    real rc0002 = 273.16f;

    integer newnode;

    newnode = 0;
    if (m.f0390 == 0 && m.f0369 > rc0000) {
	m.f0375 += m.f0370 * m.f0005;
	m.f0376 += m.f0369 * m.f0005;
    }
    if (m.f0390 == 0 && m.f0369 > 0.f && m.f0375 >= rc0001)
	    {
	m.f0390 = -1;
	newnode = 1;
	m.f0027[2] = m.f0375;
	m.f0375 = rc0000;
	m.f0217[2] = min(rc0002,m.f0181);
	m.f0400[2] = m.f0376;
	m.f0401[2] = rc0000;
    }
    if (m.f0390 < 0 && newnode == 0 && m.f0369 > rc0000) {
	m.f0400[m.f0390 + 3] += m.f0369 * m.f0005;
	m.f0027[m.f0390 + 3] += m.f0370 * m.f0005;
    }
    return 0;
} /* r0054_ */

__device__ __forceinline__ /* Subroutine */ int r0055_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1e3f;
    real rc0002 = 500.f;
    real rc0003 = 50.f;
    real rc0004 = 1e-8f;
    real rc0005 = 1e-6f;
    real rc0006 = 0.f;
    real rc0007 = 1.f;
    real rc0008 = 1.f;

    /* System generated locals */
    real r__1, r__2, r__3, r__4;

    /* Local variables */
    integer j;
    real qin, qout, wgdif, epore[3], propor, temp, vol_ice__[3];

    real vol_liq__[3];

    if (m.f0376 == rc0000) {
	m.f0393[0] += (m.f0373 - m.f0374) * m.f0005 /
		(m.f0027[3] * rc0001);
	if (m.f0393[0] < rc0000) {
	    m.f0394[0] += m.f0393[0];
	    m.f0393[0] = rc0000;
	}
    }
    if (m.f0390 == 0 && m.f0376 > rc0000) {
	temp = m.f0376;
	m.f0376 = m.f0376 - m.f0374 * m.f0005 +
		m.f0373 * m.f0005;
	propor = m.f0376 / temp;
/* Computing MAX */
	r__1 = rc0000, r__2 = propor * m.f0375;
	m.f0375 = max(r__1,r__2);
/* Computing MIN */
/* Computing MAX */
	r__3 = m.f0375, r__4 = m.f0376 / rc0002;
	r__1 = max(r__3,r__4), r__2 = m.f0376 / rc0003;
	m.f0375 = min(r__1,r__2);
	if (m.f0376 < rc0000) {
	    m.f0393[0] += m.f0376 / (m.f0027[3] * rc0001);
	    m.f0376 = rc0000;
	    m.f0375 = rc0000;
	}
	if (m.f0393[0] < rc0000) {
	    m.f0394[0] += m.f0393[0];
	    m.f0393[0] = rc0000;
	}
    }
    if (m.f0375 <= rc0004 || m.f0376 <= rc0005) {
	m.f0375 = rc0006;
	m.f0376 = rc0006;
    }
    if (m.f0390 < 0) {
	wgdif = m.f0400[m.f0390 + 3] - m.f0374 *
		m.f0005 + m.f0373 * m.f0005;
	m.f0400[m.f0390 + 3] = wgdif;
	if (wgdif < rc0005 && m.f0390 < 0) {
	    r0056_(m);
	}
	if (m.f0390 < 0) {
	    m.f0401[m.f0390 + 3] += m.f0368 * m.f0005;
/* Computing MAX */
	    r__1 = rc0000, r__2 = m.f0401[m.f0390 + 3];
	    m.f0401[m.f0390 + 3] = max(r__1,r__2);
	}
    }
    for (j = m.f0390 + 1; j <= 0; ++j) {
/* Computing MIN */
	r__1 = rc0007, r__2 = m.f0400[j + 2] / (m.f0027[j + 2] *
		m.f0138);
	vol_ice__[j + 2] = min(r__1,r__2);
	epore[j + 2] = rc0007 - vol_ice__[j + 2];
/* L9004: */
    }
/* L9005: */
    qin = rc0000;
    qout = rc0000;
    for (j = m.f0390 + 1; j <= 0; ++j) {
	m.f0401[j + 2] += qin;
	vol_liq__[j + 2] = m.f0401[j + 2] / (m.f0027[j + 2] *
		m.f0137);
/* Computing MAX */
	r__1 = rc0000, r__2 = (vol_liq__[j + 2] - m.f0142 * epore[j + 2]
		) * m.f0027[j + 2];
	qout = max(r__1,r__2);
	if (j == 0) {
/* Computing MAX */
	    r__1 = (vol_liq__[j + 2] - epore[j + 2]) * m.f0027[j + 2],
		    r__2 = m.f0169 * m.f0005 * qout;
	    qout = max(r__1,r__2);
	}
	qout *= m.f0137;
	m.f0401[j + 2] -= qout;
	if (m.f0401[j + 2] / (m.f0400[j + 2] + m.f0401[j +
		2]) > m.f0168) {
	    qout += m.f0401[j + 2] - m.f0168 / (rc0008 -
		    m.f0168) * m.f0400[j + 2];
	    m.f0401[j + 2] = m.f0168 / (rc0008 - m.f0168) *
		    m.f0400[j + 2];
	}
	qin = qout;
/* L9007: */
    }
/* L9008: */
    for (j = m.f0390 + 1; j <= 0; ++j) {
/* Computing MAX */
	r__1 = m.f0027[j + 2], r__2 = m.f0401[j + 2] /
		m.f0137 + m.f0400[j + 2] / m.f0138;
	m.f0027[j + 2] = max(r__1,r__2);
/* L9010: */
    }
/* L9011: */
    m.f0379 = qout / m.f0005;
    return 0;
} /* r0055_ */

__device__ __forceinline__ /* Subroutine */ int r0056_(Model &m) {
    /* Initialized data */

    real dzmin[3] = { .025f,.025f,.1f };
    real rc0000 = .1f;
    real rc0001 = 0.f;
    real rc0002 = 0.f;
    real rc0003 = 1e3f;
    real rc0004 = .025f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    integer i__, j, k, l, isnow_old__, mssi, neibor;
    real zwice, zwliq;


    isnow_old__ = m.f0390;
    for (j = isnow_old__ + 1; j <= 0; ++j) {
	if (m.f0400[j + 2] <= rc0000) {
	    if (j != 0) {
		m.f0401[j + 3] += m.f0401[j + 2];
		m.f0400[j + 3] += m.f0400[j + 2];
		m.f0027[j + 3] += m.f0027[j + 2];
	    } else {
		if (isnow_old__ < -1) {
		    m.f0401[j + 1] += m.f0401[j + 2];
		    m.f0400[j + 1] += m.f0400[j + 2];
		    m.f0027[j + 1] += m.f0027[j + 2];
		} else {
		    if (m.f0400[j + 2] >= rc0001) {
			m.f0381 = m.f0401[j + 2];
			m.f0376 = m.f0400[j + 2];
			m.f0375 = m.f0027[j + 2];
		    } else {
			m.f0381 = m.f0401[j + 2] + m.f0400[
				j + 2];
			if (m.f0381 < rc0001) {
/* Computing MAX */
			    r__1 = rc0002, r__2 = m.f0393[0] +
				    m.f0381 / (m.f0027[3] *
				    rc0003);
			    m.f0393[0] = max(r__1,r__2);
			    m.f0381 = rc0002;
			}
			m.f0376 = rc0002;
			m.f0375 = rc0002;
		    }
		    m.f0401[j + 2] = rc0002;
		    m.f0400[j + 2] = rc0002;
		    m.f0027[j + 2] = rc0002;
		}
	    }
	    if (j > m.f0390 + 1 && m.f0390 < -1) {
		i__1 = m.f0390 + 2;
		for (i__ = j; i__ >= i__1; --i__) {
		    m.f0217[i__ + 2] = m.f0217[i__ + 1];
		    m.f0401[i__ + 2] = m.f0401[i__ + 1];
		    m.f0400[i__ + 2] = m.f0400[i__ + 1];
		    m.f0027[i__ + 2] = m.f0027[i__ + 1];
/* L9007: */
		}
/* L9008: */
	    }
	    ++m.f0390;
	}
/* L9004: */
    }
/* L9005: */
    if (m.f0393[0] < rc0001) {
	m.f0394[0] += m.f0393[0];
	m.f0393[0] = rc0001;
    }
    if (m.f0390 == 0) {
	return 0;
    }
    m.f0376 = rc0001;
    m.f0375 = rc0001;
    zwice = rc0001;
    zwliq = rc0001;
    for (j = m.f0390 + 1; j <= 0; ++j) {
	m.f0376 = m.f0376 + m.f0400[j + 2] + m.f0401[
		j + 2];
	m.f0375 += m.f0027[j + 2];
	zwice += m.f0400[j + 2];
	zwliq += m.f0401[j + 2];
/* L9010: */
    }
/* L9011: */
    if (m.f0375 < rc0004 && m.f0390 < 0) {
	m.f0390 = 0;
	m.f0376 = zwice;
	m.f0382 = zwliq;
	if (m.f0376 <= rc0001) {
	    m.f0375 = rc0001;
	}
    }
    if (m.f0390 < -1) {
	isnow_old__ = m.f0390;
	mssi = 1;
	for (i__ = isnow_old__ + 1; i__ <= 0; ++i__) {
	    if (m.f0027[i__ + 2] < dzmin[mssi - 1]) {
		if (i__ == m.f0390 + 1) {
		    neibor = i__ + 1;
		} else if (i__ == 0) {
		    neibor = i__ - 1;
		} else {
		    neibor = i__ + 1;
		    if (m.f0027[i__ + 1] + m.f0027[i__ + 2] <
			    m.f0027[i__ + 3] + m.f0027[i__ + 2]) {
			neibor = i__ - 1;
		    }
		}
		if (neibor > i__) {
		    j = neibor;
		    l = i__;
		} else {
		    j = i__;
		    l = neibor;
		}
		r0058_(m, &m.f0027[j + 2], &m.f0401[j + 2], &
			m.f0400[j + 2], &m.f0217[j + 2], &
			m.f0027[l + 2], &m.f0401[l + 2], &
			m.f0400[l + 2], &m.f0217[l + 2]);
		if (j - 1 > m.f0390 + 1) {
		    i__1 = m.f0390 + 2;
		    for (k = j - 1; k >= i__1; --k) {
			m.f0217[k + 2] = m.f0217[k + 1];
			m.f0400[k + 2] = m.f0400[k + 1];
			m.f0401[k + 2] = m.f0401[k + 1];
			m.f0027[k + 2] = m.f0027[k + 1];
/* L9016: */
		    }
/* L9017: */
		}
		++m.f0390;
		if (m.f0390 >= -1) {
		    goto L9014;
		}
	    } else {
		++mssi;
	    }
/* L9013: */
	}
L9014:
	;
    }
    return 0;
} /* r0056_ */

__device__ __forceinline__ /* Subroutine */ int r0057_(Model &m) {
    /* Initialized data */

    real rc0000 = .05f;
    real rc0001 = 2.f;
    real rc0002 = .2f;
    real rc0003 = .2f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer j;
    real dz[3];
    integer msno;
    real drr, swice[3], swliq[3], tsno[3], zwice, zwliq, propor, dtdz;


    i__1 = m.f0001;
    for (j = 1; j <= i__1; ++j) {
	if (j <= abs(m.f0390)) {
	    dz[j - 1] = m.f0027[j + m.f0390 + 2];
	    swice[j - 1] = m.f0400[j + m.f0390 + 2];
	    swliq[j - 1] = m.f0401[j + m.f0390 + 2];
	    tsno[j - 1] = m.f0217[j + m.f0390 + 2];
	}
/* L9004: */
    }
/* L9005: */
    msno = abs(m.f0390);
    if (msno == 1) {
	if (dz[0] > rc0000) {
	    msno = 2;
	    dz[0] /= rc0001;
	    swice[0] /= rc0001;
	    swliq[0] /= rc0001;
	    dz[1] = dz[0];
	    swice[1] = swice[0];
	    swliq[1] = swliq[0];
	    tsno[1] = tsno[0];
	}
    }
    if (msno > 1) {
	if (dz[0] > rc0000) {
	    drr = dz[0] - rc0000;
	    propor = drr / dz[0];
	    zwice = propor * swice[0];
	    zwliq = propor * swliq[0];
	    propor = rc0000 / dz[0];
	    swice[0] = propor * swice[0];
	    swliq[0] = propor * swliq[0];
	    dz[0] = rc0000;
	    r0058_(m, &dz[1], &swliq[1], &swice[1], &tsno[1], &drr, &zwliq, &
		    zwice, tsno);
	    if (msno <= 2 && dz[1] > rc0002) {
		msno = 3;
		dtdz = (tsno[0] - tsno[1]) / ((dz[0] + dz[1]) / rc0001);
		dz[1] /= rc0001;
		swice[1] /= rc0001;
		swliq[1] /= rc0001;
		dz[2] = dz[1];
		swice[2] = swice[1];
		swliq[2] = swliq[1];
		tsno[2] = tsno[1] - dtdz * dz[1] / rc0001;
		if (tsno[2] >= m.f0125) {
		    tsno[2] = tsno[1];
		} else {
		    tsno[1] += dtdz * dz[1] / rc0001;
		}
	    }
	}
    }
    if (msno > 2) {
	if (dz[1] > rc0003) {
	    drr = dz[1] - rc0003;
	    propor = drr / dz[1];
	    zwice = propor * swice[1];
	    zwliq = propor * swliq[1];
	    propor = rc0003 / dz[1];
	    swice[1] = propor * swice[1];
	    swliq[1] = propor * swliq[1];
	    dz[1] = rc0003;
	    r0058_(m, &dz[2], &swliq[2], &swice[2], &tsno[2], &drr, &zwliq, &
		    zwice, &tsno[1]);
	}
    }
    m.f0390 = -msno;
    for (j = m.f0390 + 1; j <= 0; ++j) {
	m.f0027[j + 2] = dz[j - m.f0390 - 1];
	m.f0400[j + 2] = swice[j - m.f0390 - 1];
	m.f0401[j + 2] = swliq[j - m.f0390 - 1];
	m.f0217[j + 2] = tsno[j - m.f0390 - 1];
/* L9007: */
    }
/* L9008: */
    return 0;
} /* r0057_ */

__device__ __forceinline__ /* Subroutine */ int r0058_(Model &m, real *dz, real *wliq, real *wice, real *t, real *
	dz2, real *wliq2, real *wice2, real *t2) {
    /* Initialized data */

    real rc0000 = 0.f;

    real h__, h2, hc, tc, dzc, wliqc, wicec;

    dzc = *dz + *dz2;
    wicec = *wice + *wice2;
    wliqc = *wliq + *wliq2;
    h__ = (m.f0130 * *wice + m.f0129 * *wliq) * (*t -
	    m.f0125) + m.f0128 * *wliq;
    h2 = (m.f0130 * *wice2 + m.f0129 * *wliq2) * (*t2 -
	    m.f0125) + m.f0128 * *wliq2;
    hc = h__ + h2;
    if (hc < rc0000) {
	tc = m.f0125 + hc / (m.f0130 * wicec + m.f0129 *
		wliqc);
    } else if (hc <= m.f0128 * wliqc) {
	tc = m.f0125;
    } else {
	tc = m.f0125 + (hc - m.f0128 * wliqc) / (m.f0130 *
		wicec + m.f0129 * wliqc);
    }
    *dz = dzc;
    *wice = wicec;
    *wliq = wliqc;
    *t = tc;
    return 0;
} /* r0058_ */

__device__ __forceinline__ /* Subroutine */ int r0059_(Model &m) {
    /* Initialized data */

    real rc0000 = .021f;
    real rc0001 = 2.5e-6f;
    real rc0002 = .04f;
    real rc0003 = 2.f;
    real rc0004 = 100.f;
    real rc0005 = 8e5f;
    real rc0006 = 0.f;
    real rc0007 = 1.f;
    real rc0008 = .001f;
    real rc0009 = .1f;
    real rc0010 = 0.f;
    real rc0011 = .046f;
    real rc0012 = .01f;
    real rc0013 = .5f;
    real rc0014 = .08f;
    real rc0015 = 1e-6f;

    /* System generated locals */
    real r__1, r__2, r__3, r__4;

    /* Builtin functions */


    /* Local variables */
    integer j;
    real c2, c3, c4, c5, bi, dm, td, wx, eta0, burden, ddz1, ddz2, ddz3,
	    dexpf, pdzdtc, void__;

    c2 = rc0000;
    c3 = rc0001;
    c4 = rc0002;
    c5 = rc0003;
    dm = rc0004;
    eta0 = rc0005;
    burden = rc0006;
    for (j = m.f0390 + 1; j <= 0; ++j) {
	wx = m.f0400[j + 2] + m.f0401[j + 2];
	m.f0404[j + 2] = m.f0400[j + 2] / wx;
	void__ = rc0007 - (m.f0400[j + 2] / m.f0138 +
		m.f0401[j + 2] / m.f0137) / m.f0027[j + 2];
	if (void__ > rc0008 && m.f0400[j + 2] > rc0009) {
	    bi = m.f0400[j + 2] / m.f0027[j + 2];
/* Computing MAX */
	    r__1 = rc0010, r__2 = m.f0125 - m.f0217[j + 2];
	    td = max(r__1,r__2);
	    dexpf = (real)exp((real)(-c4 * td));
	    ddz1 = -c3 * dexpf;
	    if (bi > dm) {
		ddz1 *= (real)exp((real)(-rc0011 * (bi - dm)));
	    }
	    if (m.f0401[j + 2] > rc0012 * m.f0027[j + 2]) {
		ddz1 *= c5;
	    }
	    ddz2 = -(burden + rc0013 * wx) * (real)exp((real)(-rc0014 * td -
		    c2 * bi)) / eta0;
	    if (m.f0216[j + 2] == 1) {
/* Computing MAX */
/* Computing MAX */
		r__3 = rc0015, r__4 = m.f0399[j + 2];
		r__1 = rc0010, r__2 = (m.f0399[j + 2] - m.f0404[j
			+ 2]) / max(r__3,r__4);
		ddz3 = max(r__1,r__2);
		ddz3 = -ddz3 / m.f0005;
	    } else {
		ddz3 = rc0010;
	    }
	    pdzdtc = (ddz1 + ddz2 + ddz3) * m.f0005;
/* Computing MAX */
	    r__1 = -rc0013;
	    pdzdtc = max(r__1,pdzdtc);
	    m.f0027[j + 2] *= rc0007 + pdzdtc;
/* Computing MAX */
	    r__1 = m.f0027[j + 2], r__2 = m.f0400[j + 2] /
		    m.f0138 + m.f0401[j + 2] / m.f0137;
	    m.f0027[j + 2] = max(r__1,r__2);
	}
	burden += wx;
/* L9004: */
    }
/* L9005: */
    return 0;
} /* r0059_ */

__device__ __forceinline__ /* Subroutine */ int r0060_(Model &m) {
    /* Initialized data */

    real rc0000 = 4.f;
    real rc0001 = 0.f;
    real rc0002 = 1e-4f;
    real rc0003 = 0.f;
    real rc0004 = 1.f;
    real rc0005 = 1.f;
    real rc0006 = .95f;
    real rc0007 = 1e-6f;
    real rc0008 = 1e3f;
    real rc0009 = .01f;
    real rc0010 = .001f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */


    /* Local variables */
    real a;
    integer k;
    real ai[4], bi[4], ci[4];
    integer iz;
    real xs;
    integer iter;
    real dtfine, rhstt[4], fice, wplus, rsat, sh2omin, wtsub, mh2o, mliq[4],
	    watmin, qdrain_save__, runsrf_save__, epore;
    integer niter;
    real sice_smc_ratio__;


    a = rc0000;
    if (m.f0045 == 1) {
	m.f0342 = rc0001;
	m.f0343 = rc0001;
	m.f0350 = rc0001;
	m.f0387 = m.f0005;
	rsat = rc0001;
	i__1 = m.f0000;
	for (k = 1; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = rc0002, r__2 = m.f0047[k - 1] - m.f0393[k - 1];
	    epore = max(r__1,r__2);
/* Computing MAX */
	    r__1 = rc0003, r__2 = m.f0394[k - 1] - epore;
	    rsat += max(r__1,r__2) * m.f0027[k + 2];
/* Computing MIN */
	    r__1 = epore, r__2 = m.f0394[k - 1];
	    m.f0394[k - 1] = min(r__1,r__2);
/* L9004: */
	}
/* L9005: */
	i__1 = m.f0000;
	for (k = 1; k <= i__1; ++k) {
/* Computing MIN */
	    r__1 = rc0004, r__2 = m.f0393[k - 1] / m.f0047[k - 1];
	    fice = min(r__1,r__2);
/* Computing MAX */
	    r__1 = rc0001, r__2 = (real)exp((real)(-a * (rc0005 - fice))) - (
		    real)exp((real)(-a));
	    m.f0398[k - 1] = max(r__1,r__2) / (rc0004 - (real)exp((real)
		    (-a)));
/* L9007: */
	}
/* L9008: */
	m.f0352 = rc0001;
	m.f0348 = rc0001;
	sh2omin = m.f0047[0];
	i__1 = m.f0000;
	for (k = 1; k <= i__1; ++k) {
	    if (m.f0393[k - 1] > m.f0352) {
		m.f0352 = m.f0393[k - 1];
	    }
	    if (m.f0398[k - 1] > m.f0348) {
		m.f0348 = m.f0398[k - 1];
	    }
	    if (m.f0394[k - 1] < sh2omin) {
		sh2omin = m.f0394[k - 1];
	    }
/* L9010: */
	}
/* L9011: */
	if (m.f0031 == 2) {
	    r0084_(m);
	}
	if (m.f0072) {
	    m.f0398[0] = rc0006;
	}
	m.f0351 = rc0007;
	r0066_(m);
	niter = 1;
	niter = 3;
	if (m.f0350 * m.f0005 > m.f0027[3] * m.f0047[
		0]) {
	    niter <<= 1;
	}
	dtfine = m.f0005 / niter;
	m.f0387 = dtfine;
	m.f0351 = rc0007;
	qdrain_save__ = rc0001;
	runsrf_save__ = rc0001;
	i__1 = niter;
	for (iter = 1; iter <= i__1; ++iter) {
	    if (m.f0339 > rc0003 && (m.f0030 == 3 ||
		    m.f0030 == 6 || m.f0030 == 7 || m.f0030
		    == 8)) {
		r0066_(m);
	    }
	    r0061_(m, &dtfine, rhstt, ai, bi, ci);
	    r0062_(m, &dtfine, ai, bi, ci, rhstt, &wplus);
	    rsat += wplus;
	    qdrain_save__ += m.f0344;
	    runsrf_save__ += m.f0342;
/* L9013: */
	}
/* L9014: */
	m.f0344 = qdrain_save__ / niter;
	m.f0342 = runsrf_save__ / niter;
	m.f0342 = m.f0342 * rc0008 + rsat * rc0008 /
		m.f0005;
	m.f0344 *= rc0008;
	if (m.f0031 == 2) {
	    wtsub = rc0003;
	    i__1 = m.f0000;
	    for (k = 1; k <= i__1; ++k) {
		wtsub += m.f0397[k - 1] * m.f0027[k + 2];
/* L9016: */
	    }
/* L9017: */
	    i__1 = m.f0000;
	    for (k = 1; k <= i__1; ++k) {
		mh2o = m.f0343 * m.f0005 * (m.f0397[k - 1] *
			 m.f0027[k + 2]) / wtsub;
		m.f0394[k - 1] -= mh2o / (m.f0027[k + 2] * rc0008)
			;
/* L9019: */
	    }
/* L9020: */
	}
	if (m.f0031 != 1) {
	    i__1 = m.f0000;
	    for (iz = 1; iz <= i__1; ++iz) {
		mliq[iz - 1] = m.f0394[iz - 1] * m.f0027[iz + 2] *
			 rc0008;
/* L9022: */
	    }
/* L9023: */
	    watmin = rc0009;
	    i__1 = m.f0000 - 1;
	    for (iz = 1; iz <= i__1; ++iz) {
		if (mliq[iz - 1] < rc0003) {
		    xs = watmin - mliq[iz - 1];
		} else {
		    xs = rc0003;
		}
		mliq[iz - 1] += xs;
		mliq[iz] -= xs;
/* L9025: */
	    }
/* L9026: */
	    iz = m.f0000;
	    if (mliq[iz - 1] < watmin) {
		xs = watmin - mliq[iz - 1];
	    } else {
		xs = rc0003;
	    }
	    mliq[iz - 1] += xs;
	    m.f0343 -= xs / m.f0005;
	    if (m.f0031 == 5) {
		m.f0347 -= xs * rc0010;
	    }
	    i__1 = m.f0000;
	    for (iz = 1; iz <= i__1; ++iz) {
		m.f0394[iz - 1] = mliq[iz - 1] / (m.f0027[iz + 2]
			* rc0008);
/* L9028: */
	    }
/* L9029: */
	}
	if (m.f0031 != 2) {
	    r0084_(m);
	}
    }
    if (m.f0045 == 2) {
	i__1 = m.f0000;
	for (k = 1; k <= i__1; ++k) {
	    if (m.f0391[k - 1] != m.f0392[k - 1]) {
		sice_smc_ratio__ = m.f0393[k - 1] / m.f0391[k - 1]
			;
		m.f0391[k - 1] = m.f0392[k - 1];
		m.f0393[k - 1] = m.f0391[k - 1] *
			sice_smc_ratio__;
		m.f0394[k - 1] = m.f0391[k - 1] - m.f0393[k
			- 1];
	    }
/* L9031: */
	}
/* L9032: */
	if (m.f0345 != m.f0071) {
	    m.f0345 = m.f0071;
	}
    }
    return 0;
} /* r0060_ */

__device__ __forceinline__ /* Subroutine */ int r0061_(Model &m, real *dt, real *rhstt, real *ai, real *bi, real *
	ci) {
    /* Initialized data */

    real rc0000 = 2.f;
    real rc0001 = 0.f;
    real rc0002 = 1.f;
    real rc0003 = 2.f;
    real rc0004 = 0.f;

    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer k;
    real ddz[4], denom[4], dsmdz[4], wflux[4], wdf[4], smx[4], temp1, smxwtd,
	    smxbot;


    /* Parameter adjustments */
    --ci;
    --bi;
    --ai;
    --rhstt;

    /* Function Body */
    if (m.f0032 == 1) {
	i__1 = m.f0000;
	for (k = 1; k <= i__1; ++k) {
	    r0064_(m, &wdf[k - 1], &m.f0397[k - 1], &m.f0391[k - 1],
		    &m.f0398[k - 1], &k);
	    smx[k - 1] = m.f0391[k - 1];
/* L9004: */
	}
/* L9005: */
	if (m.f0031 == 5) {
	    smxwtd = m.f0346;
	}
    }
    if (m.f0032 == 2) {
	i__1 = m.f0000;
	for (k = 1; k <= i__1; ++k) {
	    r0065_(m, &wdf[k - 1], &m.f0397[k - 1], &m.f0394[k - 1],
		    &m.f0352, &k);
	    smx[k - 1] = m.f0394[k - 1];
/* L9007: */
	}
/* L9008: */
	if (m.f0031 == 5) {
	    smxwtd = m.f0346 * m.f0394[m.f0000 - 1] /
		    m.f0391[m.f0000 - 1];
	}
    }
    i__1 = m.f0000;
    for (k = 1; k <= i__1; ++k) {
	if (k == 1) {
	    denom[k - 1] = -m.f0026[k - 1];
	    temp1 = -m.f0026[k];
	    ddz[k - 1] = rc0000 / temp1;
	    dsmdz[k - 1] = rc0000 * (smx[k - 1] - smx[k]) / temp1;
	    wflux[k - 1] = wdf[k - 1] * dsmdz[k - 1] + m.f0397[k - 1] -
		    m.f0350 + m.f0395[k - 1] + m.f0340;
	} else if (k < m.f0000) {
	    denom[k - 1] = m.f0026[k - 2] - m.f0026[k - 1];
	    temp1 = m.f0026[k - 2] - m.f0026[k];
	    ddz[k - 1] = rc0000 / temp1;
	    dsmdz[k - 1] = rc0000 * (smx[k - 1] - smx[k]) / temp1;
	    wflux[k - 1] = wdf[k - 1] * dsmdz[k - 1] + m.f0397[k - 1] -
		    wdf[k - 2] * dsmdz[k - 2] - m.f0397[k - 2] +
		    m.f0395[k - 1];
	} else {
	    denom[k - 1] = m.f0026[k - 2] - m.f0026[k - 1];
	    if (m.f0031 == 1 || m.f0031 == 2) {
		m.f0344 = rc0001;
	    }
	    if (m.f0031 == 3 || m.f0031 == 6 || m.f0031 ==
		    7 || m.f0031 == 8) {
		m.f0344 = m.f0068 * m.f0397[k - 1];
	    }
	    if (m.f0031 == 4) {
		m.f0344 = (rc0002 - m.f0348) * m.f0397[k -
			1];
	    }
	    if (m.f0031 == 5) {
		temp1 = rc0000 * denom[k - 1];
		if (m.f0345 < m.f0026[m.f0000 - 1] - denom[
			m.f0000 - 1]) {
		    smxbot = smx[k - 1] - (smx[k - 1] - smxwtd) * denom[k - 1]
			     * rc0003 / (denom[k - 1] + m.f0026[k - 1]
			    - m.f0345);
		} else {
		    smxbot = smxwtd;
		}
		dsmdz[k - 1] = rc0000 * (smx[k - 1] - smxbot) / temp1;
		m.f0344 = wdf[k - 1] * dsmdz[k - 1] + m.f0397[k -
			1];
	    }
	    wflux[k - 1] = -(wdf[k - 2] * dsmdz[k - 2]) - m.f0397[k - 2]
		     + m.f0395[k - 1] + m.f0344;
	}
/* L9010: */
    }
/* L9011: */
    i__1 = m.f0000;
    for (k = 1; k <= i__1; ++k) {
	if (k == 1) {
	    ai[k] = rc0004;
	    bi[k] = wdf[k - 1] * ddz[k - 1] / denom[k - 1];
	    ci[k] = -bi[k];
	} else if (k < m.f0000) {
	    ai[k] = -wdf[k - 2] * ddz[k - 2] / denom[k - 1];
	    ci[k] = -wdf[k - 1] * ddz[k - 1] / denom[k - 1];
	    bi[k] = -(ai[k] + ci[k]);
	} else {
	    ai[k] = -wdf[k - 2] * ddz[k - 2] / denom[k - 1];
	    ci[k] = rc0004;
	    bi[k] = -(ai[k] + ci[k]);
	}
	rhstt[k] = wflux[k - 1] / (-denom[k - 1]);
/* L9013: */
    }
/* L9014: */
    return 0;
} /* r0061_ */

__device__ __forceinline__ /* Subroutine */ int r0062_(Model &m, real *dt, real *ai, real *bi, real *ci, real *
	rhstt, real *wplus) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;
    real rc0002 = 1e-4f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3;

    /* Local variables */
    integer k, ixarr;
    real ciin[4], epore, wminus;

    real rhsttin[4];

    /* Parameter adjustments */
    --rhstt;
    --ci;
    --bi;
    --ai;

    /* Function Body */
    *wplus = rc0000;
    i__1 = m.f0000;
    for (k = 1; k <= i__1; ++k) {
	rhstt[k] *= *dt;
	ai[k] *= *dt;
	bi[k] = rc0001 + bi[k] * *dt;
	ci[k] *= *dt;
/* L9004: */
    }
/* L9005: */
    i__1 = m.f0000;
    for (k = 1; k <= i__1; ++k) {
	rhsttin[k - 1] = rhstt[k];
	ciin[k - 1] = ci[k];
/* L9007: */
    }
/* L9008: */
    r0063_(m, &ci[1], &ai[1], &bi[1], ciin, rhsttin, &rhstt[1], &c__1, &c__0);
    i__1 = m.f0000;
    for (k = 1; k <= i__1; ++k) {
	m.f0394[k - 1] += ci[k];
/* L9010: */
    }
/* L9011: */
    if (m.f0031 == 5) {
	if (m.f0345 < m.f0026[m.f0000 - 1] - m.f0027[
		m.f0000 + 2]) {
	    m.f0347 += *dt * m.f0344;
	} else {
	    m.f0346 += *dt * m.f0344 / m.f0027[
		    m.f0000 + 2];
/* Computing MAX */
	    r__1 = m.f0346 - m.f0047[m.f0000 - 1];
	    *wplus = max(r__1,rc0000) * m.f0027[m.f0000 + 2];
/* Computing MAX */
	    r__1 = rc0002 - m.f0346;
	    wminus = max(r__1,rc0000) * m.f0027[m.f0000 + 2];
/* Computing MAX */
/* Computing MIN */
	    r__2 = m.f0346, r__3 = m.f0047[m.f0000 - 1];
	    r__1 = min(r__2,r__3);
	    m.f0346 = max(r__1,rc0002);
	    m.f0394[m.f0000 - 1] += *wplus / m.f0027[
		    m.f0000 + 2];
	    m.f0344 -= *wplus / *dt;
	    m.f0347 -= wminus;
	}
    }
    for (k = m.f0000; k >= 2; --k) {
/* Computing MAX */
	r__1 = rc0002, r__2 = m.f0047[k - 1] - m.f0393[k - 1];
	epore = max(r__1,r__2);
/* Computing MAX */
	r__1 = m.f0394[k - 1] - epore;
	*wplus = max(r__1,rc0000) * m.f0027[k + 2];
/* Computing MIN */
	r__1 = epore, r__2 = m.f0394[k - 1];
	m.f0394[k - 1] = min(r__1,r__2);
	m.f0394[k - 2] += *wplus / m.f0027[k + 1];
/* L9013: */
    }
/* L9014: */
/* Computing MAX */
    r__1 = rc0002, r__2 = m.f0047[0] - m.f0393[0];
    epore = max(r__1,r__2);
/* Computing MAX */
    r__1 = m.f0394[0] - epore;
    *wplus = max(r__1,rc0000) * m.f0027[3];
    m.f0394[0] = min(epore,m.f0394[0]);
    if (*wplus > rc0000) {
	m.f0394[1] += *wplus / m.f0027[4];
	i__1 = m.f0000 - 1;
	for (k = 2; k <= i__1; ++k) {
/* Computing MAX */
	    r__1 = rc0002, r__2 = m.f0047[k - 1] - m.f0393[k - 1];
	    epore = max(r__1,r__2);
/* Computing MAX */
	    r__1 = m.f0394[k - 1] - epore;
	    *wplus = max(r__1,rc0000) * m.f0027[k + 2];
/* Computing MIN */
	    r__1 = epore, r__2 = m.f0394[k - 1];
	    m.f0394[k - 1] = min(r__1,r__2);
	    m.f0394[k] += *wplus / m.f0027[k + 3];
/* L9016: */
	}
/* L9017: */
/* Computing MAX */
	r__1 = rc0002, r__2 = m.f0047[m.f0000 - 1] -
		m.f0393[m.f0000 - 1];
	epore = max(r__1,r__2);
/* Computing MAX */
	r__1 = m.f0394[m.f0000 - 1] - epore;
	*wplus = max(r__1,rc0000) * m.f0027[m.f0000 + 2];
/* Computing MIN */
	r__1 = epore, r__2 = m.f0394[m.f0000 - 1];
	m.f0394[m.f0000 - 1] = min(r__1,r__2);
    }
    for (ixarr = 1; ixarr <= 4; ++ixarr) {
	m.f0391[ixarr - 1] = m.f0394[ixarr - 1] + m.f0393[
		ixarr - 1];
    }
    return 0;
} /* r0062_ */

__device__ __forceinline__ /* Subroutine */ int r0063_(Model &m, real *p, real *a, real *b, real *c__, real *d__,
	real *delta, integer *ntop, integer *nnsnow) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;

    /* System generated locals */
    integer a_offset, b_offset, d_offset, c_offset, p_offset, delta_offset,
	    i__1;

    /* Local variables */
    integer k, kk;

    /* Parameter adjustments */
    delta_offset = *nnsnow + 1;
    delta -= delta_offset;
    d_offset = *nnsnow + 1;
    d__ -= d_offset;
    c_offset = *nnsnow + 1;
    c__ -= c_offset;
    b_offset = *nnsnow + 1;
    b -= b_offset;
    a_offset = *nnsnow + 1;
    a -= a_offset;
    p_offset = *nnsnow + 1;
    p -= p_offset;

    /* Function Body */
    c__[(0 + (0 + (m.f0000 << 2))) / 4] = rc0000;
    p[*ntop] = -c__[*ntop] / b[*ntop];
    delta[*ntop] = d__[*ntop] / b[*ntop];
    i__1 = m.f0000;
    for (k = *ntop + 1; k <= i__1; ++k) {
	p[k] = -c__[k] * (rc0001 / (b[k] + a[k] * p[k - 1]));
	delta[k] = (d__[k] - a[k] * delta[k - 1]) * (rc0001 / (b[k] + a[k] *
		p[k - 1]));
/* L9004: */
    }
/* L9005: */
    p[m.f0000] = delta[m.f0000];
    i__1 = m.f0000;
    for (k = *ntop + 1; k <= i__1; ++k) {
	kk = m.f0000 - k + (*ntop - 1) + 1;
	p[kk] = p[kk] * p[kk + 1] + delta[kk];
/* L9007: */
    }
/* L9008: */
    return 0;
} /* r0063_ */

__device__ __forceinline__ /* Subroutine */ int r0064_(Model &m, real *wdf, real *wcnd, real *smc, real *fcr,
	integer *isoil) {
    /* Initialized data */

    real rc0000 = .01f;
    real rc0001 = 2.f;
    real rc0002 = 1.f;
    real rc0003 = 3.f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real expon, factr;

/* Computing MAX */
    r__1 = rc0000, r__2 = *smc / m.f0047[(276 + (0 + (*isoil - 1 << 2))
	    - 276) / 4];
    factr = max(r__1,r__2);
    expon = m.f0046[*isoil - 1] + rc0001;
    d__1 = (doublereal) factr;
    d__2 = (doublereal) expon;
    *wdf = m.f0051[*isoil - 1] * (real)pow_dd(&d__1, &d__2);
    *wdf *= rc0002 - *fcr;
    expon = rc0001 * m.f0046[*isoil - 1] + rc0003;
    d__1 = (doublereal) factr;
    d__2 = (doublereal) expon;
    *wcnd = m.f0050[*isoil - 1] * (real)pow_dd(&d__1, &d__2);
    *wcnd *= rc0002 - *fcr;
    return 0;
} /* r0064_ */

__device__ __forceinline__ /* Subroutine */ int r0065_(Model &m, real *wdf, real *wcnd, real *smc, real *sice,
	integer *isoil) {
    /* Initialized data */

    real rc0000 = .05f;
    real rc0001 = .01f;
    real rc0002 = 2.f;
    real rc0003 = 0.f;
    real rc0004 = 1.f;
    real rc0005 = 500.f;
    real rc0006 = 3.f;
    real rc0007 = 3.f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real expon, factr1, factr2, vkwgt;

    factr1 = rc0000 / m.f0047[(276 + (0 + (*isoil - 1 << 2)) - 276) / 4]
	    ;
/* Computing MAX */
    r__1 = rc0001, r__2 = *smc / m.f0047[*isoil - 1];
    factr2 = max(r__1,r__2);
    factr1 = min(factr1,factr2);
    expon = m.f0046[*isoil - 1] + rc0002;
    d__1 = (doublereal) factr2;
    d__2 = (doublereal) expon;
    *wdf = m.f0051[*isoil - 1] * (real)pow_dd(&d__1, &d__2);
    if (*sice > rc0003) {
	d__1 = (doublereal) (rc0005 * *sice);
	d__2 = (doublereal) rc0006;
	vkwgt = rc0004 / (rc0004 + (real)pow_dd(&d__1, &d__2));
	d__1 = (doublereal) factr1;
	d__2 = (doublereal) expon;
	*wdf = vkwgt * *wdf + (rc0004 - vkwgt) * m.f0051[*isoil - 1] * (
		real)pow_dd(&d__1, &d__2);
    }
    expon = rc0002 * m.f0046[*isoil - 1] + rc0007;
    d__1 = (doublereal) factr2;
    d__2 = (doublereal) expon;
    *wcnd = m.f0050[*isoil - 1] * (real)pow_dd(&d__1, &d__2);
    return 0;
} /* r0065_ */

__device__ __forceinline__ /* Subroutine */ int r0066_(Model &m) {


    if (m.f0030 == 1) {
	r0067_(m);
    } else if (m.f0030 == 2) {
	r0068_(m);
    } else if (m.f0030 == 3) {
	r0069_(m);
    } else if (m.f0030 == 4) {
	r0070_(m);
    } else if (m.f0030 == 5) {
	r0071_(m);
    } else if (m.f0030 == 6) {
	r0072_(m);
    } else if (m.f0030 == 7) {
	r0073_(m);
    } else if (m.f0030 == 8) {
	r0074_(m);
    }
    return 0;
} /* r0066_ */

__device__ __forceinline__ /* Subroutine */ int r0067_(Model &m) {
    /* Initialized data */

    real rc0000 = 6.f;
    real rc0001 = .5f;
    real rc0002 = 2.f;
    real rc0003 = 0.f;
    real rc0004 = 1.f;

    /* Builtin functions */


    /* Local variables */
    real fff, fsat;

    fff = rc0000;
    fsat = m.f0070 * (real)exp((real)(-rc0001 * fff * (m.f0345 -
	    rc0002)));
    if (m.f0339 > rc0003) {
	m.f0342 = m.f0339 * ((rc0004 - m.f0398[0]) * fsat +
		m.f0398[0]);
	m.f0350 = m.f0339 - m.f0342;
    }
    return 0;
} /* r0067_ */

__device__ __forceinline__ /* Subroutine */ int r0068_(Model &m) {
    /* Initialized data */

    real rc0000 = 2.f;
    real rc0001 = .5f;
    real rc0002 = 0.f;
    real rc0003 = 1.f;

    /* Builtin functions */


    /* Local variables */
    real fff, fsat;

    fff = rc0000;
    fsat = m.f0070 * (real)exp((real)(-rc0001 * fff * m.f0345));
    if (m.f0339 > rc0002) {
	m.f0342 = m.f0339 * ((rc0003 - m.f0398[0]) * fsat +
		m.f0398[0]);
	m.f0350 = m.f0339 - m.f0342;
    }
    return 0;
} /* r0068_ */

__device__ __forceinline__ /* Subroutine */ int r0069_(Model &m) {


    r0075_(m);
    return 0;
} /* r0069_ */

__device__ __forceinline__ /* Subroutine */ int r0070_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 2.f;
    real rc0002 = .01f;
    real rc0003 = 4.f;
    real rc0004 = 1.f;

    /* System generated locals */
    integer i__1;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    integer k;
    real smctot, dztot, fsat;

    smctot = rc0000;
    dztot = rc0000;
    i__1 = m.f0000;
    for (k = 1; k <= i__1; ++k) {
	dztot += m.f0027[k + 2];
	smctot += m.f0391[k - 1] / m.f0047[k - 1] * m.f0027[
		k + 2];
	if (dztot >= rc0001) {
	    goto L9005;
	}
/* L9004: */
    }
L9005:
    smctot /= dztot;
    d__1 = (doublereal) max(rc0002,smctot);
    d__2 = (doublereal) rc0003;
    fsat = (real)pow_dd(&d__1, &d__2);
    if (m.f0339 > rc0000) {
	m.f0342 = m.f0339 * ((rc0004 - m.f0398[0]) * fsat +
		m.f0398[0]);
	m.f0350 = m.f0339 - m.f0342;
    }
    return 0;
} /* r0070_ */

__device__ __forceinline__ /* Subroutine */ int r0071_(Model &m) {
    /* Initialized data */

    real rc0000 = 6.f;
    real rc0001 = .5f;
    real rc0002 = 2.f;
    real rc0003 = 0.f;
    real rc0004 = 1.f;

    /* System generated locals */
    real r__1;

    /* Builtin functions */


    /* Local variables */
    real fff, fsat;

    fff = rc0000;
/* Computing MAX */
    r__1 = -rc0002 - m.f0345;
    fsat = m.f0070 * (real)exp((real)(-rc0001 * fff * max(r__1,rc0003)))
	    ;
    if (m.f0339 > rc0003) {
	m.f0342 = m.f0339 * ((rc0004 - m.f0398[0]) * fsat +
		m.f0398[0]);
	m.f0350 = m.f0339 - m.f0342;
    }
    return 0;
} /* r0071_ */

__device__ __forceinline__ /* Subroutine */ int r0072_(Model &m) {


    r0076_(m);
    return 0;
} /* r0072_ */

__device__ __forceinline__ /* Subroutine */ int r0073_(Model &m) {


    r0077_(m);
    return 0;
} /* r0073_ */

__device__ __forceinline__ /* Subroutine */ int r0074_(Model &m) {


    r0078_(m);
    return 0;
} /* r0074_ */

__device__ __forceinline__ /* Subroutine */ int r0075_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 86400.f;
    real rc0002 = 1.f;
    real rc0003 = 1.f;
    real rc0004 = 0.f;
    real rc0005 = .01f;

    /* System generated locals */
    integer i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */


    /* Local variables */
    integer j, k;
    real dd;
    integer jj;
    real dt, px, dt1;
    integer ialp1;
    real val, ddt, dice, fcr1, sum, acrt, wdf, wcnd1, smcav, infmax, dmax__[4]
	    ;


    dt = m.f0387;
    if (m.f0339 > rc0000) {
	dt1 = dt / rc0001;
	smcav = m.f0047[0] - m.f0048[0];
	dmax__[0] = -m.f0026[0] * smcav;
	dice = -m.f0026[0] * m.f0393[0];
	dmax__[0] *= rc0002 - (m.f0394[0] + m.f0393[0] -
		m.f0048[0]) / smcav;
	dd = dmax__[0];
	i__1 = m.f0000;
	for (k = 2; k <= i__1; ++k) {
	    dice += (m.f0026[k - 2] - m.f0026[k - 1]) *
		    m.f0393[k - 1];
	    dmax__[k - 1] = (m.f0026[k - 2] - m.f0026[k - 1]) *
		    smcav;
	    dmax__[k - 1] *= rc0002 - (m.f0394[k - 1] + m.f0393[k
		    - 1] - m.f0048[k - 1]) / smcav;
	    dd += dmax__[k - 1];
/* L9004: */
	}
/* L9005: */
	val = rc0003 - (real)exp((real)(-m.f0060 * dt1));
	ddt = dd * val;
/* Computing MAX */
	r__1 = rc0004, r__2 = m.f0339 * dt;
	px = max(r__1,r__2);
	infmax = px * (ddt / (px + ddt)) / dt;
	fcr1 = rc0003;
	if (dice > rc0005) {
	    acrt = m.f0067 * 3 / dice;
	    sum = rc0003;
	    ialp1 = 2;
	    i__1 = ialp1;
	    for (j = 1; j <= i__1; ++j) {
		k = 1;
		i__2 = ialp1;
		for (jj = j + 1; jj <= i__2; ++jj) {
		    k *= jj;
/* L9010: */
		}
/* L9011: */
		i__2 = 3 - j;
		sum += (real)pow_ri(&acrt, &i__2) / (real) k;
/* L9007: */
	    }
/* L9008: */
	    fcr1 = rc0003 - (real)exp((real)(-acrt)) * sum;
	}
	infmax *= fcr1;
	r0065_(m, &wdf, &wcnd1, m.f0394, &m.f0352, &c__1);
	infmax = max(infmax,wcnd1);
	infmax = min(infmax,px);
/* Computing MAX */
	r__1 = rc0004, r__2 = m.f0339 - infmax;
	m.f0342 = max(r__1,r__2);
	m.f0350 = m.f0339 - m.f0342;
    }
    return 0;
} /* r0075_ */

__device__ __forceinline__ /* Subroutine */ int r0076_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;

    /* System generated locals */
    integer i__1;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real top_moist__, dt, ex;
    integer iz;
    real i_0__, i_max__, basis, top_max_moist__;

    dt = m.f0387;
    ex = rc0000;
    i_max__ = rc0000;
    i_0__ = rc0000;
    basis = rc0000;
    top_moist__ = rc0000;
    top_max_moist__ = rc0000;
    m.f0342 = rc0000;
    m.f0388 = rc0000;
    i__1 = m.f0000 - 2;
    for (iz = 1; iz <= i__1; ++iz) {
	top_moist__ += -m.f0391[iz - 1] * m.f0026[iz - 1];
	top_max_moist__ += -m.f0047[iz - 1] * m.f0026[iz - 1];
/* L9004: */
    }
/* L9005: */
    ex = m.f0053 / (m.f0053 + 1);
    d__1 = (doublereal) (rc0001 - top_moist__ / top_max_moist__);
    d__2 = (doublereal) ex;
    m.f0388 = rc0001 - (real)pow_dd(&d__1, &d__2);
    m.f0388 = max(rc0000,m.f0388);
    m.f0388 = min(rc0001,m.f0388);
    i_max__ = (rc0001 + m.f0053) * top_max_moist__;
    d__1 = (doublereal) (rc0001 - m.f0388);
    d__2 = (doublereal) (rc0001 / m.f0053);
    i_0__ = i_max__ * (rc0001 - (real)pow_dd(&d__1, &d__2));
    if (m.f0339 == rc0000) {
	m.f0342 = rc0000;
    } else if (i_max__ == rc0000) {
	m.f0342 = m.f0339 * dt;
    } else if (i_0__ + m.f0339 * dt > i_max__) {
	m.f0342 = m.f0339 * dt - top_max_moist__ + top_moist__;
    } else {
	basis = rc0001 - (i_0__ + m.f0339 * dt) / i_max__;
	d__1 = (doublereal) basis;
	d__2 = (doublereal) (rc0001 + m.f0053);
	m.f0342 = m.f0339 * dt - top_max_moist__ + top_moist__ +
		top_max_moist__ * (real)pow_dd(&d__1, &d__2);
    }
    m.f0342 /= dt;
    if (m.f0342 < rc0000) {
	m.f0342 = rc0000;
    }
    if (m.f0342 > m.f0339) {
	m.f0342 = m.f0339;
    }
    m.f0350 = m.f0339 - m.f0342;
    return 0;
} /* r0076_ */

__device__ __forceinline__ /* Subroutine */ int r0077_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 0.f;
    real rc0002 = .5f;

    /* System generated locals */
    integer i__1;
    real d__1, d__2, d__3, d__4;

    /* Builtin functions */


    /* Local variables */
    real dt, sm;
    integer iz;
    real wm, wm_max__, sm_max__, irunoff, prunoff;

    wm = rc0000;
    wm_max__ = rc0000;
    sm = rc0000;
    sm_max__ = rc0000;
    irunoff = rc0000;
    prunoff = rc0000;
    dt = m.f0387;
    m.f0342 = rc0000;
    i__1 = m.f0000 - 2;
    for (iz = 1; iz <= i__1; ++iz) {
	if (m.f0391[iz - 1] - m.f0049[iz - 1] > rc0001) {
	    sm += -(m.f0391[iz - 1] - m.f0049[iz - 1]) *
		    m.f0026[iz - 1];
	    wm += -m.f0049[iz - 1] * m.f0026[iz - 1];
	} else {
	    wm += -m.f0391[iz - 1] * m.f0026[iz - 1];
	}
	wm_max__ += -m.f0049[iz - 1] * m.f0026[iz - 1];
	sm_max__ += -(m.f0047[iz - 1] - m.f0049[iz - 1]) *
		m.f0026[iz - 1];
/* L9004: */
    }
/* L9005: */
    wm = min(wm,wm_max__);
    sm = min(sm,sm_max__);
    irunoff = m.f0398[0] * m.f0339 * dt;
    if (wm / wm_max__ <= rc0002 - m.f0054) {
	d__1 = (doublereal) (rc0002 - m.f0054);
	d__2 = (doublereal) (1 - m.f0055);
	d__3 = (doublereal) (wm / wm_max__);
	d__4 = (doublereal) m.f0055;
	prunoff = (1 - m.f0398[0]) * m.f0339 * dt * (real)pow_dd(&
		d__1, &d__2) * (real)pow_dd(&d__3, &d__4);
    } else {
	d__1 = (doublereal) (rc0002 + m.f0054);
	d__2 = (doublereal) (1 - m.f0055);
	d__3 = (doublereal) (1 - wm / wm_max__);
	d__4 = (doublereal) m.f0055;
	prunoff = (1 - m.f0398[0]) * m.f0339 * dt * (1 - (real)
		pow_dd(&d__1, &d__2) * (real)pow_dd(&d__3, &d__4));
    }
    if (m.f0339 == rc0000) {
	m.f0342 = rc0000;
    } else {
	d__1 = (doublereal) (1 - sm / sm_max__);
	d__2 = (doublereal) m.f0056;
	m.f0342 = prunoff * (1 - (real)pow_dd(&d__1, &d__2)) + irunoff;
    }
    m.f0342 /= dt;
    m.f0342 = max(rc0000,m.f0342);
    m.f0342 = min(m.f0339,m.f0342);
    m.f0350 = m.f0339 - m.f0342;
    return 0;
} /* r0077_ */

__device__ __forceinline__ /* Subroutine */ int r0078_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1.f;
    real rc0002 = 1.388889e-7f;
    real rc0003 = 2.f;

    /* System generated locals */
    integer i__1;
    real r__1;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real r1, runoffinf, runoffsat, infiltrtn, top_moist__, bb, dp, dt, yd;
    integer iz;
    real i_0__, yd0, top_max_moist__, i_max__, fsur, fmax, tempr1, tempr2,
	    yd_old__, temp1, error;
    integer izmax;

    integer inflmax;

    top_moist__ = rc0000;
    top_max_moist__ = rc0000;
    bb = rc0001;
    dp = rc0000;
    i_max__ = rc0000;
    i_0__ = rc0000;
    runoffsat = rc0000;
    runoffinf = rc0000;
    infiltrtn = rc0000;
    dt = m.f0387;
    m.f0342 = rc0000;
    izmax = 20;
    error = rc0002 * dt;
    bb = m.f0057;
    i__1 = m.f0000 - 2;
    for (iz = 1; iz <= i__1; ++iz) {
	top_moist__ += -m.f0391[iz - 1] * m.f0026[iz - 1];
	top_max_moist__ += -m.f0047[iz - 1] * m.f0026[iz - 1];
/* L9004: */
    }
/* L9005: */
    if (top_moist__ > top_max_moist__) {
	top_moist__ = top_max_moist__;
    }
    dp = m.f0339 * dt;
    i_max__ = top_max_moist__ * (m.f0053 + rc0001);
    d__1 = (doublereal) (top_moist__ / top_max_moist__);
    d__2 = (doublereal) (rc0001 / (rc0001 + m.f0053));
    i_0__ = i_max__ * (1 - (1 - (real)pow_dd(&d__1, &d__2)));
    inflmax = 0;
    if (m.f0033 == 1) {
	r0083_(m, &dt, &fsur, &inflmax);
    } else if (m.f0033 == 2) {
	r0082_(m, &fsur, &inflmax);
    } else if (m.f0033 == 3) {
	r0081_(m, &fsur, &inflmax);
    }
    fmax = (bb + rc0001) * fsur;
    if (dp <= rc0000) {
	runoffsat = rc0000;
	runoffinf = rc0000;
	infiltrtn = rc0000;
	goto L2001;
    } else {
	if (top_moist__ >= top_max_moist__ && i_0__ >= i_max__) {
	    top_moist__ = top_max_moist__;
	    i_0__ = i_max__;
	    runoffsat = dp;
	    runoffinf = rc0000;
	    infiltrtn = rc0000;
	    goto L2001;
	} else {
	    d__1 = (doublereal) (top_moist__ / top_max_moist__);
	    d__2 = (doublereal) (rc0001 / (rc0001 + m.f0053));
	    i_0__ = i_max__ * (1 - (1 - (real)pow_dd(&d__1, &d__2)));
	    if (dp + i_0__ > i_max__) {
		if (fmax * dt >= dp) {
		    yd = i_max__ - i_0__;
		    tempr1 = rc0000;
		    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
		    d__1 = (doublereal) ((dp - tempr1) / (fmax * dt));
		    d__2 = (doublereal) (bb + rc0001);
		    temp1 = i_max__ - i_0__ - tempr1 - fsur * dt * (1 - (1 - (
			    real)pow_dd(&d__1, &d__2)));
		    if (temp1 <= rc0000) {
			yd = i_max__ - i_0__;
			infiltrtn = top_max_moist__ - top_moist__;
			runoffsat = dp - infiltrtn;
			runoffinf = rc0000;
			top_moist__ = top_max_moist__;
			i_0__ = i_max__;
			goto L2001;
		    } else {
			yd = rc0000;
			i__1 = izmax;
			for (iz = 1; iz <= i__1; ++iz) {
			    yd_old__ = yd;
			    tempr1 = rc0000;
			    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			    d__1 = (doublereal) ((dp - tempr1) / (fmax * dt));
			    d__2 = (doublereal) (bb + rc0001);
			    yd = tempr1 + fsur * dt * (1 - (1 - (real)pow_dd(&
				    d__1, &d__2)));
			    if ((r__1 = yd - yd_old__, abs(r__1)) <= error ||
				    iz == izmax) {
				goto L1001;
			    }
/* L9007: */
			}
/* L9008: */
		    }
		} else {
		    tempr1 = rc0000;
		    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
		    if (tempr1 + fmax * dt <= dp) {
			if (i_max__ - i_0__ - tempr1 - fmax * dt <= rc0000) {
			    yd = i_max__ - i_0__;
			    infiltrtn = top_max_moist__ - top_moist__;
			    runoffsat = dp - infiltrtn;
			    runoffinf = rc0000;
			    top_moist__ = top_max_moist__;
			    i_0__ = i_max__;
			    goto L2001;
			} else {
			    yd = rc0000;
			    i__1 = izmax;
			    for (iz = 1; iz <= i__1; ++iz) {
				yd_old__ = yd;
				tempr1 = rc0000;
				r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
				yd = tempr1 + fsur * dt;
				if ((r__1 = yd - yd_old__, abs(r__1)) <=
					error || iz == izmax) {
				    goto L1001;
				}
/* L9010: */
			    }
/* L9011: */
			}
		    } else {
			yd = dp / rc0003;
			i__1 = izmax;
			for (iz = 1; iz <= i__1; ++iz) {
			    yd_old__ = yd;
			    tempr1 = rc0000;
			    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			    yd = yd - tempr1 - fsur * dt + dp;
			    if (yd <= rc0000) {
				yd = rc0000;
			    }
			    if (yd >= dp) {
				yd = dp;
			    }
			    if ((r__1 = yd - yd_old__, abs(r__1)) <= error ||
				    iz == izmax) {
				yd0 = yd;
				goto L9014;
			    }
/* L9013: */
			}
L9014:
			i__1 = izmax;
			for (iz = 1; iz <= i__1; ++iz) {
			    yd_old__ = yd;
			    tempr1 = rc0000;
			    tempr2 = rc0000;
			    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			    r0080_(m, &yd, &yd0, &tempr1, &fmax, &fsur, &dt, &dp,
				     &bb, &tempr2);
			    yd = dp - tempr2;
			    if ((r__1 = yd - yd_old__, abs(r__1)) <= error ||
				    iz == izmax) {
				goto L1001;
			    }
/* L9016: */
			}
/* L9017: */
L1001:
			if (yd <= rc0000) {
			    yd = rc0000;
			}
			if (yd >= dp) {
			    yd = dp;
			}
			r0079_(m, &i_0__, &i_max__, &yd, &r1);
			runoffsat = r1;
			runoffinf = dp - yd;
			infiltrtn = yd - runoffsat;
			top_moist__ += infiltrtn;
			yd = i_0__ + yd;
			if (top_moist__ <= rc0000) {
			    top_moist__ = rc0000;
			}
			if (top_moist__ >= top_max_moist__) {
			    top_moist__ = top_max_moist__;
			}
			d__1 = (doublereal) (top_moist__ / top_max_moist__);
			d__2 = (doublereal) (rc0001 / (rc0001 + m.f0053)
				);
			i_0__ = i_max__ * (1 - (1 - (real)pow_dd(&d__1, &d__2)
				));
			goto L2001;
		    }
		}
	    } else {
		if (fmax * dt >= dp) {
		    yd = dp / rc0003;
		    i__1 = izmax;
		    for (iz = 1; iz <= i__1; ++iz) {
			yd_old__ = yd;
			tempr1 = rc0000;
			r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			d__1 = (doublereal) ((dp - tempr1) / (fmax * dt));
			d__2 = (doublereal) (bb + rc0001);
			yd = tempr1 + fsur * dt * (1 - (1 - (real)pow_dd(&
				d__1, &d__2)));
			if ((r__1 = yd - yd_old__, abs(r__1)) <= error || iz
				== izmax) {
			    goto L1002;
			}
/* L9019: */
		    }
/* L9020: */
		} else {
		    tempr1 = rc0000;
		    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
		    if (tempr1 + fmax * dt <= dp) {
			yd = dp / rc0003;
			i__1 = izmax;
			for (iz = 1; iz <= i__1; ++iz) {
			    yd_old__ = yd;
			    tempr1 = rc0000;
			    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			    yd = tempr1 + fsur * dt;
			    if ((r__1 = yd - yd_old__, abs(r__1)) <= error ||
				    iz == izmax) {
				goto L1002;
			    }
/* L9022: */
			}
/* L9023: */
		    } else {
			yd = rc0000;
			i__1 = izmax;
			for (iz = 1; iz <= i__1; ++iz) {
			    yd_old__ = yd;
			    tempr1 = rc0000;
			    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			    yd = dp - fmax * dt + yd - tempr1;
			    if (yd <= rc0000) {
				yd = rc0000;
			    }
			    if (yd >= dp) {
				yd = dp;
			    }
			    tempr1 = rc0000;
			    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			    if ((r__1 = tempr1 + fmax * dt - dp, abs(r__1)) <=
				     error || iz == izmax) {
				yd0 = yd;
				goto L9026;
			    }
/* L9025: */
			}
L9026:
			i__1 = izmax;
			for (iz = 1; iz <= i__1; ++iz) {
			    yd_old__ = yd;
			    tempr1 = rc0000;
			    tempr2 = rc0000;
			    r0079_(m, &i_0__, &i_max__, &yd, &tempr1);
			    r0080_(m, &yd, &yd0, &tempr1, &fmax, &fsur, &dt, &dp,
				     &bb, &tempr2);
			    yd = dp - tempr2;
			    if ((r__1 = yd - yd_old__, abs(r__1)) <= error ||
				    iz == izmax) {
				goto L1002;
			    }
/* L9028: */
			}
/* L9029: */
		    }
		}
L1002:
		if (yd <= rc0000) {
		    yd = rc0000;
		}
		if (yd >= dp) {
		    yd = dp;
		}
		r1 = rc0000;
		r0079_(m, &i_0__, &i_max__, &yd, &r1);
		runoffsat = r1;
		runoffinf = dp - yd;
		infiltrtn = yd - runoffsat;
		top_moist__ += infiltrtn;
		if (top_moist__ <= rc0000) {
		    top_moist__ = rc0000;
		}
		if (top_moist__ >= top_max_moist__) {
		    top_moist__ = top_max_moist__;
		}
		d__1 = (doublereal) (top_moist__ / top_max_moist__);
		d__2 = (doublereal) (rc0001 / (rc0001 + m.f0053));
		i_0__ = i_max__ * (1 - (1 - (real)pow_dd(&d__1, &d__2)));
	    }
	}
    }
L2001:
    m.f0342 = (runoffsat + runoffinf) / dt;
    m.f0342 = min(m.f0342,m.f0339);
    m.f0342 = max(m.f0342,rc0000);
    m.f0350 = m.f0339 - m.f0342;
    return 0;
} /* r0078_ */

__device__ __forceinline__ /* Subroutine */ int r0079_(Model &m, real *i_0__, real *i_max__, real *yd, real *r1) {
    /* Initialized data */

    real rc0000 = 1.f;
    real rc0001 = 0.f;

    /* System generated locals */
    real d__1, d__2, d__3, d__4;

    /* Builtin functions */


    /* Local variables */
    real tdepth;

    tdepth = *i_0__ + *yd;
    if (tdepth > *i_max__) {
	tdepth = *i_max__;
    }
    d__1 = (doublereal) (1 - *i_0__ / *i_max__);
    d__2 = (doublereal) (m.f0053 + rc0000);
    d__3 = (doublereal) (1 - tdepth / *i_max__);
    d__4 = (doublereal) (m.f0053 + rc0000);
    *r1 = *yd - *i_max__ / (m.f0053 + rc0000) * ((real)pow_dd(&d__1, &
	    d__2) - (real)pow_dd(&d__3, &d__4));
    if (*r1 < rc0001) {
	*r1 = rc0001;
    }
    return 0;
} /* r0079_ */

__device__ __forceinline__ /* Subroutine */ int r0080_(Model &m, real *yd, real *y0, real *r1, real *fmax, real *
	fsur, real *dt, real *dp, real *bb, real *r2) {
    /* Initialized data */

    real rc0000 = 1.f;
    real rc0001 = 0.f;

    /* System generated locals */
    real d__1, d__2;

    /* Builtin functions */


    if (*yd >= *y0) {
	d__1 = (doublereal) (1 - (*dp - *r1) / (*fmax * *dt));
	d__2 = (doublereal) (*bb + rc0000);
	*r2 = *dp - *r1 - *fmax * *dt * (1 - (real)pow_dd(&d__1, &d__2));
    } else {
	*r2 = *dp - *r1 - *fmax * *dt;
    }
    if (*r2 < rc0001) {
	*r2 = rc0001;
    }
    return 0;
} /* r0080_ */

__device__ __forceinline__ /* Subroutine */ int r0081_(Model &m, real *fsur, integer *inflmax) {
    /* Initialized data */

    real rc0000 = .82f;
    real rc0001 = 0.f;
    real rc0002 = 1e-5f;

    /* Builtin functions */


    /* Local variables */
    real jj, wdf, wcnd, gam;
    integer isoil;


    gam = rc0000;
    isoil = 1;
    if (*inflmax == 1) {
	r0065_(m, &wdf, &wcnd, &m.f0048[isoil - 1], &rc0001, &isoil);
	jj = -(m.f0058 * (m.f0047[isoil - 1] - m.f0048[
		isoil - 1])) * m.f0026[isoil - 1];
	*fsur = m.f0050[isoil - 1] + gam * (m.f0050[isoil - 1] -
		wcnd) / ((real)exp((real)(gam * rc0002 / jj)) - 1);
	if (m.f0050[isoil - 1] < m.f0339) {
	    *fsur = min(m.f0339,*fsur);
	} else {
	    *fsur = m.f0339;
	}
	if (*fsur < rc0001) {
	    *fsur = rc0001;
	}
    } else {
	r0065_(m, &wdf, &wcnd, &m.f0391[isoil - 1], &m.f0393[isoil -
		1], &isoil);
	jj = -(m.f0058 * (m.f0047[isoil - 1] - m.f0391[
		isoil - 1])) * m.f0026[isoil - 1];
	*fsur = m.f0050[isoil - 1] + gam * (m.f0050[isoil - 1] -
		wcnd) / ((real)exp((real)(gam * m.f0351 / jj)) - 1);
	if (m.f0050[isoil - 1] < m.f0339) {
	    *fsur = min(m.f0339,*fsur);
	} else {
	    *fsur = m.f0339;
	}
	m.f0351 += *fsur;
    }
    return 0;
} /* r0081_ */

__device__ __forceinline__ /* Subroutine */ int r0082_(Model &m, real *fsur, integer *inflmax) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 1e-5f;

    real jj, wdf, wcnd;
    integer isoil;


    isoil = 1;
    if (*inflmax == 1) {
	r0065_(m, &wdf, &wcnd, &m.f0048[isoil - 1], &rc0000, &isoil);
	jj = -(m.f0058 * (m.f0047[isoil - 1] - m.f0048[
		isoil - 1])) * m.f0026[isoil - 1];
	*fsur = m.f0050[isoil - 1] + jj / rc0001 * (m.f0050[isoil
		- 1] - wcnd);
	if (*fsur < rc0000) {
	    *fsur = rc0000;
	}
    } else {
	r0065_(m, &wdf, &wcnd, &m.f0391[isoil - 1], &m.f0393[isoil -
		1], &isoil);
	jj = -(m.f0058 * (m.f0047[isoil - 1] - m.f0391[
		isoil - 1])) * m.f0026[isoil - 1];
	*fsur = m.f0050[isoil - 1] + jj / m.f0351 * (
		m.f0050[isoil - 1] - wcnd);
	if (m.f0050[isoil - 1] < m.f0339) {
	    *fsur = min(m.f0339,*fsur);
	} else {
	    *fsur = m.f0339;
	}
	m.f0351 += *fsur;
    }
    return 0;
} /* r0082_ */

__device__ __forceinline__ /* Subroutine */ int r0083_(Model &m, real *dt, real *fsur, integer *inflmax) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 2.f;
    real rc0002 = 1.f;

    /* System generated locals */
    real r__1, r__2;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    real ap, sp, wdf, wcnd;
    integer isoil;


    isoil = 1;
    if (*inflmax == 1) {
	r0065_(m, &wdf, &wcnd, &m.f0048[isoil - 1], &rc0000, &isoil);
	sp = (real)sqrt((real)((m.f0047[isoil - 1] - m.f0048[
		isoil - 1]) * 2 * (m.f0051[isoil - 1] - wdf)));
/* Computing MIN */
	r__1 = wcnd, r__2 = rc0001 / 3 * m.f0050[isoil - 1];
	ap = min(r__1,r__2);
/* Computing MAX */
	r__1 = ap, r__2 = rc0002 / 3 * m.f0050[isoil - 1];
	ap = max(r__1,r__2);
	d__1 = (doublereal) (*dt);
	d__2 = (doublereal) (-rc0002 / 2);
	*fsur = rc0002 / 2 * sp * (real)pow_dd(&d__1, &d__2) + ap;
	if (*fsur < rc0000) {
	    *fsur = rc0000;
	}
    } else {
	r0065_(m, &wdf, &wcnd, &m.f0391[isoil - 1], &m.f0393[isoil -
		1], &isoil);
	sp = (real)sqrt((real)((m.f0047[isoil - 1] - m.f0391[
		isoil - 1]) * 2 * (m.f0051[isoil - 1] - wdf)));
/* Computing MIN */
	r__1 = wcnd, r__2 = rc0001 / 3 * m.f0050[isoil - 1];
	ap = min(r__1,r__2);
/* Computing MAX */
	r__1 = ap, r__2 = rc0002 / 3 * m.f0050[isoil - 1];
	ap = max(r__1,r__2);
	d__1 = (doublereal) (*dt);
	d__2 = (doublereal) (-rc0002 / 2);
	*fsur = rc0002 / 2 * sp * (real)pow_dd(&d__1, &d__2) + ap;
	if (m.f0050[isoil - 1] < m.f0339) {
	    *fsur = min(m.f0339,*fsur);
	} else {
	    *fsur = m.f0339;
	}
	m.f0351 += *fsur;
    }
    return 0;
} /* r0083_ */

__device__ __forceinline__ /* Subroutine */ int r0084_(Model &m) {


    if (m.f0031 == 1) {
	r0085_(m);
    } else if (m.f0031 == 2) {
	r0086_(m);
    } else if (m.f0031 == 3) {
	r0087_(m);
    } else if (m.f0031 == 4) {
	r0087_(m);
    } else if (m.f0031 == 5) {
	r0088_(m);
    } else if (m.f0031 == 6) {
	r0087_(m);
    } else if (m.f0031 == 7) {
	r0087_(m);
    } else if (m.f0031 == 8) {
	r0087_(m);
    }
    return 0;
} /* r0084_ */

__device__ __forceinline__ /* Subroutine */ int r0085_(Model &m) {
    /* Initialized data */

    real rc0000 = 6.f;
    real rc0001 = 5.f;
    real rc0002 = 1.f;
    real rc0003 = 2.f;

    /* Builtin functions */


    /* Local variables */
    real fff, rsbmx;

    fff = rc0000;
    rsbmx = rc0001;
    m.f0343 = (rc0002 - m.f0348) * rsbmx * (real)exp((real)(
	    -m.f0069)) * (real)exp((real)(-fff * (m.f0345 -
	    rc0003)));
    return 0;
} /* r0085_ */

__device__ __forceinline__ /* Subroutine */ int r0086_(Model &m) {
    /* Initialized data */

    real rc0000 = 2.f;
    real rc0001 = 4.f;
    real rc0002 = 1.f;

    /* Builtin functions */


    /* Local variables */
    real fff, rsbmx;


    fff = rc0000;
    rsbmx = rc0001;
    r0089_(m);
    m.f0343 = (rc0002 - m.f0348) * rsbmx * (real)exp((real)(
	    -m.f0069)) * (real)exp((real)(-fff * m.f0345));
    return 0;
} /* r0086_ */

__device__ __forceinline__ /* Subroutine */ int r0087_(Model &m) {
    m.f0343 += m.f0344;
    return 0;
} /* r0087_ */

__device__ __forceinline__ /* Subroutine */ int r0088_(Model &m) {
    m.f0343 += m.f0344;
    return 0;
} /* r0088_ */

__device__ __forceinline__ /* Subroutine */ int r0089_(Model &m) {
    /* Initialized data */

    real rc0000 = 0.f;
    real rc0001 = 3.f;
    real rc0002 = 3.f;
    real rc0003 = .001f;
    real rc0004 = 1.f;

    /* System generated locals */
    integer i__1;
    real r__1;
    real d__1, d__2;

    /* Builtin functions */


    /* Local variables */
    integer k;
    real wd1, wd2, dzfine, temp, zfine[100];

    wd1 = rc0000;
    i__1 = m.f0000;
    for (k = 1; k <= i__1; ++k) {
	wd1 += (m.f0047[0] - m.f0394[k - 1]) * m.f0027[k +
		2];
/* L9004: */
    }
/* L9005: */
    dzfine = rc0001 * (-m.f0026[m.f0000 - 1]) / 100;
    for (k = 1; k <= 100; ++k) {
	zfine[k - 1] = (real) k * dzfine;
/* L9007: */
    }
/* L9008: */
    m.f0345 = -rc0002 * m.f0026[m.f0000 - 1] - rc0003;
    wd2 = rc0000;
    for (k = 1; k <= 100; ++k) {
	temp = rc0004 + (m.f0345 - zfine[k - 1]) / m.f0052[0];
	d__1 = (doublereal) temp;
	d__2 = (doublereal) (-rc0004 / m.f0046[0]);
	wd2 += m.f0047[0] * (rc0004 - (real)pow_dd(&d__1, &d__2)) *
		dzfine;
	if ((r__1 = wd2 - wd1, abs(r__1)) <= .01f) {
	    m.f0345 = zfine[k - 1];
	    goto L9011;
	}
/* L9010: */
    }
L9011:
    return 0;
} /* r0089_ */

__device__ __forceinline__ /* Subroutine */ int r0090_(Model &m) {
    /* Initialized data */

    real dtheta_max__ = 0.f;
    real totalwat = 0.f;
    real tw0 = 0.f;
    real acsrf = 0.f;
    real acsub = 0.f;
    real acpcp = 0.f;
    real errwat = 0.f;
    real rc0000 = 0.f;
    real rc0001 = 0.f;
    real rc0002 = 1e3f;
    real rc0003 = 1e3f;
    real rc0004 = .001f;
    real rc0005 = 3.402823466e38f;

    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3;

    /* Local variables */
    integer iz, ixarr;
    real red001, smcold[4];


    for (ixarr = 1; ixarr <= 4; ++ixarr) {
/* Computing MAX */
	r__1 = rc0000, r__2 = m.f0391[ixarr - 1] - m.f0394[ixarr
		- 1];
	m.f0393[ixarr - 1] = max(r__1,r__2);
    }
    m.f0377 = m.f0376;
/* Computing MAX */
    r__1 = m.f0320 / m.f0299;
    m.f0383 = max(r__1,rc0001);
/* Computing MIN */
    r__2 = m.f0320 / m.f0299;
    m.f0384 = (r__1 = min(r__2,rc0001), abs(r__1));
    if (m.f0209 > m.f0125) {
	m.f0214 = FALSE_;
    } else {
	m.f0214 = TRUE_;
    }
    if (m.f0210 > m.f0125) {
	m.f0215 = FALSE_;
    } else {
	m.f0215 = TRUE_;
    }
    red001 = rc0000;
    for (ixarr = 1; ixarr <= 4; ++ixarr) {
	red001 += m.f0027[ixarr + 2] * m.f0391[ixarr - 1] *
		rc0002;
    }
    tw0 = red001;
    for (ixarr = 1; ixarr <= 4; ++ixarr) {
	smcold[ixarr - 1] = rc0000;
    }
    r0013_(m);
    m.f0374 = rc0000;
    if (m.f0376 > rc0001) {
/* Computing MIN */
	r__1 = m.f0383, r__2 = m.f0376 / m.f0005;
	m.f0374 = min(r__1,r__2);
    }
    m.f0340 = m.f0383 - m.f0374;
    m.f0373 = rc0000;
    if (m.f0376 > rc0001) {
	m.f0373 = m.f0384;
    }
    m.f0385 = m.f0384 - m.f0373;
    r0053_(m);
    if (m.f0215) {
	m.f0393[0] += (m.f0385 - m.f0340) * m.f0005 /
		(m.f0027[3] * rc0003);
	m.f0385 = rc0000;
	m.f0340 = rc0000;
	if (m.f0393[0] < rc0001) {
	    m.f0394[0] += m.f0393[0];
	    m.f0393[0] = rc0001;
	}
    }
    m.f0339 = (m.f0380 + m.f0381 + m.f0382) /
	    m.f0005 * rc0004;
    m.f0389 = m.f0380 + m.f0381 + m.f0382 +
	    m.f0379 * m.f0005;
    if (m.f0390 == 0) {
	m.f0339 += (m.f0379 + m.f0385 + m.f0368) *
		rc0004;
    } else {
	m.f0339 += (m.f0379 + m.f0385) * rc0004;
    }
    m.f0340 *= rc0004;
    i__1 = m.f0078;
    for (iz = 1; iz <= i__1; ++iz) {
	m.f0395[iz - 1] = m.f0372 * m.f0396[iz - 1] *
		rc0004;
/* L9004: */
    }
/* L9005: */
    if (m.f0025 == 2) {
	m.f0342 = rc0001;
	if (m.f0386 >= m.f0167) {
	    m.f0342 = m.f0339 * rc0003;
	}
	m.f0386 = m.f0386 + (m.f0339 - m.f0340) *
		rc0003 * m.f0005 - m.f0342 * m.f0005;
    } else {
	for (ixarr = 1; ixarr <= 4; ++ixarr) {
	    smcold[ixarr - 1] = m.f0391[ixarr - 1];
	}
	r0060_(m);
    }
    for (ixarr = 1; ixarr <= 4; ++ixarr) {
	m.f0391[ixarr - 1] = m.f0394[ixarr - 1] + m.f0393[
		ixarr - 1];
    }
    m.f0341 = m.f0340 + m.f0372 * rc0004;
    if (m.f0045 == 1) {
	m.f0343 += m.f0349;
	acsrf = m.f0342 * m.f0005;
	acsub = m.f0343 * m.f0005;
	acpcp = m.f0339 * m.f0005 * rc0002;
	red001 = -rc0005;
	for (ixarr = 1; ixarr <= 4; ++ixarr) {
/* Computing MAX */
	    r__2 = red001, r__3 = (r__1 = m.f0391[ixarr - 1] - smcold[
		    ixarr - 1], abs(r__1));
	    red001 = max(r__2,r__3);
	}
	dtheta_max__ = red001;
	red001 = rc0000;
	i__1 = m.f0000;
	for (ixarr = 1; ixarr <= i__1; ++ixarr) {
	    red001 += m.f0027[ixarr + 2] * m.f0391[ixarr - 1] *
		    rc0002;
	}
	totalwat = red001;
	errwat = acpcp - acsrf - acsub - (totalwat - tw0);
    }
    return 0;
} /* r0090_ */


__device__ __forceinline__ void noahowp_step_cell(
       int c, int n, float *f32, int *i32, double *f64, int *status, int stage, int advance,
       int *sticky_status){
    Model m;
    m.f0011=f64[0*n+c]; m.f0012=f64[1*n+c]; m.f0013=f64[2*n+c]; m.f0016=f64[3*n+c]; m.f0000=i32[0*n+c]; m.f0001=i32[1*n+c]; m.f0002=i32[2*n+c]; m.f0003=i32[3*n+c]; m.f0004=i32[4*n+c]; m.f0005=f32[0*n+c]; m.f0006=i32[5*n+c]; m.f0007=i32[6*n+c]; m.f0008=i32[7*n+c]; m.f0009=i32[8*n+c]; m.f0010=i32[9*n+c]; m.f0014=i32[10*n+c]; m.f0015=i32[11*n+c]; m.f0017=f32[1*n+c]; m.f0018=f32[2*n+c]; m.f0019=f32[3*n+c]; m.f0020=f32[4*n+c]; m.f0021=f32[5*n+c]; m.f0022=i32[12*n+c]; m.f0023=i32[13*n+c]; m.f0024=i32[14*n+c]; m.f0025=i32[15*n+c]; m.f0026[0]=f32[6*n+c]; m.f0026[1]=f32[7*n+c]; m.f0026[2]=f32[8*n+c]; m.f0026[3]=f32[9*n+c]; m.f0027[0]=f32[10*n+c]; m.f0027[1]=f32[11*n+c]; m.f0027[2]=f32[12*n+c]; m.f0027[3]=f32[13*n+c]; m.f0027[4]=f32[14*n+c]; m.f0027[5]=f32[15*n+c]; m.f0027[6]=f32[16*n+c]; m.f0028[0]=f32[17*n+c]; m.f0028[1]=f32[18*n+c]; m.f0028[2]=f32[19*n+c]; m.f0028[3]=f32[20*n+c]; m.f0028[4]=f32[21*n+c]; m.f0028[5]=f32[22*n+c]; m.f0028[6]=f32[23*n+c]; m.f0029=i32[16*n+c]; m.f0030=i32[17*n+c]; m.f0031=i32[18*n+c]; m.f0032=i32[19*n+c]; m.f0033=i32[20*n+c]; m.f0034=i32[21*n+c]; m.f0035=i32[22*n+c]; m.f0036=i32[23*n+c]; m.f0037=i32[24*n+c]; m.f0038=i32[25*n+c]; m.f0039=i32[26*n+c]; m.f0040=i32[27*n+c]; m.f0041=i32[28*n+c]; m.f0042=i32[29*n+c]; m.f0043=i32[30*n+c]; m.f0044=i32[31*n+c]; m.f0045=i32[32*n+c]; m.f0046[0]=f32[24*n+c]; m.f0046[1]=f32[25*n+c]; m.f0046[2]=f32[26*n+c]; m.f0046[3]=f32[27*n+c]; m.f0047[0]=f32[28*n+c]; m.f0047[1]=f32[29*n+c]; m.f0047[2]=f32[30*n+c]; m.f0047[3]=f32[31*n+c]; m.f0048[0]=f32[32*n+c]; m.f0048[1]=f32[33*n+c]; m.f0048[2]=f32[34*n+c]; m.f0048[3]=f32[35*n+c]; m.f0049[0]=f32[36*n+c]; m.f0049[1]=f32[37*n+c]; m.f0049[2]=f32[38*n+c]; m.f0049[3]=f32[39*n+c]; m.f0050[0]=f32[40*n+c]; m.f0050[1]=f32[41*n+c]; m.f0050[2]=f32[42*n+c]; m.f0050[3]=f32[43*n+c]; m.f0051[0]=f32[44*n+c]; m.f0051[1]=f32[45*n+c]; m.f0051[2]=f32[46*n+c]; m.f0051[3]=f32[47*n+c]; m.f0052[0]=f32[48*n+c]; m.f0052[1]=f32[49*n+c]; m.f0052[2]=f32[50*n+c]; m.f0052[3]=f32[51*n+c]; m.f0053=f32[52*n+c]; m.f0054=f32[53*n+c]; m.f0055=f32[54*n+c]; m.f0056=f32[55*n+c]; m.f0057=f32[56*n+c]; m.f0058=f32[57*n+c]; m.f0059=f32[58*n+c]; m.f0060=f32[59*n+c]; m.f0061=f32[60*n+c]; m.f0062=f32[61*n+c]; m.f0063=f32[62*n+c]; m.f0064=f32[63*n+c]; m.f0065=f32[64*n+c]; m.f0066=f32[65*n+c]; m.f0067=f32[66*n+c]; m.f0068=f32[67*n+c]; m.f0069=f32[68*n+c]; m.f0070=f32[69*n+c]; m.f0071=f32[70*n+c]; m.f0072=i32[33*n+c]; m.f0073[0]=f32[71*n+c]; m.f0073[1]=f32[72*n+c]; m.f0073[2]=f32[73*n+c]; m.f0073[3]=f32[74*n+c]; m.f0073[4]=f32[75*n+c]; m.f0073[5]=f32[76*n+c]; m.f0073[6]=f32[77*n+c]; m.f0073[7]=f32[78*n+c]; m.f0073[8]=f32[79*n+c]; m.f0073[9]=f32[80*n+c]; m.f0073[10]=f32[81*n+c]; m.f0073[11]=f32[82*n+c]; m.f0074[0]=f32[83*n+c]; m.f0074[1]=f32[84*n+c]; m.f0074[2]=f32[85*n+c]; m.f0074[3]=f32[86*n+c]; m.f0074[4]=f32[87*n+c]; m.f0074[5]=f32[88*n+c]; m.f0074[6]=f32[89*n+c]; m.f0074[7]=f32[90*n+c]; m.f0074[8]=f32[91*n+c]; m.f0074[9]=f32[92*n+c]; m.f0074[10]=f32[93*n+c]; m.f0074[11]=f32[94*n+c]; m.f0075=f32[95*n+c]; m.f0076=f32[96*n+c]; m.f0077=f32[97*n+c]; m.f0078=i32[34*n+c]; m.f0079=f32[98*n+c]; m.f0080=f32[99*n+c]; m.f0081=f32[100*n+c]; m.f0082=f32[101*n+c]; m.f0083=f32[102*n+c]; m.f0084=f32[103*n+c]; m.f0085=f32[104*n+c]; m.f0086=f32[105*n+c]; m.f0087=f32[106*n+c]; m.f0088=f32[107*n+c]; m.f0089=f32[108*n+c]; m.f0090=f32[109*n+c]; m.f0091=f32[110*n+c]; m.f0092=f32[111*n+c]; m.f0093=f32[112*n+c]; m.f0094=f32[113*n+c]; m.f0095=f32[114*n+c]; m.f0096=f32[115*n+c]; m.f0097=f32[116*n+c]; m.f0098=f32[117*n+c]; m.f0099=f32[118*n+c]; m.f0100=f32[119*n+c]; m.f0101=f32[120*n+c]; m.f0102=f32[121*n+c]; m.f0103=f32[122*n+c]; m.f0104=f32[123*n+c]; m.f0105=f32[124*n+c]; m.f0106=f32[125*n+c]; m.f0107=i32[35*n+c]; m.f0108=f32[126*n+c]; m.f0109[0]=f32[127*n+c]; m.f0109[1]=f32[128*n+c]; m.f0110[0]=f32[129*n+c]; m.f0110[1]=f32[130*n+c]; m.f0111[0]=f32[131*n+c]; m.f0111[1]=f32[132*n+c]; m.f0112[0]=f32[133*n+c]; m.f0112[1]=f32[134*n+c]; m.f0113=i32[36*n+c]; m.f0114=i32[37*n+c]; m.f0115=i32[38*n+c]; m.f0116=i32[39*n+c]; m.f0117=i32[40*n+c]; m.f0118=i32[41*n+c]; m.f0119=i32[42*n+c]; m.f0120=i32[43*n+c]; m.f0121=i32[44*n+c]; m.f0122=i32[45*n+c]; m.f0123=f32[135*n+c]; m.f0124=f32[136*n+c]; m.f0125=f32[137*n+c]; m.f0126=f32[138*n+c]; m.f0127=f32[139*n+c]; m.f0128=f32[140*n+c]; m.f0129=f32[141*n+c]; m.f0130=f32[142*n+c]; m.f0131=f32[143*n+c]; m.f0132=f32[144*n+c]; m.f0133=f32[145*n+c]; m.f0134=f32[146*n+c]; m.f0135=f32[147*n+c]; m.f0136=f32[148*n+c]; m.f0137=f32[149*n+c]; m.f0138=f32[150*n+c]; m.f0139=f32[151*n+c]; m.f0140=f32[152*n+c]; m.f0141=f32[153*n+c]; m.f0142=f32[154*n+c]; m.f0143=f32[155*n+c]; m.f0144=f32[156*n+c]; m.f0145=f32[157*n+c]; m.f0146=f32[158*n+c]; m.f0147=f32[159*n+c]; m.f0148=f32[160*n+c]; m.f0149=f32[161*n+c]; m.f0150=f32[162*n+c]; m.f0151=f32[163*n+c]; m.f0152=f32[164*n+c]; m.f0153=f32[165*n+c]; m.f0154=f32[166*n+c]; m.f0155=f32[167*n+c]; m.f0156=f32[168*n+c]; m.f0157=f32[169*n+c]; m.f0158=f32[170*n+c]; m.f0159[0]=f32[171*n+c]; m.f0159[1]=f32[172*n+c]; m.f0160[0]=f32[173*n+c]; m.f0160[1]=f32[174*n+c]; m.f0161[0]=f32[175*n+c]; m.f0161[1]=f32[176*n+c]; m.f0162[0]=f32[177*n+c]; m.f0162[1]=f32[178*n+c]; m.f0163[0]=f32[179*n+c]; m.f0163[1]=f32[180*n+c]; m.f0164=f32[181*n+c]; m.f0165=f32[182*n+c]; m.f0166[0]=f32[183*n+c]; m.f0166[1]=f32[184*n+c]; m.f0167=f32[185*n+c]; m.f0168=f32[186*n+c]; m.f0169=f32[187*n+c]; m.f0170=i32[46*n+c]; m.f0171=f32[188*n+c]; m.f0172=f32[189*n+c]; m.f0173=f32[190*n+c]; m.f0174=f32[191*n+c]; m.f0175=f32[192*n+c]; m.f0176=f32[193*n+c]; m.f0177=f32[194*n+c]; m.f0178=f32[195*n+c]; m.f0179=f32[196*n+c]; m.f0180=f32[197*n+c]; m.f0181=f32[198*n+c]; m.f0182=f32[199*n+c]; m.f0183=f32[200*n+c]; m.f0184=f32[201*n+c]; m.f0185=f32[202*n+c]; m.f0186=f32[203*n+c]; m.f0187=f32[204*n+c]; m.f0188=f32[205*n+c]; m.f0189=f32[206*n+c]; m.f0190=f32[207*n+c]; m.f0191=f32[208*n+c]; m.f0192=f32[209*n+c]; m.f0193=f32[210*n+c]; m.f0194=f32[211*n+c]; m.f0195=f32[212*n+c]; m.f0196=f32[213*n+c]; m.f0197=f32[214*n+c]; m.f0198=f32[215*n+c]; m.f0199=f32[216*n+c]; m.f0200=f32[217*n+c]; m.f0201=f32[218*n+c]; m.f0202=f32[219*n+c]; m.f0203=f32[220*n+c]; m.f0204=f32[221*n+c]; m.f0205=f32[222*n+c]; m.f0206=i32[47*n+c]; m.f0207[0]=f32[223*n+c]; m.f0207[1]=f32[224*n+c]; m.f0208[0]=f32[225*n+c]; m.f0208[1]=f32[226*n+c]; m.f0209=f32[227*n+c]; m.f0210=f32[228*n+c]; m.f0211=f32[229*n+c]; m.f0212=f32[230*n+c]; m.f0213=f32[231*n+c]; m.f0214=i32[48*n+c]; m.f0215=i32[49*n+c]; m.f0216[0]=i32[50*n+c]; m.f0216[1]=i32[51*n+c]; m.f0216[2]=i32[52*n+c]; m.f0216[3]=i32[53*n+c]; m.f0216[4]=i32[54*n+c]; m.f0216[5]=i32[55*n+c]; m.f0216[6]=i32[56*n+c]; m.f0217[0]=f32[232*n+c]; m.f0217[1]=f32[233*n+c]; m.f0217[2]=f32[234*n+c]; m.f0217[3]=f32[235*n+c]; m.f0217[4]=f32[236*n+c]; m.f0217[5]=f32[237*n+c]; m.f0217[6]=f32[238*n+c]; m.f0218[0]=f32[239*n+c]; m.f0218[1]=f32[240*n+c]; m.f0218[2]=f32[241*n+c]; m.f0218[3]=f32[242*n+c]; m.f0218[4]=f32[243*n+c]; m.f0218[5]=f32[244*n+c]; m.f0218[6]=f32[245*n+c]; m.f0219[0]=f32[246*n+c]; m.f0219[1]=f32[247*n+c]; m.f0219[2]=f32[248*n+c]; m.f0219[3]=f32[249*n+c]; m.f0219[4]=f32[250*n+c]; m.f0219[5]=f32[251*n+c]; m.f0219[6]=f32[252*n+c]; m.f0220[0]=f32[253*n+c]; m.f0220[1]=f32[254*n+c]; m.f0220[2]=f32[255*n+c]; m.f0220[3]=f32[256*n+c]; m.f0220[4]=f32[257*n+c]; m.f0220[5]=f32[258*n+c]; m.f0220[6]=f32[259*n+c]; m.f0221=f32[260*n+c]; m.f0222=f32[261*n+c]; m.f0223=f32[262*n+c]; m.f0224=f32[263*n+c]; m.f0225=f32[264*n+c]; m.f0226=f32[265*n+c]; m.f0227=f32[266*n+c]; m.f0228=f32[267*n+c]; m.f0229=f32[268*n+c]; m.f0230[0]=f32[269*n+c]; m.f0230[1]=f32[270*n+c]; m.f0231[0]=f32[271*n+c]; m.f0231[1]=f32[272*n+c]; m.f0232[0]=f32[273*n+c]; m.f0232[1]=f32[274*n+c]; m.f0233[0]=f32[275*n+c]; m.f0233[1]=f32[276*n+c]; m.f0234[0]=f32[277*n+c]; m.f0234[1]=f32[278*n+c]; m.f0235[0]=f32[279*n+c]; m.f0235[1]=f32[280*n+c]; m.f0236[0]=f32[281*n+c]; m.f0236[1]=f32[282*n+c]; m.f0237[0]=f32[283*n+c]; m.f0237[1]=f32[284*n+c]; m.f0238[0]=f32[285*n+c]; m.f0238[1]=f32[286*n+c]; m.f0239[0]=f32[287*n+c]; m.f0239[1]=f32[288*n+c]; m.f0240[0]=f32[289*n+c]; m.f0240[1]=f32[290*n+c]; m.f0241[0]=f32[291*n+c]; m.f0241[1]=f32[292*n+c]; m.f0242[0]=f32[293*n+c]; m.f0242[1]=f32[294*n+c]; m.f0243[0]=f32[295*n+c]; m.f0243[1]=f32[296*n+c]; m.f0244[0]=f32[297*n+c]; m.f0244[1]=f32[298*n+c]; m.f0245[0]=f32[299*n+c]; m.f0245[1]=f32[300*n+c]; m.f0246[0]=f32[301*n+c]; m.f0246[1]=f32[302*n+c]; m.f0247[0]=f32[303*n+c]; m.f0247[1]=f32[304*n+c]; m.f0248=f32[305*n+c]; m.f0249=f32[306*n+c]; m.f0250=f32[307*n+c]; m.f0251=f32[308*n+c]; m.f0252=f32[309*n+c]; m.f0253=f32[310*n+c]; m.f0254=f32[311*n+c]; m.f0255=f32[312*n+c]; m.f0256=f32[313*n+c]; m.f0257=f32[314*n+c]; m.f0258=f32[315*n+c]; m.f0259=f32[316*n+c]; m.f0260=f32[317*n+c]; m.f0261=f32[318*n+c]; m.f0262=f32[319*n+c]; m.f0263=f32[320*n+c]; m.f0264=f32[321*n+c]; m.f0265=f32[322*n+c]; m.f0266=f32[323*n+c]; m.f0267=f32[324*n+c]; m.f0268=f32[325*n+c]; m.f0269=f32[326*n+c]; m.f0270=f32[327*n+c]; m.f0271=f32[328*n+c]; m.f0272=f32[329*n+c]; m.f0273=f32[330*n+c]; m.f0274=f32[331*n+c]; m.f0275=f32[332*n+c]; m.f0276=f32[333*n+c]; m.f0277=f32[334*n+c]; m.f0278=f32[335*n+c]; m.f0279=f32[336*n+c]; m.f0280=f32[337*n+c]; m.f0281=f32[338*n+c]; m.f0282=f32[339*n+c]; m.f0283=f32[340*n+c]; m.f0284=f32[341*n+c]; m.f0285=f32[342*n+c]; m.f0286=f32[343*n+c]; m.f0287=f32[344*n+c]; m.f0288=f32[345*n+c]; m.f0289=f32[346*n+c]; m.f0290=f32[347*n+c]; m.f0291=f32[348*n+c]; m.f0292=f32[349*n+c]; m.f0293=f32[350*n+c]; m.f0294=f32[351*n+c]; m.f0295=f32[352*n+c]; m.f0296=f32[353*n+c]; m.f0297=f32[354*n+c]; m.f0298=f32[355*n+c]; m.f0299=f32[356*n+c]; m.f0300=f32[357*n+c]; m.f0301=f32[358*n+c]; m.f0302=f32[359*n+c]; m.f0303=f32[360*n+c]; m.f0304=f32[361*n+c]; m.f0305=f32[362*n+c]; m.f0306=f32[363*n+c]; m.f0307=f32[364*n+c]; m.f0308=f32[365*n+c]; m.f0309=f32[366*n+c]; m.f0310=f32[367*n+c]; m.f0311=f32[368*n+c]; m.f0312=f32[369*n+c]; m.f0313=f32[370*n+c]; m.f0314=f32[371*n+c]; m.f0315=f32[372*n+c]; m.f0316=f32[373*n+c]; m.f0317=f32[374*n+c]; m.f0318=f32[375*n+c]; m.f0319=f32[376*n+c]; m.f0320=f32[377*n+c]; m.f0321=f32[378*n+c]; m.f0322=f32[379*n+c]; m.f0323=f32[380*n+c]; m.f0324=f32[381*n+c]; m.f0325=f32[382*n+c]; m.f0326=f32[383*n+c]; m.f0327=f32[384*n+c]; m.f0328=f32[385*n+c]; m.f0329=f32[386*n+c]; m.f0330=f32[387*n+c]; m.f0331=f32[388*n+c]; m.f0332=f32[389*n+c]; m.f0333=f32[390*n+c]; m.f0334=f32[391*n+c]; m.f0335=f32[392*n+c]; m.f0336=f32[393*n+c]; m.f0337=f32[394*n+c]; m.f0338=i32[57*n+c]; m.f0339=f32[395*n+c]; m.f0340=f32[396*n+c]; m.f0341=f32[397*n+c]; m.f0342=f32[398*n+c]; m.f0343=f32[399*n+c]; m.f0344=f32[400*n+c]; m.f0345=f32[401*n+c]; m.f0346=f32[402*n+c]; m.f0347=f32[403*n+c]; m.f0348=f32[404*n+c]; m.f0349=f32[405*n+c]; m.f0350=f32[406*n+c]; m.f0351=f32[407*n+c]; m.f0352=f32[408*n+c]; m.f0353=f32[409*n+c]; m.f0354=f32[410*n+c]; m.f0355=f32[411*n+c]; m.f0356=f32[412*n+c]; m.f0357=f32[413*n+c]; m.f0358=f32[414*n+c]; m.f0359=f32[415*n+c]; m.f0360=f32[416*n+c]; m.f0361=f32[417*n+c]; m.f0362=f32[418*n+c]; m.f0363=f32[419*n+c]; m.f0364=f32[420*n+c]; m.f0365=f32[421*n+c]; m.f0366=f32[422*n+c]; m.f0367=f32[423*n+c]; m.f0368=f32[424*n+c]; m.f0369=f32[425*n+c]; m.f0370=f32[426*n+c]; m.f0371=f32[427*n+c]; m.f0372=f32[428*n+c]; m.f0373=f32[429*n+c]; m.f0374=f32[430*n+c]; m.f0375=f32[431*n+c]; m.f0376=f32[432*n+c]; m.f0377=f32[433*n+c]; m.f0378=f32[434*n+c]; m.f0379=f32[435*n+c]; m.f0380=f32[436*n+c]; m.f0381=f32[437*n+c]; m.f0382=f32[438*n+c]; m.f0383=f32[439*n+c]; m.f0384=f32[440*n+c]; m.f0385=f32[441*n+c]; m.f0386=f32[442*n+c]; m.f0387=f32[443*n+c]; m.f0388=f32[444*n+c]; m.f0389=f32[445*n+c]; m.f0390=i32[58*n+c]; m.f0391[0]=f32[446*n+c]; m.f0391[1]=f32[447*n+c]; m.f0391[2]=f32[448*n+c]; m.f0391[3]=f32[449*n+c]; m.f0392[0]=f32[450*n+c]; m.f0392[1]=f32[451*n+c]; m.f0392[2]=f32[452*n+c]; m.f0392[3]=f32[453*n+c]; m.f0393[0]=f32[454*n+c]; m.f0393[1]=f32[455*n+c]; m.f0393[2]=f32[456*n+c]; m.f0393[3]=f32[457*n+c]; m.f0394[0]=f32[458*n+c]; m.f0394[1]=f32[459*n+c]; m.f0394[2]=f32[460*n+c]; m.f0394[3]=f32[461*n+c]; m.f0395[0]=f32[462*n+c]; m.f0395[1]=f32[463*n+c]; m.f0395[2]=f32[464*n+c]; m.f0395[3]=f32[465*n+c]; m.f0396[0]=f32[466*n+c]; m.f0396[1]=f32[467*n+c]; m.f0396[2]=f32[468*n+c]; m.f0396[3]=f32[469*n+c]; m.f0397[0]=f32[470*n+c]; m.f0397[1]=f32[471*n+c]; m.f0397[2]=f32[472*n+c]; m.f0397[3]=f32[473*n+c]; m.f0398[0]=f32[474*n+c]; m.f0398[1]=f32[475*n+c]; m.f0398[2]=f32[476*n+c]; m.f0398[3]=f32[477*n+c]; m.f0399[0]=f32[478*n+c]; m.f0399[1]=f32[479*n+c]; m.f0399[2]=f32[480*n+c]; m.f0400[0]=f32[481*n+c]; m.f0400[1]=f32[482*n+c]; m.f0400[2]=f32[483*n+c]; m.f0401[0]=f32[484*n+c]; m.f0401[1]=f32[485*n+c]; m.f0401[2]=f32[486*n+c]; m.f0402[0]=f32[487*n+c]; m.f0402[1]=f32[488*n+c]; m.f0402[2]=f32[489*n+c]; m.f0403[0]=f32[490*n+c]; m.f0403[1]=f32[491*n+c]; m.f0403[2]=f32[492*n+c]; m.f0404[0]=f32[493*n+c]; m.f0404[1]=f32[494*n+c]; m.f0404[2]=f32[495*n+c]; m.f0405[0]=f32[496*n+c]; m.f0405[1]=f32[497*n+c]; m.f0405[2]=f32[498*n+c]; m.f0406=f32[499*n+c]; m.f0407=f32[500*n+c];
    if(m.f0000!=4||m.f0001!=3||m.f0170!=2||m.f0078<1||m.f0078>4){
        status[c]=4;if(sticky_status)sticky_status[c]|=4;return;
    }
    m.status=0;
    if(stage==5)m.f0013=m.f0011+(m.f0014-1)*(double)m.f0005;
    if(stage==0||stage==5)r0000_(m,&m.f0014);
    if(stage==1||stage==5)r0009_(m);
    if(stage==2||stage==5)r0010_(m);
    if(stage==3||stage==5)r0014_(m);
    if((stage==4||stage==5)&&!(m.status&2))r0090_(m);
    if(stage==6)r0027_(m);
    if(stage==7)r0028_(m);
    if(advance&&!(m.status&2)){m.f0014+=1;m.f0016+=m.f0005;}
    f64[0*n+c]=m.f0011; f64[1*n+c]=m.f0012; f64[2*n+c]=m.f0013; f64[3*n+c]=m.f0016; i32[0*n+c]=m.f0000; i32[1*n+c]=m.f0001; i32[2*n+c]=m.f0002; i32[3*n+c]=m.f0003; i32[4*n+c]=m.f0004; f32[0*n+c]=m.f0005; i32[5*n+c]=m.f0006; i32[6*n+c]=m.f0007; i32[7*n+c]=m.f0008; i32[8*n+c]=m.f0009; i32[9*n+c]=m.f0010; i32[10*n+c]=m.f0014; i32[11*n+c]=m.f0015; f32[1*n+c]=m.f0017; f32[2*n+c]=m.f0018; f32[3*n+c]=m.f0019; f32[4*n+c]=m.f0020; f32[5*n+c]=m.f0021; i32[12*n+c]=m.f0022; i32[13*n+c]=m.f0023; i32[14*n+c]=m.f0024; i32[15*n+c]=m.f0025; f32[6*n+c]=m.f0026[0]; f32[7*n+c]=m.f0026[1]; f32[8*n+c]=m.f0026[2]; f32[9*n+c]=m.f0026[3]; f32[10*n+c]=m.f0027[0]; f32[11*n+c]=m.f0027[1]; f32[12*n+c]=m.f0027[2]; f32[13*n+c]=m.f0027[3]; f32[14*n+c]=m.f0027[4]; f32[15*n+c]=m.f0027[5]; f32[16*n+c]=m.f0027[6]; f32[17*n+c]=m.f0028[0]; f32[18*n+c]=m.f0028[1]; f32[19*n+c]=m.f0028[2]; f32[20*n+c]=m.f0028[3]; f32[21*n+c]=m.f0028[4]; f32[22*n+c]=m.f0028[5]; f32[23*n+c]=m.f0028[6]; i32[16*n+c]=m.f0029; i32[17*n+c]=m.f0030; i32[18*n+c]=m.f0031; i32[19*n+c]=m.f0032; i32[20*n+c]=m.f0033; i32[21*n+c]=m.f0034; i32[22*n+c]=m.f0035; i32[23*n+c]=m.f0036; i32[24*n+c]=m.f0037; i32[25*n+c]=m.f0038; i32[26*n+c]=m.f0039; i32[27*n+c]=m.f0040; i32[28*n+c]=m.f0041; i32[29*n+c]=m.f0042; i32[30*n+c]=m.f0043; i32[31*n+c]=m.f0044; i32[32*n+c]=m.f0045; f32[24*n+c]=m.f0046[0]; f32[25*n+c]=m.f0046[1]; f32[26*n+c]=m.f0046[2]; f32[27*n+c]=m.f0046[3]; f32[28*n+c]=m.f0047[0]; f32[29*n+c]=m.f0047[1]; f32[30*n+c]=m.f0047[2]; f32[31*n+c]=m.f0047[3]; f32[32*n+c]=m.f0048[0]; f32[33*n+c]=m.f0048[1]; f32[34*n+c]=m.f0048[2]; f32[35*n+c]=m.f0048[3]; f32[36*n+c]=m.f0049[0]; f32[37*n+c]=m.f0049[1]; f32[38*n+c]=m.f0049[2]; f32[39*n+c]=m.f0049[3]; f32[40*n+c]=m.f0050[0]; f32[41*n+c]=m.f0050[1]; f32[42*n+c]=m.f0050[2]; f32[43*n+c]=m.f0050[3]; f32[44*n+c]=m.f0051[0]; f32[45*n+c]=m.f0051[1]; f32[46*n+c]=m.f0051[2]; f32[47*n+c]=m.f0051[3]; f32[48*n+c]=m.f0052[0]; f32[49*n+c]=m.f0052[1]; f32[50*n+c]=m.f0052[2]; f32[51*n+c]=m.f0052[3]; f32[52*n+c]=m.f0053; f32[53*n+c]=m.f0054; f32[54*n+c]=m.f0055; f32[55*n+c]=m.f0056; f32[56*n+c]=m.f0057; f32[57*n+c]=m.f0058; f32[58*n+c]=m.f0059; f32[59*n+c]=m.f0060; f32[60*n+c]=m.f0061; f32[61*n+c]=m.f0062; f32[62*n+c]=m.f0063; f32[63*n+c]=m.f0064; f32[64*n+c]=m.f0065; f32[65*n+c]=m.f0066; f32[66*n+c]=m.f0067; f32[67*n+c]=m.f0068; f32[68*n+c]=m.f0069; f32[69*n+c]=m.f0070; f32[70*n+c]=m.f0071; i32[33*n+c]=m.f0072; f32[71*n+c]=m.f0073[0]; f32[72*n+c]=m.f0073[1]; f32[73*n+c]=m.f0073[2]; f32[74*n+c]=m.f0073[3]; f32[75*n+c]=m.f0073[4]; f32[76*n+c]=m.f0073[5]; f32[77*n+c]=m.f0073[6]; f32[78*n+c]=m.f0073[7]; f32[79*n+c]=m.f0073[8]; f32[80*n+c]=m.f0073[9]; f32[81*n+c]=m.f0073[10]; f32[82*n+c]=m.f0073[11]; f32[83*n+c]=m.f0074[0]; f32[84*n+c]=m.f0074[1]; f32[85*n+c]=m.f0074[2]; f32[86*n+c]=m.f0074[3]; f32[87*n+c]=m.f0074[4]; f32[88*n+c]=m.f0074[5]; f32[89*n+c]=m.f0074[6]; f32[90*n+c]=m.f0074[7]; f32[91*n+c]=m.f0074[8]; f32[92*n+c]=m.f0074[9]; f32[93*n+c]=m.f0074[10]; f32[94*n+c]=m.f0074[11]; f32[95*n+c]=m.f0075; f32[96*n+c]=m.f0076; f32[97*n+c]=m.f0077; i32[34*n+c]=m.f0078; f32[98*n+c]=m.f0079; f32[99*n+c]=m.f0080; f32[100*n+c]=m.f0081; f32[101*n+c]=m.f0082; f32[102*n+c]=m.f0083; f32[103*n+c]=m.f0084; f32[104*n+c]=m.f0085; f32[105*n+c]=m.f0086; f32[106*n+c]=m.f0087; f32[107*n+c]=m.f0088; f32[108*n+c]=m.f0089; f32[109*n+c]=m.f0090; f32[110*n+c]=m.f0091; f32[111*n+c]=m.f0092; f32[112*n+c]=m.f0093; f32[113*n+c]=m.f0094; f32[114*n+c]=m.f0095; f32[115*n+c]=m.f0096; f32[116*n+c]=m.f0097; f32[117*n+c]=m.f0098; f32[118*n+c]=m.f0099; f32[119*n+c]=m.f0100; f32[120*n+c]=m.f0101; f32[121*n+c]=m.f0102; f32[122*n+c]=m.f0103; f32[123*n+c]=m.f0104; f32[124*n+c]=m.f0105; f32[125*n+c]=m.f0106; i32[35*n+c]=m.f0107; f32[126*n+c]=m.f0108; f32[127*n+c]=m.f0109[0]; f32[128*n+c]=m.f0109[1]; f32[129*n+c]=m.f0110[0]; f32[130*n+c]=m.f0110[1]; f32[131*n+c]=m.f0111[0]; f32[132*n+c]=m.f0111[1]; f32[133*n+c]=m.f0112[0]; f32[134*n+c]=m.f0112[1]; i32[36*n+c]=m.f0113; i32[37*n+c]=m.f0114; i32[38*n+c]=m.f0115; i32[39*n+c]=m.f0116; i32[40*n+c]=m.f0117; i32[41*n+c]=m.f0118; i32[42*n+c]=m.f0119; i32[43*n+c]=m.f0120; i32[44*n+c]=m.f0121; i32[45*n+c]=m.f0122; f32[135*n+c]=m.f0123; f32[136*n+c]=m.f0124; f32[137*n+c]=m.f0125; f32[138*n+c]=m.f0126; f32[139*n+c]=m.f0127; f32[140*n+c]=m.f0128; f32[141*n+c]=m.f0129; f32[142*n+c]=m.f0130; f32[143*n+c]=m.f0131; f32[144*n+c]=m.f0132; f32[145*n+c]=m.f0133; f32[146*n+c]=m.f0134; f32[147*n+c]=m.f0135; f32[148*n+c]=m.f0136; f32[149*n+c]=m.f0137; f32[150*n+c]=m.f0138; f32[151*n+c]=m.f0139; f32[152*n+c]=m.f0140; f32[153*n+c]=m.f0141; f32[154*n+c]=m.f0142; f32[155*n+c]=m.f0143; f32[156*n+c]=m.f0144; f32[157*n+c]=m.f0145; f32[158*n+c]=m.f0146; f32[159*n+c]=m.f0147; f32[160*n+c]=m.f0148; f32[161*n+c]=m.f0149; f32[162*n+c]=m.f0150; f32[163*n+c]=m.f0151; f32[164*n+c]=m.f0152; f32[165*n+c]=m.f0153; f32[166*n+c]=m.f0154; f32[167*n+c]=m.f0155; f32[168*n+c]=m.f0156; f32[169*n+c]=m.f0157; f32[170*n+c]=m.f0158; f32[171*n+c]=m.f0159[0]; f32[172*n+c]=m.f0159[1]; f32[173*n+c]=m.f0160[0]; f32[174*n+c]=m.f0160[1]; f32[175*n+c]=m.f0161[0]; f32[176*n+c]=m.f0161[1]; f32[177*n+c]=m.f0162[0]; f32[178*n+c]=m.f0162[1]; f32[179*n+c]=m.f0163[0]; f32[180*n+c]=m.f0163[1]; f32[181*n+c]=m.f0164; f32[182*n+c]=m.f0165; f32[183*n+c]=m.f0166[0]; f32[184*n+c]=m.f0166[1]; f32[185*n+c]=m.f0167; f32[186*n+c]=m.f0168; f32[187*n+c]=m.f0169; i32[46*n+c]=m.f0170; f32[188*n+c]=m.f0171; f32[189*n+c]=m.f0172; f32[190*n+c]=m.f0173; f32[191*n+c]=m.f0174; f32[192*n+c]=m.f0175; f32[193*n+c]=m.f0176; f32[194*n+c]=m.f0177; f32[195*n+c]=m.f0178; f32[196*n+c]=m.f0179; f32[197*n+c]=m.f0180; f32[198*n+c]=m.f0181; f32[199*n+c]=m.f0182; f32[200*n+c]=m.f0183; f32[201*n+c]=m.f0184; f32[202*n+c]=m.f0185; f32[203*n+c]=m.f0186; f32[204*n+c]=m.f0187; f32[205*n+c]=m.f0188; f32[206*n+c]=m.f0189; f32[207*n+c]=m.f0190; f32[208*n+c]=m.f0191; f32[209*n+c]=m.f0192; f32[210*n+c]=m.f0193; f32[211*n+c]=m.f0194; f32[212*n+c]=m.f0195; f32[213*n+c]=m.f0196; f32[214*n+c]=m.f0197; f32[215*n+c]=m.f0198; f32[216*n+c]=m.f0199; f32[217*n+c]=m.f0200; f32[218*n+c]=m.f0201; f32[219*n+c]=m.f0202; f32[220*n+c]=m.f0203; f32[221*n+c]=m.f0204; f32[222*n+c]=m.f0205; i32[47*n+c]=m.f0206; f32[223*n+c]=m.f0207[0]; f32[224*n+c]=m.f0207[1]; f32[225*n+c]=m.f0208[0]; f32[226*n+c]=m.f0208[1]; f32[227*n+c]=m.f0209; f32[228*n+c]=m.f0210; f32[229*n+c]=m.f0211; f32[230*n+c]=m.f0212; f32[231*n+c]=m.f0213; i32[48*n+c]=m.f0214; i32[49*n+c]=m.f0215; i32[50*n+c]=m.f0216[0]; i32[51*n+c]=m.f0216[1]; i32[52*n+c]=m.f0216[2]; i32[53*n+c]=m.f0216[3]; i32[54*n+c]=m.f0216[4]; i32[55*n+c]=m.f0216[5]; i32[56*n+c]=m.f0216[6]; f32[232*n+c]=m.f0217[0]; f32[233*n+c]=m.f0217[1]; f32[234*n+c]=m.f0217[2]; f32[235*n+c]=m.f0217[3]; f32[236*n+c]=m.f0217[4]; f32[237*n+c]=m.f0217[5]; f32[238*n+c]=m.f0217[6]; f32[239*n+c]=m.f0218[0]; f32[240*n+c]=m.f0218[1]; f32[241*n+c]=m.f0218[2]; f32[242*n+c]=m.f0218[3]; f32[243*n+c]=m.f0218[4]; f32[244*n+c]=m.f0218[5]; f32[245*n+c]=m.f0218[6]; f32[246*n+c]=m.f0219[0]; f32[247*n+c]=m.f0219[1]; f32[248*n+c]=m.f0219[2]; f32[249*n+c]=m.f0219[3]; f32[250*n+c]=m.f0219[4]; f32[251*n+c]=m.f0219[5]; f32[252*n+c]=m.f0219[6]; f32[253*n+c]=m.f0220[0]; f32[254*n+c]=m.f0220[1]; f32[255*n+c]=m.f0220[2]; f32[256*n+c]=m.f0220[3]; f32[257*n+c]=m.f0220[4]; f32[258*n+c]=m.f0220[5]; f32[259*n+c]=m.f0220[6]; f32[260*n+c]=m.f0221; f32[261*n+c]=m.f0222; f32[262*n+c]=m.f0223; f32[263*n+c]=m.f0224; f32[264*n+c]=m.f0225; f32[265*n+c]=m.f0226; f32[266*n+c]=m.f0227; f32[267*n+c]=m.f0228; f32[268*n+c]=m.f0229; f32[269*n+c]=m.f0230[0]; f32[270*n+c]=m.f0230[1]; f32[271*n+c]=m.f0231[0]; f32[272*n+c]=m.f0231[1]; f32[273*n+c]=m.f0232[0]; f32[274*n+c]=m.f0232[1]; f32[275*n+c]=m.f0233[0]; f32[276*n+c]=m.f0233[1]; f32[277*n+c]=m.f0234[0]; f32[278*n+c]=m.f0234[1]; f32[279*n+c]=m.f0235[0]; f32[280*n+c]=m.f0235[1]; f32[281*n+c]=m.f0236[0]; f32[282*n+c]=m.f0236[1]; f32[283*n+c]=m.f0237[0]; f32[284*n+c]=m.f0237[1]; f32[285*n+c]=m.f0238[0]; f32[286*n+c]=m.f0238[1]; f32[287*n+c]=m.f0239[0]; f32[288*n+c]=m.f0239[1]; f32[289*n+c]=m.f0240[0]; f32[290*n+c]=m.f0240[1]; f32[291*n+c]=m.f0241[0]; f32[292*n+c]=m.f0241[1]; f32[293*n+c]=m.f0242[0]; f32[294*n+c]=m.f0242[1]; f32[295*n+c]=m.f0243[0]; f32[296*n+c]=m.f0243[1]; f32[297*n+c]=m.f0244[0]; f32[298*n+c]=m.f0244[1]; f32[299*n+c]=m.f0245[0]; f32[300*n+c]=m.f0245[1]; f32[301*n+c]=m.f0246[0]; f32[302*n+c]=m.f0246[1]; f32[303*n+c]=m.f0247[0]; f32[304*n+c]=m.f0247[1]; f32[305*n+c]=m.f0248; f32[306*n+c]=m.f0249; f32[307*n+c]=m.f0250; f32[308*n+c]=m.f0251; f32[309*n+c]=m.f0252; f32[310*n+c]=m.f0253; f32[311*n+c]=m.f0254; f32[312*n+c]=m.f0255; f32[313*n+c]=m.f0256; f32[314*n+c]=m.f0257; f32[315*n+c]=m.f0258; f32[316*n+c]=m.f0259; f32[317*n+c]=m.f0260; f32[318*n+c]=m.f0261; f32[319*n+c]=m.f0262; f32[320*n+c]=m.f0263; f32[321*n+c]=m.f0264; f32[322*n+c]=m.f0265; f32[323*n+c]=m.f0266; f32[324*n+c]=m.f0267; f32[325*n+c]=m.f0268; f32[326*n+c]=m.f0269; f32[327*n+c]=m.f0270; f32[328*n+c]=m.f0271; f32[329*n+c]=m.f0272; f32[330*n+c]=m.f0273; f32[331*n+c]=m.f0274; f32[332*n+c]=m.f0275; f32[333*n+c]=m.f0276; f32[334*n+c]=m.f0277; f32[335*n+c]=m.f0278; f32[336*n+c]=m.f0279; f32[337*n+c]=m.f0280; f32[338*n+c]=m.f0281; f32[339*n+c]=m.f0282; f32[340*n+c]=m.f0283; f32[341*n+c]=m.f0284; f32[342*n+c]=m.f0285; f32[343*n+c]=m.f0286; f32[344*n+c]=m.f0287; f32[345*n+c]=m.f0288; f32[346*n+c]=m.f0289; f32[347*n+c]=m.f0290; f32[348*n+c]=m.f0291; f32[349*n+c]=m.f0292; f32[350*n+c]=m.f0293; f32[351*n+c]=m.f0294; f32[352*n+c]=m.f0295; f32[353*n+c]=m.f0296; f32[354*n+c]=m.f0297; f32[355*n+c]=m.f0298; f32[356*n+c]=m.f0299; f32[357*n+c]=m.f0300; f32[358*n+c]=m.f0301; f32[359*n+c]=m.f0302; f32[360*n+c]=m.f0303; f32[361*n+c]=m.f0304; f32[362*n+c]=m.f0305; f32[363*n+c]=m.f0306; f32[364*n+c]=m.f0307; f32[365*n+c]=m.f0308; f32[366*n+c]=m.f0309; f32[367*n+c]=m.f0310; f32[368*n+c]=m.f0311; f32[369*n+c]=m.f0312; f32[370*n+c]=m.f0313; f32[371*n+c]=m.f0314; f32[372*n+c]=m.f0315; f32[373*n+c]=m.f0316; f32[374*n+c]=m.f0317; f32[375*n+c]=m.f0318; f32[376*n+c]=m.f0319; f32[377*n+c]=m.f0320; f32[378*n+c]=m.f0321; f32[379*n+c]=m.f0322; f32[380*n+c]=m.f0323; f32[381*n+c]=m.f0324; f32[382*n+c]=m.f0325; f32[383*n+c]=m.f0326; f32[384*n+c]=m.f0327; f32[385*n+c]=m.f0328; f32[386*n+c]=m.f0329; f32[387*n+c]=m.f0330; f32[388*n+c]=m.f0331; f32[389*n+c]=m.f0332; f32[390*n+c]=m.f0333; f32[391*n+c]=m.f0334; f32[392*n+c]=m.f0335; f32[393*n+c]=m.f0336; f32[394*n+c]=m.f0337; i32[57*n+c]=m.f0338; f32[395*n+c]=m.f0339; f32[396*n+c]=m.f0340; f32[397*n+c]=m.f0341; f32[398*n+c]=m.f0342; f32[399*n+c]=m.f0343; f32[400*n+c]=m.f0344; f32[401*n+c]=m.f0345; f32[402*n+c]=m.f0346; f32[403*n+c]=m.f0347; f32[404*n+c]=m.f0348; f32[405*n+c]=m.f0349; f32[406*n+c]=m.f0350; f32[407*n+c]=m.f0351; f32[408*n+c]=m.f0352; f32[409*n+c]=m.f0353; f32[410*n+c]=m.f0354; f32[411*n+c]=m.f0355; f32[412*n+c]=m.f0356; f32[413*n+c]=m.f0357; f32[414*n+c]=m.f0358; f32[415*n+c]=m.f0359; f32[416*n+c]=m.f0360; f32[417*n+c]=m.f0361; f32[418*n+c]=m.f0362; f32[419*n+c]=m.f0363; f32[420*n+c]=m.f0364; f32[421*n+c]=m.f0365; f32[422*n+c]=m.f0366; f32[423*n+c]=m.f0367; f32[424*n+c]=m.f0368; f32[425*n+c]=m.f0369; f32[426*n+c]=m.f0370; f32[427*n+c]=m.f0371; f32[428*n+c]=m.f0372; f32[429*n+c]=m.f0373; f32[430*n+c]=m.f0374; f32[431*n+c]=m.f0375; f32[432*n+c]=m.f0376; f32[433*n+c]=m.f0377; f32[434*n+c]=m.f0378; f32[435*n+c]=m.f0379; f32[436*n+c]=m.f0380; f32[437*n+c]=m.f0381; f32[438*n+c]=m.f0382; f32[439*n+c]=m.f0383; f32[440*n+c]=m.f0384; f32[441*n+c]=m.f0385; f32[442*n+c]=m.f0386; f32[443*n+c]=m.f0387; f32[444*n+c]=m.f0388; f32[445*n+c]=m.f0389; i32[58*n+c]=m.f0390; f32[446*n+c]=m.f0391[0]; f32[447*n+c]=m.f0391[1]; f32[448*n+c]=m.f0391[2]; f32[449*n+c]=m.f0391[3]; f32[450*n+c]=m.f0392[0]; f32[451*n+c]=m.f0392[1]; f32[452*n+c]=m.f0392[2]; f32[453*n+c]=m.f0392[3]; f32[454*n+c]=m.f0393[0]; f32[455*n+c]=m.f0393[1]; f32[456*n+c]=m.f0393[2]; f32[457*n+c]=m.f0393[3]; f32[458*n+c]=m.f0394[0]; f32[459*n+c]=m.f0394[1]; f32[460*n+c]=m.f0394[2]; f32[461*n+c]=m.f0394[3]; f32[462*n+c]=m.f0395[0]; f32[463*n+c]=m.f0395[1]; f32[464*n+c]=m.f0395[2]; f32[465*n+c]=m.f0395[3]; f32[466*n+c]=m.f0396[0]; f32[467*n+c]=m.f0396[1]; f32[468*n+c]=m.f0396[2]; f32[469*n+c]=m.f0396[3]; f32[470*n+c]=m.f0397[0]; f32[471*n+c]=m.f0397[1]; f32[472*n+c]=m.f0397[2]; f32[473*n+c]=m.f0397[3]; f32[474*n+c]=m.f0398[0]; f32[475*n+c]=m.f0398[1]; f32[476*n+c]=m.f0398[2]; f32[477*n+c]=m.f0398[3]; f32[478*n+c]=m.f0399[0]; f32[479*n+c]=m.f0399[1]; f32[480*n+c]=m.f0399[2]; f32[481*n+c]=m.f0400[0]; f32[482*n+c]=m.f0400[1]; f32[483*n+c]=m.f0400[2]; f32[484*n+c]=m.f0401[0]; f32[485*n+c]=m.f0401[1]; f32[486*n+c]=m.f0401[2]; f32[487*n+c]=m.f0402[0]; f32[488*n+c]=m.f0402[1]; f32[489*n+c]=m.f0402[2]; f32[490*n+c]=m.f0403[0]; f32[491*n+c]=m.f0403[1]; f32[492*n+c]=m.f0403[2]; f32[493*n+c]=m.f0404[0]; f32[494*n+c]=m.f0404[1]; f32[495*n+c]=m.f0404[2]; f32[496*n+c]=m.f0405[0]; f32[497*n+c]=m.f0405[1]; f32[498*n+c]=m.f0405[2]; f32[499*n+c]=m.f0406; f32[500*n+c]=m.f0407;
    status[c]=m.status;
    if(sticky_status)sticky_status[c]|=m.status;
}

// Preserve the original raw module ABI for native replay callers.
extern "C" __global__ void noahowp_step(
       int n, float *f32, int *i32, double *f64, int *status, int stage, int advance){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    noahowp_step_cell(c,n,f32,i32,f64,status,stage,advance,nullptr);
}

extern "C" __global__ void noahowp_step_monitored(
       int n, float *f32, int *i32, double *f64, int *status, int stage, int advance,
       int *sticky_status){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    noahowp_step_cell(c,n,f32,i32,f64,status,stage,advance,sticky_status);
}

extern "C" __global__ void noahowp_export_bmi(int n, const float*f32, float*out){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    out[c]=f32[428*n+c]*f32[0*n+c];
    out[n+c]=f32[396*n+c]*1000.f;
    out[2*n+c]=f32[427*n+c]*f32[0*n+c];
}

extern "C" __global__ void noahowp_copy_forcing(int n,float*f32,const float*a0,const float*a1,const float*a2,const float*a3,const float*a4,const float*a5,const float*a6,const float*a7){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;f32[212*n+c]=a0[c]; f32[213*n+c]=a1[c]; f32[198*n+c]=a2[c]; f32[197*n+c]=a3[c]; f32[199*n+c]=a4[c]; f32[208*n+c]=a5[c]; f32[207*n+c]=a6[c]; f32[202*n+c]=a7[c];}

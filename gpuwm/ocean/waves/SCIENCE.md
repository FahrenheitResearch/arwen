Equations and numerical choices
===============================

All CUDA and Rust code in this package is original. Published equations were
consulted; no WW3, SWAN or ecWAM implementation was copied or translated.

The prognostic spectrum is elevation variance E(f,theta), m^2 Hz^-1 rad^-1,
with travel direction theta counterclockwise from east and absolute intrinsic
frequency sigma = 2 pi f. Bathymetry is static and currents are absent, so
multiplication of the action equation by sigma gives the conservative variance
balance. Spatial fluxes use finite-depth group velocity, spherical cell areas,
and upwind face values. Directional flux includes great-circle rotation and
bathymetric refraction. Frequency shifting is zero with static depth and no
current. Coastlines absorb outgoing variance; open edges have zero incident
spectrum unless a caller provides incident edge spectra. Spatial propagation is first-order explicit upwind. Directional turning uses
conservative backward-Euler upwind, solved as a cyclic tridiagonal system on
CUDA, one thread per cell/frequency ring. Its M-matrix retains positivity and
directional variance without a directional Courant restriction; coastal depth
gradients therefore do not force subsecond steps on every ocean cell. Frequency
is geometric and direction periodic.
Dispersion sigma^2 = g k tanh(k h) is solved by Newton iteration on CUDA.

Wind uses the WAM cycle 4 Janssen formulation: B = beta rho_a/rho_w
(u*/c)^2 max(cos(theta-theta_w),0)^2 sigma, with beta = 1.2/kappa^2 lambda
(log lambda)^4, lambda = g z_e/c^2 exp(kappa c/|u* cos|), lambda < 1.
The log profile and wave stress determine u* and effective z_e using eight
underrelaxed iterations. z_e = z0/sqrt(1 - tau_w/tau), z0 = .01 u*^2/g plus
the viscous smooth-surface contribution .11 nu/u*. Wave stress is the magnitude
of rho_w g integral (k/sigma) B E (cos theta,sin theta) df dtheta.
The stress ratio is bounded at .95 to keep the surface-layer fixed point finite.
A Cavaleri-Malanotte-Rizzoli linear source seeds calm water, with the
Pierson-Moskowitz low-frequency filter. The per-radian-frequency source's 1/(2pi)
is removed when converting to per-Hz density. Constants: kappa=.41,
rho_a=1.225 kg/m^3, rho_w=1025 kg/m^3, nu=1.5e-5 m^2/s, g=9.80665 m/s^2.

Whitecapping is -Gamma sigma_tilde (k/k_tilde) E, Gamma = 4.10e-5
(.5 + .5 k/k_tilde) (k_tilde sqrt(m0)/sqrt(.00302))^4.
sigma_tilde is the inverse variance-weighted inverse sigma; k_tilde is the
inverse square of the weighted inverse square root of k. JONSWAP bottom
friction is -.038 (sigma/g/sinh(kh))^2 E. Depth-limited breaking uses
Hmax=.73h, (1-Qb)/log(Qb)=-8m0/Hmax^2 and a sink
-.25 Qb f_mean Hmax^2 E/m0. The probability root is solved in log space.
Wind growth is exponential and dissipation implicit. A source step uses B dt <=
.2 from the current diagnosed growth bound, in addition to the propagation CFL
and configured maximum step. The shared asynchronous ocean ABI instead performs
that growth-limited local subcycling inside each cell block on CUDA, with spatial
transport substeps chosen from a conservative group-speed envelope. It uses the
same source equations, and evaluates DIA at each transport substep. All source
terms are on by default.

DIA uses two mirrored resonant configurations, lambda=.25, angles
(+11.48,-33.56) degrees and their negatives, Cnl=3e7, and f^11/g^4 scaling.
With E in per-Hz units the cubic transfer is
Cnl f^11/g^4 [E0^2(Eplus/1.25^4 + Eminus/.75^4)
-2 E0 Eplus Eminus/(1-.25^2)^4]. A positive transfer removes two action units
from the central bin and adds one at each partner. Physical-frequency linear
interpolation makes the partner frequencies sum to 2 f0; deposition uses the
same weights and converts action back to variance at each recipient's actual
frequency. Thus integrated action and variance conserve to floating-point
summation error. Discrete angular interpolation does not exactly conserve vector
momentum. Finite-depth correction uses kd=max(.5,.75 k_tilde h) and
1+5.5/kd (1-5kd/6) exp(-1.25kd).

The finite frequency domain contains complete in-band quadruplets only; no
extrapolated high-frequency tail is imposed. Whole-quadruplet donor limits and a
cell-wide positive update factor retain both discrete invariants while preventing
negative bins. This limiter changes transfer magnitude if the step is too long;
receipts record its minimum. The default grid is 36 frequencies from .035 Hz
with ratio1.1 and 36 directions, covering approximately .035 to .984 Hz.
The statistical partition is an age criterion, not watershed partitions of each
spectral peak. Hs=4 sqrt(m0); Tp is the inverse of the peak of direction-integrated
spectral density, and direction is the integrated meteorological FROM bearing.

Primary technical references consulted (equations, not implementation code):

* SWAN technical documentation, wind equations 2.33-2.43:
  https://swanmodel.sourceforge.io/online_doc/swantech/node15.html
* Whitecapping, bottom friction and depth breaking:
  https://swanmodel.sourceforge.io/online_doc/swantech/node16.html
* DIA configurations and depth scaling, equations 2.76-2.80:
  https://swanmodel.sourceforge.io/online_doc/swantech/node18.html
* NOAA WAVEWATCH III technical manual, spectral balance and DIA discussion:
  https://polar.ncep.noaa.gov/waves/wavewatch/manual.v4.18.pdf

Numerical conservation and positivity tests establish implementation properties.
They do not establish forecast skill. A storm run and public-buoy comparison must
accompany qualification, with calm spin-up, resolution, boundary effects, the
ERA5 wind representation and the observation definitions stated explicitly.

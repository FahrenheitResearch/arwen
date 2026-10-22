"""Pinned WRF Besselian windows and CUDA eclipse obscuration.

Geometry follows WRF v4.6.1 phys/module_ra_eclipse.F, solar_eclipse and
compute_besselian_t, including WRF's linear penumbra convention.
Undimmed shortwave is retained for recovery from totality without division.

WRF notice: WRF was developed at the National Center for Atmospheric Research
(NCAR) which is operated by the University Corporation for Atmospheric Research
(UCAR). NCAR and UCAR make no proprietary claims, either statutory or otherwise,
to this version and release of WRF and consider WRF to be in the public domain
for use by any person or entity for any purpose without any fee or charge.
UCAR requests that any WRF user include this notice on any partial or full
copies of WRF. WRF is provided on an "AS IS" basis and any warranties, either
express or implied, including but not limited to implied warranties of
non-infringement, originality, merchantability and fitness for a particular
purpose, are disclaimed. In no event shall UCAR be liable for any damages,
whatsoever, whether direct, indirect, consequential or special, that arise out
of or in connection with the access, use or performance of WRF, including
infringement actions. WRF is a registered trademark of UCAR.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json

from gpuwm.data_assets import data_path

TABLE_PATH = data_path("wrf_radiation/eclipse_besselian_elements.dat")
MANIFEST_PATH = TABLE_PATH.with_name("eclipse_manifest.json")


@dataclass(frozen=True)
class EclipseWindow:
    """One table row's validity window, in UTC."""

    start: datetime
    end: datetime

    def label(self) -> str:
        return (f"{self.start:%Y-%m-%d %H:%M}-{self.end:%H:%M} UTC"
                if self.start.date() == self.end.date() or
                self.end == datetime.combine(self.start.date(),
                                             datetime.min.time())
                + timedelta(days=1)
                else f"{self.start:%Y-%m-%d %H:%M} to "
                     f"{self.end:%Y-%m-%d %H:%M} UTC")


@lru_cache(maxsize=1)
def eclipse_windows() -> tuple[EclipseWindow, ...]:
    """Every row of the pinned table as a UTC window, in table order.

    Refuses a table whose bytes do not match the manifest: a changed table
    moves or invents eclipses, and an import decision read from it would be
    about some other sky.
    """
    pin = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = TABLE_PATH.read_bytes()
    if (hashlib.sha256(data).hexdigest() != pin["sha256"]
            or len(data) != int(pin["bytes"])):
        raise ValueError(
            f"{TABLE_PATH} does not match {MANIFEST_PATH.name}: a changed "
            "Besselian table moves or invents eclipses")
    windows = []
    for number, line in enumerate(data.decode("ascii").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 32:
            raise ValueError(f"{TABLE_PATH}:{number} has {len(fields)} "
                             "columns; WRF's rows have 32")
        year, julday = int(fields[0]), int(fields[1])
        tmin, tmax = float(fields[3]), float(fields[4])
        day = datetime(year, 1, 1) + timedelta(days=julday - 1)
        windows.append(EclipseWindow(start=day + timedelta(hours=tmin),
                                     end=day + timedelta(hours=tmax)))
    return tuple(windows)


def eclipses_during(start: datetime, end: datetime
                    ) -> tuple[EclipseWindow, ...]:
    """The table windows that overlap ``[start, end]`` (inclusive)."""
    return tuple(window for window in eclipse_windows()
                 if window.start <= end and window.end >= start)


@lru_cache(maxsize=1)
def besselian_rows():
    """Verified metadata in the original table's first-match order."""
    eclipse_windows()
    return tuple(tuple(float(v) for v in line.split())
                 for line in TABLE_PATH.read_text(encoding="ascii").splitlines()
                 if line.split())


def besselian_row_at(valid_time):
    """Select by UTC date and hour, including both table endpoints."""
    from datetime import timezone
    if valid_time.tzinfo is not None:
        valid_time = valid_time.astimezone(timezone.utc)
    doy = valid_time.timetuple().tm_yday
    hour = (valid_time.hour + valid_time.minute / 60
            + (valid_time.second + valid_time.microsecond / 1e6) / 3600)
    for row in besselian_rows():
        if (int(row[0]) == valid_time.year and int(row[1]) == doy
                and row[3] <= hour <= row[4]):
            return row, hour
    return None, hour


_CUDA = r"""
extern "C" __global__ void eclipse_obscuration(
    const float* lat, const float* lon, const double* b,
    const double hour, const long long n, float* obscur, int* mask) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double rad = 0.017453292519943295769;
    const double eps = 0.0818192, E = sqrt(1.0 - eps*eps);
    const double t = hour + b[31]/3600.0 - b[2];
    double v[6] = {0,0,0,0,0,0};
    for (int k=0; k<4; ++k) {
        double tk = k==0 ? 1.0 : (k==1 ? t : (k==2 ? t*t : t*t*t));
        for (int j=0; j<6; ++j) v[j] += b[5+4*j+k]*tk;
    }
    const double rd=v[2]*rad, rd1=atan(tan(rd)/E), rd2=atan(E*tan(rd));
    // Stable sin(rd)/sin(rd1), including zero declination.
    const double rho1=sqrt(E*E*cos(rd)*cos(rd)+sin(rd)*sin(rd));
    const double rlat=lat[i]*rad;
    const double longitude=-lon[i]-1.002738*15.0*b[31]/3600.0;
    const double H=v[5]*rad-longitude*rad;
    const double A=sqrt(1.0-eps*eps*sin(rlat)*sin(rlat));
    const double clat=cos(rlat)/A, slat=sin(rlat)*E/A;
    const double xi=clat*sin(H);
    const double eta=slat*cos(rd1)*E-clat*sin(rd1)*cos(H);
    const double zeta=slat*sin(rd2)*E+clat*cos(rd2)*cos(H);
    const double dx=xi-v[0], dy=eta-v[1]/rho1;
    const double delta=sqrt(dx*dx+dy*dy);
    const double l1=v[3]-zeta*b[29], l2=v[4]-zeta*b[30];
    double obsc=0.0; int kind=0;
    // Do not project the visible-sun shadow onto the night side.
    if (zeta>0.0 && delta<=l1) {
        if (delta<=fabs(l2) && l2<0.0) { obsc=1.0; kind=2; }
        else { obsc=(l1-delta)/(l1+l2); kind=delta<=fabs(l2)?3:1; }
    }
    obscur[i]=(float)fmin(1.0,fmax(0.0,obsc)); mask[i]=kind;
}
"""


class EclipseGeometry:
    """CUDA field calculation; the host selects immutable table metadata."""
    def __init__(self, latitude_deg, longitude_deg):
        import cupy as cp
        self.latitude = cp.ascontiguousarray(cp.asarray(latitude_deg, dtype=cp.float32))
        self.longitude = cp.ascontiguousarray(cp.asarray(longitude_deg, dtype=cp.float32))
        if self.latitude.shape != self.longitude.shape:
            raise ValueError("eclipse latitude and longitude shapes must match")
        self.obscuration = cp.zeros_like(self.latitude)
        self.mask = cp.zeros(self.latitude.shape, dtype=cp.int32)
        self._row = None
        self._device_row = None
        from gpuwm.certify.kernel_manifest import record_module
        module = cp.RawModule(code=_CUDA, options=())
        record_module("gpuwm.eclipse_windows:obscuration", source=_CUDA,
                      options=(), module=module)
        self._kernel = module.get_function("eclipse_obscuration")

    def at(self, valid_time):
        import cupy as cp
        import numpy as np
        row, hour = besselian_row_at(valid_time)
        if row is None:
            self.obscuration.fill(0)
            self.mask.fill(0)
        else:
            if row != self._row:
                self._device_row = cp.asarray(row, dtype=cp.float64)
                self._row = row
            n = self.latitude.size
            if n:
                self._kernel(((n+255)//256,), (256,),
                             (self.latitude, self.longitude, self._device_row,
                              np.float64(hour), np.int64(n), self.obscuration, self.mask))
        return self.obscuration


class EclipseRadiation:
    """Per-step shadow on shortwave heating, surface and boundary fluxes.

    SWINT fits undimmed sunlight; obscuration applies after interpolation.
    No division by transmission occurs. Undimmed fields live in the driver
    field inventory, so restart retains the exact cache and radiation cadence.
    """
    @classmethod
    def for_run(cls, cfg, radiation):
        from gpuwm.config import radiation_scheme_ids
        if not int(getattr(cfg, "ra_sw_eclipse", 1)) or not radiation_scheme_ids(cfg)[1]:
            return None
        if radiation is None:
            return None
        # Analytic attachment callbacks may have no solar geometry or clock.
        # Production radiation adapters publish all three metadata attributes.
        start = getattr(radiation, "start_time", None)
        if start is None:
            if any(hasattr(radiation, name) for name in
                   ("latitude_deg", "longitude_deg")):
                raise ValueError("eclipse radiation geometry requires start_time")
            return None
        if start.tzinfo is not None:
            from datetime import timezone
            start = start.astimezone(timezone.utc).replace(tzinfo=None)
        if not eclipses_during(start, start+timedelta(seconds=cfg.run_seconds)):
            return None
        return cls(radiation)

    def __init__(self, radiation):
        self.start_time = radiation.start_time
        self.geometry = EclipseGeometry(radiation.latitude_deg, radiation.longitude_deg)
        self.base_fields = None
        self.base_heating = None
        self.base_diagnostics = None
        self.steps = 0

    def bind(self, driver):
        """Allocate stable checkpoint-owned storage before the first step."""
        import cupy as cp
        names = {"swdown", "gsw", "swddir", "swddif", "swddni",
                 "swupt", "swdnt", "swupb", "swdnb",
                 "swuptc", "swdntc", "swupbc", "swdnbc"}
        self.base_fields = {name: driver.fields.setdefault(
            "eclipse_clear_"+name, cp.zeros_like(driver.fields[name]))
            for name in names if name in driver.fields}
        self.base_heating = driver.fields.setdefault(
            "eclipse_clear_heating", cp.zeros_like(driver.rthratensw))
        diagnostics = getattr(driver, "radiation_flux_diagnostics", None)
        if diagnostics is not None:
            self.base_diagnostics = driver.fields.setdefault(
                "eclipse_clear_diagnostics", cp.zeros_like(diagnostics))
        self.initialized = driver.fields.setdefault(
            "eclipse_initialized", cp.zeros_like(self.geometry.obscuration))
        # Geometry output aliases also stay stable across checkpoint restore.
        driver.fields["ecobsc"] = self.geometry.obscuration
        driver.fields["ecmask"] = self.geometry.mask
        self._ready = None

    def ready(self):
        if self._ready is None:
            self._ready = bool(self.initialized.ravel()[0])
        return self._ready

    def capture(self, driver):
        if self.base_fields is None:
            self.bind(driver)
        for name, base in self.base_fields.items():
            base[...] = driver.fields[name]
        self.base_heating[...] = driver.rthratensw
        if self.base_diagnostics is not None:
            self.base_diagnostics[...] = driver.radiation_flux_diagnostics
        self.initialized.fill(1)
        self._ready = True

    def apply(self, driver, state, cfg):
        import cupy as cp
        from gpuwm.core.physics import couple_column_tendencies
        DTYPE = cp.float32
        obscur = self.geometry.at(self.start_time+timedelta(seconds=float(state.elapsed_seconds)))
        transmission = DTYPE(1.0)-obscur
        driver.fields["ecobsc"] = self.geometry.obscuration
        driver.fields["ecmask"] = self.geometry.mask
        interpolated = {"swdown", "gsw", "swddir", "swddif", "swddni"}
        for name, base in self.base_fields.items():
            source = driver.fields[name] if driver.swint is not None and name in interpolated else base
            cp.multiply(source, transmission, out=driver.fields[name])
        previous_heating = driver.rthratensw.copy()
        cp.multiply(self.base_heating, transmission[None], out=driver.rthratensw)
        change = driver.rthratensw - previous_heating
        changed = change != DTYPE(0)

        if self.base_diagnostics is not None:
            for k, name in enumerate(driver.radiation_callable.diagnostics):
                if "SW" in name:
                    cp.multiply(self.base_diagnostics[k], transmission,
                                out=driver.radiation_flux_diagnostics[k])
        # Preserve the held carrier, including its normal dry-mass round trip.
        # Reconstructing LW+SW here resets that history on every clear-sun
        # step and perturbs McICA/cloud evolution before the shadow arrives.
        # Apply only the change in shortwave; exact zero leaves every word.
        driver.rthraten[...] = cp.where(changed, driver.rthraten + change,
                                       driver.rthraten)
        carrier = getattr(driver, "_wrf_rthraten", None)
        if carrier is not None:
            # compute() already uncoupled this carrier; its usual end-of-step
            # coupling remains the single authority.
            carrier[...] = cp.where(changed, carrier + change, carrier)
        else:
            coupled = couple_column_tendencies(
                state, cfg, rtheta=driver.rthraten,
                shared_zero_components=("ru", "rv", "rqv", "rqc"))
            driver.radiation_tendencies.rtheta[...] = cp.where(
                changed, coupled.rtheta, driver.radiation_tendencies.rtheta)
        self.steps += 1

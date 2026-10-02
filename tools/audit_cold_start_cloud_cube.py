"""Test explicit cube against C-library pow on seeded cloud entry lambdas."""
import numpy as np

from gpuwm.core import thompson_entry as te
from gpuwm.core.host_libm import power
from gpuwm.core.noahmp_libm import powf_array

rng = np.random.default_rng(318)
n = 200000
f = np.float32
mass = (10.0 ** rng.uniform(-12, -2, n)).astype(f)
alt = (10.0 ** rng.uniform(-1, 1, n)).astype(f)
aerosol = (10.0 ** rng.uniform(6, 11, n)).astype(f)
aerosol[::3] = 0
xland = rng.choice(np.array([1, 2], dtype=f), n)
rho = (f(1) / alt).astype(f)
seed = (te.make_droplet_number(mass * rho, aerosol * rho, xland) / rho).astype(f)
density = 1.0 / alt.astype(np.float64)
rc = (mass * density).astype(f)
nc = np.maximum(f(2), np.minimum((seed * density).astype(f), f(te.NT_C_MAX)))
nu = te._nu_c(nc)
lam = powf_array((nc * f(te.AM_R) * te.CCG2[nu].astype(f)
                  * te.OCG1[nu].astype(f) / rc).astype(f), f(f(1) / f(3))).astype(np.float64)
dc = ((f(te.BM_R) + nu.astype(f) + f(1)).astype(np.float64) / lam).astype(f)
small, large = dc < f(te.D0C), dc > f(te.D0R) * f(2)
lam = np.where(small, (te.CCE2[nu].astype(f) / f(te.D0C)).astype(np.float64), lam)
lam = np.where(large & ~small,
               (te.CCE2[nu].astype(f) / (f(te.D0R) * f(2))).astype(np.float64), lam)
expected = power(lam, np.float64(3))
multiplied = (lam * lam) * lam
different = expected.view(np.uint64) != multiplied.view(np.uint64)
print(f"CLOUD_CUBE inputs={n} differing_pow_bits={np.count_nonzero(different)}")
for i in np.flatnonzero(different)[:5]:
    print(f"lambda={lam[i].hex()} pow={expected[i].hex()} multiply={multiplied[i].hex()}")

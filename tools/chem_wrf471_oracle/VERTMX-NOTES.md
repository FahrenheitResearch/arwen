# WRF bounds, gates and integration contract

Pinned source: WRF f52c197ed39d12e087d02c50f412d90d418f6186.

chem_driver.F:449-450 gets k_start/k_end from kps/kpe. At :777-778 it sets
kts=k_start and kte=min(k_end,kde-1) for chemistry. The chem_prep call at
:784-794 instead passes k_start,k_end. For a complete EM column, k_start=1,
k_end=kde and the chemistry kte=kde-1. Here nz=kde-1 is the number of real
mass levels, not the number of levels written by this WRF mixing block.

module_chem_utilities.F:71-72 computes k_end=min(kte,kde-1). At :79-83 it
forms full pressure, total-theta temperature, moist density and destaggered
winds. At :93-97 it copies those five mass outputs to kde. At :103-106 it
writes z_at_w through kde; at :111-121 it fills dz8w through kde-1 and
sets dz8w(kde)=0. At :126-132 it writes z/rh only through kde-1. The arrays
z/rh(kde) are not initialized here. They are not exposed by this port.
p8w/t8w are interpolated at :144-145 and extrapolated at :176-177.

The port's 10 mass fields have nz cells. The three interface fields,
z_at_w/p8w/t8w, have nz+1. WRF's extra mass-output copies and dz8w padding
zero are omitted. This avoids treating uninitialized z/rh padding as a
physical cell. The physical mass level at chemistry kte remains present.

In dry_dep_driver.F:681-688, zzfull reads z_at_w through kte+1=kde;
ekmfull reads exch_h through kte=nz and writes zero at kts and kte+1.
At :730-744, zz reads z and dryrho_1d reads alt through kte, both valid
mass inputs. module_vertmx_wrf.F:190-195 reads zsigma through ktem1+1
and zsigma_half/dryrho through ktem1. Here ktem1 is chemistry kte.
Bookkeeping at dry_dep_driver.F:779-803 reads dz8w and writes chem only
through kte-1, omitting the final real mass cell. No uninitialized padding
is needed for this complete-column path. Partial vertical tile semantics
are not implemented; each array passed here is a complete column.

exch_h is (nz,ny,nx), allocated at gpuwm/core/physics.py:5357. Its indices
are WRF lower w interfaces, not layer means. MYNN supplies k_h at
mynn_pbl.py:3972; :1918-1922 sets its first interface zero and computes
subsequent values using dfh times the neighboring average dz. The mixing
kernel supplies the missing physical top interface zero explicitly.

Gates:
- chem_driver.F:1046-1047: vertmix_onoff>0 and ktau>2.
- dry_dep_driver.F:672: this path requires num_vert_mix==0.
- :741: CAM microphysics-owned constituents are skipped when
  is_CAMMGMP_used and not vertMixAero(nv).
- :747-769: this port is CASE DEFAULT. The listed aqueous families have
  another gas/aerosol selection and are outside this module's contract.
- :765: mynn_chem_vertmx suppresses the default solver call.
The process honors the first, second and MYNN gates. Integration must keep
CAM-owned rows out of table.rows_for('mixing.vertmx'), matching the ownership
gate without species-name/index tests. Registry defaults for missing cfg
fields are vertmix_onoff=1 and mynn_chem_vertmx=False
(Registry/registry.chem:3832,3793).

Mixing floors follow dry_dep_driver.F:701-727 exactly. Optional per-column
planes anth_co_kts and fire_co_k1 encode Registry presence. The two PM
channels, anth_pm25_pair=(i,j) and anth_pm25 (single), preserve :709-717's
separate >param_first_scalar gates upstream. Both PM branches are nested
inside anthropogenic CO presence and sf_urban_physics==0, even with CO=0.
CO>0 raises Fortran indices kts+1:kts+10 to 1; CO>200, either PM threshold
>8.19e-4*200, or fire CO>0 raises kts+1:kte/2 to 2. Strict inequalities
are retained. No species name or slot determines a port branch. Integration
selects source rows carrying raises_mixing_floor and supplies these planes
through ctx.physics_fields. Schema/source selection is not changed here.

pblst=max(row.floor,chem) follows :743. The fixtures use the WRF epsilc
1e-16 (chem/module_data_radm2.F:5). dryrho=1/alt follows :744. Gas/aerosol
bookkeeping is selected by row.phase, not numgas or a species slot. mwdry
is 28.966 g mol-1 (share/module_model_constants.F:34). The gas expression
at :785 and aerosol expression at :790 keep their exact operation order.
At :799 ddmassn is per-step max(0,old-new), including positive rounding
loss. The port adds it to a serialized rows_2d accumulation, like WRF's
conditional dvel additions beginning at :808. Gas rows store mol m-2,
aerosol rows store ug m-2, so there is no single common output unit.
The mass ledger remains driver's responsibility, preventing double booking.

The device solver uses one thread per column, one launch per active row.
There is no species-index branch or dependence on arena contiguity. Local
arrays are specialized to nz using the existing integer-define loader.
ChemPrep device buffers are reused across refresh calls. Their inventory
is 10*nz + 3*(nz+1) floats per column, rebuilt from state, not serialized.
The lead must add allocation, digest, frame and dependency inventory entries
at integration. No frozen physics kernel or config/restart/dycore is edited.

Conservation conflict:
module_vertmx_wrf.F's complete solver conserves dry column mass to float32
rounding when vd=0. The default driver's truncated writeback does not when
exch_h(kte)>0. The port retains that WRF behavior for bitwise parity.
With the fixture-derived 49-level column, dt=18 s, exch_h=100 and vd=0,
the solver's relative mass error is 1.554398759103571e-8, but the driver's
writeback error is 0.0022939130984717293. With the internal interface into
the frozen top cell closed, writeback error is 1.1164620563612016e-8.
tests/test_chem_vertmx_wrf471_parity.py::test_cpu_conservation_and_wrf_top_writeback
holds these numbers. Therefore an
unconditional conservation claim for the requested WRF writeback cannot
be made. Changing this would change WRF words and requires an integration
decision. No conservation correction is silently applied here.

Commands, run from the repository root unless shown:
- GPUWM_NO_LOCAL_GPU=1 python -B -m pytest -q tests/test_chem_prep_wrf471_parity.py tests/test_chem_vertmx_wrf471_parity.py -m 'not gpu'
- On a Linux host with gfortran, in a scratch directory:
  bash tools/chem_wrf471_oracle/build.sh source build run_chem_prep run_vertmx
  python3 tools/chem_wrf471_oracle/check_cuda_host_vertmx.py . host-check
The C++ check compiles the
unchanged CUDA bodies as host functions at -O0 -ffp-contract=off. It checks
six columns per fixture with explicit host RN operation shims. It never
uses CUDA, and is not a substitute for NVRTC or device parity.

To reproduce the oracle, create an owned source overlay holding byte copies
of the pinned chem/module_chem_utilities.F and chem/module_vertmx_wrf.F.
Download share/module_model_constants.F from the same pinned WRF commit:
https://raw.githubusercontent.com/wrf-model/WRF/f52c197ed39d12e087d02c50f412d90d418f6186/share/module_model_constants.F
Its hash is in SOURCES-vertmx.sha256. eta-vertmx.inc is derived from the
pinned dyn_em/module_initialize_real.F:7654-7663, also pinned there. The
original public source comments in the verbatim block are retained as part
of its required byte identity; the wrapper adds no attribution names.

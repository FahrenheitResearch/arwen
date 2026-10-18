# WRF v4.6.1 slope_rad / topo_shading column oracle

WOOF's `gpuwm/core/kernels/topo_radiation.cu` against WRF v4.6.1's own
Fortran, bit for bit, over the whole chain: slope and slope azimuth
(start_em.F), the terrain shadow (toposhad_init + toposhad), the diffuse
fraction (radiation driver, with the Ruiz-Arias split), the surface
adjustment (TOPO_RAD_ADJ_DRVR + TOPO_RAD_ADJ) and the restore after the land
surface.

| file | what it does |
|---|---|
| `extract.py` | Cuts the WRF statements verbatim out of a pristine v4.6.1 checkout. Every file and every block is pinned by sha256; the blocks carry the same digests as v4.7.1's (`../wrf_topo_radiation_v471_oracle/extract.py`). |
| `build.sh` | Compiles them with WRF's own GNU flags (no FMA, checked on the objects; any SIMD math call listed) into `oracle` (the box's libm) and `oracle_crlibm` (correctly rounded diagnostic). |
| `column_oracle.py` | The case set (20 patches, 32,100 columns, labelled by regime), the WRF chain through the oracle, WOOF's chain through its public entry points, and the comparison. |
| `libm_digest.c` / `libm_digest.py` | Whole-domain check of each libm word the unit uses: every float32 input of the word's domain, digested per binade on the host (under each libm) and on the device (WOOF's own unit, as the loader composes it). |

## Rerun (box W1 recipe)

```
T=/work/pverify/<lane>/src; source /work/pverify/base/env.sh
O=$T/tools/wrf_topo_radiation_v461_oracle
bash $O/build.sh /work/pverify/wrf-build/src/WRF-4.6.1 build
# a second libm: Ubuntu's libc6 2.43-2ubuntu2.4 package, unpacked
curl -sfO http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.43-2ubuntu2.4_amd64.deb
dpkg-deb -x libc6_2.43-2ubuntu2.4_amd64.deb glibc243
CUDA_VISIBLE_DEVICES= python $O/column_oracle.py oracle build refs.npz \
    --stock glibc239 --libm glibc243=$PWD/glibc243
# GPU, through the mutex: strict and default arithmetic
GPUWM_WRF_EXACT=1 python $O/column_oracle.py woof refs.npz woof_strict.npz
python $O/column_oracle.py woof refs.npz woof_default.npz
python $O/column_oracle.py compare refs.npz woof_strict.npz woof_default.npz --json compare.json
# libm words: host digests per libm, then the device check
gcc -O2 -fopenmp -fno-builtin -o build/libm_digest $O/libm_digest.c -lm
for w in sinf cosf sincosf tanf asinf acosf atanf atan2f_x1000 atan2f_1000_y expf \
         powf_m1.6364 powf_0.5 atan2f_resolve_x1000 atan2f_pairs atan2f_grad; do
  build/libm_digest $w >> glibc239.txt
  G=$PWD/glibc243/usr/lib/x86_64-linux-gnu
  $G/ld-linux-x86-64.so.2 --library-path $G:/usr/lib/x86_64-linux-gnu build/libm_digest $w >> glibc243.txt
done
GPUWM_WRF_EXACT=1 python $O/libm_digest.py --exe build/libm_digest \
    --libm glibc239 glibc239.txt --libm glibc243 glibc243.txt $PWD/glibc243
# the test fixture
python $O/column_oracle.py fixture refs.npz \
    $T/tests/fixtures/wrf_topo_radiation_v461/topo_radiation_chain_v461.npz
```

`tests/test_topo_radiation.py` holds the port to the fixture's
`wrf_glibc243` side on a card.

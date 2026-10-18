WRF 4.6.1 OML column oracle and production CUDA comparison.

Build the byte-pinned public module with:
    bash tools/oml_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR

The committed word receipt records 112 columns, including 96 water columns and 16 land controls, three initialization branches and 16 steps. It is a conformance fixture, not a forecast or observation qualification. The source SHA256 is checked before compilation. The oracle uses gfortran -O0 -fno-tree-vectorize -ffp-contract=off.

On a GPU box through its card queue:
    GPUWM_WRF_EXACT=1 python tools/oml_wrf461_oracle/validate.py tools/oml_wrf461_oracle/receipts --out strict.json
    env -u GPUWM_WRF_EXACT python tools/oml_wrf461_oracle/validate.py tools/oml_wrf461_oracle/receipts --out default.json
    GPUWM_WRF_EXACT=1 python tools/oml_wrf461_oracle/validate.py tools/oml_wrf461_oracle/receipts --mutation --out mutation.json

Both modes require zero ULP, with per-field differences and full output hashes retained. The mutation changes the friction-velocity multiplier from float32 0x3f000000 to 0x3f000001 in the actual compiled kernel. A nonzero difference must be detected.

tests/test_oml_gpu.py runs the actual driver cadence and full engine checkpoint writer and restore path. It compares all numeric serialized arrays after 16 uninterrupted steps against 8 plus restart plus 8, and preserves per-array SHA256 receipts.

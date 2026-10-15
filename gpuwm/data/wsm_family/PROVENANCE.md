# WSM-family scheme table and WRF words

`schemes.json` admits each WRF single-moment WSM-family scheme the engine
runs through `gpuwm/core/wsm_family.py`.  `wsm7-wrf461-words.json` holds the
float32 words WRF's `wsm7init` computes and the `radar_init` words
`refl10cm_wsm7` reads, dumped bit for bit by the column oracle and never
recomputed on the host.

- WRF release: v4.6.1
- upstream repository: https://github.com/wrf-model/WRF
- source commit: `d66e442fccc04111067e29274c9f9eaccc3cef28`
- `phys/module_mp_wsm7.F` SHA-256:
  `d3ae349fc5e122660a3b1bfafef6ae82dae17c1bd79581688190d41daadac807`
- `phys/module_mp_radar.F` SHA-256:
  `aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688`
- dumped by `tools/wsm7_wrf461_column_oracle/dump_wsm7_init.F90` (gfortran
  13.3.0 -O0, glibc 2.39) and frozen by
  `tools/wsm7_wrf461_column_oracle/freeze_words.py`.

The CUDA transcription is `gpuwm/core/kernels/wsm7.cu`.  `LICENSE.txt` is
copied from the WRF v4.6.1 source root; WRF is public domain.

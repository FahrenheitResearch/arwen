# Fast SBM lookup table provenance

`SBM_input_33/` holds the 18 table files that WRF's fast spectral bin
microphysics (FSBM-2, `mp_physics = 30`) opens, byte for byte as NCAR
distributes them. They are not in the WRF repository; NCAR serves them in a
separate archive.

WRF is NCAR/UCAR public domain and asks that its notice travel with partial
copies; the notice is `LICENSE-WRF.txt` beside this file (a byte copy of
`gpuwm/data/p3/LICENSE-WRF.txt`) and the repository `NOTICE`.

## Source

| | |
| --- | --- |
| URL | `https://www2.mmm.ucar.edu/wrf/src/wrf_files/SBM_input_33.tgz` |
| Archive size | 1,339,105 bytes |
| Archive SHA-256 | `25ed63ab3550aed110ae24df229faece20eafedfdf21cde01d8973826a7294bf` |
| Downloaded and verified | 2026-09-22 |
| Members | 54: the directory `SBM_input_33/` and 53 regular files |
| Reader | WRF v4.8.0 `phys/module_mp_fast_sbm.F`, LF SHA-256 `fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e` (identical in v4.6.1 and v4.7.1) |

The 18 files were extracted with Python's `tarfile` straight from the verified
archive into this directory, never through an editor or a Windows copy tool
(the P3 table in `gpuwm/data/p3/` picked up CR bytes that way). Every member
is LF text with no CR byte. `SBM_input_33/MANIFEST.sha256` lists the digests
in `sha256sum -c` form; the authority the loader checks is `TABLE_PINS` in
`tools/fast_sbm_host/tables.py`, and `tests/test_fast_sbm_tables.py` holds
the two equal.

## Packaged files

READ is the line in `module_mp_fast_sbm.F` that consumes the file.

| File | Bytes | SHA-256 | READ | What it fills |
| --- | ---: | --- | ---: | --- |
| `BLKD_SDC.dat` | 2280 | `6992497ee5b95f0fd215b762ad4af8611079d1e13dbe3db78687fe04a128abf2` | 5445 | `bin_mass`, `tab_colum`, `tab_dendr`, `tab_snow` (REAL*8), list-directed, first 33 of 43 data records |
| `capacity33.asc` | 3042 | `33759dbb0e238b37403ce2ad62759cca57c03538d72d34c2cc04bb7b2547c951` | 5500 | `RLEC(33)`, `RIEC(33,3)`, `RSEC`, `RGEC`, `RHEC`, FORMAT(6E13.5) |
| `masses33.asc` | 3042 | `f53d3acdaa0a3e2a265272fa22a2a0bd54292f51815149370bbf315e3fa1bee4` | 5548 | `XL(33)`, `XI(33,3)`, `XS`, `XG`, `XH` |
| `termvels33_corrected.asc` | 3042 | `bf4209175492a3019563671676787afc3befd948fddc6435db58f7a8d7f702e3` | 5598 | `VR1(33)`, `VR2(33,3)`, `VR3`, `VR4`, `VR5` |
| `constants33.asc` | 3911 | `d3d7cf345ebf850fa12376a7f173ea12b169f6413910939682494bcc5874a4fa` | 5648 | `SLIC(33,6)`, `TLIC(33,2)`, `COEFIN(33)`; read, never used |
| `kernLL_z33.asc` | 43016 | `ce4e140595ff6db2434ecd5376a16c50ea410d03ddc1700bfa95a3490c15106f` | 5694 | `YWLL_1000MB`, `YWLL_750MB`, `YWLL_500MB`, each (33,33) |
| `ckli_33_300mb_500mb_750mb.asc` | 129047 | `59ce82054f29df7d9f982249b915201f3f0385ae559c5ef2b7cdb1907a46f81a` | 5799 | `YWLI_300MB`, `_500MB`, `_750MB`, each (33,33,3) |
| `cklg_33_300mb_500mb_750mb.asc` | 43016 | `f3e8149a7e1bac947706f945a1d2ed0fb81ef2737404a0ef8c8c0c682fb4290f` | 5815 | `YWLG_300MB`, `_500MB`, `_750MB`, each (33,33) |
| `cklh_33_300mb_500mb_750mb.asc` | 43016 | `a2ffe0b96e4acfcd06dc1a0b10cb28f60601ebe255f8e00a70ce1797705fb50e` | 5831 | `YWLH_300MB`, `_500MB`, `_750MB`, each (33,33) |
| `ckls_33_300mb_500mb_750mb.asc` | 43016 | `ae282329a306d384bd6de958f1e0bf745c496d286237e8d22c85926081e0456a` | 5847 | `YWLS_300MB`, `_500MB`, `_750MB`, each (33,33) |
| `ckii_33_300mb_500mb_750mb.asc` | 387140 | `546777c064e7687bbaaaf30ab417dfc146e2decaabc41b270439e33328a75c7f` | 5864 | `YWII_300MB`, `_500MB`, `_750MB`, each (33,33,3,3) |
| `ckis_33_300mb_500mb_750mb.asc` | 129047 | `657428fd7bbc0dcf5e7810bb0e25af42645f6122a4b24313404b805cfcdec44d` | 5880 | `YWIS_300MB`, `_500MB`, `_750MB`, each (33,33,3) |
| `cksg_33_300mb_500mb_750mb.asc` | 43016 | `85b5fd667242ad216dd1b06f0f1c5ddfe15d797ae6c14dcc45e1097743bb37b9` | 5896 | `YWSG_300MB`, `_500MB`, `_750MB`, each (33,33) |
| `ckss_33_300mb_500mb_750mb.asc` | 43016 | `1946554a0e1341cd12c9ae6be2cd2c786bfc3e4622b8fe87f2e85fcc64373847` | 5912 | `YWSS_300MB`, `_500MB`, `_750MB`, each (33,33) |
| `bulkdens33.asc` | 3042 | `a5607b3d0b9eb1fcdba30e50b714e348a81aabb64d963b0a00ff4f7b2a24caa6` | 5988 | `RO1BL(33)`, `RO2BL(33,3)`, `RO3BL`, `RO4BL`, `RO5BL` |
| `bulkradii33.asc` | 3512 | `d432d8eed8760f5000c6c86ac1b602687feb0783f558adbf10811fe7040a9de0` | 6030 | `RADXXO(33,7)`, list-directed; read, never used |
| `coeff_p_new_33.dat` | 107730 | `0fc78f82cfed4033eab9bcaf3f5d6bf07a4f1bbbbf3aff708ede6a0b2e921e34` | 8725 | `PKIJ(18,18,18)`, (3I6,1E16.8), 3078 records, j <= i |
| `coeff_q_new_33.dat` | 9396 | `cbda1e0fb2c604d6c8d61ad0abc22be7b1552be93bf5daed10b5186c4ba07d9e` | 8755 | `QKJ(18,18)`, (2I6,1E16.8), 324 records |

The OPEN statements are at :5443, :5497, :5546, :5596, :5646, :5693, :5798,
:5814, :5830, :5846, :5863, :5879, :5895, :5911, :5986, :6028, :8718 and
:8751, in the order of the table. `BREAKINIT_KS` names the two coefficient
files by the hard-coded path `SBM_input_33/` (:8696-8697). The FORMAT(6E13.5)
label is `900` at :5499.

## Members not packaged

Of the archive's 54 members, 36 are not packaged: the directory entry
`SBM_input_33/` itself and these 35 regular files, none of which the code
opens.

| Member | Bytes | SHA-256 | Why it is not packaged |
| --- | ---: | --- | --- |
| `ckii_300mb_As` | 219107 | `0ba0124f7a559b2afae24f864f1d9e9a82105749d918b62c171bfc09ef779fee` | 43-bin per-pressure kernel; its file name and READ are commented out (:5851) |
| `ckii_500mb_As` | 219107 | `d8494231edfc26827a47097442d6749b6068ba00624ee73b9bbc8024359b9530` | as above (:5854) |
| `ckii_750mb_As` | 219107 | `c6b9aeda0e2e4c32505d7e7ef0c75de7692cb3b60bb5208bd822dadf510ca8b3` | as above (:5857) |
| `ckis_300mb_As` | 73036 | `09466218b8945a6f01ad14f6254ea52f3ee3cf09429afd8cddca39e233133d7b` | as above (:5868) |
| `ckis_500mb_As` | 73036 | `46f9588c4583b3cb35e2d2b25f3d7cb8423b96503a6707a2b8fc29b961a84eab` | as above (:5871) |
| `ckis_750mb_As` | 73036 | `a4efffc0ee38601031ff6c5aaf802bda160a29b105558bfa4404856016c282c7` | as above (:5874) |
| `cklg_300mb_As` | 24346 | `59915828ca4dc16e1b758fd4ee59f7a370db289b529259273a9af26fca2b21c2` | as above (:5803) |
| `cklg_500mb_As` | 24346 | `acd652f8195fa98bff237be85f6a295b3cc70b008492b00124b46c4dd1fc13d5` | as above (:5806) |
| `cklg_750mb_As` | 24346 | `bf3553b2e8e8816ffd74a1677a233df89143d4b59db0999e432cc3925f48c4cf` | as above (:5809) |
| `cklh_300mb_As` | 24346 | `3f8b20ae0146ddf55acc7b44474b7ac34bca10832c975bef81733debae17a5f2` | as above (:5819) |
| `cklh_500mb_As` | 24346 | `1f6840042db5a20542b908c5dae4f2840fb5ed0e69dc0336ede58077bc58f9d3` | as above (:5822) |
| `cklh_750mb_As` | 24346 | `42b0ded453bbede5e017bae9e67a04bda5237789f26b1f157b605dcdc9b3d17a` | as above (:5825) |
| `ckli_300mb_As` | 73036 | `26463102ca93134ba209d6161a4943872a1abde0d9d3fa9dc9afbc80adcfe89c` | as above (:5787) |
| `ckli_500mb_As` | 73036 | `f34187a8f79b0db358d2d05822d3b78249a143284da132aab8300e27d7d5d958` | as above (:5790) |
| `ckli_750mb_As` | 73036 | `d853bfc30c2fccdae60872839c2b815a8f729dfdd470b306099208a650a71db8` | as above (:5793) |
| `ckls_300mb_As` | 24346 | `388523aa97ae7b9a33e146a784f3645dffe54a5f527ddd1be991c4a6bd3a0e5a` | as above (:5835) |
| `ckls_500mb_As` | 24346 | `2654b28a7a90d75de01af36d69112596e96530ce2442efaa5fb377008427413f` | as above (:5838) |
| `ckls_750mb_As` | 24346 | `1165c5fde2b7c84bfae5ff63c2cfbf32ce494c9802f5472e0a4a493a288ff319` | as above (:5841) |
| `cksg_300mb_As` | 24346 | `5dd654397168cc9eae8196837b11a279d7d7183e1ce1914ceb92b5780815d6ef` | as above (:5884) |
| `cksg_500mb_As` | 24346 | `de5ae8c4cb2599a2f06c15b7159b4d78407817bae279a92623b352785db367bb` | as above (:5887) |
| `cksg_750mb_As` | 24346 | `d4b577ee4bcfe5d5c396d0a5ae240d4c22bb6275f6af18bfc7c7a718d261659b` | as above (:5890) |
| `ckss_300mb_As` | 24346 | `6cd96d872a62ab70aabb5414eb93873076fe6c72013e57b625f862bc61192dee` | as above (:5900) |
| `ckss_500mb_As` | 24346 | `7c44985e0de89bc1192508513515b7c514734cb8acb61c40b2d9e75560fcca66` | as above (:5903) |
| `ckss_750mb_As` | 24346 | `f608a057f02fab7de45f714bc3acc69ad7fd02593c808a1c0ec4bab94e786a93` | as above (:5906) |
| `cklg_33_1000mb.asc` | 14339 | `a32cb39be1b85d6f3a3d55e7cb89bdfa6656d05c4b3c1a5cf8e0240b5941273d` | a 33-bin drop-graupel kernel at 1000 hPa that no statement names |
| `kernLL_z43.asc` | 73036 | `c1540bff710c61ddfa69a888d6f5df2f7deace53676c2a47b0821a07b94c180d` | 43-bin `YWLL`; its file name is commented out (:5692) |
| `kernels43.asc` | 1192914 | `fb626b27989c31457ace26aaa25ea86744ee1009dfa574f40c03331058809b26` | 49 kernel tables of 43 x 43; no statement names it |
| `coeff_p43.dat` | 761656 | `624a7f1dca364117a16382b105c88149ead3c8c68feaebafbfeaf61fba296800` | 43-bin breakup gain coefficients, opened only when `nkr == 43` and then from `SBM_input_43/` (:8694) |
| `coeff_q43.dat` | 22736 | `6fcb4ecd796e330affe16ebef93ff0919dfc157bc6bacc8c1f64509f17c47d4e` | as above (:8695) |
| `coeff_p.dat` | 206226 | `ab7d89a49361bff72877c47b55cac227f10dbc1f4e8a40b757d92753725a7fa0` | an older breakup gain table with three values per record; no statement names it |
| `koeff_p.dat` | 206226 | `ab7d89a49361bff72877c47b55cac227f10dbc1f4e8a40b757d92753725a7fa0` | byte-identical to `coeff_p.dat`; no statement names it |
| `coeff_p_Barry.asc` | 206228 | `22e2385ba836176919514fed221db94681fdd169182acec14af121426527b2ad` | same layout as `coeff_p.dat` with different values (7,300 of 18,468 tokens differ); no statement names it |
| `coeff_q.dat` | 9396 | `28b215ab5acb1adc5e68e4b65c7bd3b61c096b1d65781b2d021c48743b1bc3b3` | an older breakup loss table; no statement names it |
| `koeff_q.dat` | 9396 | `28b215ab5acb1adc5e68e4b65c7bd3b61c096b1d65781b2d021c48743b1bc3b3` | byte-identical to `coeff_q.dat`; no statement names it |
| `coeff_q_Barry.asc` | 9398 | `83ab3d88f266443b1bd3691771c9e99f907db1fbc9729b1c300edc85ad90ee3f` | same layout as `coeff_q.dat` with different values (269 of 972 tokens differ); no statement names it |

The 43-bin variant (the 43-bin line at :3901 is commented out) and the full
SBM (`mp_physics = 32`) are refused by name until ported; if either is ported,
its tables are added here with their own pins.

## Relation to WRF's own `run/` tables

The full SBM (`mp_physics = 32`) reads tables from WRF's `run/` directory.
Three of them are the same bytes as fast SBM files once the carriage returns
of the CRLF science-refs checkout are removed (measured 2026-09-22 against
the WRF v4.8.0 checkout, commit 06d4240):

| WRF `run/` file | Bytes as checked out (CR) | LF bytes | LF SHA-256 | Packaged file |
| --- | ---: | ---: | --- | --- |
| `masses.asc` | 3081 (39) | 3042 | `f53d3acdaa0a3e2a265272fa22a2a0bd54292f51815149370bbf315e3fa1bee4` | `masses33.asc`, identical |
| `capacity.asc` | 3081 (39) | 3042 | `33759dbb0e238b37403ce2ad62759cca57c03538d72d34c2cc04bb7b2547c951` | `capacity33.asc`, identical |
| `kernels_z.asc` | 43561 (545) | 43016 | `ce4e140595ff6db2434ecd5376a16c50ea410d03ddc1700bfa95a3490c15106f` | `kernLL_z33.asc`, identical |
| `termvels.asc` | 3083 (40) | 3043 | `2b22e4ccf63e21e7dd7b381f46ed0f406b7e313b1e5fb43902423708e25443f3` | `termvels33_corrected.asc`, differs |

`termvels33_corrected.asc` differs from `run/termvels.asc` in 13 values and in
one trailing blank line that `run/termvels.asc` carries. The drop fall speeds
`VR1` are identical. The differing values, as `run/termvels.asc` then the
packaged file (cm/s at 1000 hPa):

| Array (bin) | `run/termvels.asc` | `termvels33_corrected.asc` |
| --- | ---: | ---: |
| `VR2(7,1)`, columns | 0.60000E+00 | 0.17000E+00 |
| `VR2(8,1)` | 0.10000E+01 | 0.18000E+00 |
| `VR2(14,1)` | 0.15000E+01 | 0.15000E+02 |
| `VR2(12,3)`, dendrites | 0.38000E+00 | 0.38000E+01 |
| `VR2(16,3)` | 0.11000E+01 | 0.11000E+02 |
| `VR2(18,3)` | 0.17000E+00 | 0.17000E+02 |
| `VR3(29)`, snow | 0.11000E+03 | 0.10750E+03 |
| `VR3(30)` | 0.11000E+03 | 0.11500E+03 |
| `VR3(31)` | 0.12000E+03 | 0.12500E+03 |
| `VR3(32)` | 0.13000E+03 | 0.13500E+03 |
| `VR3(33)` | 0.14000E+03 | 0.14500E+03 |
| `VR4(32)`, graupel | 0.35000E+03 | 0.34828E+03 |
| `VR4(33)` | 0.40000E+03 | 0.40007E+03 |

So the corrections are in the crystal columns and dendrites of `VR2` (several
of them look like decimal-point slips that the corrected file fixes), in the
five largest snow bins and in the two largest graupel bins. The warm path of
milestone 1 reads only `VR1`.

## Decimal to binary

Every value token is a decimal with an explicit decimal point. The float64
loader view rounds each one correctly to double (what WRF compiled with
`-freal-4-real-8` reads); the float32 view rounds each one correctly to
float32 in one step (what WRF's native REAL*4 READ gives), except the four
`BLKD_SDC.dat` arrays, which WRF declares REAL*8. Across the 73,461 REAL*4
tokens of the other 17 files, measured 2026-09-22, no token rounds
differently by way of double than directly to float32; the count is held by
`tests/test_fast_sbm_tables.py`.

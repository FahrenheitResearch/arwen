Each row covers every explicit FMA on that header line. Addresses are offsets in the installed libm ELF. Multiple calls on one line appear as multiple addresses.

| Function | Header line | Calculation | Installed instructions | Evidence |
|---|---:|---|---|---|
| exp | glibc_flt64.cuh:254 | overflow scaled result | 0xa0d0a | exp.objdump.txt:115 |
| exp | glibc_flt64.cuh:304 | multiply and Shift | 0xa0b81 | exp.objdump.txt:19 |
| exp | glibc_flt64.cuh:307 | two reduction steps | 0xa0baa, 0xa0bb9 | exp.objdump.txt:28, exp.objdump.txt:32 |
| exp | glibc_flt64.cuh:315 | polynomial and tail | 0xa0bc9, 0xa0bea, 0xa0bf3, 0xa0bfc | exp.objdump.txt:36, exp.objdump.txt:44, exp.objdump.txt:46, exp.objdump.txt:48 |
| exp | glibc_flt64.cuh:319 | normal result | 0xa0c0a | exp.objdump.txt:52 |
| log | glibc_flt64.cuh:545 | near-one polynomial | 0xa12eb, 0xa12f4, 0xa1309, 0xa1312, 0xa131b, 0xa1328, 0xa1331, 0xa133a, 0xa1347 | log.objdump.txt:78, log.objdump.txt:80, log.objdump.txt:85, log.objdump.txt:87, log.objdump.txt:89, log.objdump.txt:92, log.objdump.txt:94, log.objdump.txt:96, log.objdump.txt:99 |
| log | glibc_flt64.cuh:547 | near-one split | 0xa1350, 0xa1355 | log.objdump.txt:101, log.objdump.txt:102 |
| log | glibc_flt64.cuh:550 | near-one high result | 0xa136e | log.objdump.txt:108 |
| log | glibc_flt64.cuh:551 | near-one low result | 0xa137b | log.objdump.txt:111 |
| log | glibc_flt64.cuh:552 | near-one low correction | 0xa1384 | log.objdump.txt:113 |
| log | glibc_flt64.cuh:553 | near-one final correction | 0xa1389 | log.objdump.txt:114 |
| log | glibc_flt64.cuh:575 | table reduction | 0xa126f | log.objdump.txt:49 |
| log | glibc_flt64.cuh:577 | high logarithm scale | 0xa1264 | log.objdump.txt:47 |
| log | glibc_flt64.cuh:579 | low logarithm scale | 0xa128e | log.objdump.txt:56 |
| log | glibc_flt64.cuh:581 | general polynomial | 0xa1275, 0xa129b, 0xa12a4, 0xa12ad, 0xa12b2 | log.objdump.txt:50, log.objdump.txt:59, log.objdump.txt:61, log.objdump.txt:63, log.objdump.txt:64 |
| pow | glibc_flt64.cuh:972 | log table reduction | 0xa1921 | pow.objdump.txt:58 |
| pow | glibc_flt64.cuh:973 | high log scale | 0xa190a | pow.objdump.txt:53 |
| pow | glibc_flt64.cuh:975 | low log scale | 0xa1910 | pow.objdump.txt:54 |
| pow | glibc_flt64.cuh:982 | log residual | 0xa1956 | pow.objdump.txt:70 |
| pow | glibc_flt64.cuh:984 | log polynomial | 0xa192f, 0xa1938, 0xa195f, 0xa196c, 0xa1975 | pow.objdump.txt:61, pow.objdump.txt:63, pow.objdump.txt:72, pow.objdump.txt:75, pow.objdump.txt:77 |
| pow | glibc_flt64.cuh:985 | log polynomial low accumulation | 0xa1988 | pow.objdump.txt:81 |
| pow | glibc_flt64.cuh:998 | overflow result | 0xa1d2a | pow.objdump.txt:304 |
| pow | glibc_flt64.cuh:1044 | multiply and Shift | 0xa19e3 | pow.objdump.txt:101 |
| pow | glibc_flt64.cuh:1047 | two exp reduction steps | 0xa19f5, 0xa1a05 | pow.objdump.txt:105, pow.objdump.txt:109 |
| pow | glibc_flt64.cuh:1054 | exp polynomial and tail | 0xa1a2c, 0xa1a43, 0xa1a4c, 0xa1a55 | pow.objdump.txt:119, pow.objdump.txt:124, pow.objdump.txt:126, pow.objdump.txt:128 |
| pow | glibc_flt64.cuh:1058 | normal result | 0xa1a67 | pow.objdump.txt:132 |
| pow | glibc_flt64.cuh:1133 | product residual and tail | 0xa19ba, 0xa19d2 | pow.objdump.txt:92, pow.objdump.txt:97 |

CORE-MATH FMA sites use the upstream explicit fma calls. They are not matched to glibc cos or acos instructions. Every other floating-point operation in the function bodies is pinned separately.

| CORE-MATH function | Header lines containing explicit FMA |
|---|---|
| cos | 2218, 2223, 2229, 2230, 2245, 2246, 2257, 2258, 2447, 2494, 2519, 2619 |
| acos | 2784, 2816, 2835, 2852, 2857, 2864, 2865, 2866, 2867 |

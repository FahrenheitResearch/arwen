# Third-party notices

recast-radar-tools is licensed under MIT OR Apache-2.0 (`LICENSE-MIT`,
`LICENSE-APACHE`). Parts of it derive from the software below, whose
licenses require these notices.

## Py-ART

`crates/recast-radar-correct/src/dealias_pyart.rs` is a Rust port, modified
from the original, of the region-based velocity dealiasing in Py-ART
(`pyart/correct/region_dealias.py`, https://github.com/ARM-DOE/pyart).

```
Copyright (c) 2013, UChicago Argonne, LLC
All rights reserved.

Copyright 2013 UChicago Argonne, LLC. This software was produced under U.S.
Government contract DE-AC02-06CH11357 for Argonne National Laboratory (ANL),
which is operated by UChicago Argonne, LLC for the U.S. Department of Energy.
The U.S. Government has rights to use, reproduce, and distribute this
software.  NEITHER THE GOVERNMENT NOR UCHICAGO ARGONNE, LLC MAKES ANY
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
SOFTWARE.  If software is modified to produce derivative works, such modified
software should be clearly marked, so as not to confuse it with the version
available from ANL.

Additionally, redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following conditions
are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UCHICAGO ARGONNE, LLC AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL UCHICAGO ARGONNE, LLC OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## bzip2 / libbzip2

`crates/recast-radar-bzip2/src/rand.rs` reuses the 512 values of the block
randomisation table in bzip2/libbzip2 1.0.8 `randtable.c`
(https://sourceware.org/bzip2/). The code is original.

```
Copyright (C) 1996-2019 Julian R Seward. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software in a
   product, an acknowledgment in the product documentation would be
   appreciated but is not required.
3. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
4. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Test data

The files under `testdata/` are real radar data from their publishers, not
covered by this repository's license. Each manifest entry
(`testdata/manifest.toml`, `testdata/*/manifest.toml`) names its source and
license, and `docs/testdata/corpus.md` lists them. NOAA/NWS NEXRAD data is in
the public domain. Files under CC BY 4.0 (OPERA, FMI, DWD, SMHI, the Bureau
of Meteorology via NCI, Zenodo deposits) and IMGW-PIB data require attribution
to their publishers, as those listings give. Entries tagged
`not-redistributed` are not included in the repository.

## BUFR tables

`crates/recast-radar-io-bufr/tables/` embeds BUFR tables, generated by
`tools/bufr_tables.py`:

- `wmo_b.tsv`, `wmo_d.tsv`: the WMO BUFR edition 4 master tables B and D
  (https://github.com/wmo-im/BUFR4, commit dcc0316b), under the MIT
  License:

  ```
  Copyright (c) 2020-2024

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

- `meteofrance_v11_*.tsv`, `meteofrance_v12_*.tsv`: Meteo-France's local
  tables B and D, versions 11 and 12, as numpy_bufr ships them
  (https://github.com/Bram94/numpy_bufr, commit 0641ed67, `Tables.zip`),
  under the MIT License:

  ```
  Copyright (c) 2018 Bram94

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

The decoder itself is written from the WMO specification (WMO-No. 306,
FM 94 BUFR); it contains no code from numpy_bufr or any other decoder.

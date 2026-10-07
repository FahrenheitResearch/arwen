"""Assert the wrapper contains each pinned WRF routine byte for byte."""
import hashlib
from pathlib import Path
import sys

source=Path(sys.argv[1]).read_bytes()
wrapper=Path(sys.argv[2]).read_bytes()
lines=source.splitlines(keepends=True)
for name in (b"advect_scalar_mono",b"advect_scalar_pd"):
    start=next(i for i,line in enumerate(lines) if line.strip().lower().startswith(b"subroutine "+name+b" "))
    end=next(i for i in range(start,len(lines)) if lines[i].strip().lower()==b"end subroutine "+name)
    body=b"".join(lines[start:end+1])
    assert body in wrapper, name.decode()+": wrapper changed source bytes"
    print(name.decode(),"source_lines",start+1,end+1,"bytes",len(body),
          "sha256",hashlib.sha256(body).hexdigest(),"byte-identical")

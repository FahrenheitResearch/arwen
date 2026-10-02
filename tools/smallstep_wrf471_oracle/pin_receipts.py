"""Pin oracle inputs, answers, builders and launch adapters in one manifest."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    fixture=root/"tests/data/wrf471_smallstep"
    manifest=fixture/"oracle-sha256sums.json"
    files=[p for p in fixture.rglob("*") if p.is_file() and p!=manifest]
    files += [p for p in Path(__file__).parent.iterdir()
              if p.is_file() and p.suffix in (".py",".F90",".sh",".md")]
    files += sorted((root/"gpuwm/verify").glob("smallstep*_oracle.py"))
    files += sorted((root/"tests").glob("test_smallstep*.py"))
    files.append(root/"gpuwm/core/kernels/acoustic.cu")
    content={str(p.relative_to(root)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(set(files))}
    manifest.write_text(json.dumps(content,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"files":len(content),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()}))


if __name__=="__main__":
    main()

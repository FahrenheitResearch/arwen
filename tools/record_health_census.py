"""Record measured health census constants from the full constructor owner."""
import argparse,ast,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("receipt",type=Path);a=p.parse_args()
 raw=a.receipt.read_bytes();j=json.loads(raw)
 assert not j["over_ceiling"]
 rows={r["selection"]:r for r in j["rows"]}
 peak=j["worst_count"];worst=rows[j["worst_selection"]]
 assert peak==max(r["peak"] for r in rows.values())
 constants={"WORST_MEASURED_COUNT":peak,"WORST_ROOT_COUNT":worst["per_domain"]["1"],
 "WORST_CHILD_COUNT":worst["per_domain"]["4"],
 "CALIBRATION_CHILD_COUNT":rows["mp18-lsm2-pbl0-sfclay1-cu0-km1"]["per_domain"]["4"]}
 path=ROOT/"tests/test_health_field_census.py";s=path.read_text()
 for name,value in constants.items():
  s,n=re.subn(r"(?m)^"+name+r" = \d+$",name+" = "+str(value),s);assert n==1,name
 peaks=sorted(k for k,r in rows.items() if r["peak"]==peak)
 s,n=re.subn(r"WORST_SELECTIONS = frozenset\(\{.*?\}\)","WORST_SELECTIONS = frozenset("+repr(set(peaks))+")",s,flags=re.S);assert n==1
 note="# Full constructor census: "+str(len(j["rows"]))+" measured rows, "+str(len(j["rejected"]))+" refusals, peak "+str(peak)+", root "+str(constants["WORST_ROOT_COUNT"])+", child "+str(constants["WORST_CHILD_COUNT"])+".\n# Receipt sha256: "+hashlib.sha256(raw).hexdigest()+"\n"
 s=re.sub(r"(?m)^# Full constructor census:.*\n# Receipt sha256:.*\n","",s)
 s=s.replace("WORST_MEASURED_COUNT =",note+"WORST_MEASURED_COUNT =",1)
 ast.parse(s);path.write_text(s)
 print(json.dumps({"constants":constants,"peaks":peaks,"receipt_sha256":hashlib.sha256(raw).hexdigest(),"rows":len(j["rows"]),"refusals":len(j["rejected"])},sort_keys=True))
if __name__=="__main__":main()

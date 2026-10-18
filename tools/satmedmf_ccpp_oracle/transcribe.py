"""Pinned fixed-form transcription for the CCPP column routines.
Only the constructs present in the pinned files are admitted. Unknown syntax fails.
The resulting CUDA is reviewable source with upstream file and line references.
"""
from __future__ import annotations
import argparse,json,re,struct
from pathlib import Path

def split(s):
    out=[]; start=depth=0
    for i,c in enumerate(s):
        if c=="(":depth+=1
        elif c==")":depth-=1
        elif c=="," and depth==0:out.append(s[start:i].strip()); start=i+1
    out.append(s[start:].strip())
    return out

def statements(path):
    out=[]
    for line,text in enumerate(path.read_text().splitlines(),1):
        if not text.strip() or text[0] in "cC!*":continue
        body=text[6:72].split("!")[0].strip().lower()
        if not body:continue
        if text[:5].strip():raise ValueError(f"{path}:{line}: unsupported label")
        if text[5:6] not in (""," ","0"):
            if not out:raise ValueError("orphan continuation")
            out[-1]=(out[-1][0],out[-1][1]+" "+body)
        else:out.append((line,body))
    return out

TOKEN=re.compile(r"\s*(\.[a-z]+\.|\*\*|<=|>=|==|/=|[()+*/,\-<>:]|(?:\d+\.\d*|\.\d+|\d+)(?:[ed][+\-]?\d+)?(?:_[a-z0-9_]+)?|[a-z][a-z0-9_]*)")
PRE={".or.":10,".and.":20,".eqv.":5,".neqv.":5,"==":30,"/=":30,"<":30,">":30,"<=":30,">=":30,
     ".eq.":30,".ne.":30,".lt.":30,".gt.":30,".le.":30,".ge.":30,"+":40,"-":40,"*":50,"/":50,"**":70}
TYPE={"i":"int","f":"float","d":"double","b":"bool"}
def common(a,b):
    return "d" if "d" in (a,b) else "f" if "f" in (a,b) else "i"

class Expr:
    def __init__(self,text,symbols):
        text=text.strip(); self.t=[]; pos=0; self.symbols=symbols
        while pos<len(text):
            m=TOKEN.match(text,pos)
            if not m:raise ValueError(f"unknown expression suffix {text[pos:]!r}")
            self.t.append(m.group(1)); pos=m.end()
        self.pos=0
    def pop(self):
        if self.pos>=len(self.t):raise ValueError("expression ended early")
        v=self.t[self.pos]; self.pos+=1; return v
    def peek(self):
        return self.t[self.pos] if self.pos<len(self.t) else None
    def parse(self,level=0):
        token=self.pop()
        if token=="(":
            a=self.parse(); assert self.pop()==")"
        elif token in ("-","+",".not."):
            a=self.parse(60 if token!=".not." else 25)
            value=None if a[2] is None or token==".not." else -a[2] if token=="-" else a[2]
            a=((f"(!{a[0]})" if token==".not." else f"({token}{a[0]})"),a[1],value)
        elif token in (".true.",".false."):
            a=("true" if token==".true." else "false","b",None)
        elif re.match(r"\d|\.",token):
            kind="d" if "_" in token or "d" in token else "f" if "." in token or "e" in token else "i"
            number=token.split("_")[0].replace("d","e")
            if kind=="i":a=(number,kind,int(number))
            else:
                value=float(number)
                if kind=="f":value=struct.unpack("f",struct.pack("f",value))[0]
                number=value.hex()+("f" if kind=="f" else "")
                a=(number,kind,float(token.split("_")[0].replace("d","e")))
        else:
            if self.peek()=="(":
                self.pop(); args=[]
                while self.peek()!=")":
                    args.append(self.parse())
                    if self.peek()!=",":break
                    self.pop()
                assert self.pop()==")"
                if token in self.symbols and self.symbols[token]["dims"]:
                    a=(token+"("+",".join(v[0] for v in args)+")",self.symbols[token]["type"],None)
                elif token in ("min","max"):
                    kind=common(args[0][1],args[1][1])
                    value=args[0][0]
                    for b in args[1:]:
                        kind=common(kind,b[1])
                        value=f"sm_{token}<{TYPE[kind]}>({value},{b[0]})"
                    a=(value,kind,None)
                elif token in ("sqrt","exp","log","abs"):
                    kind=args[0][1]; a=(f"sm_{token}({args[0][0]})",kind,None)
                elif token in ("float","real"):
                    kind="d" if len(args)>1 else "f"
                    a=(f"(({TYPE[kind]})({args[0][0]}))",kind,None)
                elif token=="fpvs":
                    a=(f"sm_fpvs({args[0][0]},ws.table)","d",None)
                elif token=="int":
                    a=(f"((int)({args[0][0]}))","i",None)
                else:raise ValueError(f"unknown intrinsic {token}")
            else:
                if token not in self.symbols:raise ValueError(f"undeclared symbol {token}")
                a=(token,self.symbols[token]["type"],None)
        while self.peek() in PRE and PRE[self.peek()]>=level:
            op=self.pop(); b=self.parse(PRE[op]+(0 if op=="**" else 1))
            kind=common(a[1],b[1]); val=None
            if op=="**":
                if b[1]=="i":
                    code=f"sm_powi(({TYPE[kind]})({a[0]}),{b[0]})"
                elif b[2] in (2.0,1.0,-1.0):
                    code=f"sm_mul(({TYPE[kind]})({a[0]}),({TYPE[kind]})({a[0]}))" if b[2]==2 else a[0] if b[2]==1 else f"sm_div(({TYPE[kind]})1,({TYPE[kind]})({a[0]}))"
                else:code=f"sm_pow(({TYPE[kind]})({a[0]}),({TYPE[kind]})({b[0]}))"
            elif op in ("+","-","*","/") and kind in ("f","d"):
                name={"+":"add","-":"sub","*":"mul","/":"div"}[op]
                code=f"sm_{name}(({TYPE[kind]})({a[0]}),({TYPE[kind]})({b[0]}))"
            else:
                logical={".or.":"||",".and.":"&&",".eqv.":"==",".neqv.":"!="}
                rel={".eq.":"==",".ne.":"!=","/=":"!=",".lt.":"<",".gt.":">",".le.":"<=",".ge.":">="}
                cop=logical.get(op,rel.get(op,op))
                code=f"({a[0]} {cop} {b[0]})"
                if op in logical or op in rel or op in ("==","<",">","<=",">="):kind="b"
            a=(code,kind,val)
        return a
    def all(self):
        a=self.parse()
        if self.peek() is not None:raise ValueError(f"unused expression tokens {self.t[self.pos:]}")
        return a[0]

DUMMY_DIMS={
 "kinver":["im"],"kpbl":["im"],"dtidx":["100+ntrac","1"],"dtend":["im","km","ntrac"],
 **{a:["im"] for a in "usfco vsfco xmu garea zvfun sigmaf psk rbsoil zorl tsea u10m v10m fm fh evap heat stress spd1 dusfc dvsfc dtsfc dqsfc hpbl".split()},
 **{a:["im","km"] for a in "dv du tdt u1 v1 t1 swh hlw del prsl prslk phil dkt dku".split()},
 "q1":["im","km","ntrac"],"rtg":["im","km","ntrac"],"prsi":["im","km+1"],"phii":["im","km+1"]}

CONSTANTS={"grav":"sm_g","cp":"sm_cp","rv":"sm_rv","hvap":"sm_hvap","fv":"sm_fvirt","eps":"sm_eps","epsm1":"sm_epsm1"}

class Routine:
    def __init__(self,path,name):
        self.path,self.name=path,name
        st=statements(path)
        first=next(i for i,(_,s) in enumerate(st) if s.startswith("subroutine "+name+"("))
        last=next(i for i in range(first+1,len(st)) if st[i][1].startswith("end subroutine") or st[i][1]=="end")
        self.st=st[first+1:last]
        self.args=split(st[first][1].split("(",1)[1].rsplit(")",1)[0])
        self.symbols={}; self.parameters=[]; self.body=[]
        if name in ("mfpbltq","mfscuq"):
            self.symbols.update({k:{"type":"d","dims":[],"constant":v} for k,v in CONSTANTS.items()})
        for line,s in self.st:
            m=re.match(r"(real\s*\(kind=kind_phys\)|integer|logical|character\s*\(len=\*\))(.*)",s)
            if m:
                kind={"integer":"i","logical":"b"}.get(m[1],"d")
                if m[1].startswith("character"):kind="c"
                rest=m[2]
                global_dims=[]
                if "::" in rest:
                    attributes,rest=rest.split("::",1)
                    dm=re.search(r"dimension\((.*?)\)",attributes)
                    if dm:global_dims=split(dm[1])
                else:attributes=""
                for term in split(rest):
                    if not term:continue
                    if "=" in term:
                        term,value=term.split("=",1); self.parameters.append((line,term.strip(),value.strip()))
                    if "(" in term:
                        var=term.split("(",1)[0].strip(); dims=split(term.split("(",1)[1].rsplit(")",1)[0])
                    else:var=term.strip(); dims=global_dims
                    if any(d==":" for d in dims):
                        dims=DUMMY_DIMS[var]
                    self.symbols[var]={"type":kind,"dims":dims,"ref":("intent(out)" in attributes or "intent(inout)" in attributes) and not dims}
            elif s.startswith("parameter("):
                for term in split(s[10:-1]):
                    var,value=term.split("=",1); self.parameters.append((line,var.strip(),value.strip()))
            elif s.startswith(("use ","implicit ","module ","contains")):pass
            else:self.body.append((line,s))
        missing=set(self.args)-set(self.symbols)
        if missing:raise ValueError(f"{name}: missing declarations {missing}")
    def ex(self,text):return Expr(text,self.symbols).all()
    def ctype(self,var):
        info=self.symbols[var]
        typ="char" if info["type"]=="c" else TYPE[info["type"]]
        if info["dims"]:return typ+"*"
        if typ=="char":return "char*"
        return typ+("&" if info.get("ref") else "")
    def dims(self,var):
        dims=self.symbols[var]["dims"]; lower=[]; size=[]
        for dim in dims:
            if ":" in dim:
                lo,hi=dim.split(":",1); lower.append(lo); size.append(f"(({hi})-({lo})+1)")
            else:lower.append("1"); size.append(dim)
        return size+["1"]*(3-len(size)),lower+["1"]*(3-len(lower))
    def emit(self,routines):
        header="__device__ void sm_"+self.name+"("+", ".join(self.ctype(a)+" "+("raw_"+a if self.symbols[a]["dims"] else a) for a in self.args)+", SMWorkspace& ws)"
        out=[header+" {","SMFrame frame(ws);"]
        for var,info in self.symbols.items():
            if "constant" in info:out.append(f"const double {var}={info['constant']};");continue
            if var in self.args and not info["dims"]:continue
            if info["dims"]:
                sizes,lower=self.dims(var)
                ptr="raw_"+var if var in self.args else f"ws.alloc<{TYPE[info['type']]}>("+ "*".join("("+s+")" for s in sizes)+")"
                out.append(f"SMView<{TYPE[info['type']]}> {var}({ptr}, "+", ".join(sizes+lower)+");")
            elif var not in self.args:out.append(TYPE[info["type"]]+" "+var+";")
        for line,var,val in self.parameters:out.append(f"{var} = {self.ex(val)}; // {self.path.name}:{line}")
        def stmt(s):
            if s=="return":return "return;"
            if "=" in s:
                left,right=s.split("=",1)
                if left.strip()=="errmsg":return "/* CCPP success message is empty. */"
                return self.ex(left.strip())+" = "+self.ex(right.strip())+";"
            raise ValueError(f"unknown statement {s}")
        for line,s in self.body:
            prefix=f"// {self.path.name}:{line}\n"
            if s.startswith("do "):
                var,values=s[3:].split("=",1); var=var.strip(); parts=split(values); lo,hi=parts[:2]; step=parts[2] if len(parts)>2 else "1"
                bound="<=" if not step.strip().startswith("-") else ">="
                code=f"for({var}={self.ex(lo)}; {var}{bound}{self.ex(hi)}; {var}+={self.ex(step)}) {{"
            elif s in ("enddo","end do","endif","end if"):code="}"
            elif s=="else":code="} else {"
            elif s.startswith(("elseif","else if")):
                cond=s[s.index("(")+1:s.rindex(")")]
                code="} else if("+self.ex(cond)+") {"
            elif s.startswith("if"):
                depth=0; end=None
                for idx in range(s.index("("),len(s)):
                    if s[idx]=="(":depth+=1
                    elif s[idx]==")":
                        depth-=1
                        if depth==0:end=idx;break
                cond=s[s.index("(")+1:end]; tail=s[end+1:].strip()
                code="if("+self.ex(cond)+") "+("{" if tail=="then" else stmt(tail))
            elif s.startswith("call "):
                name=s[5:s.index("(")].strip(); args=split(s[s.index("(")+1:-1])
                called=routines[name]
                if len(args)!=len(called.args):raise ValueError("call argument mismatch")
                params=[]
                for formal,actual in zip(called.args,args):
                    params.append(actual+".p" if called.symbols[formal]["dims"] else self.ex(actual))
                code="sm_"+name+"("+", ".join(params)+", ws);"
            elif ":" in s:
                # Only the CCPP optional diagnostic slice assignment uses this form.
                # Expand explicitly, retaining the scalar operation order.
                if not s.startswith("dtend("):raise ValueError("unknown array assignment")
                expanded=s.replace("1:im","sm_i").replace("1:km","sm_k")
                self.symbols.update(sm_i={"type":"i","dims":[]},sm_k={"type":"i","dims":[]})
                code="for(int sm_k=1;sm_k<=km;++sm_k) for(int sm_i=1;sm_i<=im;++sm_i) "+stmt(expanded)
            else:code=stmt(s)
            out.append(prefix+code)
        out.append("}")
        return "\n".join(out),header+";"

def transcribe(source,out):
    spec=[("physics/PBL/tridi.f",n) for n in ("tridi1","tridi2","tridin","tridit")]
    spec += [("physics/PBL/mfpbltq.f","mfpbltq"),("physics/PBL/SATMEDMF/mfscuq.f","mfscuq"),
             ("physics/PBL/SATMEDMF/satmedmfvdifq.F","satmedmfvdifq_run")]
    routines={n:Routine(source/p,n) for p,n in spec}
    bodies=[]; prototypes=[]
    for name,routine in routines.items():
        try:body,proto=routine.emit(routines)
        except Exception as e:raise ValueError(f"{name}: {e}") from e
        bodies.append(body);prototypes.append(proto)
    text="// Apache-2.0. Pinned CCPP transcription. See tools/satmedmf_ccpp_oracle/source.json.\n"+"\n".join(prototypes+bodies)+"\n"
    out.write_text(text)
    return routines

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("source",type=Path);p.add_argument("output",type=Path)
 a=p.parse_args();transcribe(a.source,a.output)

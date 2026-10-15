"""Instrument assigned scalars in the first defined-reference column.

This separate diagnostic listing never replaces the pristine reference.
"""
from pathlib import Path
import argparse, re
from transcribe import statements, Routine


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    original=list(statements(a.source.read_text()))
    start=next(i for i,(_,s) in enumerate(original) if s.startswith('subroutine urban('))
    end=next(i for i in range(start+1,len(original)) if original[i][1].startswith('end subroutine urban'))
    routine=Routine(original[start:end+1])
    targets={s:(n,re.fullmatch(r'(\w+)\s*=(.*)',s)[1]) for n,s in routine.body
             if re.fullmatch(r'(\w+)\s*=(.*)',s)
             and routine.types.get(re.fullmatch(r'(\w+)\s*=(.*)',s)[1]) in ('float','double','int')
             and re.fullmatch(r'(\w+)\s*=(.*)',s)[1] not in routine.arrays}
    raw=a.candidate.read_text().splitlines(True); parsed=list(statements(''.join(raw)))
    inserts={};active=False
    for i,(line,s) in enumerate(parsed):
        if s.startswith('subroutine urban('):active=True
        if s.startswith('end subroutine urban'):active=False
        if active and s in targets:
            number,name=targets[s]
            source_name='lambda' if name=='f_lambda' else name
            at=parsed[i+1][0]-1
            inserts.setdefault(at,[]).append(f'   if (dbg_count==1) write(*,\'(A,I0,1X,A,1X,ES26.17E3)\') "TRACE ", {number}, "{name}", real({source_name},8)\n')
    text=''.join(''.join(inserts.get(i,[]))+line for i,line in enumerate(raw))
    anchor='   IF (distributed_aerodynamics_option .and. groption == 1) THEN'
    text=text.replace(anchor,'   integer, save :: dbg_count=0\n   dbg_count=dbg_count+1\n'+anchor,1)
    a.output.write_text(text)


if __name__=='__main__':main()

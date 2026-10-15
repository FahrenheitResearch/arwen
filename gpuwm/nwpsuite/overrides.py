"""Data-declared door flags, with admission controls kept ahead of transport."""
from copy import deepcopy
import re
from .registry import SuiteRefusal

def selection(selector):
    match=re.fullmatch(r'(?:members\[(\d+)(?:\.\.(\d+))?\]\.)?(physics|door)\.([\w-]+)',selector)
    if not match:
        raise SuiteRefusal(f'unknown override key {selector!r}; prevents a door override being silently dropped')
    return match

def validate_flags(row, shape):
    for key,value in row.get('door_flags',{}).items():
        spec=shape.get('door_flags',{}).get(key)
        if spec is None:
            raise SuiteRefusal(f"{row['id']}: unknown door flag {key!r}; prevents a silently dropped argument")
        valid={'boolean':isinstance(value,bool),'integer':isinstance(value,int) and not isinstance(value,bool),
               'number':isinstance(value,(int,float)) and not isinstance(value,bool),'string':isinstance(value,str)}[spec['type']]
        if not valid or ('choices' in spec and value not in spec['choices']):
            raise SuiteRefusal(f"{row['id']}: invalid door flag {key!r} value {value!r}")
        if spec['admission']=='refuse':
            raise SuiteRefusal(f"{row['id']}: door flag {key!r} bypasses admission or is an ignored base slot; prevents HAILCAST/mp_tend_radar style silent drops")
        if spec['admission']=='placement':
            raise SuiteRefusal(f"{row['id']}: door flag {key!r} is owned by card placement; prevents launching outside reserved lanes")

def effective_row(row, overrides, shape, member=None):
    result=deepcopy(row)
    for override in overrides:
        if override['system']!=row['id']: continue
        for selector,value in override['set'].items():
            match=selection(selector)
            if match[1] is not None:
                if member is None or not int(match[1])<=member<=int(match[2] or match[1]): continue
            result.setdefault('physics' if match[3]=='physics' else 'door_flags',{})[match[4]]=value
    for key,value in result.get('door_flags',{}).items():
        spec=shape['door_flags'][key]
        if spec['admission']!='row': continue
        path=spec['row_key'].split('.'); cursor=result
        for part in path[:-1]: cursor=cursor.setdefault(part,{})
        if spec['unit']=='h': value=str(value)+'h'
        elif spec['unit']=='source': value=[value]
        elif path[-1]=='members' and isinstance(cursor.get('members'),dict):
            cursor['members']['perturbed']=value; continue
        cursor[path[-1]]=value
    return result

def argv_flags(argv, row, shape):
    argv=list(argv)
    for key,value in sorted(row.get('door_flags',{}).items()):
        spec=shape['door_flags'][key]
        if spec['admission']=='row': continue # already priced and carried through the template
        flag='--'+key
        while flag in argv:
            index=argv.index(flag)
            del argv[index:index+(1 if spec['type']=='boolean' else 2)]
        if spec['type']=='boolean':
            if value: argv.append(flag)
            elif spec['action']=='boolean_optional': argv.append('--no-'+key)
        else: argv.extend([flag,str(value)])
    return argv

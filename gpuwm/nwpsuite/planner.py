"""Deterministic reservations. Engines own physics and allocation estimates."""
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from itertools import combinations
import re
from .registry import Registries, ROOT, SuiteRefusal
from .validator import cycle_row, suite_hash
from .overrides import argv_flags, effective_row

BLOCK = 21600

def seconds(value):
    m = re.fullmatch(r"(?:ready\+)?([0-9]+(?:\.[0-9]+)?)([smhd])", value)
    if not m:
        raise SuiteRefusal(f"invalid duration {value!r}")
    return float(m[1]) * {"s":1, "m":60, "h":3600, "d":86400}[m[2]]

@lru_cache(maxsize=64)
def recipe_memory(recipe, cards, memory, lead=0, physics=()):
    from gpuwm.experiment import load_experiment
    from gpuwm.core.devices import DeviceOptions
    from gpuwm.core.devices_memory import estimate_devices
    exp = load_experiment(ROOT / "configs/recipes" / (recipe + ".toml"))
    if physics:
        exp = replace(exp, domains=(replace(exp.root, run=replace(exp.root.run, **dict(physics))),))
    if lead:
        exp = replace(exp, run_seconds=lead)
    report = estimate_devices(exp, options=DeviceOptions(count=cards), vram_gib=memory/1024, streaming_boundaries=True)
    return max(r["total_bytes"] for r in report["cards"])/1048576, report["host_bytes"]/1048576

def runtime(rate, lead):
    startup=0 if rate.get('all_in_includes_startup') else rate['startup_s']
    return max(startup + (rate["all_in_s_per_fh"] or 0)*lead/3600, rate.get("reservation_floor_s", 0))

def place(lanes, count, ready, length, deadline, eligible, avoid=(), horizon=BLOCK):
    best = None
    for ids in combinations(eligible, count):
        for start in sorted({ready, *(b for i in ids for _, b in lanes[i] if b >= ready)}):
            end = start + length
            if end > min(horizon, deadline)+1e-7:
                break
            if all(all(end <= a+1e-7 or start >= b-1e-7 for a,b in lanes[i]) for i in ids):
                item = (start, ids, end)
                score = (start, len(set(ids) & set(avoid)), ids)
                if best is None or score < (best[0], len(set(best[1]) & set(avoid)), best[1]):
                    best = item
                break
    if best:
        for i in best[1]:
            lanes[i].append((best[0],best[2]))
            lanes[i].sort()
    return best

def argv_for(row, shape, ids, hour, member):
    values = {"mesh":row.get("mesh",""), "preset":row.get("preset",""), "source":"{selected.ic}",
              "lead_hours":str(seconds(row["cycle"]["lead"])/3600), "device_ids":",".join(map(str,range(len(ids)))),
              "cards":str(len(ids)), "members":"1", "run_plan":f"{{artifacts.{row['id']}.{hour:02d}.{member}.run_plan}}"}
    values.update({"prepared."+k:v for k,v in row.get("prepared",{}).items()})
    if shape.get('override_transport') == 'manifest.forecast_flags':
        values['prepared.manifest']=f"{{artifacts.{row['id']}.{hour:02d}.{member}.manifest}}"
    template=[*shape['alternate_argv_template' if shape.get('override_transport')=='derived_config' and row.get('door_flags') else 'argv_template']]
    values.update(derived_config=f"{{artifacts.{row['id']}.{hour:02d}.{member}.config}}",cycle='{cycle}')
    values['perturb.recipe']=row.get('perturb',{}).get('recipe','{perturb.recipe}')
    if len(ids)>1:
        template.extend(shape.get('multicard_argv_template',[]))
    argv = [re.sub(r"\{([^{}]+)\}",lambda m:values.get(m[1],m[0]),t) for t in template]
    if shape.get("override_transport") == "--physics KEY=VALUE":
        for k,v in sorted(row.get("physics",{}).items()):
            argv.extend(["--physics",f"{k}={v}"])
    if row.get("lts") == "one-card":
        argv.append("--fullstep-local-timestep")
    argv=argv_flags(argv,row,shape)
    return argv, sorted({m for t in argv for m in re.findall(r"\{([^{}]+)\}",t)})

def plan_suite(suite, *, registries=None, base=None, **options):
    reg = registries or Registries(suite.get("policy",{}).get("registry_dir"))
    rates = reg.rates(suite.get("policy",{}).get("rates"),base)
    p = dict(schema="gpuwm.suite.plan.v1", suite=suite.get("name"), region=suite["region"], suite_sha256=suite_hash(suite),
             admitted=False, exit_code=2, block_seconds=BLOCK, cycle_hours=[0,6,12,18], boxes=[], rows=[], jobs=[],
             measurement_slots=[], refusals=[], daily_card_hours=0, what_if=None, execution_ready=False,
             ready_basis="source_cycles: actual posting readiness", defaults=suite.get("defaults",{}), overrides=suite.get("override",[]))
    boxes = {b["id"]:deepcopy(b) for b in suite["box"]}
    for b in boxes.values():
        if not {"cards","memory_mib","card_model"} <= b.keys():
            raise SuiteRefusal(f"box {b['id']}: missing check inventory; prevents unknown hardware admission")
        b.setdefault("inventory",[{"index":i,"name":b["card_model"],"memory_mib":b["memory_mib"]} for i in range(b["cards"])])
        if len(b["inventory"]) != b["cards"] or len({r["index"] for r in b["inventory"]}) != b["cards"]:
            raise SuiteRefusal(f"box {b['id']}: inconsistent inventory count or duplicate card index")
        b.update(block_budget_card_hours=b["cards"]*6*(1-suite.get("policy",{}).get("headroom",.2)),daily_card_hours=0,blocks=[])
        p["boxes"].append(b)
    def refuse(row, reason, rate, budget, required, unit):
        over = max(0,required-budget) if budget is not None and required is not None else None
        p["refusals"].append(dict(row=row,reason=reason,rate=rate,budget=budget,required=required,overrun=over,unit=unit,
            message=f"{row}: {reason}; rate {rate}, budget {budget} {unit}, required {required}, overrun {over}; prevents an unadmitted run"))
    rows = sorted(suite["system"],key=lambda r:(r.get("priority",2),r["id"]))
    sums = {r["id"]:dict(id=r["id"],shape=r["shape"],box=r["box"],rate=None,daily_card_hours=0,settings=r,
              dependencies=[r["from"]] if r.get("from") else [],stages=["da","forecast","ensemble","products","verify","publish","retain"]) for r in rows}
    p["rows"] = list(sums.values())
    for block in p["cycle_hours"]:
        lanes = {k:{r["index"]:[] for r in b["inventory"]} for k,b in boxes.items()}
        used = {k:0. for k in boxes}
        def reserve(row, rate, count, length, ready, deadline, phase, hour, member=0, deps=()):
            b = boxes[row["box"]]; key=b["id"]; peak=rate["peak_mib_per_card"]; host=rate.get("host_memory_mib",0)
            shape=reg.shapes[row['shape']]
            changed={k:v for k,v in row.get('door_flags',{}).items() if shape['door_flags'][k]['admission']=='rate'}
            if phase=='forecast' and any(rate.get('door_flags',{}).get(k)!=v for k,v in changed.items()):
                refuse(row['id'],'door flag change lacks a matching measured rate: '+','.join(changed),rate['id'],None,None,'qualified rate'); return
            if peak is None:
                if rate.get("memory_admission") == "recipe":
                    try:
                        translated=tuple(sorted((reg.catalog['physics_keys'][k]['config_key'],v) for k,v in row.get('physics',{}).items()))
                        peak,host=recipe_memory(row["recipe"],count,b["memory_mib"],seconds(row['cycle']['lead']),translated)
                    except (ValueError,RuntimeError,ImportError,OSError,KeyError) as e:
                        refuse(row["id"],"CPU memory admission unavailable: "+str(e),rate["id"],b["memory_mib"],None,"MiB/card"); return
                else:
                    peak=rate.get("capacity_bound_mib")
            eligible=[r["index"] for r in b["inventory"] if peak is not None and r["memory_mib"]>=peak and r["name"]==rate["card_model"]]
            if len(eligible)<count:
                model_memory=[r['memory_mib'] for r in b['inventory'] if r['name']==rate['card_model']]
                capacity=max(model_memory,default=b['memory_mib'])
                if peak is None or (model_memory and peak>capacity):
                    refuse(row["id"],"memory per card",rate["id"],capacity,peak,"MiB/card")
                else:
                    refuse(row["id"],"eligible card count",rate["id"],len(eligible),count,"cards")
                return
            if host>b.get("host_memory_mib",0):
                refuse(row["id"],"host memory",rate["id"],b.get("host_memory_mib",0),host,"MiB"); return
            charge=count*length/3600
            if used[key]+charge>b["block_budget_card_hours"]+1e-7:
                refuse(row["id"],f"block {block:02d} headroom",rate["id"],b["block_budget_card_hours"]-used[key],charge,"card-hours"); return
            previous=[j for j in p['jobs'] if j['row']==row['id'] and j['phase']==phase and j['block_hour']==block]
            avoid=previous[-1]['card_ids'] if previous and seconds(row['cycle']['every']) < BLOCK else ()
            spot=place(lanes[key],count,ready,length,deadline,eligible,avoid)
            if not spot:
                relaxed=place(deepcopy(lanes[key]),count,ready,length,float('inf'),eligible,avoid,horizon=float('inf'))
                needed=relaxed[2]-ready if relaxed else length
                refuse(row["id"],f"block {block:02d} deadline/card lanes",rate["id"],min(BLOCK,deadline)-ready,needed,"seconds"); return
            start,ids,end=spot
            argv,bindings=argv_for(row,reg.shapes[row["shape"]],ids,hour,member) if phase=="forecast" else ([],["da.shadow.command"])
            job=dict(id=f"{row['id']}:{hour:02d}:{phase}:{member}",row=row["id"],phase=phase,cycle_hour=hour,block_hour=block,
                ready_offset_s=(hour-block)*3600,start_s=start,end_s=end,deadline_s=deadline,box=key,card_ids=list(ids),rate=rate["id"],
                card_hours=charge,argv=argv,env={"CUDA_VISIBLE_DEVICES":",".join(map(str,ids))},dependencies=list(deps),bindings=bindings,
                execution_ready=not bindings,member=member,reservation_group=f"{row['id']}:{hour:02d}",settings=row,peak_mib_per_card=peak,
                override_transport=reg.shapes[row["shape"]].get("override_transport"))
            job['logical_card_ids']=list(range(count))
            job['source_selection']={'ic':row.get('ic',[]),'lbc':row.get('lbc',[]),'rule':'first-ready','requires_prepared_source_receipt':True}
            job['qualification_notes']=[rate.get('door_qualification')] if rate.get('door_qualification') else []
            job['rate_build_matches_inventory']=rate['build']==b.get('build')
            if reg.shapes[row['shape']].get('override_transport')=='derived_config':
                edits={'experiment.run_seconds':seconds(row['cycle']['lead']), 'fetch.hours':seconds(row['cycle']['lead'])/3600,
                       'fetch.cycle':'{cycle}', 'experiment.start_time':'{start_time}', 'devices.count':count}
                edits['fetch.source']='{selected.fetch.source}'
                job['bindings'].append('selected.fetch.source'); job['execution_ready']=False
                edits.update({'shared.'+reg.catalog['physics_keys'][k]['config_key']:v for k,v in row.get('physics',{}).items()})
                if row.get('perturb'):
                    edits['ensemble.recipe']=row['perturb']['recipe']; edits['ensemble.members']=1
                    edits['ensemble.base_seed']=member
                    edits['ensemble.stochastic']=row['perturb'].get('stochastic',[])
                job['config_edits']=edits
                job['recipe_reference']=reg.catalog['recipes'].get(row['recipe'],{}).get('path','configs/recipes/'+row['recipe']+'.toml')
                job['run_plan_document']={'schema':'gpuwm.run-plan.v1','name':job['id'],'route':'prepared',
                    'config':{'path':'{derived_config}'},'run_options':{'devices':count,'cycle':'{cycle}'},'output_root':'{prepared.outdir}'}
            if reg.shapes[row['shape']].get('override_transport')=='manifest.forecast_flags':
                job['manifest_forecast_flags']=[token for k,v in row.get('physics',{}).items() for token in ('--physics',reg.catalog['physics_keys'][k]['config_key']+'='+str(v))]
                job['manifest_selection']={'source':row.get('prepared',{}).get('manifest','{source_manifest}'),'member':member,'count':1}
            if reg.catalog['meshes'].get(row.get('mesh'),{}).get('runtime_registration_required'):
                job['env']['GPUWM_HEX_MESH_ROWS']=row.get('prepared',{}).get('mesh_rows','{prepared.mesh_rows}')
                if '{prepared.mesh_rows}' in job['env'].values():
                    job['bindings'].append('prepared.mesh_rows'); job['execution_ready']=False
            p["jobs"].append(job); used[key]+=charge; sums[row["id"]]["daily_card_hours"]+=charge
            return job
        shadows=[]
        for original in rows:
            for offset in range(6):
                hour=block+offset; row=cycle_row(original,hour)
                row=effective_row(row,suite.get('override',[]),reg.shapes[row['shape']])
                if not row.get("enabled",True): continue
                every=seconds(row["cycle"]["every"]); measure=row.get("admit")=="measure"
                if measure:
                    if block or offset: continue
                elif (every>86400 and (block or offset)) or (every<=86400 and (hour*3600 % every or hour not in row["cycle"].get("hours",range(24)))): continue
                b=boxes[row["box"]]; shape=reg.shapes[row["shape"]]
                count=row.get("cards_per_member",1) if shape["ensembles"] else row.get("cards",1)
                grid=row.get("mesh",row.get("domain",{}).get("grid"))
                candidates=[r for r in rates if r.get("role","forecast")=="forecast" and r["shape"]==row["shape"] and r["grid"]==grid
                            and r["card_model"]==b["card_model"] and r["cards"]==count and (not row.get("rate") or r["id"]==row["rate"])]
                if not row.get("rate"):
                    exact=[r for r in candidates if r["build"]==b.get("build")]
                    candidates=exact or candidates
                if len(candidates)!=1:
                    refuse(row["id"],"missing unambiguous shape/grid/cards/build rate",row.get("rate"),None,None,"card-hours"); continue
                rate=candidates[0]; sums[row["id"]]["rate"]=rate; length=runtime(rate,seconds(row["cycle"]["lead"]))
                if rate["kind"]=="P" and not measure:
                    refuse(row["id"],"projected rate requires admit=measure",rate["id"],0,length,"seconds"); continue
                if measure:
                    p["measurement_slots"].append(dict(row=row["id"],rate=rate,box=b["id"],estimated_seconds=length,cards=count,
                        cadence=row["cycle"]["every"],launch_allowed=False,qualification="measurement only")); continue
                ready=offset*3600; deadline=ready+seconds(row["cycle"].get("deadline","ready+6h")); deps=[]
                if row.get("from") and shape["ensembles"]:
                    parent=[j for j in p["jobs"] if j["row"]==row["from"] and j["cycle_hour"]==hour and j["phase"]=="forecast"]
                    if parent: ready=max(ready,max(j["end_s"] for j in parent)); deps=[j["id"] for j in parent]
                members=row.get("members",1); n=(members if isinstance(members,int) else members.get("perturbed",0)) if shape["ensembles"] else 1
                for member in range(n):
                    effective=effective_row(row,suite.get('override',[]),shape,member)
                    if not reserve(effective,rate,count,length,ready,deadline,"forecast",hour,member,deps): break
                if row.get("da",{}).get("method","none")!="none" and hour*3600 % seconds(row['da'].get('every',row['cycle']['every']))==0:
                    shadows.append((row,hour))
        for row,hour in shadows:
            da=row["da"]; b=boxes[row["box"]]
            dr=[r for r in rates if r.get("role")=="da-forecast" and r["grid"]==da.get("mesh") and r["card_model"]==b["card_model"]
                and (not da.get('rate') or r['id']==da['rate']) and (not da.get('cards') or r['cards']==da['cards'])]
            if da.get("mode")!="shadow" or len(dr)!=1:
                refuse(row["id"],"unqualified production DA or missing shadow rate",da.get("rate"),0,None,"receipt"); continue
            if dr[0]['kind']!='M':
                refuse(row['id'],'projected DA rate requires measurement-only work',dr[0]['id'],0,None,'qualified rate'); continue
            if 'cpu' not in da.get('backend',['cpu']):
                refuse(row['id'],'DA solve backend has no qualified memory rate',da.get('backend'),b.get('memory_mib'),None,'MiB/card'); continue
            shadow_rate=dict(dr[0],host_memory_mib=max(dr[0].get('host_memory_mib',0),da.get('host_memory_mib',0)))
            job=reserve(row,shadow_rate,shadow_rate["cards"],runtime(shadow_rate,seconds(da.get("lead","30h"))),0,BLOCK,"da-shadow",hour)
            if job:
                job["background_dependency"]={"row":da.get("b_from"),"cycle":"previous"}
                job["cpu_solve_seconds"]=next((r["startup_s"] for r in rates if r.get("role")=="da-solve" and r["cards"]==0),None)
                job['selected_solve_backend']='cpu'
        for k,b in boxes.items():
            b["blocks"].append(dict(cycle_hour=block,card_hours=used[k],budget_card_hours=b["block_budget_card_hours"],lanes={str(i):v for i,v in lanes[k].items()}))
            b["daily_card_hours"]+=used[k]
    p["daily_card_hours"]=sum(b["daily_card_hours"] for b in boxes.values())
    for slot in p['measurement_slots']:
        slot['windows']=[]
        box=boxes[slot['box']]
        for block in box['blocks']:
            copy={int(card):deepcopy(intervals) for card,intervals in block['lanes'].items()}
            spot=place(copy,slot['cards'],0,slot['estimated_seconds'],BLOCK,sorted(copy))
            remaining=block['budget_card_hours']-block['card_hours']
            if spot and slot['cards']*slot['estimated_seconds']/3600 <= remaining:
                slot['windows'].append(dict(block_hour=block['cycle_hour'],start_s=spot[0],end_s=spot[2],card_ids=list(spot[1])))
    p["admitted"]=not p["refusals"]; p["exit_code"]=0 if p["admitted"] else 2
    return p

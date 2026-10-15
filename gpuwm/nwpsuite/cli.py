"""Suite runtime front doors. Planner verbs are registered by the spec lane."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from .ledger import SuiteLedger, halt_row
from .run import SuiteRunner, apply_plan, load_plan
from .status import publish_status


def active_plan(root):
    config = SuiteLedger(root).state()["configuration"]
    if config is None:
        raise ValueError("suite_not_started: run or rehearse FILE first in this state root")
    return json.loads(Path(config["plan"]).read_text())


def runtime_main(args):
    try:
        root = Path(args.state_dir).resolve()
        verb = args.suite_verb
        if verb == "run":
            if args.bindings and not args.file:
                raise ValueError("suite_bindings_file: provide a suite FILE with --bindings, or resume the pinned plan")
            plan = load_plan(args.file, args.bindings) if args.file else active_plan(root)
            if args.detach:
                root.mkdir(parents=True, exist_ok=True)
                argv = [sys.executable, "-m", "gpuwm.cli", "suite", "run", "--state-dir", str(root), "--resume"]
                if args.file:
                    argv.append(str(Path(args.file).resolve()))
                if args.bindings:
                    argv.extend(["--bindings", str(Path(args.bindings).resolve())])
                with (root / "scheduler.log").open("ab") as log:
                    process = subprocess.Popen(argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
                print(json.dumps({"pid": process.pid, "state_dir": str(root)}))
                return 0
            window = root / "replay-window.json"
            replay = json.loads(window.read_text()) if args.resume and window.exists() else {}
            runner = SuiteRunner(plan, root, rehearsal=bool(replay))
            runner.run(cycles=replay.get("cycles"), once=args.once, poll=args.poll_seconds)
            return publish_status(runner.plan, root)["exit_code"]
        if verb == "status":
            document = publish_status(active_plan(root), root)
            print(json.dumps(document, indent=2))
            return document["exit_code"]
        if verb == "events":
            cursor = args.after
            while True:
                entries = SuiteLedger(root).settle()
                if entries and cursor < entries[0]["seq"]:
                    print(json.dumps({"event": "events_expired", "after": cursor, "first_available": entries[0]["seq"]}), flush=True)
                    cursor = entries[0]["seq"]
                for entry in entries:
                    if entry["seq"] < cursor:
                        continue
                    print(json.dumps(entry), flush=True)
                    cursor = entry["seq"] + 1
                if not args.follow:
                    return 0
                time.sleep(1)
        if verb in {"halt", "resume"}:
            plan = active_plan(root)
            targets = {r["id"] for r in plan["rows"] if r["id"] == args.row or r.get("suite_row") == args.row}
            if not targets:
                raise ValueError(f"suite_row_unknown: {args.row}")
            ledger = SuiteLedger(root)
            if verb == "halt":
                for target in sorted(targets):
                    halt_row(root, target, "operator", plan["suite_hash"])
            else:
                for target in sorted(targets):
                    ledger.append(verb, row=target, reason="operator", suite_hash=plan["suite_hash"])
            if verb == "resume":
                for (row_id, cycle), state in ledger.state()["jobs"].items():
                    if row_id in targets and (state.get("phase") == "halted" or
                            state.get("phase") == "skipped" and state.get("reason") in {"dependency_halted", "dependency_skipped"}):
                        ledger.append("phase", row=row_id, cycle=cycle, suite_hash=state["suite_hash"],
                                      phase="planned", attempt=state.get("attempt", 0) + 1, retries=0,
                                      checkpoint_tried=False, degrade_index=-1)
            print(json.dumps({"row": args.row, "action": verb, "targets": sorted(targets)}))
            return 0
        if verb == "apply":
            print(json.dumps(apply_plan(root, load_plan(args.file, args.bindings)), indent=2))
            return 0
        if verb == "rehearse":
            from .rehearse import rehearse
            result = rehearse(load_plan(args.file, args.bindings), root, start=args.start, days=args.days, cycles_limit=args.cycles)
            print(json.dumps(result, indent=2))
            return 0 if result["accepted"] else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, ImportError) as error:
        print(f"gpuwm suite: {error}", file=sys.stderr)
        return 2


#: One line per verb for `gpuwm suite --help`; without them the listing named
#: only `init`, so the other nine doors were invisible to a reader of --help.
SUITE_VERB_HELP = {
    "run": "start or continue the suite controller (needs --bindings for real launches)",
    "status": "aggregate and per-row state; exit 0 green, 1 degraded, 2 halted, 75 waiting on a source",
    "events": "the journal's events, optionally followed live",
    "halt": "stop new launches of one row and signal its active attempt",
    "resume": "release a halted row and start its attempt with a new launch token",
    "apply": "install a changed suite file at a quiescent cycle boundary",
    "rehearse": "replay past cycles end to end and score them",
    "validate": "check a suite file against the schema and the shipped tables",
    "plan": "price the suite against its boxes and admit or refuse it (exit 2 names the overrun)",
}


def register_runtime(sub):
    for verb in ("run", "status", "events", "halt", "resume", "apply", "rehearse"):
        parser = sub.add_parser(verb, help=SUITE_VERB_HELP[verb])
        parser.add_argument("--state-dir", default=".suite-state")
        parser.set_defaults(func=runtime_main, suite_verb=verb)
        if verb in {"run", "apply", "rehearse"}:
            parser.add_argument("file", nargs="?" if verb == "run" else None)
            parser.add_argument("--bindings", help="prepared/deployment bindings tied to the resolved suite hash")
        if verb == "run":
            parser.add_argument("--resume", action="store_true")
            parser.add_argument("--detach", action="store_true")
            parser.add_argument("--once", action="store_true")
            parser.add_argument("--poll-seconds", type=float, default=1)
        if verb == "status":
            parser.add_argument("--json", action="store_true")
        if verb == "events":
            parser.add_argument("--follow", action="store_true")
            parser.add_argument("--after", type=int, default=0)
        if verb in {"halt", "resume"}:
            parser.add_argument("row")
        if verb == "rehearse":
            parser.add_argument("--from", dest="start", required=True)
            parser.add_argument("--days", type=float, required=True)
            parser.add_argument("--cycles", type=int, default=None, help="limit replay to first N cycles")


def register_cli(subparsers):
    parser = subparsers.add_parser("suite", help="plan and operate an unattended weather suite")
    sub = parser.add_subparsers(dest="suite_command", required=True)
    register_runtime(sub)
    register_spec(sub)
    # gpuwm/cli.py's one layering convention: every command takes --explain,
    # so the refusal pointer "... --explain for the reason" is true here too.
    # `gpuwm suite` dispatches before gpuwm/cli.py's sweep adds the flag, so
    # it registers its own: on `suite`, and on each verb typed after it.
    from gpuwm.explain import add_explain_flag
    add_explain_flag(parser)
    for verb in sub.choices.values():
        add_explain_flag(verb, nested=True)


"""CPU-only suite commands. Parsing does not import numerical engines."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import tomllib
from .hardware import box_from_check
from .planner import plan_suite
from .registry import Registries, SuiteRefusal
from .validator import load_suite, template_path, validate_suite

def register_spec(verbs):
    init=verbs.add_parser("init",help="write a table-selected country suite")
    init.add_argument("--region",required=True)
    init.add_argument("--boxes",help="comma-separated gpuwm check --json receipts")
    init.add_argument("-o","--output",required=True)
    init.add_argument("--registry-dir",help=argparse.SUPPRESS)
    for verb in ("validate","plan"):
        p=verbs.add_parser(verb,help=SUITE_VERB_HELP[verb])
        p.add_argument("file")
        p.add_argument("--json",action="store_true")
        if verb=="plan":
            p.add_argument("--what-if",action="append",default=[])
            p.add_argument("--emit-commands",action="store_true")

def literal(value):
    if isinstance(value,bool): return str(value).lower()
    if isinstance(value,(int,float)): return str(value)
    if isinstance(value,str): return json.dumps(value)
    if isinstance(value,list): return "["+", ".join(literal(v) for v in value)+"]"
    if isinstance(value,dict): return "{ "+", ".join(json.dumps(k)+" = "+literal(v) for k,v in value.items())+" }"
    raise SuiteRefusal(f"cannot serialize suite value {value!r}")

def write_suite(path,suite):
    lines=["# Generated planning file. Validate inventory and prepared bindings before execution."]
    lines.extend(json.dumps(k)+" = "+literal(v) for k,v in suite.items() if k not in ("system","box","override","extends"))
    for key in ("box","system","override"):
        for row in suite.get(key,[]):
            lines.append("\n[["+key+"]]")
            lines.extend(json.dumps(k)+" = "+literal(v) for k,v in row.items())
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")

def initialize(region,boxes=None,*,registry_dir=None):
    reg=Registries(registry_dir); geography=reg.region(region)
    hardware=[box_from_check(path,chr(97+i)) for i,path in enumerate(boxes.split(","))] if boxes else []
    if len(hardware)>2:
        raise SuiteRefusal("at most two boxes; prevents an unqualified cross-host forecast")
    count=sum(b["cards"] for b in hardware) if hardware else 8
    # Template names come from the region row; country ids never select code.
    profiles=geography["profiles"]
    ordered=list(profiles)
    profile=ordered[1] if count>=16 and len(hardware)==2 else ordered[0] if count>=8 else ordered[-1]
    template=profiles[profile]
    suite=load_suite(template_path("template:"+template,reg),registry_dir=registry_dir)
    suite["name"]=region.lower()+"-"+template
    suite["region"]=region
    if hardware:
        old=[b["id"] for b in suite["box"]]
        mapping={key:hardware[min(i,len(hardware)-1)]["id"] for i,key in enumerate(old)}
        suite["box"]=hardware
        for row in suite["system"]: row["box"]=mapping[row["box"]]
    for row in suite["system"]:
        if row.get("domain",{}).get("from")=="region":
            row["mesh"]=geography["mesh"]
            row["ic"]=geography["sources"]["regional_ic"]
    if registry_dir:
        suite.setdefault("policy",{})["registry_dir"]=str(Path(registry_dir).resolve())
    return validate_suite(suite,registries=reg)

def apply_what_if(suite,expression,*,base=None):
    try:
        left,right=expression.split("=",1); identifier,key=left.split(".",1)
        value=tomllib.loads("value = "+right)["value"]
        row=next(r for r in suite["system"] if r["id"]==identifier)
        cursor=row; parts=key.split(".")
        for part in parts[:-1]: cursor=cursor[part]
        if parts[-1] not in cursor: raise KeyError(parts[-1])
        cursor[parts[-1]]=value
    except (ValueError,KeyError,TypeError,StopIteration,tomllib.TOMLDecodeError) as error:
        raise SuiteRefusal(f"unknown or invalid what-if {expression!r}; prevents silently ignored edits") from error
    return validate_suite(suite,base=base)

def spec_main(args):
    try:
        if args.suite_command=="init":
            suite=initialize(args.region,args.boxes,registry_dir=args.registry_dir)
            write_suite(args.output,suite)
            print(f"Wrote {args.output}. Planning targets require live check receipts before launch.")
            return 0
        suite=load_suite(args.file)
        if args.suite_command=="validate":
            result=dict(valid=True,schema=suite["schema"],region=suite["region"],systems=len(suite["system"]))
            print(json.dumps(result) if args.json else f"Valid: {args.file} ({len(suite['system'])} rows)")
            return 0
        baseline=plan_suite(suite,base=Path(args.file).resolve().parent) if args.what_if else None
        for expression in args.what_if: suite=apply_what_if(suite,expression,base=Path(args.file).resolve().parent)
        plan=plan_suite(suite,base=Path(args.file).resolve().parent)
        if baseline:
            plan["what_if"]=dict(expressions=args.what_if,baseline_daily_card_hours=baseline["daily_card_hours"],
                                 delta_daily_card_hours=plan["daily_card_hours"]-baseline["daily_card_hours"])
        if args.json:
            print(json.dumps(plan,sort_keys=True,allow_nan=False))
        else:
            print("ADMITTED" if plan["admitted"] else "REFUSED")
            for box in plan["boxes"]:
                print(f"Box {box['id']}: {box['daily_card_hours']:.3f} card-hours/day")
                for block in box["blocks"]:
                    print(f"  {block['cycle_hour']:02d}: {block['card_hours']:.3f}/{block['budget_card_hours']:.3f} card-hours")
                    for card,intervals in block["lanes"].items():
                        print(f"    card {card}: "+", ".join(f"{a/60:.1f}-{b/60:.1f}m" for a,b in intervals))
            for row in plan["rows"]: print(f"{row['id']}: {row['daily_card_hours']:.3f} card-hours/day")
            for refusal in plan["refusals"]: print(refusal["message"])
            for slot in plan["measurement_slots"]: print(f"{slot['row']}: measurement only ({slot['estimated_seconds']:.1f}s)")
            if args.emit_commands:
                for job in plan["jobs"]:
                    if job["block_hour"]==0:
                        print(job["id"]+": "+shlex.join(job["argv"] or ["{da.shadow.command}"]))
        return plan["exit_code"]
    except (SuiteRefusal,OSError) as error:
        if getattr(args,"json",False):
            print(json.dumps(dict(schema="gpuwm.suite.plan.v1",suite=None,region=None,suite_sha256=None,admitted=False,exit_code=2,
                block_seconds=21600,cycle_hours=[0,6,12,18],boxes=[],rows=[],jobs=[],measurement_slots=[],daily_card_hours=0,what_if=None,
                refusals=[dict(row="*",reason=str(error),rate=None,budget=None,required=None,overrun=None,unit="validation",message=str(error))])))
        else: print("REFUSED: "+str(error))
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpuwm")
    register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["suite", *(argv or [])])
    from gpuwm.explain import explain_enabled, set_explain
    set_explain(explain_enabled(args))   # inside gpuwm.cli.main's explain_scope
    return args.func(args) if hasattr(args, "func") else spec_main(args)

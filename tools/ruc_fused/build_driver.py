"""Write ``gpuwm/core/kernels/ruc_fused_driver.cuh``.

The fused RUC driver's CUDA is ``tools/ruc_fused/driver_body.cuh``, written
by hand; its pointer aliases, admission checks and commit list are generated
here from the name inventories in ``gpuwm/core/ruc_fused.py`` and
``gpuwm/core/ruc.py``, so the kernel and its launcher cannot disagree on a
slot.  CPU-only, stdlib-only: ``python tools/ruc_fused/build_driver.py``.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
source = ast.parse((root / "gpuwm/core/ruc.py").read_text())
constants = {}
classes = {}
for node in source.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        try:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    if isinstance(node, ast.ClassDef):
        classes[node.name] = SimpleNamespace(__dataclass_fields__={
            item.target.id: None for item in node.body if isinstance(item, ast.AnnAssign)})
environment = {"ruc": SimpleNamespace(**constants, **classes)}
module = ast.parse((root / "gpuwm/core/ruc_fused.py").read_text())
for node in module.body:
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if isinstance(target, ast.Name) and target.id in (
                "_COLUMNS", "_EXTRAS", "_LOCALS", "_PROFILES", "_WORK_PROFILES",
                "_SCRATCH_NAMES", "_INPUT_NAMES", "_SF_OUTPUTS", "_OUTPUT_NAMES"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "abi", "exec"), environment)

prefix = "// Generated pointer aliases.  Regenerate with tools/ruc_fused/build_driver.py.\n"
prefix += "#define D_F(name) d_##name[i]\n#define D_P(name,k) d_##name[(k)*n+i]\n"
declarations = ["float* d_storage = (float*)sp[0];"]
scalar_rows = profile_rows = 0
for name in environment["_SCRATCH_NAMES"]:
    declarations.append(f"float* d_{name} = d_storage+({scalar_rows}+{profile_rows}*RUC_NZS)*n;")
    if name in environment["_PROFILES"] + environment["_WORK_PROFILES"]:
        profile_rows += 1
    else:
        scalar_rows += 1
prefix += "#define D_DECLARE_SCRATCH " + (" " + chr(92) + chr(10)).join(declarations) + chr(10)
prefix += "#define D_DECLARE_OUTPUT " + " \\\n+".join(
    f"const float* o_{name} = (const float*)op[{index}];" for index, name in enumerate(environment["_SF_OUTPUTS"])) + "\n"
prefix += "#define D_COPY_INPUT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) D_P({name},k)=((const float*)ip[{index}])[k*n+i];"
     if name in environment["_PROFILES"] else
     f"D_F({name})=((const float*)ip[{index}])[i];")
    for index, name in enumerate(environment["_INPUT_NAMES"])) + "\n"
prefix += "#define D_ADMIT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P({name},k))) {{ d_flag(flags,{index}); admitted=false; }}"
     if name in environment["_PROFILES"] else
     f"if (!isfinite(D_F({name}))) {{ d_flag(flags,{index}); admitted=false; }}")
    for index, name in enumerate(environment["_PROFILES"] + environment["_COLUMNS"])) + "\n"
prefix += f"#define D_CAT_INPUT {len(environment['_INPUT_NAMES'])}\n"
prefix += f"#define D_CAT_FLAG {len(environment['_PROFILES'] + environment['_COLUMNS'])}\n"
prefix += "#define D_CHECK_OUTPUT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P({name},k))) d_flag(flags,{1056+index});"
     if name in environment["_PROFILES"] else
     f"if (!isfinite(D_F({name}))) d_flag(flags,{1056+index});")
    for index, name in enumerate(environment["_OUTPUT_NAMES"])) + "\n"
targets = constants["RUC_DRIVER_COLUMN_STATE"] + ("albbck", "chs", "flhc", "flqc") + environment["_EXTRAS"] + environment["_PROFILES"] + (
    "infiltr", "smelt", "runoff1", "runoff2", "t2", "th2", "q2")
prefix += "#define D_COMMIT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) ((float*)cp[{index}])[k*n+i]=D_P({name},k);"
     if name in environment["_PROFILES"] else
     f"((float*)cp[{index}])[i]=D_F({name});")
    for index, name in enumerate(targets)) + "\n"
body = (root / "tools/ruc_fused/driver_body.cuh").read_text()
(root / "gpuwm/core/kernels/ruc_fused_driver.cuh").write_text(prefix.replace("\n+", "\n") + body, newline="\n")

"""Write ``gpuwm/core/kernels/ruc_fused_driver.cuh``.

The fused RUC driver's CUDA is ``tools/ruc_fused/driver_body.cuh``, written
by hand; its pointer aliases, admission checks and commit list are generated
here from the name inventories in ``gpuwm/core/ruc_memory.py`` and
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
module = ast.parse((root / "gpuwm/core/ruc_memory.py").read_text())
for node in module.body:
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if isinstance(target, ast.Name) and target.id in (
                "DRIVER_COLUMNS", "DRIVER_EXTRAS", "DRIVER_LOCALS", "DRIVER_PROFILES", "DRIVER_WORK_PROFILES",
                "DRIVER_SCRATCH_NAMES", "DRIVER_INPUT_NAMES", "SFCTMP_OUTPUT_NAMES", "DRIVER_OUTPUT_NAMES",
                "DRIVER_TARGET_NAMES"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "abi", "exec"), environment)

prefix = "// Generated pointer aliases.  Regenerate with tools/ruc_fused/build_driver.py.\n"
prefix += "#define D_F(name) d_##name[i]\n#define D_P(name,k) d_##name[(k)*n+i]\n"
declarations = ["float* d_storage = (float*)sp[0];"]
scalar_rows = profile_rows = 0
for name in environment["DRIVER_SCRATCH_NAMES"]:
    declarations.append(f"float* d_{name} = d_storage+({scalar_rows}+{profile_rows}*RUC_NZS)*n;")
    if name in environment["DRIVER_PROFILES"] + environment["DRIVER_WORK_PROFILES"]:
        profile_rows += 1
    else:
        scalar_rows += 1
prefix += "#define D_DECLARE_SCRATCH " + (" " + chr(92) + chr(10)).join(declarations) + chr(10)
prefix += "#define D_DECLARE_OUTPUT " + " \\\n+".join(
    f"const float* o_{name} = (const float*)op[{index}];" for index, name in enumerate(environment["SFCTMP_OUTPUT_NAMES"])) + "\n"
prefix += "#define D_COPY_INPUT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) D_P({name},k)=((const float*)ip[{index}])[k*n+i];"
     if name in environment["DRIVER_PROFILES"] else
     f"D_F({name})=((const float*)ip[{index}])[i];")
    for index, name in enumerate(environment["DRIVER_INPUT_NAMES"])) + "\n"
prefix += "#define D_ADMIT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P({name},k))) {{ d_flag(flags,{index}); admitted=false; }}"
     if name in environment["DRIVER_PROFILES"] else
     f"if (!isfinite(D_F({name}))) {{ d_flag(flags,{index}); admitted=false; }}")
    for index, name in enumerate(environment["DRIVER_PROFILES"] + environment["DRIVER_COLUMNS"])) + "\n"
prefix += f"#define D_CAT_INPUT {len(environment['DRIVER_INPUT_NAMES'])}\n"
prefix += f"#define D_CAT_FLAG {len(environment['DRIVER_PROFILES'] + environment['DRIVER_COLUMNS'])}\n"
prefix += "#define D_CHECK_OUTPUT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) if (!isfinite(D_P({name},k))) d_flag(flags,{1056+index});"
     if name in environment["DRIVER_PROFILES"] else
     f"if (!isfinite(D_F({name}))) d_flag(flags,{1056+index});")
    for index, name in enumerate(environment["DRIVER_OUTPUT_NAMES"])) + "\n"
targets = environment["DRIVER_TARGET_NAMES"]
prefix += "#define D_COMMIT " + " \\\n+".join(
    (f"for (int k=0;k<RUC_NZS;++k) ((float*)cp[{index}])[k*n+i]=D_P({name},k);"
     if name in environment["DRIVER_PROFILES"] else
     f"((float*)cp[{index}])[i]=D_F({name});")
    for index, name in enumerate(targets)) + "\n"
body = (root / "tools/ruc_fused/driver_body.cuh").read_text()
(root / "gpuwm/core/kernels/ruc_fused_driver.cuh").write_text(prefix.replace("\n+", "\n") + body, newline="\n")

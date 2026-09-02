"""COMPILE-ONLY frame probe.  Nothing is launched, so nothing is reserved."""
import sys, json
import cupy as cp

MIB = 1024 ** 2
props = cp.cuda.runtime.getDeviceProperties(0)
SM   = int(props["multiProcessorCount"])
TPSM = int(props["maxThreadsPerMultiProcessor"])
CAP  = SM * TPSM
name = props["name"]; name = name.decode() if isinstance(name, bytes) else name
print(f"device: {name}   {SM} SMs x {TPSM} = {CAP:,} resident threads")
print(f"reservation law: (frame - 1024) x {CAP:,}  ->  1 B of frame = "
      f"{CAP/1024:.0f} KiB\n")

rows = []
for path, sym, kps in (("nt_probe_stack.cu", "nt_skeleton_stack", (49, 62)),
                       ("nt_probe_ws.cu",    "nt_skeleton_ws",    (49, 62))):
    src = open(path).read()
    for kp in kps:
        code = f"#define NT_KP {kp}\n" + src
        mod = cp.RawModule(code=code, options=("-std=c++17",))
        mod.compile()
        fn = mod.get_function(sym)
        a = fn.attributes
        frame = int(a["local_size_bytes"])
        regs  = int(a["num_regs"])
        resv  = max(0, frame - 1024) * CAP / MIB
        rows.append((sym, kp, frame, regs, resv,
                     int(a.get("shared_size_bytes", 0)),
                     int(a.get("const_size_bytes", 0))))

print(f"{'kernel':<20} {'NT_KP':>6} {'frame B':>10} {'regs':>6} "
      f"{'shared':>7} {'reserves':>12}")
for sym, kp, frame, regs, resv, sh, cn in rows:
    print(f"{sym:<20} {kp:>6} {frame:>10,} {regs:>6} {sh:>7} "
          f"{resv:>9,.1f} MiB")

print()
stack49 = [r for r in rows if r[0].endswith("stack") and r[1] == 49][0]
ws49    = [r for r in rows if r[0].endswith("_ws")   and r[1] == 49][0]
print(f"stack -> workspace at nz=49 : {stack49[2]:,} B -> {ws49[2]:,} B")
print(f"reservation                 : {stack49[4]:,.1f} MiB -> {ws49[4]:,.1f} MiB")
print(f"under the 1,024 B default stack? {'YES' if ws49[2] < 1024 else 'NO'}")

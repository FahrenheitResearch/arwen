# Startup profiling aid: when GPUWM_PISTARTUP_SAMPLE names a folder, a daemon thread
# writes every thread stack (gpuwm/tilestream/cupy frames, last 6) every 2 s, one file per pid.
# Used because py-spy needs SYS_PTRACE, which rented containers do not grant.
import os
_out = os.environ.get("GPUWM_PISTARTUP_SAMPLE")
if _out:
    import sys, threading, time, traceback
    def _sample():
        path = os.path.join(_out, f"stacks-{os.getpid()}.txt")
        t0 = time.time()
        with open(path, "a", buffering=1) as handle:
            while True:
                time.sleep(2.0)
                frames = sys._current_frames()
                names = {t.ident: t.name for t in threading.enumerate()}
                handle.write(f"@ {time.time() - t0:.1f} {time.strftime('%H:%M:%S')}\n")
                for ident, frame in frames.items():
                    if ident == threading.get_ident():
                        continue
                    stack = traceback.extract_stack(frame)
                    mine = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in stack
                            if ("gpuwm" in f.filename or "tilestream" in f.filename or "cupy" in f.filename or "netCDF" in f.filename or "h5" in f.filename)]
                    if not mine:
                        mine = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in stack[-3:]]
                    handle.write(f"  [{names.get(ident, ident)}] " + " > ".join(mine[-6:]) + "\n")
    threading.Thread(target=_sample, name="pistartup-sampler", daemon=True).start()

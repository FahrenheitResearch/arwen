"""Fresh-node provisioning probe: CUDA major, cgroup memory, NUMA topology.

Answers the three questions a bootstrap must DETECT rather than hardcode:

  1. Which CuPy wheel does this box need?  Read the CUDA major off the
     DRIVER (cuDriverGetVersion via ctypes -- works with no CuPy installed
     and opens no device), never off `nvcc`: the toolkit major and the
     driver major are independent and the wheel follows the driver.

  2. How much host RAM may this process actually use?  /proc/meminfo
     reports the HOST inside a container (measured: 1007 GiB reported for
     a 128 GB entitlement).  Walk the process's OWN cgroup path upward
     and take the minimum limit found.  Reading the literal path
     /sys/fs/cgroup/memory.max is WRONG on a bare host -- the cgroup-v2
     root has no memory.max at all (verified on this dev box).

  3. Which cores are local to each GPU?  Per-device sysfs, never a
     hardcoded core list.  Two measured boxes disagree in opposite
     directions: the 8x4090 has two NUMA nodes with every GPU on node 0
     (binding to node 1 is the bug), the single-4090 box is dual-socket
     with the GPU on node 1 (cores 64-127 are the only correct ones).

Every probe fails soft: an unanswerable question returns None with a
reason, and the caller decides.  A bootstrap that guesses here produces a
node that looks provisioned and dies on the first matmul or the first
radiation step.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
from pathlib import Path

GIB = 1 << 30


# --------------------------------------------------------------- CUDA major

def driver_cuda_major() -> tuple[int | None, str]:
    """(major, how) read straight off the driver library.

    cuDriverGetVersion is the one entry point that answers without
    cuInit: no context, no device opened, nothing that could disturb a
    card another process is using.
    """
    names = ("nvcuda.dll",) if sys.platform == "win32" else (
        "libcuda.so.1", "libcuda.so")
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            ver = ctypes.c_int(0)
            if lib.cuDriverGetVersion(ctypes.byref(ver)) != 0:
                continue
        except (OSError, AttributeError, ValueError):
            continue
        if ver.value > 0:
            return ver.value // 1000, f"{name} cuDriverGetVersion={ver.value}"
    return None, "no NVIDIA driver library could be loaded"


def cupy_wheel_for(major: int | None) -> str | None:
    return {12: "cupy-cuda12x", 13: "cupy-cuda13x"}.get(major or 0)


# ------------------------------------------------------------ cgroup memory

def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text in ("max", ""):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def cgroup_memory_limit() -> tuple[int | None, str]:
    """Smallest memory limit binding on THIS process, in bytes.

    cgroup v2: read /proc/self/cgroup for the relative path, then check
    memory.max at every level from that path up to the mount root, taking
    the minimum -- a limit set on an ancestor binds us just as hard as one
    set on our own leaf.  cgroup v1: memory.limit_in_bytes, where an
    "unlimited" cgroup reports a huge sentinel (~2^63-1 rounded to page
    size) rather than the string "max", so anything at or above the
    sentinel threshold is treated as absent.
    """
    root = Path("/sys/fs/cgroup")
    rel = ""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[1] == "":          # v2 line: "0::/path"
                rel = parts[2].strip().lstrip("/")
                break
    except OSError:
        pass

    limits: list[tuple[int, str]] = []
    node = root / rel if rel else root
    seen = 0
    while True:
        value = _read_int(node / "memory.max")
        if value is not None:
            limits.append((value, str(node / "memory.max")))
        if node == root or seen > 64:
            break
        node = node.parent
        seen += 1

    if limits:
        value, where = min(limits)
        return value, f"cgroup v2 {where}"

    # cgroup v1
    SENTINEL = 1 << 62
    for candidate in (root / "memory" / "memory.limit_in_bytes",
                      root / "memory.limit_in_bytes"):
        value = _read_int(candidate)
        if value is not None and value < SENTINEL:
            return value, f"cgroup v1 {candidate}"

    return None, ("no cgroup memory limit binds this process "
                  "(bare host, or the limit is on an ancestor outside the "
                  "namespace); /proc/meminfo is the only remaining source "
                  "and must be labelled untrusted in a container")


def meminfo_total() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------- NUMA topology

def _parse_cpulist(text: str) -> list[int]:
    out: list[int] = []
    for chunk in text.strip().split(","):
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


def numa_nodes() -> dict[int, list[int]]:
    """node id -> cpu list, from /sys/devices/system/node."""
    nodes: dict[int, list[int]] = {}
    base = Path("/sys/devices/system/node")
    if not base.is_dir():
        return nodes
    for entry in sorted(base.glob("node[0-9]*")):
        match = re.match(r"node(\d+)$", entry.name)
        if not match:
            continue
        try:
            nodes[int(match.group(1))] = _parse_cpulist(
                (entry / "cpulist").read_text())
        except OSError:
            continue
    return nodes


def gpu_pci_addresses() -> list[tuple[int, str]]:
    """[(device index, 'DDDD:BB:DD.F')] from cudaDeviceProp, via CuPy.

    cudaDeviceProp carries pciDomainID/pciBusID/pciDeviceID as separate
    integers; the function is always 0 for a GPU.  Falls back to
    nvidia-smi if CuPy is unavailable.
    """
    out: list[tuple[int, str]] = []
    try:
        import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        for i in range(count):
            p = cp.cuda.runtime.getDeviceProperties(i)
            out.append((i, "%04x:%02x:%02x.0" % (
                p["pciDomainID"], p["pciBusID"], p["pciDeviceID"])))
        return out
    except Exception:
        pass
    try:
        import subprocess
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout
        for line in text.strip().splitlines():
            idx, bdf = (part.strip() for part in line.split(",", 1))
            out.append((int(idx), bdf.lower()))
    except Exception:
        pass
    return out


def gpu_affinity() -> list[dict]:
    """Per GPU: its NUMA node and its local cpu list, straight from sysfs.

    /sys/bus/pci/devices/<BDF>/numa_node is the authority.  -1 means the
    firmware did not report an affinity, which is the ordinary answer on a
    single-socket box and is NOT node 0 -- treat it as "no binding
    required".  Under WSL2 the BDF from cudaDeviceProp does not exist in
    sysfs at all (measured: CuPy says 0000:01:00.0, sysfs holds
    4fbe:00:00.0), so absence must fail soft.
    """
    nodes = numa_nodes()
    all_cpus = sorted(os.sched_getaffinity(0))
    rows = []
    for index, bdf in gpu_pci_addresses():
        dev = Path("/sys/bus/pci/devices") / bdf
        node = _read_int(dev / "numa_node")
        local = None
        try:
            local = _parse_cpulist((dev / "local_cpulist").read_text())
        except OSError:
            local = None
        if local is None and node is not None and node >= 0:
            local = nodes.get(node)
        rows.append({
            "device": index,
            "pci": bdf,
            "sysfs_present": dev.is_dir(),
            "numa_node": node,
            "local_cpus": local,
            "binding_required": bool(len(nodes) > 1 and node is not None
                                     and node >= 0 and local),
            "usable_cpus": local if local else all_cpus,
        })
    return rows


def core_plan(rows: list[dict], per_rank: int | None = None) -> dict[int, list[int]]:
    """rank -> cores, derived from measured affinity, never hardcoded.

    Ranks are grouped by the node their card is actually on, and each
    rank gets a disjoint contiguous slice of that node's cpus.  With one
    node (or no reported affinity) every rank slices the process's own
    allowed cpu set, which is itself already narrowed by any cpuset
    cgroup -- so a container that was given 16 of 128 cores does not hand
    out cores it may not run on.
    """
    by_node: dict[int | None, list[int]] = {}
    for row in rows:
        key = row["numa_node"] if row["binding_required"] else None
        by_node.setdefault(key, []).append(row["device"])
    plan: dict[int, list[int]] = {}
    for key, devices in by_node.items():
        pool = None
        for row in rows:
            if row["device"] == devices[0]:
                pool = sorted(row["usable_cpus"])
                break
        pool = pool or sorted(os.sched_getaffinity(0))
        share = per_rank or max(1, len(pool) // max(1, len(devices)))
        stride = max(1, len(pool) // max(1, len(devices)))
        for slot, device in enumerate(sorted(devices)):
            start = slot * stride
            plan[device] = pool[start:start + share] or pool[:1]
    return plan


def main() -> int:
    major, how = driver_cuda_major()
    mem, mem_how = cgroup_memory_limit()
    rows = gpu_affinity()
    nodes = numa_nodes()
    host = meminfo_total()

    report = {
        "cuda_driver_major": major,
        "cuda_driver_read_from": how,
        "cupy_wheel": cupy_wheel_for(major),
        "cgroup_memory_limit_bytes": mem,
        "cgroup_memory_read_from": mem_how,
        "proc_meminfo_memtotal_bytes": host,
        "proc_meminfo_trustworthy": mem is None,
        "numa_nodes": {str(k): v for k, v in nodes.items()},
        "gpus": rows,
        "rank_core_plan": {str(k): v for k, v in core_plan(rows).items()},
    }
    print(json.dumps(report, indent=2, default=str))

    print("\n--- human ---", file=sys.stderr)
    print(f"CUDA driver major : {major}  ({how})", file=sys.stderr)
    print(f"CuPy wheel        : {cupy_wheel_for(major)}", file=sys.stderr)
    if mem is not None:
        print(f"host RAM budget   : {mem / GIB:.1f} GiB  ({mem_how})",
              file=sys.stderr)
    else:
        print(f"host RAM budget   : UNBOUNDED by cgroup; {mem_how}",
              file=sys.stderr)
        if host:
            print(f"                    /proc/meminfo says "
                  f"{host / GIB:.1f} GiB", file=sys.stderr)
    print(f"NUMA nodes        : {len(nodes)}  "
          f"{ {k: f'{len(v)} cpus' for k, v in nodes.items()} }",
          file=sys.stderr)
    for row in rows:
        node = row["numa_node"]
        note = ("bind" if row["binding_required"]
                else "no binding required" if len(nodes) <= 1
                else "affinity not reported (-1): do NOT assume node 0")
        cpus = row["local_cpus"]
        span = (f"{cpus[0]}-{cpus[-1]} ({len(cpus)})" if cpus else "-")
        print(f"  gpu{row['device']} {row['pci']} numa_node={node} "
              f"local_cpus={span}  [{note}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

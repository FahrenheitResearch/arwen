#!/usr/bin/env python3
"""Passive process-tree/GPU sampling, under the node's exclusive GPU mutex."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def tree(pid):
    pending, cpu, rss, threads = [pid], 0., 0, 0
    while pending:
        current = pending.pop()
        try:
            fields = Path(f'/proc/{current}/stat').read_text().rsplit(')', 1)[1].split()
            cpu += (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
            rss += int(fields[21])*os.sysconf('SC_PAGE_SIZE')
            threads += int(fields[17])
            # A queue worker creates Popen from a non-main Python thread;
            # Linux attributes those children to that task, not task/pid.
            for task in Path(f'/proc/{current}/task').iterdir():
                try:
                    pending += [int(p) for p in (task/'children').read_text().split()]
                except OSError:
                    pass
        except (OSError, ValueError, IndexError):
            pass
    return cpu, rss, threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mutex', type=Path, required=True)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--timeout-seconds', type=float, default=14400)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    def ledger(message):
        if args.ledger:
            with args.ledger.open('a') as stream:
                stream.write(f'trailer-ian {utc()} {message} pid {os.getpid()}\n')
    with args.mutex.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger('start')
        started, start_utc = time.perf_counter(), utc()
        peak_rss, peak_gpu, suspected = 0, 0, None
        alerts = []
        try:
            with (args.out/'command.log').open('w') as log, (args.out/'samples.csv').open('w') as samples:
                writer = csv.writer(samples)
                writer.writerow(['utc', 'wall_seconds', 'tree_cpu_seconds', 'rss_bytes', 'threads', 'gpu_util_pct', 'gpu_memory_mib'])
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                previous = None
                while True:
                    now = time.perf_counter()
                    cpu, rss, threads = tree(process.pid)
                    query = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                    cards = [tuple(map(int, row.split(','))) for row in query.stdout.strip().splitlines()]
                    util, memory = max(c[0] for c in cards), sum(c[1] for c in cards)
                    peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, memory)
                    writer.writerow([utc(), now-started, cpu, rss, threads, util, memory])
                    samples.flush()
                    if previous is not None:
                        cores = (cpu-previous[1])/(now-previous[0])
                        if .7 <= cores <= 1.3 and util < 10:
                            suspected = suspected or now
                            if now-suspected >= 30:
                                alert = {'utc': utc(), 'wall_seconds': now-started, 'cpu_cores': cores,
                                    'gpu_util_pct': util, 'message': 'possible single-core stage; inspect command.log and process threads'}
                                alerts.append(alert)
                                with (args.out/'SINGLE-CORE-ALERTS.jsonl').open('a') as stream:
                                    stream.write(json.dumps(alert)+'\n')
                                suspected = now
                        else:
                            suspected = None
                    previous = now, cpu
                    if process.poll() is not None:
                        break
                    if now-started > args.timeout_seconds:
                        # A launcher owns card workers and native children.
                        # Retire this command's process group before releasing
                        # the host mutex, including after its leader exits.
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        break
                    time.sleep(2)
            result = {'start_utc': start_utc, 'end_utc': utc(), 'wall_seconds': time.perf_counter()-started,
                'exit_code': process.returncode, 'peak_process_tree_rss_bytes': peak_rss,
                'peak_gpu_memory_mib': peak_gpu, 'single_core_alerts': alerts,
                'measurement': 'passive 2-second /proc and nvidia-smi sampling; no deep profiler or added CUDA synchronization'}
            (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
        finally:
            ledger('release')
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()

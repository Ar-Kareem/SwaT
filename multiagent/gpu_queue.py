#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


LOG_ROOT = Path(__file__).resolve().parent / "gpu_queue_logs"
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
RUN_DIR_RE = re.compile(r"^(\d+)_")


def run_name(value: str) -> str:
    if value in {".", ".."} or not RUN_NAME_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "--name must be one filename component using letters, digits, '.', '_', '+', or '-'; paths are forbidden"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commands", default="gpu_queue_commands/gpu_queue_commands.json", help="JSON file with the command queue.")
    parser.add_argument("--gpus", help="Comma-separated physical GPU indices the queue may use. Default: all visible GPUs.")
    parser.add_argument("--free-seconds", type=float, default=10.0, help="How long a GPU must look idle before it counts as free.")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="nvidia-smi polling interval.")
    parser.add_argument("--util-threshold", type=int, default=0, help="GPU utilization (%%) above this counts as busy.")
    parser.add_argument("--grace-mib", type=int, default=6000, help="Other processes may hold up to this much GPU memory (MiB) combined and the GPU still counts as free.")
    parser.add_argument("--mem-threshold-mib", type=int, default=500, help="Total GPU memory (MiB) above max(this, --grace-mib) counts as busy (catches non-compute allocations).")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Do not launch any later jobs after one job exits nonzero; already-running jobs are allowed to finish.",
    )
    parser.add_argument(
        "--name",
        type=run_name,
        default="jobs",
        help="Y in gpu_queue_logs/X_Y (default: jobs); must be one filename component.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the parsed queue and exit without running anything.")
    return parser.parse_args()


@dataclass
class Job:
    index: int
    cmd: str
    gpus_needed: int
    assigned: list[int] = field(default_factory=list)
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    exit_code: int | None = None
    started: float | None = None
    name: str | None = None
    after: list[str] = field(default_factory=list)


class Tee:

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.log_file.write(text)
        try:
            self.terminal.write(text)
        except OSError:
            pass
        return len(text)

    def flush(self) -> None:
        self.log_file.flush()
        try:
            self.terminal.flush()
        except OSError:
            pass

    def __getattr__(self, name: str):
        return getattr(self.terminal, name)


@contextlib.contextmanager
def mirror_queue_output(path: Path):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with path.open("w", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def load_jobs(path: Path) -> list[Job]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON list of commands")
    jobs = []
    for raw in entries:
        if isinstance(raw, str):
            if raw.lstrip().startswith("#"):
                continue
            cmd, gpus_needed = raw, 1
        elif isinstance(raw, dict):
            cmd, gpus_needed = raw["cmd"], int(raw.get("gpus", 1))
            if isinstance(cmd, list):
                if not cmd or not all(isinstance(part, str) for part in cmd):
                    raise ValueError(f"cmd list must be non-empty strings: {raw!r}")
                cmd = " ".join(part.strip() for part in cmd)
        else:
            raise ValueError(f"Unsupported queue entry: {raw!r}")
        if gpus_needed < 1:
            raise ValueError(f"gpus must be >= 1: {raw!r}")
        name = raw.get("name") if isinstance(raw, dict) else None
        after = raw.get("after", []) if isinstance(raw, dict) else []
        if isinstance(after, str):
            after = [after]
        jobs.append(Job(index=len(jobs), cmd=cmd, gpus_needed=gpus_needed, name=name, after=list(after)))
    names = [job.name for job in jobs if job.name]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate job names: {sorted(n for n in names if names.count(n) > 1)}")
    known = set(names)
    for job in jobs:
        for dep in job.after:
            if dep not in known:
                raise ValueError(f"job{job.index:03d} waits on unknown job name {dep!r}")
            if next(j for j in jobs if j.name == dep).index >= job.index:
                raise ValueError(f"job{job.index:03d} waits on {dep!r}, which comes later in the file")
    return jobs


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GpuLocks:

    def __init__(self, directory: Path):
        self.directory = directory
        self.held: set[int] = set()
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, gpu: int) -> Path:
        return self.directory / f"gpu{gpu}.pid"

    def blocked(self, gpu: int) -> bool:
        if gpu in self.held:
            return False
        path = self._path(gpu)
        try:
            owner = int(path.read_text().split()[0])
        except FileNotFoundError:
            return False
        except (OSError, ValueError, IndexError):
            owner = None
        if owner is not None and owner != os.getpid() and _pid_alive(owner):
            return True
        try:
            path.unlink()
        except OSError:
            pass
        return False

    def acquire(self, gpus: list[int]) -> bool:
        got: list[int] = []
        for gpu in gpus:
            try:
                fd = os.open(self._path(gpu), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self.release(got)
                return False
            with os.fdopen(fd, "w") as handle:
                handle.write(f"{os.getpid()}\n")
            got.append(gpu)
            self.held.add(gpu)
        return True

    def release(self, gpus: list[int]) -> None:
        for gpu in gpus:
            if gpu not in self.held:
                continue
            try:
                self._path(gpu).unlink()
            except OSError:
                pass
            self.held.discard(gpu)

    def release_all(self) -> None:
        self.release(list(self.held))


def allocate_run_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    largest = 0
    for child in root.iterdir():
        match = RUN_DIR_RE.match(child.name)
        if match:
            largest = max(largest, int(match.group(1)))
    number = largest + 1
    while True:
        candidate = root / f"{number}_{name}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            number += 1


def _parse_mib(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 10**9


def _query_gpu_state_pynvml() -> dict[int, dict]:
    import pynvml

    pynvml.nvmlInit()
    try:
        gpus: dict[int, dict] = {}
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                util = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except pynvml.NVMLError:
                util = 10**9
            mem_mib = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used) // (1024 * 1024)
            proc_mem = 0
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                used = proc.usedGpuMemory
                proc_mem += 10**9 if used is None else int(used) // (1024 * 1024)
            gpus[index] = {"util": util, "mem_mib": mem_mib, "proc_mem_mib": proc_mem}
        return gpus
    finally:
        pynvml.nvmlShutdown()


def query_gpu_state() -> dict[int, dict]:
    if shutil.which("nvidia-smi") is None:
        return _query_gpu_state_pynvml()
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus: dict[int, dict] = {}
    uuid_to_index: dict[str, int] = {}
    for line in out.strip().splitlines():
        idx_s, uuid, util_s, mem_s = [part.strip() for part in line.split(",")]
        index = int(idx_s)
        uuid_to_index[uuid] = index
        gpus[index] = {"util": _parse_mib(util_s), "mem_mib": _parse_mib(mem_s), "proc_mem_mib": 0}
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        uuid, used_s = [part.strip() for part in line.split(",")]
        if uuid in uuid_to_index:
            gpus[uuid_to_index[uuid]]["proc_mem_mib"] += _parse_mib(used_s)
    return gpus


def launch(job: Job, assigned: list[int], log_dir: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in assigned)
    job.assigned = assigned
    job.log_path = log_dir / f"job{job.index:03d}.log"
    log_file = job.log_path.open("w")
    log_file.write(f"# gpus={assigned} cmd={job.cmd}\n")
    log_file.flush()
    job.proc = subprocess.Popen(
        job.cmd,
        shell=True,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job.started = time.monotonic()
    print(f"[{stamp()}] job{job.index:03d} START gpus={assigned} log={job.log_path} :: {job.cmd}", flush=True)


def _signal_group(job: Job, sig: int) -> None:
    try:
        os.killpg(os.getpgid(job.proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def kill_all(running: list[Job]) -> None:
    alive = [job for job in running if job.proc is not None and job.proc.poll() is None]
    if not alive:
        return
    for job in alive:
        _signal_group(job, signal.SIGTERM)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(job.proc.poll() is None for job in alive):
            time.sleep(0.2)
    except KeyboardInterrupt:
        note("second interrupt — SIGKILL now")
    for job in alive:
        while True:
            try:
                if job.proc.poll() is None:
                    _signal_group(job, signal.SIGKILL)
                job.proc.wait()
                break
            except KeyboardInterrupt:
                continue
        note(f"job{job.index:03d} killed (gpus {job.assigned})")


def _clip_cmd(cmd: str, width: int = 100) -> str:
    preview = re.sub(r"^(?:\w+=\S+\s+)*\S*python[\d.]*\s+(?:-u\s+)?", "", cmd)
    if len(preview) <= width:
        return preview
    head = width // 3
    tail = width - head - 5
    return f"{preview[:head]} ... {preview[-tail:]}"


def stamp() -> str:
    return time.strftime("%F %T")


def note(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def run_queue(args: argparse.Namespace, jobs: list[Job], allowed: list[int], log_dir: Path) -> int:
    running_marker = log_dir / ".running"
    running_marker.write_text(f"{os.getpid()} {stamp()}\n")
    print(f"job logs: {log_dir}")

    pending = list(jobs)
    running: list[Job] = []
    done: list[Job] = []
    failure_seen = False
    free_since: dict[int, float | None] = {gpu: None for gpu in allowed}
    last_wait_note = 0.0
    locks = GpuLocks(LOG_ROOT / "locks")

    def _terminate(signum: int, _frame) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGHUP, _terminate)

    try:
        while pending or running:
            for job in list(running):
                code = job.proc.poll()
                if code is None:
                    continue
                running.remove(job)
                done.append(job)
                job.exit_code = code
                locks.release(job.assigned)
                minutes = (time.monotonic() - job.started) / 60
                status = "DONE" if code == 0 else f"FAILED (exit {code})"
                print(f"[{stamp()}] job{job.index:03d} {status} after {minutes:.1f} min, gpus {job.assigned} released", flush=True)
                if code != 0 and args.stop_on_failure:
                    failure_seen = True

            if failure_seen and not running:
                print(f"[{stamp()}] stopping queue after job failure; {len(pending)} later job(s) will not start", flush=True)
                break

            try:
                state = query_gpu_state()
            except (OSError, subprocess.SubprocessError) as error:
                now = time.monotonic()
                if now - last_wait_note > 60:
                    print(
                        f"[{stamp()}] GPU status query failed ({error}); jobs remain tracked and the queue will retry",
                        flush=True,
                    )
                    last_wait_note = now
                time.sleep(args.poll_seconds)
                continue
            now = time.monotonic()
            reserved = {gpu for job in running for gpu in job.assigned}
            free: list[int] = []
            for gpu in allowed:
                info = state.get(gpu)
                busy = (
                    gpu in reserved
                    or info is None
                    or info["proc_mem_mib"] > args.grace_mib
                    or info["util"] > args.util_threshold
                    or info["mem_mib"] > max(args.mem_threshold_mib, args.grace_mib)
                    or locks.blocked(gpu)
                )
                if busy:
                    free_since[gpu] = None
                    continue
                if free_since[gpu] is None:
                    free_since[gpu] = now
                if now - free_since[gpu] >= args.free_seconds:
                    free.append(gpu)

            while not failure_seen and pending:
                head = pending[0]
                if head.after:
                    by_name = {j.name: j for j in done + running if j.name}
                    deps = [by_name.get(dep) for dep in head.after]
                    if any(d is None or d.exit_code is None for d in deps):
                        break
                    failed_deps = [dep for dep, d in zip(head.after, deps) if d.exit_code != 0]
                    if failed_deps:
                        pending.pop(0)
                        head.exit_code = -1
                        done.append(head)
                        print(f"[{stamp()}] job{head.index:03d} FAILED (dependency {failed_deps} failed), not started", flush=True)
                        if args.stop_on_failure:
                            failure_seen = True
                        continue
                if len(free) < head.gpus_needed:
                    break
                job = pending.pop(0)
                assigned = [free.pop(0) for _ in range(job.gpus_needed)]
                if not locks.acquire(assigned):
                    pending.insert(0, job)
                    for gpu in assigned:
                        free_since[gpu] = None
                    print(f"[{stamp()}] lost lock race for gpus {assigned}, will retry", flush=True)
                    break
                launch(job, assigned, log_dir)
                running.append(job)
                for gpu in assigned:
                    free_since[gpu] = None

            if now - last_wait_note > 60:
                gpu_to_job = {gpu: job for job in running for gpu in job.assigned}
                lines = [f"[{stamp()}] status:"]
                for gpu in allowed:
                    job = gpu_to_job.get(gpu)
                    line = (
                        f"  gpu {gpu}: job{job.index:03d} :: {_clip_cmd(job.cmd)}"
                        if job
                        else f"  gpu {gpu}: (free)"
                    )
                    lines.append(line)
                for job in pending:
                    lines.append(f"  queued job{job.index:03d} :: {_clip_cmd(job.cmd)}")
                lines.append(f"  {len(running)} running, {len(pending)} queued, {len(free)} gpu(s) free")
                print("\n".join(lines), flush=True)
                last_wait_note = now

            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        note(f"\n[{stamp()}] interrupted")
    finally:
        if running:
            note(f"[{stamp()}] killing {len(running)} running job(s)")
            kill_all(running)
            for job in running:
                job.exit_code = job.proc.returncode
                done.append(job)
            running.clear()
        locks.release_all()
        try:
            running_marker.unlink(missing_ok=True)
            note(f"job logs: {log_dir}")
        except OSError as error:
            note(f"could not remove running marker {running_marker}: {error}")

    print(f"\n[{stamp()}] queue finished — summary:")
    for job in sorted(done, key=lambda j: j.index):
        status = "ok" if job.exit_code == 0 else f"FAILED (exit {job.exit_code})"
        print(f"  job{job.index:03d} [{status}] {job.cmd}")
    for job in pending:
        print(f"  job{job.index:03d} [never started] {job.cmd}")
    return 0 if all(job.exit_code == 0 for job in done) and not pending else 1


def main() -> int:
    args = parse_args()
    jobs = load_jobs(Path(args.commands))
    if not jobs:
        print("Queue is empty, nothing to do.")
        return 0

    while True:
        try:
            all_gpus = sorted(query_gpu_state())
            break
        except (OSError, subprocess.SubprocessError) as error:
            print(f"[{stamp()}] GPU status query failed ({error}); retrying in 10 seconds", flush=True)
            time.sleep(10)
    allowed = sorted(int(g) for g in args.gpus.split(",")) if args.gpus else all_gpus
    unknown = [g for g in allowed if g not in all_gpus]
    if unknown:
        print(f"Unknown GPU indices {unknown}; visible GPUs are {all_gpus}", file=sys.stderr)
        return 1
    widest = max(job.gpus_needed for job in jobs)
    if widest > len(allowed):
        print(f"A job needs {widest} GPUs but only {len(allowed)} are allowed ({allowed})", file=sys.stderr)
        return 1

    queue_message = (
        f"{len(jobs)} job(s) queued on GPUs {allowed}; "
        f"a GPU must be idle {args.free_seconds:g}s to count as free."
    )
    if args.dry_run:
        print(queue_message)
        for job in jobs:
            dep = f" after={job.after}" if job.after else ""
            nm = f" name={job.name}" if job.name else ""
            print(f"  job{job.index:03d} gpus={job.gpus_needed}{nm}{dep} :: {job.cmd}")
        return 0

    log_dir = allocate_run_dir(LOG_ROOT, args.name)
    with mirror_queue_output(log_dir / "entrypoint.log"):
        print(queue_message)
        return run_queue(args, jobs, allowed, log_dir)


if __name__ == "__main__":
    sys.exit(main())

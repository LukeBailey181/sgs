from __future__ import annotations

import math
import os
import platform
import re
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence
from collections import defaultdict

import wandb
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch, Rectangle

from sgs.utils import export

__all__ = ["GPUSample", "SystemSample", "UtilizationReport", "ResourceMonitor"]

# Optional dependencies
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore

    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

from sgs.data.dataset_types import UtilizationReport, SystemSample, GPUSample


# ----------------------------- Helpers ----------------------------- #


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


def _parse_cpuset_list(s: str) -> int:
    # e.g., "0-3,8,10-11" -> count
    total = 0
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            total += int(b) - int(a) + 1
        else:
            total += 1
    return total


def _detect_cgroup_paths() -> Dict[str, str]:
    # Returns dict with keys: root (mount), path (this proc cgroup rel path), version ("v1"|"v2")
    # Try cgroup v2 unified
    mi = _read_file("/proc/self/mountinfo") or ""
    cg2_root = None
    for line in mi.splitlines():
        if " - cgroup2 " in line:
            parts = line.split()
            if len(parts) >= 5:
                cg2_root = parts[4]
                break

    rel: str | None
    cg = _read_file("/proc/self/cgroup") or ""
    if cg2_root:
        # v2 lines look like: "0::/slurm/uid_xxx/job_xxx/step_xxx"
        for line in cg.splitlines():
            if line.startswith("0::"):
                rel = line.split("::", 1)[1].strip()
                return {"version": "v2", "root": cg2_root, "path": rel}

    # Fallback v1: need specific controllers
    # e.g., "2:cpu,cpuacct:/slurm/uid_xxx/job_xxx/step_xxx"
    root = "/sys/fs/cgroup"
    rel = None
    for line in cg.splitlines():
        if ":" not in line:
            continue
        _, controllers, p = line.split(":", 2)
        if "cpuacct" in controllers or "cpu" in controllers:
            rel = p.strip()
    if rel:
        return {"version": "v1", "root": root, "path": rel}
    return {}


def _slurm_env() -> Dict[str, str]:
    keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_NNODES",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_CPUS_ON_NODE",
        "SLURM_TASKS_PER_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS",
        "SLURM_GPUS_PER_TASK",
        "SLURM_JOB_PARTITION",
        "SLURM_SUBMIT_DIR",
    ]
    return {k: v for k, v in os.environ.items() if k in keys}


def _safe_cmdline(p: "psutil.Process") -> str:
    try:
        cmd = p.cmdline()
        if not cmd:
            return p.name()
        return " ".join(cmd[:6])
    except Exception:
        try:
            return p.name()
        except Exception:
            return "<unknown>"


# ----------------------------- Monitor ----------------------------- #


class ResourceMonitor:
    def __init__(
        self,
        interval_sec: float = 5.0,
        track_children: bool = False,
        per_process_top_n: int = 0,
        track_cgroup: bool = True,
        profile_gpu: bool = False,
        timings_only: bool = False,
    ) -> None:
        export()

        self.interval = max(0.25, float(interval_sec))
        self.track_children = bool(track_children)
        self.per_process_top_n = int(per_process_top_n)
        self.track_cgroup = bool(track_cgroup)

        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self.samples: List[SystemSample] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # psutil process + baselines
        self._proc = None
        self._io0 = None
        self._net0 = None
        self._proc_cache: Dict[int, "psutil.Process"] = {} if psutil is not None else {}

        # cgroup state
        self._cg: Optional[Dict[str, str]] = None
        self._cg_prev_usage: Optional[int] = None  # usec (v2) or ns (v1)
        self._cg_prev_ts: Optional[float] = None
        self._cg_assigned_cpus: Optional[int] = None

        # NVML state
        self._nvml_enabled = _HAS_NVML and profile_gpu
        self._gpu_handles: Optional[List[Any]] = None

        if psutil is not None:
            try:
                self._proc = psutil.Process(os.getpid())
            except Exception:
                self._proc = None

        self.timings_only = timings_only

        self.has_started = False

    def start(self) -> None:
        # psutil primes

        if not self.timings_only:
            if psutil is not None:
                try:
                    psutil.cpu_percent(interval=None)
                except Exception:
                    pass
                try:
                    if self._proc is not None:
                        self._proc.cpu_percent(interval=None)
                except Exception:
                    pass
                try:
                    self._io0 = psutil.disk_io_counters()
                except Exception:
                    self._io0 = None
                try:
                    self._net0 = psutil.net_io_counters()
                except Exception:
                    self._net0 = None

                # Children prime
                if self._proc is not None and self.track_children:
                    try:
                        for ch in self._proc.children(recursive=True):
                            try:
                                self._proc_cache[ch.pid] = ch
                                ch.cpu_percent(interval=None)
                            except Exception:
                                pass
                    except Exception:
                        pass

            # cgroup detection
            if self.track_cgroup and os.name == "posix":
                self._cg = _detect_cgroup_paths()
                if self._cg:
                    base = os.path.join(self._cg["root"], self._cg["path"].lstrip("/"))
                    try:
                        if self._cg["version"] == "v2":
                            cpus = (
                                _read_file(os.path.join(base, "cpuset.cpus.effective"))
                                or _read_file(os.path.join(base, "cpuset.cpus"))
                                or ""
                            ).strip()
                            self._cg_assigned_cpus = (
                                _parse_cpuset_list(cpus) if cpus else None
                            )
                            usage = _read_file(os.path.join(base, "cpu.stat")) or ""
                            m = re.search(r"usage_usec\s+(\d+)", usage)
                            if m:
                                self._cg_prev_usage = int(m.group(1))  # usec
                                self._cg_prev_ts = time.time()
                        else:
                            # v1
                            cpus = (
                                _read_file(
                                    os.path.join(
                                        self._cg["root"],
                                        "cpuset",
                                        self._cg["path"].lstrip("/"),
                                        "cpuset.cpus",
                                    )
                                )
                                or ""
                            ).strip()
                            self._cg_assigned_cpus = (
                                _parse_cpuset_list(cpus) if cpus else None
                            )
                            usage_ns = _read_file(
                                os.path.join(
                                    self._cg["root"],
                                    "cpuacct",
                                    self._cg["path"].lstrip("/"),
                                    "cpuacct.usage",
                                )
                            )
                            if usage_ns and usage_ns.strip().isdigit():
                                self._cg_prev_usage = int(usage_ns.strip())  # ns
                                self._cg_prev_ts = time.time()
                    except Exception:
                        # Ignore cgroup setup errors
                        self._cg = None

            # NVML init
            if self._nvml_enabled:
                try:
                    pynvml.nvmlInit()
                    cnt = pynvml.nvmlDeviceGetCount()
                    self._gpu_handles = [
                        pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(cnt)
                    ]
                except Exception:
                    self._nvml_enabled = False
                    self._gpu_handles = None

        self.start_time = time.time()

        self.has_started = True

        if not self.timings_only:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()

    def report(self, num_examples: Optional[int] = None) -> UtilizationReport:
        end_time = time.time()
        node_name = os.environ.get("SLURMD_NODENAME") or socket.gethostname()

        # Host info
        mem_total_gb = None
        cpu_count = None
        if psutil is not None:
            try:
                mem_total_gb = psutil.virtual_memory().total / (1024**3)
            except Exception:
                pass
            try:
                cpu_count = psutil.cpu_count(logical=True)
            except Exception:
                pass

        if not self.timings_only:
            with self._samples_lock:
                summary = self._compute_summary()
                samples = list(self.samples)
        else:
            summary = {}
            samples = []

        return UtilizationReport(
            start_time=self.start_time or 0.0,
            end_time=end_time,
            duration_sec=(end_time or 0.0) - (self.start_time or 0.0),
            hostname=socket.gethostname(),
            platform=f"{platform.system()} {platform.release()}",
            node_name=node_name,
            cpu_count=cpu_count,
            mem_total_gb=mem_total_gb,
            slurm_env=_slurm_env(),
            samples=samples,
            summary=summary,
            num_examples=num_examples,
        )

    def stop_and_report(
        self,
        num_examples: Optional[int] = None,
    ) -> UtilizationReport:
        if not self.timings_only:
            self._stop.set()
            if self._t:
                self._t.join()
        self.end_time = time.time()

        # NVML shutdown
        if not self.timings_only:
            if self._nvml_enabled:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        # Host info
        mem_total_gb = None
        cpu_count = None
        if psutil is not None:
            try:
                mem_total_gb = psutil.virtual_memory().total / (1024**3)
            except Exception:
                pass
            try:
                cpu_count = psutil.cpu_count(logical=True)
            except Exception:
                pass

        if not self.timings_only:
            summary = self._compute_summary()
            samples = self.samples
        else:
            summary = {}
            samples = []

        node_name = os.environ.get("SLURMD_NODENAME") or socket.gethostname()

        return UtilizationReport(
            start_time=self.start_time or 0.0,
            end_time=self.end_time or 0.0,
            duration_sec=(self.end_time or 0.0) - (self.start_time or 0.0),
            hostname=socket.gethostname(),
            platform=f"{platform.system()} {platform.release()}",
            node_name=node_name,
            cpu_count=cpu_count,
            mem_total_gb=mem_total_gb,
            slurm_env=_slurm_env(),
            samples=samples,
            summary=summary,
            num_examples=num_examples,
        )

    def _run(self) -> None:
        if not self.timings_only:
            while not self._stop.is_set():
                self._collect_once()
                self._stop.wait(self.interval)

    # ----------------------------- Sampling methods ----------------------------- #

    def _collect_gpu_samples(self) -> List[GPUSample]:
        out: List[GPUSample] = []
        if not self._nvml_enabled or not self._gpu_handles:
            return out
        for i, h in enumerate(self._gpu_handles):
            try:
                name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                power = None
                temp = None
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    pass
                try:
                    temp = float(
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    )
                except Exception:
                    pass
                out.append(
                    GPUSample(
                        index=i,
                        name=name,
                        util_percent=float(util),
                        mem_used_mb=mem.used / (1024**2),
                        mem_total_mb=mem.total / (1024**2),
                        power_watts=power,
                        temperature_c=temp,
                    )
                )
            except Exception:
                continue
        return out

    def _collect_once(self) -> None:
        ts = time.time()
        cpu_percent = None
        mem_used_gb = None
        mem_percent = None
        load1 = load5 = load15 = None
        proc_cpu = proc_mem = None
        proc_threads = None
        io_read = io_write = None
        net_sent = net_recv = None

        if psutil is not None:
            try:
                cpu_percent = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            try:
                vm = psutil.virtual_memory()
                mem_used_gb = vm.used / (1024**3)
                mem_percent = float(vm.percent)
            except Exception:
                pass
            try:
                l1, l5, l15 = os.getloadavg()
                load1, load5, load15 = float(l1), float(l5), float(l15)
            except Exception:
                load1 = load5 = load15 = None
            try:
                if self._proc is not None:
                    proc_cpu = float(self._proc.cpu_percent(interval=None))
                    pm = self._proc.memory_info()
                    proc_mem = pm.rss / (1024**2)
                    proc_threads = self._proc.num_threads()
            except Exception:
                pass
            try:
                if self._io0 is not None:
                    io = psutil.disk_io_counters()
                    io_read = (io.read_bytes - self._io0.read_bytes) / (1024**2)
                    io_write = (io.write_bytes - self._io0.write_bytes) / (1024**2)
            except Exception:
                pass
            try:
                if self._net0 is not None:
                    net = psutil.net_io_counters()
                    net_sent = (net.bytes_sent - self._net0.bytes_sent) / (1024**2)
                    net_recv = (net.bytes_recv - self._net0.bytes_recv) / (1024**2)
            except Exception:
                pass

        # cgroup-based CPU utilization normalized by assigned CPUs
        cgroup_util_percent = None
        cgroup_assigned_cpus = self._cg_assigned_cpus
        if (
            self._cg
            and self._cg_prev_usage is not None
            and self._cg_prev_ts is not None
            and self._cg_assigned_cpus
        ):
            now = ts
            base = os.path.join(self._cg["root"], self._cg["path"].lstrip("/"))
            try:
                if self._cg["version"] == "v2":
                    usage = _read_file(os.path.join(base, "cpu.stat")) or ""
                    m = re.search(r"usage_usec\s+(\d+)", usage)
                    if m:
                        cur = int(m.group(1))  # usec
                        du = max(0, cur - self._cg_prev_usage) / 1_000_000.0  # sec
                        dt = max(1e-6, now - self._cg_prev_ts)  # sec
                        frac = (du / dt) / self._cg_assigned_cpus
                        cgroup_util_percent = min(100.0, max(0.0, frac * 100.0))
                        self._cg_prev_usage, self._cg_prev_ts = cur, now
                else:
                    usage_ns = _read_file(
                        os.path.join(
                            self._cg["root"],
                            "cpuacct",
                            self._cg["path"].lstrip("/"),
                            "cpuacct.usage",
                        )
                    )
                    if usage_ns and usage_ns.strip().isdigit():
                        cur = int(usage_ns.strip())  # ns
                        du = max(0, cur - self._cg_prev_usage) / 1_000_000_000.0  # sec
                        dt = max(1e-6, now - self._cg_prev_ts)  # sec
                        frac = (du / dt) / self._cg_assigned_cpus
                        cgroup_util_percent = min(100.0, max(0.0, frac * 100.0))
                        self._cg_prev_usage, self._cg_prev_ts = cur, now
            except Exception:
                pass

        # Process tree aggregation
        proc_tree_cpu = None
        proc_tree_mem = None
        top_children = None
        if psutil is not None and self._proc is not None and self.track_children:
            try:
                current_children = {
                    ch.pid: ch for ch in self._proc.children(recursive=True)
                }
            except Exception:
                current_children = {}

            # Remove dead
            for pid in list(self._proc_cache.keys()):
                if pid not in current_children:
                    self._proc_cache.pop(pid, None)

            # Add new and prime
            for pid, ch in current_children.items():
                if pid not in self._proc_cache:
                    self._proc_cache[pid] = ch
                    try:
                        ch.cpu_percent(interval=None)
                    except Exception:
                        pass

            # Aggregate
            cpu_sum = 0.0
            mem_sum_mb = 0.0
            child_rows = []
            for pid, ch in list(self._proc_cache.items()):
                try:
                    cpu = float(ch.cpu_percent(interval=None))
                    rss_mb = ch.memory_info().rss / (1024**2)
                    cpu_sum += cpu
                    mem_sum_mb += rss_mb
                    if self.per_process_top_n > 0:
                        child_rows.append(
                            {
                                "pid": pid,
                                "name": ch.name(),
                                "cpu_percent": cpu,
                                "mem_mb": rss_mb,
                                "cmd": _safe_cmdline(ch),
                            }
                        )
                except Exception:
                    # Process may have exited
                    self._proc_cache.pop(pid, None)
                    continue

            proc_tree_cpu = cpu_sum
            proc_tree_mem = mem_sum_mb
            if self.per_process_top_n > 0 and child_rows:
                child_rows.sort(
                    key=lambda r: (r["cpu_percent"], r["mem_mb"]), reverse=True
                )
                top_children = child_rows[: self.per_process_top_n]

        gpus = self._collect_gpu_samples()

        with self._samples_lock:
            self.samples.append(
                SystemSample(
                    timestamp=ts,
                    cpu_percent=cpu_percent or 0.0,
                    mem_used_gb=mem_used_gb or 0.0,
                    mem_percent=mem_percent or 0.0,
                    load_avg_1=load1,
                    load_avg_5=load5,
                    load_avg_15=load15,
                    process_cpu_percent=proc_cpu,
                    process_mem_mb=proc_mem,
                    process_num_threads=proc_threads,
                    io_read_mb=io_read,
                    io_write_mb=io_write,
                    net_sent_mb=net_sent,
                    net_recv_mb=net_recv,
                    gpus=gpus,
                    process_tree_cpu_percent=proc_tree_cpu,
                    process_tree_mem_mb=proc_tree_mem,
                    top_children=top_children,
                    cgroup_cpu_util_percent=cgroup_util_percent,
                    cgroup_assigned_cpus=cgroup_assigned_cpus,
                )
            )

    # ----------------------------- Summary ----------------------------- #

    def _compute_summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {}

        def avg(xs: List[float]) -> Optional[float]:
            xs = [x for x in xs if x is not None]
            return (sum(xs) / len(xs)) if xs else None

        def peak(xs: List[float]) -> Optional[float]:
            xs = [x for x in xs if x is not None]
            return max(xs) if xs else None

        cpu = [s.cpu_percent for s in self.samples]
        mem = [s.mem_percent for s in self.samples]
        proc_cpu = [
            s.process_cpu_percent
            for s in self.samples
            if s.process_cpu_percent is not None
        ]
        proc_mem = [
            s.process_mem_mb for s in self.samples if s.process_mem_mb is not None
        ]
        tree_cpu = [
            s.process_tree_cpu_percent
            for s in self.samples
            if s.process_tree_cpu_percent is not None
        ]
        tree_mem = [
            s.process_tree_mem_mb
            for s in self.samples
            if s.process_tree_mem_mb is not None
        ]
        cg_util = [
            s.cgroup_cpu_util_percent
            for s in self.samples
            if s.cgroup_cpu_util_percent is not None
        ]

        # GPU flatten
        gpu_utils: List[float] = []
        gpu_mem_used: List[float] = []
        for s in self.samples:
            for g in s.gpus:
                gpu_utils.append(g.util_percent)
                gpu_mem_used.append(g.mem_used_mb)

        return {
            # System-wide
            "cpu_percent_avg": avg(cpu),
            "cpu_percent_peak": peak(cpu),
            "mem_percent_avg": avg(mem),
            "mem_percent_peak": peak(mem),
            # Parent process
            "proc_cpu_percent_avg": avg(proc_cpu),
            "proc_cpu_percent_peak": peak(proc_cpu),
            "proc_mem_mb_avg": avg(proc_mem),
            "proc_mem_mb_peak": peak(proc_mem),
            # Process tree
            "proc_tree_cpu_percent_avg": avg(tree_cpu),
            "proc_tree_cpu_percent_peak": peak(tree_cpu),
            "proc_tree_mem_mb_avg": avg(tree_mem),
            "proc_tree_mem_mb_peak": peak(tree_mem),
            # cgroup (SLURM CPUs)
            "cgroup_cpu_util_percent_avg": avg(cg_util),
            "cgroup_cpu_util_percent_peak": peak(cg_util),
            "cgroup_assigned_cpus": self._cg_assigned_cpus,
            # GPU
            "gpu_util_percent_avg": (sum(gpu_utils) / len(gpu_utils))
            if gpu_utils
            else None,
            "gpu_util_percent_peak": max(gpu_utils) if gpu_utils else None,
            "gpu_mem_mb_avg": (sum(gpu_mem_used) / len(gpu_mem_used))
            if gpu_mem_used
            else None,
            "gpu_mem_mb_peak": max(gpu_mem_used) if gpu_mem_used else None,
        }


# ----------------------------- Logging ----------------------------- #


def log_utilization_report_timings_DEP(
    reports: Sequence[UtilizationReport | None],
    wandb_run: wandb.Run | None,
    wandb_prefix: str,
):
    if wandb_run is None:
        return

    # Filter valid intervals and keep node_name
    intervals: List[Tuple[float, float, str]] = []
    for r in reports:
        if r is None:
            continue
        s, e = getattr(r, "start_time", None), getattr(r, "end_time", None)
        if s is None or e is None:
            continue
        if e < s:
            s, e = e, s
        node = (
            getattr(r, "node_name", None) or getattr(r, "hostname", None) or "<unknown>"
        )
        intervals.append((float(s), float(e), str(node)))

    if not intervals:
        return

    # Normalize to start at 0 and convert to minutes
    intervals.sort(key=lambda x: x[0])
    t0 = intervals[0][0]
    norm_min: List[Tuple[float, float, str]] = [
        ((s - t0) / 60.0, (e - t0) / 60.0, node) for s, e, node in intervals
    ]

    # Color map per node_name (stable order of first appearance)
    unique_nodes: List[str] = []
    for _, _, node in intervals:
        if node not in unique_nodes:
            unique_nodes.append(node)
    cmap = plt.get_cmap("tab20")
    node2color = {node: cmap(i % cmap.N) for i, node in enumerate(unique_nodes)}

    # Assign to lanes: lowest-index lane whose latest end <= start, else new lane
    lane_ends: List[float] = []
    assigned: List[Tuple[int, float, float, str]] = []  # (lane, start, end, node)
    for s, e, node in norm_min:
        placed = False
        for i, end in enumerate(lane_ends):
            if s >= end:
                lane_ends[i] = e
                assigned.append((i, s, e, node))
                placed = True
                break
        if not placed:
            lane_ends.append(e)
            assigned.append((len(lane_ends) - 1, s, e, node))

    num_lanes = len(lane_ends)
    max_x = max(e for _, _, e, _ in assigned) if assigned else 0.0

    # Plot
    fig_h = max(2.0, 0.4 * num_lanes + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bar_h = 0.35
    for lane, s, e, node in assigned:
        y = lane + 1
        w = max(1e-6, e - s)
        ax.add_patch(
            Rectangle(
                (s, y - bar_h / 2),
                w,
                bar_h,
                facecolor=node2color[node],
                edgecolor="black",
                linewidth=1.5,
                zorder=2,
            )
        )

    # y-axis: integers only
    ax.set_ylim(0.5, num_lanes + 0.5)
    ax.set_yticks(list(range(1, num_lanes + 1)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # x-axis: minutes
    ax.set_xlim(0.0, max_x * 1.02 if max_x > 0 else 1.0)
    ax.set_xlabel("Job duration (minutes)")
    ax.set_ylabel("Concurrent jobs (lane)")
    title_prefix = (wandb_prefix + " ") if wandb_prefix else ""
    ax.set_title(
        f"{title_prefix}Job timeline (jobs={len(intervals)}, peak concurrency={num_lanes})"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Legend for nodes (outside to the right)
    legend_handles = [
        Patch(facecolor=node2color[n], edgecolor="black", linewidth=1, label=n)
        for n in unique_nodes
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Node",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
        )

    # Reserve space on the right for the outside legend
    fig.tight_layout(rect=(0, 0, 0.8, 1.0))

    wandb_run.log({f"{wandb_prefix}/job_timeline": wandb.Image(fig)})

    plt.close(fig)

    # ------------------------------------------------------------ #
    # Log table of per node data
    # ------------------------------------------------------------ #

    # Now we will get the host name and throughput of each utilization report and log this as a table
    node_data: List[
        Tuple[
            str,
            float,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
        ]
    ] = []

    inner_node_throughputs: Dict[str, List[float]] = defaultdict(list)
    outer_node_throughputs: Dict[str, List[float | None]] = defaultdict(list)

    for r in reports:
        if r is None or r.num_examples is None:
            continue

        inner_throughput = r.num_examples / r.duration_sec

        if r.outer_job_start_time is not None and r.outer_job_end_time is not None:
            if r.outer_job_start_time == r.outer_job_end_time:
                # These are in terms of seconds, so they can be equal for short jobs
                outer_throughput = float(r.num_examples)
            else:
                outer_throughput = r.num_examples / (
                    r.outer_job_end_time - r.outer_job_start_time
                )

            unpickle_input_time = r.start_time - r.outer_job_start_time
            pickle_output_time = r.outer_job_end_time - r.end_time
        else:
            outer_throughput = None
            unpickle_input_time = None
            pickle_output_time = None

        node_data.append(
            (
                r.node_name,
                r.duration_sec,
                inner_throughput,
                outer_throughput,
                r.num_examples,
                unpickle_input_time,
                pickle_output_time,
            )
        )

        inner_node_throughputs[r.node_name].append(inner_throughput)
        outer_node_throughputs[r.node_name].append(outer_throughput)

    node_data.sort(key=lambda x: x[1], reverse=True)
    table = wandb.Table(
        data=node_data,
        columns=[
            "Node",
            "Duration",
            "Inner Throughput",
            "Outer Throughput",
            "Num Examples",
            "Unpickle Input Time",
            "Pickle Output Time",
        ],
    )

    wandb_run.log({f"{wandb_prefix}/node_data_table": table})

    # Compute per-node averages (filtering out missing outer throughputs)
    nodes = sorted(
        inner_node_throughputs.keys(),
        key=lambda n: (sum(inner_node_throughputs[n]) / len(inner_node_throughputs[n])),
        reverse=True,
    )
    inner_avgs = [
        sum(inner_node_throughputs[n]) / len(inner_node_throughputs[n]) for n in nodes
    ]
    outer_avgs = []
    for n in nodes:
        vals = [v for v in outer_node_throughputs.get(n, []) if v is not None]
        outer_avgs.append((sum(vals) / len(vals)) if vals else float("nan"))

        # Grouped bar plot: inner vs outer per node
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(nodes)))
    width = 0.4
    bars_inner = ax.bar(
        [i - width / 2 for i in x], inner_avgs, width, label="Inner Throughput"
    )
    bars_outer = ax.bar(
        [i + width / 2 for i in x], outer_avgs, width, label="Outer Throughput"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=45, ha="right")
    ax.set_xlabel("Node")
    ax.set_ylabel("Throughput (examples/s)")
    ax.set_title("Throughput per node")
    ax.legend()

    # Put values on top of bars (skip NaN/inf)
    def _autolabel(bar_container):
        for rect in bar_container:
            h = rect.get_height()
            if not math.isfinite(h):
                continue
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _autolabel(bars_inner)
    _autolabel(bars_outer)

    fig.tight_layout()

    wandb_run.log({f"{wandb_prefix}/node_throughput_plot": wandb.Image(fig)})
    plt.close(fig)


def log_utilization_report_timings(
    reports: Sequence[UtilizationReport | None],
    wandb_run: wandb.Run | None,
    wandb_prefix: str,
):
    if wandb_run is None:
        return

    # Build job records: inner (start,end), optional outer (start,end), and node
    jobs: List[Tuple[float, float, Optional[float], Optional[float], str]] = []
    for r in reports:
        if r is None:
            continue
        s, e = getattr(r, "start_time", None), getattr(r, "end_time", None)
        if s is None or e is None:
            continue
        if e < s:
            s, e = e, s
        node = (
            getattr(r, "node_name", None) or getattr(r, "hostname", None) or "<unknown>"
        )

        os_outer = getattr(r, "outer_job_start_time", None)
        oe_outer = getattr(r, "outer_job_end_time", None)
        if os_outer is not None and oe_outer is not None:
            osf, oef = float(os_outer), float(oe_outer)
            if oef < osf:
                osf, oef = oef, osf
        else:
            osf, oef = None, None

        jobs.append((float(s), float(e), osf, oef, str(node)))

    if not jobs:
        return

    # Use earliest outer start (if present) else earliest inner start to normalize to t=0
    t0_candidates = [j[2] if j[2] is not None else j[0] for j in jobs]
    t0 = min(t0_candidates)

    # Normalize times to minutes from t0
    # Each entry: (s_in_min, e_in_min, node, s_out_min|None, e_out_min|None)
    norm_jobs: List[Tuple[float, float, str, Optional[float], Optional[float]]] = []
    for s, e, osf, oef, node in jobs:
        s_in = (s - t0) / 60.0
        e_in = (e - t0) / 60.0
        s_out = ((osf - t0) / 60.0) if osf is not None else None
        e_out = ((oef - t0) / 60.0) if oef is not None else None
        norm_jobs.append((s_in, e_in, node, s_out, e_out))

    # Color map per node_name (stable order of first appearance)
    unique_nodes: List[str] = []
    for _, _, node, _, _ in norm_jobs:
        if node not in unique_nodes:
            unique_nodes.append(node)
    cmap = plt.get_cmap("tab20")
    node2color = {node: cmap(i % cmap.N) for i, node in enumerate(unique_nodes)}

    # Assign to lanes based on outer intervals when present; otherwise inner intervals
    # Sort by lane_start to ensure deterministic greedy packing
    def lane_start_end(job: Tuple[float, float, str, Optional[float], Optional[float]]):
        s_in, e_in, _node, s_out, e_out = job
        s_lane = s_out if s_out is not None else s_in
        e_lane = e_out if e_out is not None else e_in
        return s_lane, e_lane

    norm_jobs.sort(key=lambda j: lane_start_end(j)[0])

    lane_ends: List[float] = []
    # (lane, s_in, e_in, node, s_out, e_out, s_lane, e_lane)
    assigned: List[
        Tuple[int, float, float, str, Optional[float], Optional[float], float, float]
    ] = []
    for s_in, e_in, node, s_out, e_out in norm_jobs:
        s_lane = s_out if s_out is not None else s_in
        e_lane = e_out if e_out is not None else e_in

        placed = False
        for i, end in enumerate(lane_ends):
            if s_lane >= end:
                lane_ends[i] = e_lane
                assigned.append((i, s_in, e_in, node, s_out, e_out, s_lane, e_lane))
                placed = True
                break
        if not placed:
            lane_ends.append(e_lane)
            assigned.append(
                (len(lane_ends) - 1, s_in, e_in, node, s_out, e_out, s_lane, e_lane)
            )

    num_lanes = len(lane_ends)
    # x-limit based on assigned lane ends (respects outer when present)
    max_x = max(e_lane for *_, e_lane in assigned) if assigned else 0.0

    # Plot
    fig_h = max(2.0, 0.4 * num_lanes + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bar_h = 0.35
    outer_h = bar_h + 0.2  # slightly taller outline for the outer window
    for lane, s_in, e_in, node, s_out, e_out, s_lane, e_lane in assigned:
        y = lane + 1

        # Draw outer window as a dashed outline rectangle if present
        if s_out is not None and e_out is not None:
            w_out = max(1e-6, e_out - s_out)
            ax.add_patch(
                Rectangle(
                    (s_out, y - outer_h / 2),
                    w_out,
                    outer_h,
                    facecolor="none",
                    edgecolor=node2color[node],
                    linestyle="--",
                    linewidth=1.5,
                    zorder=1,
                )
            )

        # Draw inner (compute) window as a filled bar
        w_in = max(1e-6, e_in - s_in)
        ax.add_patch(
            Rectangle(
                (s_in, y - bar_h / 2),
                w_in,
                bar_h,
                facecolor=node2color[node],
                edgecolor="black",
                linewidth=1.5,
                zorder=2,
            )
        )

    # y-axis: integers only
    ax.set_ylim(0.5, num_lanes + 0.5)
    ax.set_yticks(list(range(1, num_lanes + 1)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # x-axis: minutes
    ax.set_xlim(0.0, max_x * 1.02 if max_x > 0 else 1.0)
    ax.set_xlabel("Job duration (minutes)")
    ax.set_ylabel("Concurrent jobs (lane)")
    title_prefix = (wandb_prefix + " ") if wandb_prefix else ""
    ax.set_title(
        f"{title_prefix}Job timeline (jobs={len(jobs)}, peak concurrency={num_lanes})"
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Legend for nodes (outside to the right)
    legend_handles = [
        Patch(facecolor=node2color[n], edgecolor="black", linewidth=1, label=n)
        for n in unique_nodes
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Node",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
        )

    # Reserve space on the right for the outside legend
    fig.tight_layout(rect=(0, 0, 0.8, 1.0))

    wandb_run.log({f"{wandb_prefix}/job_timeline": wandb.Image(fig)})

    plt.close(fig)

    # ------------------------------------------------------------ #
    # Log table of per node data
    # ------------------------------------------------------------ #
    node_data: List[
        Tuple[
            str,
            float,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
        ]
    ] = []

    inner_node_throughputs: Dict[str, List[float]] = defaultdict(list)
    outer_node_throughputs: Dict[str, List[float | None]] = defaultdict(list)

    for r in reports:
        if r is None or r.num_examples is None:
            continue

        inner_throughput = r.num_examples / r.duration_sec

        if r.outer_job_start_time is not None and r.outer_job_end_time is not None:
            if r.outer_job_start_time == r.outer_job_end_time:
                outer_throughput = float(r.num_examples)
            else:
                outer_throughput = r.num_examples / (
                    r.outer_job_end_time - r.outer_job_start_time
                )
            unpickle_input_time = r.start_time - r.outer_job_start_time
            pickle_output_time = r.outer_job_end_time - r.end_time
        else:
            outer_throughput = None
            unpickle_input_time = None
            pickle_output_time = None

        percent_errors = (
            r.num_errors / r.num_examples if r.num_errors is not None else None
        )

        node_data.append(
            (
                r.node_name,
                r.duration_sec,
                inner_throughput,
                outer_throughput,
                r.num_examples,
                unpickle_input_time,
                pickle_output_time,
                percent_errors,
            )
        )

        inner_node_throughputs[r.node_name].append(inner_throughput)
        outer_node_throughputs[r.node_name].append(outer_throughput)

    node_data.sort(key=lambda x: x[1], reverse=True)
    table = wandb.Table(
        data=node_data,
        columns=[
            "Node",
            "Duration",
            "Inner Throughput",
            "Outer Throughput",
            "Num Examples",
            "Unpickle Input Time",
            "Pickle Output Time",
            "Percent Errors",
        ],
    )
    wandb_run.log({f"{wandb_prefix}/node_data_table": table})

    # Compute per-node averages (filtering out missing outer throughputs)
    nodes = sorted(
        inner_node_throughputs.keys(),
        key=lambda n: (sum(inner_node_throughputs[n]) / len(inner_node_throughputs[n])),
        reverse=True,
    )
    inner_avgs = [
        sum(inner_node_throughputs[n]) / len(inner_node_throughputs[n]) for n in nodes
    ]
    outer_avgs = []
    for n in nodes:
        vals = [v for v in outer_node_throughputs.get(n, []) if v is not None]
        outer_avgs.append((sum(vals) / len(vals)) if vals else float("nan"))

    # Grouped bar plot: inner vs outer per node
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(nodes)))
    width = 0.4
    bars_inner = ax.bar(
        [i - width / 2 for i in x], inner_avgs, width, label="Inner Throughput"
    )
    bars_outer = ax.bar(
        [i + width / 2 for i in x], outer_avgs, width, label="Outer Throughput"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=45, ha="right")
    ax.set_xlabel("Node")
    ax.set_ylabel("Throughput (examples/s)")
    ax.set_title("Throughput per node")
    ax.legend()

    def _autolabel(bar_container):
        for rect in bar_container:
            h = rect.get_height()
            if not math.isfinite(h):
                continue
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    _autolabel(bars_inner)
    _autolabel(bars_outer)

    fig.tight_layout()
    wandb_run.log({f"{wandb_prefix}/node_throughput_plot": wandb.Image(fig)})
    plt.close(fig)

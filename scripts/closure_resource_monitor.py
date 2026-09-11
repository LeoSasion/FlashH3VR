"""Read-only psutil/NVML sampling. This module never loads a neural model."""
import json
import os
import time
from pathlib import Path

import psutil
import pynvml as nv


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def valid_memory(value):
    # NVML returns None or an unsigned sentinel for unsupported Windows counters.
    return int(value) if isinstance(value, int) and 0 <= value < 2**60 else None


def interval_stats(rows, key, start, end):
    """Time-weight interval readings by their intersection with the requested phase."""
    selected = []
    for row in rows:
        value = row.get(key)
        weight = max(0., min(end, row['t']) - max(start, row['interval_start']))
        if isinstance(value, (float, int)) and weight > 0:
            selected.append((value, weight))
    covered = sum(w for _, w in selected)
    return dict(mean=sum(v*w for v, w in selected)/covered if covered else None,
                sampled_peak=max((v for v, _ in selected), default=None),
                samples=len(selected), covered_seconds=covered)


class Sampler:
    def __init__(self, root_pid=None):
        nv.nvmlInit()
        self.handle = nv.nvmlDeviceGetHandleByIndex(0)
        self.root_pid = root_pid
        self.previous = {}
        self.last_t = time.perf_counter()
        self.process_catalog = {}
        psutil.cpu_percent(interval=None, percpu=True)

    def close(self):
        nv.nvmlShutdown()

    def hardware(self):
        return dict(logical_cpus=psutil.cpu_count(), physical_cores=psutil.cpu_count(logical=False),
                    gpu_name=nv.nvmlDeviceGetName(self.handle), gpu_uuid=nv.nvmlDeviceGetUUID(self.handle),
                    driver=nv.nvmlSystemGetDriverVersion(), gpu_index=0,
                    ram_total_bytes=psutil.virtual_memory().total,
                    gpu_total_bytes=nv.nvmlDeviceGetMemoryInfo(self.handle).total,
                    sampler_pid=os.getpid(), psutil_version=psutil.__version__)

    def sample(self):
        started = time.perf_counter()
        errors = {}

        def optional(name, callback):
            try:
                return callback()
            except (nv.NVMLError, psutil.Error, OSError) as error:
                errors[name] = str(error)
                return None

        cores = psutil.cpu_percent(interval=None, percpu=True)
        vm = psutil.virtual_memory()
        descendants = []
        if self.root_pid is not None:
            try:
                root = psutil.Process(self.root_pid)
                descendants = [root] + root.children(recursive=True)
            except psutil.Error as error:
                errors['process_tree'] = str(error)
        snapshots = {}
        rss = 0
        process_rows = []
        for process in descendants:
            try:
                with process.oneshot():
                    identity = (process.pid, process.create_time())
                    cpu = process.cpu_times()
                    seconds = cpu.user + cpu.system
                    memory = process.memory_info().rss
                    name = process.name()
                snapshots[identity] = seconds
                rss += memory
                self.process_catalog[str(identity)] = dict(pid=process.pid, create_time=identity[1], name=name)
                process_rows.append(dict(pid=process.pid, create_time=identity[1], cpu_seconds=seconds, rss_bytes=memory))
            except psutil.Error as error:
                errors[f'pid_{process.pid}'] = str(error)
        common = snapshots.keys() & self.previous.keys()
        cpu_delta = sum(max(0., snapshots[k] - self.previous[k]) for k in common)
        elapsed = started - self.last_t
        tree_pct = 100*cpu_delta/elapsed if common and elapsed > 0 else None
        util = optional('gpu_utilization', lambda: nv.nvmlDeviceGetUtilizationRates(self.handle))
        memory = optional('gpu_memory', lambda: nv.nvmlDeviceGetMemoryInfo(self.handle))
        encoder = optional('gpu_encoder', lambda: nv.nvmlDeviceGetEncoderUtilization(self.handle))
        decoder = optional('gpu_decoder', lambda: nv.nvmlDeviceGetDecoderUtilization(self.handle))
        gpu_processes = []
        seen = set()
        for kind, api in [('compute', nv.nvmlDeviceGetComputeRunningProcesses),
                          ('graphics', nv.nvmlDeviceGetGraphicsRunningProcesses)]:
            values = optional('gpu_processes_' + kind, lambda: api(self.handle))
            for p in values or []:
                value = valid_memory(p.usedGpuMemory)
                try:
                    name = psutil.Process(p.pid).name()
                except psutil.Error:
                    name = None
                gpu_processes.append(dict(pid=p.pid, name=name, kind=kind, used_bytes=value))
                seen.add(p.pid)
        owned = {p.pid for p in descendants}
        owned_memory = {p['pid']: p['used_bytes'] for p in gpu_processes if p['pid'] in owned}
        tree_gpu_memory = sum(owned_memory.values()) if owned_memory and all(v is not None for v in owned_memory.values()) else None
        row = dict(t=started, interval_start=self.last_t, unix_time=time.time(),
                   cpu_total_percent=sum(cores)/len(cores), cpu_per_core_percent=cores,
                   cpu_busiest_core_percent=max(cores), process_cpu_one_core_percent=tree_pct,
                   process_cpu_machine_percent=tree_pct/len(cores) if tree_pct is not None else None,
                   process_interval_complete=bool(snapshots) and snapshots.keys() == self.previous.keys(),
                   process_rss_bytes=rss if process_rows else None, processes=process_rows,
                   system_memory_used_bytes=vm.total-vm.available, system_memory_percent=vm.percent,
                   gpu_util_percent=util.gpu if util else None,
                   gpu_memory_controller_percent=util.memory if util else None,
                   gpu_memory_used_bytes=memory.used if memory else None,
                   process_gpu_memory_bytes=tree_gpu_memory,
                   gpu_encoder_percent=encoder[0] if encoder else None,
                   gpu_encoder_window_us=encoder[1] if encoder else None,
                   gpu_decoder_percent=decoder[0] if decoder else None,
                   gpu_decoder_window_us=decoder[1] if decoder else None,
                   gpu_processes=gpu_processes, background_gpu_pids=sorted(seen-owned), errors=errors)
        row['sampler_call_seconds'] = time.perf_counter()-started
        self.previous, self.last_t = snapshots, started
        return row


def wait_idle(directory, label):
    """Bounded historical idle criterion; no warmup or forward is performed."""
    nv.nvmlInit()
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(0)
        rows, good = [], 0
        for _ in range(30):
            value = nv.nvmlDeviceGetUtilizationRates(handle).gpu
            rows.append(dict(t=time.perf_counter(), gpu_util_percent=value))
            good = good+1 if value <= 10 else 0
            if good >= 10:
                save(Path(directory)/f'{label}_idle.json', rows)
                return
            time.sleep(1)
        save(Path(directory)/f'{label}_idle.json', rows)
        raise RuntimeError('GPU idle criterion not met within 30 observations; no automatic retry')
    finally:
        nv.nvmlShutdown()

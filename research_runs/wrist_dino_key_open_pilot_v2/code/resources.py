"""Measured process peaks. RSS is not whole-cgroup memory or Slurm MaxRSS."""
from __future__ import annotations
import os
import resource
import time

def snapshot():
    r = resource.getrusage(resource.RUSAGE_SELF)
    value = {"process_rss_hwm_bytes": int(r.ru_maxrss * 1024),
             "process_cpu_seconds": r.ru_utime + r.ru_stime,
             "process_id": os.getpid(), "sample_time_ns": time.time_ns()}
    import torch
    if torch.cuda.is_initialized():
        value.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                     cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                     cuda_current_allocated_bytes=int(torch.cuda.memory_allocated()))
    return value

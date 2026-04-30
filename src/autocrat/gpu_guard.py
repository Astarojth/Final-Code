from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStatus:
    name: str
    total_mib: int
    used_mib: int
    util_percent: int

    @property
    def used_ratio(self) -> float:
        if self.total_mib <= 0:
            return 0.0
        return self.used_mib / float(self.total_mib)


def parse_nvidia_smi_row(row: str) -> GpuStatus:
    # Expected format:
    # GPU_NAME, TOTAL_MIB, USED_MIB, UTIL_PERCENT
    fields = [part.strip() for part in row.split(",")]
    if len(fields) != 4:
        raise ValueError(f"Unexpected nvidia-smi row: {row}")
    name = fields[0]
    total = int(re.sub(r"[^\d]", "", fields[1]) or "0")
    used = int(re.sub(r"[^\d]", "", fields[2]) or "0")
    util = int(re.sub(r"[^\d]", "", fields[3]) or "0")
    return GpuStatus(name=name, total_mib=total, used_mib=used, util_percent=util)


def query_gpu_status() -> GpuStatus | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if not output:
        return None
    first_line = output.splitlines()[0]
    return parse_nvidia_smi_row(first_line)


def is_gpu_busy(
    status: GpuStatus,
    max_used_ratio: float = 0.75,
    max_util_percent: int = 40,
) -> bool:
    return status.used_ratio > max_used_ratio or status.util_percent > max_util_percent

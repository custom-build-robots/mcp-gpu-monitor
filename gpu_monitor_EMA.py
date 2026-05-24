"""
GPU Monitor MCP Server
======================

A lightweight Model Context Protocol (MCP) server that exposes NVIDIA GPU
metrics — utilization, memory, temperature, power draw — over a standard
protocol. 
Multi-GPU support with EMA smoothing.

Author:   Ingmar Stapel
Blog:     https://ai-box.eu
Version:  1.0.0
Date:     2026-05-23
License:  MIT

This small tool was developed with the help of AI.
"""
import pynvml
from fastmcp import FastMCP



# --- EMA smoothing ---
class EMASmoother:
    """
    Exponential Moving Average for individual, named measurement series.
    A separate smoothed value is kept per key (e.g. "gpu0_compute").
    """
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._values: dict[str, float] = {}
    
    def update(self, key: str, raw_value: float) -> float:
        """Records a new raw value and returns the smoothed result."""
        if key not in self._values:
            # First value for this key → take it as-is
            self._values[key] = float(raw_value)
        else:
            self._values[key] = (
                self.alpha * float(raw_value)
                + (1.0 - self.alpha) * self._values[key]
            )
        return self._values[key]



pynvml.nvmlInit()
GPU_COUNT = pynvml.nvmlDeviceGetCount()

smoother = EMASmoother(alpha=0.3)

print(f"Server bereit, {GPU_COUNT} GPU(s) erkannt, EMA-Alpha = {smoother.alpha}")

mcp = FastMCP(name="GPU Monitor")

# Helper: return GPU name as a clean string
def _gpu_name(handle) -> str:
    name = pynvml.nvmlDeviceGetName(handle)
    # Older pynvml versions return bytes, newer ones return strings
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name


@mcp.tool()
def get_gpu_count() -> int:
    """Returns the number of available GPUs."""
    return GPU_COUNT


@mcp.tool()
def get_gpu_info(gpu_id: int = 0) -> dict:
    """
    Returns name and ID of the specified GPU.

    :param gpu_id: Index of the GPU (0 to GPU_COUNT - 1)
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    return {
        "gpu_id": gpu_id,
        "name": _gpu_name(handle),
    }


@mcp.tool()
def get_gpu_utilization(gpu_id: int = 0, smoothed: bool = True) -> dict:
    """
    Returns the compute and memory I/O utilization of a GPU in percent.

    :param gpu_id: Index of the GPU
    :param smoothed: If True (default), the value is EMA-smoothed.
                     If False, the raw value is returned.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    
    compute_raw = float(util.gpu)
    memory_raw = float(util.memory)
    
    if smoothed:
        compute = smoother.update(f"gpu{gpu_id}_compute", compute_raw)
        memory = smoother.update(f"gpu{gpu_id}_memory", memory_raw)
    else:
        compute = compute_raw
        memory = memory_raw
    
    return {
        "gpu_id": gpu_id,
        "compute_percent": round(compute, 1),
        "memory_io_percent": round(memory, 1),
        "smoothed": smoothed,
    }


@mcp.tool()
def get_gpu_memory(gpu_id: int = 0) -> dict:
    """
    Returns the VRAM usage of a GPU in MB and percent.

    :param gpu_id: Index of the GPU
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    return {
        "gpu_id": gpu_id,
        "used_mb": round(mem.used / 1024**2, 0),
        "total_mb": round(mem.total / 1024**2, 0),
        "used_percent": round((mem.used / mem.total) * 100, 1),
    }


@mcp.tool()
def get_gpu_thermals(gpu_id: int = 0) -> dict:
    """
    Returns temperature (°C) and power draw (W) of a GPU.

    :param gpu_id: Index of the GPU
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
    
    return {
        "gpu_id": gpu_id,
        "temperature_c": temp,
        "power_w": round(power_mw / 1000, 1),
    }

@mcp.tool()
def get_all_gpus_summary() -> list:
    """
    Returns a compact status overview of ALL GPUs in a single call.
    Ideal for embedded clients that don't want to call each tool
    individually. Compute utilization is EMA-smoothed.
    """
    result = []
    for i in range(GPU_COUNT):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        
        # Compute-Auslastung glätten, Memory-Werte sind stabil genug
        compute_smoothed = smoother.update(f"gpu{i}_compute", float(util.gpu))
        
        result.append({
            "gpu_id": i,
            "compute_percent": round(compute_smoothed, 1),
            "memory_used_percent": round((mem.used / mem.total) * 100, 1),
            "temperature_c": temp,
            "power_w": round(power_mw / 1000, 1),
        })
    return result

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8765)

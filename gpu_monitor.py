"""
GPU Monitor MCP Server
======================

A lightweight Model Context Protocol (MCP) server that exposes NVIDIA GPU
metrics — utilization, memory, temperature, power draw — over a standard
protocol. 
Multi-GPU support with NO EMA smoothing.

Author:   Ingmar Stapel
Blog:     https://ai-box.eu
Version:  1.0.0
Date:     2026-05-23
License:  MIT

This small tool was developed with the help of AI.
"""
import pynvml
from fastmcp import FastMCP

pynvml.nvmlInit()
GPU_COUNT = pynvml.nvmlDeviceGetCount()
print(f"Server bereit, {GPU_COUNT} GPU(s) erkannt")

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
def get_gpu_utilization(gpu_id: int = 0) -> dict:
    """
    Returns the compute and memory I/O utilization of a GPU in percent.

    :param gpu_id: Index of the GPU
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_id": gpu_id,
        "compute_percent": float(util.gpu),
        "memory_io_percent": float(util.memory),
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


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8765)
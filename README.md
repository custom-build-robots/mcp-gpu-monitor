# mcp-gpu-monitor

> Lightweight MCP server for NVIDIA multi-GPU monitoring. Built with Python, FastMCP and pynvml.

A small but useful **Model Context Protocol** server that exposes NVIDIA GPU metrics utilization, memory, temperature, power draw over a standard protocol. Any MCP-compatible client can read the data: Claude Desktop, NeMo Agent Toolkit, LangChain agents, ESP32 microcontrollers, custom scripts.

![MCP Server - GPU Monitor architecture](https://ai-box.eu/wp-content/uploads/2026/05/MCP_Server_GPU_load.jpg)

## What's inside

- Six MCP tools covering utilization, memory, temperature, power, and a single-call aggregated summary
- EMA-smoothed values for jitter-free LED ring or chart displays
- Multi-GPU support
- Network-accessible via SSE transport
- systemd service template for permanent operation

## 📖 Full walkthrough on my blog

This README sticks to the essentials. The complete story — the *why*, design decisions, step-by-step build, and the pitfalls I hit along the way lives on my blog:

**→ [An MCP Server for Multi-GPU Monitoring — Step by Step with Python, pynvml and EMA Smoothing](https://ai-box.eu/en/news/an-mcp-server-for-multi-gpu-monitoring-step-by-step-with-python-pynvml-and-ema-smoothing/2354/)**

If you want to understand *how* and *why* this is built the way it is, head over there.

## Quick start

```bash
git clone https://github.com/<your-user>/mcp-gpu-monitor.git
cd mcp-gpu-monitor
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python gpu_monitor.py
```

The server starts on `0.0.0.0:8765`. Test it with the official MCP Inspector:

```bash
npx @modelcontextprotocol/inspector
```

Connect via **SSE** to `http://<your-host>:8765/sse`.

## License

MIT — see [LICENSE](LICENSE).

## Author

Ingmar Stapel — [ai-box.eu](https://ai-box.eu/) · author of *Roboter-Autos mit dem ESP32* (Rheinwerk Verlag).

"""Write the machine specs and library versions used for the run (the assignment asks for this)."""
import platform

import faiss
import numpy
import pandas

import config

cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
mem_kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
text = f"""CPU: {cpu}
Logical cores: {faiss.omp_get_max_threads()}
RAM: {mem_kb / 1e6:.0f} GB
OS: {platform.platform()}
Python: {platform.python_version()}
faiss-cpu: {faiss.__version__}
numpy: {numpy.__version__}
pandas: {pandas.__version__}
Timing: one query per call, FAISS limited to 1 thread, median of {config.REPEATS} repeats
"""
(config.RES_DIR / "environment.txt").write_text(text)
print(text)

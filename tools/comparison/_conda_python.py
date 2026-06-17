"""Resolve sibling conda-env python executables without hardcoding machine paths.

run_*_comparison.py 脚本需要用 subprocess 跨 conda 环境调用另一个解释器（如
scanvi311 调 sandbox310，或反过来）。直接硬编码 "D:/setup/anaconda/envs/..."
在换机器/换 OS（本机 Windows ↔ 服务器 Linux）时就会失效，这里改为相对当前
正在运行的解释器所在的 conda envs 目录去定位，同机器上换路径、跨平台都不用改。
"""
from __future__ import annotations

import sys
from pathlib import Path


def conda_python(env_name: str) -> str:
    """Return the python executable of conda env ``env_name``.

    定位方式：从当前解释器路径向上找名为 "envs" 的目录，再拼接
    "<env_name>/python.exe"（Windows）或 "<env_name>/bin/python"（Linux/macOS）。
    找不到时回退为环境名本身，交给 PATH / 调用方处理。
    """
    current = Path(sys.executable).resolve()
    envs_dir = next((p for p in current.parents if p.name == "envs"), None)
    if envs_dir is not None:
        for candidate in (envs_dir / env_name / "python.exe",
                           envs_dir / env_name / "bin" / "python"):
            if candidate.exists():
                return str(candidate)
    return env_name

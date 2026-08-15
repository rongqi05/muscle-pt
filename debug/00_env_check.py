"""Level 0: 环境诊断脚本。

检查 Python / PyTorch / CUDA / Isaac Lab / Isaac Sim / MuJoCo / GPU,
以及仓库关键依赖(protomotions, isaac_utils, poselib)是否可导入。

注意:
  - `isaacsim` / `pxr` 只有在 SimulationApp 启动后才能 import,
    因此本脚本不 import 它们,而是通过 importlib.metadata 读取版本。
  - 运行方式(在仓库根目录):
      PYTHONPATH=. python debug/00_env_check.py
"""

import importlib.metadata as md
import platform
import subprocess
import sys


def get_dist_version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "NOT INSTALLED"


def try_import(name: str) -> str:
    try:
        mod = __import__(name)
        return f"OK ({getattr(mod, '__version__', getattr(mod, '__file__', '?'))})"
    except Exception as e:  # noqa: BLE001
        return f"FAIL: {type(e).__name__}: {e}"


def main() -> None:
    print("=" * 70)
    print("Level 0 — 环境诊断")
    print("=" * 70)
    print(f"Python            : {sys.version.split()[0]}  ({sys.executable})")
    print(f"OS                : {platform.system()} {platform.release()}")

    import torch
    print(f"PyTorch           : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version      : {torch.version.cuda}")
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA version      : N/A (GPU 不可用)")

    # GPU 名称(直接查 nvidia-smi,失败不致命)
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"nvidia-smi        : {gpu}")
    except Exception:  # noqa: BLE001
        print("nvidia-smi        : 不可用")

    print("-" * 70)
    print(f"isaaclab          : {get_dist_version('isaaclab')}")
    print(f"isaacsim          : {get_dist_version('isaacsim')}")
    print(f"isaacsim-core     : {get_dist_version('isaacsim-core')}")
    print(f"mujoco            : {get_dist_version('mujoco')}")
    print(f"hydra-core        : {get_dist_version('hydra-core')}")
    print(f"lightning         : {get_dist_version('lightning')}")
    print(f"pydantic          : {get_dist_version('pydantic')}")
    print(f"trimesh           : {get_dist_version('trimesh')}")
    print(f"rtree             : {get_dist_version('rtree')}")

    print("-" * 70)
    print(f"protomotions      : {try_import('protomotions')}")
    print(f"isaac_utils       : {try_import('isaac_utils')}")
    print(f"poselib           : {try_import('poselib')}")
    print(f"torch             : {try_import('torch')}")
    print(f"mujoco            : {try_import('mujoco')}")
    print(f"hydra             : {try_import('hydra')}")
    print(f"lightning         : {try_import('lightning')}")

    print("-" * 70)
    # 关键:确认上游 README 的安装步骤是否有问题
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    has_setup = os.path.exists(os.path.join(root, "setup.py")) or os.path.exists(
        os.path.join(root, "pyproject.toml")
    )
    print(f"repo root         : {root}")
    print(f"setup.py/pyproject: {'存在' if has_setup else '不存在 (pip install -e . 会失败!)'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

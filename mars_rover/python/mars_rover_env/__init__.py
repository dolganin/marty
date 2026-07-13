import os
import sys
from pathlib import Path

# MinGW builds need their runtime DLL directory explicitly registered on Python 3.8+.
if os.name == "nt":
    _dll_dir = Path(sys.prefix) / "Lib" / "site-packages" / "_mars_rover_dlls"
    if _dll_dir.is_dir():
        os.add_dll_directory(str(_dll_dir))

from .envs.mars_rover_env import MarsRoverEnv
from .envs.mars_rover_vec_env import MarsRoverVecEnv

__all__ = ["MarsRoverEnv", "MarsRoverVecEnv"]

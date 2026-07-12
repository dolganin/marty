from __future__ import annotations

from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    c_opts = {
        "unix": ["-O3", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic"],
    }

    def build_extensions(self):
        import pybind11

        for ext in self.extensions:
            ext.include_dirs.append(str(Path("cpp/include").resolve()))
            ext.include_dirs.append(pybind11.get_include())
            ext.extra_compile_args = self.c_opts.get(self.compiler.compiler_type, [])
        super().build_extensions()


sources = [
    "cpp/bindings/pybind_module.cpp",
    "cpp/src/batch_env.cpp",
    "cpp/src/env.cpp",
    "cpp/src/mechanics.cpp",
    "cpp/src/physics.cpp",
    "cpp/src/renderer.cpp",
    "cpp/src/reward.cpp",
    "cpp/src/rover_rig.cpp",
    "cpp/src/terrain.cpp",
]

setup(
    ext_modules=[
        Extension(
            "_mars_rover_cpp",
            sources=sources,
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
)

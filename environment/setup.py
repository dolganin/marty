from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        import pybind11

        build_type = os.environ.get("MARS_ROVER_BUILD_TYPE", "Release").lower()
        debug = build_type in {"debug", "relwithdebinfo"}
        use_openmp = os.environ.get("MARS_ROVER_OPENMP", "1") not in {"0", "false", "False"}
        bank_include = os.environ.get("MARS_ROVER_BANK_INCLUDE")
        if bank_include:
            # A run bank is an external generated header. Distutils does not
            # reliably include such headers in its incremental dependency
            # graph, especially for editable Windows builds, so stale .obj/.pyd
            # files can otherwise survive a bank switch.
            self.force = True
        for ext in self.extensions:
            if bank_include:
                ext.include_dirs.insert(0, str(Path(bank_include).resolve()))
            ext.include_dirs.append(str(Path("cpp/include").resolve()))
            ext.include_dirs.append(pybind11.get_include())
            if self.compiler.compiler_type == "msvc":
                ext.extra_compile_args = [
                    "/std:c++20", "/EHsc", "/W4", "/permissive-",
                    *(["/Od", "/Zi"] if debug else ["/O2"]),
                ]
                if debug:
                    ext.extra_link_args = ["/DEBUG"]
                else:
                    ext.extra_compile_args.append("/fp:fast")
                if use_openmp:
                    ext.extra_compile_args.append("/openmp")
                    ext.define_macros.append(("MARS_ROVER_HAS_OPENMP", "1"))
            else:
                ext.extra_compile_args = [
                    "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
                    *(["-O0", "-g3", "-fno-omit-frame-pointer"] if debug else ["-O3"]),
                ]
                if os.name == "nt":
                    ext.extra_compile_args.append("-Wa,-mbig-obj")
                if not debug:
                    ext.extra_compile_args.append("-ffast-math")
                if use_openmp:
                    ext.extra_compile_args.append("-fopenmp")
                    ext.extra_link_args.append("-fopenmp")
                    ext.define_macros.append(("MARS_ROVER_HAS_OPENMP", "1"))
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
    ext_modules=[] if os.environ.get("MARS_ROVER_SKIP_NATIVE") == "1" else [
        Extension(
            "_mars_rover_cpp",
            sources=sources,
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
)

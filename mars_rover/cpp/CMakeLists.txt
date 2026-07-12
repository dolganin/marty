cmake_minimum_required(VERSION 3.20)
project(mars_rover LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mars_rover_core
  src/batch_env.cpp
  src/env.cpp
  src/mechanics.cpp
  src/physics.cpp
  src/renderer.cpp
  src/reward.cpp
  src/rover_rig.cpp
  src/terrain.cpp
)

target_include_directories(mars_rover_core PUBLIC include)
target_compile_options(mars_rover_core PRIVATE -O3 -Wall -Wextra -Wpedantic)

find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(_mars_rover_cpp bindings/pybind_module.cpp)
  target_link_libraries(_mars_rover_cpp PRIVATE mars_rover_core)
endif()

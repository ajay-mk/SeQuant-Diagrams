cmake_minimum_required(VERSION 3.24)   # FIND_PACKAGE_ARGS needs >=3.24
project(sq-diagrams CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shareable build: find_package(SeQuant) first (short-circuits to a local build
# when -DCMAKE_PREFIX_PATH=<SeQuant build dir> is passed), else clone+build once.
include(FetchContent)
# SeQuant's build-tree export names Eigen3::Eigen in its link interface but does
# not find_package it; harmless no-op when SeQuant is fetched+built instead.
find_package(Eigen3 CONFIG QUIET)
FetchContent_Declare(
  sequant
  GIT_REPOSITORY https://github.com/ValeevGroup/SeQuant.git
  GIT_TAG master  # ponytail: pin a SHA once green, for reproducibility
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(sequant)

add_executable(sq-diagram-extract sq-diagrams/extract.cpp)
target_link_libraries(sq-diagram-extract PRIVATE SeQuant::SeQuant)
set_target_properties(sq-diagram-extract PROPERTIES CXX_SCAN_FOR_MODULES OFF)

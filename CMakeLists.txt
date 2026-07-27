cmake_minimum_required(VERSION 3.24)
project(SeQuant-Diagrams CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# SeQuant needs Eigen
find_package(Eigen3 CONFIG QUIET)
FetchContent_Declare(
  sequant
  GIT_REPOSITORY https://github.com/ValeevGroup/SeQuant.git
  GIT_TAG 1e033dfc3bf21aaa30eaef9afb02ab382f6d6a07
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(sequant)

add_executable(sq-diagram-topology src/topology.cpp)
target_link_libraries(sq-diagram-topology PRIVATE SeQuant::core SeQuant::mbpt)
set_target_properties(sq-diagram-topology PROPERTIES CXX_SCAN_FOR_MODULES OFF)

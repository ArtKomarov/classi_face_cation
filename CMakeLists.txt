cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(classi_face_cation)

SET(GCC_COVERAGE_COMPILE_FLAGS "-Wall -Wextra -g -fno-elide-constructors -pthread")

#-fsanitize=address -fno-omit-frame-pointer -O2

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

set(HEADER clfc.hpp nets.hpp customdataset.hpp)

add_executable(classi_face_cation main.cpp customdataset.cpp nets.cpp)
target_link_libraries(classi_face_cation "${TORCH_LIBRARIES}")
set_property(TARGET classi_face_cation PROPERTY CXX_STANDARD 14)

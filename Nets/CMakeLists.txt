cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(classi_face_cation)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(classi_face_cation main.cpp nets.cpp customdataset.cpp)
target_link_libraries(classi_face_cation  "${TORCH_LIBRARIES}")
set_property(TARGET classi_face_cation PROPERTY CXX_STANDARD 14)

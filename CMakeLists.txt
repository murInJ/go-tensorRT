cmake_minimum_required(VERSION 3.13)
project(TensorRTLib)
set(CMAKE_CXX_STANDARD 11)
# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add TensorRT8
include_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/usr/src/tensorrt/samples/common/)
set(TENSORRT_LIB_PATH "/usr/lib/x86_64-linux-gnu/")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

# Find CUDA
find_package(CUDA 12.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Build TensorRTLib as a SHARED library
add_library(TensorRT SHARED
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorRTBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CTensorRT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensorRT.cpp
)

# Link TensorRTLib with required libraries
target_link_libraries(TensorRT ${LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

# Rename the generated SHARED library to libTensorRT.so
set_target_properties(TensorRT PROPERTIES OUTPUT_NAME TensorRT)

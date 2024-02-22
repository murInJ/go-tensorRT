# cmake_minimum_required(VERSION 3.13)
# project(TensorRTLib)
# set(CMAKE_CXX_STANDARD 11)
# # Find OpenCV
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# # Add TensorRT8
# include_directories(/usr/lib/x86_64-linux-gnu/)
# include_directories(/usr/src/tensorrt/samples/common/)
# set(TENSORRT_LIB_PATH "/usr/lib/x86_64-linux-gnu/")
# file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

# # Find CUDA
# find_package(CUDA 12.1 REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})

# # Build TensorRTLib as a SHARED library
# add_library(TensorRT SHARED
#     # ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorRTBuffer.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/CTensorRT.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/tensorRT.cpp
# )

# # Link TensorRTLib with required libraries
# target_link_libraries(TensorRT ${LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

# # Rename the generated SHARED library to libTensorRT.so
# set_target_properties(TensorRT PROPERTIES OUTPUT_NAME TensorRT)


cmake_minimum_required(VERSION 3.13)
project(TensorRTLib)
set(CMAKE_CXX_STANDARD 11)
# Find OpenCV
set(OpenCV_DIR D:/packages/c/opencv4.7/build/x64/vc16/lib)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add TensorRT8
include_directories(D:/packages/c/TensorRT-8.5.1.7/include D:/diannaozhongde/zuoye/daizuo/dai/TensorRT/samples/common)
include_directories(/usr/src/tensorrt/samples/common/)
link_directories(D:/packages/c/TensorRT-8.5.1.7/lib)
set(TENSORRT_LIB_PATH "D:/packages/c/TensorRT-8.5.1.7/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Build TensorRTLib as a SHARED library
add_library(TensorRT SHARED
#add_executable(TensorRT 
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/TensorRTBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CTensorRT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensorRT.cpp
)

# Link TensorRTLib with required libraries
target_link_libraries(TensorRT ${LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} nvonnxparser nvinfer nvparsers)

# Rename the generated SHARED library to libTensorRT.so
set_target_properties(TensorRT PROPERTIES OUTPUT_NAME TensorRT)

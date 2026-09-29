cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0079 NEW)

project(infer)
add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
# TODO(Call for PR): make cmake compatible with Windows
find_package(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_120,code=sm_120
    -gencode arch=compute_120,code=compute_120
)
include(../cmake/modules/libnvinferConfig.cmake)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${nvinfer_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_library(infer  ${SRCS})
target_link_libraries(infer ${nvinfer_LIBRARY} cudart nvonnxparser ${OpenCV_LIBS})
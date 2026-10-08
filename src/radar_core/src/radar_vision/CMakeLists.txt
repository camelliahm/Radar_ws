# 指定 CMake 最低版本为 3.8
cmake_minimum_required(VERSION 3.8)
# 定义项目名称
project(radar_vision)

# 要求编译器必须支持指定的 C++ 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 指定 C++ 标准为 20
set(CMAKE_CXX_STANDARD 20)

# 生成位置无关代码，用于共享库
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 添加全局头文件搜索路径
include_directories(  
  "detect/include"
  "resolve/include"
  "calibrate/include"
)

# 查找 ROS 2 ament_cmake_auto 包
find_package(ament_cmake_auto REQUIRED)
# 自动查找 package.xml 中声明的构建依赖
ament_auto_find_build_dependencies()


# 设置 CUDA 运行时为动态链接
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# 构建类型不再硬编码：交给 colcon / 命令行的 -DCMAKE_BUILD_TYPE 决定（本项目约定为 Release）
# 仅在未指定时回退为 Release，避免默认以无优化（-g、无 -O、断言开启）方式编译
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 指定 CUDA 编译器路径
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# 添加 CUDA 头文件路径
include_directories(/usr/local/cuda/include)
# 添加 CUDA 库路径
link_directories(/usr/local/cuda/lib64)

# 再次查找 ament_cmake_auto（重复）
find_package(ament_cmake_auto REQUIRED)
# 再次自动查找构建依赖（重复）
ament_auto_find_build_dependencies()

# 添加 utils 子目录
add_subdirectory(utils)
# 添加 utils 头文件路径
include_directories(${PROJECT_SOURCE_DIR}/utils/include)

# 定义雷达检测库名称
set(RADAR_DETECT "radar_detect")
# 递归查找 detect/src 下所有 .cpp 文件
file(GLOB_RECURSE RADAR_DETECT_SRCS "detect/src/*.cpp")
# 创建共享库 radar_detect
ament_auto_add_library(${RADAR_DETECT} SHARED
    ${RADAR_DETECT_SRCS}
)
# 查找 CUDA 包（旧式）
find_package(CUDA)
# 包含 TensorRT 的 CMake 配置文件
include(${PROJECT_SOURCE_DIR}/cmake/modules/libnvinferConfig.cmake)
# 打印 CUDA 和 TensorRT 版本信息
message("Using Cuda Version: ${CUDA_VERSION} and TensorRT Version: ${nvinfer_VERSION}")
# 添加编译定义 use_tensorrt
add_definitions(-D use_tensorrt)
# 启用 CUDA 语言支持
enable_language(CUDA)
# 添加 CUDA 头文件路径
include_directories(${CUDA_INCLUDE_DIRS})
# 添加 TensorRT 头文件路径
include_directories(${nvinfer_INCLUDE_DIRS})

# 添加 infer 子目录
add_subdirectory(infer)
# 添加 infer 头文件路径
include_directories(${PROJECT_SOURCE_DIR}/infer/include)

# 链接 radar_detect 所需的库
target_link_libraries(${RADAR_DETECT}
    utils
    infer 
    ${nvinfer_LIBRARY} 
    cudart 
    nvonnxparser 
    fmt 
    ${PCL_LIBRARIES}
    )
# 注册 ROS 2 组件节点
rclcpp_components_register_node(${RADAR_DETECT}
    PLUGIN "radar_core::Detect"
    EXECUTABLE ${RADAR_DETECT}_node
)

# 定义雷达解算库名称
set(RADAR_RESOLVE "radar_resolve")
# 递归查找 resolve/src 下所有 .cpp 文件
file(GLOB_RECURSE RADAR_RESOLVE_SRCS "resolve/src/*.cpp")

# 创建共享库 radar_resolve
ament_auto_add_library(${RADAR_RESOLVE} SHARED
    ${RADAR_RESOLVE_SRCS}
)
# 注册 ROS 2 组件节点
rclcpp_components_register_node(${RADAR_RESOLVE}
    PLUGIN "radar_core::Resolve"
    EXECUTABLE ${RADAR_RESOLVE}_node
)

# 定义雷达标定库名称
set(RADAR_CALIBRATE "radar_calibrate")
# 递归查找 calibrate/src 下所有 .cpp 文件
file(GLOB_RECURSE RADAR_CALIBRATE_SRCS "calibrate/src/*.cpp")
# 创建共享库 radar_calibrate
ament_auto_add_library(${RADAR_CALIBRATE} SHARED
    ${RADAR_CALIBRATE_SRCS}
)
# 注册 ROS 2 组件节点
rclcpp_components_register_node(${RADAR_CALIBRATE}
    PLUGIN "radar_core::Calibrate"
    EXECUTABLE ${RADAR_CALIBRATE}_node
)


# 自动安装包，并将 launch 目录安装到 share
ament_auto_package(
    INSTALL_TO_SHARE
    launch
)

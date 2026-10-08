# AGENTS.md

## 项目概述

本目录是 RoboMaster 雷达站的 ROS 2 工作空间，面向激光雷达、单目工业相机、目标识别和多传感器后融合。主要能力包括：

- Livox 点云采集、GICP 地图配准、动态点云提取、欧几里得聚类和目标跟踪；
- 海康工业相机图像采集；
- 基于 YOLO/分类网络的机器人与装甲板识别，以及透视变换定位；
- 使用卡尔曼滤波匹配并融合雷达与视觉结果；
- rosbag 回放、地图调试、Foxglove 可视化和 PyQt5 控制界面。

工作空间根目录 `radar_ws/` 本身就是一个 Git 仓库（`origin` = `https://github.com/camelliahm/Radar_ws.git`），`src/radar_core/` 只是其中的雷达主工程目录，不再有独立的嵌套仓库。修改前先在**根目录**检查工作树，保留用户已有改动。

## 技术栈与依赖

- 平台：Ubuntu 24.04、ROS 2 Jazzy，使用 `colcon`、CMake 和 `ament_cmake`/`ament_cmake_auto`；
- 核心语言：C++、CUDA C++；启动文件与工具使用 Python 3；根目录 UI 使用 PyQt5；
- C++ 标准因包而异：Livox 驱动和 `rosbag_player` 为 C++14，`hik_camera` 与接口/推理子库主要为 C++17，`radar_vision` 顶层为 C++20；不要在未验证全部包的情况下统一修改标准；
- ROS 组件：`rclcpp`、`rclcpp_components`、`sensor_msgs`、`geometry_msgs`、`visualization_msgs`、`tf2`、`cv_bridge`、`image_transport`、`rosbag2` 和自定义 `rosidl` 消息；
- 点云与视觉：PCL 1.14、OpenCV 4.6；
- 推理：CUDA、cuDNN、TensorRT（工作空间说明记录为 CUDA 13.4、TensorRT 11.3；README 要求 TensorRT 10）；
- 硬件 SDK：Livox SDK 1；海康 MVS SDK（CMake 期望头文件在 `/opt/MVS/include`，库在 `/opt/MVS/lib/64` 或 `/opt/MVS/lib/aarch64`）；
- 其他原生依赖：APR、Boost、fmt、Foxglove Bridge、Nav2 map server；
- UI Python 依赖：`rclpy`、`PyQt5`、`cv_bridge`、OpenCV/NumPy（由图像转换间接使用）。仓库中没有 Python requirements/pyproject 文件。

`livox_sdk_vendor` 在系统找不到 Livox SDK 时会通过 CMake `ExternalProject` 从 GitHub 获取 v2.3.0，因此首次构建可能需要网络；项目说明仍将运行时驱动描述为基于 Livox SDK 1。

## 构建（Build）

所有工作空间命令均从仓库根目录执行：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1
source install/setup.bash
```

`src/radar_core/build.sh` 执行相同的 `colcon build` 命令，也可以运行：

```bash
bash src/radar_core/build.sh
```

为 VS Code 更新编译数据库：

```bash
ln -sf /home/camelliahm/radar_ws/build/compile_commands.json /home/camelliahm/radar_ws/compile_commands.json
```

构建视觉推理前需确认 `src/radar_core/src/radar_vision/infer/CMakeLists.txt` 中的 `CMAKE_CUDA_ARCHITECTURES` 与目标 GPU 一致；当前值为 `120`。`radar_vision/CMakeLists.txt` 还固定使用 `/usr/local/cuda/bin/nvcc`。

## 测试与验证（Test）

仓库未发现独立的 gtest/pytest 测试源码。若要运行各包已注册的 ament lint 和 rosidl 接口测试：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

README 提供的集成/冒烟测试如下；它们依赖已构建并 source 的工作空间，部分还依赖真实硬件、模型或 rosbag：

```bash
# rosbag 驱动视觉链路
ros2 launch radar_vision run_rosbag.launch.py

# 激光雷达识别链路
ros2 launch dynamic_cloud lidar.launch.py

# 地图结果可视化
ros2 run debug_map debug_map

# Livox 驱动
ros2 launch livox_ros2_driver livox_lidar_launch.py

# 相机外参标定（交互式）
ros2 launch radar_vision calib_rosbag.launch.py
```

注意：`run_rosbag.launch.py` 和 `calib_rosbag.launch.py` 里的 rosbag 路径是硬编码绝对路径（当前为 `~/radar_ws/bags/merged_bag_1.db3`），换机器或换数据包时必须修改。`radar.launch.py` 还引用当前工作空间未提供的 `radar_vision::NodeCamera`、`radar_vision::NodeDebug` 和 `databag_tool`，不能假定该 launch 文件开箱即用。

海康相机节点和 UI 可分别这样启动：

```bash
ros2 run hik_camera hik_camera_node --ros-args -p exposure_time:=50000.0 -p gain:=10.0
python3 camera_ui.py
```

UI 订阅 `/image`，并通过 `/hik_camera/set_parameters` 修改曝光和增益。若使用项目记录的虚拟环境，先执行 `source ~/radar_venv/bin/activate`。

## 目录结构

```text
radar_ws/                            # Git 仓库根（origin: camelliahm/Radar_ws）
├── .gitignore                        # 忽略 build/ install/ log/ bags/ *.engine 等
├── AGENTS.md                         # AI 协作上下文（本文件）
├── WORKSPACE.md                      # 本机工作空间、硬件和使用说明
├── camera_ui.py                      # PyQt5 + rclpy 雷达站控制界面
├── compile_commands.json             # 指向 build 中编译数据库的软链接（已忽略）
├── .vscode/                          # VS Code C++/CMake 与 git 配置
├── bags/                             # rosbag 数据包（已忽略，约 10 GB）
├── build/ install/ log/              # colcon 生成物（已忽略）
└── src/
    ├── hik_camera/                   # 海康 MVS 相机 ROS 2 组件
    │   ├── config/                   # 相机内参与不同镜头配置
    │   ├── include/hik_camera/       # 驱动头文件
    │   └── src/                      # 相机节点与 CameraInfo 发布器
    └── radar_core/                   # 雷达主工程（点云 / 视觉 / 后融合算法）
        ├── README.md                 # 项目说明、话题和运行方法
        ├── build.sh                  # colcon 一键构建脚本
        ├── .clang-format             # C/C++ 格式规则
        ├── config/                   # 检测、标定、场地图与点云配置
        ├── model/
        │   ├── ONNX/                 # 原始检测/分类模型
        │   └── TensorRT/             # 生成的 TensorRT engine（不应提交）
        └── src/
            ├── interface/
            │   ├── base_interface/   # 基础 Point/Polar 自定义消息
            │   └── vision_interface/ # 检测、融合、哨兵通信消息
            ├── lidar/
            │   ├── localization/     # GICP 地图配准与定位
            │   ├── dynamic_cloud/    # 静态背景剔除、动态点云/预警
            │   └── cluster/          # 点云聚类
            ├── fusion/
            │   ├── kalman_filter/    # 目标跟踪及雷视融合
            │   └── debug_map/        # 2D 地图调试与最终消息输出
            ├── radar_vision/
            │   ├── detect/           # 三级神经网络检测节点
            │   ├── infer/            # TensorRT/CUDA 推理实现
            │   ├── resolve/          # 图像结果到场地坐标的解算
            │   ├── calibrate/        # 五点相机外参标定
            │   ├── utils/            # 视觉侧公共工具
            │   ├── launch/           # 视觉、回放与标定启动文件
            │   └── maps/             # Nav2 2D 地图配置
            ├── livox_driver/
            │   ├── livox_interfaces/ # Livox CustomMsg/CustomPoint
            │   ├── livox_ros2_driver/# Livox ROS 2 驱动及 launch/config
            │   └── livox_sdk_vendor/ # Livox SDK CMake vendor 包
            └── utils/
                ├── rosbag_player/     # 进程内 rosbag 播放组件
                └── onnx2trt.py        # ONNX 到 TensorRT 转换辅助脚本
```

`build/`、`install/`、`log/` 和 `__pycache__/` 是生成物，不应手工编辑，也不应作为源码上下文优先读取。

## ROS 2 包清单

`colcon list` 当前识别 13 个包：

- 接口：`base_interface`、`vision_interface`、`livox_interfaces`；
- 视觉/相机：`radar_vision`、`hik_camera`；
- 点云：`localization`、`dynamic_cloud`、`cluster`；
- 融合/调试：`kalman_filter`、`debug_map`；
- 驱动/工具：`livox_ros2_driver`、`livox_sdk_vendor`、`rosbag_player`。

主要数据流为：Livox 点云 → `localization` → `dynamic_cloud` → `cluster` → `kalman_filter`；相机图像 → `radar_vision` 的 `Detect` → `Resolve` → `kalman_filter`/`debug_map`。修改话题名或消息字段时，应同步检查发布者、订阅者、launch 文件及 `vision_interface/msg/`。

## 项目约定

### C/C++ 与 CUDA

- 对 `src/radar_core/` 下的 C/C++/CUDA 改动使用其 `.clang-format`；核心规则为 4 空格、禁止 Tab、76 列、指针靠左、自动排序 include、函数定义的大括号另起一行；
- 现有核心 ROS 节点主要位于 `radar_core` 命名空间，类用 PascalCase，局部变量/方法多用 snake_case；新增代码应遵循所在文件的既有风格；
- ROS 节点优先实现为 `rclcpp_components` 组件，并通过 `RCLCPP_COMPONENTS_REGISTER_NODE` 注册；launch 文件大量依赖进程内通信，修改插件类名时同步更新 CMake 和 launch；
- 头文件放在对应模块的 `include/`，实现放在 `src/`；新增 ROS 依赖同时更新 `package.xml` 和 `CMakeLists.txt`；
- 不要随意改动 `livox_ros2_driver/common/` 中内嵌的 RapidJSON、RapidXML、FastCRC 等第三方源码。

### Python、配置与资源

- Python 没有仓库级 formatter/linter 配置；保持现有 4 空格缩进、snake_case 函数/变量和 PascalCase 类名；
- launch 文件命名采用 `*.launch.py`，可组合节点优先保持 `ComposableNodeContainer` 结构；
- 相机、检测与标定参数使用 YAML/JSON；更改配置键时必须搜索相应 C++ 读取点。OpenCV YAML 带 `%YAML:1.0` 和 `!!opencv-matrix`，不要用会丢失这些标签的通用 YAML 格式化器重写；
- ONNX 模型是源资源；TensorRT `.engine` 与具体 GPU/TensorRT 版本绑定，属于生成物。首次运行可能自动生成 engine，耗时较长；
- 避免提交个人绝对路径。现有 launch 中的硬编码路径属于已知遗留问题，新增路径应通过 launch 参数、ROS 参数或 package share 目录解析；
- `config/RM2025.pcd`、图片和模型为大文件，除非任务明确要求，不要重新编码或改写。

## 修改与交付检查

1. 在根目录运行 `colcon list`，确认新包可被发现；
2. 使用完整构建命令，或用 `colcon build --packages-select <包名>` 快速验证局部改动；接口包变化后还需重建所有下游包；
3. 执行 `colcon test` 和 `colcon test-result --verbose`；没有自动测试覆盖的硬件/算法路径，使用上述 launch 命令做针对性冒烟测试；
4. C/C++ 提交前按 `.clang-format` 格式化仅本次修改的文件，避免对第三方目录或整个仓库做无关格式化；
5. 在**根目录**（仓库根是 `radar_ws/`）检查 `git status` 和 `git diff`，不提交 `build/`、`install/`、`log/`、`bags/`、`*.engine`、缓存或本机路径；
6. 涉及相机、雷达、CUDA 或 TensorRT 的验证结果要注明硬件/驱动环境；缺少硬件时明确说明未执行硬件验证。

## 已知注意事项

- 海康 MVS SDK 和 Livox SDK 属于系统级依赖，CI 或新机器上通常需要单独安装；
- README 的“本机硬件”记录 Livox Mid-70 + 海康 MV-CS060-10UC-PRO（3072×2048），与上游参考硬件 Livox Avia + CH-120-10UC 不同；涉及设备参数时以实际部署硬件为准；
- `detect_params.yaml` 中是相对模型路径；运行目录变化可能影响模型定位，应在改动时验证；
- 相机外参标定会写入 `config/out_matrix.yaml`，运行交互式标定前先确认是否允许覆盖现有标定结果；
- 根仓库用 `radar_ws/.gitignore` 忽略 `build/`、`install/`、`log/`、`bags/`（约 10 GB rosbag）等大目录；新增生成物目录时同步更新该文件，避免误提交；
- 不要在 `~/radar_ws` 根目录用 IDE 的“初始化仓库”功能（仓库已存在，误操作会生成第二个仓库并把 build/bags 全加进去）；
- `detach_from_tdt.sh`、`promote_git_to_root.sh` 是一次性脚本，已被 `.gitignore` 排除，不参与版本管理；
- 不要在相机工作时热插拔 USB；雷达供电建议 12V、持续放电能力至少 5A，并保证共地。

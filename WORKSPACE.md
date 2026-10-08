# RoboMaster 雷达站工作空间说明

本工作空间用于 RoboMaster 雷达站开发，集成激光雷达、工业相机、视觉识别与多传感器融合算法。

---

## 📁 目录结构

    radar_ws/
      src/
        radar_core/                     # 雷达主工程
          config/                       # 全局配置目录
            detect_params.yaml          # 模型路径、检测阈值配置
          model/                        # 模型目录
            ONNX/                       # 原始 ONNX 模型（armor_yolo、classify、RM2025）
            TensorRT/                   # 运行时自动生成的 .engine 引擎
          src/                          # 源码目录
            fusion/                     # 视觉+雷达后融合（卡尔曼滤波）
            interface/                  # 自定义 ROS 消息/服务（base_interface、vision_interface）
            lidar/                      # 点云聚类、配准、定位（localization、dynamic_cloud、cluster）
            livox_driver/               # Livox 雷达驱动（基于 SDK1）
            radar_vision/                 # 视觉识别与 YOLO 推理
            utils/                      # 通用工具（rosbag_player、onnx2trt.py）
          build.sh                      # 一键编译脚本
        hik_camera/                     # 海康工业相机 ROS 2 驱动
          config/                       # 相机内参、外参配置
          include/                      # 头文件
          src/                          # 驱动源码（hik_camera.cpp、info_pub.cpp）
      camera_ui.py                      # 自研 PyQt5 雷达站控制界面
      compile_commands.json             # C++ 编译数据库（指向 build/，供 VS Code 用）
      build/                            # 编译中间产物（自动生成，勿手动编辑）
      install/                          # 编译安装产物（自动生成，勿手动编辑）
      log/                              # 编译日志（自动生成，勿手动编辑）
      WORKSPACE.md                      # 本文件
      AGENTS.md                         # AI 协作上下文说明
      .vscode/                          # VS Code 工作区配置
        c_cpp_properties.json           # C++ 智能提示配置
        settings.json                   # 屏蔽 build/install/log 目录

---

## 🧩 各模块职责

| 目录/文件 | 职责 |
|---|---|
| `radar_core/` | 主雷达站算法：点云处理、视觉识别、后融合、决策 |
| `radar_core/config/` | 全局配置，含模型路径与检测阈值 |
| `radar_core/model/ONNX/` | 原始 ONNX 模型（源资源，需保留） |
| `radar_core/model/TensorRT/` | TensorRT 引擎（生成物，勿提交，与 GPU 绑定） |
| `radar_core/src/fusion/` | 雷视后融合（卡尔曼滤波、目标跟踪、调试地图） |
| `radar_core/src/interface/` | 自定义 ROS 2 消息与服务定义 |
| `radar_core/src/lidar/` | 点云配准、动态点云提取、欧几里得聚类 |
| `radar_core/src/livox_driver/` | Livox Mid-70 雷达 ROS 2 驱动 |
| `radar_core/src/radar_vision/` | 视觉检测（YOLO 推理、坐标解算、标定） |
| `radar_core/src/utils/` | 通用工具（rosbag 播放、ONNX 转 TRT 脚本） |
| `radar_core/build.sh` | 一键编译脚本 |
| `hik_camera/` | 海康相机 ROS 2 驱动，发布 `/image` 话题 |
| `hik_camera/config/` | 相机内参与不同镜头参数 |
| `hik_camera/src/` | 相机节点与 CameraInfo 发布器 |
| `camera_ui.py` | PyQt5 控制界面，显示相机画面、点云 2D 投影、设备状态 |
| `compile_commands.json` | C++ 编译数据库软链接（供 VS Code 智能提示） |
| `AGENTS.md` | AI 协作上下文，描述项目结构、约定与注意事项 |
| `.vscode/` | VS Code 工作区配置（C++ 路径、目录屏蔽） |
| `build/` | 编译中间产物（自动生成，勿手动编辑） |
| `install/` | 编译安装产物（自动生成，勿手动编辑） |
| `log/` | 编译日志（自动生成，勿手动编辑） |

---

## 🔧 Git 仓库

- 仓库根：`~/radar_ws`（`origin` = `https://github.com/camelliahm/Radar_ws.git`），原 `src/radar_core/.git` 已提升到这里
- 已忽略：`build/`、`install/`、`log/`、`bags/`（约 10 GB rosbag）、`*.engine`、`compile_commands.json` 等，规则见根目录 `.gitignore`
- 常用命令：

    cd ~/radar_ws
    git status -sb
    git log --oneline

- 注意：不要在根目录点 IDE 的“初始化仓库”，仓库已存在

---

## 🚀 编译方式

    cd ~/radar_ws
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1

编译完成后，重新建立编译数据库软链接：

    ln -sf ~/radar_ws/build/compile_commands.json ~/radar_ws/compile_commands.json

---

## 🖥️ VS Code 开发环境

- `c_cpp_properties.json`：指向 `compile_commands.json`，提供 C++ 智能提示
- `settings.json`：屏蔽 `build/`、`install/`、`log/` 目录，提升性能

建议使用专属 Profile：`Robomaster radar`。

---

## 📷 相机驱动

- 相机型号：海康 MV-CS060-10UC-PRO
- SDK：MVS V5.0.2 Linux
- 驱动包：`hik_camera`
- 启动命令：

    ros2 run hik_camera hik_camera_node --ros-args -p exposure_time:=50000.0 -p gain:=10.0

- 注意：不要热插拔相机 USB

---

## 📡 雷达驱动

- 雷达型号：Livox Mid-70
- 底层 SDK：Livox SDK1 + Livox SDK2
- 驱动包：`radar_core/src/livox_driver`
- 启动命令：

    cd ~/radar_ws/src/radar_core
    source ~/radar_ws/install/setup.bash
    ros2 launch livox_ros2_driver livox_lidar_launch.py

- 雷达 IP：192.168.96.100
- 电脑 IP：192.168.96.50
- 点云话题：`/livox/lidar`，约 10 Hz

---

## 🎨 UI 界面

    source ~/radar_venv/bin/activate
    cd ~/radar_ws
    source install/setup.bash
    python3 camera_ui.py

功能：相机画面、HSV 调参、曝光/增益、点云 2D 投影、设备状态、累积帧数、显示范围、雷达倒置。

---

## ⚠️ 注意事项

1. 模型首次运行会自动将 ONNX 转为 TensorRT 引擎，需 10~30 分钟，勿中断。
2. CUDA 架构已设为 `sm_120`。
3. 显存限制：`--memPoolSize=workspace:1G`。
4. 雷达供电：12V，BMS ≥ 5A，注意共地。
5. 相机帧率优化：调整曝光至 10000us，检查 `lsusb -t`。

---

## 📌 当前状态

- ✅ Ubuntu 24.04 + ROS 2 Jazzy
- ✅ CUDA 13.4 + TensorRT 11.3
- ✅ Livox SDK1 + SDK2 共存
- ✅ radar_core 编译通过
- ✅ hik_camera 编译通过
- ✅ 相机已出图（3072×2048，约 6.4 FPS）
- ✅ 雷达已接入（192.168.96.100，点云 10Hz）
- ✅ UI 框架完成
- ✅ 视觉推理跑通
- 🟡 相机帧率待优化
- 🟡 融合算法待验证

---

## 🔗 常用命令速查

    # 雷达驱动
    cd ~/radar_ws/src/radar_core
    source ~/radar_ws/install/setup.bash
    ros2 launch livox_ros2_driver livox_lidar_launch.py

    # 相机驱动
    cd ~/radar_ws
    source install/setup.bash
    ros2 run hik_camera hik_camera_node --ros-args -p exposure_time:=50000.0 -p gain:=10.0

    # 视觉推理
    cd ~/radar_ws/src/radar_core
    source ~/radar_ws/install/setup.bash
    ros2 run radar_vision radar_detect_node --ros-args -r camera_image:=/image

    # UI
    source ~/radar_venv/bin/activate
    cd ~/radar_ws
    source install/setup.bash
    python3 camera_ui.py
<div align="center">

# Radar Core

> RoboMaster 雷达站：激光雷达 + 单目相机目标识别与多传感器后融合（ROS 2 Jazzy）

<a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow"></a>

</div>

<br>

--------

<br>

<div align="left">

# 版本记录

**v1.0**
- 初始版本：雷达点云处理（GICP 配准 / 动态点云 / 聚类）+ 单目相机识别解算 + 雷视后融合
- 本机适配：CUDA 架构 `sm_120`、TensorRT 引擎路径、rosbag 路径、相机替换为海康 MV-CS060-10UC-PRO

# 项目介绍

本项目通过激光雷达和单目相机的目标检测，进行传感器后融合，实现了传感器之间的完全解耦合，避免了联合标定带来的误差，同时开发难度不随传感器数量增加而增加。

**如果没有激光雷达，也可以只使用本项目的单目相机方案。**

<p align="center">
  <img src=".github/icp.png" alt="配准效果"  width="60%">
  <br>
  <em>图1：点云配准效果</em>
</p>

## 项目优势
- 1.即插即用，不依赖联合标定，脱离空间(机械结构)上的限制
- 2.不依赖相机和雷达之间的帧间匹配，脱离时间上的限制
- 3.直接使用直角坐标系的信息，更加直观
- 4.三层神经网络实现了更好的鲁棒性和可修复性，极大地降低了模型训练和数据集整理的难度和时间。
- 5.低耦合，易于维护和扩展
- 6.雷达全自动配准，节约3分钟部署时间

## 本机硬件

- 激光雷达 Livox Mid-70
- 单目相机 海康 MV-CS060-10UC-PRO（3072×2048）
- 系统 Ubuntu 24.04 + ROS 2 Jazzy
- 推理 CUDA 13.4 + TensorRT 11.3，CUDA 架构 `sm_120`

## 项目结构说明

**本项目提供了除串口、相机驱动、模型训练外雷达站的全部功能**

![项目结构](.github/processon.png)

### 单目相机

- 五点标定(键盘微调)
- 透视变换方案

### 识别

- 三层神经网络结构

| 名称 | 大小 | 用途 |
| --- | --- | --- |
| yolov8s | 1280x1280 | 识别机器人 |
| yolov8s | 192x192 | 识别装甲板 |
| resnet18 | 224x224 | 数字分类 |

建议根据相机分辨率调整模型大小，以提高推理速度。

由于使用了时间同步，只要推理速度>10Hz 也能正常使用。

**模型存储在 model/ONNX 文件夹下**
- 提供了 onnx 自动转换 trt，如果没有检测到 TensorRT 编译的模型，会自动编译对应模型。

### 激光雷达

- GICP配准 **RM2025场地地图(有墙版)存储在config/RM2025.pcd**
- KdTree离群点检测
- 欧几里得聚类
- 飞镖检测
- 空中机器人检测

### 传感器融合

- 使用卡尔曼滤波器对激光雷达识别到的目标进行跟踪，同时将相机识别结果向卡尔曼轨迹进行匹配，最后融合卡尔曼滤波器结果和相机识别结果，输出最终结果。

### 工具包
- 进程内播放 rosbag (ros2 jazzy 已支持)

## 模块介绍

| 模块 | 说明 |
| --- | --- |
| [`lidar`](./src/lidar/) | 激光雷达模块 |
| [`camera`](./src/radar_vision/) | 相机模块（无相机驱动） |
| [`interface`](./src/interface/) | 自定义消息接口 |
| [`livox_driver`](./src/livox_driver/) | Livox 驱动 |
| [`fusion`](./src/fusion/) | 传感器后融合模块 |
| [`utils`](./src/utils/) | 工具包 |
| [~~`llm_decision`~~](./src/llm_decision/) | ~~大模型决策模块~~ |

## 依赖

```bash
Ubuntu 24.04
ROS2 (Jazzy)
CUDA+CUDNN+TensorRT10
OpenCV 4.6.0
PCL 1.14.0
Livox_SDK(1)
```
> Livox SDK：https://github.com/Livox-SDK/Livox-SDK

## 进程间通信消息名称及用途

### 1. ROS2 通信

#### 激光雷达

| 名称 | 类型 | 用途 |
| --- | --- | --- |
| livox/lidar | topic< sensor_msgs::msg::PointCloud2 > | Livox驱动接口 |
| livox/map | topic< sensor_msgs::msg::PointCloud2 > | 3D地图可视化 |
| livox/lidar_dynamic | topic< sensor_msgs::msg::PointCloud2 > | 动态点云 |
| livox/cluster | topic< sensor_msgs::msg::PointCloud2 > | 聚类结果 |
| livox/lidar_detect | topic< vision_interface::msg::RadarWarn > | 激光雷达预警 |



#### 相机

| 名称 | 类型 | 用途 |
| --- | --- | --- |
| camera_image | topic< sensor_msgs::msg::Image > | 相机驱动接口 |
| detect_result | topic< vision_interface::msg::DetectResult > | 识别结果 |
| resolve_result | topic< vision_interface::msg::DetectResult > | 解算结果 |


#### 传感器融合

| 名称 | 类型 | 用途 |
| --- | --- | --- |
| kalman_detect | topic<vision_interface::msg::DetectResult> | 卡尔曼节点输出 |
| match_info | topic<vision_interface::msg::MatchInfo > | 当前比赛的实时信息 |
| Radar2Sentry | topic<vision_interface::msg::Radar2Sentry> | 发送给串口的最终结果 |


## 测试

```bash
ros2 launch radar_vision run_rosbag.launch.py #通过rosbag启动相机
ros2 launch dynamic_cloud lidar.launch.py #启动激光雷达识别
ros2 run debug_map debug_map #启动地图可视化
ros2 launch livox_ros2_driver livox_lidar_launch.py #启动Livox驱动
```

`run_rosbag.launch.py` 与 `calib_rosbag.launch.py` 里的 rosbag 路径目前是硬编码绝对路径（默认 `~/radar_ws/bags/merged_bag_1.db3`），换机器或换数据包时需要改这两个 launch 文件。

### 相机外参标定
```bash
ros2 launch radar_vision calib_rosbag.launch.py
```
按Enter键开始标定,依次点击我方堡垒靠近我方的底角，我方前哨站血条底部，敌方基地引导灯，敌方前哨战引导灯，敌方四十三度坡围挡左底角。


每次点击后可使用wasd调节上下左右，按n键保存当前点，保存5个点后自动计算外参并保存在config/out_matrix.yaml
## 可视化
Launch文件已集成foxglove-bridge,启动后直接打开foxglove-studio即可查看
## TODO
- 多相机/雷达从当前逻辑(结构)上可以实现，但是并没有进行对应ros2接口的适配
- 改进聚类与匹配算法
- 使用ros参数，实时调参
- 把 rosbag 路径改为 launch 参数，去掉硬编码绝对路径
# 仓库

https://github.com/camelliahm/Radar_ws

</div>

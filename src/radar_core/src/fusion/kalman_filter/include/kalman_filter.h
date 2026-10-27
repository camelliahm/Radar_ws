#pragma once

#include <rclcpp/publisher.hpp>                  // 发布者句柄
#include <rclcpp_components/register_node_macro.hpp>  // 组件注册宏
#include <rcl_interfaces/msg/set_parameters_result.hpp>  // 参数回调返回
#include <vision_interface/msg/detect_result.hpp>  // 视觉/融合结果消息
#include <vision_interface/msg/match_info.hpp>     // 比赛信息消息
#include <vision_interface/msg/radar2_sentry.hpp>  // 发往哨兵的消息
#include <vision_interface/msg/radar_warn.hpp>     // 雷达预警消息
#include "filter_plus.h"                        // 单目标卡尔曼滤波器
#include "opencv2/opencv.hpp"                   // cv::Mat / cv::Scalar
#include "pcl/io/pcd_io.h"                      // PCL 点云读写
#include "pcl/point_cloud.h"                    // pcl::PointCloud
#include "pcl/point_types.h"                    // PointXYZ / PointXY
#include "pcl_conversions/pcl_conversions.h"    // ROS 与 PCL 互转
#include "rclcpp/rclcpp.hpp"                    // 节点基类
#include "sensor_msgs/msg/point_cloud2.hpp"     // 点云消息
#include "sensor_msgs/point_cloud2_iterator.hpp"  // 点云字段遍历

namespace radar_core {
// 多目标跟踪节点：消费聚类点云与视觉结果，输出带颜色编号的目标轨迹
class KalmanFilter : public rclcpp::Node {
public:
    // 构造函数负责建立订阅/发布，并初始化节点成员
    KalmanFilter(const rclcpp::NodeOptions& node_options);
    ~KalmanFilter() {}

private:
    // 订阅：聚类后的目标质心点云 /livox/lidar_cluster
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    // 订阅：视觉解算结果 /resolve_result
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        sub_detect_;
    // 订阅：雷达预警 /lidar_detect，目前只缓存不参与计算
    rclcpp::Subscription<vision_interface::msg::RadarWarn>::SharedPtr
        sub_lidar_;
    // 订阅：比赛信息 /match_info，取己方颜色决定是否做坐标镜像
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        sub_match_;
    // 发布：发往哨兵的敌方坐标 /radar2sentry
    rclcpp::Publisher<vision_interface::msg::Radar2Sentry>::SharedPtr
        radar_pub_;
    // 发布：带颜色与编号的融合结果 /kalman_detect
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr
        radar_detect_pub_;
    // 发布：滤波后的彩色点云 /livox/lidar_kalman，供可视化
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    // 点云回调：预测、关联、校正、剔除、发布的主流程
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    // 视觉回调：把视觉点位投给时间与空间都匹配的航迹
    void detect_callback(
        const vision_interface::msg::DetectResult::SharedPtr msg);
    // 预警回调：仅把消息存入成员
    void
    lidar_callback(const vision_interface::msg::RadarWarn::SharedPtr msg);
    // 比赛信息回调：缓存己方颜色等场上状态
    void
    match_callback(const vision_interface::msg::MatchInfo::SharedPtr msg);
    // 全部存活航迹，一个元素代表一个目标
    std::vector<Kalman_filter_plus>  KFs;
    // 最近一帧雷达预警
    vision_interface::msg::RadarWarn lidar_detect;
    // 最近一帧比赛信息
    vision_interface::msg::MatchInfo match_info;
    // 回调：外部通过参数服务写入己方颜色
    rcl_interfaces::msg::SetParametersResult on_parameters(
        const std::vector<rclcpp::Parameter>& params);
    // 参数回调句柄，需保留以免回调被析构注销
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
        param_cb_;
    // 收到裁判系统颜色后，允许其接管参数的时长，单位秒
    double self_color_hold = 3.0;
    // 最近一次由参数指定颜色的时刻，用于判断裁判系统是否已接管
    rclcpp::Time self_color_stamp;
};
}  // namespace radar_core

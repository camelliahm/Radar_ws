#ifndef RADAR_RESOLVE_H
#define RADAR_RESOLVE_H

#include <memory>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_interface/msg/match_info.hpp>
#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "opencv2/opencv.hpp"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "radar_utils.h"
#include "vision_interface/msg/detect_result.hpp"
#include "vision_interface/msg/radar2_sentry.hpp"
namespace radar_core {

// 视觉解算节点：把检测节点输出的像素坐标，按标定外参换算成场地世界坐标，
// 并以点云（可视化）和 DetectResult（给融合/调试）两种形式发布。
class Resolve final : public rclcpp::Node {
public:
    explicit Resolve(const rclcpp::NodeOptions& options);
    // 单点调试入口：直接喂一个像素点，输出对应的世界坐标点云
    void callback(const std::shared_ptr<geometry_msgs::msg::Vector3> msg);
    // 主业务入口：处理一整帧检测结果（蓝方 6 个 + 红方 6 个）
    void DetectCallback(
        const vision_interface::msg::DetectResult::SharedPtr msg);
    // 缓存比赛信息（易伤状态、血量、剩余时间）
    void MatchInfoCallback(
        const vision_interface::msg::MatchInfo::SharedPtr msg);
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr point_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_sub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
            match_info_sub;
    parser* parser_;            // 像素->世界 坐标解算器
    int     EnemyColor = 1;     // 1 = 对方为红方，2 = 对方为蓝方
    cv::Mat minimap;            // RM2025.png 场地俯视图，用于画点调试
    int     markers[6];         // 6 个易伤区的易伤状态
    int16_t match_time;         // 比赛剩余时间（s）
    uint8_t robot_hp[16];       // 16 台机器人的血量

private:
    // 发布解算后的世界坐标点（PointCloud2，供 RViz/Foxglove 观察）
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub;
    // 发布解算后的场地坐标（DetectResult，供 kalman_filter / debug_map 使用）
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr
        pub_radar;
};

// 俯视图上的一个目标点（用于内部调试绘制）
class map_car {
public:
    float x;
    float y;
    int   id;
};
}  // namespace radar_core

#endif

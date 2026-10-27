#ifndef RADAR_CALIBRATE_H
#define RADAR_CALIBRATE_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/photo.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/opencv.hpp"
#include "radar_utils.h"
#include "tf2_msgs/msg/tf_message.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace radar_core {
// 以下为标定过程的全局状态，鼠标回调只能通过全局变量与节点交互
static cv::Mat                  cvimage_;        // 当前显示帧（已缩放到标定窗口尺寸）
static std::vector<cv::Point2f> pick_points;     // 用户在图像上点选的特征点
static cv::Mat                  camera_matrix;   // 相机内参
static cv::Mat                  dist_coeffs;     // 畸变系数
static cv::Mat                  rvec;            // solvePnP 求出的旋转向量
static cv::Mat                  tvec;            // solvePnP 求出的平移向量
static bool                     is_calibrating = false;  // 是否处于交互选点阶段

// 相机外参标定节点：在画面上手动点选 5 个已知世界坐标的场地特征点，
// 用 solvePnP 求出世界->相机的旋转/平移，写入 config/out_matrix.yaml。
class Calibrate final : public rclcpp::Node {
public:
    std::vector<cv::Point3f> real_points;  // 5 个特征点的世界坐标（单位 m）

    // 特征点的世界坐标，按点选顺序一一对应；
    // 选取原则是分布分散、边界清晰、易于在图像上精确定位
    cv::Point3f self_FORTRESS = cv::Point3f(5.471, -7.5, 0.0);        // 己方基地地面角点
    cv::Point3f self_Tower = cv::Point3f(10.936, -11.161, 0.868);     // 己方哨兵塔
    cv::Point3f enemy_Base = cv::Point3f(25.49, -7.5, 1.24524);       // 敌方基地
    cv::Point3f enemy_Tower = cv::Point3f(16.925, -3.625, 1.745);     // 敌方前哨站
    cv::Point3f enemy_High = cv::Point3f(20.20, -10.8, 0.8);          // 敌方高地

    explicit Calibrate(const rclcpp::NodeOptions& options);
    // 原始图像回调（Hik 相机直出）
    void callback(const sensor_msgs::msg::Image::SharedPtr msg);
    // 压缩图像回调（rosbag 回放时使用）
    void compressed_callback(
        const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
                                                   compressed_image_sub;
    // 点选齐全后执行 solvePnP 并写文件
    void                                           solve();
    parser*                                        parser_;  // 解算器，标定完成后需刷新
    std::shared_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
    geometry_msgs::msg::TransformStamped           transformStamped;
};
// OpenCV 鼠标回调：左键进入 WASD 微调，按 n 确认选点
void mousecallback(int event, int x, int y, int flags, void* userdata);

}  // namespace radar_core
#endif

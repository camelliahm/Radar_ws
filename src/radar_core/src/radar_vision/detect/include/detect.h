#ifndef RADAR_DETECT_H
#define RADAR_DETECT_H

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "classify.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "NvidiaInterface.hpp"
#include "opencv2/opencv.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "vision_interface/msg/detect_result.hpp"
#include "yolos.hpp"
#include "BaseInfer.hpp"
#include <fstream>
namespace radar_core {

// 三级视觉检测节点：整车 YOLO -> 装甲板 YOLO -> 装甲板数字分类。
// 输出 DetectResult（按编号 1..6 存放的蓝/红方装甲板中心像素坐标）。
class Detect final : public rclcpp::Node {
public:
    explicit Detect(const rclcpp::NodeOptions& options);
    void callback(const std::shared_ptr<sensor_msgs::msg::Image> msg);
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr
        compressed_image_sub;

private:
    std::shared_ptr<Infer<yolo::BoxArray>>     yolo;        // 整车检测
    std::shared_ptr<Infer<yolo::BoxArray>>     armor_yolo;  // 装甲板检测（批量）
    std::shared_ptr<Infer<int>> classifier;                 // 装甲板数字分类
    rclcpp::Publisher<vision_interface::msg::DetectResult>::SharedPtr pub;

    bool        if_rosbag = false;  // 是否处于 rosbag 回放模式
    int         EnemyColor;         // 对方颜色
    int         debug;              // 调试绘制开关，运行时按 r 切换
    std::string yolo_path;          // TensorRT engine 路径
    std::string armor_path;
    std::string classify_path;
};

// 中间结构：一台车的整车框、装甲板框、以及最终选出的最佳装甲板信息
class Car {
public:
    cv::Rect       car_rect;     // 裁剪到图像范围内的整车框
    yolo::Box      car;          // 整车检测结果（原图坐标）
    yolo::BoxArray armors;       // 该车内的装甲板（相对整车裁剪图坐标）
    cv::Point2f    center;       // 最佳装甲板中心（原图坐标）
    cv::Rect       center_rect;  // 最佳装甲板框（原图坐标）
    int            number = 0;   // 装甲板数字，对应机器人编号
    int            color = 1;    // 0 蓝 / 2 红 / 1 无法判定
};
}  // namespace radar_core

#endif

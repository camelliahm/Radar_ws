#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <chrono>
#include <vector>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <vision_interface/msg/radar_warn.hpp>
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

namespace radar_core {
// 动态点云提取节点：把实时点云变换到场地坐标系 rm_frame，
// 与静态地图做最近邻比对，距离超过阈值的点判为动态点（场上机器人/飞镖等）。
// 同时负责飞镖与空中机器人（fly）的区域入侵检测，并发布预警消息。
class DynamicCloud : public rclcpp::Node {
public:
    DynamicCloud(const rclcpp::NodeOptions& node_options);
    ~DynamicCloud() {}

private:
    int accumulate_time  = 3;  // 滑动窗口累积帧数，用多帧叠加提升点云密度
    int accumulate_count = 0;  // 当前已累积帧数，达到 accumulate_time 后转为滑窗
    // 静态背景地图（RM2025.pcd 经体素下采样），用于最近邻比对
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud;
    // 动态点云的滑窗缓存，每帧一份
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
    // 场外/特殊区域点云的滑窗缓存（飞镖区、空中机器人飞行区）
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
        other_accumulated_clouds_;
    // 订阅原始点云 /livox/lidar
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    // 发布动态点云 /livox/lidar_dynamic（供 cluster 聚类）
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    // 发布特殊区域点云 /livox/lidar_other（飞镖、空中目标）
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr other_pub_;
    // 发布飞镖/空中机器人预警 /lidar_detect
    rclcpp::Publisher<vision_interface::msg::RadarWarn>::SharedPtr
         detect_pub_;
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    // 多线程最近邻查询：与地图距离大于 threshold 的点写入 output_cloud
    void GetDynamicCloud(pcl::PointCloud<pcl::PointXYZ>& input_cloud,
                         pcl::PointCloud<pcl::PointXYZ>& output_cloud,
                         float threshold, int thread_num);
    // 以静态地图建立的 KdTree，用于最近邻距离查询
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_Tree;
    tf2_ros::Buffer                 tf_buffer_;
    tf2_ros::TransformListener      tf_listener_;
};
}  // namespace radar_core
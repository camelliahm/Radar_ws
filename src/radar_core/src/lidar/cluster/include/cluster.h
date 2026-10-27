#include <pcl/kdtree/kdtree.h>
#include <chrono>
#include <numeric>
#include <vector>
#include <pcl/impl/point_types.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
namespace radar_core {
// 点云聚类节点：对 dynamic_cloud 输出的动态点云做欧式聚类，
// 每个簇用其质心代表一个目标，供后续卡尔曼滤波跟踪使用。
class Cluster : public rclcpp::Node {
public:
    Cluster(const rclcpp::NodeOptions& node_options);
    ~Cluster() {}

private:
    // 订阅 /livox/lidar_dynamic（动态点云），发布 /livox/lidar_cluster（目标质心点云）
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_;
    // 单帧回调：聚类 + 求质心 + 发布
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    // 预留的多帧累积缓存，当前实现未使用
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> accumulated_clouds_;
};
}  // namespace radar_core

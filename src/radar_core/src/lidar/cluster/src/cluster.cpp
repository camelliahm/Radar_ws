#include "cluster.h"
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <rclcpp/duration.hpp>
namespace radar_core {

Cluster::Cluster(const rclcpp::NodeOptions& node_options)
    : Node("cluster_node", node_options)
{
    RCLCPP_INFO(this->get_logger(), "cluster_node start");
    // 输入为动态点云（已剔除静态地图），输出为每个目标簇的质心
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10,
        std::bind(&Cluster::callback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10);
}

void Cluster::callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) {
        return;  // 无动态点时直接返回，避免下游收到空质心云
    }
    // 聚类搜索使用的 KdTree，以当前帧点云建立索引
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    auto time = std::chrono::system_clock::now();

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.25);  // 邻域半径 0.25 m：远小于机器人间距，避免粘连
    ec.setMinClusterSize(5);       // 少于 5 点视为噪点丢弃
    ec.setMaxClusterSize(1000);    // 超过 1000 点视为背景/地面残留
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);
    // 输出点云：每个簇只保留一个质心点
    pcl::PointCloud<pcl::PointXYZ>* out_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    for (auto it = cluster_indices.begin(); it != cluster_indices.end();
         ++it) {
        // 按索引从原始点云中取出当前簇的所有点
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
            new pcl::PointCloud<pcl::PointXYZ>);
        for (auto pit = it->indices.begin(); pit != it->indices.end();
             ++pit) {
            cloud_cluster->points.push_back(cloud->points[*pit]);
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // 累加求取簇内点的几何中心（质心）
        pcl::PointXYZ move_point;
        for (auto point : cloud_cluster->points) {
            move_point.x += point.x;
            move_point.y += point.y;
            move_point.z += point.z;
        }
        move_point.x /= cloud_cluster->points.size();
        move_point.y /= cloud_cluster->points.size();
        move_point.z /= cloud_cluster->points.size();
        out_cloud->points.push_back(move_point);
    }
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*out_cloud, output);
    output.header.frame_id = "rm_frame";  // 上游已完成到 rm_frame 的变换
    output.header.stamp = msg->header.stamp;
    pub_->publish(output);
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    RCLCPP_INFO(
        this->get_logger(), "Cluster callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace radar_core

RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::Cluster)
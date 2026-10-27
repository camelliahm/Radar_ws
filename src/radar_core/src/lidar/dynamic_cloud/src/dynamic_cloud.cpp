#include "dynamic_cloud.h"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/time.hpp>
namespace radar_core {
DynamicCloud::DynamicCloud(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("dynamic_cloud_node", node_options),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
{
    RCLCPP_INFO(this->get_logger(), "Dynamic_cloud Node start");
    // 载入场地静态点云地图，作为"什么是背景"的基准
    auto temp_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("config/RM2025.pcd",
                                            *temp_cloud) == -1) {
        PCL_ERROR("Couldn't read file map.pcd \n");
    }
    // 体素下采样到 0.1 m：降低地图点数，加快最近邻查询；0.1 m 与判定阈值同量级
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(temp_cloud);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
    auto result = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    sor.filter(*result);
    map_cloud = result;
    kd_Tree.setInputCloud(map_cloud);  // 地图只需建一次索引

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar", 10,
        std::bind(&DynamicCloud::callback, this, std::placeholders::_1));
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_dynamic", 10);
    other_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_other", 10);
    detect_pub_ = this->create_publisher<vision_interface::msg::RadarWarn>(
        "/lidar_detect", 10);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Dynamic Cloud Launch!");
}

void DynamicCloud::GetDynamicCloud(
    pcl::PointCloud<pcl::PointXYZ>& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>& output_cloud, float threshold,
    int thread_num)
{
    int                                         K = 1;  // 只查最近 1 个邻居
    std::vector<std::thread>                    threads;
    // 每个线程写自己独立的分片容器，避免加锁
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);
    auto start = std::chrono::system_clock::now();
    int  cloud_size = input_cloud.points.size();
    int  step = cloud_size / thread_num;
    for (int i = 0; i < thread_num; i++) {
        // 第 i 个线程处理 [i*step, (i+1)*step) 区间；
        // 末尾不足 thread_num 整除的余数点不参与处理
        threads.push_back(std::thread([i, step, cloud_size, &clouds,
                                       &input_cloud, this, threshold, K]() {
            for (int j = i * step; j < (i + 1) * step; j++) {
                std::vector<int>   pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);
                // 与静态地图的最近邻距离超过 threshold（单位 m 的平方）
                // 说明该位置在原始地图中不存在，判为动态点
                if (kd_Tree.nearestKSearch(input_cloud.points[j], K,
                                           pointIdxNKNSearch,
                                           pointNKNSquaredDistance) > 0) {
                    if (pointNKNSquaredDistance[0] > threshold) {
                        clouds[i].points.push_back(input_cloud.points[j]);
                    }
                }
            }
        }));
    }
    for (auto& t : threads) {
        t.join();
    }
    // 合并各线程分片结果（+= 会追加 points）
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }
    auto end = std::chrono::system_clock::now();
}

// 自由函数：多线程把点云从雷达坐标系变换到 rm_frame。
// 与 callback 中内联的那次变换逻辑一致，当前主流程未调用。
void TransformCloud(pcl::PointCloud<pcl::PointXYZ>&      input_cloud,
                    pcl::PointCloud<pcl::PointXYZ>&      output_cloud,
                    geometry_msgs::msg::TransformStamped transform_stamped,
                    int                                  thread_num)
{
    std::vector<std::thread>                    threads;
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clouds(thread_num);
    auto start = std::chrono::system_clock::now();
    int  cloud_size = input_cloud.points.size();
    int  step = cloud_size / thread_num;

    // 由平移 + 四元数旋量构造 4x4 仿射变换
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);

    for (int i = 0; i < thread_num; i++) {
        threads.push_back(std::thread(
            [i, step, cloud_size, &clouds, &input_cloud, transform]() {
                // 多一个 j < cloud_size 边界判断，防止余数段越界
                for (int j = i * step; j < (i + 1) * step && j < cloud_size;
                     j++) {
                    pcl::PointXYZ   point = input_cloud.points[j];
                    Eigen::Vector3f point_vec(point.x, point.y, point.z);
                    Eigen::Vector3f point_out = transform * point_vec;

                    pcl::PointXYZ transformed_point;
                    transformed_point.x = point_out.x();
                    transformed_point.y = point_out.y();
                    transformed_point.z = point_out.z();

                    clouds[i].points.push_back(transformed_point);
                }
            }));
    }

    for (auto& t : threads) {
        t.join();
    }
    for (auto& cloud : clouds) {
        output_cloud += cloud;
    }
    auto end = std::chrono::system_clock::now();
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "transform time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            1000.0);
}

void DynamicCloud::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    vision_interface::msg::RadarWarn lidar_detect;
    // 飞镖检测区：己方半场靠近基地的窄条空间，尺寸约 0.19 x 0.6 x 0.76 m
    auto dart_cloud_filter = [](pcl::PointXYZ& point) {
        return (point.x > 28 - 0.5889 - 0.1885 && point.x < 28 - 0.5889) &&
               (point.y > 3.925 && point.y < 4.525) &&
               (point.z > 2.4722 - 0.859 + 0.1 && point.z < 2.4722);
    };  // 飞镖检测
    // 空中机器人起飞区（最靠近己方基地的一档）
    auto fly_safe_filter = [](pcl::PointXYZ& point) {
        return (point.x > 28 - 2.775 && point.x < 27.5) &&
               (point.y > 0.2 && point.y < 2.2) &&
               (point.z > 1.7 && point.z < 3);
    };  // 飞机起飞

    // 空中机器人越过中线飞到半场（第二档告警）
    auto fly_warn_filter = [](pcl::PointXYZ& point) {
        return (point.x > 19.83 && point.x < 28 - 2.7) &&
               (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
               (point.z > 1.7 && point.z < 3);
    };  // 飞机飞到半场

    // 空中机器人逼近我方（第三档，最高级告警）
    auto fly_alarm_filter = [](pcl::PointXYZ& point) {
        return (point.x > 13 && point.x < 20.5) &&
               (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
               (point.z > 1.7 && point.z < 3);
    };  // 飞机飞到中场

    auto receive_cloud = pcl::PointCloud<pcl::PointXYZ>();
    pcl::fromROSMsg(*msg, receive_cloud);

    std::chrono::steady_clock::time_point t1 =
        std::chrono::steady_clock::now();
    geometry_msgs::msg::TransformStamped transform_stamped;
    auto ta = std::chrono::steady_clock::now();
    try {
        // 雷达装在车体上，需查当前时刻从雷达坐标系到场地坐标系 rm_frame 的变换；
        // TimePointZero 表示取最新可用（不使用消息自带时间戳，避免累积延迟）
        transform_stamped = tf_buffer_.lookupTransform(
            "rm_frame", msg->header.frame_id, tf2::TimePointZero);
    }
    catch (tf2::TransformException& ex) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Transform error: %s",
                     ex.what());
        return;
    }
    // 把整帧点云变换到场地坐标系
    pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
    auto                           transform = Eigen::Affine3f::Identity();
    transform.translation() << transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z;
    Eigen::Quaternionf rotation(transform_stamped.transform.rotation.w,
                                transform_stamped.transform.rotation.x,
                                transform_stamped.transform.rotation.y,
                                transform_stamped.transform.rotation.z);
    transform.rotate(rotation);
    pcl::transformPointCloud(receive_cloud, transformed_cloud, transform);
    // filtered_cloud：场内有效区域点云，参与动态点判定
    // other_filtered_cloud：被裁掉的、但仍可能是飞镖/空中机的点，单独保留
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    pcl::PointCloud<pcl::PointXYZ> other_filtered_cloud;
    for (size_t i = 0; i < transformed_cloud.size(); i++) {
        auto& point = transformed_cloud.points[i];
        // 以下条件表示该点落在"有效场地"之外，需要裁剪：
        //   x/y/z 超出场地包围盒；己方基地旁的空区；场地中央的斜坡/立柱排除区（45° 旋转矩形）
        if (point.x < 3 || point.x > 28 || point.y < 0 || point.y > 15 ||
            point.z < 0 || point.z > 1.4 ||
            (point.y > 0 && point.y < 5 && point.x > 25) ||
            ((21.5 - 2.9 / sqrt(2)) < (point.x + point.y) &&
             (point.x + point.y) < (21.5 + 2.9 / sqrt(2)) &&
             (-6.5 - 0.9 / sqrt(2)) < (point.y - point.x) &&
             (point.y - point.x) < (-6.5 + 0.9 / sqrt(2)))) {
            // 虽然在场外，但如果落在飞镖区或空中机飞行区，仍要保留下来做入侵判断。
            // 注意这里 && 优先于 ||，实际分组为：
            //   (飞镖区三条件全满足) || (飞行区三条件全满足)
            if ((point.x > 28 - 0.5889 - 0.1885 && point.x < 28 - 0.5889) &&
                    (point.y > 3.925 && point.y < 4.525) &&
                    (point.z > 2.4722 - 0.859 + 0.1 && point.z < 2.4722) ||
                (point.x > 13 && point.x < 27.5) &&
                    (point.y > 0.2 && point.y < 1.356 + 2.4 + 0.8) &&
                    (point.z > 1.7 && point.z < 3)) {
                other_filtered_cloud.push_back(point);
            }
            continue;
        }
        filtered_cloud.push_back(point);
    }
    // 与静态地图比对提取动态点；阈值 0.1 为距离平方（约 0.32 m），12 线程并行
    pcl::PointCloud<pcl::PointXYZ> dynamic_pointcloud;
    GetDynamicCloud(filtered_cloud, dynamic_pointcloud, 0.1, 12);

    // 滑动窗口累积：先用满 accumulate_time 帧填窗，之后每帧替换最旧一帧。
    // 单帧动态点较稀疏，多帧叠加才能凑够判定用的点数阈值。
    if (accumulate_count < accumulate_time) {
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.push_back(
            other_filtered_cloud.makeShared());
        accumulate_count++;
    } else {
        accumulated_clouds_.erase(accumulated_clouds_.begin());
        accumulated_clouds_.push_back(dynamic_pointcloud.makeShared());
        other_accumulated_clouds_.erase(other_accumulated_clouds_.begin());
        other_accumulated_clouds_.push_back(
            other_filtered_cloud.makeShared());
    }
    // 把窗口内所有帧拼成一帧输出
    pcl::PointCloud<pcl::PointXYZ> accumulated_cloud;
    for (auto it = accumulated_clouds_.begin();
         it != accumulated_clouds_.end(); ++it) {
        accumulated_cloud += **it;
    }
    pcl::PointCloud<pcl::PointXYZ> other_accumulated_cloud;
    for (auto it = other_accumulated_clouds_.begin();
         it != other_accumulated_clouds_.end(); ++it) {
        other_accumulated_cloud += **it;
    }
    ta = std::chrono::steady_clock::now();
    sensor_msgs::msg::PointCloud2 output;
    accumulated_cloud.header.frame_id = "rm_frame";
    other_accumulated_cloud.header.frame_id = "rm_frame";
    pcl::toROSMsg(accumulated_cloud, output);
    output.header.frame_id = "rm_frame";
    output.header.stamp = msg->header.stamp;
    pub_->publish(output);

    pcl::toROSMsg(other_accumulated_cloud, output);
    other_pub_->publish(output);
    // 从特殊区域点云中挑出落在飞镖区内的点，点数超过 5 认为有飞镖来袭
    pcl::PointCloud<pcl::PointXYZ> dart_cloud;
    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (dart_cloud_filter(point)) {
            dart_cloud.push_back(point);
        }
    }

    if (dart_cloud.size() > 5) {
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Find dart cloud!");
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "Dart cloud size: %d",
                    dart_cloud.size());
        lidar_detect.dart_state = 1;
    }

    // 三级空中机区域分别统计点数，点数超过 40 判为有效目标
    pcl::PointCloud<pcl::PointXYZ> fly_safe_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_warn_cloud;
    pcl::PointCloud<pcl::PointXYZ> fly_alarm_cloud;

    for (size_t i = 0; i < other_accumulated_cloud.size(); i++) {
        auto& point = other_accumulated_cloud.points[i];
        if (fly_safe_filter(point)) {
            fly_safe_cloud.push_back(point);
        }
        if (fly_warn_filter(point)) {
            fly_warn_cloud.push_back(point);
        }
        if (fly_alarm_filter(point)) {
            fly_alarm_cloud.push_back(point);
        }
    }
    // fly_state：0 无目标，1 起飞区，2 半场（警告），3 中场（告警）。
    // 三个 if 依次覆盖，越近的区域优先级越高
    if (fly_safe_cloud.size() > 40) {
        lidar_detect.fly_state = 1;
    }

    if (fly_warn_cloud.size() > 40) {
        lidar_detect.fly_state = 2;
    }

    if (fly_alarm_cloud.size() > 40) {
        lidar_detect.fly_state = 3;
    }

    // 按告警等级打印不同严重性的日志，fly_state == 0 时静默
    switch (lidar_detect.fly_state) {
    case 0:
        break;
    case 1:
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
                    "Safe fly object detected!");
        break;
    case 2:
        RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
                    "Warn fly object detected!");
        break;
    case 3:
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                     "Alarm fly object detected!");
        break;
    default:
        break;
    }

    // 无论是否检测到目标都发布一次，下游可据此复位状态
    detect_pub_->publish(lidar_detect);
    std::chrono::steady_clock::time_point t2 =
        std::chrono::steady_clock::now();
    RCLCPP_INFO(
        rclcpp::get_logger("rclcpp"), "Dynamic callback time: %f",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count() /
            1000.0);
}
}  // namespace radar_core

RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::DynamicCloud)

#include "resolve.h"

#define TDT_INFO(msg) std::cout << msg << std::endl

namespace radar_core {

Resolve::Resolve(const rclcpp::NodeOptions& node_options)
    : Node("radar_resolve_node", node_options)
{
    // 解算器内部会读取 config/camera_params.yaml 与 out_matrix.yaml
    parser_ = new parser();
    minimap = cv::imread("config/RM2025.png");
    point_sub = this->create_subscription<geometry_msgs::msg::Vector3>(
        "camera_point2D", rclcpp::SensorDataQoS(),
        std::bind(&Resolve::callback, this, std::placeholders::_1));

    pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "camera_point3D", rclcpp::SensorDataQoS());
    match_info_sub =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "match_info", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::MatchInfoCallback, this,
                      std::placeholders::_1));

    detect_sub =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "detect_result", rclcpp::SensorDataQoS(),
            std::bind(&Resolve::DetectCallback, this,
                      std::placeholders::_1));
    pub_radar = this->create_publisher<vision_interface::msg::DetectResult>(
        "resolve_result", rclcpp::SensorDataQoS());
    TDT_INFO("Load radar resolve node success!");
}

void Resolve::MatchInfoCallback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    // 整块缓存裁判系统信息，供解算时参考
    for (int i = 0; i < 6; i++) {
        markers[i] = msg->marks[i];
    }
    for (int i = 0; i < 16; i++) {
        robot_hp[i] = msg->robot_hp[i];
    }
    match_time = msg->match_time;
}

void Resolve::callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    // 调试通道：外部给一个像素坐标（x,y），这里解算并发布单点世界坐标
    cv::Point2f point;
    point.x = msg->x;
    point.y = msg->y;
    auto center_point = parser_->parse(point);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ send_point;
    send_point.x = center_point.x;
    // 注意：此处用 15 - y，而 DetectCallback 用 15 + y，两条通道的 y 轴方向处理不同
    send_point.y = 15 - center_point.y;
    std::cout << center_point << std::endl;
    send_point.z = 1;  // z 固定为 1，仅作可视化用
    cloud->points.push_back(send_point);
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "rm_frame";
    pub->publish(output);
}

void Resolve::DetectCallback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    cv::Mat                                 Map_clone = minimap.clone();
    std::vector<map_car>                    cars;
    vision_interface::msg::DetectResult     send_data;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGBA>);
    // 检测结果按编号 1..6 存放，索引 i 对应编号 i+1
    for (int i = 0; i < 6; i++) {
        // 蓝方：坐标为 0 表示该编号未检测到，用乘积做有效性判断
        cv::Point2f blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];
        if (blue_point.x * blue_point.y) {
            auto center_point = parser_->parse(blue_point);
            // 解析得到的是半场坐标，加 15 平移到全场坐标系（对称半场镜像）
            center_point.y = 15 + center_point.y;
            send_data.blue_x[i] = center_point.x;
            send_data.blue_y[i] = center_point.y;
            // 点云染色：蓝方 (B=255)，便于可视化区分
            pcl::PointXYZRGBA send_point;
            send_point.x = center_point.x;
            send_point.y = center_point.y;
            send_point.z = 1;
            send_point.r = 0;
            send_point.g = 0;
            send_point.b = 255;
            send_point.a = 255;
            cloud->points.push_back(send_point);
            // 在场地俯视图上画点：28m x 15m 映射到整幅图像
            cv::circle(
                Map_clone,
                cv::Point((Map_clone.cols * center_point.x) / 28,
                          Map_clone.rows * (15 - center_point.y) / 15),
                20, cv::Scalar(200, 0, 0), -1);
            cv::putText(
                Map_clone, std::to_string(i + 1),
                cv::Point((Map_clone.cols * center_point.x) / 28 - 10,
                          Map_clone.rows * (15 - center_point.y) / 15 + 10),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
        // 红方：逻辑与蓝方一致，仅颜色不同（点云 R=255，俯视图红点）
        cv::Point2f red_point;
        red_point.x = msg->red_x[i];
        red_point.y = msg->red_y[i];
        if (red_point.x * red_point.y) {
            auto center_point = parser_->parse(red_point);
            center_point.y = 15 + center_point.y;
            send_data.red_x[i] = center_point.x;
            send_data.red_y[i] = center_point.y;
            pcl::PointXYZRGBA send_point;
            send_point.x = center_point.x;
            send_point.y = center_point.y;
            send_point.z = 1;
            send_point.r = 255;
            send_point.g = 0;
            send_point.b = 0;
            send_point.a = 255;
            cloud->points.push_back(send_point);
            cv::circle(
                Map_clone,
                cv::Point((Map_clone.cols * center_point.x) / 28,
                          Map_clone.rows * (15 - center_point.y) / 15),
                20, cv::Scalar(0, 0, 200), -1);
            cv::putText(
                Map_clone, std::to_string(i + 1),
                cv::Point((Map_clone.cols * center_point.x) / 28 - 10,
                          Map_clone.rows * (15 - center_point.y) / 15 + 10),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
    }

    // 沿用原始消息时间戳，保证下游时间对齐
    send_data.header.stamp = msg->header.stamp;
    pub_radar->publish(send_data);
    auto cloud_msg = sensor_msgs::msg::PointCloud2();
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "rm_frame";
    cloud_msg.header.stamp = msg->header.stamp;
    pub->publish(cloud_msg);
}
}  // namespace radar_core
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::Resolve)
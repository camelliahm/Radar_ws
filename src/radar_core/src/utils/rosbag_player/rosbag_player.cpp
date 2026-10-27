#include <thread>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/detail/image__struct.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_interface/msg/match_info.hpp>

using namespace std::chrono_literals;

// Ctrl+C 时干净退出：关闭 rclcpp 让工作线程的 while 条件失效
void on_exit([[maybe_unused]] int sig)
{
    RCUTILS_LOG_INFO("Exit by Ctrl+C");
    rclcpp::shutdown();
    exit(0);
}

// rosbag 回放组件：把录制包里的点云、压缩图像、比赛信息重新发布成实时话题，
// 让下游算法节点无需真实硬件即可联调。播放线程与 ROS 执行器分离，互不阻塞。
class RosbagPlayer : public rclcpp::Node {
public:
    RosbagPlayer(const rclcpp::NodeOptions& options)
        : Node("rosbag_player_node", options)
    {
        // 包路径由 launch 参数传入，不在此处硬编码
        this->declare_parameter<std::string>("rosbag_file", "");
        this->get_parameter("rosbag_file", rosbag_file);

        pointcloud_publisher_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>(
                "/livox/lidar", 10);
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
            "camera_image", rclcpp::SensorDataQoS());
        match_info_publisher_ =
            this->create_publisher<vision_interface::msg::MatchInfo>(
                "/match_info", 10);
        signal(SIGINT, on_exit);
        reader_.open(rosbag_file);

        // 后台线程持续读包，主线程仍可正常处理 ROS 回调
        processing_thread_ =
            std::make_shared<std::thread>(&RosbagPlayer::play_bag, this);
    }

    ~RosbagPlayer()
    {
        if (processing_thread_ && processing_thread_->joinable()) {
            processing_thread_->join();
        }
    }

private:
    void play_bag()
    {
        while (rclcpp::ok()) {
            // 读到包尾后重新打开，实现循环播放
            if (!reader_.has_next())
                reader_.open(rosbag_file);

            auto start_time = std::chrono::high_resolution_clock::now();
            auto bag_message = reader_.read_next();
            // 时间戳统一替换为当前时刻：包的原始时间戳已过期，
            // 下游的 TF 查询、超时判断都依赖"现在"
            auto ros_time = rclcpp::Clock().now();

            if (bag_message->topic_name == "/livox/lidar") {
                // 反序列化后直接转发点云
                auto pointcloud_msg =
                    std::make_shared<sensor_msgs::msg::PointCloud2>();
                rclcpp::Serialization<sensor_msgs::msg::PointCloud2>
                                          serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  pointcloud_msg.get());
                pointcloud_msg->header.stamp = ros_time;
                pointcloud_publisher_->publish(*pointcloud_msg);
            }
            else if (bag_message->topic_name == "/compressed_image") {
                // 包里存的是 JPEG，需解码成 bgr8 图像再发布
                auto image_msg =
                    std::make_shared<sensor_msgs::msg::CompressedImage>();
                rclcpp::Serialization<sensor_msgs::msg::CompressedImage>
                                          serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  image_msg.get());
                auto img = cv::imdecode(image_msg->data, cv::IMREAD_COLOR);
                auto msg =
                    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img)
                        .toImageMsg();
                msg->header.stamp = ros_time;
                image_publisher_->publish(*msg);
            }
            else if (bag_message->topic_name == "/match_info") {
                auto match_info_msg =
                    std::make_shared<vision_interface::msg::MatchInfo>();
                rclcpp::Serialization<vision_interface::msg::MatchInfo>
                                          serialization;
                rclcpp::SerializedMessage serialized_msg(
                    *bag_message->serialized_data);
                serialization.deserialize_message(&serialized_msg,
                                                  match_info_msg.get());
                match_info_publisher_->publish(*match_info_msg);
            }

            // 简单限速：处理耗时不足 100ms 时补足，避免回放速度远超实时导致下游丢帧
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time)
                    .count();
            if ((duration < 100) && (duration > 1)) {
                std::this_thread::sleep_for(
                    100ms - std::chrono::milliseconds(duration));
            }
        }
        RCLCPP_INFO(this->get_logger(), "No more messages in the bag.");
        rclcpp::shutdown();
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        pointcloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<vision_interface::msg::MatchInfo>::SharedPtr
                                 match_info_publisher_;
    rosbag2_cpp::Reader          reader_;
    std::shared_ptr<std::thread> processing_thread_;
    std::string                  rosbag_file;
};

RCLCPP_COMPONENTS_REGISTER_NODE(RosbagPlayer)

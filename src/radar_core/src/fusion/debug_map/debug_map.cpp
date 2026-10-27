#include <fstream>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_interface/msg/detect_result.hpp>
#include <vision_interface/msg/match_info.hpp>
#include <vision_interface/msg/radar2_sentry.hpp>
#include <vision_interface/msg/radar_warn.hpp>
namespace radar_core {
// 2D 地图调试与最终消息输出节点。
// 输入：/kalman_detect（雷达融合结果）、/resolve_result（相机解算结果）、/match_info
// 输出：/map_2d（俯视图调试画面）、/hero_state（英雄位置预警）、/Radar2Sentry（发给哨兵）
class DebugMap : public rclcpp::Node {
public:
    explicit DebugMap(const rclcpp::NodeOptions& options)
        : Node("debug_map", options)
    {
        detect_result_sub =
            this->create_subscription<vision_interface::msg::DetectResult>(
                "/kalman_detect", 10,
                std::bind(&DebugMap::callback, this,
                          std::placeholders::_1));
        camera_detect_sub =
            this->create_subscription<vision_interface::msg::DetectResult>(
                "/resolve_result", rclcpp::SensorDataQoS(),
                std::bind(&DebugMap::camera_callback, this,
                          std::placeholders::_1));
        map = cv::imread("config/RM2025.png");
        match_info_sub =
            this->create_subscription<vision_interface::msg::MatchInfo>(
                "/match_info", 10,
                std::bind(&DebugMap::save_match_info, this,
                          std::placeholders::_1));
        radar_warn_pub =
            this->create_publisher<vision_interface::msg::RadarWarn>(
                "/hero_state", 10);
        debug_map_pub =
            this->create_publisher<sensor_msgs::msg::Image>("/map_2d", 10);
        radar2sentry_pub =
            this->create_publisher<vision_interface::msg::Radar2Sentry>(
                "/Radar2Sentry", rclcpp::SensorDataQoS());
        // 每个像素代表 1/25 m，即 28m x 15m 场地映射到 700 x 375 的图像
        cv::resize(map, map, cv::Size(28 * 25, 15 * 25));
    }
    // 缓存裁判系统信息，并把己方颜色做 1<->2 互换，
    // 这样后续逻辑统一按"1/2 表示谁"来判断，不必在各处再取反
    void save_match_info(
        const std::shared_ptr<vision_interface::msg::MatchInfo> msg)
    {
        this->match_info = *msg;
        if (msg->self_color == 1)
            match_info.self_color = 2;
    }

    // 把 6 组蓝/红方点位画到场地俯视图上：位置过期超过 0.5s 的点不再显示
    void show_map()
    {
        auto   now_time = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now_time.time_since_epoch())
                          .count() /
                      1000.0;
        auto clone_map = map.clone();
        for (int i = 0; i < 6; i++) {
            // 数组下标 0..5 对应编号 1..6，但场上没有编号 6，
            // 因此跳过 6 直接显示为 7（英雄/工程等大号机器人）
            int number = i + 1;
            if (number == 6)
                number++;
            // 坐标 (0,0) 表示本帧无该目标，用乘积判断有效性
            if (blue_point[i].x * blue_point[i].y &&
                time - blue_update[i] < 0.5) {
                cv::Point2f point = cv::Point2f(
                    clone_map.cols * blue_point[i].x / 28,
                    clone_map.rows * (15 - blue_point[i].y) / 15);
                cv::circle(clone_map, point, 10, cv::Scalar(200, 0, 0), -1);
                cv::putText(clone_map, std::to_string(number),
                            cv::Point(point.x - 6, point.y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255));
            }
            if (red_point[i].x * red_point[i].y &&
                time - red_update[i] < 0.5) {
                cv::Point2f point = cv::Point2f(
                    clone_map.cols * red_point[i].x / 28,
                    clone_map.rows * (15 - red_point[i].y) / 15);
                cv::circle(clone_map, point, 10, cv::Scalar(0, 0, 200), -1);
                cv::putText(clone_map, std::to_string(number),
                            cv::Point(point.x - 6, point.y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255));
            }
        }
        cv::imshow("map", clone_map);
        cv::waitKey(1);
    }
    // 相机分支：与雷达结果互斥。若距上次雷达更新不足 5s，相机结果不覆盖，
    // 避免两路结果在画面上来回跳变
    void camera_callback(
        const std::shared_ptr<vision_interface::msg::DetectResult> msg)
    {
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;
        for (int i = 0; i < 6; i++) {
            if (msg->blue_x[i] * msg->blue_y[i]) {
                if (time - blue_time[i] > 5) {
                    blue_point[i] =
                        cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
                    // self_color 为 0 表示己方为红方，需把坐标翻到对面半场
                    if (!match_info.self_color) {
                        blue_point[i] = cv::Point2f(28 - msg->blue_x[i],
                                                    15 - msg->blue_y[i]);
                    }
                    blue_update[i] = time;
                }
            }
            if (msg->red_x[i] * msg->red_y[i]) {
                if (time - red_time[i] > 5) {
                    red_point[i] =
                        cv::Point2f(msg->red_x[i], msg->red_y[i]);
                    if (!match_info.self_color) {
                        red_point[i] = cv::Point2f(28 - msg->red_x[i],
                                                   15 - msg->red_y[i]);
                    }
                    red_update[i] = time;
                }
            }
        }
        show_map();
    }

    void
    callback(const std::shared_ptr<vision_interface::msg::DetectResult> msg)
    {
        auto   now = std::chrono::system_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count() /
                      1000.0;
        // 雷达结果直接刷新缓存点位（blue_time/red_time 同时作为互斥时间戳）
        for (int i = 0; i < 6; i++) {
            if (msg->blue_x[i] * msg->blue_y[i]) {
                blue_point[i] = cv::Point2f(msg->blue_x[i], msg->blue_y[i]);
                blue_time[i] = time;
                blue_update[i] = time;
            }
            if (msg->red_x[i] * msg->red_y[i]) {
                red_point[i] = cv::Point2f(msg->red_x[i], msg->red_y[i]);
                red_time[i] = time;
                red_update[i] = time;
            }
        }
        show_map();
        // 英雄位置滞回计数：连续 10 帧以上处于某区域才发出预警，抑制抖动误报
        vision_interface::msg::RadarWarn radar_warn;
        if (hero_count1 > 10) {
            radar_warn.hero_state = 1;
            radar_warn_pub->publish(radar_warn);
        } else if (hero_count2 > 10) {
            radar_warn.hero_state = 2;
            radar_warn_pub->publish(radar_warn);
        }
        // self_color == 0：己方为红方，敌方为蓝方，此时关注 red_point[0]（编号 1）
        if (match_info.self_color == 0) {
            if (red_point[0].x > (28 - 8.668)) {
                // 敌方英雄深入我方半场（x 靠近 28）
                hero_count1++;
                hero_count2--;
            } else if (red_point[0].x < (28 - 20.3) &&
                       red_point[0].x > (28 - 25.075) &&
                       red_point[0].y < 15 && red_point[0].y > 10.3) {
                // 敌方英雄位于我方基地附近区域
                hero_count1--;
                hero_count2++;
            } else {
                hero_count1--;
                hero_count2--;
            }
        }
        // self_color == 2：己方为蓝方，对称地检查 blue_point[0]
        if (match_info.self_color == 2) {
            if (blue_point[0].x < 8.668) {
                hero_count1++;
                hero_count2--;
            } else if (blue_point[0].x > 20.3 && blue_point[0].x < 25.075 &&
                       blue_point[0].y > 0 &&
                       blue_point[0].y < (15 - 10.3)) {
                hero_count1--;
                hero_count2++;
            } else {
                hero_count1--;
                hero_count2--;
            }
        }

        // 组装发给哨兵的目标信息。
        // 易伤机制：marks[i] >= 117 时该编号进入"易伤"状态（relax），
        // 此时不发送其坐标；marks 降到 105 以下解除易伤。
        // 处于易伤但仍未解除的过渡期，每 0.35s 补发一次坐标。
        vision_interface::msg::Radar2Sentry radar2sentry;
        if (match_info.self_color == 0) {
            for (int i = 0; i < 6; i++) {
                if (!relax[i]) {
                    if (match_info.marks[i] >= 117) {
                        relax[i] = true;
                        relax_time[i] = time;
                    } else {
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    }
                } else {
                    if (match_info.marks[i] < 105) {
                        relax[i] = false;
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    } else if (time - relax_time[i] > 0.35) {
                        relax_time[i] = time;
                        if (time - red_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = red_point[i].x;
                            radar2sentry.radar_enemy_y[i] = red_point[i].y;
                        }
                    }
                }
            }
        }
        // 己方为蓝方时逻辑相同，只是数据源换成 blue_point/blue_update
        if (match_info.self_color == 2) {
            for (int i = 0; i < 6; i++) {
                if (!relax[i]) {
                    if (match_info.marks[i] >= 117) {
                        relax[i] = true;
                        relax_time[i] = time;
                    } else {
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    }
                } else {
                    if (match_info.marks[i] < 105) {
                        relax[i] = false;
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    } else if (time - relax_time[i] > 0.35) {
                        relax_time[i] = time;
                        if (time - blue_update[i] < 0.5) {
                            radar2sentry.radar_enemy_x[i] = blue_point[i].x;
                            radar2sentry.radar_enemy_y[i] = blue_point[i].y;
                        }
                    }
                }
            }
        }
        radar2sentry_pub->publish(radar2sentry);
    }
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
        detect_result_sub;
    rclcpp::Subscription<vision_interface::msg::DetectResult>::SharedPtr
                                                          camera_detect_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_map_pub;
    rclcpp::Publisher<vision_interface::msg::RadarWarn>::SharedPtr
        radar_warn_pub;
    rclcpp::Publisher<vision_interface::msg::Radar2Sentry>::SharedPtr
        radar2sentry_pub;
    rclcpp::Subscription<vision_interface::msg::MatchInfo>::SharedPtr
        match_info_sub;

    // 最近一次雷达结果的时间戳，用于与相机结果做 5s 互斥
    double blue_time[6];
    double red_time[6];

    // 易伤状态标记及其进入时间
    bool   relax[6];
    double relax_time[6];

    // 最近一次有效坐标的时间戳，超过 0.5s 视为过期不再发送/显示
    double blue_update[6];
    double red_update[6];

    // 英雄位置滞回计数器，需要连续多帧才触发预警
    int hero_count1;
    int hero_count2;

    // 当前画面上使用的点位（已按己方颜色做场地翻转）
    cv::Point2f blue_point[6];
    cv::Point2f red_point[6];

    vision_interface::msg::MatchInfo match_info;
    cv::Mat                          map;
    int                              count = 0;
};
}  // namespace radar_core

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node_options = rclcpp::NodeOptions();
    rclcpp::spin(std::make_shared<radar_core::DebugMap>(node_options));
    rclcpp::shutdown();
    return 0;
}

// 多目标跟踪节点实现：聚类点云 -> 卡尔曼滤波 -> 视觉打标签 -> 输出
#include "kalman_filter.h"
#include <pcl_conversions/pcl_conversions.h>
#include "filter_plus.h"

namespace radar_core {

// 构造：建立四路订阅与三路发布
KalmanFilter::KalmanFilter(const rclcpp::NodeOptions& node_options)
    : rclcpp::Node("kalman_filter_node", node_options)
{
    // 输入：聚类后的目标质心点云，队列深度 10 帧
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_cluster", 10,
        std::bind(&KalmanFilter::callback, this, std::placeholders::_1));
    // 输出：滤波后的彩色点云，用于 Foxglove 观察跟踪效果
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/livox/lidar_kalman", 10);
    // 输出：发往哨兵的敌方坐标
    radar_pub_ =
        this->create_publisher<vision_interface::msg::Radar2Sentry>(
            "/radar2sentry", 10);
    // 输出：带颜色与编号的融合结果
    radar_detect_pub_ =
        this->create_publisher<vision_interface::msg::DetectResult>(
            "/kalman_detect", 10);
    // 输入：视觉解算出的红蓝点位，SensorDataQoS 只保留最新帧以降低延迟
    sub_detect_ =
        this->create_subscription<vision_interface::msg::DetectResult>(
            "/resolve_result", rclcpp::SensorDataQoS(),
            std::bind(&KalmanFilter::detect_callback, this,
                      std::placeholders::_1));
    // 输入：雷达预警状态
    sub_lidar_ =
        this->create_subscription<vision_interface::msg::RadarWarn>(
            "/lidar_detect", 10,
            std::bind(&KalmanFilter::lidar_callback, this,
                      std::placeholders::_1));
    // 输入：比赛信息，提供己方颜色
    sub_match_ =
        this->create_subscription<vision_interface::msg::MatchInfo>(
            "/match_info", 10,
            std::bind(&KalmanFilter::match_callback, this,
                      std::placeholders::_1));

    // 己方颜色参数：0 蓝、2 红，默认蓝方，可由 UI 通过参数服务改写
    this->declare_parameter<int>("self_color", 0);
    // 注册参数回调，写入时立即生效
    param_cb_ = this->add_on_set_parameters_callback(
        std::bind(&KalmanFilter::on_parameters, this, std::placeholders::_1));

    // 启动完成日志，便于确认节点已进入运行态
    RCLCPP_INFO(this->get_logger(), "Kalman_filter_Node has been started.");
}

// 参数写入校验：只接受 0 或 2，其余取值拒绝并回报失败
rcl_interfaces::msg::SetParametersResult KalmanFilter::on_parameters(
    const std::vector<rclcpp::Parameter>& params)
{
    rcl_interfaces::msg::SetParametersResult result;
    // 默认通过，遇到非法值再改写
    result.successful = true;
    for (auto& p : params) {
        // 只关心己方颜色，其他参数直接放过
        if (p.get_name() != "self_color")
            continue;
        // 颜色约定只有蓝(0)与红(2)
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_INTEGER ||
            (p.as_int() != 0 && p.as_int() != 2)) {
            result.successful = false;
            result.reason = "self_color must be 0 (blue) or 2 (red)";
            return result;
        }
        // 写入比赛信息，后续输出与镜像都以它为准
        int color = static_cast<int>(p.as_int());
        match_info.self_color = static_cast<int8_t>(color);
        // 记录改写时刻，裁判系统颜色在保持期过后优先
        self_color_stamp = this->now();
        RCLCPP_INFO(this->get_logger(), "self_color set to %d (%s)", color,
                    color == 0 ? "blue" : "red");
    }
    return result;
}

// 缓存整条比赛信息，后续镜像变换需要其中的 self_color
void KalmanFilter::match_callback(
    const vision_interface::msg::MatchInfo::SharedPtr msg)
{
    // 裁判系统颜色只有蓝(0)与红(2)两种有效值，其他值视为未连接
    bool judge_ok = (msg->self_color == 0 || msg->self_color == 2);
    // UI 刚写入颜色的保持期内，不让裁判系统覆盖，避免手动选择被冲掉
    bool ui_holding =
        (this->now() - self_color_stamp).seconds() < self_color_hold;
    // 保留当前颜色，仅更新场上状态字段
    int8_t keep = match_info.self_color;
    // 整条消息保存，保证 marks、robot_hp 等字段始终跟随裁判系统
    this->match_info = *msg;
    // UI 未接管且裁判系统颜色有效时才采纳，否则沿用已有颜色
    if (!judge_ok || ui_holding) {
        match_info.self_color = keep;
    }
    // 每帧打印一次，用于确认裁判系统数据在持续到达
    RCLCPP_INFO(this->get_logger(), "Match_info_callback");
}

// 把视觉结果按编号拆成点位，逐个投给已存在的航迹
void KalmanFilter::detect_callback(
    const vision_interface::msg::DetectResult::SharedPtr msg)
{
    // 视觉帧时间戳，用于在航迹历史里找到同一时刻的雷达位置
    rclcpp::Time time = msg->header.stamp;
    // 场上最多 6 块装甲板，下标即编号
    for (int i = 0; i < 6; i++) {
        // 编号 i 的红方点位·
        pcl::PointXY red_point;
        red_point.x = msg->red_x[i];
        red_point.y = msg->red_y[i];
        // 坐标为 0 表示该编号本帧无目标
        if (red_point.x == 0 || red_point.y == 0)
            continue;
        // 每条航迹各自判断是否与这个红点配对
        for (auto& kf : KFs) {
            // 颜色 2 表示红方，编号 i
            kf.camera_match(time, red_point, 2, i);
        }

        // 编号 i 的蓝方点位
        pcl::PointXY blue_point;
        blue_point.x = msg->blue_x[i];
        blue_point.y = msg->blue_y[i];
        // 同样以 0 表示无效
        if (blue_point.x == 0 || blue_point.y == 0)
            continue;
        for (auto& kf : KFs) {
            // 颜色 0 表示蓝方，编号 i
            kf.camera_match(time, blue_point, 0, i);
        }
    }
}

// 仅缓存雷达预警，当前不参与融合决策
void KalmanFilter::lidar_callback(
    const vision_interface::msg::RadarWarn::SharedPtr msg)
{
    // 直接覆盖上一帧
    this->lidar_detect = *msg;
}

// 主流程：每帧点云驱动一次完整的跟踪与输出
void KalmanFilter::callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 点云时间戳，会写回全部输出消息
    rclcpp::Time time = msg->header.stamp;
    // 本帧处理耗时的计时起点
    auto         now_time = std::chrono::steady_clock::now();
    // 原始 XYZ 点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    // 降维后的平面点云，跟踪只在 XY 平面进行
    pcl::PointCloud<pcl::PointXY>::Ptr cloud_xy(
        new pcl::PointCloud<pcl::PointXY>);
    // ROS 消息转 PCL 容器
    pcl::fromROSMsg(*msg, *cloud);
    for (auto point : cloud->points) {
        // 丢掉 z 与强度，只保留平面位置
        pcl::PointXY point_xy;
        point_xy.x = point.x;
        point_xy.y = point.y;
        // 追加到平面点云
        cloud_xy->points.push_back(point_xy);
    }
    // 本帧无目标：既不推进预测，也不做航迹剔除
    if (cloud_xy->points.size() == 0)
        return;
    // 所有航迹先按当前时间向前推进一步
    for (auto& kf : KFs) {
        kf.update_predict_point();
        // 清空更新标志，等待本帧观测
        kf.has_updated = false;
    }

    // 逐点关联：决定每个观测归属哪条航迹或新建航迹
    for (auto point : cloud_xy->points) {
        // 落在门限内的航迹下标集合
        std::vector<int> match_kf_indexs;
        for (int i = 0; i < this->KFs.size(); i++) {
            if (KFs[i].match(point)) {
                match_kf_indexs.push_back(i);
            }
        }
        // 没有任何航迹命中：视为新出现的目标
        if (match_kf_indexs.size() == 0) {
            // 用该观测初始化一条新航迹
            Kalman_filter_plus kf(point, time);
            KFs.push_back(kf);
        } else if (match_kf_indexs.size() == 1) {
            // 唯一命中：直接用该观测校正
            KFs[match_kf_indexs[0]].update(point, time);
        } else {
            // 多条航迹命中：取预测位置最近的一条
            float min_distance = 1000000;
            int   min_index = 0;
            for (auto index : match_kf_indexs) {
                // 观测点到该航迹预测位置的距离
                float distance =
                    KFs[index].Distance(KFs[index].predict_point, point);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = index;
                }
            }
            // 由最近的航迹吸收该观测
            KFs[min_index].update(point, time);
        }
    }

    // 输出点云按航迹身份着色
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    // 逆序遍历，便于在循环中安全删除元素
    for (int i = KFs.size() - 1; i >= 0; i--) {
        // 超过 1.5 秒没有观测则认为目标已离场
        if ((KFs[i].last_time) > 1.5) {
            KFs.erase(KFs.begin() + i);
        } else {
            pcl::PointXYZRGB point;
            // 用滤波后的位置，而不是原始观测
            point.x = KFs[i].predict_point.x;
            point.y = KFs[i].predict_point.y;
            // z 固定，只为在三维视图里可见
            point.z = 1.5;
            // 视觉投票得到的颜色：0 蓝 / 2 红 / 1 未知
            int color = KFs[i].get_color();
            switch (color) {
            case 0:
                // 蓝方：蓝色通道拉满
                point.b = 255;
                break;

            case 2:
                // 红方：红色通道拉满
                point.r = 255;
                break;

            default:
                // 颜色未知：用构造时分配的随机色区分航迹
                point.r = KFs[i].color[0];
                point.g = KFs[i].color[1];
                point.b = KFs[i].color[2];
                break;
            }
            cloud_filtered->points.push_back(point);
        }
    }
    // 统一使用场地坐标系
    cloud_filtered->header.frame_id = "rm_frame";
    sensor_msgs::msg::PointCloud2 output;
    // PCL 点云转回 ROS 消息
    pcl::toROSMsg(*cloud_filtered, output);
    output.header.frame_id = "rm_frame";
    // 沿用输入时间戳，方便下游对齐
    output.header.stamp = msg->header.stamp;
    pub_->publish(output);
    // 耗时统计终点
    auto  end_time = std::chrono::steady_clock::now();
    // 本帧处理耗时，单位毫秒，当前仅计算未使用
    float dur_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - now_time)
                         .count();
    // 融合结果：按编号填写两个阵营各 6 组坐标
    vision_interface::msg::DetectResult detect_msg;
    for (auto kf : KFs) {
        // 从未被视觉命中过的航迹不对外发布
        if (kf.detect_history.size() == 0)
            continue;
        // 蓝方目标写入 blue 数组
        if (kf.get_color() == 0) {
            // 装甲板编号即数组下标
            int number = kf.get_number();
            detect_msg.blue_x[number] = kf.predict_point.x;
            detect_msg.blue_y[number] = kf.predict_point.y;
        }
        // 红方目标写入 red 数组
        if (kf.get_color() == 2) {
            int number = kf.get_number();
            detect_msg.red_x[number] = kf.predict_point.x;
            detect_msg.red_y[number] = kf.predict_point.y;
        }
    }
    // 己方为蓝时，需要把己方半场坐标系翻到对方半场
    if (match_info.self_color == 0) {
        for (int i = 0; i < 6; i++) {
            if (detect_msg.blue_x[i] != 0 && detect_msg.blue_y[i] != 0) {
                // 场地长 28 米，x 轴翻转
                detect_msg.blue_x[i] = 28 - detect_msg.blue_x[i];
                // 场地宽 15 米，y 轴翻转
                detect_msg.blue_y[i] = 15 - detect_msg.blue_y[i];
            }
            if (detect_msg.red_x[i] != 0 && detect_msg.red_y[i] != 0) {
                detect_msg.red_x[i] = 28 - detect_msg.red_x[i];
                detect_msg.red_y[i] = 15 - detect_msg.red_y[i];
            }
        }
    }
    // 发布带编号的融合结果
    radar_detect_pub_->publish(detect_msg);

    // 发往哨兵的消息只携带敌方坐标
    vision_interface::msg::Radar2Sentry radar_msg;
    // 己方为蓝时敌方即红方
    if (match_info.self_color == 0) {
        for (int i = 0; i < 6; i++) {
            radar_msg.radar_enemy_x[i] = detect_msg.red_x[i];
            radar_msg.radar_enemy_y[i] = detect_msg.red_y[i];
        }

        // 发布敌方坐标
        radar_pub_->publish(radar_msg);
    }
}
}  // namespace radar_core
// 注册为可组合节点，供 launch 文件在进程内加载
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::KalmanFilter)

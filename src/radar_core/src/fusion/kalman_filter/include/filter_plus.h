#include <opencv2/core/hal/interface.h>  // CV_32F 等矩阵类型标记
#include <vector>                        // std::vector 历史队列
#include <opencv2/core/types.hpp>        // cv::Scalar
#include <pcl/impl/point_types.hpp>      // pcl::PointXY
#include <rclcpp/rclcpp.hpp>             // rclcpp::Time
#include <rclcpp/time.hpp>
#include "opencv2/opencv.hpp"            // cv::KalmanFilter
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#pragma once

// 单目标卡尔曼滤波器
// 状态向量 [x, vx, y, vy]，观测向量 [x, y]，匀速(CV)运动模型
// 一个实例对应一条航迹，航迹的增删由节点侧负责
class Kalman_filter_plus {
private:
    // OpenCV 卡尔曼核心：4 维状态 / 2 维观测 / 无控制量
    cv::KalmanFilter KF;

public:
    // 计算两个平面点的欧氏距离，量纲与输入一致（米）
    float Distance(pcl::PointXY& a, pcl::PointXY& b)
    {
        // 横向差与纵向差的平方和开方
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
    }

    // 距上次调用本函数（或构造时刻）的真实耗时，单位秒
    float get_time()
    {
        // steady_clock 不受系统时间跳变影响，适合测时间间隔
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - timer);
        // 毫秒换算为秒
        return duration.count() / 1000.0;
    }

    // 距上次量测更新的累计时间，超过阈值即判定航迹丢失
    float                                        last_time = 0;
    // 计时基准点，每次预测后刷新
    std::chrono::steady_clock::time_point        timer;
    // 航迹允许的最大无观测时长，单位秒（当前由节点侧硬编码替代）
    float                                        delete_time = 2.0;
    // 位置历史队列：(ROS 时间戳, 雷达点)，供视觉做时间对齐查询
    std::vector<std::pair<double, pcl::PointXY>> history;
    // 视觉标签历史队列：(颜色, 装甲板编号)，供颜色与编号投票
    // 颜色约定 0 为蓝、2 为红
    std::vector<std::pair<int, int>>             detect_history;
    // 两个历史队列的最大长度，超出后丢弃最旧元素
    int                                          max_history = 20;

    // 当前时刻的滤波输出位置，关联门限与发布都用它
    pcl::PointXY predict_point;
    // 视觉与雷达判为同一目标的空间容差，单位米
    float        detect_r = 1;
    // 目标典型速度，当前未参与门限计算
    float        car_speed = 2;
    // 目标可能的最大速度，用于按采样间隔放大关联门限
    float        car_max_speed = 2.5;
    // 颜色未知时用于可视化区分航迹的随机色
    cv::Scalar   color;
    // 本帧是否已被点云观测更新（当前未被读取）
    bool         has_updated = false;
    // 过程噪声矩阵 4x4，仅作成员占位，实际使用 KF 内部矩阵
    cv::Mat      Q = cv::Mat::zeros(4, 4, CV_32F);
    // 观测噪声矩阵 2x2，同上
    cv::Mat      R = cv::Mat::zeros(2, 2, CV_32F);
    // 标称采样间隔，单位秒，用于构造状态转移矩阵与过程噪声
    float        dt_ = 0.1f;
    // x 方向过程噪声强度
    float        sigma_q_x = 50.0f;
    // y 方向过程噪声强度
    float        sigma_q_y = 50.0f;
    // x 方向观测噪声方差
    float        sigma_r_x = 0.1f;
    // y 方向观测噪声方差
    float        sigma_r_y = 0.1f;

    // 用首个观测点建立航迹：位置取观测值，速度置零
    Kalman_filter_plus(pcl::PointXY& input, rclcpp::Time time)
    {
        // 尚无速度估计，直接用观测点作为当前位置
        predict_point = input;
        // 记录初始观测，供后续视觉时间对齐
        history.push_back(std::make_pair(GetTimeByRosTime(time), input));
        // 启动计时，首次预测的 dt 由此算起
        timer = std::chrono::steady_clock::now();
        // 三通道随机取值，仅用于可视化时区分不同航迹
        color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        // 状态维度：x、vx、y、vy
        int          stateSize = 4;
        // 观测维度：x、y
        int          measSize = 2;
        // 无控制输入
        int          contrSize = 0;
        // 统一单精度浮点，与 PCL/ROS 侧的 float 数据一致
        unsigned int type = CV_32F;
        // 分配内部矩阵并置零
        KF.init(stateSize, measSize, contrSize, type);

        // 状态列向量，随后写入初始位置
        cv::Mat state(stateSize, 1, type);
        // 观测列向量
        cv::Mat meas(measSize, 1, type);
        // 观测 x
        meas.at<float>(0) = input.x;
        // 观测 y
        meas.at<float>(1) = input.y;

        // 初始位置 x
        state.at<float>(0) = meas.at<float>(0);
        // 初始位置 y，注意状态第 2 个元素是 y 而不是 vy
        state.at<float>(2) = meas.at<float>(1);
        // 直接作为后验状态，跳过首次预测
        KF.statePost = state;
        // 状态转移矩阵 F：位置 = 位置 + 速度 * dt
        KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, dt_, 0, 0, 0, 1,
                               0, 0, 0, 0, 1, dt_, 0, 0, 0, 1);

        // 观测矩阵 H：只取状态中的 x 与 y
        KF.measurementMatrix =
            (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);

        // 过程噪声 Q：连续白噪声加速度模型，x/y 两块对角
        // 块内元素依次为 dt^3/3、dt^2/2、dt^2/2、dt
        KF.processNoiseCov =
            (cv::Mat_<float>(4, 4) << sigma_q_x * pow(dt_, 3) / 3,
             sigma_q_x * pow(dt_, 2) / 2, 0, 0, sigma_q_x * pow(dt_, 2) / 2,
             sigma_q_x * pow(dt_, 1), 0, 0, 0, 0,
             sigma_q_y * pow(dt_, 3) / 3, sigma_q_y * pow(dt_, 2) / 2, 0, 0,
             sigma_q_y * pow(dt_, 2) / 2, sigma_q_y * pow(dt_, 1));

        // 观测噪声 R：x/y 相互独立，对角即各自方差
        KF.measurementNoiseCov =
            (cv::Mat_<float>(2, 2) << sigma_r_x, 0, 0, sigma_r_y);

        // 初始后验协方差取单位阵，表示初始位置不确定度为 1
        setIdentity(KF.errorCovPost, cv::Scalar::all(1));
        // 构造即视为已完成一次更新
        has_updated = true;
    }

    ~Kalman_filter_plus() {}

    // 返回航迹颜色：0 蓝、2 红、1 未知，取视觉标签的多数票
    int get_color()
    {
        // 还没有任何视觉标签时返回 1，表示颜色未知
        if (detect_history.size() == 0) {
            return 1;
        }
        // 红方票数
        int red = 0;
        // 蓝方票数
        int blue = 0;
        for (auto& color : detect_history) {
            // first 为颜色，约定 0 表示蓝方
            if (color.first == 0) {
                blue++;
            } else {
                // 其余标签一律按红方计票
                red++;
            }
        }
        // 红票多于蓝票则判定为红方
        if (red > blue) {
            return 2;
        } else {
            // 蓝票更多或两方平票时判定为蓝方
            return 0;
        }
    }

    // 返回装甲板编号 0-5，取同色标签中出现次数最多的编号
    int get_number()
    {
        // 先定颜色，编号只在同色标签内统计
        int                color = get_color();
        // 编号 -> 出现次数
        std::map<int, int> number_map;
        for (auto& number : detect_history) {
            // first 为颜色，second 为编号，只统计与判定颜色一致的票
            if (number.first == color) {
                number_map[number.second]++;
            }
        }
        // 记录最高票数
        int max_count = 0;
        // 记录最高票数对应的编号
        int max_number = 0;
        for (auto& entry : number_map) {
            if (entry.second > max_count) {
                max_count = entry.second;
                max_number = entry.first;
            }
        }
        // 没有任何同色标签时返回 0
        return max_number;
    }

    // 用一次雷达观测校正状态，并重置丢失计时
    void update(pcl::PointXY& input, rclcpp::Time time)
    {
        // 构造 2x1 观测列向量
        cv::Mat meas = cv::Mat::zeros(2, 1, CV_32F);
        // 观测 x
        meas.at<float>(0) = input.x;
        // 观测 y
        meas.at<float>(1) = input.y;
        // 卡尔曼校正：融合先验预测与本次观测，更新 statePost
        KF.correct(meas);
        // 输出位置取校正后的 x
        predict_point.x = KF.statePost.at<float>(0);
        // 输出位置取校正后的 y
        predict_point.y = KF.statePost.at<float>(2);
        // 标记本帧已有观测
        has_updated = true;
        // 收到观测，丢失计时清零
        last_time = 0;
        // 复制一份观测点，避免改动调用方数据
        auto temp_point = input;
        // 记录本次观测的时间戳与位置
        history.push_back(
            std::make_pair(GetTimeByRosTime(time), temp_point));
        // 历史超长时丢弃最旧的一条
        if (history.size() > max_history) {
            history.erase(history.begin());
        }
    }

    // 无观测时把状态向前推进一步，维持航迹的预测位置
    void update_predict_point()
    {
        // 用实测间隔更新 dt，供关联门限与丢失计时使用
        dt_ = get_time();
        // 刷新计时基准点
        timer = std::chrono::steady_clock::now();
        // 状态一步预测，返回预测位置
        auto result = KF.predict();
        // 累加无观测时长
        last_time += dt_;
        // 预测位置 x
        predict_point.x = result.at<float>(0);
        // 预测位置 y
        predict_point.y = result.at<float>(2);
    }

    // 判断观测点是否落在这条航迹的关联门限内
    bool match(pcl::PointXY& input)
    {
        // 门限 = 一个采样周期内的最大位移 + 固定容差
        if (Distance(predict_point, input) <
            car_max_speed * dt_ + detect_r) {
            return true;
        } else {
            return false;
        }
    }

    // 用一帧视觉结果给本航迹投一票颜色与编号
    // 视觉只提供标签，不参与状态校正
    void camera_match(rclcpp::Time& time, pcl::PointXY& input, int color,
                      int number)
    {
        // 雷达与视觉允许的最大时间差，单位秒
        const double TIME_THRESHOLD = 1.0f;
        // 视觉帧时间戳，统一换算为秒
        double       input_time = GetTimeByRosTime(time);
        // 当前最小时间差，初值取一个足够大的数
        double       differ_time = 1000;
        // 与视觉帧时间最接近的历史雷达点
        pcl::PointXY match_point;
        for (auto& point : history) {
            // first 为时间戳，时间差取绝对值
            auto differ = abs(point.first - input_time);
            // 保留时间上更接近的一帧
            if (differ < differ_time) {
                differ_time = differ;
                match_point = point.second;
            }
        }
        // 时间差过大说明两者不同步，放弃本次投票
        if (differ_time > TIME_THRESHOLD) {
            return;
        }
        // 空间上足够近才认为是同一个目标
        if (Distance(match_point, input) < detect_r) {
            // 记下颜色与装甲板编号
            detect_history.push_back(std::make_pair(color, number));
            // 投票窗口超长时丢弃最旧的一票
            if (detect_history.size() > max_history) {
                detect_history.erase(detect_history.begin());
            }
        }
    }

    // ROS 时间转秒，使其可与纯数值时间戳比较
    static double GetTimeByRosTime(rclcpp::Time& ros_time)
    {
        // 纳秒换算为秒
        double ros_time_value = ros_time.nanoseconds() / 1e9;
        return ros_time_value;
    }
};

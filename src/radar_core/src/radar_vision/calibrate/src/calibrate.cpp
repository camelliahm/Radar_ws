#include "calibrate.h"
#include <string>

namespace radar_core {

Calibrate::Calibrate(const rclcpp::NodeOptions& options)
    : Node("radar_calibrate_node", options)
{
    std::cout << "Calibrate start" << std::endl;
    // 主窗口显示整幅图像，"ROI" 小窗口显示鼠标附近放大后的局部，便于精确点选
    cv::namedWindow("calibrate", cv::WINDOW_AUTOSIZE);
    cv::resizeWindow("calibrate", 1440, 900);
    cv::moveWindow("calibrate", 1920 - 1440, 1080 - 900);
    cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE);
    cv::resizeWindow("ROI", 400, 400);
    cv::moveWindow("ROI", 0, 0);
    cv::setMouseCallback("calibrate", mousecallback, 0);
    cv::FileStorage fs;
    fs.open("./config/camera_params.yaml", cv::FileStorage::READ);
    fs["camera_matrix"] >> camera_matrix;
    fs["dist_coeffs"] >> dist_coeffs;
    fs.release();

    // 世界坐标点与用户点选顺序严格对应，不能调换
    real_points.push_back(self_FORTRESS);
    real_points.push_back(self_Tower);
    real_points.push_back(enemy_Base);
    real_points.push_back(enemy_Tower);
    real_points.push_back(enemy_High);
    parser_ = new parser();

    // 启动时打印 ASCII 横幅，确认节点已正常拉起
    RCLCPP_INFO(this->get_logger(), "\n"
                                    "  ______    ____  ______\n"
                                    " /_  __/   / __ \\/_  __/\n"
                                    "  / /_____/ / / / / /   \n"
                                    " / /_____/ /_/ / / /    \n"
                                    "/_/     /_____/ /_/     ");

    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Calibrate::callback, this, std::placeholders::_1));
    compressed_image_sub =
        this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "compressed_image", rclcpp::SensorDataQoS(),
            std::bind(&Calibrate::compressed_callback, this,
                      std::placeholders::_1));
    std::cout << "Calibrate end" << std::endl;
}

void Calibrate::callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    auto    img = cv_bridge::toCvCopy(msg, "bgr8")->image;
    // 统一缩放到固定尺寸，保证鼠标坐标与后续缩放换算关系恒定
    cv::Mat calib_img;
    cv::resize(img, calib_img, cv::Size(1536, 1125));
    cvimage_ = calib_img;
    if (is_calibrating) {
        // 标定模式：显示已点选数量，并提示按 n 确认
        cv::putText(img, std::to_string(pick_points.size()),
                    cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(0, 0, 255), 2);
        cv::putText(img, "Press 'n' to add good point", cv::Point(50, 400),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
        // 点数凑齐即求解外参，并让解算器立即用新外参刷新区域投影
        if (pick_points.size() == real_points.size()) {
            solve();
            parser_->Change_Matrix();
        }
    } else {
        // 预览模式：绘制当前外参下各区域的多边形，方便判断是否需要重标
        parser_->draw_ui(img);
        cv::putText(img, "Press Enter to Calibrate !!!", cv::Point(50, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
    }
    auto temp = img.clone();
    cv::resize(img, img, cv::Size(1536, 1125));
    cv::imshow("calibrate", img);
    auto key = cv::waitKey(10);
    switch (key) {
    case 13:  // 回车：进入标定模式
        is_calibrating = true;
        break;
    default:
        break;
    }
}

void Calibrate::compressed_callback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    // 与 callback 逻辑一致，区别只是先 JPEG 解码
    auto    img = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    cv::Mat calib_img;
    cv::resize(img, calib_img, cv::Size(1536, 1125));
    cvimage_ = calib_img;
    if (is_calibrating) {
        cv::putText(img, std::to_string(pick_points.size()),
                    cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3,
                    cv::Scalar(0, 0, 255), 2);
        if (pick_points.size() == real_points.size()) {
            solve();
            parser_->Change_Matrix();
        }
    } else {
        parser_->draw_ui(img);
        cv::putText(img, "Press Enter to Calibrate !!!", cv::Point(50, 200),
                    cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 2);
    }
    auto temp = img.clone();
    cv::resize(img, img, cv::Size(1536, 1125));
    cv::imshow("calibrate", img);
    auto key = cv::waitKey(10);
    switch (key) {
    case 13:
        is_calibrating = true;
        break;
    default:
        break;
    }
}

void mousecallback(int event, int x, int y, int flags, void* userdata)
{
    int temp_key = 0;

    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        // 左键点击后进入微调循环：用 WASD 逐像素修正位置，按 n 确认落点
        if (is_calibrating) {
            do {
                temp_key = cv::waitKey(10);
                switch (temp_key) {
                case 'w':
                    y -= 1;
                    break;
                case 'a':
                    x -= 1;
                    break;
                case 's':
                    y += 1;
                    break;
                case 'd':
                    x += 1;
                    break;
                }
                // 限制在图像内并留 50px 边距，保证 ROI 裁剪不越界
                x = std::max(50, std::min(x, cvimage_.cols - 50));
                y = std::max(50, std::min(y, cvimage_.rows - 50));
                // 取 100x100 局部并放大到 400x400，中心十字线指示精确位置
                cv::Mat roi = cvimage_(cv::Rect(x - 50, y - 50, 100, 100));
                cv::Mat dst;
                cv::resize(roi, dst, cv::Size(400, 400));
                cv::line(dst, cv::Point(200, 100), cv::Point(200, 300),
                         cv::Scalar(0, 0, 255), 1);
                cv::line(dst, cv::Point(100, 200), cv::Point(300, 200),
                         cv::Scalar(0, 0, 255), 1);
                cv::imshow("ROI", dst);

            } while (temp_key != 'n');

            // 窗口显示的是 1536x1125 缩放图，原图为 3072x2048 量级，
            // 故需乘回缩放倍数还原到原图坐标系
            x *= 1.3333333333 * 2;
            y *= 1.3333333333 * 2;
            std::cout << "x:" << x << " y:" << y << std::endl;
            pick_points.push_back(cv::Point2f(x, y));
        }
        break;

    case cv::EVENT_MOUSEMOVE:
        // 鼠标移动时仅刷新 ROI 放大窗口，不做选点
        if (x > cvimage_.cols - 50 || y > cvimage_.rows - 50 || x < 50 ||
            y < 50)
            break;
        cv::Mat roi = cvimage_(cv::Rect(x - 50, y - 50, 100, 100));
        cv::Mat dst;
        cv::resize(roi, dst, cv::Size(400, 400));
        cv::line(dst, cv::Point(200, 100), cv::Point(200, 300),
                 cv::Scalar(0, 0, 255), 1);
        cv::line(dst, cv::Point(100, 200), cv::Point(300, 200),
                 cv::Scalar(0, 0, 255), 1);
        cv::imshow("ROI", dst);
        break;
    }
}

// 用 5 组「世界坐标 - 像素坐标」求出外参，EPNP 是点位少时的稳定选择
void Calibrate::solve()
{
    cv::solvePnP(real_points, pick_points, camera_matrix, dist_coeffs, rvec,
                 tvec, 0, cv::SOLVEPNP_EPNP);
    std::cout << "rvec:" << rvec << std::endl;
    std::cout << "tvec:" << tvec << std::endl;
    // 覆盖写入外参文件，解算节点下次启动或 Change_Matrix 时会读取
    cv::FileStorage fs;
    fs.open("./config/out_matrix.yaml", cv::FileStorage::WRITE);
    fs << "world_rvec" << rvec;
    fs << "world_tvec" << tvec;
    fs.release();
    pick_points.clear();
    is_calibrating = false;
}
}  // namespace radar_core

RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::Calibrate);

#include "detect.h"

#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/utilities.hpp>
#define TDT_INFO(msg) std::cout << msg << std::endl
#define MAX_CARS 12    // 单帧最多处理的整车数，超出视为误检
#define MAX_ARMORS 20  // 单帧最多处理的装甲板数
namespace radar_core {
unsigned int count_img = 0;  // 调试存图的全局计数器

// 用蓝/红通道的平均差值判断装甲板颜色，返回 0=蓝方、2=红方
int getColor(cv::Mat& img)
{
    std::vector<cv::Mat> channels;
    cv::split(img, channels);  // BGR 顺序
    cv::Mat    blueMinusRed = channels[0] - channels[2];
    cv::Mat    redMinusBlue = channels[2] - channels[0];
    cv::Scalar avgBlueMinusRed = cv::mean(blueMinusRed);
    cv::Scalar avgRedMinusBlue = cv::mean(redMinusBlue);
    cv::Scalar avgGreen = cv::mean(channels[1]);
    // 蓝减红的均值更大说明画面整体偏蓝
    if (avgBlueMinusRed[0] > avgRedMinusBlue[0]) {
        return 0;
    } else {
        return 2;
    }
}

// small 的四个角点是否都落在 big 内部
bool isRectInside(const cv::Rect& small, const cv::Rect& big)
{
    bool topLeftInside = big.contains(small.tl());
    bool topRightInside =
        big.contains(cv::Point(small.x + small.width, small.y));
    bool bottomLeftInside =
        big.contains(cv::Point(small.x, small.y + small.height));
    bool bottomRightInside = big.contains(
        cv::Point(small.x + small.width, small.y + small.height));
    return (topLeftInside && topRightInside && bottomLeftInside &&
            bottomRightInside);
}

// yolo::Box 版本的包含判断，用于确认装甲板框在整车框内
bool isBoxInside(const yolo::Box& small, const yolo::Box& big)
{
    cv::Rect small_rect(small.left, small.top, small.right - small.left,
                        small.bottom - small.top);
    cv::Rect big_rect(big.left, big.top, big.right - big.left,
                      big.bottom - big.top);
    return isRectInside(small_rect, big_rect);
}

// 把矩形裁剪到图像范围内，避免 ROI 取值越界崩溃
cv::Rect getSafeRect(cv::Mat& image, cv::Rect& rect)
{
    cv::Rect save_rect;
    save_rect.x = std::max(0, rect.x);
    save_rect.y = std::max(0, rect.y);
    save_rect.width = std::min(image.cols - save_rect.x, rect.width);
    save_rect.height = std::min(image.rows - save_rect.y, rect.height);
    return save_rect;
}

Detect::Detect(const rclcpp::NodeOptions& node_options)
    : Node("radar_detect_node", node_options)
{
    //cv::namedWindow("detect", cv::WINDOW_NORMAL);

    // TensorRT 推理依赖 CUDA，先做一次可用性检查
    std::cout << "Checking CUDA with nvidia-smi...\n";
    if (system("nvidia-smi") == 0) {
        RCLCPP_INFO(this->get_logger(), "CUDA is available.");
    } else {
        RCLCPP_ERROR(this->get_logger(), "CUDA is not available. Exiting.");
        rclcpp::shutdown();
    }
    // 三个模型各自的 engine 路径
    cv::FileStorage fs;
    fs.open("./config/detect_params.yaml", cv::FileStorage::READ);
    fs["yolo_path"] >> yolo_path;
    fs["armor_path"] >> armor_path;
    fs["classify_path"] >> classify_path;
    fs.release();

    // engine 文件不存在时调用 onnx2trt.py 现场转换。
    // 批大小按各级实际数量确定：整车最多 2 台、装甲板最多 12 块、分类最多 20 块
    std::ifstream file1(yolo_path.c_str());
    if (!file1.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/RM2025.onnx "
               "--saveEngine=model/TensorRT/yolo.engine "
               "--minBatch 1 "
               "--optBatch 1 "
               "--maxBatch 2 "
               "--Shape=1280x1280 "
               "--input_name=images");
    } else {
        TDT_INFO("Load yolo engine!");
    }
    std::ifstream file2(armor_path.c_str());
    if (!file2.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/armor_yolo.onnx "
               "--saveEngine=model/TensorRT/armor_yolo.engine "
               "--minBatch 1 "
               "--optBatch 5 "
               "--maxBatch 12 "
               "--Shape=192x192 "
               "--input_name=images");
    } else {
        TDT_INFO("Load armor_yolo engine!");
    }
    std::ifstream file3(classify_path.c_str());
    if (!file3.good()) {
        system("python3 src/utils/onnx2trt.py "
               "--onnx=model/ONNX/classify.onnx "
               "--saveEngine=model/TensorRT/classify.engine "
               "--minBatch 1 "
               "--optBatch 10 "
               "--maxBatch 20 "
               "--Shape=224x224 "
               "--input_name=input");
    } else {
        TDT_INFO("Load classify engine!");
    }
    std::cout << "yolo_path:" << yolo_path << "\n";
    std::cout << "armor_path:" << armor_path << "\n";
    std::cout << "classify_path:" << classify_path << "\n";
    this->classifier =
        classify::load(classify_path, classify::Type::densenet121);
    TDT_INFO("Load classify engine success!");

    // 装甲板阈值放宽（0.4）以便召回，整车阈值收紧（0.65）以减少误检
    this->armor_yolo = yolo::load(armor_path, yolo::Type::V8, 0.4f, 0.45f);
    TDT_INFO("Load armor_yolo engine success!");
    this->yolo = yolo::load(yolo_path, yolo::Type::V8, 0.65f, 0.45f);
    TDT_INFO("Load yolo engine success!");
    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "camera_image", rclcpp::SensorDataQoS(),
        std::bind(&Detect::callback, this, std::placeholders::_1));
    pub = this->create_publisher<vision_interface::msg::DetectResult>(
        "detect_result", rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "Detect node has been started.");
}

void Detect::callback(const std::shared_ptr<sensor_msgs::msg::Image> msg)
{
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::cout << "Detecting..." << std::endl;
    auto             img = cv_bridge::toCvShare(msg, "bgr8")->image;
    radar_core::Image image(img.data, img.cols, img.rows);

    // 第一级：整图检测机器人与装甲板
    auto result = yolo->forward(image);
    if (result.size() == 0) {
        RCLCPP_INFO(this->get_logger(), "No Car!");
        return;
    } else if (result.size() > MAX_CARS) {
        RCLCPP_INFO(this->get_logger(), "Too Many Car!");
        return;
    }

    std::vector<radar_core::Image> images;
    std::vector<cv::Mat>          car_imgs;
    std::vector<Car>              cars;
    // 只保留类别 0/1（机器人）的框，丢弃其他类别
    for (auto& box : result) {
        if (box.class_label == 0 || box.class_label == 1) {
            Car car;
            car.car = box;
            cars.push_back(car);
        }
    }

    // 按整车框裁出子图，作为第二级检测的批量输入
    for (auto& car : cars) {
        auto     temp_rect = cv::Rect(car.car.left, car.car.top,
                                      car.car.right - car.car.left,
                                      car.car.bottom - car.car.top);
        cv::Rect temp_car_rect = getSafeRect(img, temp_rect);
        auto     car_img = img(temp_car_rect);
        car_imgs.push_back(car_img.clone());  // clone 保证数据连续
        car.car_rect = temp_car_rect;
    }

    for (auto& car_img : car_imgs) {
        auto image =
            radar_core::Image(car_img.data, car_img.cols, car_img.rows);
        images.push_back(image);
    }

    // 第二级：对每台车的子图批量检测装甲板，结果按输入顺序返回
    auto armor_boxes = armor_yolo->forwards(images);
    bool has_armor = false;
    for (int i = 0; i < armor_boxes.size(); i++) {
        if (armor_boxes[i].size() == 0) {
            continue;
        } else {
            cars[i].armors = armor_boxes[i];
            has_armor = true;
        }
    }  // 将armor_boxes存储到cars中
    if (!has_armor) {
        RCLCPP_INFO(this->get_logger(), "No Armor!");
        return;
    }
    // for(int i=0;i<cars.size();i++){
    //   if(cars[i].armors.size()==0){continue;}
    //   for(auto &armor:cars[i].armors){
    //     cv::Rect rect_img(
    //       armor.left+cars[i].car.left,
    //       armor.top+cars[i].car.top,
    //       armor.right-armor.left,
    //       armor.bottom-armor.top);
    //     // cv::rectangle(img,rect_img,cv::Scalar(255,255,255),2);
    //     cv::imwrite("debug_out/armor_detect/"+std::to_string(count_img++)+".jpg",img(rect_img));
    //     }
    // }//这段是用来打印或者保存armor的图片
    std::vector<cv::Mat>         armor_imgs;
    std::vector<classify::Image> armor_images;

    // 装甲板框是相对整车裁剪图的坐标，需加回整车左上角偏移才能取到原图
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor : car.armors) {
            cv::Rect rect_img_1(
                armor.left + car.car.left, armor.top + car.car.top,
                armor.right - armor.left, armor.bottom - armor.top);
            cv::Rect rect_img = getSafeRect(img, rect_img_1);
            auto     armor_img = img(rect_img);
            armor_imgs.push_back(armor_img.clone());
        }
    }
    for (auto& armor_img : armor_imgs) {
        auto image =
            classify::Image(armor_img.data, armor_img.cols, armor_img.rows);
        armor_images.push_back(image);
    }
    // 第三级：批量识别装甲板上的数字（机器人编号）
    auto armor_result = classifier->forwards(armor_images);

    // 保存用来训练分类
    //  for(int i=0;i<armor_imgs.size();i++){
    //    cv::imwrite("debug_out/fenqu_armor/"+std::to_string(armor_result[i])+"/"+std::to_string(count_img++)+".jpg",armor_imgs[i]);
    //  }

    // 把分类结果按顺序回填到各装甲板；armor_result 边取边删以保持与遍历顺序一致
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        for (auto& armor : car.armors) {
            armor.class_label = armor_result[0];
            armor_result.erase(armor_result.begin());
            if (debug) {
                cv::putText(img, std::to_string(armor.class_label),
                            cv::Point(armor.left + car.car.left,
                                      armor.top + car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
    }
    vision_interface::msg::DetectResult detect_result;
    for (auto& car : cars) {
        if (car.armors.size() == 0) {
            continue;
        }
        cv::Rect max_rect;
        float    max_confidence = 0;
        // 每台车取置信度最高的装甲板作为该车的代表；
        // 排除 0（背景）和 5（可能为未训练类别）两类无效标签
        for (auto& armor : car.armors) {
            if (armor.class_label != 0 && armor.class_label != 5 &&
                armor.confidence > max_confidence) {
                max_rect = cv::Rect(
                    armor.left + car.car.left, armor.top + car.car.top,
                    armor.right - armor.left, armor.bottom - armor.top);
                max_confidence = armor.confidence;
                car.number = armor.class_label;
            }
        }
        if (max_confidence == 0) {
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);
            continue;
        }
        auto safe_rect = getSafeRect(img, max_rect);
        auto max_mat = img(safe_rect);

        // 用最佳装甲板区域的颜色判断红蓝方，并计算其中心像素
        car.color = getColor(max_mat);
        car.center = cv::Point2f(max_rect.x + max_rect.width / 2,
                                 max_rect.y + max_rect.height / 2);
        if (car.color == 0) {
            // 按编号寻址：number 1..6 对应数组下标 0..5
            detect_result.blue_x[car.number - 1] = car.center.x;
            detect_result.blue_y[car.number - 1] = car.center.y;
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: blue car center is 0");
            }
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 0, 0), 2);
                cv::putText(img, std::to_string(car.car.confidence),
                            cv::Point(car.car.left, car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
        if (car.color == 2) {
            detect_result.red_x[car.number - 1] = car.center.x;
            detect_result.red_y[car.number - 1] = car.center.y;
            if (car.center.x * car.center.y == 0 && car.number != 0) {
                RCLCPP_ERROR(this->get_logger(),
                             "Error: red car center is 0");
            }
            if (debug) {
                cv::rectangle(img, car.car_rect, cv::Scalar(0, 0, 255), 2);
                cv::putText(img, std::to_string(car.car.confidence),
                            cv::Point(car.car.left, car.car.top),
                            cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255), 2);
            }
        }
        if (car.color == 1) {
            if (debug)
                cv::rectangle(img, car.car_rect, cv::Scalar(255, 255, 255),
                              2);
        }
    }
    // 未检测到的编号保持数组默认值 0，下游据此判断"本帧无该编号"
    detect_result.header.stamp = msg->header.stamp;
    pub->publish(detect_result);
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                  begin);
    std::cout << "Detect Time: " << time_used.count() * 1000 << "ms"
              << std::endl;
    cv::Mat final_img;
    cv::resize(img, final_img, cv::Size(1536, 1125));
    //cv::imshow("detect", final_img);
    // 键盘 'r' 切换调试绘制开关
    auto key = cv::waitKey(1);
    if (key == 'r') {
        debug = !debug;
    }
}
}  // namespace radar_core
RCLCPP_COMPONENTS_REGISTER_NODE(radar_core::Detect)
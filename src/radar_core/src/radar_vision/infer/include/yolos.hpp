#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include <NvInfer.h>
#include <future>
#include <memory>
#include <string>
#include <vector>
#include "BaseInfer.hpp"
#include "InferTool.hpp"
#include "logging.h"

using namespace nvinfer1;

// TensorRT 日志回调实例，engine 反序列化时需要传入
static yolov5_predictor::Logger gLogger;

namespace yolo {

// 各检测头后处理方式不同，需按模型类型选择对应的 decode kernel
enum class Type : int {
    V5 = 0,      // 含 objectness
    X = 1,       // 不含归一化，直接输出 0-255
    V3 = 2,      // 类似 V5
    V7 = 3,      // 类似 V5
    V8 = 5,      // 无 objectness，类别置信度直接从索引 4 起
    V8Seg = 6,   // V8 + 实例分割 mask
    V5Face = 7   // V5 + 人脸关键点
};

// 实例分割结果位图，尺寸与原图一致，逐像素 0-255
struct InstanceSegmentMap {
    int            width = 0, height = 0;
    unsigned char* data = nullptr;

    InstanceSegmentMap(int width, int height);
    virtual ~InstanceSegmentMap();
};

// 单个检测框：坐标已映射回原图像素；seg 与 points 依模型类型可选
struct Box {
    float left, top, right, bottom, confidence;
    int   class_label;
    std::shared_ptr<InstanceSegmentMap> seg;     // V8Seg 时的分割掩码
    std::vector<float>                  points;  // V5Face 时的关键点
    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence,
        int class_label)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label)
    {
    }
    Box(float left, float top, float right, float bottom, float confidence,
        int class_label, std::vector<float> points)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label),
          points(points)
    {
    }
};

typedef std::vector<Box> BoxArray;

// 加载 engine 并返回检测推理器。
// confidence_threshold 过滤低置信度框，nms_threshold 控制去重强度
std::shared_ptr<radar_core::Infer<BoxArray>>
load(const std::string& engine_file, Type type,
     float confidence_threshold = 0.45f, float nms_threshold = 0.6f);

// 类型名，用于打印日志
const char*                           type_name(Type type);
// HSV 转 BGR，供随机配色使用
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
// 按 id 生成稳定的伪随机颜色，保证同一类别每次画出的颜色一致
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
};  // namespace yolo

#endif  // __YOLO_HPP__
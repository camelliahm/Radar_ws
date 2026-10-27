#ifndef __BASEINFER_HPP__
#define __BASEINFER_HPP__

#include <tuple>
#include <vector>

namespace radar_core {

// GPU kernel 的默认线程数，用于计算 grid/block 划分
#define GPU_BLOCK_THREADS 512

// 对输入图像的非拥有视图：只记录数据指针和尺寸，
// 不拷贝像素，调用方需保证 forward 期间内存有效
struct Image {
    const void* bgrptr = nullptr;
    int         width = 0, height = 0;

    Image() = default;
    Image(const void* bgrptr, int width, int height)
        : bgrptr(bgrptr), width(width), height(height)
    {
    }
};

// 归一化方式：MeanStd 为 (x*alpha - mean)/std，AlphaBeta 为 x*alpha + beta
enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

// 通道顺序：SwapRB 表示输出时交换 R/B 通道（BGR 输入转 RGB 模型）
enum class ChannelType : int { None = 0, SwapRB = 1 };

// 推理器统一接口：单图 forward 与批量 forwards，
// stream 允许调用方注入自己的 CUDA 流以实现异步
template <typename T>
class Infer {
public:
    virtual T forward(const Image& image, void* stream = nullptr) = 0;
    virtual std::vector<T> forwards(const std::vector<Image>& images,
                                    void* stream = nullptr) = 0;
};

// 归一化参数集合，配合 kernel 里的分支使用
struct Norm {
    float       mean[3];   // 各通道均值
    float       std[3];    // 各通道标准差
    float       alpha, beta;  // 线性缩放/偏移
    NormType    type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // 常用配置：先乘 alpha（一般为 1/255）再减均值除标准差
    static Norm mean_std(const float mean[3], const float std[3],
                         float       alpha = 1 / 255.0f,
                         ChannelType channel_type = ChannelType::None);

    // 仅做线性变换 x*alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0,
                           ChannelType channel_type = ChannelType::None);

    // 不做任何归一化（如部分模型直接把 0-255 送入）
    static Norm None();
};

// 2x3 仿射矩阵，同时保存正向（原图->模型输入）和逆向（模型输出->原图）两套系数。
// 前向用于 warp 预处理，逆向用于把检测框映射回原图坐标。
struct AffineMatrix {
    float i2d[6];
    float d2i[6];

    // 按"保持宽高比 + 居中填充"（letterbox）计算双向矩阵
    void compute(const std::tuple<int, int>& from,
                 const std::tuple<int, int>& to);
};

}  // namespace radar_core

#endif  //__BASEINFER_HPP__

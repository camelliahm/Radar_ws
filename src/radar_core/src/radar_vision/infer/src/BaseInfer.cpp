#include "BaseInfer.hpp"
#include <cstring>

using namespace radar_core;

// 构造 MeanStd 归一化配置：(x*alpha - mean) / std
Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type)
{
    Norm out;
    out.type = NormType::MeanStd;
    out.alpha = alpha;
    out.channel_type = channel_type;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std, std, sizeof(out.std));
    return out;
}

// 构造线性归一化配置：x*alpha + beta
Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type)
{
    Norm out;
    out.type = NormType::AlphaBeta;
    out.alpha = alpha;
    out.beta = beta;
    out.channel_type = channel_type;
    return out;
}

// 不做归一化
Norm Norm::None()
{
    return Norm();
}

// 计算 letterbox 仿射矩阵。
// 取 x/y 缩放比的较小值，保证整幅图都能放进目标尺寸内，
// 剩余边缘用常量填充，避免直接拉伸导致的宽高比失真。
void AffineMatrix::compute(const std::tuple<int, int>& from,
                           const std::tuple<int, int>& to)
{
    float scale_x = std::get<0>(to) / (float)std::get<0>(from);
    float scale_y = std::get<1>(to) / (float)std::get<1>(from);
    float scale = std::min(scale_x, scale_y);
    // 前向矩阵 i2d：原图坐标 -> 模型输入坐标（含居中平移，+0.5 对齐像素中心）
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 +
             scale * 0.5 - 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 +
             scale * 0.5 - 0.5;

    // 求 2x3 仿射矩阵的逆（D 为行列式），得到逆向矩阵 d2i：
    // 模型输出坐标 -> 原图坐标，用于把检测框映射回去
    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    D = D != 0. ? double(1.) / D : double(0.);
    double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D,
           A21 = -i2d[3] * D;
    double b1 = -A11 * i2d[2] - A12 * i2d[5];
    double b2 = -A21 * i2d[2] - A22 * i2d[5];

    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = b1;
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = b2;
}

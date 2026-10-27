#ifndef INFERTOOL_H
#define INFERTOOL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <BaseInfer.hpp>

namespace radar_core {

// 检查 CUDA Runtime 调用是否成功，失败则打印错误名/错误码并直接 abort。
// 推理链路出错继续跑没有意义，故采用 fail-fast 策略。
#define checkRuntime(call)                                                 \
    do {                                                                   \
        auto ___call__ret_code__ = (call);                                 \
        if (___call__ret_code__ != cudaSuccess) {                          \
            INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,  \
                 cudaGetErrorString(___call__ret_code__),                  \
                 cudaGetErrorName(___call__ret_code__),                    \
                 ___call__ret_code__);                                     \
            abort();                                                       \
        }                                                                  \
    } while (0)

// kernel 启动没有返回值，需在调用后主动查询最近一次错误
#define checkKernel(...)                                                   \
    do {                                                                   \
        {                                                                  \
            (__VA_ARGS__);                                                 \
        }                                                                  \
        checkRuntime(cudaPeekAtLastError());                               \
    } while (0)

// 预处理 kernel：对源图做仿射变换 + 双线性插值 + 归一化，
// 输出 CHW 布局的浮点平面数据，直接作为模型输入
__global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_width, int dst_height, uint8_t const_value_st,
    float* warp_affine_matrix_2_3, Norm norm);

// 上述 kernel 的启动封装：按 32x32 的 block 划分 grid
void warp_affine_bilinear_and_normalize_plane(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_width, int dst_height, float* matrix_2_3,
    uint8_t const_value, const radar_core::Norm& norm, cudaStream_t stream);

}  // namespace radar_core

#endif

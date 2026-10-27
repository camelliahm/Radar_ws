#include <iostream>
#include "NvidiaInterface.hpp"
#include "classify.hpp"

namespace classify {

using namespace std;

// 分类后处理：取输出向量中最大值的下标即为预测类别
int postprocess(vector<float>& output_array)
{
    int   max_index = 0;
    float max_value = output_array[0];
    for (int i = 1; i < output_array.size(); i++) {
        if (output_array[i] > max_value) {
            max_value = output_array[i];
            max_index = i;
        }
    }
    return max_index;
}

// 向上对齐到 align 的整数倍，用于给 GPU workspace 内的数据块做 32 字节对齐
inline int upbound(int n, int align = 32)
{
    return (n + align - 1) / align * align;
}

// 分类推理器：预处理 + TensorRT 前向 + argmax
class InferImpl : public Infer<int> {
public:
    shared_ptr<trt::Infer>                         trt_;
    string                                         engine_file_;
    Type                                           type_;
    // 每张图一份预处理临时缓冲（pinned host + device），避免多图串行时互相覆盖
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    trt::Memory<float> input_buffer_, output_array_;  // 模型输入/输出
    int                num_class_ = 0;                // 分类数
    int                network_input_width_, network_input_height_;
    bool               isdynamic_model_ = false;      // 是否为动态 batch
    Norm               normalize_;                    // 归一化配置
    float              mean_[3], std_[3];

    virtual ~InferImpl() = default;

    // 按实际 batch 调整缓冲大小；BaseMemory 内部按容量复用，不会每帧真分配
    void adjust_memory(int batch_size)
    {
        size_t input_numel =
            network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        output_array_.gpu(batch_size * num_class_);
        output_array_.cpu(batch_size * num_class_);

        if ((int)preprocess_buffers_.size() < batch_size) {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(
                    make_shared<trt::Memory<unsigned char>>());
        }
    }

    // 单图预处理：letterbox 缩放到网络输入尺寸，同步做归一化并写入 CHW 布局
    void
    preprocess(int ibatch, const Image& image,
               shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
               AffineMatrix& affine, void* stream = nullptr)
    {
        // 保持宽高比缩放到网络输入尺寸
        affine.compute(
            make_tuple(image.width, image.height),
            make_tuple(network_input_width_, network_input_height_));

        size_t input_numel =
            network_input_width_ * network_input_height_ * 3;
        // 第 ibatch 张图的输入位置
        float*   input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t   size_image = image.width * image.height * 3;
        // workspace 布局：前段放仿射矩阵（32 字节对齐），后段放原图
        size_t   size_matrix = upbound(sizeof(affine.d2i), 32);
        uint8_t* gpu_workspace =
            preprocess_buffer->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device = gpu_workspace + size_matrix;

        uint8_t* cpu_workspace =
            preprocess_buffer->cpu(size_matrix + size_image);
        float*   affine_matrix_host = (float*)cpu_workspace;
        uint8_t* image_host = cpu_workspace + size_matrix;

        cudaStream_t stream_ = (cudaStream_t)stream;
        std::chrono ::high_resolution_clock::time_point a1 =
            std::chrono::high_resolution_clock::now();
        // 先拷到 pinned host 内存，再由 cudaMemcpyAsync 异步上传，
        // 相比直接从普通内存拷贝更容易实现真正的异步
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        std::chrono ::high_resolution_clock::time_point a2 =
            std::chrono::high_resolution_clock::now();
        auto time_used2 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a2 -
                                                                      a1);
        checkRuntime(cudaMemcpyAsync(image_device, image_host, size_image,
                                     cudaMemcpyHostToDevice, stream_));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device,
                                     affine_matrix_host, sizeof(affine.d2i),
                                     cudaMemcpyHostToDevice, stream_));
        // 卷积核内完成缩放 + 双线性插值 + 归一化，边界填 114（与 YOLO 惯例一致）
        warp_affine_bilinear_and_normalize_plane(
            image_device, image.width * 3, image.width, image.height,
            input_device, network_input_width_, network_input_height_,
            affine_matrix_device, 114, normalize_, stream_);
    }

    bool load(const string& engine_file, Type type)
    {
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr)
            return false;

        // 打印张量信息，便于确认 engine 与代码期望是否一致
        trt_->print();

        this->type_ = type;

        // 假定 binding 0 为输入、binding 1 为输出，格式统一为 NCHW
        auto input_dim = trt_->static_dims(0);
        auto output_dims_ = trt_->static_dims(1);
        puts("input_dim");
        for (int i = 0; i < 4; i++) {
            std::cout << input_dim[i] << std::endl;
        }

        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        // ImageNet 标准均值方差，输入需从 BGR 转 RGB
        mean_[0] = 0.485;
        mean_[1] = 0.456;
        mean_[2] = 0.406;
        std_[0] = 0.229;
        std_[1] = 0.224;
        std_[2] = 0.225;
        normalize_ =
            Norm::mean_std(mean_, std_, 1.0 / 255.0, ChannelType::SwapRB);
        num_class_ = output_dims_[1];

        return true;
    }

    virtual int forward(const Image& image, void* stream = nullptr) override
    {
        auto output = forwards({image}, stream);
        if (output.empty())
            return {};
        return output[0];
    }

    virtual vector<int> forwards(const vector<Image>& images,
                                 void* stream = nullptr) override
    {
        int num_image = images.size();
        if (num_image == 0)
            return {};

        // 动态 batch 可任意指定；静态 batch 只能小于等于 engine 的最大批
        auto input_dims = trt_->static_dims(0);
        int  infer_batch_size = input_dims[0];
        if (infer_batch_size != num_image) {
            if (isdynamic_model_) {
                infer_batch_size = num_image;
                input_dims[0] = num_image;
                if (!trt_->set_run_dims(0, input_dims))
                    return {};
            } else {
                if (infer_batch_size < num_image) {
                    INFO("When using static shape model, number of "
                         "images[%d] must be "
                         "less than or equal to the maximum batch[%d].",
                         num_image, infer_batch_size);
                    return {};
                }
            }
        }

        std::chrono ::high_resolution_clock::time_point a1 =
            std::chrono::high_resolution_clock::now();
        adjust_memory(infer_batch_size);  // 调用内存
        std::chrono ::high_resolution_clock::time_point a2 =
            std::chrono::high_resolution_clock::now();
        auto time_used2 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a2 -
                                                                      a1);

        vector<AffineMatrix> affine_matrixs(num_image);
        std::chrono ::high_resolution_clock::time_point a3 =
            std::chrono::high_resolution_clock::now();
        cudaStream_t stream_ = (cudaStream_t)stream;
        // 逐图预处理，结果依次写入 input_buffer_ 的第 i 段
        for (int i = 0; i < num_image; ++i)
            preprocess(i, images[i], preprocess_buffers_[i],
                       affine_matrixs[i], stream);

        // binding 顺序必须与 engine 一致：先输入后输出
        vector<void*> bindings{input_buffer_.gpu(), output_array_.gpu()};

        std::chrono ::high_resolution_clock::time_point a3d1 =
            std::chrono::high_resolution_clock::now();
        auto time_used3 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a3d1 -
                                                                      a3);
        if (!trt_->forward(bindings, stream)) {
            INFO("Failed to tensorRT forward.");
            return {};
        }
        std::chrono ::high_resolution_clock::time_point a3d2 =
            std::chrono::high_resolution_clock::now();
        auto time_used3d2 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a3d2 -
                                                                      a3d1);
        std::chrono ::high_resolution_clock::time_point a3d3 =
            std::chrono::high_resolution_clock::now();
        auto time_used3d3 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a3d3 -
                                                                      a3d2);

        std::chrono ::high_resolution_clock::time_point a3d4 =
            std::chrono::high_resolution_clock::now();
        // 异步拷回后同步一次，保证后续在 CPU 上读到完整结果
        checkRuntime(cudaMemcpyAsync(
            output_array_.cpu(), output_array_.gpu(),
            output_array_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));
        std::chrono ::high_resolution_clock::time_point a4 =
            std::chrono::high_resolution_clock::now();
        auto time_used3d4 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a4 -
                                                                      a3d4);

        auto time_used4 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a4 -
                                                                      a3);
        INFO("forward and decode_kernel_invoker time: %f",
             time_used4.count() * 1000);
        std::chrono ::high_resolution_clock::time_point a5 =
            std::chrono::high_resolution_clock::now();
        // 逐 batch 取该类别的 argmax
        vector<int> arrout(num_image);
        for (int ib = 0; ib < num_image; ++ib) {
            float*        parray = output_array_.cpu() + ib * num_class_;
            vector<float> output(num_class_);
            for (int o = 0; o < num_class_; o++) {
                output[o] = parray[o];
            }

            arrout[ib] = postprocess(output);
        }

        return arrout;
    }
};
Infer<int>* loadraw(const std::string& engine_file, Type type)
{
    InferImpl* impl = new InferImpl();
    if (!impl->load(engine_file, type)) {
        delete impl;
        impl = nullptr;
    }
    return impl;
}
shared_ptr<Infer<int>> load(const string& engine_file, Type type)
{
    return std::shared_ptr<InferImpl>(
        (InferImpl*)loadraw(engine_file, type));
}
}  // namespace classify

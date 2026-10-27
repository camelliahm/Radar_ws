#include "BaseInfer.hpp"
#include "NvidiaInterface.hpp"
#include "yolos.hpp"

namespace yolo {

using namespace std;
using namespace radar_core;
// 人脸关键点数量（仅 V5Face 使用）
static const int point_num = 4;

// 输出缓冲中每个框占用的 float 数量：
// 依次为 left/top/right/bottom/confidence/label/keep，共 7 个有效字段，
// 后 8 个位置预留关键点，因此固定为 15 便于统一寻址
const int NUM_BOX_ELEMENT = 15;
// 单张图最多保留的候选框数，同时也是输出缓冲的容量上限
const int MAX_IMAGE_BOXES = 1024;

// 向上对齐到 align 的整数倍
inline int upbound(int n, int align = 32)
{
    return (n + align - 1) / align * align;
}

// 2x3 仿射变换，host 与 device 都能调用
static __host__ __device__ void
affine_project(float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

// 通用 decode（YOLOv3/v5/v7/x）：每个候选点由 objectness 与类别置信度相乘得最终分数。
// 每个线程处理一个 anchor 位置，通过 atomicAdd 抢占输出缓冲中的槽位。
static __global__ void
decode_kernel_common(float* predict, int num_bboxes, int num_classes,
                     int output_cdim, float confidence_threshold,
                     float* invert_affine_matrix, float* parray,
                     int MAX_IMAGE_BOXES)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // 当前候选点的数据起点（每个点 output_cdim 个 float）
    float* pitem = predict + output_cdim * position;

    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    // 类别置信度从索引 5 开始，取最大值对应的类别
    float* class_confidence = pitem + 5;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    // 原子自增分配槽位；parray[0] 即当前已写入的框数量
    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    // 模型输出为 cx/cy/w/h，需转成左上右下
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;

    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    // 用逆向仿射矩阵把框从模型输入坐标系映射回原图
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
}

// V5Face 专用 decode：在通用逻辑基础上额外输出 point_num 个关键点，
// 因此类别置信度起点要整体后移 2*point_num
static __global__ void
decode_kernel_v5_face(float* predict, int num_bboxes, int num_classes,
                      int output_cdim, float confidence_threshold,
                      float* invert_affine_matrix, float* parray,
                      int MAX_IMAGE_BOXES)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;

    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5 + 2 * point_num;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }

    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    objectness = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    // 关键点以 (x, y) 成对存放
    float landmark_array[point_num * 2];

    for (int i = 0; i < point_num; i++) {
        landmark_array[2 * i] = *pitem++;
        landmark_array[2 * i + 1] = *pitem++;
    }

    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
    // 关键点也要做同样的逆变换
    for (int i = 0; i < point_num; i++) {
        affine_project(invert_affine_matrix, landmark_array[2 * i],
                       landmark_array[2 * i + 1], &landmark_array[2 * i],
                       &landmark_array[2 * i + 1]);
    }
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    for (int i = 0; i < point_num; i++) {
        *pout_item++ = landmark_array[2 * i];
        *pout_item++ = landmark_array[2 * i + 1];
    }
}

// YOLOv8 decode：输出无 objectness，类别置信度直接从索引 4 开始，
// 并在末尾多记一个 position（行号），供后续取分割 mask 权重使用
static __global__ void decode_kernel_v8(float* predict, int num_bboxes,
                                        int num_classes, int output_cdim,
                                        float  confidence_threshold,
                                        float* invert_affine_matrix,
                                        float* parray, int MAX_IMAGE_BOXES)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + output_cdim * position;
    float* class_confidence = pitem + 4;
    float  confidence = *class_confidence++;
    int    label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) {
        if (*class_confidence > confidence) {
            confidence = *class_confidence;
            label = i;
        }
    }
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;  // 供 seg 分支定位该框的 mask 权重
}

// 计算两个框的交并比 IoU
static __device__ float box_iou(float aleft, float atop, float aright,
                                float abottom, float bleft, float btop,
                                float bright, float bbottom)
{
    // 相交矩形
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

// GPU 端 NMS：每个线程对应一个框，与其他同类框比较 IoU，
// 若存在分数更高且重叠过大的框，则把当前框标记为忽略（keep=0）。
// 只对同 label 去重，不同类别互不抑制。
static __global__ void fast_nms_kernel(float* bboxes, int MAX_IMAGE_BOXES,
                                       float threshold)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, MAX_IMAGE_BOXES);
    if (position >= count)
        return;

    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        // 跳过自己和不同类别
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        // 只在对方分数更高时才可能抑制自己（分数相等按索引顺序决定，保证结果稳定）
        if (pitem[4] >= pcurrent[4]) {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou =
                box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                        pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold) {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

// 按任务数计算 grid：每块最多 GPU_BLOCK_THREADS 个线程
static dim3 grid_dims(int numJobs)
{
    int numBlockThreads =
        numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

// 块内线程数不超过 GPU_BLOCK_THREADS
static dim3 block_dims(int numJobs)
{
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

// 按模型类型分发对应的 decode kernel，随后统一执行 NMS
static void decode_kernel_invoker(float* predict, int num_bboxes,
                                  int num_classes, int output_cdim,
                                  float  confidence_threshold,
                                  float  nms_threshold,
                                  float* invert_affine_matrix,
                                  float* parray, int MAX_IMAGE_BOXES,
                                  Type type, cudaStream_t stream)
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    if (type == Type::V8 || type == Type::V8Seg) {
        checkKernel(decode_kernel_v8<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray,
            MAX_IMAGE_BOXES));
    } else if (type == Type::V5Face) {
        checkKernel(decode_kernel_v5_face<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray,
            MAX_IMAGE_BOXES));
    } else {
        checkKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim,
            confidence_threshold, invert_affine_matrix, parray,
            MAX_IMAGE_BOXES));
    }

    // NMS 的线程数按最大框数开，实际有效范围由 count 限制
    grid = grid_dims(MAX_IMAGE_BOXES);
    block = block_dims(MAX_IMAGE_BOXES);
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(
        parray, MAX_IMAGE_BOXES, nms_threshold));
}

// 实例分割 mask 解码：对每个像素的 mask_dim 维特征做加权求和，再过 sigmoid 转成 0-255。
// 该 kernel 逐像素计算，因此按输出 mask 尺寸划分线程。
static __global__ void
decode_single_mask_kernel(int left, int top, float* mask_weights,
                          float* mask_predict, int mask_width,
                          int mask_height, unsigned char* mask_out,
                          int mask_dim, int out_width, int out_height)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= out_width || dy >= out_height)
        return;

    // mask 在原图坐标系中的位置
    int sx = left + dx;
    int sy = top + dy;
    if (sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height) {
        mask_out[dy * out_width + dx] = 0;
        return;
    }

    // 逐通道加权累加
    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ++ic) {
        float cval =
            mask_predict[(ic * mask_height + sy) * mask_width + sx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }

    // sigmoid 得到概率，再缩放到 0-255
    float alpha = 1.0f / (1.0f + exp(-cumprod));
    mask_out[dy * out_width + dx] = alpha * 255;
}

static void decode_single_mask(float left, float top, float* mask_weights,
                               float* mask_predict, int mask_width,
                               int mask_height, unsigned char* mask_out,
                               int mask_dim, int out_width, int out_height,
                               cudaStream_t stream)
{
    // mask_weights is mask_dim(32 element) gpu pointer
    dim3 grid((out_width + 31) / 32, (out_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(decode_single_mask_kernel<<<grid, block, 0, stream>>>(
        left, top, mask_weights, mask_predict, mask_width, mask_height,
        mask_out, mask_dim, out_width, out_height));
}

const char* type_name(Type type)
{
    switch (type) {
    case Type::V5:
        return "YoloV5";
    case Type::V3:
        return "YoloV3";
    case Type::V7:
        return "YoloV7";
    case Type::X:
        return "YoloX";
    case Type::V8:
        return "YoloV8";
    case Type::V5Face:
        return "YoloV5Face";
    default:
        return "Unknow";
    }
}

// 本文件内另有一份 AffineMatrix 定义，与 BaseInfer.hpp 中的同名结构重复；
// 两者算法一致，此处为 yolo namespace 私有副本
struct AffineMatrix {
    float i2d[6];
    float d2i[6];

    void compute(const std::tuple<int, int>& from,
                 const std::tuple<int, int>& to)
    {
        float scale_x = get<0>(to) / (float)get<0>(from);
        float scale_y = get<1>(to) / (float)get<1>(from);
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 +
                 scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 +
                 scale * 0.5 - 0.5;

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
};

// 分割掩码使用 pinned memory，方便直接作为 D2H 拷贝目标
InstanceSegmentMap::InstanceSegmentMap(int width, int height)
{
    this->width = width;
    this->height = height;
    checkRuntime(cudaMallocHost(&this->data, width * height));
}

InstanceSegmentMap::~InstanceSegmentMap()
{
    if (this->data) {
        checkRuntime(cudaFreeHost(this->data));
        this->data = nullptr;
    }
    this->width = 0;
    this->height = 0;
}

// YOLO 推理器：预处理 + TensorRT 前向 + decode/NMS 后处理
class InferImpl : public Infer<BoxArray> {
public:
    shared_ptr<trt::Infer>                         trt_;
    string                                         engine_file_;
    Type                                           type_;
    float                                          confidence_threshold_;
    float                                          nms_threshold_;
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    // input_buffer_：模型输入；bbox_predict_：检测头原始输出；
    // output_boxarray_：decode 后的框数组（首 4 字节存数量）
    trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
    // 分割头原始输出（仅 V8Seg）
    trt::Memory<float> segment_predict_;
    int                network_input_width_, network_input_height_;
    Norm               normalize_;
    vector<int>        bbox_head_dims_;     // 检测头输出维度
    vector<int>        segment_head_dims_;  // 分割头输出维度
    int                num_classes_ = 0;
    bool               has_segment_ = false;
    bool               has_keyPoint = false;
    bool               isdynamic_model_ = false;
    // 每个框的 mask 输出临时缓冲，按需增长
    vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;

    virtual ~InferImpl() = default;

    // 按 batch 调整各级缓冲；output_boxarray_ 每张图需 4 字节计数 + 框数组
    void adjust_memory(int batch_size)
    {
        size_t input_numel =
            network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] *
                          bbox_head_dims_[2]);
        output_boxarray_.gpu(batch_size *
                             (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size *
                             (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        if (has_segment_)
            segment_predict_.gpu(batch_size * segment_head_dims_[1] *
                                 segment_head_dims_[2] *
                                 segment_head_dims_[3]);

        if ((int)preprocess_buffers_.size() < batch_size) {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
                preprocess_buffers_.push_back(
                    make_shared<trt::Memory<unsigned char>>());
        }
    }

    // 预处理：letterbox + 双线性插值 + 归一化，与 classify.cu 中实现一致
    void
    preprocess(int ibatch, const Image& image,
               shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
               AffineMatrix& affine, void* stream = nullptr)
    {
        affine.compute(
            make_tuple(image.width, image.height),
            make_tuple(network_input_width_, network_input_height_));

        size_t input_numel =
            network_input_width_ * network_input_height_ * 3;
        float*   input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t   size_image = image.width * image.height * 3;
        size_t   size_matrix = upbound(sizeof(affine.d2i), 32);
        // workspace 布局：仿射矩阵（已对齐）+ 原图数据
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
        warp_affine_bilinear_and_normalize_plane(
            image_device, image.width * 3, image.width, image.height,
            input_device, network_input_width_, network_input_height_,
            affine_matrix_device, 114, normalize_, stream_);
    }

    bool load(const string& engine_file, Type type,
              float confidence_threshold, float nms_threshold)
    {
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr)
            return false;

        trt_->print();

        this->type_ = type;
        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;

        // binding 0 为输入，binding 1 通常为检测头输出
        auto input_dim = trt_->static_dims(0);
        bbox_head_dims_ = trt_->static_dims(1);

        has_segment_ = type == Type::V8Seg;
        has_keyPoint = type == Type::V5Face;
        if (has_segment_) {
            // V8Seg 的 binding 顺序为：输入、分割头、检测头
            bbox_head_dims_ = trt_->static_dims(2);
            segment_head_dims_ = trt_->static_dims(1);
        }
        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        // 各模型输出通道构成不同，类别数由此推算
        if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5;  // cx,cy,w,h,obj + n*cls
        } else if (type == Type::V8) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4;  // 无 objectness
        } else if (type == Type::V8Seg) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
        } else if (type == Type::X) {
            normalize_ = Norm::None();  // YOLOX 输出已是归一化数值
            num_classes_ = bbox_head_dims_[2] - 5;
        } else if (type == Type::V5Face) {
            normalize_ =
                Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
            num_classes_ = bbox_head_dims_[2] - 5 - 2 * point_num;
        } else {
            INFO("Unsupport type %d", type);
        }
        return true;
    }

    virtual BoxArray forward(const Image& image,
                             void*        stream = nullptr) override
    {
        auto output = forwards({image}, stream);
        if (output.empty())
            return {};
        return output[0];
    }

    virtual vector<BoxArray> forwards(const vector<Image>& images,
                                      void* stream = nullptr) override
    {
        int num_image = images.size();
        if (num_image == 0)
            return {};

        // 动态 batch 可自由指定，静态 batch 不得超出 engine 最大批次
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
        adjust_memory(infer_batch_size);
        std::chrono ::high_resolution_clock::time_point a2 =
            std::chrono::high_resolution_clock::now();
        auto time_used2 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a2 -
                                                                      a1);

        vector<AffineMatrix> affine_matrixs(num_image);
        std::chrono ::high_resolution_clock::time_point a3 =
            std::chrono::high_resolution_clock::now();
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i)
            preprocess(i, images[i], preprocess_buffers_[i],
                       affine_matrixs[i], stream);

        float*        bbox_output_device = bbox_predict_.gpu();
        // 常规模型两个 binding，分割模型三个
        vector<void*> bindings{input_buffer_.gpu(), bbox_output_device};

        if (has_segment_) {
            bindings = {input_buffer_.gpu(), segment_predict_.gpu(),
                        bbox_output_device};
        }
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

        // 逐张图做 decode + NMS。各图输出缓冲相互独立，都在同一 stream 上串行排布
        for (int ib = 0; ib < num_image; ++ib) {
            float* boxarray_device =
                output_boxarray_.gpu() +
                ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            // 预处理时把逆向仿射矩阵放在 workspace 首部，此处直接复用
            float* affine_matrix_device =
                (float*)preprocess_buffers_[ib]->gpu();
            float* image_based_bbox_output =
                bbox_output_device +
                ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);

            // parray[0] 是框计数，decode 前必须清零
            checkRuntime(
                cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            decode_kernel_invoker(
                image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES,
                type_, stream_);
        }
        std::chrono ::high_resolution_clock::time_point a3d3 =
            std::chrono::high_resolution_clock::now();
        auto time_used3d3 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a3d3 -
                                                                      a3d2);
        std::chrono ::high_resolution_clock::time_point a3d4 =
            std::chrono::high_resolution_clock::now();
        // 整块框数组拷回 host
        checkRuntime(cudaMemcpyAsync(
            output_boxarray_.cpu(), output_boxarray_.gpu(),
            output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
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
        vector<BoxArray> arrout(num_image);
        int              imemory = 0;
        for (int ib = 0; ib < num_image; ++ib) {
            float* parray = output_boxarray_.cpu() +
                            ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            // 首元素为计数，需裁剪到容量上限以内
            int       count = min(MAX_IMAGE_BOXES, (int)*parray);
            BoxArray& output = arrout[ib];
            output.reserve(count);
            for (int i = 0; i < count; ++i) {
                float*             pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                int                a1 = pbox[0];
                int                a2 = pbox[1];
                int                a3 = pbox[2];
                int                a4 = pbox[3];
                int                a5 = pbox[4];
                int                label = pbox[5];
                int                keepflag = pbox[6];
                // 后 8 个位置存放关键点，无论模型是否有关键点都一并读出
                std::vector<float> points;
                for (int i = 1; i < 9; i++) {
                    points.push_back(pbox[6 + i]);
                }
                // NMS 中被抑制的框不再输出
                if (keepflag == 1) {
                    Box result_object_box(pbox[0], pbox[1], pbox[2],
                                          pbox[3], pbox[4], label, points);

                    if (has_segment_) {
                        // pbox[7] 为 decode 时记下的候选点行号，
                        // 据此定位该框对应的 mask 权重
                        int    row_index = pbox[7];
                        int    mask_dim = segment_head_dims_[1];
                        float* mask_weights =
                            bbox_output_device +
                            (ib * bbox_head_dims_[1] + row_index) *
                                bbox_head_dims_[2] +
                            num_classes_ + 4;

                        float* mask_head_predict = segment_predict_.gpu();
                        // mask 是在模型输入坐标系下生成的，需按正向仿射换算出框的位置
                        float  left, top, right, bottom;
                        float* i2d = affine_matrixs[ib].i2d;
                        affine_project(i2d, pbox[0], pbox[1], &left, &top);
                        affine_project(i2d, pbox[2], pbox[3], &right,
                                       &bottom);

                        float box_width = right - left;
                        float box_height = bottom - top;

                        // 模型输入尺寸与 mask 分辨率之比，用于把框尺寸换算到 mask 空间
                        float scale_to_predict_x =
                            segment_head_dims_[3] /
                            (float)network_input_width_;
                        float scale_to_predict_y =
                            segment_head_dims_[2] /
                            (float)network_input_height_;
                        int mask_out_width =
                            box_width * scale_to_predict_x + 0.5f;
                        int mask_out_height =
                            box_height * scale_to_predict_y + 0.5f;

                        if (mask_out_width > 0 && mask_out_height > 0) {
                            if (imemory >= (int)box_segment_cache_.size()) {
                                box_segment_cache_.push_back(
                                    std::make_shared<
                                        trt::Memory<unsigned char>>());
                            }

                            int bytes_of_mask_out =
                                mask_out_width * mask_out_height;
                            auto box_segment_output_memory =
                                box_segment_cache_[imemory];
                            result_object_box.seg =
                                make_shared<InstanceSegmentMap>(
                                    mask_out_width, mask_out_height);

                            unsigned char* mask_out_device =
                                box_segment_output_memory->gpu(
                                    bytes_of_mask_out);
                            unsigned char* mask_out_host =
                                result_object_box.seg->data;
                            // mask 起点也要换算到 mask 空间
                            decode_single_mask(
                                left * scale_to_predict_x,
                                top * scale_to_predict_y, mask_weights,
                                mask_head_predict +
                                    ib * segment_head_dims_[1] *
                                        segment_head_dims_[2] *
                                        segment_head_dims_[3],
                                segment_head_dims_[3],
                                segment_head_dims_[2], mask_out_device,
                                mask_dim, mask_out_width, mask_out_height,
                                stream_);
                            checkRuntime(cudaMemcpyAsync(
                                mask_out_host, mask_out_device,
                                box_segment_output_memory->gpu_bytes(),
                                cudaMemcpyDeviceToHost, stream_));
                        }
                    }
                    output.emplace_back(result_object_box);
                }
            }
        }
        std::chrono ::high_resolution_clock::time_point a6 =
            std::chrono::high_resolution_clock::now();
        auto time_used6 =
            std::chrono::duration_cast<std::chrono::duration<double>>(a6 -
                                                                      a5);
        // mask 的 D2H 是异步提交的，返回前必须等待完成
        if (has_segment_)
            checkRuntime(cudaStreamSynchronize(stream_));

        return arrout;
    }
};

Infer<BoxArray>* loadraw(const std::string& engine_file, Type type,
                         float confidence_threshold, float nms_threshold)
{
    InferImpl* impl = new InferImpl();
    if (!impl->load(engine_file, type, confidence_threshold,
                    nms_threshold)) {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

shared_ptr<Infer<BoxArray>> load(const string& engine_file, Type type,
                                 float confidence_threshold,
                                 float nms_threshold)
{
    return std::shared_ptr<InferImpl>((InferImpl*)loadraw(
        engine_file, type, confidence_threshold, nms_threshold));
}

// HSV 转 BGR，输入 h/s 归一化到 0-1，v 为亮度
std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int   h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float       r, g, b;
    switch (h_i) {
    case 0:
        r = v, g = t, b = p;
        break;
    case 1:
        r = q, g = v, b = p;
        break;
    case 2:
        r = p, g = v, b = t;
        break;
    case 3:
        r = p, g = q, b = v;
        break;
    case 4:
        r = t, g = p, b = v;
        break;
    case 5:
        r = v, g = p, b = q;
        break;
    default:
        r = 1, g = 1, b = 1;
        break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255),
                      static_cast<uint8_t>(g * 255),
                      static_cast<uint8_t>(r * 255));
}

// 由 id 生成稳定的伪随机颜色：同一 id 每次得到同一颜色，便于区分不同类别
std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

};  // namespace yolo
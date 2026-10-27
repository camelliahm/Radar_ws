#ifndef __INFER_HPP__
#define __INFER_HPP__

#include <NvInfer.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace trt {

// 带文件名与行号的日志宏
#define INFO(...) trt::__log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char* file, int line, const char* fmt, ...);
// 矩阵转置（在 GPU 上执行），用于部分模型需要 NHWC/HWC 转换的场合
void transpose(float* input, float* output, int rows, int cols,
               void* stream);

// TensorRT 张量数据类型
enum class DType : int {
    FLOAT = 0,
    HALF = 1,
    INT8 = 2,
    INT32 = 3,
    BOOL = 4,
    UINT8 = 5
};

// 基于 CUDA event 的计时器，可测 GPU 侧耗时（不受 CPU 异步影响）
class Timer {
public:
    Timer();
    virtual ~Timer();
    void  start(void* stream = nullptr);
    // 返回距上次 start 的毫秒数；print 为 true 时同时打印
    float stop(const char* prefix = "Timer", bool print = true);

private:
    void *start_, *stop_;  // cudaEvent_t，此处以 void* 隐藏 CUDA 类型
    void* stream_;
};

// 内存管理基类，同时维护三种内存：普通 CPU、GPU 显存、统一寻址内存。
// owner_* 标记表示该指针由本对象分配、需要负责释放；
// 若只是引用外部内存（reference 接口），owner 为 false，析构时不释放。
class BaseMemory {
public:
    BaseMemory() = default;
    BaseMemory(void* cpu, size_t cpu_bytes, void* gpu, size_t gpu_bytes);
    virtual ~BaseMemory();

    // 容量不足时才重新分配，避免每帧反复 malloc/free
    virtual void* gpu_realloc(size_t bytes);
    virtual void* cpu_realloc(size_t bytes);      // 使用 pinned memory，加快 H2D
    virtual void* unified_realloc(size_t bytes);  // 使用 managed memory

    void release_gpu();
    void release_cpu();
    void release_unified();
    void release();

    inline bool owner_gpu() const
    {
        return owner_gpu_;
    }
    inline bool owner_cpu() const
    {
        return owner_cpu_;
    }
    inline bool owner_unified() const
    {
        return owner_unified_;
    }

    inline size_t cpu_bytes() const
    {
        return cpu_bytes_;
    }
    inline size_t gpu_bytes() const
    {
        return gpu_bytes_;
    }
    inline size_t unified_bytes() const
    {
        return unified_bytes_;
    }

    virtual inline void* get_gpu() const
    {
        return gpu_;
    }
    virtual inline void* get_cpu() const
    {
        return cpu_;
    }
    virtual inline void* get_unified() const
    {
        return unified_;
    }

    // 仅引用外部内存，不接管所有权
    void reference(void* cpu, size_t cpu_bytes, void* gpu,
                   size_t gpu_bytes);
    void reference(void* unified, size_t unified_bytes);

protected:
    void*  cpu_ = nullptr;
    size_t cpu_bytes_ = 0, cpu_capacity_ = 0;  // 实际字节数与已分配容量
    bool   owner_cpu_ = true;

    void*  gpu_ = nullptr;
    size_t gpu_bytes_ = 0, gpu_capacity_ = 0;
    bool   owner_gpu_ = true;

    void*  unified_ = nullptr;
    size_t unified_bytes_ = 0, unified_capacity_ = 0;
    bool   owner_unified_ = true;
};

// 按元素类型包装 BaseMemory：对外接口用"元素个数"而非字节数，内部自动乘 sizeof
template <typename _DT>
class Memory : public BaseMemory {
public:
    Memory() = default;
    Memory(const Memory& other) = delete;
    Memory&      operator=(const Memory& other) = delete;
    virtual _DT* gpu(size_t size)
    {
        return (_DT*)BaseMemory::gpu_realloc(size * sizeof(_DT));
    }
    virtual _DT* cpu(size_t size)
    {
        return (_DT*)BaseMemory::cpu_realloc(size * sizeof(_DT));
    }
    virtual _DT* unified(size_t size)
    {
        return (_DT*)BaseMemory::unified_realloc(size * sizeof(_DT));
    }

    inline size_t cpu_size() const
    {
        return cpu_bytes_ / sizeof(_DT);
    }
    inline size_t gpu_size() const
    {
        return gpu_bytes_ / sizeof(_DT);
    }
    inline size_t unified_size() const
    {
        return unified_bytes_ / sizeof(_DT);
    }

    virtual inline _DT* gpu() const
    {
        return (_DT*)gpu_;
    }
    virtual inline _DT* cpu() const
    {
        return (_DT*)cpu_;
    }
    virtual inline _DT* unified() const
    {
        return (_DT*)unified_;
    }
};

// TensorRT 执行上下文封装。
// bindings 为各输入/输出张量的设备指针数组，顺序与 engine 的 binding 索引一致。
class Infer {
public:
    virtual bool             forward(const std::vector<void*>& bindings,
                                     void*                     stream = nullptr,
                                     void* input_consum_event = nullptr) = 0;
    virtual int              index(const std::string& name) = 0;
    // run_dims 为当前运行时的实际维度（动态 shape 会变化），
    // static_dims 为 engine 编译时的固定维度
    virtual std::vector<int> run_dims(const std::string& name) = 0;
    virtual std::vector<int> run_dims(int ibinding) = 0;
    virtual std::vector<int> static_dims(const std::string& name) = 0;
    virtual std::vector<int> static_dims(int ibinding) = 0;
    // 元素总数（各维乘积）
    virtual int              numel(const std::string& name) = 0;
    virtual int              numel(int ibinding) = 0;
    virtual int              num_bindings() = 0;
    virtual bool             is_input(int ibinding) = 0;
    // 为动态 shape 的输入指定本次推理的维度
    virtual bool             set_run_dims(const std::string&      name,
                                          const std::vector<int>& dims) = 0;
    virtual bool             set_run_dims(int                     ibinding,
                                          const std::vector<int>& dims) = 0;
    virtual DType            dtype(const std::string& name) = 0;
    virtual DType            dtype(int ibinding) = 0;
    // 是否存在动态维度（任一一维为 -1）
    virtual bool             has_dynamic_dim() = 0;
    // 打印所有输入/输出名称与维度，便于调试
    virtual void             print() = 0;
};

// 从 .engine 文件反序列化出执行上下文
std::shared_ptr<Infer> load(const std::string& file);
// 把维度向量格式化为 [a, b, c] 形式的字符串
std::string            format_shape(const std::vector<int>& shape);

}  // namespace trt

#endif  // __INFER_HPP__
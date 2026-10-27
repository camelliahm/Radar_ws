#ifndef __CLASSIFY_HPP__
#define __CLASSIFY_HPP__

#include <future>
#include <memory>
#include <string>
#include <vector>
#include "BaseInfer.hpp"
#include "InferTool.hpp"

namespace classify {

using namespace radar_core;

// 支持的分类网络类型，新增模型时在此扩展
enum class Type : int { densenet121 = 0 };

// 加载 engine 并返回分类推理器；forward 输出该图所属类别下标
std::shared_ptr<Infer<int>> load(const std::string& engine_file, Type type);

// 类型名，用于打印日志
const char* type_name(Type type);

}  // namespace classify

#endif  // __CLASSIFY_HPP__
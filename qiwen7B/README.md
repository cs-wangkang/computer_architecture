# Qwen7B 推理优化项目

本项目实现了Qwen7B模型的基础推理和VLLM优化推理，并提供了性能基准测试脚本。

## 项目结构

```
qiwen7B/
├── basic_inference.py          # 基础推理实现（使用transformers）
├── vllm_inference.py           # VLLM优化推理实现
├── benchmark_basic.py          # 基础推理性能测试脚本
├── benchmark_vllm.py           # VLLM优化推理性能测试脚本
├── requirements.txt            # 依赖包列表
└── README.md                   # 项目说明文档
```

## 环境设置

1. 激活conda虚拟环境：
```bash
conda activate qiwen7B_optimaze
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基础推理测试

运行基础推理测试：
```bash
python benchmark_basic.py --num_samples 10 --warmup 2 --max_length 512
```

参数说明：
- `--model_name`: 模型名称（默认：Qwen/Qwen-7B-Chat）
- `--num_samples`: 测试样本数（默认：10）
- `--warmup`: 预热轮数（默认：2）
- `--max_length`: 最大生成长度（默认：512）
- `--output`: 结果输出文件（默认：benchmark_basic_results.json）

### 2. VLLM优化推理测试

运行VLLM优化推理测试：
```bash
python benchmark_vllm.py --num_samples 10 --warmup 2 --max_tokens 512
```

参数说明：
- `--model_name`: 模型名称（默认：Qwen/Qwen-7B-Chat）
- `--num_samples`: 测试样本数（默认：10）
- `--warmup`: 预热轮数（默认：2）
- `--max_tokens`: 最大生成token数（默认：512）
- `--tensor_parallel_size`: 张量并行大小（默认：1）
- `--gpu_memory_utilization`: GPU内存利用率（默认：0.9）
- `--output`: 结果输出文件（默认：benchmark_vllm_results.json）

### 3. 单独使用推理类

#### 基础推理
```python
from basic_inference import Qwen7BInference

inference = Qwen7BInference()
result = inference.generate("介绍一下你的模型")
print(result)
```

#### VLLM优化推理
```python
from vllm_inference import Qwen7BVLLMInference

inference = Qwen7BVLLMInference()
result = inference.generate("介绍一下你的模型")
print(result)
```

## 性能对比

运行两个benchmark脚本后，可以对比以下指标：
- 模型加载时间
- 平均延迟（P50, P95, P99）
- 吞吐量（tokens/秒）
- 批量推理性能（VLLM）

## 注意事项

1. 确保有足够的GPU内存（建议至少16GB）
2. VLLM需要CUDA环境
3. 首次运行会下载模型，需要一定时间
4. 建议先运行少量样本测试，确认环境配置正确

## VLLM优化特性

VLLM提供了以下优化：
- PagedAttention：高效的内存管理
- 连续批处理：动态批处理请求
- 量化支持：降低内存占用
- 张量并行：多GPU加速


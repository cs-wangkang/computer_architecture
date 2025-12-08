# Qwen7B 推理性能优化项目

本项目实现了Qwen7B模型的基础推理和优化推理，并提供性能基准测试工具。

## 项目结构

```
qiwen7B_project/
├── basic_inference.py          # 基础推理实现
├── benchmark_basic.py           # 基础推理基准测试脚本
├── optimized_inference.py       # 优化推理实现（量化、KV缓存等）
├── benchmark_optimized.py       # 优化推理基准测试脚本
├── requirements.txt             # 依赖包列表
└── README.md                    # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU加速推荐)
- Conda环境: qiwen7B

## 安装依赖

```bash
# 激活conda环境
conda activate qiwen7B

# 安装依赖
pip install -r requirements.txt

# 可选：安装Flash Attention 2（需要CUDA）
pip install flash-attn --no-build-isolation
```

## 使用方法

### 1. 基础推理

#### 直接运行测试
```bash
python basic_inference.py
```

#### 运行基准测试
```bash
python benchmark_basic.py \
    --model-name Qwen/Qwen-7B-Chat \
    --num-samples 10 \
    --max-length 512 \
    --output benchmark_basic_results.json
```

### 2. 优化推理

#### 直接运行测试
```bash
python optimized_inference.py
```

#### 运行基准测试（4-bit量化）
```bash
python benchmark_optimized.py \
    --model-name Qwen/Qwen-7B-Chat \
    --num-samples 10 \
    --max-length 512 \
    --quantization 4bit \
    --batch-size 1 \
    --output benchmark_optimized_results.json
```

#### 运行基准测试（8-bit量化）
```bash
python benchmark_optimized.py \
    --model-name Qwen/Qwen-7B-Chat \
    --quantization 8bit \
    --num-samples 10
```

#### 使用torch.compile优化
```bash
python benchmark_optimized.py \
    --use-torch-compile \
    --quantization 4bit
```

### 3. 基准测试参数说明

#### benchmark_basic.py 参数
- `--model-name`: 模型名称或路径（默认: Qwen/Qwen-7B-Chat）
- `--device`: 设备类型，auto/cuda/cpu（默认: auto）
- `--warmup`: 预热轮数（默认: 2）
- `--num-samples`: 测试样本数量（默认: 10）
- `--max-length`: 最大生成长度（默认: 512）
- `--output`: 结果输出文件（默认: benchmark_basic_results.json）

#### benchmark_optimized.py 参数
- `--model-name`: 模型名称或路径（默认: Qwen/Qwen-7B-Chat）
- `--device`: 设备类型，auto/cuda/cpu（默认: auto）
- `--warmup`: 预热轮数（默认: 2）
- `--num-samples`: 测试样本数量（默认: 10）
- `--max-length`: 最大生成长度（默认: 512）
- `--batch-size`: 批处理大小（默认: 1）
- `--quantization`: 量化类型，4bit/8bit/none（默认: 4bit）
- `--no-flash-attention`: 禁用Flash Attention
- `--use-torch-compile`: 使用torch.compile优化
- `--output`: 结果输出文件（默认: benchmark_optimized_results.json）

## 优化技术说明

### 1. 量化（Quantization）
- **4-bit量化**: 使用BitsAndBytesConfig进行4-bit量化，显著减少显存占用
- **8-bit量化**: 使用8-bit量化，平衡性能和精度
- 量化可以大幅减少显存占用，适合在资源受限的环境中运行

### 2. KV缓存（Key-Value Cache）
- 自动启用KV缓存，避免重复计算已生成的token的key-value
- 显著提升生成速度，特别是长文本生成

### 3. Flash Attention
- 使用Flash Attention 2加速注意力计算
- 减少显存占用并提升计算效率
- 需要安装flash-attn包

### 4. Torch Compile
- 使用PyTorch 2.0+的torch.compile进行图优化
- 可以进一步提升推理速度
- 需要PyTorch 2.0+

### 5. 批处理优化
- 支持批量处理多个输入
- 提高GPU利用率

## 性能对比

运行基准测试后，可以对比以下指标：
- **延迟（Latency）**: 单次生成的平均延迟（ms）
- **吞吐量（Throughput）**: 每秒生成的token数（tokens/s）
- **显存占用**: 量化可以大幅减少显存占用
- **P50/P95/P99延迟**: 不同百分位的延迟分布

## 注意事项

1. 首次运行会下载模型，需要较长时间和足够的磁盘空间
2. 4-bit量化需要CUDA环境，CPU环境会自动禁用量化
3. Flash Attention需要特定的CUDA版本和编译环境
4. 批处理可能会增加延迟，但可以提高总体吞吐量
5. 模型加载时间不计入生成延迟统计

## 结果示例

基准测试会生成JSON格式的结果文件，包含以下指标：
- 平均/最小/最大延迟
- P50/P95/P99延迟
- 平均生成token数
- 吞吐量（tokens/s）
- 模型加载时间
- 使用的优化技术


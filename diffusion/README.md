# Stable Diffusion 3.5 Large Turbo 图像生成项目

## 项目简介

本项目实现了 Stability AI 发布的 Stable Diffusion 3.5 Large Turbo 模型的推理功能，支持通过文本提示词快速生成高质量图像。

## 模型背景

### Stable Diffusion 3.5 系列

Stable Diffusion 3.5 是 Stability AI 于 2024 年 10 月发布的最新文本到图像生成模型系列，该系列包括三个版本：

- **Large**: 标准版本，提供最佳图像质量
- **Large Turbo**: 优化版本，在保持高质量的同时显著提升生成速度
- **Medium**: 轻量级版本，适合资源受限的环境

### Stable Diffusion 3.5 Large Turbo 特点

1. **高效生成**: 仅需 4 个推理步骤即可生成高质量图像，速度比 Large 版本快数倍
2. **高质量输出**: 在提示词遵循性和图像质量方面表现出色
3. **大模型规模**: 拥有 80 亿参数，能够理解复杂的文本描述并生成细节丰富的图像
4. **灵活配置**: 支持多种分辨率（最高 1024x1024）和生成参数调整

## 功能特性

- ✅ **文本到图像生成**: 根据文本提示词生成高质量图像
- ✅ **负向提示词支持**: 通过负向提示词排除不想要的元素
- ✅ **批量生成**: 支持一次性生成多张图像
- ✅ **灵活参数配置**: 可调整推理步数、图像尺寸、随机种子等参数
- ✅ **GPU/CPU 自动适配**: 自动检测并使用可用设备
- ✅ **内存优化**: 支持模型 CPU offload，降低显存占用
- ✅ **HTTP API 服务**: 提供RESTful API接口，支持通过HTTP请求生成图片

## 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐，用于 GPU 加速)
- 至少 16GB 显存 (推荐，用于 Large Turbo 模型)
- 或 32GB+ 系统内存 (CPU 模式)

## 安装步骤

### 1. 激活 conda 虚拟环境

```bash
conda activate qiwen7B
```

### 2. 安装依赖

```bash
cd /root/autodl-tmp/project/diffusion
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; import diffusers; print('安装成功！')"
```

## 使用方法

### HTTP API 服务（推荐）

项目提供了HTTP API服务，可以通过RESTful接口生成图片。

#### 启动服务

```bash
# 方式1: 使用启动脚本
conda activate diffusion
./start_server.sh

# 方式2: 直接运行
conda activate diffusion
python app.py --host 0.0.0.0 --port 5000
```

服务启动后，默认监听 `http://0.0.0.0:5000`

#### API 接口

**1. 健康检查**
```bash
GET http://localhost:5000/health
```

**2. 服务信息**
```bash
GET http://localhost:5000/info
```

**3. 生成图片（POST方式，推荐）**
```bash
POST http://localhost:5000/generate
Content-Type: application/json

{
  "prompt": "a beautiful sunset over the ocean, vibrant colors",
  "negative_prompt": "blurry, low quality",
  "steps": 4,
  "width": 512,
  "height": 512,
  "seed": 42
}
```

**4. 生成图片（GET方式，简单测试）**
```bash
GET http://localhost:5000/generate?prompt=a%20beautiful%20sunset&steps=4&width=512&height=512
```

#### 使用示例

**Python 示例:**
```python
import requests

# POST方式
response = requests.post(
    "http://localhost:5000/generate",
    json={
        "prompt": "a cute cat playing with yarn",
        "steps": 4,
        "width": 512,
        "height": 512
    },
    timeout=300
)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
    print("图片已保存")
```

**curl 示例:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful landscape", "steps": 4, "width": 512, "height": 512}' \
  --output image.png
```

**GET方式:**
```bash
curl "http://localhost:5000/generate?prompt=a%20beautiful%20sunset&steps=4&width=512&height=512" \
  --output image.png
```

#### 图片存储

- 生成的图片自动保存到 `generated_image/` 目录
- 图片命名格式: `image_000001.png`, `image_000002.png` ... (自动递增)
- 索引信息保存在 `image_index.json` 文件中

#### 测试API

```bash
# 运行测试脚本
python test_api.py
```

### 命令行使用

### 基础使用

```bash
python inference.py --prompt "a beautiful sunset over the ocean, vibrant colors, peaceful atmosphere"
```

### 完整参数示例

```bash
python inference.py \
    --prompt "a futuristic cityscape at night, neon lights, cyberpunk style" \
    --negative-prompt "blurry, low quality, distorted" \
    --output output.png \
    --steps 4 \
    --width 1024 \
    --height 1024 \
    --seed 42
```

### 参数说明

- `--prompt`: 图像生成提示词（必需）
- `--negative-prompt`: 负向提示词，用于排除不想要的元素
- `--output`: 输出图像路径（默认: output.png）
- `--steps`: 推理步数（默认: 4，适合 turbo 模型）
- `--width`: 图像宽度（默认: 1024）
- `--height`: 图像高度（默认: 1024）
- `--seed`: 随机种子，用于可重复生成
- `--model-id`: 模型 ID（默认: stabilityai/stable-diffusion-3.5-large-turbo）
- `--device`: 设备类型，auto/cuda/cpu（默认: auto）
- `--dtype`: 数据类型，float16/float32（默认: float16）

### Python API 使用

```python
from inference import StableDiffusion35Inference

# 初始化模型
inference = StableDiffusion35Inference()

# 生成单张图像
image = inference.generate(
    prompt="a cute cat playing with a ball of yarn",
    output_path="cat.png",
    seed=42
)

# 批量生成
prompts = [
    "a serene mountain landscape",
    "a bustling city street at night",
    "a peaceful garden with flowers"
]
images = inference.generate_batch(
    prompts=prompts,
    output_dir="outputs"
)
```

## 性能优化建议

1. **使用 GPU**: 确保 CUDA 可用，GPU 模式比 CPU 模式快 10-100 倍
2. **使用 float16**: 默认使用 float16 精度，可显著降低显存占用
3. **调整推理步数**: Turbo 模型在 4 步时已能生成高质量图像，增加步数提升有限
4. **启用 CPU offload**: 如果显存不足，会自动启用模型 CPU offload

## 示例输出

生成的图像将保存在指定的输出路径。图像格式为 PNG，支持透明背景（如果模型支持）。

## 常见问题

### Q: 显存不足怎么办？

A: 可以尝试以下方法：
- 使用 `--dtype float32` 降低精度（但会降低速度）
- 减小图像尺寸（如 512x512）
- 确保启用了 CPU offload（代码中已自动处理）

### Q: 生成速度慢怎么办？

A: 
- 确保使用 GPU 而非 CPU
- 使用默认的 4 步推理（Turbo 模型优化）
- 检查 CUDA 和 cuDNN 是否正确安装

### Q: 如何提高图像质量？

A:
- 使用更详细和具体的提示词
- 适当使用负向提示词排除不想要的元素
- 可以尝试增加到 6-8 步（但提升有限）

## 技术架构

本项目使用以下技术栈：

- **diffusers**: Hugging Face 的扩散模型库，提供模型加载和推理接口
- **transformers**: 用于模型组件加载
- **torch**: PyTorch 深度学习框架
- **PIL/Pillow**: 图像处理

## 模型信息

- **模型名称**: stable-diffusion-3.5-large-turbo
- **发布机构**: Stability AI
- **模型大小**: 约 8B 参数
- **许可证**: 请参考 Stability AI 官方许可证
- **Hugging Face**: https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo

## 参考资料

- [Stability AI 官网](https://stability.ai/)
- [Hugging Face Diffusers 文档](https://huggingface.co/docs/diffusers)
- [Stable Diffusion 3.5 发布公告](https://stability.ai/news/stable-diffusion-3-5)

## 许可证

本项目代码遵循 MIT 许可证。模型权重遵循 Stability AI 的许可证条款。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

- **2024-12-08**: 
  - 初始版本，实现基础推理功能
  - 添加HTTP API服务，支持RESTful接口生成图片
  - 实现图片自动索引管理


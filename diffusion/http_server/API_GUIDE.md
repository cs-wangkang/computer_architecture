# HTTP API 使用指南

## 快速开始

### 1. 启动服务

```bash
conda activate diffusion
python app.py
```

服务将在 `http://0.0.0.0:5000` 启动

### 2. 生成图片

**方式1: POST请求（推荐）**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over the ocean",
    "steps": 4,
    "width": 512,
    "height": 512
  }' \
  --output image.png
```

**方式2: GET请求（简单测试）**
```bash
curl "http://localhost:5000/generate?prompt=a%20beautiful%20sunset&steps=4&width=512&height=512" \
  --output image.png
```

## API 接口

### POST /generate

生成图片（推荐方式）

**请求参数（JSON）:**
- `prompt` (string, 必需): 图片生成提示词
- `negative_prompt` (string, 可选): 负向提示词
- `steps` (int, 可选): 推理步数，默认4
- `width` (int, 可选): 图片宽度，默认512
- `height` (int, 可选): 图片高度，默认512
- `seed` (int, 可选): 随机种子

**响应:**
- 成功 (200): 返回PNG图片文件
- 失败 (400/500): 返回JSON错误信息

**示例:**
```python
import requests

response = requests.post(
    "http://localhost:5000/generate",
    json={
        "prompt": "a cute cat playing with yarn",
        "negative_prompt": "blurry, low quality",
        "steps": 4,
        "width": 512,
        "height": 512,
        "seed": 42
    },
    timeout=300
)

if response.status_code == 200:
    with open("output.png", "wb") as f:
        f.write(response.content)
```

### GET /generate

生成图片（简单测试用）

**查询参数:**
- `prompt` (string, 必需): 图片生成提示词
- `negative_prompt` (string, 可选): 负向提示词
- `steps` (int, 可选): 推理步数，默认4
- `width` (int, 可选): 图片宽度，默认512
- `height` (int, 可选): 图片高度，默认512
- `seed` (int, 可选): 随机种子

**响应:**
- 成功 (200): 返回PNG图片文件
- 失败 (400/500): 返回JSON错误信息

### GET /health

健康检查接口

**响应:**
```json
{
  "status": "ok",
  "message": "Service is running"
}
```

### GET /info

获取服务信息

**响应:**
```json
{
  "service": "Stable Diffusion 3.5 Large Turbo API",
  "version": "1.0.0",
  "generated_images_count": 10,
  "endpoints": {
    "POST /generate": "生成图片（JSON格式）",
    "GET /generate": "生成图片（查询参数）",
    "GET /health": "健康检查",
    "GET /info": "服务信息"
  }
}
```

## 图片存储

- 所有生成的图片自动保存到 `generated_image/` 目录
- 图片命名格式: `image_000001.png`, `image_000002.png`, ...
- 索引自动递增，保存在 `image_index.json` 文件中
- 即使服务重启，索引也会继续递增

## 错误处理

**常见错误码:**
- `400`: 请求参数错误（如缺少prompt）
- `500`: 服务器内部错误（如模型加载失败、生成失败）

**错误响应格式:**
```json
{
  "error": "错误描述信息"
}
```

## 性能建议

1. **首次请求**: 需要加载模型，可能需要1-2分钟
2. **后续请求**: 模型已加载，生成速度更快（约20-30秒）
3. **并发请求**: 服务支持多线程，但建议控制并发数量
4. **超时设置**: 建议设置5分钟（300秒）超时

## 测试

运行测试脚本:
```bash
python test_api.py
```

## 注意事项

1. 模型首次加载需要较长时间和大量内存
2. 建议使用GPU加速，CPU模式会非常慢
3. 图片生成时间取决于硬件配置和图片尺寸
4. 服务启动时会自动初始化模型（单例模式）


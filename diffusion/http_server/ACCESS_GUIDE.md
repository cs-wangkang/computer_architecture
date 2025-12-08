# 服务访问指南

## 服务地址

服务默认运行在：**http://0.0.0.0:5000** 或 **http://localhost:5000**

## 访问方式

### 方式1: Web界面（推荐）

在浏览器中打开：
```
http://localhost:5000/
```

或者如果从其他机器访问：
```
http://<服务器IP>:5000/
```

Web界面提供了友好的图形界面，可以直接输入提示词生成图片。

### 方式2: 命令行访问

#### 健康检查
```bash
curl http://localhost:5000/health
```

#### 查看服务信息
```bash
curl http://localhost:5000/info
```

#### 生成图片（GET方式）
```bash
curl "http://localhost:5000/generate?prompt=a%20beautiful%20sunset&steps=4&width=512&height=512" \
  --output image.png
```

#### 生成图片（POST方式）
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

### 方式3: Python脚本

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

### 方式4: 使用测试脚本

```bash
cd /root/autodl-tmp/project/diffusion/http_server
python test_api.py
```

## 查看已生成的图片

所有生成的图片都保存在：
```
/root/autodl-tmp/project/diffusion/generated_image/
```

图片命名格式：`image_000001.png`, `image_000002.png`, ...

你可以通过以下方式查看：

1. **直接访问文件系统**
   ```bash
   ls -lh /root/autodl-tmp/project/diffusion/generated_image/
   ```

2. **使用文件管理器**（如果有图形界面）

3. **通过SSH下载**
   ```bash
   scp user@server:/root/autodl-tmp/project/diffusion/generated_image/image_*.png ./
   ```

## API接口说明

### POST /generate
生成图片（推荐方式）

**请求体（JSON）:**
```json
{
  "prompt": "图片描述",
  "negative_prompt": "不想要的元素（可选）",
  "steps": 4,
  "width": 512,
  "height": 512,
  "seed": 42
}
```

**响应:** PNG图片文件

### GET /generate
生成图片（简单测试）

**查询参数:**
- `prompt` (必需): 图片描述
- `negative_prompt` (可选): 负向提示词
- `steps` (可选): 推理步数，默认4
- `width` (可选): 宽度，默认512
- `height` (可选): 高度，默认512
- `seed` (可选): 随机种子

**响应:** PNG图片文件

### GET /health
健康检查

**响应:**
```json
{
  "status": "ok",
  "message": "Service is running"
}
```

### GET /info
服务信息

**响应:**
```json
{
  "service": "Stable Diffusion 3.5 Large Turbo API",
  "version": "1.0.0",
  "generated_images_count": 10,
  "endpoints": {...}
}
```

## 注意事项

1. **首次请求**: 需要加载模型，可能需要1-2分钟
2. **后续请求**: 模型已加载，生成速度更快（约20-30秒）
3. **超时设置**: 建议设置5分钟（300秒）超时
4. **图片大小**: 默认512x512，可以根据需要调整

## 故障排查

### 服务无法访问

1. 检查服务是否运行：
   ```bash
   ps aux | grep "python.*app.py"
   ```

2. 检查端口是否被占用：
   ```bash
   netstat -tlnp | grep 5000
   ```

3. 检查防火墙设置

### 生成失败

1. 查看服务日志
2. 检查GPU/内存是否充足
3. 尝试减小图片尺寸（如256x256）

## 更多信息

详细API文档请参考：`API_GUIDE.md`


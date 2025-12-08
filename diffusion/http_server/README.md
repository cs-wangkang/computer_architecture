# HTTP 服务目录

本目录包含 Stable Diffusion 3.5 Large Turbo 的 HTTP API 服务相关文件。

## 文件说明

- `app.py` - HTTP 服务主程序
- `test_api.py` - API 测试脚本
- `start_server.sh` - 服务启动脚本
- `API_GUIDE.md` - API 使用指南
- `image_index.json` - 图片索引文件（自动生成）

## 目录结构

```
diffusion/
├── http_server/          # HTTP 服务目录
│   ├── app.py           # 服务主程序
│   ├── test_api.py      # 测试脚本
│   ├── start_server.sh  # 启动脚本
│   ├── API_GUIDE.md     # API 文档
│   └── image_index.json # 索引文件（自动生成）
├── generated_image/     # 生成的图片存储目录
│   └── image_*.png      # 生成的图片文件
├── inference.py         # 模型推理模块（被http_server引用）
└── requirements.txt     # 依赖文件
```

## 路径说明

- **图片存储**: `../generated_image/` (相对于 http_server 目录)
- **索引文件**: `./image_index.json` (在 http_server 目录内)
- **模型模块**: 从父目录导入 `inference.py`

## 启动服务

```bash
# 方式1: 使用启动脚本
conda activate diffusion
cd /root/autodl-tmp/project/diffusion/http_server
./start_server.sh

# 方式2: 直接运行
conda activate diffusion
cd /root/autodl-tmp/project/diffusion/http_server
python app.py --host 0.0.0.0 --port 5000
```

## 使用说明

详细的使用说明请参考 `API_GUIDE.md` 文件。


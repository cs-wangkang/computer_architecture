#!/bin/bash
# 启动Stable Diffusion HTTP服务

# 激活conda环境
echo "激活conda环境 diffusion..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate diffusion

# 进入http_server目录
cd /root/autodl-tmp/project/diffusion/http_server

# 检查依赖
echo "检查依赖..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖..."
    pip install -r ../requirements.txt -i https://pypi.org/simple
fi

# 启动服务
echo "启动HTTP服务..."
python app.py --host 0.0.0.0 --port 5000


#!/bin/bash
# Stable Diffusion 3.5 Large Turbo 快速启动脚本

# 激活conda环境
echo "激活conda环境 qiwen7B..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qiwen7B

# 检查是否已安装依赖
echo "检查依赖..."
python -c "import diffusers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖..."
    pip install -r requirements.txt
else
    echo "依赖已安装"
fi

# 运行推理（如果提供了参数）
if [ $# -gt 0 ]; then
    echo "运行推理..."
    python inference.py "$@"
else
    echo "使用方法:"
    echo "  ./run.sh --prompt 'your prompt here'"
    echo ""
    echo "或直接运行:"
    echo "  python inference.py --prompt 'your prompt here'"
fi



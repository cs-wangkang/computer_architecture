#!/bin/bash
# 运行所有benchmark测试的脚本

echo "=========================================="
echo "Qwen7B 推理性能基准测试"
echo "=========================================="

# 激活conda环境
echo "激活conda环境: qiwen7B_optimaze"

# 初始化conda（使用conda的完整路径）
CONDA_BASE="/root/autodl-tmp/miniconda3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    # 如果标准路径不存在，尝试从PATH中找到conda
    CONDA_BASE=$(dirname $(dirname $(which conda 2>/dev/null || echo "/root/autodl-tmp/miniconda3")))
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "错误: 无法找到conda初始化脚本"
        echo "请手动激活conda环境后运行此脚本"
        exit 1
    fi
fi

conda activate qiwen7B_optimaze

# 检查环境
if [ $? -ne 0 ]; then
    echo "错误: 无法激活conda环境 qiwen7B_optimaze"
    exit 1
fi

echo "当前环境: $(conda info --envs | grep '*')"
echo ""

# 运行基础推理测试
echo "=========================================="
echo "1. 运行基础推理性能测试"
echo "=========================================="
python benchmark_basic.py --num_samples 10 --warmup 2 --max_length 512

if [ $? -ne 0 ]; then
    echo "错误: 基础推理测试失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "2. 运行VLLM优化推理性能测试"
echo "=========================================="
echo "注意: VLLM需要transformers>=4.56.0，但当前环境使用transformers 4.40.0以支持transformers_stream_generator"
echo "如果测试失败，这是预期的版本冲突。可以使用基础推理测试。"
echo ""

python benchmark_vllm.py --num_samples 10 --warmup 2 --max_tokens 512

if [ $? -ne 0 ]; then
    echo ""
    echo "警告: VLLM优化推理测试失败（可能是版本冲突）"
    echo "这是预期的，因为："
    echo "  - transformers_stream_generator 需要 transformers<=4.40.0"
    echo "  - VLLM 需要 transformers>=4.56.0"
    echo ""
    echo "建议："
    echo "  1. 使用基础推理测试（已完成）"
    echo "  2. 如需使用VLLM，请创建单独的conda环境并升级transformers"
    echo ""
fi

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo "结果文件:"
echo "  - benchmark_basic_results.json"
echo "  - benchmark_vllm_results.json"
echo "=========================================="


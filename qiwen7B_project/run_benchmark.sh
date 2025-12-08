#!/bin/bash
# Qwen7B 基准测试运行脚本

# 激活conda环境
source /root/miniconda3/bin/activate qiwen7B

# 进入项目目录
cd /root/autodl-tmp/project/qiwen7B_project

echo "=========================================="
echo "Qwen7B 性能基准测试"
echo "=========================================="

# 运行基础推理基准测试
echo ""
echo "1. 运行基础推理基准测试..."
python benchmark_basic.py \
    --model-name Qwen/Qwen-7B-Chat \
    --num-samples 5 \
    --max-length 256 \
    --warmup 1 \
    --output benchmark_basic_results.json

# 运行优化推理基准测试（4-bit量化）
echo ""
echo "2. 运行优化推理基准测试（4-bit量化）..."
python benchmark_optimized.py \
    --model-name Qwen/Qwen-7B-Chat \
    --num-samples 5 \
    --max-length 256 \
    --warmup 1 \
    --quantization 4bit \
    --batch-size 1 \
    --output benchmark_optimized_4bit_results.json

echo ""
echo "基准测试完成！"
echo "结果文件："
echo "  - benchmark_basic_results.json"
echo "  - benchmark_optimized_4bit_results.json"


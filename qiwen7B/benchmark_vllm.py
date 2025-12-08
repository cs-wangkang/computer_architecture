"""
VLLM优化推理性能基准测试脚本
"""
import json
import time
import argparse
import sys
import os

# 禁用VLLM V1引擎（Qwen模型不支持V1引擎的generate runner）
# 必须在导入vllm之前设置
os.environ["VLLM_USE_V1"] = "0"

try:
    from vllm_inference import Qwen7BVLLMInference
except ImportError as e:
    print("=" * 60)
    print("错误: 无法导入VLLM推理模块")
    print("=" * 60)
    print(str(e))
    print("=" * 60)
    sys.exit(1)


def run_benchmark(model_name: str = "Qwen/Qwen-7B-Chat",
                  num_samples: int = 10,
                  warmup: int = 2,
                  max_tokens: int = 512,
                  tensor_parallel_size: int = 1,
                  gpu_memory_utilization: float = 0.9,
                  output_file: str = "benchmark_vllm_results.json"):
    """
    运行VLLM优化推理性能测试
    
    Args:
        model_name: 模型名称
        num_samples: 测试样本数
        warmup: 预热轮数
        max_tokens: 最大生成token数
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU内存利用率
        output_file: 结果输出文件
    """
    # 测试提示词
    test_prompts = [
        "介绍一下你的模型结构和参数",
        "请解释一下什么是深度学习",
        "如何优化大语言模型的推理性能？",
        "Python中如何实现多线程编程？",
        "请介绍一下Transformer架构的核心思想",
        "什么是注意力机制？",
        "如何训练一个语言模型？",
        "请解释一下梯度下降算法",
        "什么是过拟合？如何防止？",
        "请介绍一下BERT模型的特点"
    ]
    
    # 如果测试样本数少于提示词数量，只使用前num_samples个
    test_prompts = test_prompts[:num_samples]
    
    print("=" * 60)
    print("VLLM优化推理性能基准测试")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"测试样本数: {num_samples}")
    print(f"预热轮数: {warmup}")
    print(f"最大生成Tokens: {max_tokens}")
    print(f"张量并行大小: {tensor_parallel_size}")
    print(f"GPU内存利用率: {gpu_memory_utilization}")
    print("=" * 60)
    
    # 记录模型加载时间
    load_start = time.time()
    inference = Qwen7BVLLMInference(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )
    load_time = time.time() - load_start
    
    print(f"\n模型加载时间: {load_time:.2f} 秒\n")
    
    # 预热
    print(f"预热 {warmup} 轮...")
    for i in range(warmup):
        _ = inference.generate(test_prompts[0], max_tokens=max_tokens)
        print(f"预热完成 {i+1}/{warmup}")
    
    # 单样本测试
    print("\n开始单样本性能测试...")
    latencies = []
    token_counts = []
    results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"测试样本 {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        start_time = time.time()
        result = inference.generate(prompt, max_tokens=max_tokens)
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
        
        # 计算生成的token数量（简单估算）
        token_count = len(result.split())  # 简单估算
        token_counts.append(token_count)
        
        results.append({
            "prompt": prompt,
            "response": result[:100] + "..." if len(result) > 100 else result,
            "latency_ms": latency * 1000,
            "tokens": token_count
        })
        
        print(f"  延迟: {latency*1000:.2f}ms, Tokens: {token_count}")
    
    # 批量测试
    print("\n开始批量推理性能测试...")
    batch_start = time.time()
    batch_results = inference.generate_batch(test_prompts, max_tokens=max_tokens)
    batch_time = time.time() - batch_start
    
    # 统计结果
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    throughput = avg_tokens / avg_latency if avg_latency > 0 else 0
    batch_throughput = (avg_tokens * len(test_prompts)) / batch_time if batch_time > 0 else 0
    
    sorted_latencies = sorted(latencies)
    p50_idx = len(sorted_latencies) // 2
    p95_idx = int(len(sorted_latencies) * 0.95)
    p99_idx = int(len(sorted_latencies) * 0.99)
    
    benchmark_results = {
        "model_name": model_name,
        "inference_type": "vllm_optimized",
        "total_samples": len(test_prompts),
        "warmup_rounds": warmup,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "model_load_time_sec": load_time,
        "avg_latency_ms": avg_latency * 1000,
        "min_latency_ms": min(latencies) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "p50_latency_ms": sorted_latencies[p50_idx] * 1000,
        "p95_latency_ms": sorted_latencies[p95_idx] * 1000,
        "p99_latency_ms": sorted_latencies[p99_idx] * 1000 if len(sorted_latencies) > 1 else sorted_latencies[0] * 1000,
        "avg_tokens": avg_tokens,
        "throughput_tokens_per_sec": throughput,
        "batch_time_sec": batch_time,
        "batch_throughput_tokens_per_sec": batch_throughput,
        "total_time_sec": sum(latencies),
        "samples": results
    }
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("性能测试结果摘要")
    print("=" * 60)
    print(f"模型加载时间: {load_time:.2f} 秒")
    print(f"平均延迟: {avg_latency*1000:.2f} ms")
    print(f"最小延迟: {min(latencies)*1000:.2f} ms")
    print(f"最大延迟: {max(latencies)*1000:.2f} ms")
    print(f"P50延迟: {sorted_latencies[p50_idx]*1000:.2f} ms")
    print(f"P95延迟: {sorted_latencies[p95_idx]*1000:.2f} ms")
    print(f"P99延迟: {sorted_latencies[p99_idx]*1000:.2f} ms")
    print(f"平均生成Tokens: {avg_tokens:.2f}")
    print(f"单样本吞吐量: {throughput:.2f} tokens/秒")
    print(f"批量推理时间: {batch_time:.2f} 秒")
    print(f"批量推理吞吐量: {batch_throughput:.2f} tokens/秒")
    print(f"总测试时间: {sum(latencies):.2f} 秒")
    print("=" * 60)
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM优化推理性能基准测试")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B-Chat",
                        help="模型名称")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="测试样本数")
    parser.add_argument("--warmup", type=int, default=2,
                        help="预热轮数")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="最大生成token数")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU内存利用率")
    parser.add_argument("--output", type=str, default="benchmark_vllm_results.json",
                        help="结果输出文件")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_name=args.model_name,
        num_samples=args.num_samples,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_file=args.output
    )


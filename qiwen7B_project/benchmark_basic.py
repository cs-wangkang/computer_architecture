"""
Qwen7B 基础推理性能基准测试脚本
"""
import argparse
import json
import time
from basic_inference import Qwen7BInference


def main():
    parser = argparse.ArgumentParser(description="Qwen7B基础推理性能基准测试")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen-7B-Chat", 
                       help="模型名称或路径")
    parser.add_argument("--device", type=str, default="auto", 
                       help="设备类型 (auto/cuda/cpu)")
    parser.add_argument("--warmup", type=int, default=2, 
                       help="预热轮数")
    parser.add_argument("--num-samples", type=int, default=10, 
                       help="测试样本数量")
    parser.add_argument("--max-length", type=int, default=512, 
                       help="最大生成长度")
    parser.add_argument("--output", type=str, default="benchmark_basic_results.json", 
                       help="结果输出文件")
    
    args = parser.parse_args()
    
    # 测试提示词列表
    test_prompts = [
        "请介绍一下人工智能的发展历程。",
        "解释一下什么是深度学习。",
        "Python和Java有什么区别？",
        "如何优化数据库查询性能？",
        "什么是机器学习中的过拟合？",
        "介绍一下Transformer架构。",
        "解释一下注意力机制的工作原理。",
        "什么是梯度下降算法？",
        "介绍一下自然语言处理的应用场景。",
        "如何训练一个语言模型？",
    ]
    
    # 如果样本数超过提示词数量，循环使用
    if args.num_samples > len(test_prompts):
        test_prompts = test_prompts * (args.num_samples // len(test_prompts) + 1)
    test_prompts = test_prompts[:args.num_samples]
    
    print("=" * 60)
    print("Qwen7B 基础推理性能基准测试")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"设备: {args.device}")
    print(f"测试样本数: {args.num_samples}")
    print(f"最大生成长度: {args.max_length}")
    print(f"预热轮数: {args.warmup}")
    print("=" * 60)
    
    # 初始化模型
    start_time = time.time()
    inference = Qwen7BInference(model_name=args.model_name, device=args.device)
    load_time = time.time() - start_time
    
    print(f"\n模型加载时间: {load_time:.2f} 秒\n")
    
    # 运行基准测试
    results = inference.benchmark(
        prompts=test_prompts,
        warmup=args.warmup,
        max_length=args.max_length
    )
    
    # 添加模型加载时间
    results["model_load_time_sec"] = load_time
    
    # 打印结果
    print("\n" + "=" * 60)
    print("性能测试结果")
    print("=" * 60)
    print(f"测试样本数: {results['total_samples']}")
    print(f"平均延迟: {results['avg_latency_ms']:.2f} ms")
    print(f"最小延迟: {results['min_latency_ms']:.2f} ms")
    print(f"最大延迟: {results['max_latency_ms']:.2f} ms")
    print(f"P50延迟: {results['p50_latency_ms']:.2f} ms")
    print(f"P95延迟: {results['p95_latency_ms']:.2f} ms")
    print(f"P99延迟: {results['p99_latency_ms']:.2f} ms")
    print(f"平均生成token数: {results['avg_tokens']:.2f}")
    print(f"吞吐量: {results['throughput_tokens_per_sec']:.2f} tokens/s")
    print(f"模型加载时间: {results['model_load_time_sec']:.2f} s")
    print("=" * 60)
    
    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()


"""
Qwen7B 基础推理实现
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from typing import List, Optional


class Qwen7BInference:
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", device: str = "auto"):
        """
        初始化Qwen7B模型
        
        Args:
            model_name: HuggingFace模型名称
            device: 设备类型，"auto"表示自动选择
        """
        print(f"正在加载模型: {model_name}")
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"模型加载完成，设备: {self.device}")
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.8, do_sample: bool = True) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否使用采样
            
        Returns:
            生成的文本
        """
        # 格式化输入（Qwen聊天格式）
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids
            )
        ]
        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def generate_batch(self, prompts: List[str], max_length: int = 512, 
                       temperature: float = 0.7, top_p: float = 0.8, 
                       do_sample: bool = True) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示词列表
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否使用采样
            
        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_length, temperature, top_p, do_sample)
            results.append(result)
        return results
    
    def benchmark(self, prompts: List[str], warmup: int = 2, 
                  max_length: int = 512, **kwargs) -> dict:
        """
        性能基准测试
        
        Args:
            prompts: 测试提示词列表
            warmup: 预热轮数
            max_length: 最大生成长度
            **kwargs: 其他生成参数
            
        Returns:
            性能统计字典
        """
        # 预热
        print(f"预热 {warmup} 轮...")
        for i in range(warmup):
            _ = self.generate(prompts[0] if prompts else "Hello", max_length=max_length, **kwargs)
        
        # 测试
        print("开始性能测试...")
        latencies = []
        token_counts = []
        
        for prompt in prompts:
            start_time = time.time()
            result = self.generate(prompt, max_length=max_length, **kwargs)
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            # 计算生成的token数量
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            output_ids = self.tokenizer.encode(result, return_tensors="pt")
            token_count = output_ids.shape[1] - input_ids.shape[1]
            token_counts.append(token_count)
        
        # 统计结果
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(token_counts) / len(token_counts)
        throughput = avg_tokens / avg_latency if avg_latency > 0 else 0
        
        return {
            "total_samples": len(prompts),
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "avg_tokens": avg_tokens,
            "throughput_tokens_per_sec": throughput,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] * 1000,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000 if len(latencies) > 1 else latencies[0] * 1000
        }


if __name__ == "__main__":
    # 测试代码
    inference = Qwen7BInference()
    
    test_prompt = "介绍一下你的模型结构和参数"
    print(f"输入: {test_prompt}")
    print("生成中...")
    
    result = inference.generate(test_prompt, max_length=256)
    print(f"输出: {result}")


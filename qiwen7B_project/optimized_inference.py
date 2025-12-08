"""
Qwen7B 优化推理实现
包括量化、KV缓存优化等
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import time
from typing import List, Optional
import os


class Qwen7BOptimizedInference:
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", device: str = "auto",
                 use_quantization: bool = True, quantization_type: str = "4bit",
                 use_flash_attention: bool = True, use_torch_compile: bool = False):
        """
        初始化优化后的Qwen7B模型
        
        Args:
            model_name: HuggingFace模型名称
            device: 设备类型，"auto"表示自动选择
            use_quantization: 是否使用量化
            quantization_type: 量化类型，"4bit"或"8bit"
            use_flash_attention: 是否使用Flash Attention（如果可用）
            use_torch_compile: 是否使用torch.compile优化
        """
        print(f"正在加载优化模型: {model_name}")
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantization = use_quantization and self.device == "cuda"
        self.quantization_type = quantization_type
        self.use_flash_attention = use_flash_attention
        self.use_torch_compile = use_torch_compile
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 配置量化（仅CUDA）
        quantization_config = None
        if self.use_quantization:
            if quantization_type == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("使用4-bit量化")
            elif quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                print("使用8-bit量化")
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if self.use_quantization:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
            model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
        
        # Flash Attention设置
        if self.use_flash_attention and self.device == "cuda":
            try:
                # 检查是否安装了flash-attn
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("使用Flash Attention 2")
            except ImportError:
                print("Flash Attention未安装，使用默认注意力机制")
                self.use_flash_attention = False
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not self.use_quantization and self.device != "cuda":
            self.model = self.model.to(self.device)
        
        # Torch compile优化（PyTorch 2.0+）
        if self.use_torch_compile:
            try:
                if hasattr(torch, "compile"):
                    print("使用torch.compile优化")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                else:
                    print("当前PyTorch版本不支持torch.compile")
                    self.use_torch_compile = False
            except Exception as e:
                print(f"torch.compile优化失败: {e}")
                self.use_torch_compile = False
        
        self.model.eval()
        print(f"优化模型加载完成，设备: {self.device}")
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.8, do_sample: bool = True, use_cache: bool = True) -> str:
        """
        生成文本（使用KV缓存优化）
        
        Args:
            prompt: 输入提示词
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否使用采样
            use_cache: 是否使用KV缓存（默认开启以提升性能）
            
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
        
        # 生成（KV缓存自动启用）
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,  # KV缓存优化
                pad_token_id=self.tokenizer.eos_token_id,
                num_beams=1 if do_sample else 1,  # 贪婪解码更快
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
                       do_sample: bool = True, batch_size: int = 1) -> List[str]:
        """
        批量生成文本（批量处理优化）
        
        Args:
            prompts: 输入提示词列表
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否使用采样
            batch_size: 批处理大小
            
        Returns:
            生成的文本列表
        """
        results = []
        
        # 批量处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # 格式化批量输入
            batch_texts = []
            for prompt in batch_prompts:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_texts.append(text)
            
            # Tokenize批量输入
            model_inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # 批量生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码批量输出
            batch_results = []
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                generated_token_ids = output_ids[len(input_ids):]
                response = self.tokenizer.decode(
                    generated_token_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                batch_results.append(response)
            
            results.extend(batch_results)
        
        return results
    
    def benchmark(self, prompts: List[str], warmup: int = 2, 
                  max_length: int = 512, batch_size: int = 1, **kwargs) -> dict:
        """
        性能基准测试
        
        Args:
            prompts: 测试提示词列表
            warmup: 预热轮数
            max_length: 最大生成长度
            batch_size: 批处理大小
            **kwargs: 其他生成参数
            
        Returns:
            性能统计字典
        """
        # 预热
        print(f"预热 {warmup} 轮...")
        for i in range(warmup):
            _ = self.generate(prompts[0] if prompts else "Hello", max_length=max_length, **kwargs)
        
        # 如果是批处理，先预热批处理路径
        if batch_size > 1:
            print(f"预热批处理 (batch_size={batch_size})...")
            batch_prompts = prompts[:batch_size] if len(prompts) >= batch_size else prompts
            _ = self.generate_batch(batch_prompts, max_length=max_length, batch_size=batch_size, **kwargs)
        
        # 测试
        print("开始性能测试...")
        latencies = []
        token_counts = []
        
        if batch_size > 1:
            # 批处理测试
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                
                start_time = time.time()
                results = self.generate_batch(
                    batch_prompts, 
                    max_length=max_length, 
                    batch_size=batch_size,
                    **kwargs
                )
                end_time = time.time()
                
                latency = end_time - start_time
                latencies.append(latency / len(batch_prompts))  # 每个样本的平均延迟
                
                for prompt, result in zip(batch_prompts, results):
                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                    output_ids = self.tokenizer.encode(result, return_tensors="pt")
                    token_count = output_ids.shape[1] - input_ids.shape[1]
                    token_counts.append(token_count)
        else:
            # 单个处理
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
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        throughput = avg_tokens / avg_latency if avg_latency > 0 else 0
        
        return {
            "total_samples": len(prompts),
            "batch_size": batch_size,
            "quantization": f"{self.quantization_type}bit" if self.use_quantization else "none",
            "flash_attention": self.use_flash_attention,
            "torch_compile": self.use_torch_compile,
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min(latencies) * 1000 if latencies else 0,
            "max_latency_ms": max(latencies) * 1000 if latencies else 0,
            "avg_tokens": avg_tokens,
            "throughput_tokens_per_sec": throughput,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] * 1000 if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000 if latencies else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000 if latencies and len(latencies) > 1 else (latencies[0] * 1000 if latencies else 0)
        }


if __name__ == "__main__":
    # 测试代码
    inference = Qwen7BOptimizedInference(use_quantization=True, quantization_type="4bit")
    
    test_prompt = "请介绍一下人工智能的发展历程。"
    print(f"输入: {test_prompt}")
    print("生成中...")
    
    result = inference.generate(test_prompt, max_length=256)
    print(f"输出: {result}")


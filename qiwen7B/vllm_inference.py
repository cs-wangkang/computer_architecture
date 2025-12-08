"""
Qwen7B VLLM优化推理实现
"""
from typing import List, Optional
import time
import os

# 禁用VLLM V1引擎（Qwen模型不支持V1引擎的generate runner）
# 必须在导入vllm之前设置环境变量
os.environ["VLLM_USE_V1"] = "0"

# 尝试导入VLLM，如果失败则给出清晰的错误提示
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    VLLM_ERROR = str(e)
    # 创建占位符类以避免导入错误
    class LLM:
        pass
    class SamplingParams:
        pass


class Qwen7BVLLMInference:
    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", 
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None,
                 trust_remote_code: bool = True):
        """
        初始化Qwen7B模型（使用VLLM优化）
        
        Args:
            model_name: HuggingFace模型名称
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU内存利用率
            max_model_len: 最大模型长度
            trust_remote_code: 是否信任远程代码
        """
        if not VLLM_AVAILABLE:
            error_msg = (
                f"VLLM不可用，无法使用VLLM优化推理。\n"
                f"错误信息: {VLLM_ERROR}\n\n"
                f"可能的原因：\n"
                f"1. VLLM需要transformers>=4.56.0，但当前安装的是transformers 4.40.0（为了支持transformers_stream_generator）\n"
                f"2. 这两个库的版本要求冲突：\n"
                f"   - transformers_stream_generator 需要 transformers<=4.40.0\n"
                f"   - VLLM 需要 transformers>=4.56.0\n\n"
                f"解决方案：\n"
                f"1. 如果不需要transformers_stream_generator，可以升级transformers：\n"
                f"   pip install transformers>=4.56.0\n"
                f"2. 如果不需要VLLM，可以使用基础推理（basic_inference.py）\n"
                f"3. 或者创建两个不同的conda环境分别用于基础推理和VLLM推理"
            )
            raise ImportError(error_msg)
        
        print(f"正在加载模型: {model_name} (使用VLLM优化)")
        
        # 使用VLLM 0.11.2版本，兼容Qwen模型
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype="float16"
        )
        
        print("模型加载完成（VLLM优化）")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.8, stop: Optional[List[str]] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            stop: 停止词列表
            
        Returns:
            生成的文本
        """
        # 格式化输入（Qwen聊天格式）
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm.llm_engine.model_config.hf_config.hf_model_name,
            trust_remote_code=True
        )
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )
        
        # 生成
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        
        return outputs[0].outputs[0].text
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 512, 
                       temperature: float = 0.7, top_p: float = 0.8,
                       stop: Optional[List[str]] = None) -> List[str]:
        """
        批量生成文本（VLLM支持高效的批量推理）
        
        Args:
            prompts: 输入提示词列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            stop: 停止词列表
            
        Returns:
            生成的文本列表
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm.llm_engine.model_config.hf_config.hf_model_name,
            trust_remote_code=True
        )
        
        # 格式化所有提示词
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )
        
        # 批量生成
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]


if __name__ == "__main__":
    # 测试代码
    inference = Qwen7BVLLMInference()
    
    test_prompt = "介绍一下你的模型结构和参数"
    print(f"输入: {test_prompt}")
    print("生成中...")
    
    result = inference.generate(test_prompt, max_tokens=256)
    print(f"输出: {result}")


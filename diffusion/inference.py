"""
Stable Diffusion 3.5 Large Turbo 推理实现
"""
import torch
from PIL import Image
import argparse
import os
from typing import Optional, Tuple

try:
    from diffusers import StableDiffusion3Pipeline
except ImportError:
    try:
        from diffusers import StableDiffusionPipeline as StableDiffusion3Pipeline
    except ImportError:
        raise ImportError(
            "无法导入diffusers库。请运行: pip install -r requirements.txt"
        )


class StableDiffusion35Inference:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-large-turbo",
        device: str = "auto",
        dtype: str = "float16"
    ):
        """
        初始化Stable Diffusion 3.5 Large Turbo模型
        
        Args:
            model_id: HuggingFace模型ID
            device: 设备类型，"auto"表示自动选择
            dtype: 数据类型，"float16"或"float32"
        """
        print(f"正在加载模型: {model_id}")
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if dtype == "float16" else torch.float32
        
        # 加载pipeline
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        
        # 移动到指定设备
        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)
            # 启用内存优化（如果可用）
            if hasattr(self.pipe, 'enable_model_cpu_offload'):
                self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        
        print(f"模型加载完成，设备: {self.device}, 数据类型: {dtype}")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        生成图像
        
        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_inference_steps: 推理步数（turbo模型通常只需要4步）
            guidance_scale: 引导尺度（turbo模型通常为0.0）
            width: 图像宽度
            height: 图像高度
            seed: 随机种子
            output_path: 输出路径（可选）
            
        Returns:
            生成的PIL图像
        """
        # 设置随机种子
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        print(f"正在生成图像...")
        print(f"提示词: {prompt}")
        if negative_prompt:
            print(f"负向提示词: {negative_prompt}")
        
        # 生成图像
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            ).images[0]
        
        # 保存图像
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            image.save(output_path)
            print(f"图像已保存到: {output_path}")
        
        return image
    
    def generate_batch(
        self,
        prompts: list,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        width: int = 1024,
        height: int = 1024,
        output_dir: str = "outputs"
    ) -> list:
        """
        批量生成图像
        
        Args:
            prompts: 提示词列表
            negative_prompt: 负向提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度
            width: 图像宽度
            height: 图像高度
            output_dir: 输出目录
            
        Returns:
            生成的图像列表
        """
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n处理第 {i+1}/{len(prompts)} 个提示词...")
            output_path = os.path.join(output_dir, f"image_{i+1:03d}.png")
            image = self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                output_path=output_path
            )
            images.append(image)
        
        return images


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion 3.5 Large Turbo 推理")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="图像生成提示词"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="负向提示词"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="输出图像路径"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="推理步数（默认4步，适合turbo模型）"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="图像宽度"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="图像高度"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-3.5-large-turbo",
        help="模型ID"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备类型（auto/cuda/cpu）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="数据类型"
    )
    
    args = parser.parse_args()
    
    # 初始化模型
    inference = StableDiffusion35Inference(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype
    )
    
    # 生成图像
    image = inference.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        output_path=args.output
    )
    
    print("\n生成完成！")


if __name__ == "__main__":
    main()


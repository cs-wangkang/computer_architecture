"""
Stable Diffusion 3.5 Large Turbo 使用示例
"""
from inference import StableDiffusion35Inference


def example_single():
    """单张图像生成示例"""
    print("=" * 50)
    print("示例 1: 单张图像生成")
    print("=" * 50)
    
    # 初始化模型
    inference = StableDiffusion35Inference()
    
    # 生成图像
    image = inference.generate(
        prompt="a beautiful landscape with mountains and a lake, sunset, peaceful, high quality",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=4,
        width=1024,
        height=1024,
        seed=42,
        output_path="example_landscape.png"
    )
    
    print("图像已生成: example_landscape.png")


def example_batch():
    """批量生成示例"""
    print("\n" + "=" * 50)
    print("示例 2: 批量图像生成")
    print("=" * 50)
    
    # 初始化模型
    inference = StableDiffusion35Inference()
    
    # 定义多个提示词
    prompts = [
        "a cute cat playing with yarn, photorealistic",
        "a futuristic robot in a cyberpunk city, neon lights",
        "a serene Japanese garden with cherry blossoms, peaceful"
    ]
    
    # 批量生成
    images = inference.generate_batch(
        prompts=prompts,
        negative_prompt="blurry, low quality",
        num_inference_steps=4,
        width=1024,
        height=1024,
        output_dir="outputs"
    )
    
    print(f"已生成 {len(images)} 张图像，保存在 outputs/ 目录")


if __name__ == "__main__":
    # 运行示例
    example_single()
    # example_batch()  # 取消注释以运行批量生成示例


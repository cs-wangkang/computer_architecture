"""
测试环境配置和依赖安装
"""
import sys

def test_imports():
    """测试必要的库是否可以导入"""
    print("正在测试依赖库...")
    
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ torch 未安装: {e}")
        return False
    
    try:
        import diffusers
        print(f"✓ diffusers {diffusers.__version__}")
    except ImportError as e:
        print(f"✗ diffusers 未安装: {e}")
        print("  请运行: pip install -r requirements.txt")
        return False
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers 未安装: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow")
    except ImportError as e:
        print(f"✗ PIL/Pillow 未安装: {e}")
        return False
    
    try:
        import accelerate
        print(f"✓ accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"✗ accelerate 未安装: {e}")
        return False
    
    return True

def test_model_import():
    """测试模型类是否可以导入"""
    print("\n正在测试模型类导入...")
    try:
        from inference import StableDiffusion35Inference
        print("✓ StableDiffusion35Inference 类导入成功")
        return True
    except ImportError as e:
        print(f"✗ 无法导入模型类: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Stable Diffusion 3.5 Large Turbo 环境测试")
    print("=" * 50)
    print()
    
    success = test_imports()
    if success:
        test_model_import()
        print("\n" + "=" * 50)
        print("环境配置检查完成！")
        print("=" * 50)
        print("\n下一步:")
        print("1. 运行示例: python example.py")
        print("2. 或使用命令行: python inference.py --prompt 'your prompt'")
    else:
        print("\n" + "=" * 50)
        print("环境配置不完整，请安装缺失的依赖")
        print("=" * 50)
        print("\n运行: pip install -r requirements.txt")
        sys.exit(1)




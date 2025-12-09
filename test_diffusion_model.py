#!/usr/bin/env python3
"""
测试Stable Diffusion模型是否可以正常加载
"""
import sys
import os

# 添加diffusion目录到路径
sys.path.insert(0, '/root/autodl-tmp/project/diffusion')

def test_model_loading():
    """测试模型加载"""
    print("=" * 70)
    print("测试 Stable Diffusion 3.5 Large Turbo 模型加载")
    print("=" * 70)
    
    try:
        print("\n1. 导入必要的库...")
        import torch
        print(f"   ✓ PyTorch版本: {torch.__version__}")
        print(f"   ✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA设备: {torch.cuda.get_device_name(0)}")
        
        print("\n2. 导入diffusers库...")
        from diffusers import StableDiffusion3Pipeline
        print("   ✓ diffusers库导入成功")
        
        print("\n3. 导入模型类...")
        from inference import StableDiffusion35Inference
        print("   ✓ StableDiffusion35Inference类导入成功")
        
        print("\n4. 初始化模型（这可能需要一些时间）...")
        print("   正在从缓存加载模型...")
        
        # 设置环境变量使用本地缓存
        os.environ['HF_HOME'] = '/root/autodl-tmp/project/hf_cache'
        
        inference = StableDiffusion35Inference(
            model_id="stabilityai/stable-diffusion-3.5-large-turbo",
            device="auto",
            dtype="float16"
        )
        
        print("\n" + "=" * 70)
        print("✓ 模型加载成功！")
        print("=" * 70)
        print(f"   设备: {inference.device}")
        print(f"   数据类型: {inference.dtype}")
        print("\n模型已准备就绪，可以开始生成图像了！")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ 文件未找到错误: {e}")
        print("   可能缺少模型文件，请检查下载是否完整")
        return False
    except ImportError as e:
        print(f"\n✗ 导入错误: {e}")
        print("   请检查依赖是否已正确安装")
        return False
    except Exception as e:
        print(f"\n✗ 错误: {type(e).__name__}: {e}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)



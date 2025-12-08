"""
快速测试脚本 - 验证环境配置
"""
import sys

def test_imports():
    """测试必要的包是否已安装"""
    print("检查依赖包...")
    
    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "vllm": "VLLM (可选，用于优化推理)"
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} 已安装")
        except ImportError:
            print(f"✗ {name} 未安装")
            if package != "vllm":  # vllm是可选的
                missing.append(package)
    
    if missing:
        print(f"\n缺少必要的包: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("\n所有必要的包已安装！")
    return True

def test_cuda():
    """测试CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  当前GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✗ CUDA 不可用（将使用CPU，速度较慢）")
            return False
    except Exception as e:
        print(f"✗ 检查CUDA时出错: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("环境配置检查")
    print("=" * 60)
    
    imports_ok = test_imports()
    print()
    cuda_ok = test_cuda()
    
    print("\n" + "=" * 60)
    if imports_ok:
        print("✓ 环境配置正确，可以开始运行benchmark测试")
    else:
        print("✗ 请先安装依赖包")
    print("=" * 60)


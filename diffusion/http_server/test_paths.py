"""
测试路径配置是否正确
"""
import sys
from pathlib import Path

# 模拟app.py中的路径设置
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

print("=" * 50)
print("路径配置测试")
print("=" * 50)
print(f"当前目录: {current_dir}")
print(f"父目录: {parent_dir}")
print(f"generated_image目录: {parent_dir / 'generated_image'}")
print(f"索引文件: {current_dir / 'image_index.json'}")
print()

# 测试导入
try:
    from inference import StableDiffusion35Inference
    print("✓ inference模块导入成功")
except ImportError as e:
    print(f"✗ inference模块导入失败: {e}")

print()
print("=" * 50)
print("路径配置验证完成")
print("=" * 50)


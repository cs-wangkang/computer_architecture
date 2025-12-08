"""
测试HTTP API的示例脚本
"""
import requests
import json

# 服务地址
BASE_URL = "http://localhost:5000"


def test_health():
    """测试健康检查接口"""
    print("测试健康检查接口...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()


def test_info():
    """测试服务信息接口"""
    print("测试服务信息接口...")
    response = requests.get(f"{BASE_URL}/info")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_generate_post():
    """测试POST方式生成图片"""
    print("测试POST方式生成图片...")
    
    data = {
        "prompt": "a beautiful sunset over the ocean, vibrant colors, peaceful atmosphere",
        "negative_prompt": "blurry, low quality",
        "steps": 4,
        "width": 512,
        "height": 512,
        "seed": 42
    }
    
    response = requests.post(
        f"{BASE_URL}/generate",
        json=data,
        timeout=300  # 5分钟超时
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        # 保存图片
        output_path = "test_api_output.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"图片已保存到: {output_path}")
        print(f"图片大小: {len(response.content)} bytes")
    else:
        print(f"错误: {response.json()}")
    print()


def test_generate_get():
    """测试GET方式生成图片"""
    print("测试GET方式生成图片...")
    
    params = {
        "prompt": "a cute cat playing with yarn, photorealistic",
        "steps": 4,
        "width": 512,
        "height": 512
    }
    
    response = requests.get(
        f"{BASE_URL}/generate",
        params=params,
        timeout=300  # 5分钟超时
    )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        # 保存图片
        output_path = "test_api_output_get.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"图片已保存到: {output_path}")
        print(f"图片大小: {len(response.content)} bytes")
    else:
        print(f"错误: {response.json()}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Stable Diffusion API 测试")
    print("=" * 50)
    print()
    
    try:
        # 测试健康检查
        test_health()
        
        # 测试服务信息
        test_info()
        
        # 测试POST生成
        # test_generate_post()
        
        # 测试GET生成
        # test_generate_get()
        
        print("测试完成！")
        print("\n提示: 取消注释 test_generate_post() 或 test_generate_get() 来测试图片生成")
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到服务器")
        print("请确保服务已启动: python app.py")
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()


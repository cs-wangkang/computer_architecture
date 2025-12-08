"""
Stable Diffusion 3.5 Large Turbo HTTP 服务
提供RESTful API接口，接收prompt参数，生成图片并返回
"""
import os
import json
import sys
from pathlib import Path
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
import threading

# 添加父目录到路径，以便导入inference模块
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from inference import StableDiffusion35Inference

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置 - 使用绝对路径
BASE_DIR = Path(__file__).parent.parent  # http_server的父目录（diffusion目录）
GENERATED_IMAGE_DIR = BASE_DIR / "generated_image"
INDEX_FILE = BASE_DIR / "http_server" / "image_index.json"
DEFAULT_STEPS = 4
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512

# 确保输出目录存在
os.makedirs(GENERATED_IMAGE_DIR, exist_ok=True)

# 全局模型实例（单例模式）
_model_instance = None
_model_lock = threading.Lock()


def get_model():
    """获取模型实例（单例模式）"""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                print("正在初始化模型...")
                _model_instance = StableDiffusion35Inference()
                print("模型初始化完成")
    return _model_instance


def get_next_index():
    """获取下一个图片索引"""
    index_file = str(INDEX_FILE)
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                index = data.get('index', 0)
        except:
            index = 0
    else:
        index = 0
    
    # 自增索引
    next_index = index + 1
    
    # 保存新索引
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({'index': next_index}, f, ensure_ascii=False, indent=2)
    
    return next_index


def get_image_path(index):
    """根据索引获取图片路径"""
    return str(GENERATED_IMAGE_DIR / f"image_{index:06d}.png")


@app.route('/', methods=['GET'])
def index():
    """首页 - 返回Web界面"""
    web_interface_path = Path(__file__).parent / 'web_interface.html'
    if web_interface_path.exists():
        return send_file(str(web_interface_path), mimetype='text/html')
    else:
        return jsonify({
            'service': 'Stable Diffusion 3.5 Large Turbo API',
            'message': 'Web interface not found. Use /info for API documentation.',
            'endpoints': {
                'GET /': 'Web interface (if available)',
                'POST /generate': 'Generate image (JSON)',
                'GET /generate': 'Generate image (query params)',
                'GET /health': 'Health check',
                'GET /info': 'Service info'
            }
        })


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'message': 'Service is running'
    })


@app.route('/generate', methods=['POST'])
def generate_image():
    """
    生成图片接口
    
    请求参数（JSON）:
    - prompt: 图片生成提示词（必需）
    - negative_prompt: 负向提示词（可选）
    - steps: 推理步数（可选，默认4）
    - width: 图片宽度（可选，默认512）
    - height: 图片高度（可选，默认512）
    - seed: 随机种子（可选）
    
    返回:
    - 成功: 返回生成的图片文件
    - 失败: 返回JSON错误信息
    """
    try:
        # 获取请求数据
        if request.is_json:
            data = request.get_json()
        else:
            # 支持form-data格式
            data = request.form.to_dict()
        
        # 验证必需参数
        if 'prompt' not in data or not data['prompt']:
            return jsonify({
                'error': 'Missing required parameter: prompt'
            }), 400
        
        prompt = data['prompt']
        negative_prompt = data.get('negative_prompt', None)
        steps = int(data.get('steps', DEFAULT_STEPS))
        width = int(data.get('width', DEFAULT_WIDTH))
        height = int(data.get('height', DEFAULT_HEIGHT))
        seed = int(data['seed']) if 'seed' in data and data['seed'] else None
        
        # 参数验证
        if steps < 1 or steps > 50:
            return jsonify({
                'error': 'steps must be between 1 and 50'
            }), 400
        
        if width < 64 or width > 2048 or height < 64 or height > 2048:
            return jsonify({
                'error': 'width and height must be between 64 and 2048'
            }), 400
        
        # 获取下一个索引
        image_index = get_next_index()
        image_path = get_image_path(image_index)
        
        # 获取模型实例
        model = get_model()
        
        # 生成图片
        print(f"收到生成请求: prompt='{prompt}', index={image_index}")
        image = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            seed=seed,
            output_path=str(image_path)
        )
        
        # 返回图片文件
        return send_file(
            str(image_path),
            mimetype='image/png',
            as_attachment=False,
            download_name=f"image_{image_index:06d}.png"
        )
        
    except Exception as e:
        print(f"生成图片时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to generate image: {str(e)}'
        }), 500


@app.route('/generate', methods=['GET'])
def generate_image_get():
    """
    GET方式生成图片接口（用于简单测试）
    
    查询参数:
    - prompt: 图片生成提示词（必需）
    - negative_prompt: 负向提示词（可选）
    - steps: 推理步数（可选，默认4）
    - width: 图片宽度（可选，默认512）
    - height: 图片高度（可选，默认512）
    - seed: 随机种子（可选）
    """
    try:
        prompt = request.args.get('prompt')
        if not prompt:
            return jsonify({
                'error': 'Missing required parameter: prompt'
            }), 400
        
        negative_prompt = request.args.get('negative_prompt', None)
        steps = int(request.args.get('steps', DEFAULT_STEPS))
        width = int(request.args.get('width', DEFAULT_WIDTH))
        height = int(request.args.get('height', DEFAULT_HEIGHT))
        seed = int(request.args['seed']) if 'seed' in request.args else None
        
        # 获取下一个索引
        image_index = get_next_index()
        image_path = get_image_path(image_index)
        
        # 获取模型实例
        model = get_model()
        
        # 生成图片
        print(f"收到生成请求: prompt='{prompt}', index={image_index}")
        image = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            seed=seed,
            output_path=str(image_path)
        )
        
        # 返回图片文件
        return send_file(
            str(image_path),
            mimetype='image/png',
            as_attachment=False,
            download_name=f"image_{image_index:06d}.png"
        )
        
    except Exception as e:
        print(f"生成图片时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to generate image: {str(e)}'
        }), 500


@app.route('/info', methods=['GET'])
def get_info():
    """获取服务信息"""
    # 统计已生成的图片数量
    image_count = 0
    index_file = str(INDEX_FILE)
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                image_count = data.get('index', 0)
        except:
            pass
    
    return jsonify({
        'service': 'Stable Diffusion 3.5 Large Turbo API',
        'version': '1.0.0',
        'generated_images_count': image_count,
        'endpoints': {
            'POST /generate': '生成图片（JSON格式）',
            'GET /generate': '生成图片（查询参数）',
            'GET /health': '健康检查',
            'GET /info': '服务信息'
        }
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Stable Diffusion HTTP服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5000, help='监听端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Stable Diffusion 3.5 Large Turbo HTTP 服务")
    print("=" * 50)
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/info")
    print("=" * 50)
    
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


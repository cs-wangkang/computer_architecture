# CUDA 矩阵相乘测试项目

这是一个简单的 CUDA 矩阵相乘示例项目。

## 文件说明

- `matrix_multiply.cu`: CUDA 矩阵相乘核心代码
- `Makefile`: 编译脚本
- `README.md`: 项目说明文档

## 使用方法

1. 激活 conda 环境：
```bash
conda activate cuda_test
```

2. 编译代码：
```bash
make
```

3. 运行程序：
```bash
make run
# 或者
./matrix_multiply
```

4. 清理编译文件：
```bash
make clean
```

## 功能说明

程序实现了两个矩阵的相乘运算：
- 矩阵 A: M × N
- 矩阵 B: N × K
- 结果矩阵 C: M × K

程序会：
1. 在 GPU 上执行矩阵相乘
2. 在 CPU 上执行相同的计算用于验证
3. 比较 GPU 和 CPU 的结果
4. 显示执行时间和结果样本


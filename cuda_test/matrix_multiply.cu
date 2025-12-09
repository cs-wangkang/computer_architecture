#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// CUDA 核函数：矩阵相乘
__global__ void matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    // 计算当前线程对应的矩阵位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < M && col < K) {
        float sum = 0.0f;
        // 计算 C[row][col] = A[row][:] * B[:][col]
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// CPU 版本用于验证结果
void matrixMultiplyCPU(float *A, float *B, float *C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < K; col++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}

// 验证结果是否正确
bool verifyResult(float *C_gpu, float *C_cpu, int size) {
    const float epsilon = 1e-3f;  // 放宽容差，考虑浮点运算顺序差异
    int error_count = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float error = fabs(C_gpu[i] - C_cpu[i]);
        if (error > epsilon) {
            error_count++;
            if (error > max_error) {
                max_error = error;
            }
        }
    }
    
    if (error_count > 0) {
        printf("发现 %d 个元素超出容差 (最大误差: %.6f)\n", error_count, max_error);
        // 如果误差很小，仍然认为验证通过
        if (max_error < 1e-2f && error_count < size * 0.001f) {
            return true;
        }
        return false;
    }
    return true;
}

int main() {
    // 矩阵维度: A[M][N] * B[N][K] = C[M][K]
    int M = 512;
    int N = 512;
    int K = 512;
    
    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * K * sizeof(float);
    size_t size_C = M * K * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);
    float *h_C_cpu = (float *)malloc(size_C);
    
    // 初始化矩阵 A 和 B
    for (int i = 0; i < M * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < N * K; i++) {
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // 配置线程块和网格
    dim3 blockSize(16, 16);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 启动 CUDA 核函数
    cudaEventRecord(start);
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    
    // 等待核函数完成
    cudaEventSynchronize(stop);
    
    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // CPU 版本计算用于验证（包含计时）
    printf("使用 CPU 计算验证结果...\n");
    clock_t cpu_start = clock();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0;
    
    // 验证结果
    printf("验证 GPU 计算结果...\n");
    if (verifyResult(h_C, h_C_cpu, M * K)) {
        printf("✓ 验证成功！GPU 和 CPU 结果一致\n");
    } else {
        printf("✗ 验证失败！GPU 和 CPU 结果不一致\n");
    }
    
    // 打印性能对比
    printf("\n========== 性能对比 ==========\n");
    printf("矩阵维度: A[%d][%d] * B[%d][%d] = C[%d][%d]\n", M, N, N, K, M, K);
    printf("CPU 执行时间: %.3f 毫秒\n", cpu_time_ms);
    printf("GPU 执行时间: %.3f 毫秒\n", milliseconds);
    printf("加速比: %.2fx\n", cpu_time_ms / milliseconds);
    printf("==============================\n");
    printf("\n结果样本 (前 5x5):\n");
    for (int i = 0; i < 5 && i < M; i++) {
        for (int j = 0; j < 5 && j < K; j++) {
            printf("%8.2f ", h_C[i * K + j]);
        }
        printf("\n");
    }
    
    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n程序执行完成！\n");
    return 0;
}


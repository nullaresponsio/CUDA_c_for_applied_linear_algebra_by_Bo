#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    int blockRow = blockIdx.y, blockCol = blockIdx.x;
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0;
    int row = threadIdx.y, col = threadIdx.x;
    for (int m = 0; m < A.width / BLOCK_SIZE; ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
}

void MatMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A, d_B, d_C;
    size_t size = A.width * A.height * sizeof(float);
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatAddKernel(Matrix A, Matrix B, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.height && col < A.width)
        SetElement(C, row, col, GetElement(A, row, col) + GetElement(B, row, col));
}

void MatAdd(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A, d_B, d_C;
    size_t size = A.width * A.height * sizeof(float);
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    cudaMalloc(&d_C.elements, size);
    dim3 block(16,16), grid((A.width+15)/16,(A.height+15)/16);
    MatAddKernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatSubKernel(Matrix A, Matrix B, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.height && col < A.width)
        SetElement(C, row, col, GetElement(A, row, col) - GetElement(B, row, col));
}

void MatSub(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A, d_B, d_C;
    size_t size = A.width * A.height * sizeof(float);
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    cudaMalloc(&d_C.elements, size);
    dim3 block(16,16), grid((A.width+15)/16,(A.height+15)/16);
    MatSubKernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatElemMulKernel(Matrix A, Matrix B, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.height && col < A.width)
        SetElement(C, row, col, GetElement(A, row, col) * GetElement(B, row, col));
}

void MatElemMul(const Matrix A, const Matrix B, Matrix C) {
    Matrix d_A, d_B, d_C;
    size_t size = A.width * A.height * sizeof(float);
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    cudaMalloc(&d_C.elements, size);
    dim3 block(16,16), grid((A.width+15)/16,(A.height+15)/16);
    MatElemMulKernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void MatScalarMulKernel(Matrix A, float scalar, Matrix C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.height && col < A.width)
        SetElement(C, row, col, GetElement(A, row, col) * scalar);
}

void MatScalarMul(const Matrix A, float scalar, Matrix C) {
    Matrix d_A, d_C;
    size_t size = A.width * A.height * sizeof(float);
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    cudaMalloc(&d_C.elements, size);
    dim3 block(16,16), grid((A.width+15)/16,(A.height+15)/16);
    MatScalarMulKernel<<<grid, block>>>(d_A, scalar, d_C);
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_C.elements);
}

__global__ void MatTransposeKernel(Matrix A, Matrix At) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A.height && col < A.width)
        SetElement(At, col, row, GetElement(A, row, col));
}

void MatTranspose(const Matrix A, Matrix At) {
    Matrix d_A, d_At;
    size_t size = A.width * A.height * sizeof(float);
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    d_At.width = d_At.stride = At.width; d_At.height = At.height;
    cudaMalloc(&d_At.elements, size);
    dim3 block(16,16), grid((A.width+15)/16,(A.height+15)/16);
    MatTransposeKernel<<<grid, block>>>(d_A, d_At);
    cudaMemcpy(At.elements, d_At.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_At.elements);
}

bool IsIdentity(const Matrix M) {
    for (int i = 0; i < M.height; ++i)
        for (int j = 0; j < M.width; ++j) {
            float v = M.elements[i * M.stride + j];
            if (i == j ? v != 1.f : v != 0.f) return false;
        }
    return true;
}

void MatInverse(const Matrix A, Matrix Ainv) {
    int n = A.width;
    assert(A.width == A.height);
    float* aug = (float*)malloc(sizeof(float) * n * 2 * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            aug[i * (2*n) + j]     = A.elements[i * A.stride + j];
            aug[i * (2*n) + j + n] = (i == j) ? 1.f : 0.f;
        }
    for (int i = 0; i < n; ++i) {
        float diag = aug[i * (2*n) + i];
        for (int j = 0; j < 2*n; ++j) aug[i * (2*n) + j] /= diag;
        for (int k = 0; k < n; ++k) if (k != i) {
            float factor = aug[k * (2*n) + i];
            for (int j = 0; j < 2*n; ++j)
                aug[k * (2*n) + j] -= factor * aug[i * (2*n) + j];
        }
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Ainv.elements[i * Ainv.stride + j] = aug[i * (2*n) + j + n];
    free(aug);
}

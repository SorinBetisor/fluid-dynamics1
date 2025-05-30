/**
 * @file cuda_kernels.cu
 * @brief CUDA kernels for GPU-accelerated fluid dynamics computations
 * 
 * This file contains CUDA kernel implementations for the most computationally
 * intensive operations in the fluid dynamics simulation, providing significant
 * performance improvements over CPU implementations.
 * 
 * Key optimizations implemented:
 * - Shared memory usage for matrix operations
 * - Coalesced memory access patterns
 * - Thread block optimization
 * - Reduction operations for min/max finding
 * 
 * @author Fluid Dynamics Simulation Team
 * @date 2024
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <float.h>
#include "linearalg.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Global CUDA configuration
static int g_cuda_enabled = 0;
static cublasHandle_t cublas_handle = NULL;

/**
 * @brief Initialize CUDA runtime and cuBLAS
 */
void cuda_init() {
    if (g_cuda_enabled) {
        CUDA_CHECK(cudaSetDevice(0));
        cublasCreate(&cublas_handle);
        printf("CUDA initialized with cuBLAS support\n");
    }
}

/**
 * @brief Cleanup CUDA resources
 */
void cuda_cleanup() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = NULL;
    }
    cudaDeviceReset();
}

/**
 * @brief Set CUDA configuration
 * @param enabled 1 to enable CUDA, 0 to disable
 */
void set_cuda_config(int enabled) {
#ifdef CUDA_ENABLED
    g_cuda_enabled = enabled;
    if (enabled && !cublas_handle) {
        cuda_init();
    }
#else
    g_cuda_enabled = 0;
#endif
}

/**
 * @brief Get CUDA enabled status
 * @return 1 if CUDA is enabled, 0 otherwise
 */
int get_cuda_enabled() {
    return g_cuda_enabled;
}

// CUDA Kernels

/**
 * @brief CUDA kernel for matrix zeroing
 */
__global__ void cuda_zeros_kernel(double* A, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < m && idy < n) {
        A[idx * n + idy] = 0.0;
    }
}

/**
 * @brief CUDA kernel for matrix negation
 */
__global__ void cuda_invsig_kernel(double* A, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < m && idy < n) {
        A[idx * n + idy] = -A[idx * n + idy];
    }
}

/**
 * @brief CUDA kernel for matrix copy
 */
__global__ void cuda_copy_kernel(double* A, const double* B, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < m && idy < n) {
        A[idx * n + idy] = B[idx * n + idy];
    }
}

/**
 * @brief CUDA kernel for finding maximum element (reduction)
 */
__global__ void cuda_max_reduction_kernel(const double* A, double* result, int n) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? A[i] : -DBL_MAX;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result back to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief CUDA kernel for finding minimum element (reduction)
 */
__global__ void cuda_min_reduction_kernel(const double* A, double* result, int n) {
    extern __shared__ double sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? A[i] : DBL_MAX;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result back to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief CUDA kernel for Euler time advancement
 */
__global__ void cuda_euler_kernel(double* w, const double* dwdx, const double* dwdy,
                                 const double* d2wdx2, const double* d2wdy2,
                                 const double* u, const double* v,
                                 double Re_inv, double dt, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < m && idy < n) {
        int id = idx * n + idy;
        
        // Vorticity transport: ∂ω/∂t = -(u·∇ω) + (1/Re)∇²ω
        double convection = -(u[id] * dwdx[id] + v[id] * dwdy[id]);
        double diffusion = Re_inv * (d2wdx2[id] + d2wdy2[id]);
        
        w[id] = w[id] + dt * (convection + diffusion);
    }
}

/**
 * @brief CUDA kernel for continuity calculation
 */
__global__ void cuda_continuity_kernel(double* result, const double* dudx, 
                                      const double* dvdy, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < m && idy < n) {
        int id = idx * n + idy;
        result[id] = dudx[id] + dvdy[id];
    }
}

/**
 * @brief CUDA kernel for vorticity calculation
 */
__global__ void cuda_vorticity_kernel(double* result, const double* dvdx, 
                                     const double* dudy, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < m && idy < n) {
        int id = idx * n + idy;
        result[id] = dvdx[id] - dudy[id];
    }
}

/**
 * @brief CUDA kernel for Jacobi iteration (Poisson solver)
 */
__global__ void cuda_jacobi_kernel(double* u_new, const double* u_old, const double* f,
                                  double dx2, double dy2, double dx2_dy2, 
                                  int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < nx - 1 && j < ny - 1) {
        int id = i * ny + j;
        
        double u_east = u_old[(i + 1) * ny + j];
        double u_west = u_old[(i - 1) * ny + j];
        double u_north = u_old[i * ny + (j + 1)];
        double u_south = u_old[i * ny + (j - 1)];
        
        u_new[id] = (dy2 * (u_east + u_west) + dx2 * (u_north + u_south) - 
                     dx2 * dy2 * f[id]) / (2.0 * dx2_dy2);
    }
}

// Host interface functions

/**
 * @brief GPU matrix multiplication using cuBLAS
 */
mtrx cuda_matrix_multiply(mtrx A, mtrx B) {
    if (!g_cuda_enabled || !cublas_handle) {
        printf("Error: CUDA not initialized\n");
        exit(1);
    }
    
    mtrx C = initm(A.m, B.n);
    
    double *d_A, *d_B, *d_C;
    size_t size_A = A.m * A.n * sizeof(double);
    size_t size_B = B.m * B.n * sizeof(double);
    size_t size_C = C.m * C.n * sizeof(double);
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy2D(d_A, A.n * sizeof(double), A.M[0], A.n * sizeof(double), 
                           A.n * sizeof(double), A.m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_B, B.n * sizeof(double), B.M[0], B.n * sizeof(double), 
                           B.n * sizeof(double), B.m, cudaMemcpyHostToDevice));
    
    // Perform matrix multiplication using cuBLAS
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                B.n, A.m, A.n,
                &alpha, d_B, B.n, d_A, A.n,
                &beta, d_C, B.n);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy2D(C.M[0], C.n * sizeof(double), d_C, C.n * sizeof(double), 
                           C.n * sizeof(double), C.m, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}

/**
 * @brief GPU matrix zeroing
 */
void cuda_zeros_matrix(mtrx A) {
    if (!g_cuda_enabled) return;
    
    double *d_A;
    size_t size = A.m * A.n * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((A.m + blockSize.x - 1) / blockSize.x, 
                  (A.n + blockSize.y - 1) / blockSize.y);
    
    cuda_zeros_kernel<<<gridSize, blockSize>>>(d_A, A.m, A.n);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy2D(A.M[0], A.n * sizeof(double), d_A, A.n * sizeof(double), 
                           A.n * sizeof(double), A.m, cudaMemcpyDeviceToHost));
    
    cudaFree(d_A);
}

/**
 * @brief GPU matrix negation
 */
void cuda_invsig_matrix(mtrx A) {
    if (!g_cuda_enabled) return;
    
    double *d_A;
    size_t size = A.m * A.n * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMemcpy2D(d_A, A.n * sizeof(double), A.M[0], A.n * sizeof(double), 
                           A.n * sizeof(double), A.m, cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((A.m + blockSize.x - 1) / blockSize.x, 
                  (A.n + blockSize.y - 1) / blockSize.y);
    
    cuda_invsig_kernel<<<gridSize, blockSize>>>(d_A, A.m, A.n);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy2D(A.M[0], A.n * sizeof(double), d_A, A.n * sizeof(double), 
                           A.n * sizeof(double), A.m, cudaMemcpyDeviceToHost));
    
    cudaFree(d_A);
}

/**
 * @brief GPU matrix copy
 */
void cuda_copy_matrix(mtrx A, mtrx B) {
    if (!g_cuda_enabled) return;
    
    double *d_A, *d_B;
    size_t size = A.m * A.n * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    
    CUDA_CHECK(cudaMemcpy2D(d_B, B.n * sizeof(double), B.M[0], B.n * sizeof(double), 
                           B.n * sizeof(double), B.m, cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((A.m + blockSize.x - 1) / blockSize.x, 
                  (A.n + blockSize.y - 1) / blockSize.y);
    
    cuda_copy_kernel<<<gridSize, blockSize>>>(d_A, d_B, A.m, A.n);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy2D(A.M[0], A.n * sizeof(double), d_A, A.n * sizeof(double), 
                           A.n * sizeof(double), A.m, cudaMemcpyDeviceToHost));
    
    cudaFree(d_A);
    cudaFree(d_B);
}

/**
 * @brief GPU maximum element finding
 */
double cuda_max_element(mtrx A) {
    if (!g_cuda_enabled) return 0.0;
    
    double *d_A, *d_result;
    int n = A.m * A.n;
    size_t size = n * sizeof(double);
    
    // Calculate grid size for reduction
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_result, gridSize * sizeof(double)));
    
    // Copy matrix data in row-major format
    double *flat_A = (double*)malloc(size);
    for (int i = 0; i < A.m; i++) {
        for (int j = 0; j < A.n; j++) {
            flat_A[i * A.n + j] = A.M[i][j];
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, flat_A, size, cudaMemcpyHostToDevice));
    
    // Launch reduction kernel
    cuda_max_reduction_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_A, d_result, n);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy partial results and find final maximum on CPU
    double *partial_results = (double*)malloc(gridSize * sizeof(double));
    CUDA_CHECK(cudaMemcpy(partial_results, d_result, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    double max_val = partial_results[0];
    for (int i = 1; i < gridSize; i++) {
        if (partial_results[i] > max_val) {
            max_val = partial_results[i];
        }
    }
    
    // Cleanup
    free(flat_A);
    free(partial_results);
    cudaFree(d_A);
    cudaFree(d_result);
    
    return max_val;
}

/**
 * @brief GPU minimum element finding
 */
double cuda_min_element(mtrx A) {
    if (!g_cuda_enabled) return 0.0;
    
    double *d_A, *d_result;
    int n = A.m * A.n;
    size_t size = n * sizeof(double);
    
    // Calculate grid size for reduction
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_result, gridSize * sizeof(double)));
    
    // Copy matrix data in row-major format
    double *flat_A = (double*)malloc(size);
    for (int i = 0; i < A.m; i++) {
        for (int j = 0; j < A.n; j++) {
            flat_A[i * A.n + j] = A.M[i][j];
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, flat_A, size, cudaMemcpyHostToDevice));
    
    // Launch reduction kernel
    cuda_min_reduction_kernel<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_A, d_result, n);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy partial results and find final minimum on CPU
    double *partial_results = (double*)malloc(gridSize * sizeof(double));
    CUDA_CHECK(cudaMemcpy(partial_results, d_result, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    double min_val = partial_results[0];
    for (int i = 1; i < gridSize; i++) {
        if (partial_results[i] < min_val) {
            min_val = partial_results[i];
        }
    }
    
    // Cleanup
    free(flat_A);
    free(partial_results);
    cudaFree(d_A);
    cudaFree(d_result);
    
    return min_val;
}

/**
 * @brief GPU Euler time advancement
 */
void cuda_euler_step(mtrx w, mtrx dwdx, mtrx dwdy, mtrx d2wdx2, mtrx d2wdy2, 
                     mtrx u, mtrx v, double Re, double dt) {
    if (!g_cuda_enabled) return;
    
    int n = w.m * w.n;
    size_t size = n * sizeof(double);
    
    double *d_w, *d_dwdx, *d_dwdy, *d_d2wdx2, *d_d2wdy2, *d_u, *d_v;
    
    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_w, size));
    CUDA_CHECK(cudaMalloc(&d_dwdx, size));
    CUDA_CHECK(cudaMalloc(&d_dwdy, size));
    CUDA_CHECK(cudaMalloc(&d_d2wdx2, size));
    CUDA_CHECK(cudaMalloc(&d_d2wdy2, size));
    CUDA_CHECK(cudaMalloc(&d_u, size));
    CUDA_CHECK(cudaMalloc(&d_v, size));
    
    // Copy data to GPU (convert to flat arrays)
    for (int i = 0; i < w.m; i++) {
        CUDA_CHECK(cudaMemcpy(d_w + i * w.n, w.M[i], w.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dwdx + i * w.n, dwdx.M[i], w.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dwdy + i * w.n, dwdy.M[i], w.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_d2wdx2 + i * w.n, d2wdx2.M[i], w.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_d2wdy2 + i * w.n, d2wdy2.M[i], w.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u + i * w.n, u.M[i], w.n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v + i * w.n, v.M[i], w.n * sizeof(double), cudaMemcpyHostToDevice));
    }
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((w.m + blockSize.x - 1) / blockSize.x, 
                  (w.n + blockSize.y - 1) / blockSize.y);
    
    cuda_euler_kernel<<<gridSize, blockSize>>>(d_w, d_dwdx, d_dwdy, d_d2wdx2, d_d2wdy2, 
                                               d_u, d_v, 1.0/Re, dt, w.m, w.n);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    for (int i = 0; i < w.m; i++) {
        CUDA_CHECK(cudaMemcpy(w.M[i], d_w + i * w.n, w.n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Cleanup
    cudaFree(d_w);
    cudaFree(d_dwdx);
    cudaFree(d_dwdy);
    cudaFree(d_d2wdx2);
    cudaFree(d_d2wdy2);
    cudaFree(d_u);
    cudaFree(d_v);
} 
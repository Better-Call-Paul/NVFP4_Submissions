#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using fp4_e2m1 = __nv_fp4_e2m1;
using fp4x2_e2m1 = __nv_fp4x2_e2m1;
using fp8_e4m3 = __nv_fp8_e4m3;
using fp16 = __half;

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

struct gemm_params
{
    int M, N, K, L;
    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ sfa_ptr;
    void *__restrict__ sfb_ptr;
    void *__restrict__ c_ptr;
};

__device__ static __forceinline__ void mma(uint32_t d_tt_addr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc)
{
    
    // non block scaling
    asm volatile(
        "tcgen05.mma.cta_group::1.kindf8f6f4 [%0], [%1], %2, %3\n"
        :: "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
    );

    // block scaling
    // block16, block32
    // scale_vec::1X, scale_vec::2X, scale_vec::4X
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block16{.scale_vec::1X} [%0], [%1], %2, %3\n"
        :: "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idsec), "n"(acc)
    );
}

template<>
__global__ void __launch_bounds__(1, 1)
gemm_kernel(const __grid_constant__ gemm_params params)
{

}

torch::Tensor cuda_nvfp4_gemm(torch::Tensor A,
                              torch::Tensor B,
                              torch::Tensor C,
                              torch::Tensor SFA,
                              torch::Tensor SFB)
{
    const auto a_sizes = A.sizes();
    const auto b_sizes = B.sizes();

    const int N = b_sizes[0];
    const int M = a_sizes[0];
    const int K = a_sizes[1];
    const int L = a_sizes[2];

    gemm_params params{};

    params.M = M;
    params.N = N;
    params.K = K;
    params.L = L;

    params.a_ptr = A.data_ptr();
    params.b_ptr = B.data_ptr();
    pararms.c_ptr = C.data_ptr();
    params.sfa_ptr = SFA.data_ptr();
    params.sfb_ptr = SFB.data_ptr();

    //TODO: set specific dims
    dim3 grid(1, 1);
    dim3 block(1, 1, 1);

    gemm_kernel<><<<grid, block>>>(params);

}

int run_benchmark(size_t M, size_t N, size_t K, size_t L)
{
    cudaError_t cudaStatus;

    float *host_a = new float[M * K];
    float *host_b = new float[K * N];

    float *host_sfa = new float[M * CEIL_DIV(K, 16)];
    float *host_sfb = new float[CEIL_DIV(K, 16) * N];

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i = 0; i < M * K; ++i) host_a[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) host_b[i] = dis(gen);
    for (int i = 0; i < M * CEIL_DIV(K, 16); ++i) host_sfa[i] = dis(gen);
    for (int i = 0; i < CEIL_DIV(K, 16) * N; ++i) host_sfb[i] = dis(gen);

    __nv_fp4x2_e2m1 *host_a_fp4x2 = new __nv_fp4x2_e2m1[M * CEIL_DIV(K, 2)];
    __nv_fp4x2_e2m1 *host_b_fp4x2 = new __nv_fp4x2_e2m1[K * CEIL_DIV(N, 2)];
    __nv_fp8_e4m3 *host_sfa_fp8 = new __nv_fp8_e4m3[M * CEIL_DIV(K, 16)];
    __nv_fp8_e4m3 *host_sfb_fp8 = new __nv_fp8_e4m3[CEIL_DIV(K, 16) * N];

    __half *host_c = new __half[M * N];
    __half *host_c_ref = new __half[M * N];

    // ISSUE:
    for (int i = 0; i + 1 < M * K; i += 2) host_a_fp4x2[CEIL_DIV(i, 2)] = __nv_fp4x2_e2m1(__floats2half2_rn(host_a[i], host_a[i + 1]));
    for (int i = 0; i + 1 < K * N; i += 2) host_b_fp4x2[CEIL_DIV(i, 2)] = __nv_fp4x2_e2m1(__floats2half2_rn(host_b[i], host_b[i + 1]));

    for (int i = 0; i < M * CEIL_DIV(K, 16); ++i) host_sfa_fp8[i] = __nv_fp8_e4m3(host_sfa[i]);
    for (int i = 0; i < CEIL_DIV(K, 16) * N; ++i) host_sfb_fp8[i] = __nv_fp8_e4m3(host_sfb[i]);
    
    __nv_fp4x2_e2m1 *d_a, *d_b;
    __half *d_c, *d_c_ref;
    __nv_fp8_e4m3 *d_sfa, *d_sfb;

    cudaMalloc(&d_a, M * CEIL_DIV(K, 2) * sizeof(__nv_fp4x2_e2m1));
    cudaMalloc(&d_b, K * CEIL_DIV(N, 2) * sizeof(__nv_fp4x2_e2m1));

    cudaMalloc(&d_c, K * N * sizeof(__half));
    cudaMalloc(&d_c_ref, K * N * sizeof(__half));

    cudaMalloc(&d_sfa, M * CEIL_DIV(K, 16) * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_sfb, CEIL_DIV(K, 16) * N * sizeof(__nv_fp8_e4m3));

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << "\n";
        return -1;
    }

    cudaMemcpy(d_a, host_a_fp4x2, M * CEIL_DIV(K, 2) * sizeof(__nv_fp4x2_e2m1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b_fp4x2, K * CEIL_DIV(N, 2) * sizeof(__nv_fp4x2_e2m1), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sfa, host_sfa_fp8, M * CEIL_DIV(K, 16) * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfb, host_sfb_fp8, CEIL_DIV(K, 16) * N * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);

    // kernels

    // cublas check

    // check

    // FLOP CALC

    delete[] host_a;
    delete[] host_a_fp4x2;
    delete[] host_b;
    delete[] host_b_fp4x2;
    delete[] host_c;
    delete[] host_c_ref;
    delete[] host_sfa;
    delete[] host_sfa_fp8;
    delete[] host_sfb;
    delete[] host_sfb_fp8;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sfa);
    cudaFree(d_sfb);
    cudaFree(d_c);
    cudaFree(d_c_ref);

    return 0;
}

int main()
{

    /*
    M N K L time[us]
    128 7168 16384 1 8.994
    128 4096 7168 1 2.354
    128 7168 2048 1 1.333
    */

    //size_t M = 128, N = 7168, K = 16384, L = 1;
    //size_t M = 128, N = 4096, K = 7168, L = 1;
    size_t M = 128, N = 7168, K = 2048, L = 1;

    run_benchmark(M, N, K, L);

    std::cout << "Stable\n";

    return 0;
}
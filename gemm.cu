#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using fp4_e2m1 = __nv_fp4_e2m1;
using fp8_e4m3 = __nv_fp8_e4m3;
using fp16 = __half;

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

struct gemm_params
{
    int M, N, K;
    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ sfa_ptr;
    void *__restrict__ sfb_ptr;
    void *__restrict__ c_ptr;
};
/*
__device__ static __forceinline__ void mma()
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
}*/

template<typename M, typename N, typename K>
__global__ void __launch_bounds__(1, 1) gemm()
{

}

int run_benchmark(size_t M, size_t N, size_t K)
{
    cudaError_t cudaStatus;

    float *generator_a = new float[M * K];
    float *generator_b = new float[K * N];

    float *generator_sfa = new float[M * CEIL_DIV(K, 16)];
    float *generator_sfb = new float[CEIL_DIV(K, 16) * N];

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i = 0; i < M * K; ++i) generator_a[i] = gen(dis);
    for (int i = 0; i < K * N; ++i) generator_b[i] = gen(dis);
    for (int i = 0; i < M * CEIL_DIV(K, 16); ++i) generator_sfa[i] = gen(dis);
    for (int i = 0; i < CEIL_DIV(K, 16) * N; ++i) generator_sfb[i] = gen(dis);

    __nv_fp4_e2m1 *h_a = new __nv_fp4_e2m1[M * K];
    __nv_fp4_e2m1 *h_b = new __nv_fp4_e2m1[K * N];
    __half *h_c = new __half[M * N];
    __half *h_c_ref = new __half[M * N];

    // TODO: check that there is not a critical flaw in just checking via cuBLAS after I converted them

    __nv_fp8_e4m3 *h_sfa = new __nv_fp8_e4m3[M, CEIL_DIV(K, 16)];
    __nv_fp8_e4m3 *h_sfb = new __nv_fp8_e4m3[N, CEIL_DIV(K, 16)];

    for (int i = 0; i < M * K; ++i) h_a[i] = dis(gen);

    // either start with fp16 and then populate -> then convert and store on device
    // or just convert during population?
    // might actually be easier just to do it the first way
    // nvrm just do first way 

    // so only supposed to use __nv_fp4x2_e2m1 

    __nv_fp4_e2m1 *d_a, *d_b;
    __half *d_c, *d_c_ref
    __nv_fp8_e4m3 *d_sfa, *d_sfb;

    cudaMalloc(&d_a, M * K * sizeof(__nv_fp4_e2m1));
    cudaMalloc(&d_b, K * N * sizeof(__nv_fp4_e2m1));

    cudaMalloc(&d_c, K * N * sizeof(__half));
    cudaMalloc(&d_c, K * N * sizeof(__half));

    cudaMalloc(&d_sfa, M * CEIL_DIV(K, 16));
    cudaMalloc(&d_sfb, N * CEIL_DIV(K, 16));




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










    std::cout << "Stable\n";

    return 0;
}
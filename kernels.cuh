#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

/*
 * fp4 (e2m1): a, b
 * fp8 (e4m3nuz): sfa, sfb 
 * fp16: c
*/

// TODO: might need to do some vectorization/packing later so will prob add more dtypes
using __nv_fp4_e2m1 = fp4;
using __nv_fp8_e4m3 = fp8;
using half = fp16;

template<typename D, typename AB, int M, int N, bool trans_a, bool trans_b, bool neg=false>
__device__ static inline uint32_t instruction_descriptor() {
    uint32_t desc = 0;
    if constexpr (sizeof(AB) == 2) { // kind::f16
        // either accumulate to float, or the input is half and the output is half
        static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D matrix is FP32
        }
        else {
            desc |= 0b00  << 4; // D matrix is FP16
        }
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, half>) {
            desc |= 0b000 << 7;  // 16-bit A input type as FP16
            desc |= 0b000 << 10; // 16-bit B input type as FP16
        } else if constexpr (std::is_same_v<AB, bf16>) {
            desc |= 0b001 << 7;  // 16-bit A input type as BF16
            desc |= 0b001 << 10; // 16-bit B input type as BF16
        } else if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
        }
        /* fp6 and fp4
        else if constexpr (std::is_same_v<AB, fp6e2m3>) {
            desc |= 0b011 << 7;  // 6-bit A input type as FP6 e2m3
            desc |= 0b011 << 10; // 6-bit B input type as FP6 e2m3
        }
        else if constexpr (std::is_same_v<AB, fp4e2m3>) {
            desc |= 0b100 << 7;  // 6-bit A input type as FP6 e3m2
            desc |= 0b100 << 10; // 6-bit B input type as FP6 e3m2
        }
        else if constexpr (std::is_same_v<AB, fp4e3m1>) {
            desc |= 0b101 << 7;  // 4-bit A input type as FP4 e3m1
            desc |= 0b101 << 10; // 4-bit B input type as FP4 e3m1
        }
        */
        if constexpr (neg) {
            desc |= 0b1   << 13; // Do negate A matrix
        }
        else {
            desc |= 0b0   << 13; // Don't negate A matrix
        }
        desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
    } else if constexpr (sizeof(AB) == 1) { // kind::f8f6f4
        static_assert(std::is_same_v<D, float> || std::is_same_v<D, half>); // FP8/6/4 has to accumulate to float or half
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D matrix is FP32
        }
        else {
            desc |= 0b00  << 4; // D matrix is FP16
        }
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
        }
        /* fp6 and fp4
        else if constexpr (std::is_same_v<AB, fp6e2m3>) {
            desc |= 0b011 << 7;  // 6-bit A input type as FP6 e2m3
            desc |= 0b011 << 10; // 6-bit B input type as FP6 e2m3
        }
        else if constexpr (std::is_same_v<AB, fp4e2m3>) {
            desc |= 0b100 << 7;  // 6-bit A input type as FP6 e3m2
            desc |= 0b100 << 10; // 6-bit B input type as FP6 e3m2
        }
        else if constexpr (std::is_same_v<AB, fp4e3m1>) {
            desc |= 0b101 << 7;  // 4-bit A input type as FP4 e3m1
            desc |= 0b101 << 10; // 4-bit B input type as FP4 e3m1
        }
        */
        if constexpr (neg) {
            desc |= 0b1   << 13; // Do negate A matrix
        }
        else {
            desc |= 0b0   << 13; // Don't negate A matrix
        }
        desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
    }
    else {
        static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet.");
    }
    return desc;
};

// B shmem, A TMEM, D TMEM
template<int acc, int ncta=1>
__device__ static inline void tensor_shared_mma(uint32_t d_tt_addr, uint32_t a_tt_addr, uint64_t b_desc, uint32_t idesc)
{
    //mxf8f6f4, mxf4nvf4

    // sequence of MMA can reuse A or B, reuse it without multiple reloads, loaded into TensorCore collector buffer

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


/*
 * A (M X K), fp4
 * B (K X 1) fp4
 * sfa (M X (K // 16)) * L, fp16
 * sfb ((K// 16) * 1) * L, fp16
 * C (M X 1), fp16
 * 
*/
__global__ void gemv(fp4* A, fp4* B, fp16* sfa, fp16* sfb, fp16* C);




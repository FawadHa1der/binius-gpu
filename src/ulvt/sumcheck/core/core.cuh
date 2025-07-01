#pragma once
#include <cstdint>

#include "../utils/constants.hpp"

__host__ __device__ void evaluate_composition_on_batch_row(
	const uint32_t* first_batch_of_row,
	uint32_t* batch_composition_destination,
	const uint32_t composition_size,
	const uint32_t original_evals_per_col
);

__host__ __device__ void fold_batch(
	const uint32_t lower_batch[BITS_WIDTH],
	const uint32_t upper_batch[BITS_WIDTH],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const bool is_interpolation
);

// __host__ __device__ void fold_batch_gpu(
// 	const uint32_t lower_batch[BITS_WIDTH],
// 	const uint32_t upper_batch[BITS_WIDTH],
// 	uint32_t dst_batch[BITS_WIDTH],
// 	const uint32_t coefficient[BITS_WIDTH],
// 	const bool is_interpolation);


void fold_small(
	const uint32_t source[BITS_WIDTH],
	uint32_t destination[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const uint32_t list_len
);

__host__ __device__ void compute_sum(
	uint32_t sum[INTS_PER_VALUE],
	uint32_t bitsliced_batch[BITS_WIDTH],
	const uint32_t num_eval_points_being_summed_unpadded
);

//  __device__  __host__  void precompute_A4(const uint32_t a[4], uint32_t out[4]);
typedef struct pre_compute_a4_b4 {
	uint32_t v5ANDv7;  // a0 ^ a2
	uint32_t v21ANDv22;  // a1 ^ a3
	uint32_t v25;
	uint32_t v25XORv26ANDv27;  // a2 ^ a3 ^ (a0 ^ a1)
} pre_compute_a4_b4_t;

__device__ __host__  void precompute_B4(const uint32_t b[4], uint32_t out[4]);
__device__ __host__  pre_compute_a4_b4_t precomputeA4_B4(const uint32_t pre_compute_a[4], const uint32_t pre_compute_b[4]);
__device__ __host__ void fold_batch_gpu(
								const uint32_t lower_batch[BITS_WIDTH],
								const uint32_t *preA_chunks,  // IN
								const uint32_t *xor_chunks,   // IN
								uint32_t *preB,
								const pre_compute_a4_b4 [32],
								uint32_t dst_batch[BITS_WIDTH],
								const uint32_t coefficient[BITS_WIDTH],
                               bool            is_interp);

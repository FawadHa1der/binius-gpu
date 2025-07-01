#include <cstdint>
#include <iostream>

#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../../utils/bitslicing.cuh"
#include "../utils/constants.hpp"
#include "core.cuh"

__host__ __device__ void evaluate_composition_on_batch_row(
	const uint32_t* first_batch_of_row,
	uint32_t* batch_composition_destination,
	const uint32_t composition_size,
	const uint32_t original_evals_per_col
) {
	memcpy(batch_composition_destination, first_batch_of_row, BITS_WIDTH * sizeof(uint32_t));

	for (int operand_in_composition = 1; operand_in_composition < composition_size; ++operand_in_composition) {
		const uint32_t* nth_batch_of_row =
			first_batch_of_row + operand_in_composition * original_evals_per_col * INTS_PER_VALUE;

		multiply_unrolled<TOWER_HEIGHT>(batch_composition_destination, nth_batch_of_row, batch_composition_destination);
	}
}


/*------------------------------------------------------------------*/
/*  b: 4-word bit-sliced element                                    */
/*  out[4] contains {v16, v18, v22, v7}                              */
/*------------------------------------------------------------------*/
 __device__ __host__  void precompute_B4(const uint32_t b[4], uint32_t out[4])
{
    out[0] = b[0] ^ b[2];   /* v16 */
    out[1] = b[1] ^ b[3];   /* v18 */
    out[2] = b[0] ^ b[1];   /* v22 */
    out[3] = b[2] ^ b[3];   /* v7  */
}



__device__ __host__  pre_compute_a4_b4_t precomputeA4_B4(const uint32_t pre_compute_a[4], const uint32_t pre_compute_b[4])
{
	pre_compute_a4_b4_t ret;
    ret.v5ANDv7 = pre_compute_a[3] & pre_compute_b[3];   /* v16 */
	ret.v21ANDv22 = pre_compute_a[2] & pre_compute_b[2]; /* v18 */
	ret.v25 = pre_compute_a[1] & pre_compute_b[1];       /* v22 */
	ret.v25XORv26ANDv27 = ret.v25 ^ (pre_compute_a[0] ^ pre_compute_a[1]) & (pre_compute_b[0] ^ pre_compute_b[1]); /* v7  */

    // ret[1] = b[0] ^ b[1];   /* v18 */
    // ret[2] = b[0] ^ b[1];   /* v22 */
    // ret[3] = b[2] ^ b[3];   /* v7  */
}


/**********************************************************************
* multiply_unrolled2_fast : same result as multiply_unrolled<2>       *
*                                                                 ... *
**********************************************************************/
	__device__ __host__ __forceinline__
	void multiply_unrolled2_fast(const uint32_t  a[4],   /* bit-slice */
								const uint32_t  preA[4],/* {v12,v14,v21,v5} */
								const uint32_t  b[4],
								const uint32_t  preB[4],/* {v16,v18,v22,v7} */
								const pre_compute_a4_b4_t pre_compute_a4_b4,
								uint32_t        dst[4])
	{
		uint32_t v1  = a[3] & b[3];
		uint32_t v4  = v1;
		uint32_t v5  = preA[3];            // a2 ^ a3
		uint32_t v7  = preB[3];            // b2 ^ b3
		v1         ^= a[2] & b[2];
		uint32_t v9  = v1;
		v4         ^= v1 ^ (pre_compute_a4_b4.v5ANDv7);
		uint32_t v10 = v4;
		uint32_t v11 = v9 ^ v4;

		uint32_t v12 = preA[0];            // a0 ^ a2
		uint32_t v14 = preA[1];            // a1 ^ a3
		uint32_t v16 = preB[0];            // b0 ^ b2
		uint32_t v18 = preB[1];            // b1 ^ b3

		uint32_t v20 = a[1] & b[1];
		v4          ^= v20;
		uint32_t v21 = preA[2];            // a0 ^ a1
		uint32_t v22 = preB[2];            // b0 ^ b1
		v20         ^= a[0] & b[0];
		v9          ^= v20;
		v4          ^= v20 ^ (pre_compute_a4_b4.v21ANDv22);
		uint32_t v23 = v9;
		uint32_t v24 = v4;
		v10         ^= v9;
		v11         ^= v4;

		uint32_t v25 = pre_compute_a4_b4.v25;
		v11         ^= v25;
		uint32_t v26 = v12 ^ v14;
		uint32_t v27 = v16 ^ v18;
		v25         ^= v12 & v16;
		v10         ^= v25;
		v11         ^= pre_compute_a4_b4.v25XORv26ANDv27;

		dst[0] = v23;
		dst[1] = v24;
		dst[2] = v10;
		dst[3] = v11;
	}

/*  xor_chunks   : 32×4 words  (flattened)  – the 32 four-plane slices
 *  preA_chunks  : 32×4 words               – {v12,v14,v21,v5} for each chunk */
__device__ __host__ void fold_batch_gpu(
								const uint32_t lower_batch[BITS_WIDTH],
								const uint32_t *preA_chunks,  // IN
								const uint32_t *xor_chunks,   // IN
								 uint32_t *preB,
								 const pre_compute_a4_b4 preAB_row[32],
							uint32_t dst_batch[BITS_WIDTH],
							const uint32_t coefficient[BITS_WIDTH],
                               bool            is_interp)
{
	uint32_t product[BITS_WIDTH];
	memset(product, 0, BITS_WIDTH * sizeof(uint32_t));

    /* ---- iterate over 32 chunks --------------------------------- */
    #pragma unroll
	for (int i = 0; i < BITS_WIDTH; i += INTERPOLATION_BITS_WIDTH) {
        const uint32_t *a     = xor_chunks + i;
        const uint32_t *preA  = preA_chunks + i;
        // uint32_t        prod4[4];
		const pre_compute_a4_b4 ab = preAB_row[i/4];
        multiply_unrolled2_fast(a, preA, coefficient, preB, ab, product + i);
    }
	
	for (int i = 0; i < BITS_WIDTH; ++i) {
		dst_batch[i] = lower_batch[i] ^ product[i];
	}
}

__host__ __device__ void fold_batch(
	const uint32_t lower_batch[BITS_WIDTH],
	const uint32_t upper_batch[BITS_WIDTH],
	uint32_t dst_batch[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const bool is_interpolation
) {
	uint32_t xor_of_halves[BITS_WIDTH];

	for (int i = 0; i < BITS_WIDTH; ++i) {
		xor_of_halves[i] = lower_batch[i] ^ upper_batch[i];
	}

	uint32_t product[BITS_WIDTH];
	memset(product, 0, BITS_WIDTH * sizeof(uint32_t));

	// Multiply chunk-wise based on field height of coefficient
	// For random challenges this will be the full 7
	// For interpolation points this will be no more than 2

	if (is_interpolation) {
		for (int i = 0; i < BITS_WIDTH; i += INTERPOLATION_BITS_WIDTH) {
			multiply_unrolled<INTERPOLATION_TOWER_HEIGHT>(xor_of_halves + i, coefficient, product + i);
		}
	} else {
		multiply_unrolled<TOWER_HEIGHT>(xor_of_halves, coefficient, product);
	}

	for (int i = 0; i < BITS_WIDTH; ++i) {
		dst_batch[i] = lower_batch[i] ^ product[i];
	}
}

void fold_small(
	const uint32_t source[BITS_WIDTH],
	uint32_t destination[BITS_WIDTH],
	const uint32_t coefficient[BITS_WIDTH],
	const uint32_t list_len
) {
	uint32_t half_len = list_len / 2;

	uint32_t batch_to_be_multiplied[BITS_WIDTH];

	memcpy(batch_to_be_multiplied, source, BITS_WIDTH * sizeof(uint32_t));

	for (int i = 0; i < BITS_WIDTH; ++i) {
		batch_to_be_multiplied[i] >>= half_len;  // Move the upper half into the lower half of this operand
		batch_to_be_multiplied[i] ^= source[i];  // Add two halves before multiplying
	}

	uint32_t product[BITS_WIDTH];

	multiply_unrolled<TOWER_HEIGHT>(batch_to_be_multiplied, coefficient, product);

	for (int i = 0; i < BITS_WIDTH; ++i) {
		destination[i] = source[i] ^ product[i];
	}
}

__host__ __device__ void compute_sum(
	uint32_t sum[INTS_PER_VALUE],
	uint32_t bitsliced_batch[BITS_WIDTH],
	const uint32_t num_eval_points_being_summed_unpadded
) {
	BitsliceUtils<BITS_WIDTH>::bitslice_untranspose(bitsliced_batch);

	memset(sum, 0, INTS_PER_VALUE * sizeof(uint32_t));

	for (uint32_t i = 0; i < min(BITS_WIDTH, INTS_PER_VALUE * num_eval_points_being_summed_unpadded); ++i) {
		sum[i % INTS_PER_VALUE] ^= bitsliced_batch[i];
	}
}
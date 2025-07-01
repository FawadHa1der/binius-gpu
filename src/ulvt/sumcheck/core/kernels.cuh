#include <cstdint>
#include "core.cuh"
#include "../utils/constants.hpp"

 __device__  __host__ __forceinline__ void precompute_A4(const uint32_t a[4], uint32_t out[4])
{
    out[0] = a[0] ^ a[2];   /* v12 */
    out[1] = a[1] ^ a[3];   /* v14 */
    out[2] = a[0] ^ a[1];   /* v21 */
    out[3] = a[2] ^ a[3];   /* v5  */
}


template <uint32_t INTERPOLATION_POINTS, uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
__global__ void compute_compositions(
	const uint32_t* multilinear_evaluations,
	uint32_t* multilinear_products_sums,
	uint32_t* folded_products_sums,
	const uint32_t coefficients[INTERPOLATION_POINTS * BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t active_threads,
	const uint32_t active_threads_folded
) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // start the batch index off at the tid

	uint32_t folded_products_sums_this_thread[INTERPOLATION_POINTS * BITS_WIDTH];

	uint32_t multilinear_products_sums_this_thread[BITS_WIDTH];

	memset(folded_products_sums_this_thread, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));

	memset(multilinear_products_sums_this_thread, 0, BITS_WIDTH * sizeof(uint32_t));

	for (uint32_t row_idx = tid; row_idx < num_batch_rows; row_idx += gridDim.x * blockDim.x) {
		uint32_t this_multilinear_product[BITS_WIDTH];

		evaluate_composition_on_batch_row(
			multilinear_evaluations + BITS_WIDTH * row_idx,
			this_multilinear_product,
			COMPOSITION_SIZE,
			EVALS_PER_MULTILINEAR
		);

		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			multilinear_products_sums_this_thread[i] ^= this_multilinear_product[i];
		}

		uint32_t num_batch_rows_to_fold = num_batch_rows / 2;

		if (row_idx < num_batch_rows_to_fold) {
			// Fold each batch in the batch row
			uint32_t folded_batch_row[INTERPOLATION_POINTS * COMPOSITION_SIZE * BITS_WIDTH];

			// Fold this batch with the corresponding one
			for (int column_idx = 0; column_idx < COMPOSITION_SIZE; ++column_idx) {
				uint32_t batches_fitting_into_original_column = EVALS_PER_MULTILINEAR / 32;
				const uint32_t* lower_batch =
					multilinear_evaluations +
					BITS_WIDTH * (batches_fitting_into_original_column * column_idx + row_idx);
				const uint32_t* upper_batch = lower_batch + BITS_WIDTH * num_batch_rows_to_fold;
								/* -------- inside the `column_idx` loop, BEFORE ip loop ----------- */
				uint32_t xor_chunks[128];            // holds 32×4 planes
				uint32_t preA_chunks[128];           // same layout, pre-computed A terms

				for (int off = 0; off < 128; off += 4) {
					/* xor once */
					xor_chunks[off+0] = lower_batch[off+0] ^ upper_batch[off+0];
					xor_chunks[off+1] = lower_batch[off+1] ^ upper_batch[off+1];
					xor_chunks[off+2] = lower_batch[off+2] ^ upper_batch[off+2];
					xor_chunks[off+3] = lower_batch[off+3] ^ upper_batch[off+3];

					/* pre-compute A terms once */
					precompute_A4(&xor_chunks[off], &preA_chunks[off]);
				}

				uint32_t preB[4*INTERPOLATION_POINTS];
				pre_compute_a4_b4_t pre_compute_a4_b4[INTERPOLATION_POINTS][32];

				for (int ip = 0; ip < INTERPOLATION_POINTS; ++ip) {
					// per interpolation point calculate pre_b
					precompute_B4(coefficients + ip * BITS_WIDTH, &preB[ip * 4]);
				}

				for (int off = 0; off < 128; off += 4) {
					pre_compute_a4_b4[ip][off/4] = precomputeA4_B4(
						&preA_chunks[off],
						&preB[ip * 4]
					);
				}

				/* -------- interpolation-point loop ------------------------------- */
				for (int ip = 0; ip < INTERPOLATION_POINTS; ++ip) {

					fold_batch_gpu(
						lower_batch,
						preA_chunks,                        // 32×4 preA values
						xor_chunks,    //a                     // 32×4 planes
						&preB[ip * 4],
						pre_compute_a4_b4[ip], // 32×4 pre-computed A terms
						folded_batch_row +
							BITS_WIDTH * (column_idx * INTERPOLATION_POINTS + ip),
						coefficients + ip * BITS_WIDTH,     // coeff slice
						/* is_interpolation = */ true);
				}
			}

			// Take the folded batches and evaluate the compositions on them

			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				uint32_t this_interpolation_point_product_batch[BITS_WIDTH];
				evaluate_composition_on_batch_row(
					folded_batch_row + BITS_WIDTH * interpolation_point,
					this_interpolation_point_product_batch,
					COMPOSITION_SIZE,
					INTERPOLATION_POINTS * 32
				);

				// Add this product to the sum of all products taken by the thread
				uint32_t* this_interpolation_point_sum_location =
					folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

				for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
					this_interpolation_point_sum_location[i] ^= this_interpolation_point_product_batch[i];
				}
			}
		}
	}

	if (tid < active_threads) {
		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			atomicXor(multilinear_products_sums + i, multilinear_products_sums_this_thread[i]);
		}
	}

	if (tid < active_threads_folded) {
		for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
			uint32_t* batch_to_copy_to = folded_products_sums + BITS_WIDTH * interpolation_point;
			uint32_t* batch_to_copy_from = folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

			for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
				atomicXor(batch_to_copy_to + i, batch_to_copy_from[i]);
			}
		}
	}
}

__global__ void fold_large_list_halves(
	uint32_t* source,
	uint32_t* destination,
	uint32_t coefficient[BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t src_evals_per_column,
	const uint32_t dst_evals_per_column,
	const uint32_t num_cols
);
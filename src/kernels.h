#ifndef SUM_VECTOR_FUNC_H
#define SUM_VECTOR_FUNC_H

/**
*   function: sum_vectors
*   ---------------------
*   Sum two vectors of integers and store the result in a third vector
*
*   @param[in] vec1 first vector
*   @param[in] vec2 second vector (store the inverse of the first vector)
*   @param[out] output_vec vector that store the result of the sum
*   @param[in] size size of the vectors
*
*   @return void
*/
__global__ void sum_vectors (int *vec1, int *vec2, int *output_vec, int size);

/**
*   function: matrix_generator
*   --------------------------
*   Generate one matrix from other matrix keeping the same values in the even columns
*   and zeros in the odd columns
*
*   @param[in] mtxA matrix to be copied
*   @param[out] mtxB matrix that store the result of the copy
*
*   @return void
*/
__global__ void matrix_generator (int *mtxA, int *mtxB);

/**
*   function: reduction
*   -------------------
*   Sum all the elements of a vector
*
*   @param[in] arr vector to be summed
*   @param[out] output pointer that store the result of the sum
*
*   @return void
*/
__global__ void parallel_reduction (float *vec, float *output);

#endif

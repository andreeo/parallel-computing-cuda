#ifndef SUM_VECTOR_FUNC_H
#define SUM_VECTOR_FUNC_H

/**
 *  function: sum_vectors
 *  ---------------------
 *  Sum two vectors of integers and store the result in a third vector
 *
 *  @param[in] vec1 first vector
 *  @param[in] vec2 second vector (store the inverse of the first vector)
 *  @param[out] output_vec vector that store the result of the sum
 *  @param[in] size size of the vectors
 *
 *  @return void
 */
__global__ void sum_vectors (int *vec1, int *vec2, int *output_vec, int size);

/**
 *  function: matrix_generator
 *  --------------------------
 *  Generate one matrix from other matrix keeping the same values in the even columns
 *  and zeros in the odd columns
 *
 *  @param[in] mtxA matrix to be copied
 *  @param[out] mtxB matrix that store the result of the copy
 *
 *  @return void
 */
__global__ void matrix_generator (int *mtxA, int *mtxB);

/**
 *  function: reduction
 *  -------------------
 *  Sum all the elements of a vector
 *
 *  @param[in] arr vector to be summed
 *  @param[out] output pointer that store the result of the sum
 *
 *  @return void
 */
__global__ void parallel_reduction (float *vec, float *output);


/**
 *  function: displacer_matrix
 *  --------------------------
 *  displacer one position to down in all rows of a matrix
 * 
 *  @param[in] mtx matrix to be displaced
 *  @param[out] mtx_output matrix that store the result of the displacement
 *  @param[in] MTX_COL_SIZE number of columns of the matrix
 *  @param[in] MTX_ROW_SIZE number of rows of the matrix
 *  
 *  @return void
 */
__global__ void displacer_matrix (int *mtx, int *mtx_output, int MTX_COL_SIZE,
                                  int MTX_ROW_SIZE);


/**
 *  function: rank_sort_algorithm
 *  ---------------------
 *  Sort a vector of integers using the rank sort algorithm
 * 
 *  @param[in] vec vector to be sorted
 *  @param[out] vec_output vector that store the result of the sort
 *  @param[in] size size of the vector
 *  
 *  @return void
 */
__global__ void rank_sort_algorithm (int *vec, int *vec_output, int size);

#endif

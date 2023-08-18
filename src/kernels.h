#ifndef SUM_VECTOR_FUNC_H
#define SUM_VECTOR_FUNC_H

/*
*   function: sum_vectors
*   ---------------------
*   Sum two vectors of integers and store the result in a third vector
*
*   parameters:
*       vec1: first vector
*       vec2: second vector (store the inverse of the first vector)
*       output_vec: vector that store the result of the sum
*       size: size of the vectors
*
*   returns: void
*/
__global__ void sum_vectors (int *vec1, int *vec2, int *output_vec, int size);

/*
*   function: matrix_generator
*   --------------------------
*   Generate one matrix from other matrix keeping the same values in the even columns
*   and zeros in the odd columns
*
*   parameters:
*       mtxA: matrix to be copied
*       mtxB: matrix that store the result of the copy
*
*   returns: void
*/
__global__ void matrix_generator (int *mtxA, int *mtxB);

#endif

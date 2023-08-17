#include "kernels.h"

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
__global__ void
sum_vectors (int *vec1, int *vec2, int *output_vec, int size)
{
  // compute the global thread index of univoque way
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  // check if the thread is in the range of the vector
  if (threadId < size)
  {
    // generate second vector of way inverse
    vec2[threadId] = vec1[size - 1 - threadId];

    // compute the addition of the two vectors
    output_vec[threadId] = vec1[threadId] + vec2[threadId];
  }
}

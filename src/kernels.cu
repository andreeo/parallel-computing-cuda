#include "kernels.h"

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

__global__ void
matrix_generator (int *mtxA, int *mtxB)
{
  // index - col and row
  int col = threadIdx.x;
  int row = threadIdx.y;

  // compute the global thread index of univoque way using lineal index
  int idx = col + row * blockDim.x;

  // check if the column is even or odd
  if (col % 2 != 0)
    mtxB[idx] = 0;
}

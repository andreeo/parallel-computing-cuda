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

__global__ void
parallel_reduction (float *vec, float *output)
{

  int block_dim, threadId, calc, jump;

  // set the block dimension
  block_dim = blockDim.x;

  // set the thread index
  threadId = threadIdx.x;

  // compute the value
  calc = (threadId + 1) * (threadId + 1);       // which means pow(threadId + 1, 2)

  // set data
  vec[threadId] = (float) 1 / calc;

  // sync all threads
  __syncthreads ();

  /*
   * parallel reduction algorithm
   *
   * explination:
   * implement the algorithm where the length of N is a power of 2
   * so, the algorithm will be executed in log2(N) steps
   * instead of N steps in the serial version.
   * in each step each thread will add the value corresponding to its
   * position and its position plus the jump (half)
   * storing the partial result in its respective position.
   */

  // compute the jump
  jump = block_dim / 2;

  // do reduction
  while (jump > 0)
  {
    if (threadId < jump)
    {
      // add the value of the position and the position plus the jump
      vec[threadId] = vec[threadId] + vec[threadId + jump];
    }
    __syncthreads ();

    // update the jump
    jump /= 2;
  }

  // set the output
  if (threadId == 0)
  {
    *output = vec[0];
  }

}

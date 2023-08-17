#include <iostream>
#include <cuda_runtime.h>
#include "../cuda_device.h"
#include "common.h"
#include "kernels.h"

#define RANDOM_MAX 10
#define THREADS_PER_BLOCK 10

using namespace std;

int
main ()
{

  // declarate the host and device vectors
  int
    *hst_vec1, *hst_vec2, *hst_vec_output,
    *dev_vec1, *dev_vec2, *dev_vec_output;

  // declarate the size of the vectors
  int vecSize;

  // declare the quantity of blocks
  int blocks;

  // declare the counter for the for loop
  int i;

  // ask the user for the size of the vectors
  cout << "Enter the size of the vectors: ";
  cin >> vecSize;

  // allocate and initilizate the host and device vectors

  hst_vec1 = (int *) calloc (vecSize, sizeof (int));
  hst_vec2 = (int *) calloc (vecSize, sizeof (int));
  hst_vec_output = (int *) calloc (vecSize, sizeof (int));

  cudaMalloc ((void **) &dev_vec1, vecSize * sizeof (int));
  cudaMalloc ((void **) &dev_vec2, vecSize * sizeof (int));
  cudaMalloc ((void **) &dev_vec_output, vecSize * sizeof (int));

  cudaMemset ((void *) &dev_vec1, 0, vecSize * sizeof (int));
  cudaMemset ((void *) &dev_vec2, 0, vecSize * sizeof (int));
  cudaMemset ((void *) &dev_vec_output, 0, vecSize * sizeof (int));

  // seed the random number generator to get different results each time
  // and fill the host vector (1) with random numbers
  srand (time (NULL));
  for (i = 0; i < vecSize; i++)
  {
    hst_vec1[i] = rand () % RANDOM_MAX;
  }

  // compute the quantity of blocks

  // if the size of the vector is multiple of the threads per block
  blocks = vecSize / THREADS_PER_BLOCK;

  // if not add one more block
  if (vecSize % THREADS_PER_BLOCK != 0)
  {
    blocks++;
  }

  // transfer data from host (vector 1) to device (vector 1)
  cudaMemcpy (dev_vec1, hst_vec1, vecSize * sizeof (int),
              cudaMemcpyHostToDevice);
  // call  the kernel
  sum_vectors <<< blocks, THREADS_PER_BLOCK >>> (dev_vec1, dev_vec2,
                                                 dev_vec_output, vecSize);
  // transfer data from device (vector output) to host (vector output)
  cudaMemcpy (hst_vec2, dev_vec2, vecSize * sizeof (int),
              cudaMemcpyDeviceToHost);
  cudaMemcpy (hst_vec_output, dev_vec_output, vecSize * sizeof (int),
              cudaMemcpyDeviceToHost);

  // print the results
  print_int_vector ((char *) "Vector 1", hst_vec1, vecSize);
  print_int_vector ((char *) "Vector 2", hst_vec2, vecSize);
  print_int_vector ((char *) "Output Vector", hst_vec_output, vecSize);
}

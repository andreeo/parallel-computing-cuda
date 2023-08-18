#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common.h"
#include "kernels.h"

#define MTX_COL_SIZE 21
#define MTX_ROW_SIZE 6

/* 
    * This program will take in a matrix of size COL x ROW initializated with random values
    * and then generate a new matrix in kernel with the same size.
    * The values of odd columns will be the same and even columns will be zero.
*/

int
main (int argc, char **argv)
{
  // declarate the host and device matrix
  int *hst_mtxA, *hst_mtxR, *dev_mtxB;

  // declare the counter for the loops
  int i;

  // declare and initialize size of the matrix
  int mtxSize = MTX_COL_SIZE * MTX_ROW_SIZE;

  // allocate and initialize the host and device matrix
  hst_mtxA = (int *) calloc (mtxSize, sizeof (int));
  hst_mtxR = (int *) calloc (mtxSize, sizeof (int));

  cudaMalloc ((void **) &dev_mtxB, mtxSize * sizeof (int));

  cudaMemset (dev_mtxB, 0, mtxSize * sizeof (int));

  //seed the random number generator to get different results each time
  srand (time (NULL));
  for (i = 0; i < mtxSize; i++)
  {
    hst_mtxA[i] = rand () % 10;
  }

  //transfer data from host to device
  cudaMemcpy (dev_mtxB, hst_mtxA, mtxSize * sizeof (int),
              cudaMemcpyHostToDevice);

  //declare the block and grid size
  dim3 block (1);
  dim3 threads (MTX_COL_SIZE, MTX_ROW_SIZE);

  //call the kernel
  matrix_generator <<< block, threads >>> (hst_mtxA, dev_mtxB);

  //transfer data from device to host
  cudaMemcpy (hst_mtxR, dev_mtxB, mtxSize * sizeof (int),
              cudaMemcpyDeviceToHost);

  //print the results
  print_int_matrix ((char *) "Original Matrix", hst_mtxA, mtxSize,
                    MTX_COL_SIZE);
  print_int_matrix ((char *) "Output Matrix", hst_mtxR, mtxSize,
                    MTX_COL_SIZE);

  return 0;
}

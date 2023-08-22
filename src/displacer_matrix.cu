#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "common.h"
#include "kernels.h"
#include "../cuda_device.h"

#define MTX_COL_SIZE 25
#define MTX_ROW_SIZE 7
#define BLOCKS 1

using namespace std;

int
main (int argc, char **argv)
{
  // declarate the host and device matrix
  int *hst_mtxA, *hst_mtx_output, *dev_mtxB, *dev_mtx_output;

  // declarate variable to store the random number
  int randNumber;

  // declarate the counter variables for the loops
  int i, j;

  // declare the accumulator variable to index the matrix
  int acc;

  // declare the size variable of the matrix
  int mtxSize = MTX_COL_SIZE * MTX_ROW_SIZE;

  // declare events for timing
  cudaEvent_t start, stop;

  // declare the elapsed time variable
  float elapsedTime;

  // create event objects
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  // allocate memory for the host
  hst_mtxA = (int *) malloc (mtxSize * sizeof (int));
  hst_mtx_output = (int *) malloc (mtxSize * sizeof (int));

  // start the timer
  cudaEventRecord (start, 0);

  // allocate memory for the device
  cudaMalloc ((void **) &dev_mtxB, mtxSize * sizeof (int));
  cudaMalloc ((void **) &dev_mtx_output, mtxSize * sizeof (int));

  // seed the random number generator
  srand (time (NULL));

  // initialize the host matrix
  acc = 0;
  for (i = 0; i < MTX_ROW_SIZE; i++)
  {
    randNumber = rand () % RANDOM_MAX;
    for (j = 0; j < MTX_COL_SIZE && acc < mtxSize; j++)
    {
      hst_mtxA[acc] = randNumber;
      acc++;
    }
  }

  // transfer data from host to device
  cudaMemcpy (dev_mtxB, hst_mtxA, mtxSize * sizeof (int),
              cudaMemcpyHostToDevice);

  // config kernel
  dim3 blocks (BLOCKS);
  dim3 threads (MTX_COL_SIZE, MTX_ROW_SIZE);

  // launch the kernel
  displacer_matrix <<< blocks, threads >>> (dev_mtxB, dev_mtx_output,
                                            MTX_COL_SIZE, MTX_ROW_SIZE);

  // wait for compute device to finish
  cudaDeviceSynchronize ();

  // stop the timer
  cudaEventRecord (stop, 0);

  // syncronize the CPU-GPU
  cudaEventSynchronize (stop);

  // calculate the elapsed time (ms)
  cudaEventElapsedTime (&elapsedTime, start, stop);

  // transfer data from device to host
  cudaMemcpy (hst_mtx_output, dev_mtx_output, mtxSize * sizeof (int),
              cudaMemcpyDeviceToHost);

  // print the device props
  deviceProperties (DEVICE_SELECTED);

  cout << endl;
  cout << "> Execution time: " << elapsedTime << " ms" << endl;

  // destroy the event objects
  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  // print the matrix
  print_int_matrix ((char *) "> Original matrix", hst_mtxA, mtxSize,
                    MTX_COL_SIZE);
  print_int_matrix ((char *) "> Displaced matrix", hst_mtx_output, mtxSize,
                    MTX_COL_SIZE);

  // free the memory
  free (hst_mtxA);
  free (hst_mtx_output);
  cudaFree (dev_mtxB);
  cudaFree (dev_mtx_output);

  return 0;
}

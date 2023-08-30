#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>
#include "common.h"
#include "../cuda_device.h"
#include "kernels.h"

#undef RANDOM_MAX
#define RANDOM_MAX 50
#define BLOCK 1

using namespace std;

int
main (int argc, char **argv)
{
  // declarate the host and device vectors
  int *hst_vecA, *hst_vecR, *dev_vecB, *dev_vecR;

  // declarate the device properties
  struct cudaDeviceProp deviceProps;

  // declarate variable for the size of the vector input by user
  int size;

  // declarate events for time measure
  cudaEvent_t start, stop;

  // declarate the elapsed time variable
  float elapsedTime;

  // declarate the counter variable for loop
  int i;

  // create the objects for time measure
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  // get the device properties
  deviceProps = deviceProperties (DEVICE_SELECTED);

  // ask the user for the size of the vector
  cout << "Enter the size of the vector to sort: ";
  cin >> size;

  // check if the size is valid and does not exceed the limit of threads
  while (size > deviceProps.maxThreadsPerBlock)
  {
    cout <<
      "> ERROR: the maximum number of threads per block has been exceeded ("
      << deviceProps.maxThreadsPerBlock << " threads)" << endl;
    cout << "Enter the size of the vector to sort: ";
    cin >> size;
  }

  // allocate memory for the host vectors
  hst_vecA = (int *) calloc (size, sizeof (int));
  hst_vecR = (int *) calloc (size, sizeof (int));

  // allocate memory for the device vectors
  cudaMalloc ((void **) &dev_vecB, size * sizeof (int));
  cudaMalloc ((void **) &dev_vecR, size * sizeof (int));

  // seed the random number generator
  srand (time (NULL));

  for (i = 0; i < size; i++)
  {
    hst_vecA[i] = rand () % RANDOM_MAX;
  }

  // transfer data from host to device
  cudaMemcpy (dev_vecB, hst_vecA, size * sizeof (int),
              cudaMemcpyHostToDevice);

  // config kernel
  dim3 block (BLOCK);
  dim3 threadsPerBlock (size);

  // start the timer
  cudaEventRecord (start, 0);

  // call the kernel
  rank_sort_algorithm <<< block, threadsPerBlock >>> (dev_vecB, dev_vecR,
                                                      size);

  // wait for compute device to finish
  cudaDeviceSynchronize ();

  // stop the timer
  cudaEventRecord (stop, 0);

  // synchronize the GPU-CPU
  cudaEventSynchronize (stop);

  // calculate the elapsed time
  cudaEventElapsedTime (&elapsedTime, start, stop);

  // transfer data from device to host
  cudaMemcpy (hst_vecR, dev_vecR, size * sizeof (int),
              cudaMemcpyDeviceToHost);

  cout << endl;
  cout << "Execution time: " << elapsedTime << " ms" << endl;

  // destroy the event objects
  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  print_int_vector ((char *) "Original vector", hst_vecA, size);
  print_int_vector ((char *) "Sorted vector", hst_vecR, size);

  // free host memory
  free (hst_vecA);
  free (hst_vecR);

  cout << "Press [INTRO] to finish...";
  cin.get ();

  return 0;
}

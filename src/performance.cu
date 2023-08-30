#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include "common.h"
#include "../cuda_device.h"
#include "kernels.h"
#include "host.h"

#undef RANDOM_MAX
#define RANDOM_MAX 50

using namespace std;

int
main (int argc, char **argv)
{
  // declarate the host and device vectors
  int *hst_vecA, *hst_vecR, *hst_vecR_other, *dev_vecB, *dev_vecR;

  // declarate the device properties
  struct cudaDeviceProp deviceProps;

  // declarate variable for the size of the vector input by user
  int size;

  // declarate events for the time measure
  cudaEvent_t start, stop;

  //  declarate variable for the elapsed time in gpu
  float elapsedTimeGPU;

  // declarate variable for the time measure in cpu
  event start_cpu, stop_cpu;

  // declarate variable for the time measure in cpu
  double elapsedTimeCPU;

  // declarate variable for the gain
  float gain;


  // declarate variable for the choice of the user to print the vectors
  int print;

  // declarate the counter variable for loop
  int i;

  // declarate the variable for the number of blocks and threads
  int numBlocks, numThreads;

  // create the objets for the time measure
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  // get the device properties
  deviceProps = deviceProperties (DEVICE_SELECTED);

  // ask the user for the size of the vector
  cout << "Enter the size of the vector to sort: ";
  cin >> size;

  // ask the user if he wants to print the vectors
  cout << "Do you want to print the vectors? (1 = yes, 0 = no): ";
  cin >> print;

  // check if the print is valid (0 or 1)
  while (print != 0 && print != 1)
  {
    cout << "> ERROR: invalid option" << endl;

    cout << "Do you want to print the vectors? (1 = yes, 0 = no): ";
    cin >> print;
  }

  // allocate and initialize the host vectors
  hst_vecA = (int *) calloc (size, sizeof (int));
  hst_vecR = (int *) calloc (size, sizeof (int));
  hst_vecR_other = (int *) calloc (size, sizeof (int));

  // allocate the device vectors
  cudaMalloc ((void **) &dev_vecB, size * sizeof (int));
  cudaMalloc ((void **) &dev_vecR, size * sizeof (int));

  // seed the random number generator
  srand (time (NULL));

  // set the host vector with random numbers
  for (i = 0; i < size; i++)
  {
    hst_vecA[i] = rand () % RANDOM_MAX;
  }

  // start event
  setEvent (&start_cpu);
  // sort the vector
  rank_sort_algorithm_cpu (hst_vecA, hst_vecR, size);
  // stop event
  setEvent (&stop_cpu);

  // transfer data from host to device
  cudaMemcpy (dev_vecB, hst_vecA, size * sizeof (int),
              cudaMemcpyHostToDevice);

  // compute the threads and blocks
  if (size < deviceProps.maxThreadsPerBlock)
  {
    numThreads = size;
    numBlocks = 1;
  }
  else
  {
    numBlocks =
      ceil ((double) size / (double) deviceProps.maxThreadsPerBlock);
    numThreads = deviceProps.maxThreadsPerBlock;
  }

  // config kernel
  dim3 blocks (numBlocks);
  dim3 threads (numThreads);

  // start event
  cudaEventRecord (start, 0);
  // sort the vector
  rank_sort_algorithm <<< blocks, threads >>> (dev_vecB, dev_vecR, size);

  // wait for compute device to finish
  cudaDeviceSynchronize ();

  // stop event
  cudaEventRecord (stop, 0);

  // synchronize the GPU-CPU
  cudaEventSynchronize (stop);

  // calculate the elapsed time in gpu
  cudaEventElapsedTime (&elapsedTimeGPU, start, stop);

  // calculate the elapsed time in cpu
  elapsedTimeCPU = (float) eventDiff (&start_cpu, &stop_cpu);
  gain = elapsedTimeCPU / elapsedTimeGPU;



  // transfer data from device to host
  cudaMemcpy (hst_vecR_other, dev_vecR, size * sizeof (int),
              cudaMemcpyDeviceToHost);

  cout << "> Size of numbers to sort: [" << size << "]" << endl;
  cout << "> Lauch: " << numThreads << " threads per block and " << numBlocks
    << " blocks" << endl;
  cout << "> Total: " << numThreads * numBlocks << " threads" << endl;
  cout << "*********************************************" << endl;

  // print the vectors if the user wants
  if (print != 0)
  {
    print_int_vector ((char *) "Initial vector", hst_vecA, size);
    print_int_vector ((char *) "Final vector(CPU)", hst_vecR, size);
    print_int_vector ((char *) "Final vector(GPU)", hst_vecR_other, size);
  }

  cout << "*********************************************" << endl;
  cout << "EXECUTION GPU..." << endl;
  cout << "> Execution time: " << elapsedTimeGPU << " ms" << endl;
  cout << endl;
  cout << "*********************************************" << endl;
  cout << "EXECUTION CPU..." << endl;
  cout << "> Execution time: " << elapsedTimeCPU << " ms" << endl;
  cout << endl;
  cout << "Gain GPU/CPU: " << gain << endl;
  cout << "*********************************************" << endl;
  cout << "RESULTS SUMMARY" << endl;
  cout << "*********************************************" << endl;
  cout << "N - " << size << " [GPU: " << elapsedTimeGPU << " ms] [CPU: " <<
    elapsedTimeCPU << " ms]" << " [Gain GPU/CPU = " << gain << "]" << endl;
  cout << "*********************************************" << endl;

  // destroy the event objects
  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  // free 
  free (hst_vecA);
  free (hst_vecR);
  free (hst_vecR_other);
  cudaFree (dev_vecB);
  cudaFree (dev_vecR);

  return 0;
}

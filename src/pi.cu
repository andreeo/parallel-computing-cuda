#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "common.h"
#include "kernels.h"
#include "../cuda_device.h"

#define BLOCKS 1

using namespace std;

int
main (int argc, char **argv)
{
  // declarate and initializate the variable to store the user input
  // that's the number of terms that's power of 2
  int nTerms = 0;

  // declare the vector to store the terms in the device
  float *dev_vec;

  // declare the output variable in the host and device
  float *hst_output, *dev_output;

  // declare the variable to store the output
  double approxPI;

  // declare the variable to store the PI value
  double PI;

  // print the device props
  deviceProperties (DEVICE_SELECTED);

  // ask the user for the number of terms
  printf ("Enter the number of terms (power of 2):");
  scanf ("%d", &nTerms);

  //fflush(stdin);

  // allocate mem in the host 
  hst_output = (float *) malloc (sizeof (float));

  // allocate mem in the device
  cudaMalloc ((void **) &dev_vec, nTerms * sizeof (float));
  cudaMalloc ((void **) &dev_output, sizeof (float));

  // call the kernel to generate the sucessive terms and
  // compute the sum of them

  dim3 block (BLOCKS);
  dim3 thread (nTerms);

  parallel_reduction <<< block, thread >>> (dev_vec, dev_output);

  // transfer data from device to host
  cudaMemcpy (hst_output, dev_output, sizeof (float), cudaMemcpyDeviceToHost);

  // print the result
  approxPI = sqrt (double (6 * *hst_output));
  PI =
    3.141592653589793238462643383279502884197169399375105820974944592307816406286;

  cout << endl;
  cout << "> PI Value            : " << PI << endl;
  cout << "> Total amount        : " << *hst_output << endl;
  cout << "> Calculated value    : " << approxPI << endl;
  cout << "> Relative error      : " << ((approxPI - PI) / PI) * 100 << endl;

  // free the memory
  free (hst_output);
  cudaFree (dev_vec);
  cudaFree (dev_output);

  return 0;
}

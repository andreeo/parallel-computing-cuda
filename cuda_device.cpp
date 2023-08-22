#include <iostream>
#include "cuda_device.h"
#include <cuda_runtime.h>

using namespace std;

__host__ struct cudaDeviceProp
deviceProperties (int deviceId)
{
  struct cudaDeviceProp props;

  cudaGetDeviceProperties (&props, deviceId);

  const char *archName;

  int cudaCores = 0;

  switch (props.major)
  {
  case 1:
    archName = "Tesla";
    cudaCores = 8;
    break;
  case 2:
    archName = "Fermi";
    if (props.minor == 0)
      cudaCores = 32;
    else
      cudaCores = 48;
    break;
  case 3:
    archName = "Kepler";
    cudaCores = 192;
    break;
  case 4:
    archName = "Maxwell";
    cudaCores = 128;
    break;
  case 5:
    archName = "Pascal";
    cudaCores = 64;
    break;
  case 6:
    archName = "Volta";
    cudaCores = 64;
    break;
  case 7:
    archName = "Turing";
    cudaCores = 64;
    break;
  case 8:
    archName = "Ampere";
    cudaCores = 64;
    break;
  default:
    archName = "DESCONOCIDA";
  }

  cout << "DEVICE" << deviceId << " : " << props.name << endl;
  cout << "****************************************************" << endl;
  cout << "> Arquitectura CUDA            : " << archName << endl;
  cout << "> Capacidad de Computo         : " << props.major << "." << props.
    minor << endl;
  cout << "> No. MultiProcesadores        : " << props.
    multiProcessorCount << endl;
  cout << "> No. Nucleos CUDA (" << cudaCores << "x" << props.
    multiProcessorCount << ")     : " << cudaCores *
    props.multiProcessorCount << endl;
  cout << "> Memoria Global (total)       : " << props.totalGlobalMem / 1048576 << " MiB" << endl;      // to convert bytes to Mebibyte (MiB)
  cout << endl;

  return props;
}

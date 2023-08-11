#include <iostream>
#include <cuda_runtime.h>
#include "cuda_device.h"

using namespace std;

int
main (int argc, char **argv)
{
  int deviceCount;
  cudaGetDeviceCount (&deviceCount);

  if (deviceCount == 0)
  {
    cout << "||||||||| No se han encontrado dispositivos CUDA ||||||||\n";
    cout << "<pulsa [INTRO] para finalizar>";
    getchar ();
    return 1;
  }
  else
  {
    cout << "\nSe han encontrado " << deviceCount << " dispositivos CUDA" <<
      endl;
    cout << "****************************************************" << endl;

    for (int i = 0; i < deviceCount; i++)
    {
      deviceProperties (i);
    }
    time_t fecha;
    time (&fecha);
    cout << "Programa ejecutado el: " << ctime (&fecha) << endl;
    cout << "<pulsa [INTRO] para finalizar>";
    getchar ();
    return 0;
  }

}

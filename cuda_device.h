#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H
#include <cuda_runtime.h>

/*
*   Function: deviceProperties
*   --------------------------
*   Prints out the properties of the device
*
*   Parameters:
*     deviceId: int - the id of cuda device
*
*   returns: void
*/

__host__ void deviceProperties (int device_id);


#endif

#include <cstdio>
#include "common.h"

void
print_int_vector (char *vector_name, int *vector, int size)
{
  int i = 0;
  printf ("%s: [", vector_name);
  for (i = 0; i < size; i++)
  {
    if (i != size - 1)
      printf ("%2d, ", vector[i]);
    else
      printf ("%2d", vector[i]);
  }
  printf ("]\n");
}

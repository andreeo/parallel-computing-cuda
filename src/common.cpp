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

void
print_int_matrix (char *matrix_name, int *matrix, int size, int mtx_col_size)
{
  int i = 0, count = mtx_col_size;      // count is the number of columns
  printf ("-> %s:\n", matrix_name);
  for (i = 0; i < size; i++)
  {
    if (i == count)
    {
      printf ("\n");
      count += mtx_col_size;
    }

    printf (" %d ", matrix[i]);
  }
  printf ("\n");

  return;
}

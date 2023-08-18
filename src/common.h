#ifndef COMMON_H
#define COMMON_H

#define DEVICE_SELECTED 1
#define RANDOM_MAX 10

/*
*   function: print_int_vector
*   ---------------------------
*   Prints a vector of integers.
*
*   parameters:
*       vector_name: name of the vector to be printed.
*       vector: pointer to the vector to be printed.
*       size: size of the vector.
*
*   returns: void.
*/
void print_int_vector (char *vector_name, int *vector, int size);

/*
*   Function: print_int_matrix
*   ---------------------------
*   Prints a matrix of integers.
*
*   parameters:
*       matrix_name: name of the matrix to be printed.
*       matrix: pointer to the matrix to be printed.
*       size: size of the matrix.
*       col: column dimension.
*
*   returns: void.
*/
void print_int_matrix (char *matrix_name, int *matrix, int size, int col);

#endif

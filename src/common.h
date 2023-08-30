#ifndef COMMON_H
#define COMMON_H

#define DEVICE_SELECTED 1
#define RANDOM_MAX 10

#ifdef __linux__
#include <sys/time.h>
typedef struct timeval event;
#else
#include <windows.h>
typedef LARGE_INTEGER event;
#endif

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

/**
 * function: print_int_matrix
 * --------------------------
 * Prints a matrix of integers.
 * 
 * @param matrix_name name of the matrix to be printed.
 * @param matrix pointer to the matrix to be printed.
 * @param size size of the matrix.
 * @param col column dimension.
 * 
 * @return void.
 * 
 */
void print_int_matrix (char *matrix_name, int *matrix, int size, int col);


/**
 *  function: rank_sort_algorithm_cpu
 *  -------------------------
 *  Sorts a vector of integers using the rank sort algorithm.
 *  from smallest to largest.
 * 
 *  @param vector pointer to the vector to be sorted.
 *  @param output_vector pointer to the output vector.
 *  @param size size of the vector.
 * 
 * @return void.
 */
void rank_sort_algorithm_cpu (int *vector, int *output_vector, int size);

#endif

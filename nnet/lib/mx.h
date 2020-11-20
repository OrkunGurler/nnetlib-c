#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

typedef struct matrix
{
    int rows, cols;
    float **data;
} mx;
mx create_matrix(int rows, int cols);

void m_randomize(mx m, int a, int b);

void m_add_n(mx m, float n);
void m_addition(mx m1, mx m2);
void m_subtraction(mx m1, mx m2);

void m_multiply(mx m, float n);
void m_hadamard(mx m1, mx m2);
mx m_scalar(mx m1, mx m2);

float m_r_sum(mx m, int n);
float m_c_sum(mx m, int n);

float m_r_max(mx m, int n);
float m_r_min(mx m, int n);
float m_c_max(mx m, int n);
float m_c_min(mx m, int n);

// Insertion Sort
void m_ins_sort(float *arr, int size, int flag);
int compare(int num1, int num2, int flag);

mx m_transpose(mx m);

void m_map(mx m, float (*func)(float));
void m_map_v(mx m, void (*func)(float));
mx m_map_r(mx m, float (*func)(float));

mx m_copy(mx m);

float *to_array(mx m);
mx from_array(float *arr, int rows, int cols);

void print_array(float *arr, int size);
void print_matrix(mx m);
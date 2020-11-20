#include "mx.h"

mx create_matrix(int rows, int cols)
{
    mx m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float **)malloc(rows * sizeof(float *));
    int i, j;
    for (i = 0; i < rows; i++)
    {
        m.data[i] = (float *)malloc(cols * sizeof(float));
        for (j = 0; j < cols; j++)
        {
            m.data[i][j] = 0;
        }
    }
    return m;
}

void m_randomize(mx m, int init, int fin)
{
    srand(time(NULL));
    if (init > fin)
    {
        printf("\n[ERR: m_randomize] Initial value must be lower than Final value!\n");
        exit(1);
    }
    int temp;
    if (fin <= 0)
    {
        temp = init;
        init = fin;
        fin = temp;
    }
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            m.data[i][j] = (((float)rand() / (float)(RAND_MAX)) * (fin - init)) + init;
        }
    }
}

void m_add_n(mx m, float n)
{
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            m.data[i][j] += n;
        }
    }
}

void m_addition(mx m1, mx m2)
{
    int i, j;
    if ((m1.rows != m2.rows) || (m1.cols != m2.cols))
    {
        printf("\n[ERR: m_addition] Columns and Rows must match!\n");
        exit(1);
    }
    for (i = 0; i < m1.rows; i++)
    {
        for (j = 0; j < m1.cols; j++)
        {
            m1.data[i][j] += m2.data[i][j];
        }
    }
}

void m_subtraction(mx m1, mx m2)
{
    int i, j;
    if ((m1.rows != m2.rows) || (m1.cols != m2.cols))
    {
        printf("\n[ERR: m_subtraction] Columns and Rows must match!\n");
        exit(1);
    }
    for (i = 0; i < m1.rows; i++)
    {
        for (j = 0; j < m1.cols; j++)
        {
            m1.data[i][j] -= m2.data[i][j];
        }
    }
}

void m_multiply(mx m, float n)
{
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            m.data[i][j] *= n;
        }
    }
}

void m_hadamard(mx m1, mx m2)
{
    int i, j;
    if ((m1.rows != m2.rows) || (m1.cols != m2.cols))
    {
        printf("\n[ERR: m_hadamard] Columns and Rows must match!\n");
        exit(1);
    }
    for (i = 0; i < m1.rows; i++)
    {
        for (j = 0; j < m1.cols; j++)
        {
            m1.data[i][j] *= m2.data[i][j];
        }
    }
}

mx m_scalar(mx m1, mx m2)
{
    mx m;
    int i, j, k;
    if (m1.cols != m2.rows)
    {
        printf("\n[ERR: m_scalar] First matrix's Column Size and Second matrix's Row Size must match!\n");
        exit(1);
    }
    m = create_matrix(m1.rows, m2.cols);
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            for (k = 0; k < m1.cols; k++)
            {
                m.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }
    return m;
}

float m_r_sum(mx m, int n)
{
    float sum = 0;
    int i;
    for (i = 0; i < m.cols; i++)
    {
        sum += m.data[n][i];
    }
    return sum;
}

float m_c_sum(mx m, int n)
{
    float sum = 0;
    int i;
    for (i = 0; i < m.rows; i++)
    {
        sum += m.data[i][n];
    }
    return sum;
}

float m_r_max(mx m, int n)
{
    float max = 0;
    float *arr = (float *)malloc(m.cols * sizeof(float));
    int i;
    for (i = 0; i < m.cols; i++)
    {
        arr[i] = m.data[n][i];
    }
    m_ins_sort(arr, m.cols, 1);
    max = arr[m.cols - 1];
    free(arr);
    return max;
}

float m_r_min(mx m, int n)
{
    float min = 0;
    float *arr = (float *)malloc(m.cols * sizeof(float));
    int i;
    for (i = 0; i < m.cols; i++)
    {
        arr[i] = m.data[n][i];
    }
    m_ins_sort(arr, m.cols, 1);
    min = arr[0];
    free(arr);
    return min;
}

float m_c_max(mx m, int n)
{
    float max = 0;
    float *arr = (float *)malloc(m.rows * sizeof(float));
    int i;
    for (i = 0; i < m.rows; i++)
    {
        arr[i] = m.data[n][i];
    }
    m_ins_sort(arr, m.rows, 1);
    max = arr[m.rows - 1];
    free(arr);
    return max;
}

float m_c_min(mx m, int n)
{
    float min = 0;
    float *arr = (float *)malloc(m.rows * sizeof(float));
    int i;
    for (i = 0; i < m.rows; i++)
    {
        arr[i] = m.data[n][i];
    }
    m_ins_sort(arr, m.rows, 1);
    min = arr[0];
    free(arr);
    return min;
}

// Insertion Sort
void m_ins_sort(float *arr, int size, int flag)
{
    int temp;
    int i, j;
    for (i = 1; i < size; i++)
    {
        temp = arr[i];
        j = i - 1;
        while (j >= 0 && compare(arr[j], temp, flag))
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}

int compare(int num1, int num2, int flag) { return (flag == 1 ? num1 > num2 : num1 < num2); }

mx m_transpose(mx m)
{
    mx m_t = create_matrix(m.cols, m.rows);
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            m_t.data[j][i] = m.data[i][j];
        }
    }
    return m_t;
}

void m_map(mx m, float (*func)(float))
{
    float f;
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            f = m.data[i][j];
            m.data[i][j] = func(f);
        }
    }
}

void m_map_v(mx m, void (*func)(float))
{
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            func(m.data[i][j]);
        }
    }
}

mx m_map_r(mx m, float (*func)(float))
{
    float f;
    mx nm = create_matrix(m.rows, m.cols);
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            f = m.data[i][j];
            nm.data[i][j] = func(f);
        }
    }
    return nm;
}

mx m_copy(mx m)
{
    mx m_c = create_matrix(m.cols, m.rows);
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            m_c.data[i][j] = m.data[i][j];
        }
    }
    return m_c;
}

float *to_array(mx m)
{
    float *arr = (float *)malloc((m.rows * m.cols) * sizeof(float));
    int i, j, count = 0;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            arr[count] = m.data[i][j];
            count++;
        }
    }
    return arr;
}

mx from_array(float *arr, int rows, int cols)
{
    mx m = create_matrix(rows, cols);
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            m.data[i][j] = arr[i];
        }
    }
    return m;
}

void print_array(float *arr, int size)
{
    int i;
    for (i = 0; i < size - 1; i++)
    {
        printf("%.6f, ", arr[i]);
    }
    printf("%.6f\n", arr[size - 1]);
}

void print_matrix(mx m)
{
    int i, j;
    for (i = 0; i < m.rows; i++)
    {
        for (j = 0; j < m.cols; j++)
        {
            printf(" %.6f ", m.data[i][j]);
        }
        printf("\n");
    }
}
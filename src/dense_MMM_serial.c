/*
 gcc -O1 -lrt dense_MMM_serial.c -o dense_MMM_serial
 ./dense_MMM_serial
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define COEF_A 4
#define COEF_B 32
#define COEF_C 400

#define NUM_TESTS 15
#define OPTIONS 4

typedef float data_t;

/* Create abstract data type for matrix */
typedef struct
{
    long int rowlen;
    data_t* data;
} matrix_rec, *matrix_ptr;

/* Prototypes */
int clock_gettime(clockid_t clk_id, struct timespec* tp);
matrix_ptr new_matrix(long int rowlen);
int set_matrix_rowlen(matrix_ptr m, long int rowlen);
long int get_matrix_rowlen(matrix_ptr m);
int init_matrix(matrix_ptr m, long int rowlen);
int zero_matrix(matrix_ptr m, long int rowlen);
void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);

/* Time measurement */
static double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec -= 1;
        temp.tv_nsec += 1000000000;
    }
    return (double)temp.tv_sec + (double)temp.tv_nsec * 1.0e-9;
}

/* Warmup delay */
static double wakeup_delay()
{
    double meas = 0, quasi = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t0);
    int j = 100;
    while (meas < 1.0)
    {
        for (int i = 1; i < j; i++)
        {
            quasi = quasi * quasi - 1.923432;
        }
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        meas = interval(t0, t1);
        j *= 2;
    }
    return quasi;
}

int main(int argc, char* argv[])
{
    int OPTION;
    struct timespec time_start, time_stop;
    double time_stamp[OPTIONS][NUM_TESTS];
    double final_answer;
    long int x, n, alloc_size;

    printf("Matrix Multiply (serial)\n");

    final_answer = wakeup_delay();

    /* declare and initialize matrices */
    x = NUM_TESTS - 1;
    alloc_size = COEF_A * x * x + COEF_B * x + COEF_C;
    matrix_ptr a0 = new_matrix(alloc_size);
    init_matrix(a0, alloc_size);
    matrix_ptr b0 = new_matrix(alloc_size);
    init_matrix(b0, alloc_size);
    matrix_ptr c0 = new_matrix(alloc_size);
    zero_matrix(c0, alloc_size);

    for (OPTION = 0; OPTION < OPTIONS; OPTION++)
    {
        printf("Doing OPTION=%d...\n", OPTION);
        for (x = 0; x < NUM_TESTS &&
                    (n = COEF_A * x * x + COEF_B * x + COEF_C, n <= alloc_size);
             x++)
        {
            set_matrix_rowlen(a0, n);
            set_matrix_rowlen(b0, n);
            set_matrix_rowlen(c0, n);
            clock_gettime(CLOCK_REALTIME, &time_start);
            mmm_ijk(a0, b0, c0);
            clock_gettime(CLOCK_REALTIME, &time_stop);
            time_stamp[OPTION][x] = interval(time_start, time_stop);
            printf("  iter %ld done\r", x);
            fflush(stdout);
        }
        printf("\n");
    }

    printf("\nAll times are in seconds\n");
    printf("rowlen, ijk\n");
    for (int i = 0; i < x; i++)
    {
        long int rowlen = COEF_A * i * i + COEF_B * i + COEF_C;
        printf("%4ld", rowlen);
        for (int j = 0; j < OPTIONS; j++)
        {
            printf(",%10.4g", time_stamp[j][i]);
        }
        printf("\n");
    }
    printf("\nInitial delay was: %g\n", final_answer);
    return 0;
}

/* matrix routines */
matrix_ptr new_matrix(long int rowlen)
{
    matrix_ptr m = malloc(sizeof(matrix_rec));
    assert(m);
    m->rowlen = rowlen;
    if (rowlen > 0)
    {
        m->data = calloc(rowlen * rowlen, sizeof(data_t));
        assert(m->data);
    }
    else
        m->data = NULL;
    return m;
}

int set_matrix_rowlen(matrix_ptr m, long int rowlen)
{
    m->rowlen = rowlen;
    return 1;
}

long int get_matrix_rowlen(matrix_ptr m) { return m->rowlen; }

int init_matrix(matrix_ptr m, long int rowlen)
{
    m->rowlen = rowlen;
    for (long int idx = 0; idx < rowlen * rowlen; idx++)
        m->data[idx] = (data_t)idx;
    return 1;
}

int zero_matrix(matrix_ptr m, long int rowlen)
{
    m->rowlen = rowlen;
    for (long int idx = 0; idx < rowlen * rowlen; idx++)
        m->data[idx] = 0;
    return 1;
}

void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
    long int n = get_matrix_rowlen(a);
    data_t *Ad = a->data, *Bd = b->data, *Cd = c->data;
    for (long int i = 0; i < n; i++)
    {
        for (long int j = 0; j < n; j++)
        {
            data_t sum = 0;
            for (long int k = 0; k < n; k++)
                sum += Ad[i * n + k] * Bd[k * n + j];
            Cd[i * n + j] += sum;
        }
    }
}

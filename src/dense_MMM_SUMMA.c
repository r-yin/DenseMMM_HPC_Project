/***********************************************************************

  mpicc -O2 -o dense_MMM_SUMMA dense_MMM_SUMMA.c
  mpirun -np 16 ./dense_MMM_SUMMA 4 4 1

*/
#include <assert.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float data_t;

#define A 4
#define B 32
#define C 400
#define NUM_TESTS 50

// 3 x (Tile size x Tile size) x sizeof(data_t) <= Cache_size * 0.8
// 0.8 to avoid evictions from other associated code
// #define T 40 // fit in L1
// #define T 246 // fit in L2
#define T 1250          // fit in L3
static int use_omp = 1; // default omp on

// allocate float matrix
static data_t* alloc_mat(int rows, int cols)
{
    return (data_t*)malloc(sizeof(data_t) * rows * cols);
}

// general matrix matrix multiply
// C += A*B, A = A_r x A_c, B = A_c x B_c, C = A_r x B_c, leading dim = ld
static void local_genmm(data_t* A_blk, data_t* B_blk, data_t* C_blk, int A_r,
                        int A_c, int B_c, int ld)
{
#pragma omp parallel for collapse(2) schedule(                                 \
        static) if (use_omp)      // parallelize A_r * A_c number of operations
    for (int i = 0; i < A_r; ++i) // loop row
    {
        for (int k = 0; k < A_c; ++k) // loop col
        {
            data_t a = A_blk[i * A_c + k];
            for (int j = 0; j < B_c; ++j)
                C_blk[i * ld + j] += a * B_blk[k * B_c + j];
        }
    }
}

static void local_genmm_blocking(data_t* A_blk, data_t* B_blk, data_t* C_blk,
                                 int A_r, int A_c, int B_c, int ld)
{
#pragma omp parallel for collapse(2) schedule(static) if (use_omp)
    for (int ii = 0; ii < A_r; ii += T)
        for (int kk = 0; kk < A_c; kk += T)
        {
            int i_max = (ii + T > A_r ? A_r : ii + T);
            int k_max = (kk + T > A_c ? A_c : kk + T);

            for (int i = ii; i < i_max; ++i)
                for (int k = kk; k < k_max; ++k) // loop col
                {
                    data_t a = A_blk[i * A_c + k];
                    for (int j = 0; j < B_c; ++j)
                        C_blk[i * ld + j] += a * B_blk[k * B_c + j];
                }
        }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    // rank: [0,1,2..n]
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // size: n+1
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3 || argc > 4)
    {
        if (world_rank == 0) // so only 1 error message
            fprintf(stderr, "Usage: %s Pr Pc", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int Pr = atoi(argv[1]);
    int Pc = atoi(argv[2]);
    assert(Pr * Pc == world_size); // check to make sure total procs == size
    if (argc == 4)
        use_omp = atoi(argv[3]); // 1 -> enable, 0 -> off

    // making new row and column communicators
    int coord[2] = {world_rank / Pc, world_rank % Pc}; // [row, col]
    MPI_Comm row_comm, col_comm;
    // old comm, color, key, new comm name
    MPI_Comm_split(MPI_COMM_WORLD, coord[0], coord[1], &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, coord[1], coord[0], &col_comm);

    double time_stamp[NUM_TESTS];

    for (int x = 0; x < NUM_TESTS; ++x)
    {
        /*
         1. compute local block sizes, allocate memory A_block B_block C_block -
         done
         2. rank 0 calculates and sends appropriete block to each rank, so each
         rank has their designated blcok
         2. build full matrix for computation, col broadcast/row broadcast to
         each rank
         3. SUMMA multiply: each rank grab panel and col/row broadcast to the
         other ranks that lie in the same row/col
            - record timing in this stage
         4. clean up, free resources
        */
        long n = A * (long)x * (long)x + B * (long)x + C;
        if (n % Pr != 0 || n % Pc != 0)
        {
            if (world_rank == 0)
            {
                fprintf(stderr, "Skipping n=%ld (not divisible by %d×%d)\n", n,
                        Pr, Pc);
                time_stamp[x] = -1.0;
            }
            continue;
        }

        // compute block size: Br x Bc
        int Br = n / Pr, Bc = n / Pc;
        // allocate block memory for A, B, C in each rank
        data_t *A_block = alloc_mat(Br, Bc), *B_block = alloc_mat(Br, Bc),
               *C_block = alloc_mat(Br, Bc);
        memset(C_block, 0, sizeof(data_t) * Br * Bc);

        if (world_rank == 0)
        {
            // build A and B, making them all squares for simplicity
            data_t *A_full = alloc_mat(n, n), *B_full = alloc_mat(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                {
                    A_full[i * n + j] = (data_t)(i * n + j); // filler data
                    B_full[i * n + j] = (data_t)(j * n + i);
                }
            // send blocks to each rank
            // send A
            for (int r = 0; r < Pr; ++r)
                for (int c = 0; c < Pc; ++c)
                {
                    // calculate destination rank
                    int dest = r * Pc + c;
                    // if rank = 0, store first block
                    if (dest == 0)
                    {
                        for (int i = 0; i < Br; ++i)
                            for (int j = 0; j < Bc; ++j)
                            {
                                A_block[i * Bc + j] =
                                    A_full[(r * Br + i) * n + (c * Bc + j)];
                                B_block[i * Bc + j] =
                                    B_full[(r * Br + i) * n + (c * Bc + j)];
                            }
                    }
                    else
                    {
                        data_t *tmpA = alloc_mat(Br, Bc),
                               *tmpB = alloc_mat(Br, Bc);
                        for (int i = 0; i < Br; ++i)
                            for (int j = 0; j < Bc; ++j)
                            {
                                tmpA[i * Bc + j] =
                                    A_full[(r * Br + i) * n + (c * Bc + j)];
                                tmpB[i * Bc + j] =
                                    B_full[(r * Br + i) * n + (c * Bc + j)];
                            }
                        MPI_Send(tmpA, Br * Bc, MPI_FLOAT, dest, 0,
                                 MPI_COMM_WORLD);
                        MPI_Send(tmpB, Br * Bc, MPI_FLOAT, dest, 1,
                                 MPI_COMM_WORLD);
                        free(tmpA);
                        free(tmpB);
                    }
                }
            fprintf(stdout, "Done with initial setup from rank %d of test %d\n",
                    world_rank, x);

            free(A_full);
            free(B_full);
        }
        else
        {
            MPI_Recv(A_block, Br * Bc, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Recv(B_block, Br * Bc, MPI_FLOAT, 0, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        // Buffer for broadcasts
        data_t *Apanel = alloc_mat(Br, Bc), *Bpanel = alloc_mat(Br, Bc);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int k = 0; k < Pc; ++k)
        {
            // if current column is k, row broadcast on Apanel
            // if current row is k, column broadcast on Bpanel
            if (coord[1] == k)
                memcpy(Apanel, A_block, sizeof(data_t) * Br * Bc);
            MPI_Bcast(Apanel, Br * Bc, MPI_FLOAT, k,
                      row_comm); // k is rank of broadcaster
            if (coord[0] ==
                k) // if current iteration equals the current rank's row #
                memcpy(Bpanel, B_block, sizeof(data_t) * Br * Bc);
            MPI_Bcast(Bpanel, Br * Bc, MPI_FLOAT, k, col_comm);
            local_genmm(Apanel, Bpanel, C_block, Br, Bc, Bc, Bc);
            // local_genmm_blocking(Apanel, Bpanel, C_block, Br, Bc, Bc, Bc);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (world_rank == 0)
            time_stamp[x] = t1 - t0;

        free(A_block);
        free(B_block);
        free(C_block);
        free(Apanel);
        free(Bpanel);
    }

    if (world_rank == 0)
    {
        printf("\nAll times are in seconds\n");
        printf("rowlen, summa\n");
        for (int x = 0; x < NUM_TESTS; x++)
        {
            long n = A * (long)x * (long)x + B * (long)x + C;
            if (time_stamp[x] < 0)
                continue;
            printf("%8ld, %10.6g\n", n, time_stamp[x]);
        }
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}

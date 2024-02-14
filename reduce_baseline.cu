#include <iostream>
#include "error.cuh"

#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

const int THREAD_PER_BLOCK = 256;

const int N = 32 * 1024 * 1024;

__global__ void reduce0(real *d_in, real *d_out)
{
    // allocate shared_memory
    __shared__ real s_data[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared_memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    s_data[tid] = d_in[i];
    __syncthreads();    // sync

    // do reduction in shared_memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) 
    {
        if (tid % (2 * s) == 0) 
        {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();    // sync
    }

    // write result from block to global_memory
    if (tid == 0) d_out[blockIdx.x] = s_data[0];
}

bool is_true(real *out, real *res, int block_num)
{
    for (int i = 0; i < block_num; i ++) 
    {
        if (out[i] != res[i])
        {
            return false;
        }
    }
    return true;
}

int main()
{
    // allocate and initialize memory for host and device
    real *h_x = (real *)malloc(N * sizeof(real));
    real *d_x;
    CHECK(cudaMalloc((void **)&d_x, N * sizeof(real)));

    int block_num = N / THREAD_PER_BLOCK;

    real *out = (real *)malloc(N / THREAD_PER_BLOCK * sizeof(real));
    real *d_out;
    CHECK(cudaMalloc((void **)&d_out, N / THREAD_PER_BLOCK * sizeof(real)));

    real *res = (real *)malloc(N / THREAD_PER_BLOCK * sizeof(real));

    for (int i = 0; i < N; i++)
    {
        h_x[i] = 1;
    }

    for (int i = 0; i < block_num; i++)
    {
        real cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++)
        {
            cur += h_x[i * THREAD_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_x, h_x, N * sizeof(real), cudaMemcpyHostToDevice);

    dim3 grid_size(N / THREAD_PER_BLOCK, 1);
    dim3 block_size(THREAD_PER_BLOCK, 1);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    reduce0<<<grid_size, block_size>>>(d_x, d_out);
    

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Time = %g ms.\n", elapsed_time);

    // calculate the GB/s
    printf("带宽 = %f GB/s.\n", 0.032 * 4 * 1000 / elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    cudaMemcpy(out, d_out, N / THREAD_PER_BLOCK * sizeof(real), cudaMemcpyDeviceToHost);

    if (is_true(out, res, block_num))
        std::cout << "The answer is true";
    else
    {
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; i++)
        {
            printf("%lf ", out[i]);
        }
        printf("\n");
    }

    free(h_x);
    free(out);
    free(res);
    cudaFree(d_x);
    cudaFree(d_out);
}
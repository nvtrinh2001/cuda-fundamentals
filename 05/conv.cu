#include <stdio.h>

#define MASK_WIDTH 3
#define TILE_WIDTH 16
#define WIDTH 1024
#define HEIGHT 1024

__constant__ float M[MASK_WIDTH][MASK_WIDTH]; // constant memory for mask

__global__ void convolution(float *N, float *P, int width, int height) {
    __shared__ float N_ds[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - MASK_WIDTH / 2;
    int col_i = col_o - MASK_WIDTH / 2;

    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
        N_ds[ty][tx] = N[row_i * width + col_i];
    else
        N_ds[ty][tx] = 0.0;

    __syncthreads();

    float Pvalue = 0;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                Pvalue += N_ds[i + ty][j + tx] * M[i][j];
            }
        }
        if (row_o < height && col_o < width)
            P[row_o * width + col_o] = Pvalue;
    }
}

int main() {
    float *N, *P;
    float *d_N, *d_P;
    int size = WIDTH * HEIGHT * sizeof(float);

    // Allocate memory for host matrices
    N = (float*)malloc(size);
    P = (float*)malloc(size);

    // Allocate memory for device matrices
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    // Initialize N with random values
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        N[i] = rand() / (float)RAND_MAX;
    }

    // Copy N to device memory
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    // Define mask
    float h_M[MASK_WIDTH][MASK_WIDTH] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    cudaMemcpyToSymbol(M, h_M, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x, (HEIGHT + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    convolution<<<dimGrid, dimBlock>>>(d_N, d_P, WIDTH, HEIGHT);

    // Copy result matrix from device to host memory
    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_P);

    // Free host memory
    free(N);
    free(P);

    return 0;
}

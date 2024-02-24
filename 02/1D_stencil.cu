#include <iostream>
#include <algorithm>

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  temp[lindex] = in[gindex];
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  _syncthreads();

  int result = 0;
  for (int offset = -RADIUS; offset <= RADIUS; offset += 1) {
    result += temp[lindex + offset];
  }

  out[gindex] = result;
}

void fill_ints(int *x, int n) {
  std::fill_n(x, n, 1);
}

int main() {

  int *in, *out;
  int *d_in, *d_out;
  int size = (N + 2 * RADIUS) * sizeof(int);

  in = (int *)malloc(size);
  fill_ints(in, N + 2 * RADIUS);

  out = (int *)malloc(size);
  fill_ints(out, N + 2 * RADIUS);

  cudaMalloc((void **)&d_in, size);
  cudaMalloc((void **)&d_out, size);

  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

  stencil_1d<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out);

  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  // Error Checking
  for (int i = 0; i < N + 2 * RADIUS; i++) {
    if (i < RADIUS || i >= N + RADIUS) {
      if (out[i] != 1)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
    } else {
      if (out[i] != 1 + 2 * RADIUS)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2 * RADIUS);
    }
  }

  free(in); 
  free(out);
  cudaFree(d_in); 
  cudaFree(d_out);

  printf("Success!\n");
  return 0;
}

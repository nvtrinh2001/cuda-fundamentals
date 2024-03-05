#include <iostream>

#define TILE_SIZE 4
#define INPUT_SIZE 12
#define MASK_WIDTH 5
__constant__ float M[MASK_WIDTH];

__global__ void convolution_shared_memory(float *N, float *P){
	
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	__shared__ float N_s[TILE_SIZE];

	N_s[threadIdx.x]=N[i];

	__syncthreads();

	int this_title_start_point = blockIdx.x*blockDim.x;
	int next_tile_start_point = (blockIdx.x+1)*blockDim.x;
	int n_start_point = i-(MASK_WIDTH/2);
	float Pvalue = 0;

	for(int j =0; j < MASK_WIDTH; j++){

		int N_index = n_start_point+j;

		if(N_index >= 0 && N_index < INPUT_SIZE){
			if((N_index >= this_title_start_point) && (N_index < next_tile_start_point)){
				Pvalue+=N_s[threadIdx.x+j-(MASK_WIDTH/2)]*M[j];
			}
			else{
				Pvalue+=N[N_index]*M[j];
			}
		}
	}

	P[i]=Pvalue;	
}


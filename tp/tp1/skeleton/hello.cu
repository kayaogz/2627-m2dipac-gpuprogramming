#include <cstdio>
#include "cuda.h"

__global__ void cudaHello(){
  // Print "Hello World" together with blockIdx.x and threadIdx.x from every thread
  // A FAIRE ...
}

int main() {
  int numBlocks = 64;
  int blockSize = 1;
  // Experiment with different blockSize values (threads per block) over the powers of 2,
  // keeping the total number of threads equal to 64
  // A FAIRE ...
  cudaHello<<<numBlocks, blockSize>>>(); 

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  return 0;
  }

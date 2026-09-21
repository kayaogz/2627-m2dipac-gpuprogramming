// TP1, exercise 1 -- Hello World in CUDA (solution)
//
// Build:  nvcc -arch=native hello.cu -o hello
// Run:    ./hello
#include <cstdio>
#include "cuda_check.h"

__global__ void cudaHello()
{
  printf("Hello World from thread %d/%d of block %d/%d\n",
      threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);
}

int main(void)
{
  // 64 threads in total, split as numBlocks x blockSize.
  // Any of (64,1) (32,2) (16,4) (8,8) (4,16) (2,32) (1,64) works; the order
  // of the 64 lines changes from one run to the next.
  int numBlocks = 8;
  int blockSize = 64 / numBlocks;

  cudaHello<<<numBlocks, blockSize>>>();

  CUDA_CHECK_KERNEL();
  return 0;
}

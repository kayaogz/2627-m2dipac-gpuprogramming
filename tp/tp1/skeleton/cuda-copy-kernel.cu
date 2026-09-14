#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

__global__ void cudaCopyByBlocks(float *tab0, const float *tab1, int size)
{
  int idx;
  // Compute the correct idx
  // TODO ...
  // idx = ?
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

__global__ void cudaCopyByBlocksThreads(float *tab0, const float *tab1, int size)
{
  int idx;
  // Compute the correct idx in terms of blockIdx.x, threadIdx.x, and blockDim.x
  // TODO ...
  // idx = ?
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

int main(int argc, char **argv) {
  float *A, *B, *dA, *dB;
  int N, i;

  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 0;
  }
  N = atoi(argv[1]);

  // Initialization
  A = (float *) malloc(sizeof(float) * N);
  B = (float *) malloc(sizeof(float) * N);
  for (i = 0; i < N; i++) { 
    A[i] = (float)i;
    B[i] = 0.0f;
  }
  
  // Allocate dynamic arrays dA and dB of size N on the GPU with cudaMalloc
  // TODO ...

  // Copy A into dA and B into dB
  // TODO ...

  // Copy dA into dB using the kernel cudaCopyByBlocks
  // TODO ...
  // cudaCopyByBlocks<<<...,...>>>(...) ???

  // Wait for kernel cudaCopyByBlocks to finish
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // TODO ...

  // Verify the results on the CPU by comparing B with A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "The copy is incorrect!\n"; }
  else { cout << "The copy is correct!\n"; }

  // Reinitialize B to zero, then copy B into dB again to test the second copy kernel
  for (int i = 0; i < N; i++) { B[i] = 0.0f; }
  // TODO ...

  // Copy dA into dB with the kernel cudaCopyByBlocksThreads
  // TODO ...
  // cudaCopyByBlocksThreads<<<...,...>>>(...) ???

  // Wait for the kernel cudaCopyByBlocksThreads to finish
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  // TODO ...

  // Verify the results on the CPU by comparing B with A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "The copy is incorrect!\n"; }
  else { cout << "The copy is correct!\n"; }

  // Deallocate arrays dA[N] and dB[N] on the GPU
  // TODO ...

  // Deallocate A and B
  free(A);
  free(B);

  return 0;
}

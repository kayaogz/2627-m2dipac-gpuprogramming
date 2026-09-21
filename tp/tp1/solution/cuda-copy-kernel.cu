#include <cstdio>
#include <iostream>
#include "cuda.h"

using namespace std;

__global__ void cudaCopyByBlocks(float *tab0, const float *tab1, int size)
{
  int idx = blockIdx.x;
  if (idx < size) { tab0[idx] = tab1[idx]; }
}

__global__ void cudaCopyByBlocksThreads(float *tab0, const float *tab1, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
  cudaMalloc(&dA, N * sizeof(float));
  cudaMalloc(&dB, N * sizeof(float));

  // Copy A into dA and B into dB
  cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, N * sizeof(float), cudaMemcpyHostToDevice);

  // Copy dA into dB using the kernel cudaCopyByBlocks
  cudaCopyByBlocks<<<N, 1>>>(dB, dA, N);

  // Wait for kernel cudaCopyByBlocks to finish
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  cudaMemcpy(B, dB, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify the results on the CPU by comparing B with A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "The copy is incorrect!\n"; }
  else { cout << "The copy is correct!\n"; }

  // Reinitialize B to zero, then copy B into dB again to test the second copy kernel
  for (int i = 0; i < N; i++) { B[i] = 0.0f; }
  cudaMemcpy(dB, B, N * sizeof(float), cudaMemcpyHostToDevice);

  // Copy A into dA and B into dB
  cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);

  // Copy dA into dB with the kernel cudaCopyByBlocksThreads
  int blockSize = 1024;
  cudaCopyByBlocksThreads<<<(N + blockSize - 1) / blockSize, blockSize>>>(dB, dA, N);

  // Wait for the kernel cudaCopyByBlocksThreads to finish
  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("Kernel execution failed with error: \"%s\".\n", cudaGetErrorString(cudaerr));
  }

  // Copy dB into B for verification
  cudaMemcpy(B, dB, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify the results on the CPU by comparing B with A
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  if (i < N) { cout << "The copy is incorrect!\n"; }
  else { cout << "The copy is correct!\n"; }

  // Deallocate arrays dA[N] and dB[N] on the GPU
  cudaFree(dA);
  cudaFree(dB);

  // Deallocate A and B
  free(A);
  free(B);

  return 0;
}

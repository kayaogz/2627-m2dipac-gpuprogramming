// TP1, exercise 2b -- copying a dynamic array CPU -> GPU -> CPU (solution)
//
// Build:  nvcc -arch=native cuda-copy-dynamic.cu -o cuda-copy-dynamic
// Run:    ./cuda-copy-dynamic 1000000
#include <cstdio>
#include <cstdlib>
#include "cuda_check.h"

int main(int argc, char **argv)
{
  if (argc < 2) {
    printf("Usage: %s N\n", argv[0]);
    return 1;
  }
  int N = atoi(argv[1]);

  float *A = (float *) malloc(N * sizeof(float));
  float *B = (float *) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) { A[i] = (float) i; }

  // Dynamic array on the GPU: cudaMalloc fills in the device pointer
  float *dA;
  CUDA_CHECK(cudaMalloc((void **) &dA, N * sizeof(float)));

  // A[N] (CPU) -> dA[N] (GPU)
  CUDA_CHECK(cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice));

  // dA[N] (GPU) -> B[N] (CPU)
  CUDA_CHECK(cudaMemcpy(B, dA, N * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(dA));

  int i;
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  printf("The copy is %s.\n", i == N ? "correct" : "INCORRECT");

  free(A);
  free(B);
  return 0;
}

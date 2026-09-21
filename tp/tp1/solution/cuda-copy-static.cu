// TP1, exercise 2a -- copying a static array CPU -> GPU -> CPU (solution)
//
// Build:  nvcc -arch=native cuda-copy-static.cu -o cuda-copy-static
// Run:    ./cuda-copy-static
#include <cstdio>
#include "cuda_check.h"

#define N 1024

// Static array on the GPU: same declaration as on the CPU, plus __device__
__device__ float dA[N];

int main(void)
{
  float A[N], B[N];

  for (int i = 0; i < N; i++) { A[i] = (float) i; }

  // A[N] (CPU) -> dA[N] (GPU). Static GPU arrays are addressed by symbol.
  CUDA_CHECK(cudaMemcpyToSymbol(dA, A, N * sizeof(float), 0,
      cudaMemcpyHostToDevice));

  // dA[N] (GPU) -> B[N] (CPU)
  CUDA_CHECK(cudaMemcpyFromSymbol(B, dA, N * sizeof(float), 0,
      cudaMemcpyDeviceToHost));

  // cudaMemcpy is synchronous, so B is complete here: verify it.
  int i;
  for (i = 0; i < N; i++) { if (A[i] != B[i]) { break; } }
  printf("The copy is %s.\n", i == N ? "correct" : "INCORRECT");
  return 0;
}

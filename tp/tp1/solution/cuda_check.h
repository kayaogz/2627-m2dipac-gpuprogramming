#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call: on failure, print
// file:line and the message, then abort.
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } } while (0)

// A kernel launch returns nothing; check it in two steps:
//   cudaGetLastError()      -> errors of the launch itself
//   cudaDeviceSynchronize() -> errors raised while running
#define CUDA_CHECK_KERNEL() do { \
  CUDA_CHECK(cudaGetLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); } while (0)

#endif

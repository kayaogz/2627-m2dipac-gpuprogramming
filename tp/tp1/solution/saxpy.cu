#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>

using namespace std;


// saxpy with 1 thread per block, 1 operation per thread
__global__
void saxpyBlocs(const int N, float a, const float *x, float *y)
{
  int idx = blockIdx.x;
  if (idx < N) { y[idx] = a * x[idx] + y[idx]; }
}


// saxpy with blockSize threads per block, 1 operation per thread
__global__
void saxpyBlocsThreads(const int N, float a, const float *x, float *y)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { y[idx] = a * x[idx] + y[idx]; }
}


// saxpy with blockSize threads per block, each thread performing k operations
__global__
void saxpyBlocsThreadsKops(const int N, float a, const float *x, float *y, const int k)
{
  // Each block handles blockDim.x * k consecutive elements; thread t of the block
  // handles elements t, t + blockDim.x, t + 2 blockDim.x, ... of that range
  int idxBeg = blockIdx.x * blockDim.x * k + threadIdx.x;
  int idxEnd = idxBeg + blockDim.x * k;
  for (int idx = idxBeg; idx < idxEnd && idx < N; idx += blockDim.x) {
    y[idx] = a * x[idx] + y[idx];
  }
}

// Reference CPU implementation of saxpy
void saxpy(const int N, float a, float *x, float *y)
{
  for (int i = 0; i < N; i++) { y[i] = a * x[i] + y[i]; }
}

// Check that res[N] equals saxpy(N, a, x, y)
void verifySaxpy(float a, float *x, float *y, float *res, int N)
{
  int i;
  for (i = 0; i < N; i++) {
    float temp = a * x[i] + y[i];
    if (std::abs(res[i] - temp) / std::max(1e-6f, temp) > 1e-6) { 
      cout << res[i] << " " << temp << endl;
      break;
    }
  }
  if (i == N) {
    cout << "saxpy on GPU is correct." << endl;
  } else {
    cout << "saxpy on GPU is incorrect on element " << i << "." << endl;
  }
}


int main(int argc, char **argv)
{
  int blockSize;
  int k;
  float *x, *y, *res, *dx, *dy;
  float a = 2.0f;

  int N;

  if (argc < 2) {
    printf("Usage: ./saxpy N\n");
    return 0;
  }
  N = atoi(argv[1]);

  // Allocate and initialise the vectors x, y and res on the CPU
  x = (float *) malloc(N * sizeof(float));
  y = (float *) malloc(N * sizeof(float));
  res = (float *) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = 1.0f;
  }

  // Allocate the vectors dx[N] and dy[N] on the GPU, then copy x and y into dx and dy
  cudaMalloc(&dx, N * sizeof(float));
  cudaMalloc(&dy, N * sizeof(float));
  cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel saxpyBlocs with an appropriate number of blocks
  saxpyBlocs<<<N, 1>>>(N, a, dx, dy);

  // Copy dy[N] into res[N] for verification on the CPU
  cudaMemcpy(res, dy, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check the result
  verifySaxpy(a, x, y, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel saxpyBlocsThreads with some blockSize and number of blocks
  // blockSize = 32, 64, 128, 256, 512, 1024
  blockSize = 1024;
  saxpyBlocsThreads<<<(N + blockSize - 1) / blockSize, blockSize>>>(N, a, dx, dy);

  // Copy dy[N] into res[N] for verification on the CPU
  cudaMemcpy(res, dy, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check the result
  verifySaxpy(a, x, y, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel saxpyBlocsThreadsKops with some blockSize, number of blocks, and number of operations per thread (variable k)
  // blockSize = 32, 64, 128, 256, 512, 1024
  // k = 1, 2, 4, 8, 16, ...
  blockSize = 1024;
  k = 8;
  saxpyBlocsThreadsKops<<<(N + blockSize * k - 1) / (blockSize * k), blockSize>>>(N, a, dx, dy, k);

  // Copy dy[N] into res[N] for verification on the CPU
  cudaMemcpy(res, dy, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Check the result
  verifySaxpy(a, x, y, res, N);

  // Free the GPU arrays
  cudaFree(dx);
  cudaFree(dy);

  // Free the CPU arrays
  free(x);
  free(y);
  free(res);

  return 0;
}

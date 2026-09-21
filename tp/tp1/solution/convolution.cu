#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>

using namespace std;


// 1D convolution with 1 thread per block
__global__
void conv1DBlocs(const int N, const float *x, float *y)
{
  int idx = blockIdx.x;
  if (idx < N) {
    if (idx == 0 || idx == N - 1) {
      y[idx] = x[idx];
    } else {
      y[idx] = (x[idx - 1] + x[idx] + x[idx + 1]) / 3.0f;
    }
  }
}


// 1D convolution with blockSize threads per block
__global__
void conv1DBlocsThreads(const int N, const float *x, float *y)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    if (idx == 0 || idx == N - 1) {
      y[idx] = x[idx];
    } else {
      y[idx] = (x[idx - 1] + x[idx] + x[idx + 1]) / 3.0f;
    }
  }
}


// 1D convolution with blockSize threads per block et effectuant k operation par thread dans un bloc
__global__
void conv1DBlocsThreadsKops(const int N, const float *x, float *y, const int k)
{
  // Each block handles blockDim.x * k consecutive elements; thread t of the block
  // handles elements t, t + blockDim.x, t + 2 blockDim.x, ... of that range
  int idxBeg = blockIdx.x * blockDim.x * k + threadIdx.x;
  int idxEnd = idxBeg + blockDim.x * k;
  for (int idx = idxBeg; idx < idxEnd && idx < N; idx += blockDim.x) {
    if (idx == 0 || idx == N - 1) {
      y[idx] = x[idx];
    } else {
      y[idx] = (x[idx - 1] + x[idx] + x[idx + 1]) / 3.0f;
    }
  }
}


// Check that res[N] equals the 1D convolution of x
void verifyConv1D(float *x, float *res, int N)
{
  int i;
  for (i = 0; i < N; i++) {
    float temp = (i == 0 || i == N - 1) ? x[i] : (x[i - 1] + x[i] + x[i + 1]) / 3.0f;
    if (std::abs(res[i] - temp) > 1e-6) { 
      cout << res[i] << " " << temp << endl;
      break;
    }
  }
  if (i == N) {
    cout << "convolution on GPU is correct." << endl;
  } else {
    cout << "convolution on GPU is incorrect on element " << i << "." << endl;
  }
}


int main(int argc, char **argv)
{
  int blockSize;
  int k;
  float *x, *y, *res, *dx, *dy;

  int N;

  if (argc < 2) {
    printf("Usage: ./conv1d N\n");
    return 0;
  }
  N = atoi(argv[1]);

  // Allocate and initialise the vectors x, y and res on the CPU
  x = (float *) malloc(N * sizeof(float));
  y = (float *) malloc(N * sizeof(float));
  res = (float *) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    x[i] = (float)i;
    y[i] = 0.0f;
  }

  // Allocate the vectors dx[N] and dy[N] on the GPU, then copy x and y into dx and dy
  cudaMalloc(&dx, N * sizeof(float));
  cudaMalloc(&dy, N * sizeof(float));
  cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);


  // Launch the kernel conv1DBlocs with an appropriate number of blocks
  conv1DBlocs<<<N, 1>>>(N, dx, dy);


  // Copy dy[N] into res[N] for verification on the CPU
  cudaMemcpy(res, dy, N * sizeof(float), cudaMemcpyDeviceToHost);


  // Check the result
  verifyConv1D(x, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);


  // Launch the kernel conv1DBlocsThreads with some blockSize and number of blocks
  // blockSize = 32, 64, 128, 256, 512, 1024
  blockSize = 1024;
  conv1DBlocsThreads<<<(N + blockSize - 1) / blockSize, blockSize>>>(N, dx, dy);


  // Copy dy[N] into res[N] for verification on the CPU
  cudaMemcpy(res, dy, N * sizeof(float), cudaMemcpyDeviceToHost);


  // Check the result
  verifyConv1D(x, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);


  // Launch the kernel conv1DBlocsThreadsKops with some blockSize, number of blocks, and number of operations per thread (variable k)
  // blockSize = 32, 64, 128, 256, 512, 1024
  // k = 1, 2, 4, 8, 16, ...
  blockSize = 1024;
  k = 8;
  conv1DBlocsThreadsKops<<<(N + blockSize * k - 1) / (blockSize * k), blockSize>>>(N, dx, dy, k);


  // Copy dy[N] into res[N] for verification on the CPU
  cudaMemcpy(res, dy, N * sizeof(float), cudaMemcpyDeviceToHost);


  // Check the result
  verifyConv1D(x, res, N);

  // Free the GPU arrays
  cudaFree(dx);
  cudaFree(dy);


  // Free the CPU arrays
  free(x);
  free(y);
  free(res);

  return 0;
}

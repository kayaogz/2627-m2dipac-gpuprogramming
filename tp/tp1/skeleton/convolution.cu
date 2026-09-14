#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>

using namespace std;


// 1D convolution with 1 thread per block
__global__
void conv1DBlocs(const int N, const float *x, float *y)
{
  // TODO ...
}


// 1D convolution with blockSize threads per block
__global__
void conv1DBlocsThreads(const int N, const float *x, float *y)
{
  // TODO ...
}


// 1D convolution with blockSize threads per block et effectuant k operation par thread dans un bloc
__global__
void conv1DBlocsThreadsKops(const int N, const float *x, float *y, const int k)
{
  // TODO ...
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
  // TODO ...


  // Launch the kernel conv1DBlocs with an appropriate number of blocks
  // TODO ...


  // Copy dy[N] into res[N] for verification on the CPU
  // TODO ...


  // Check the result
  verifyConv1D(x, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  // TODO ...


  // Launch the kernel conv1DBlocsThreads with some blockSize and number of blocks
  // TODO ...
  // blockSize = 32, 64, 128, 256, 512, 1024
  blockSize = 1024;


  // Copy dy[N] into res[N] for verification on the CPU
  // TODO ...


  // Check the result
  verifyConv1D(x, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  // TODO ...


  // Launch the kernel conv1DBlocsThreadsKops with some blockSize, number of blocks, and number of operations per thread (variable k)
  // TODO ...
  // blockSize = 32, 64, 128, 256, 512, 1024
  // k = 1, 2, 4, 8, 16, ...
  blockSize = 1024;
  k = 8; 


  // Copy dy[N] into res[N] for verification on the CPU
  // TODO ...


  // Check the result
  verifyConv1D(x, res, N);

  // Free the GPU arrays
  // TODO ...


  // Free the CPU arrays
  free(x);
  free(y);
  free(res);

  return 0;
}

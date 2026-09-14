#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda.h>

using namespace std;


// saxpy with 1 thread per block, 1 operation per thread
__global__
void saxpyBlocs(const int N, float a, const float *x, float *y)
{
  int idx;
  // TODO ...
}


// saxpy with blockSize threads per block, 1 operation per thread
__global__
void saxpyBlocsThreads(const int N, float a, const float *x, float *y)
{
  int idx;
  // TODO ...
}


// saxpy with blockSize threads per block, each thread performing k operations
__global__
void saxpyBlocsThreadsKops(const int N, float a, const float *x, float *y, const int k)
{
  int idx;
  // TODO ...
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
  // TODO ...

  // Launch the kernel saxpyBlocs with an appropriate number of blocks
  // TODO ...

  // Copy dy[N] into res[N] for verification on the CPU
  // TODO ...

  // Check the result
  verifySaxpy(a, x, y, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  // TODO ...

  // Launch the kernel saxpyBlocsThreads with some blockSize and number of blocks
  // TODO ...
  // blockSize = 32, 64, 128, 256, 512, 1024

  // Copy dy[N] into res[N] for verification on the CPU
  // TODO ...

  // Check the result
  verifySaxpy(a, x, y, res, N);

  // Re-initialise dy[N] by copying y[N] into it again
  // TODO ...

  // Launch the kernel saxpyBlocsThreadsKops with some blockSize, number of blocks, and number of operations per thread (variable k)
  // TODO ...
  // blockSize = 32, 64, 128, 256, 512, 1024
  // k = 1, 2, 4, 8, 16, ...

  // Copy dy[N] into res[N] for verification on the CPU
  // TODO ...

  // Check the result
  verifySaxpy(a, x, y, res, N);

  // Free the GPU arrays
  // TODO ...

  // Free the CPU arrays
  free(x);
  free(y);
  free(res);

  return 0;
}

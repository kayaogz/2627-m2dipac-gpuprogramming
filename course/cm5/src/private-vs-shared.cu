// A thread's private array vs a block's shared array (session 5, slide
// "Shared memory: what it is").
//
//   privateArray  every thread declares float v[K]: K floats of its own, in
//                 registers (or in L1-cached "local" memory when K is large
//                 or indexed dynamically). No other thread can see them.
//   sharedArray   the block declares __shared__ float s[BS * K]: BS * K
//                 floats, one copy per block, that every thread of the block
//                 can read and write -- here thread t fills its K cells,
//                 then reads the cell its neighbour t + 1 filled.
// (The barrier between the write and the read is the subject of the stencil
// slides; it is here so that the kernel is correct.)
//
// Build:  nvcc -O2 -arch=native private-vs-shared.cu -o private-vs-shared
//         (godbolt: -O2)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } } while (0)

// A launch returns nothing: check the launch, then the run.
#define CUDA_CHECK_KERNEL() do { \
  CUDA_CHECK(cudaGetLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); } while (0)

#define BS 256   // threads per block
#define K  4     // floats per thread

// K floats of my own: in registers, and
// invisible to the other threads
__global__ void privateArray(int n, float *a)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v[K];                   // mine only
  if (i < n) {
    for (int j = 0; j < K; j++) v[j] = a[i] + j;
    a[i] = v[0] + v[K - 1];
  }
}

// BS * K floats for the whole block: every
// thread can read and write any of them
__global__ void sharedArray(int n, float *a)
{
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  __shared__ float s[BS * K];   // the block's
  if (i < n) {
    for (int j = 0; j < K; j++)
      s[t * K + j] = a[i] + j;
  }
  __syncthreads();              // (see later)
  int u = (t + 1) % BS;         // my neighbour
  if (i < n) a[i] = s[t * K] + s[u * K + K - 1];
}

int main(void)
{
  const int n = 1000;
  size_t bytes = n * sizeof(float);
  float *a = (float *)malloc(bytes), *r = (float *)malloc(bytes);
  for (int i = 0; i < n; i++) { a[i] = (float)(i % 50); }
  float *da;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  dim3 grid((n + BS - 1) / BS), block(BS);

  // ---- private arrays: a[i] + (a[i] + K - 1)
  CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  privateArray<<<grid, block>>>(n, da);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(r, da, bytes, cudaMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 0; i < n; i++) { if (r[i] != 2 * a[i] + K - 1) { wrong++; } }
  printf("privateArray  %s\n", wrong ? "WRONG" : "ok");

  // ---- the shared array: a[i] + (a[neighbour] + K - 1)
  CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  sharedArray<<<grid, block>>>(n, da);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(r, da, bytes, cudaMemcpyDeviceToHost));
  wrong = 0;
  for (int i = 0; i < n; i++) {
    int b = i / BS, t = i % BS, u = b * BS + (t + 1) % BS;
    float expected = (u < n) ? a[i] + a[u] + K - 1 : -1.0f;  // -1: the neighbour is past n (not written: unspecified)
    if (u < n && r[i] != expected) { wrong++; }
  }
  printf("sharedArray   %s\n", wrong ? "WRONG" : "ok");

  CUDA_CHECK(cudaFree(da));
  free(a); free(r);
  return 0;
}

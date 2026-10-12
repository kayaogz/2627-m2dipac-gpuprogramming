// a[i] to the power k (session 5, slides "a[i] to the power k: a first
// kernel" and following).
//
// Every thread replaces its element a[i] by a[i]^k, with k multiplications:
//   powerGlobal    reads a[i] from global memory k times (as written -- but
//                  nvcc hoists the loop-invariant load into a register, so the
//                  three kernels compile to the same loop and take the same time)
//   powerShared    copies a[i] to shared memory once, reads it k times there
//   powerRegister  copies a[i] to a register once: no shared memory needed,
//                  since a thread never uses another thread's element
// Any n: the grid is rounded up and the threads past n are guarded.
//
// Each kernel is launched twice: a warm-up launch (the first launch of a
// kernel also loads it onto the GPU), then a launch timed with std::chrono
// around the launch and cudaDeviceSynchronize(). a is restored before each
// launch (the kernels work in place). The result of the timed launch is
// compared with the CPU (same multiplications in the same order).
//
// Build:  nvcc -O2 -arch=native power.cu -o power
//         (godbolt: -O2)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
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

// a[i] is read from global memory k times
__global__ void powerGlobal(int n, float *a, int k)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float c = 1.0f;
    for (int j = 0; j < k; j++) {
      c *= a[i];          // global read
    }
    a[i] = c;             // global write
  }
}

// a[i]: 1 global read, k shared reads
__global__ void powerShared(int n, float *a, int k)
{
  __shared__ float s[BS];  // one per thread
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  if (i < n) {
    s[t] = a[i];          // global read
    float c = 1.0f;
    for (int j = 0; j < k; j++) {
      c *= s[t];          // shared read
    }
    a[i] = c;             // global write
  }
}

// a[i]: 1 global read, k register reads
__global__ void powerRegister(int n, float *a, int k)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i];       // global read
    float c = 1.0f;
    for (int j = 0; j < k; j++) {
      c *= x;             // register read
    }
    a[i] = c;             // global write
  }
}

// Wall-clock time in milliseconds. A launch returns at once: synchronise
// (CUDA_CHECK_KERNEL does) before reading the clock.
double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// Copy da back, compare it with the CPU, print one line
void check(const char *name, double ms, const float *da, float *a, const float *a0, int n, int k)
{
  CUDA_CHECK(cudaMemcpy(a, da, n * sizeof(float), cudaMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 0; i < n; i++) {
    float c = 1.0f;
    for (int j = 0; j < k; j++) { c *= a0[i]; }
    if (!(fabsf(a[i] - c) <= 1e-5f * fabsf(c))) { wrong++; }
  }
  if (wrong == 0) { printf("%-14s ok     %8.3f ms\n", name, ms); }
  else            { printf("%-14s WRONG (%d elements)\n", name, wrong); }
}

// GPUs raise their clock after a few hundred ms of load: without this
// spin (~0.3-0.5 s) first, the first timings would run at a lower clock.
__global__ void settleClock()
{
  long long t0 = clock64();
  while (clock64() - t0 < 400000000LL) { }   // 4e8 cycles
}

int main(void)
{
  settleClock<<<1, 32>>>();   // raise the clock first (see above)
  CUDA_CHECK_KERNEL();
  const int n = (1 << 22) + 100;   // 4 M floats, not a multiple of BS
  const int k = 100;          // the power
  size_t bytes = n * sizeof(float);
  float *a0 = (float *)malloc(bytes), *a = (float *)malloc(bytes);
  for (int i = 0; i < n; i++) { a0[i] = 1.0f + (i % 100) * 1e-4f; }
  float *da;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  dim3 grid((n + BS - 1) / BS), block(BS);
  double t0, ms;

  // ---- k reads from global memory
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  powerGlobal<<<grid, block>>>(n, da, k);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());      // the copy is finished before t0
  t0 = now();
  powerGlobal<<<grid, block>>>(n, da, k);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  check("powerGlobal", ms, da, a, a0, n, k);

  // ---- k reads from shared memory
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  powerShared<<<grid, block>>>(n, da, k);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  t0 = now();
  powerShared<<<grid, block>>>(n, da, k);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  check("powerShared", ms, da, a, a0, n, k);

  // ---- k reads from a register
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  powerRegister<<<grid, block>>>(n, da, k);    // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
  t0 = now();
  powerRegister<<<grid, block>>>(n, da, k);    // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  check("powerRegister", ms, da, a, a0, n, k);

  CUDA_CHECK(cudaFree(da));
  free(a0); free(a);
  return 0;
}

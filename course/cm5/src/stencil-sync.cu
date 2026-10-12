// The 1D stencil, without and with shared memory, and why the shared version
// needs a barrier (session 5, slides "The 1D stencil" and following).
//
//   b[i] = (a[i-1] + a[i] + a[i+1]) / 3   for 0 < i < n-1 (the ends are not
//   computed)
//
//   stencilGlobal   every thread reads its three inputs from global memory:
//                   3 global reads per output
//   stencilNoSync   every thread copies ONE element to shared memory, then
//                   the threads that are not at the extremities of the block
//                   read their three inputs there. Consecutive blocks overlap
//                   by 2 elements (block b starts at b * (BS - 2)), so that
//                   every output has an interior thread. No barrier: WRONG
//                   (a thread may read s[t-1] or s[t+1] before the thread
//                   that loads it has written it)
//   stencilShared   the same with __syncthreads() between the copy and the
//                   reads
// n - 2 is a multiple of BS - 2: no guard in the shared versions.
//
// Each kernel is launched twice: a warm-up launch (the first launch of a
// kernel also loads it onto the GPU), then a launch timed with std::chrono
// around the launch and cudaDeviceSynchronize(). The result of the timed
// launch is compared with the CPU (exactly: same operations, same order).
// stencilNoSync is then launched RUNS more times, and the number of wrong
// outputs of each launch is printed. GB/s counts the bytes the kernel must
// move: a read once, b written once.
//
// Build:  nvcc -O2 -arch=native stencil-sync.cu -o stencil-sync
//         (godbolt: -O2)
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define BS 256    // threads per block
#define RUNS 5    // extra launches of stencilNoSync

// 3 global reads per output
__global__ void stencilGlobal(
    int n, const float *a, float *b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > 0 && i < n - 1) {
    b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0f;
  }
}

// Each thread loads one element, then computes
__global__ void stencilNoSync(
    const float *a, float *b)
{
  __shared__ float s[BS];
  int t = threadIdx.x;
  // consecutive blocks overlap by 2
  int i = blockIdx.x * (BS - 2) + t;
  s[t] = a[i];                // 1 global read
  if (t > 0 && t < BS - 1) {  // not the ends
    b[i] = (s[t - 1] + s[t] + s[t + 1]) / 3.0f;
  }
}

// The same, with a barrier after the load
__global__ void stencilShared(
    const float *a, float *b)
{
  __shared__ float s[BS];
  int t = threadIdx.x;
  // consecutive blocks overlap by 2
  int i = blockIdx.x * (BS - 2) + t;
  s[t] = a[i];                // 1 global read
  __syncthreads();            // s is complete
  if (t > 0 && t < BS - 1) {  // not the ends
    b[i] = (s[t - 1] + s[t] + s[t + 1]) / 3.0f;
  }
}

// Wall-clock time in milliseconds. A launch returns at once: synchronise
// (CUDA_CHECK_KERNEL does) before reading the clock.
double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// Copy db back, count the wrong outputs (0 < i < n-1)
int countWrong(const float *db, float *b, const float *a, int n)
{
  CUDA_CHECK(cudaMemcpy(b, db, n * sizeof(float), cudaMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 1; i < n - 1; i++) {
    if (b[i] != (a[i - 1] + a[i] + a[i + 1]) / 3.0f) { wrong++; }
  }
  return wrong;
}

// Print one line: ok and the time, or WRONG and the count
void report(const char *name, double ms, int wrong, int n)
{
  double gbs = 2.0 * n * sizeof(float) / ms / 1e6;   // a read, b written
  if (wrong == 0) { printf("%-14s ok     %8.3f ms  %6.1f GB/s\n", name, ms, gbs); }
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
  const int n = 32768 * (BS - 2) + 2;   // 8.3 M floats: n - 2 is a multiple of BS - 2
  size_t bytes = n * sizeof(float);
  float *a = (float *)malloc(bytes), *b = (float *)malloc(bytes);
  for (int i = 0; i < n; i++) { a[i] = (float)(i % 100); }
  float *da, *db;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  dim3 gridGlobal((n + BS - 1) / BS);   // one thread per element
  dim3 gridShared((n - 2) / (BS - 2));  // blocks overlap by 2
  dim3 block(BS);
  double t0, ms;

  // ---- global memory only
  stencilGlobal<<<gridGlobal, block>>>(n, da, db);   // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemset(db, 0, bytes));              // check the timed run
  CUDA_CHECK(cudaDeviceSynchronize());   // cudaMemset is asynchronous: finish it before t0
  t0 = now();
  stencilGlobal<<<gridGlobal, block>>>(n, da, db);   // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  report("stencilGlobal", ms, countWrong(db, b, a, n), n);

  // ---- shared memory, no barrier
  stencilNoSync<<<gridShared, block>>>(da, db);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemset(db, 0, bytes));
  CUDA_CHECK(cudaDeviceSynchronize());
  t0 = now();
  stencilNoSync<<<gridShared, block>>>(da, db);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  report("stencilNoSync", ms, countWrong(db, b, a, n), n);

  // ---- shared memory, with the barrier
  stencilShared<<<gridShared, block>>>(da, db);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemset(db, 0, bytes));
  CUDA_CHECK(cudaDeviceSynchronize());
  t0 = now();
  stencilShared<<<gridShared, block>>>(da, db);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  report("stencilShared", ms, countWrong(db, b, a, n), n);

  // ---- the race, run after run
  printf("stencilNoSync, wrong outputs in %d more runs:", RUNS);
  for (int r = 0; r < RUNS; r++) {
    CUDA_CHECK(cudaMemset(db, 0, bytes));
    stencilNoSync<<<gridShared, block>>>(da, db);
    CUDA_CHECK_KERNEL();
    printf(" %d", countWrong(db, b, a, n));
  }
  printf("\n");

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  free(a); free(b);
  return 0;
}

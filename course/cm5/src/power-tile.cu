// a[i][j] to the power k on a 2D array, with horizontal and vertical warps,
// then through a shared tile (session 5, section "a[i][j]^k on a 2D array").
//
// a is an n x n row-major array; every element is replaced by its k-th power
// (k multiplications, reading the element every time). Blocks of T x T
// threads, one element per thread; n is a multiple of T: no guard.
//   powerH           horizontal warps: a warp handles a row of the block's
//                    tile, 32 consecutive floats: coalesced
//   powerV           vertical warps: a warp handles a column of the tile,
//                    32 floats 4n bytes apart: 32 sectors, not coalesced
//   powerTileV       the vertical mapping through a shared tile s[T][T]:
//                    the tile is loaded and stored horizontally (coalesced),
//                    and used vertically from s: thread (tx, ty) multiplies
//                    s[tx][ty] k times. A warp reads a column of s, words
//                    32 apart, all in one bank: 32-way bank conflict
//   powerTileVPadded the same with s[T][T + 1]: a column of s is now in 32
//                    different banks: 1-way, and the global accesses stay
//                    coalesced
//
// Each kernel is launched twice: a warm-up launch (the first launch of a
// kernel also loads it onto the GPU), then a launch timed with std::chrono
// around the launch and cudaDeviceSynchronize(). a is restored before each
// launch (the kernels work in place). The result of the timed launch is
// compared with the CPU (same multiplications in the same order).
//
// Build:  nvcc -O2 -arch=native power-tile.cu -o power-tile
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

#define T 32   // tile T x T, blocks of T x T threads

// Horizontal warps: a warp handles a row
__global__ void powerH(float *a, int n, int k)
{
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = blockIdx.y * T + ty;      // row
  int j = blockIdx.x * T + tx;      // column
  float c = 1.0f;
  for (int p = 0; p < k; p++) { c *= a[i * n + j]; }
  a[i * n + j] = c;
}

// Vertical warps: a warp handles a column
__global__ void powerV(float *a, int n, int k)
{
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = blockIdx.y * T + tx;      // row
  int j = blockIdx.x * T + ty;      // column
  float c = 1.0f;
  for (int p = 0; p < k; p++) { c *= a[i * n + j]; }
  a[i * n + j] = c;
}

// Vertical warps through a shared tile:
// loaded and stored by rows, used by columns
__global__ void powerTileV(float *a, int n, int k)
{
  __shared__ float s[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = blockIdx.y * T + ty;
  int j = blockIdx.x * T + tx;
  s[ty][tx] = a[i * n + j];         // load: row
  __syncthreads();
  float c = 1.0f;                   // use: column
  for (int p = 0; p < k; p++) { c *= s[tx][ty]; }
  s[tx][ty] = c;
  __syncthreads();
  a[i * n + j] = s[ty][tx];         // store: row
}

// The same, the rows of s padded to T + 1 words
__global__ void powerTileVPadded(float *a,
                                 int n, int k)
{
  __shared__ float s[T][T + 1];     // + 1: padding
  int tx = threadIdx.x, ty = threadIdx.y;
  int i = blockIdx.y * T + ty;
  int j = blockIdx.x * T + tx;
  s[ty][tx] = a[i * n + j];         // load: row
  __syncthreads();
  float c = 1.0f;                   // use: column
  for (int p = 0; p < k; p++) { c *= s[tx][ty]; }
  s[tx][ty] = c;
  __syncthreads();
  a[i * n + j] = s[ty][tx];         // store: row
}

// Wall-clock time in milliseconds. A launch returns at once: synchronise
// (CUDA_CHECK_KERNEL does) before reading the clock.
double now(void)
{
  using namespace std::chrono;
  return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// Copy da back, compare it with the reference, print one line
void check(const char *name, double ms, const float *da, float *a, const float *ref, int n)
{
  CUDA_CHECK(cudaMemcpy(a, da, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost));
  int wrong = 0;
  for (int e = 0; e < n * n; e++) {
    if (!(fabsf(a[e] - ref[e]) <= 1e-5f * fabsf(ref[e]))) { wrong++; }
  }
  double gbs = 2.0 * n * n * sizeof(float) / ms / 1e6;   // a read, a written
  if (wrong == 0) { printf("%-17s ok     %7.3f ms  %6.1f GB/s\n", name, ms, gbs); }
  else            { printf("%-17s WRONG (%d elements)\n", name, wrong); }
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
  const int n = 2048;         // 2048 x 2048 floats = 16 MB, a multiple of T
  const int k = 100;          // the power
  size_t bytes = (size_t)n * n * sizeof(float);
  float *a0 = (float *)malloc(bytes), *a = (float *)malloc(bytes);
  float *ref = (float *)malloc(bytes);
  for (int e = 0; e < n * n; e++) {
    a0[e] = 1.0f + (e % 100) * 1e-4f;
    float c = 1.0f;
    for (int p = 0; p < k; p++) { c *= a0[e]; }
    ref[e] = c;
  }
  float *da;
  CUDA_CHECK(cudaMalloc(&da, bytes));
  dim3 grid(n / T, n / T), block(T, T);
  double t0, ms;

  // ---- powerH
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  powerH<<<grid, block>>>(da, n, k);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());    // the copy is finished before t0
  t0 = now();
  powerH<<<grid, block>>>(da, n, k);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  check("powerH", ms, da, a, ref, n);

  // ---- powerV
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  powerV<<<grid, block>>>(da, n, k);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());    // the copy is finished before t0
  t0 = now();
  powerV<<<grid, block>>>(da, n, k);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  check("powerV", ms, da, a, ref, n);

  // ---- powerTileV
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  powerTileV<<<grid, block>>>(da, n, k);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());    // the copy is finished before t0
  t0 = now();
  powerTileV<<<grid, block>>>(da, n, k);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  check("powerTileV", ms, da, a, ref, n);

  // ---- powerTileVPadded
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  powerTileVPadded<<<grid, block>>>(da, n, k);      // warm-up
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaMemcpy(da, a0, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());    // the copy is finished before t0
  t0 = now();
  powerTileVPadded<<<grid, block>>>(da, n, k);      // timed
  CUDA_CHECK_KERNEL();
  ms = now() - t0;
  check("powerTileVPadded", ms, da, a, ref, n);

  CUDA_CHECK(cudaFree(da));
  free(a0); free(a); free(ref);
  return 0;
}
